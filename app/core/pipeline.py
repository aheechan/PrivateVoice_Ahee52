"""对话编排：录音 → ASR → LLM → 切句 → TTS → 播放。

严格串行：同一时刻只有一轮在跑。任意时刻打断都会取消这一轮的所有子任务
（LLM 请求、TTS 合成、播放缓冲），并直接进入下一轮 listening。
"""
from __future__ import annotations

import json
import logging
import queue
import re
import threading
import time
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np

from . import audio_io, audio_qc

logger = logging.getLogger(__name__)
from .asr import ASREngine, to_16k
from .history import HistoryStore, system_prompt
from .llm import LLMClient, LLMError
from .metrics import MetricsStore, TurnMetrics
from .player import Player
from .tts import TTSEngine, detect_lang, make_engine

SENT_END = "。！？…!?."
SENT_SOFT = "，、；,;"


class State(str, Enum):
    IDLE = "idle"
    LISTENING = "listening"
    TRANSCRIBING = "transcribing"
    THINKING = "thinking"
    SPEAKING = "speaking"
    TRAINING = "training"


# ---------------------------------------------------------------------------
# 切句
# ---------------------------------------------------------------------------
class SentenceSplitter:
    """流式累积，满足条件立刻吐出一句。"""

    def __init__(self, min_chars: int = 6, max_chars: int = 40):
        self.min_chars = min_chars
        self.max_chars = max_chars
        self.buf = ""

    @staticmethod
    def _limit(text: str) -> int:
        # 英文按字符数会更短，放宽上限，避免一句话被切成碎片
        latin = len(re.findall(r"[A-Za-z]", text))
        return 90 if latin > len(text) * 0.5 else 40

    def feed(self, text: str) -> list[str]:
        out: list[str] = []
        for ch in text:
            self.buf += ch
            if ch in SENT_END and len(self.buf.strip()) >= 2:
                out.append(self.buf)
                self.buf = ""
            elif ch in SENT_SOFT and len(self.buf.strip()) >= self.min_chars:
                out.append(self.buf)
                self.buf = ""
            elif len(self.buf) >= self._limit(self.buf):
                out.append(self.buf)
                self.buf = ""
        return out

    def flush(self) -> list[str]:
        rest, self.buf = self.buf, ""
        return [rest] if rest.strip() else []


_MD = re.compile(r"[`*_#>~\|]+")
_BRACKET = re.compile(r"（[^）]*）|\([^)]*\)|\[[^\]]*\]|【[^】]*】")
_EMOJI = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F000-\U0001F2FF"
    "\U0000FE0F\U0000200D]+")
_SPACE = re.compile(r"\s+")


def clean_for_tts(text: str) -> str:
    s = _MD.sub("", text)
    s = _BRACKET.sub("", s)
    s = _EMOJI.sub("", s)
    s = s.replace("“", "").replace("”", "").replace("\"", "")
    # 英文单词之间的空格必须保留，否则会被读成一串连写
    s = _SPACE.sub(" ", s)
    s = re.sub(r"\s+(?=[，。！？；：、）】》])", "", s)
    s = re.sub(r"(?<=[，。！？；：、（【《])[ \t]+", "", s)
    s = re.sub(r"(?<=[\u4e00-\u9fff]) (?=[\u4e00-\u9fff])", "", s)
    return s.strip()


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
@dataclass
class PipelineEvent:
    kind: str          # state | user_text | reply_delta | sentence | error | turn_done | info
    state: str = ""
    text: str = ""
    detail: str = ""


def _trim_edges(x: np.ndarray, sr: int, thresh_db: float = -50.0,
                keep_ms: float = 30.0) -> np.ndarray:
    """去掉句首句尾的静音。

    v3 每句都会自带约 300ms 尾静音（训练片段两端本来就留了 padding），
    不裁的话句与句之间会叠出 300ms+ 的空洞，听起来就是"断断续续"。
    """
    if x.size == 0:
        return x
    frame = max(1, int(sr * 0.01))
    n = x.size // frame
    if n < 2:
        return x
    rms = np.sqrt(np.mean(x[: n * frame].reshape(n, frame) ** 2, axis=1) + 1e-12)
    db = 20 * np.log10(rms + 1e-12)
    voiced = np.where(db > thresh_db)[0]
    if voiced.size == 0:
        return x
    keep = int(sr * keep_ms / 1000)
    a = max(0, voiced[0] * frame - keep)
    b = min(x.size, (voiced[-1] + 1) * frame + keep)
    return x[a:b]


class Pipeline:
    def __init__(self, config: Any, store: Any, character: Any,
                 on_event: Callable[[PipelineEvent], None] | None = None):
        self.config = config
        self.store = store
        self.character = character
        self.on_event = on_event

        self.state = State.IDLE
        self.asr = ASREngine(config.data.get("asr", {}))
        self.llm = LLMClient(config)
        self.player = Player(
            device=config.audio.get("output_device"),
            blocksize_ms=int(config.get("player.blocksize_ms", 30)),
            sentence_gap_ms=int(config.get("player.sentence_gap_ms", 80)),
            limiter_dbtp=float(config.get("player.limiter_dbtp", -1.0)),
        )
        self.recorder = audio_io.Recorder(device=config.audio.get("input_device"))
        self._tts: TTSEngine | None = None
        self._turn_audio: list[tuple[np.ndarray, int]] = []
        self._web_sources: list[str] = []
        self.style_id = str(config.get("llm.default_style", "gentle"))
        self._tts_key: tuple[str, str] = ("", "")

        self.metrics = MetricsStore(Path(config.path_of("logs")) / "turns.jsonl")
        self.history = HistoryStore(
            store.dir_of(character.id) if character else Path("."),
            context_turns=int(config.get("llm.history_turns", 12)),
        )

        self._cancel = threading.Event()
        self._lock = threading.RLock()
        self._tts_ready = threading.Event()
        self._asr_ready = threading.Event()
        self._tts_warming = False
        self._asr_warming = False
        self._turn_lock = threading.Lock()
        self._tts_lock = threading.RLock()
        self._tts_stop = threading.Event()
        self._turn_thread: threading.Thread | None = None
        self._tts_queue: "queue.Queue[Any] | None" = None
        self._tts_thread: threading.Thread | None = None
        self._t_press = 0.0
        self._t_release = 0.0
        self._t_asr = 0.0
        self._t_llm = 0.0
        self._t_tts = 0.0
        self._first_play_logged = False

    # -- 事件 ----------------------------------------------------------
    def _emit(self, kind: str, **kw: Any) -> None:
        if self.on_event:
            try:
                self.on_event(PipelineEvent(kind=kind, **kw))
            except Exception:
                pass

    def _set_state(self, state: State, detail: str = "") -> None:
        self.state = state
        self._emit("state", state=state.value, detail=detail)

    # -- 角色 ----------------------------------------------------------
    def set_training(self, busy: bool) -> None:
        """训练期间禁止进入 listening（热键按下会被忽略）。"""
        if busy:
            self.state = State.TRAINING
            self._emit("state", state="training", detail="训练进行中")
        elif self.state == State.TRAINING:
            self._set_state(State.IDLE)

    def set_character(self, character: Any) -> None:
        self.interrupt(silent=True)
        self.history = HistoryStore(
            self.store.dir_of(character.id),
            context_turns=int(self.config.get("llm.history_turns", 12)),
        )
        self.character = character
        self._emit("info", text=f"已切换到角色：{character.name}")

    def _ensure_tts(self) -> TTSEngine:
        with self._tts_lock:
            ch = self.character
            key = (ch.voice.get("gpt_ckpt", ""), ch.voice.get("sovits_ckpt", ""))
            if self._tts is None:
                self._tts = make_engine(self.config)
            if key != self._tts_key:
                self._tts.load(*key)
                self._tts_key = key
                self._emit("info", text=f"TTS 引擎已加载：{Path(key[0]).name}")
            else:
                self._emit("info", text="TTS 引擎复用（已加载）")
            return self._tts

    def vram_free_gb(self) -> float:
        try:
            import torch
            return round(torch.cuda.mem_get_info()[0] / 1024 ** 3, 2)
        except Exception:
            return 0.0

    def release_models(self) -> dict[str, Any]:
        """卸载 ASR + TTS，把显存让给训练。"""
        out: dict[str, Any] = {}
        import gc

        try:
            import torch
            torch.cuda.synchronize()
            out["vram_free_before_gb"] = round(torch.cuda.mem_get_info()[0] / 1024 ** 3, 2)
        except Exception:
            pass
        with self._tts_lock:
            engine, self._tts = self._tts, None
        if engine is not None:
            try:
                engine.unload()
            except Exception as exc:  # noqa: BLE001
                out["tts_error"] = str(exc)
            del engine
        self._tts_key = ("", "")
        self._tts_ready.clear()
        self._asr_ready.clear()
        try:
            self.asr.unload()
        except Exception as exc:  # noqa: BLE001
            out["asr_error"] = str(exc)
        try:
            import torch
            gc.collect()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect() if hasattr(torch.cuda, "ipc_collect") else None
            free, total = torch.cuda.mem_get_info()
            out["vram_free_gb"] = round(free / 1024 ** 3, 2)
            out["vram_total_gb"] = round(total / 1024 ** 3, 2)
            out["torch_allocated_gb"] = round(torch.cuda.memory_allocated() / 1024 ** 3, 2)
            out["torch_reserved_gb"] = round(torch.cuda.memory_reserved() / 1024 ** 3, 2)
        except Exception:
            pass
        self._emit("info", text=f"已释放模型：{out}")
        return out

    def load_models(self) -> dict[str, Any]:
        """重新加载 ASR + TTS（会在后台跑，期间轮次会等）。"""
        tts = self.warm_tts()
        asr = self.warm_asr()
        self._emit("info", text="模型已重新加载")
        return {**tts, **asr}

    def warm_tts(self) -> dict[str, Any]:
        """第一段预热：只加载 TTS（约 18s），完成后就可以放开输入。

        不能和 ASR 预热并发——两者同时抢 GPU 会让 v3 的 CUDA Graph 解码卡死。
        """
        out: dict[str, Any] = {}
        self._tts_warming = True
        t0 = time.perf_counter()
        try:
            t1 = time.perf_counter()
            engine = self._ensure_tts()
            out["tts_load_s"] = round(time.perf_counter() - t1, 2)
            ch = self.character
            ref = str(self.store.dir_of(ch.id) /
                      (ch.voice.get("reference") or "reference.wav"))
            t2 = time.perf_counter()
            with self._tts_lock:
                for _ in engine.synthesize_stream("你好。", ref,
                                                  ch.voice.get("reference_text", "")):
                    pass
            out["tts_prime_s"] = round(time.perf_counter() - t2, 2)
        except Exception as exc:  # noqa: BLE001
            out["tts_error"] = str(exc)
        out["total_s"] = round(time.perf_counter() - t0, 2)
        self._tts_warming = False
        self._tts_ready.set()
        return out

    def warm_asr(self) -> dict[str, Any]:
        """第二段预热：ASR 跑一段静音。它在后台跑，期间轮次会等它。"""
        out: dict[str, Any] = {}
        self._asr_warming = True
        t0 = time.perf_counter()
        try:
            out["asr_ms"] = round(self.asr.warmup(), 1)
        except Exception as exc:  # noqa: BLE001
            out["asr_error"] = str(exc)
        out["asr_total_s"] = round(time.perf_counter() - t0, 2)
        self._asr_warming = False
        self._asr_ready.set()
        return out

    def warmup(self) -> dict[str, Any]:
        """一次性跑完两段（自检/测试用）。"""
        out = self.warm_tts()
        out.update(self.warm_asr())
        return out

    # -- 一轮 ----------------------------------------------------------
    @property
    def is_warm(self) -> bool:
        """TTS 就绪即可放开输入；ASR 还在后台预热。"""
        return self._tts_ready.is_set()

    def _reset_timing(self) -> None:
        self._t_asr = self._t_llm = self._t_tts = 0.0
        self._first_play_logged = False

    def press(self) -> None:
        """按下热键：打断当前轮，开始录音。"""
        if self.state == State.TRAINING:
            self._emit("info", text="训练进行中，先等训练结束")
            return
        self.interrupt(silent=True)
        self._cancel.clear()
        self._t_press = time.perf_counter()
        self._reset_timing()
        try:
            self.recorder.start()
        except Exception as exc:  # noqa: BLE001
            self._emit("error", text=f"麦克风打不开：{exc}")
            self._set_state(State.IDLE)
            return
        self._set_state(State.LISTENING)

    def release(self) -> None:
        if self.state != State.LISTENING:
            return
        t = time.perf_counter()
        try:
            data = self.recorder.stop()
        except Exception as exc:  # noqa: BLE001
            self._emit("error", text=f"录音停止失败：{exc}")
            self._set_state(State.IDLE)
            return
        sr_rec = int(getattr(self.recorder, "samplerate", 48000))
        if data is None or len(data) < sr_rec * 0.03:
            self._emit("error", text="没听清，再说一次")
            self._set_state(State.IDLE)
            return
        self._t_release = t
        self.submit_audio(data, sr_rec)

    def submit_audio(self, samples: np.ndarray, sr: int,
                     t_press: float | None = None,
                     t_release: float | None = None) -> None:
        """录音数据入口（测试可直接调用，绕过麦克风）。"""
        self._await_previous_turn()
        if t_press:
            self._t_press = t_press
        if t_release:
            self._t_release = t_release
        self._reset_timing()
        self._set_state(State.TRANSCRIBING)
        self._turn_thread = threading.Thread(
            target=self._run_turn, args=(samples, sr), daemon=True,
            name="vlab-turn")
        self._turn_thread.start()

    def _await_previous_turn(self, timeout: float = 12.0) -> None:
        """保证上一轮彻底结束（TTS 线程也退出），否则会有两个引擎同时活着。"""
        t = self._turn_thread
        if t is not None and t.is_alive():
            self._cancel.set()
            self.player.interrupt()
            self._finish_tts_quick()
            t.join(timeout=timeout)
        self._cancel.clear()

    def submit_text(self, text: str) -> None:
        """纯文本入口（自测用：跳过录音与 ASR）。"""
        self._await_previous_turn()
        self._t_press = time.perf_counter()
        self._t_release = self._t_press
        self._t_asr = self._t_release
        self._reset_timing()
        self._t_asr = self._t_release
        self._cancel.clear()
        self._set_state(State.THINKING)
        self._turn_thread = threading.Thread(
            target=self._run_turn, args=(None, 0, text), daemon=True,
            name="vlab-turn-text")
        self._turn_thread.start()

    # -- 打断 ----------------------------------------------------------
    def interrupt(self, silent: bool = False) -> float:
        """停播放、取消 LLM/TTS、清队列。返回停止耗时（ms）。"""
        self._cancel.set()
        t0 = time.perf_counter()
        try:
            if self.recorder.recording:
                self.recorder.stop()
        except Exception:
            pass
        ms = self.player.interrupt()
        if not silent:
            self._emit("info", text="已打断")
        if self.state not in (State.IDLE, State.TRAINING):
            self._set_state(State.IDLE)
        # 清理放到最后，且不把线程 join 的时间算进"打断响应"
        self._finish_tts_quick()
        t = self._turn_thread
        if t is not None and t.is_alive():
            t.join(timeout=2.0)
        return ms

    def close(self) -> None:
        self.interrupt(silent=True)
        self.player.close()
        if self._tts is not None:
            try:
                self._tts.unload()
            except Exception:
                pass

    # -- 主流程 --------------------------------------------------------
    def set_style(self, style_id: str) -> None:
        """切换情绪档位（温柔/活泼/冷静/毒舌/低落），下一轮生效。"""
        self.style_id = style_id
        self._emit("info", text=f"情绪档位：{self._style().get('name', style_id)}")

    def _style(self) -> dict[str, Any]:
        styles = self.config.get("llm.styles", []) or []
        for s in styles:
            if s.get("id") == self.style_id:
                return s
        return styles[0] if styles else {}

    def _should_search(self, user_text: str) -> bool:
        if not bool(self.config.get("web.enabled", False)):
            return False
        from app.core import websearch

        if bool(self.config.get("web.always", False)):
            return True
        return websearch.needs_search(user_text)

    def _messages(self, user_text: str) -> list[dict[str, str]]:
        ch = self.character
        template = ""
        try:
            p = Path(self.config.path_of("assets")) / "prompts" / "companion_system.txt"
            template = p.read_text(encoding="utf-8")
        except OSError:
            template = "你是一个说话简短的语音陪伴角色。"
        sys_text = system_prompt(template, name=ch.name, age="", identity="",
                                 relation="")
        # 角色人设必须真正进 system prompt，否则 LLM 只会按通用模板说话
        persona = (getattr(ch, "persona", "") or "").strip()
        if persona:
            sys_text = (sys_text.rstrip() +
                        f"\n\n【你的设定：{ch.name}】\n" + persona +
                        "\n\n以上设定优先于前面的通用规则；说话要像这个人，不要解释设定本身。")
        greeting = (getattr(ch, "greeting", "") or "").strip()
        if greeting:
            sys_text += f"\n你的开场白是：「{greeting}」（只在第一次见面时用，不要每轮重复）。"
        if self._should_search(user_text):
            try:
                from app.core import websearch

                # 先说一句"稍等"，趁这句话合成/播放的时间去查资料
                filler = str(self.config.get("web.filler", "稍等，让我查一下。"))
                if filler and self._tts_queue is not None:
                    try:
                        self._tts_queue.put_nowait(filler)
                    except Exception:
                        pass

                results = websearch.search(user_text, self.config)
                if results:
                    sys_text += "\n\n" + websearch.format_context(results)
                    self._emit("info", text=f"已联网检索 {len(results)} 条资料")
                    self._web_sources = [r["url"] for r in results]
            except Exception as exc:  # noqa: BLE001
                logger.warning("联网检索失败（忽略）：%s", exc)
        style = self._style()
        if style.get("hint"):
            sys_text += f"\n\n【本轮情绪】{style['name']}：{style['hint']}"
        msgs = [{"role": "system", "content": sys_text.strip()}]
        budget = int(self.config.get("llm.token_budget", 1500))
        msgs += self.history.context(budget_chars=budget)
        msgs.append({"role": "user", "content": user_text})
        return msgs

    def _run_turn(self, samples: np.ndarray | None, sr: int,
                  preset_text: str | None = None) -> None:
        ch = self.character
        m = TurnMetrics(char_id=ch.id if ch else "",
                        t_press=self._t_press, t_release=self._t_release)
        self._turn_audio = []
        self._web_sources = []
        cancel = self._cancel
        try:
            # ---- ASR ----
            if preset_text is not None:
                user_text = preset_text
                self._t_asr = time.perf_counter()
            else:
                # 只有真要用 ASR 时才等它预热完（文字输入不需要）
                if self._asr_warming and not self._asr_ready.wait(timeout=90):
                    self._emit("error", text="识别模型还在加载，请再说一次")
                    self.metrics.append(m)
                    self._set_state(State.IDLE)
                    return
                y16 = to_16k(samples, sr)
                res = self.asr.transcribe(y16)
                self._t_asr = time.perf_counter()
                if not res.ok:
                    self._emit("error", text="没听清，再说一次")
                    self._emit("turn_done", detail="asr_empty")
                    self._set_state(State.IDLE)
                    self.metrics.append(m)
                    return
                user_text = res.text
            m.text_len = len(user_text)
            self._emit("user_text", text=user_text)
            if cancel.is_set():
                return

            # ---- LLM + TTS ----
            self._set_state(State.THINKING, user_text[:24])
            self.player.begin_utterance()      # 打开输出流，第一块数据到就出声
            self._first_play_logged = True
            self._start_tts_worker()
            splitter = SentenceSplitter(
                int(self.config.get("pipeline.min_sentence_chars", 6)),
                int(self.config.get("pipeline.max_sentence_chars", 40)))
            reply_parts: list[str] = []
            first_token = False
            provider = ""
            first_play = False
            tts_fail_streak = 0
            max_fail = int(self.config.get("pipeline.max_tts_fail_streak", 2))

            try:
                for delta in self.llm.chat_stream(
                        self._messages(user_text),
                        temperature=float((ch.llm or {}).get("temperature", 0.8)),
                        max_tokens=(768 if self._web_sources else
                                    int((ch.llm or {}).get("max_tokens", 320))),
                        sampling=self._style(),
                        cancel=cancel):
                    if cancel.is_set():
                        return
                    if not first_token:
                        first_token = True
                        self._t_llm = time.perf_counter()
                        provider = delta.provider_id
                        self._emit("info", text=f"provider={provider}")
                    reply_parts.append(delta.text)
                    self._emit("reply_delta", text=delta.text)
                    for sent in splitter.feed(delta.text):
                        if not self._enqueue_sentence(sent, cancel):
                            tts_fail_streak += 1
                            if tts_fail_streak >= max_fail:
                                raise RuntimeError("连续两句话合成失败")
                        else:
                            tts_fail_streak = 0
                        first_play = self._maybe_play()
                for sent in splitter.flush():
                    self._enqueue_sentence(sent, cancel)
                self._finish_tts()
            except LLMError:
                # 全部 provider 失败 → 用 TTS 说兜底句
                fallback = str(self.config.get("llm.fallback_text",
                                               "我这会儿有点接不上话。"))
                self._emit("error", text="LLM 全部不可用，播兜底句")
                self._start_tts_worker()
                self._enqueue_sentence(fallback, cancel)
                self._finish_tts()
                reply_parts.append(fallback)
                self._t_llm = self._t_llm or time.perf_counter()

            # ---- 等第一块音频 ----
            # 注意：_finish_tts 只是发信号，合成还在后台跑，这里必须等首块
            deadline = time.perf_counter() + 25.0
            while self._t_tts == 0.0 and not cancel.is_set():
                if time.perf_counter() > deadline:
                    break
                t = self._tts_thread
                if t is not None and not t.is_alive():
                    break
                time.sleep(0.02)
            if self._t_tts == 0.0 and not cancel.is_set():
                self._emit("error", text="语音合成超时，已重置引擎，请再说一次")
                self._reset_tts()
                m.error = "no_audio"
                self.metrics.append(m)
                self._set_state(State.IDLE)
                return
            self._set_state(State.SPEAKING)
            timeout = 90.0
            while not cancel.is_set() and timeout > 0:
                alive = self._tts_thread is not None and self._tts_thread.is_alive()
                if not alive and not self.player.playing:
                    break
                timeout -= 0.02
                time.sleep(0.02)
            self._join_tts()
            if cancel.is_set():
                m.interrupted = True
                return
            m.t_play_end = time.perf_counter()
            m.t_llm_first_token = self._t_llm
            m.t_tts_first_chunk = self._t_tts
            m.t_play_start = self.player.t_play_start or 0.0
            m.t_asr_done = self._t_asr
            m.provider = provider
            reply = clean_for_tts("".join(reply_parts))
            m.reply_len = len(reply)
            m.audio_sec = round((m.t_play_end - (m.t_play_start or m.t_play_end)), 3)
            if self.player.underruns:
                logger.warning("本轮播放欠载 %d 次（可能造成重复/断音）",
                               self.player.underruns)
                self._emit("info",
                           text=f"播放欠载 {self.player.underruns} 次（检查音频设备）")
            self.history.add("user", user_text)
            self.history.add("assistant", reply)
            self.metrics.append(m)
            self._save_turn_audio(m, user_text, reply)
            self._emit("turn_done", text=reply)
            self._set_state(State.IDLE)
        except Exception as exc:  # noqa: BLE001
            m.error = f"{type(exc).__name__}: {exc}"
            self.metrics.append(m)
            self._emit("error", text=f"这一轮出错了：{exc}")
            self._set_state(State.IDLE)

    # -- TTS 工作线程 ---------------------------------------------------
    def _start_tts_worker(self) -> None:
        self._finish_tts_quick()
        if self._tts_thread is not None and self._tts_thread.is_alive():
            self._emit("error", text="上一轮语音还没合成完，请稍等一秒再说")
            self._tts_queue = None
            return
        self._tts_stop.clear()
        self._tts_queue = queue.Queue(
            maxsize=max(1, int(self.config.get("pipeline.prefetch_max", 2))))
        self._tts_thread = threading.Thread(target=self._tts_loop, daemon=True,
                                            name="vlab-tts")
        self._tts_thread.start()

    def _enqueue_sentence(self, sentence: str, cancel: threading.Event) -> bool:
        text = clean_for_tts(sentence)
        if not text:
            return True
        q = self._tts_queue
        if q is None:
            return False
        deadline = time.perf_counter() + 15.0
        while not cancel.is_set() and time.perf_counter() < deadline:
            try:
                q.put(text, timeout=0.05)
                return True
            except queue.Full:
                if self._tts_thread is None or not self._tts_thread.is_alive():
                    return False
                continue
        return False

    def _synth_once(self, text: str, ref: str, ref_text: str,
                    params: dict[str, Any] | None = None,
                    cancel: threading.Event | None = None
                    ) -> tuple[np.ndarray, int] | None:
        """合成一句，返回 (音频, 采样率)；用于异常时长时的重试。"""
        buf: list[np.ndarray] = []
        sr = 24000
        gen = None
        try:
            with self._tts_lock:
                engine = self._ensure_tts()
                gen = engine.synthesize_stream(text, ref, ref_text, **(params or {}))
                for chunk in gen:
                    if cancel is not None and cancel.is_set():
                        return None
                    buf.append(np.asarray(chunk.samples, np.float32))
                    sr = int(chunk.samplerate or 24000)
        except Exception as exc:  # noqa: BLE001
            self._emit("error", text=f"TTS 重试失败：{exc}")
            return None
        finally:
            if gen is not None:
                try:
                    gen.close()
                except Exception:
                    pass
        if not buf:
            return None
        return np.concatenate(buf), sr

    def _tts_loop(self) -> None:
        ch = self.character
        ref = str(self.store.dir_of(ch.id) / (ch.voice.get("reference") or "reference.wav"))
        ref_text = ch.voice.get("reference_text", "")
        failed = 0
        # 先攒够 2 句再开播：合成比实时慢，只攒 1 句的话第二句必然等不到位
        held: list[tuple[np.ndarray, int]] = []
        play_started = False
        last_lang = ""

        def flush_held() -> None:
            nonlocal play_started
            for data, sr in held:
                self.player.push(data, sr)
                self.player.push_gap()
                self._turn_audio.append((data, sr))
            held.clear()
            play_started = True

        def offer(data: np.ndarray, sr: int) -> None:
            nonlocal play_started, last_lang
            lang = detect_lang(text) if text else "zh"
            if last_lang and lang != last_lang:
                # 换语言时插 220ms 停顿，听起来更自然
                held.append((np.zeros(int(sr * 0.22), np.float32), sr))
            last_lang = lang
            held.append((data, sr))
            if play_started or len(held) >= 2:
                flush_held()

        while True:
            q = self._tts_queue
            if q is None:
                flush_held()
                return
            try:
                text = q.get(timeout=0.03)
            except queue.Empty:
                if self._tts_stop.is_set() or self._cancel.is_set():
                    flush_held()
                    return
                continue
            if text is None:
                flush_held()
                return
            if self._cancel.is_set():
                return
            t_start = time.perf_counter()
            chunks = 0
            gen = None
            buf: list[np.ndarray] = []
            sr_out = 24000
            try:
                with self._tts_lock:
                    engine = self._ensure_tts()
                    self._set_state(State.SPEAKING, text[:20])
                    gen = engine.synthesize_stream(text, ref, ref_text)
                    try:
                        for chunk in gen:
                            if self._cancel.is_set():
                                self._emit("info",
                                           text=f"TTS 被取消（{text[:12]}…，"
                                                f"{chunks} 块 / "
                                                f"{time.perf_counter()-t_start:.1f}s）")
                                break
                            if self._t_tts == 0.0:
                                self._t_tts = time.perf_counter()
                            chunks += 1
                            sr_out = int(chunk.samplerate or 24000)
                            # 整句缓冲后再交给播放器：v3 的合成比实时慢（RTF>1），
                            # 逐块推会持续欠载，听起来就是"断断续续的颗粒感"。
                            buf.append(np.asarray(chunk.samples, np.float32))
                    finally:
                        # 关键：中断时显式关掉生成器，让上游 finally 释放
                        # CUDA Graph / KV 状态；不关的话下一句会在推理里卡死。
                        if gen is not None:
                            try:
                                gen.close()
                            except Exception:
                                pass
                            gen = None
                if self._cancel.is_set():
                    flush_held()
                    return
                if buf:
                    data = np.concatenate(buf)
                    # 去掉模型自带的 ~300ms 首尾静音，否则句间会叠出明显空洞
                    data = _trim_edges(data, sr_out)
                    # 异常延音保护：v4 偶发把语气词拖长/重复（实测 11s），
                    # 按字数估个上限截断，免得一句话卡住半天。
                    per_char = float(self.config.get("tts.max_sec_per_char", 0.55))
                    cap = per_char * max(4, len(text)) + 1.0
                    peak = float(np.max(np.abs(data))) if data.size else 0.0
                    alt_params = [
                        {"seed": int(time.time() * 1000) % 100000},
                        {"seed": int(time.time() * 1000 + 7919) % 100000,
                         "sample_steps": 8, "top_k": 15, "temperature": 0.7},
                        {"seed": int(time.time() * 1000 + 15887) % 100000,
                         "sample_steps": 4, "top_k": 10, "temperature": 0.8},
                    ]
                    for attempt, extra in enumerate(alt_params):
                        if peak >= 0.02:
                            break
                        self._emit("info",
                                   text=f"TTS 输出过轻（{peak:.3f}），第 {attempt+1} 次换参重试")
                        alt = self._synth_once(text, ref, ref_text, extra,
                                               self._cancel)
                        if alt is not None and alt[0].size:
                            cand_peak = float(np.max(np.abs(alt[0])))
                            if cand_peak > peak:
                                data, sr_out, peak = alt[0], alt[1], cand_peak
                    if peak < 0.02:
                        self._emit("error", text=f"这句合成不出声音，已跳过：{text[:12]}…")
                        failed += 1
                        continue
                    if data.size / sr_out > cap:
                        # 先用更"放开"的采样重试一次，实在不行才截断
                        self._emit("info",
                                   text=f"TTS 输出偏长（{data.size/sr_out:.1f}s），用宽松采样重试")
                        alt = self._synth_once(
                            text, ref, ref_text,
                            {"top_k": int(self.config.get("tts.retry_top_k", 15)),
                             "temperature": float(self.config.get("tts.retry_temperature", 0.7))},
                            cancel)
                        if alt is not None and 0 < alt[0].size < buf and \
                                alt[0].size / alt[1] < data.size / sr_out:
                            data, sr_out = alt
                    if data.size / sr_out > cap * 1.3:
                        self._emit("info",
                                   text=f"TTS 仍偏长（{data.size/sr_out:.1f}s），截断到 {cap:.1f}s")
                        data = data[: int(cap * sr_out)]
                    # 句首淡入、句尾长淡出：避免硬切与句尾残响被当成"底噪"
                    fi = float(self.config.get("tts.sentence_fade_in_ms", 12.0))
                    fo = float(self.config.get("tts.sentence_fade_out_ms", 25.0))
                    data = audio_qc.fade_asym(data, sr_out, fi, fo)
                    offer(data, sr_out)
                failed = 0
                self._emit("info", text=f"TTS 完成：{text[:12]}… "
                                        f"{chunks} 块 / {time.perf_counter()-t_start:.1f}s")
            except Exception as exc:  # noqa: BLE001
                failed += 1
                self._emit("error", text=f"TTS 失败（{failed}）：{exc}")
                if failed >= 2:
                    self._reset_tts()

    def _save_turn_audio(self, m: Any, user_text: str, reply: str) -> None:
        """把这一轮实际播放的音频存下来，方便事后用播放器反复检查音质。"""
        if not bool(self.config.get("audio.save_tts", True)):
            self._turn_audio = []
            return
        parts = list(self._turn_audio)
        self._turn_audio = []
        if not parts:
            return
        try:
            sr = parts[0][1]
            gap = np.zeros(int(sr * 0.08), np.float32)
            chunks: list[np.ndarray] = []
            for i, (data, _sr) in enumerate(parts):
                if i:
                    chunks.append(gap)
                chunks.append(data)
            audio = np.concatenate(chunks) if chunks else np.zeros(0, np.float32)
            root = Path(self.config.path_of("logs")).parent
            out_dir = root / str(self.config.get("audio.save_tts_dir", "results/tts"))
            out_dir.mkdir(parents=True, exist_ok=True)
            ts = time.strftime("%Y%m%d-%H%M%S")
            cid = (self.character.id if self.character else "na")[:6]
            base = out_dir / f"{ts}_{cid}_{int(m.e2e_ms)}ms"
            audio_qc.save_wav(base.with_suffix(".wav"), audio, sr, subtype="PCM_16")
            meta = {
                "time": ts, "character": getattr(self.character, "name", ""),
                "user": user_text, "reply": reply,
                "sample_rate": sr, "audio_sec": round(audio.size / sr, 3),
                "sentences": len(parts),
                "asr_ms": getattr(m, "asr_ms", None),
                "llm_ttft_ms": getattr(m, "llm_ttft_ms", None),
                "tts_first_ms": getattr(m, "tts_first_ms", None),
                "e2e_ms": getattr(m, "e2e_ms", None),
                "provider": getattr(m, "provider", None), "style": self.style_id,
            }
            base.with_suffix(".json").write_text(
                json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
            self._rotate_tts(out_dir)
        except Exception as exc:  # noqa: BLE001
            logger.warning("保存本轮音频失败：%s", exc)

    def _rotate_tts(self, out_dir: Path) -> None:
        ttl_min = float(self.config.get("audio.tts_ttl_minutes", 10))
        if ttl_min > 0:
            cutoff = time.time() - ttl_min * 60
            for old in out_dir.glob("*"):
                try:
                    if old.stat().st_mtime < cutoff:
                        old.unlink(missing_ok=True)
                except OSError:
                    pass
        keep = int(self.config.get("audio.keep_tts", 200))
        if keep <= 0:
            return
        wavs = sorted(out_dir.glob("*.wav"), key=lambda p: p.stat().st_mtime)
        for old in wavs[:-keep]:
            try:
                old.unlink(missing_ok=True)
                old.with_suffix(".json").unlink(missing_ok=True)
            except OSError:
                pass

    def _finish_tts(self) -> None:
        """只发“没有更多句子”的信号，不阻塞——合成还要继续跑，播放同时进行。"""
        q = self._tts_queue
        if q is None:
            return
        self._tts_stop.set()
        deadline = time.perf_counter() + 5.0
        while time.perf_counter() < deadline:
            try:
                q.put_nowait(None)
                return
            except queue.Full:
                time.sleep(0.01)

    def _join_tts(self, timeout: float = 30.0) -> None:
        t = self._tts_thread
        if t is not None and t.is_alive():
            t.join(timeout=timeout)
        self._tts_thread = None

    def _reset_tts(self) -> None:
        """把语音引擎整个丢掉，下次用到时重新加载。

        用于 TTS 卡死/连续失败后自救：v3 LoRA 的 CUDA Graph 一旦状态坏掉，
        继续复用只会一直失败，重载（约 12s）比一直报错好。
        """
        t = self._tts_thread
        if t is not None and t.is_alive():
            self._emit("info", text="TTS 线程仍在运行，暂不重载引擎")
            return
        with self._tts_lock:
            try:
                if self._tts is not None:
                    self._tts.unload()
            except Exception:
                pass
            self._tts = None
            self._tts_key = ("", "")
        self._emit("info", text="语音引擎已重置，下一轮会重新加载")

    def _finish_tts_quick(self) -> None:
        self._tts_stop.set()
        q = self._tts_queue
        if q is not None:
            try:
                q.put_nowait(None)
            except queue.Full:
                pass
        t = self._tts_thread
        if t is not None and t.is_alive():
            t.join(timeout=10)
        if t is not None and t.is_alive():
            self._emit("info", text="TTS 线程 10 秒内没退出，标记引擎为待重置")
        self._tts_thread = None
        self._tts_queue = None

    def _maybe_play(self) -> bool:
        """第一块音频进缓冲后立刻开播，返回是否已经开播。"""
        if self.player.buffered_sec > 0:
            if not self._first_play_logged:
                self._first_play_logged = True
                self.player.begin_utterance()
            return True
        return False


__all__ = ["Pipeline", "PipelineEvent", "State", "SentenceSplitter",
           "clean_for_tts"]
