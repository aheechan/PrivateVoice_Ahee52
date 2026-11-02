"""对话编排：录音 → ASR → LLM → 切句 → TTS → 播放。

严格串行：同一时刻只有一轮在跑。任意时刻打断都会取消这一轮的所有子任务
（LLM 请求、TTS 合成、播放缓冲），并直接进入下一轮 listening。
"""
from __future__ import annotations

import queue
import re
import threading
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np

from . import audio_io
from .asr import ASREngine, to_16k
from .history import HistoryStore, system_prompt
from .llm import LLMClient, LLMError
from .metrics import MetricsStore, TurnMetrics
from .player import Player
from .tts import TTSEngine, make_engine

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
            elif len(self.buf) >= self.max_chars:
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
    s = _SPACE.sub("", s)
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
            samplerate=48000,
            blocksize_ms=int(config.get("player.blocksize_ms", 30)),
            sentence_gap_ms=int(config.get("player.sentence_gap_ms", 80)),
            limiter_dbtp=float(config.get("player.limiter_dbtp", -1.0)),
        )
        self.recorder = audio_io.Recorder(
            samplerate=48000, device=config.audio.get("input_device"))
        self._tts: TTSEngine | None = None
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
        if data is None or len(data) < 1600:
            self._emit("error", text="没听清，再说一次")
            self._set_state(State.IDLE)
            return
        self._t_release = t
        self.submit_audio(data, 48000)

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
                        max_tokens=int((ch.llm or {}).get("max_tokens", 160)),
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
            timeout = 60.0
            while self.player.playing and not cancel.is_set():
                timeout -= 0.02
                if timeout <= 0:
                    break
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
            self.history.add("user", user_text)
            self.history.add("assistant", reply)
            self.metrics.append(m)
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

    def _tts_loop(self) -> None:
        ch = self.character
        ref = str(self.store.dir_of(ch.id) / (ch.voice.get("reference") or "reference.wav"))
        ref_text = ch.voice.get("reference_text", "")
        failed = 0
        while True:
            q = self._tts_queue
            if q is None:
                return
            try:
                text = q.get(timeout=0.03)
            except queue.Empty:
                if self._tts_stop.is_set() or self._cancel.is_set():
                    return
                continue
            if text is None:
                return
            if self._cancel.is_set():
                return
            t_start = time.perf_counter()
            chunks = 0
            gen = None
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
                            self.player.push(chunk.samples, chunk.samplerate)
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
                    return
                self.player.push_gap()
                failed = 0
                self._emit("info", text=f"TTS 完成：{text[:12]}… "
                                        f"{chunks} 块 / {time.perf_counter()-t_start:.1f}s")
            except Exception as exc:  # noqa: BLE001
                failed += 1
                self._emit("error", text=f"TTS 失败（{failed}）：{exc}")
                if failed >= 2:
                    self._reset_tts()

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
