"""全局热键：按下=开始说话，松开=结束。

为什么不是纯 RegisterHotKey：
    Win32 的 RegisterHotKey 只在**按下**时投递 WM_HOTKEY，没有 key-up 事件。
    所以这里用「RegisterHotKey 拿按下 + GetAsyncKeyState 轮询松开」的组合，
    不需要管理员权限，也不用装 pywin32（纯 ctypes）。

注意：RegisterHotKey 会全局截走这个组合键。默认 `shift+a` 意味着在别的程序里
    打大写 A 也会触发——设置页可改，本模块只负责如实执行配置。
"""
from __future__ import annotations

import ctypes
import ctypes.wintypes as wt
import threading
import time
from dataclasses import dataclass
from typing import Callable

user32 = ctypes.WinDLL("user32", use_last_error=True)

MOD_ALT = 0x0001
MOD_CONTROL = 0x0002
MOD_SHIFT = 0x0004
MOD_WIN = 0x0008
MOD_NOREPEAT = 0x4000

WM_HOTKEY = 0x0312
WM_QUIT = 0x0012

MOD_NAMES = {
    "ctrl": MOD_CONTROL, "control": MOD_CONTROL,
    "alt": MOD_ALT, "shift": MOD_SHIFT, "win": MOD_WIN,
}

VK_NAMES = {
    "space": 0x20, "enter": 0x0D, "return": 0x0D, "tab": 0x09,
    "esc": 0x1B, "escape": 0x1B, "backspace": 0x08,
    "up": 0x26, "down": 0x28, "left": 0x25, "right": 0x27,
    "insert": 0x2D, "delete": 0x2E, "home": 0x24, "end": 0x23,
    "pageup": 0x21, "pagedown": 0x22,
    "`": 0xC0, "-": 0xBD, "=": 0xBB, "[": 0xDB, "]": 0xDD,
    "\\": 0xDC, ";": 0xBA, "'": 0xDE, ",": 0xBC, ".": 0xBE, "/": 0xBF,
}
for _i in range(1, 25):
    VK_NAMES[f"f{_i}"] = 0x6F + _i


@dataclass
class HotkeySpec:
    mods: int
    vk: int
    label: str
    passive: bool = False     # True = 只按修饰键（如 shift），用轮询而不是 RegisterHotKey

    @property
    def mod_keys(self) -> list[int]:
        keys = []
        if self.mods & MOD_CONTROL:
            keys.append(0x11)      # VK_CONTROL
        if self.mods & MOD_ALT:
            keys.append(0x12)      # VK_MENU
        if self.mods & MOD_SHIFT:
            keys.append(0x10)      # VK_SHIFT
        if self.mods & MOD_WIN:
            keys += [0x5B, 0x5C]
        return keys


class HotkeyError(RuntimeError):
    pass


def parse_spec(spec: str) -> HotkeySpec:
    raw = (spec or "").strip().lower().replace(" ", "")
    if not raw:
        raise HotkeyError("热键为空")
    parts = raw.split("+")
    mods = 0
    key = ""
    for part in parts:
        if part in MOD_NAMES:
            mods |= MOD_NAMES[part]
        elif part:
            key = part
    if not key:
        # 只按修饰键（shift / ctrl+alt）：RegisterHotKey 不支持单独注册修饰键，
        # 改用轮询检测按下/松开。
        if not mods:
            raise HotkeyError(f"热键 {spec!r} 无法解析")
        if mods == MOD_SHIFT:
            vk, label = 0x10, "SHIFT"
        elif mods == MOD_CONTROL:
            vk, label = 0x11, "CTRL"
        elif mods == MOD_ALT:
            vk, label = 0x12, "ALT"
        elif mods == MOD_WIN:
            vk, label = 0x5B, "WIN"
        else:
            vk = 0x10
            label = "+".join(p.upper() for p in parts if p in MOD_NAMES)
        return HotkeySpec(mods=mods, vk=vk, label=label, passive=True)
    if len(key) == 1 and key.isalnum():
        vk = ord(key.upper())
    elif key in VK_NAMES:
        vk = VK_NAMES[key]
    elif key.isdigit():
        vk = ord(key)
    else:
        raise HotkeyError(f"不认识的按键：{key}")
    label = "+".join([p for p in parts if p in MOD_NAMES] + [key.upper()])
    return HotkeySpec(mods=mods, vk=vk, label=label)


VK_TO_NAME = {v: k for k, v in VK_NAMES.items()}


def spec_from_key(mods: set[str], key: str) -> str:
    """由设置页的按键捕获生成热键字符串，例如 {'ctrl','alt'} + 'space'。"""
    order = ["ctrl", "alt", "shift", "win"]
    parts = [m for m in order if m in mods]
    if key:
        parts.append(key.lower())
    return "+".join(parts)


class HotkeyListener:
    """注册一个全局热键，回调 press / release。

    回调发生在监听线程里，UI 侧必须自己 marshal 回主线程（Qt 信号即可）。
    """

    _next_id = 0x5A01

    def __init__(self, spec: str, on_press: Callable[[], None],
                 on_release: Callable[[], None],
                 poll_ms: int = 20, min_hold_ms: int = 150,
                 on_error: Callable[[str], None] | None = None):
        self.spec = parse_spec(spec)
        self.on_press = on_press
        self.on_release = on_release
        self.on_error = on_error
        self.poll_ms = max(5, int(poll_ms))
        self.min_hold_ms = max(0, int(min_hold_ms))
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._thread_id: int | None = None
        self._registered = False
        self._press_ts: float | None = None
        HotkeyListener._next_id += 1
        self._id = HotkeyListener._next_id

    # -- 生命周期 ------------------------------------------------------
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="vlab-hotkey",
                                        daemon=True)
        self._thread.start()
        # 等注册结果，最多 1.5s
        deadline = time.time() + 1.5
        while time.time() < deadline and self._thread.is_alive():
            if self._registered:
                return
            time.sleep(0.02)

    @property
    def registered(self) -> bool:
        return self._registered

    def stop(self) -> None:
        self._stop.set()
        if self._thread_id:
            try:
                user32.PostThreadMessageW(self._thread_id, WM_QUIT, 0, 0)
            except Exception:
                pass
        self._thread = None
        self._registered = False

    # -- 线程体 --------------------------------------------------------
    def _run(self) -> None:
        self._thread_id = ctypes.windll.kernel32.GetCurrentThreadId()
        if self.spec.passive:
            self._run_passive()
            return
        ok = user32.RegisterHotKey(None, self._id,
                                   self.spec.mods | MOD_NOREPEAT, self.spec.vk)
        if not ok:
            err = ctypes.get_last_error()
            msg = (f"热键 {self.spec.label} 注册失败（错误码 {err}）："
                   f"可能已被其它程序占用，或需要换一个组合键。")
            self._safe_error(msg)
            return
        self._registered = True
        msg = wt.MSG()
        try:
            while not self._stop.is_set():
                got = user32.GetMessageW(ctypes.byref(msg), None, 0, 0)
                if got == 0 or got == -1:
                    break
                if msg.message == WM_HOTKEY and int(msg.wParam) == self._id:
                    self._handle_press()
        finally:
            try:
                user32.UnregisterHotKey(None, self._id)
            except Exception:
                pass
            self._registered = False

    def _run_passive(self) -> None:
        """只按修饰键（如 shift）时用：轮询检测按下/松开，不抢占系统热键。"""
        self._registered = True
        keys = self.spec.mod_keys
        was_down = False
        try:
            while not self._stop.is_set():
                down = all(bool(user32.GetAsyncKeyState(k) & 0x8000) for k in keys)
                if down and not was_down:
                    self._press_ts = time.perf_counter()
                    self._safe(self.on_press)
                elif not down and was_down:
                    held = (time.perf_counter() - (self._press_ts or
                                                   time.perf_counter())) * 1000
                    self._press_ts = None
                    if held >= self.min_hold_ms:
                        self._safe(self.on_release)
                was_down = down
                time.sleep(self.poll_ms / 1000.0)
        finally:
            self._registered = False

    def _handle_press(self) -> None:
        if self._press_ts is None:
            self._press_ts = time.perf_counter()
            self._safe(self.on_press)
        # 起一个轮询线程等松开
        threading.Thread(target=self._wait_release, daemon=True,
                         name="vlab-hotkey-release").start()

    def _wait_release(self) -> None:
        vk = self.spec.vk
        mod_keys = self.spec.mod_keys
        while not self._stop.is_set():
            down = bool(user32.GetAsyncKeyState(vk) & 0x8000)
            mods_down = any(user32.GetAsyncKeyState(k) & 0x8000 for k in mod_keys) \
                if mod_keys else True
            if not down or not mods_down:
                break
            time.sleep(self.poll_ms / 1000.0)
        held = (time.perf_counter() - (self._press_ts or time.perf_counter())) * 1000
        # 太短的按压当作误触，不触发一轮对话
        self._press_ts = None
        if held >= self.min_hold_ms:
            self._safe(self.on_release)

    def _safe(self, fn: Callable[[], None]) -> None:
        try:
            fn()
        except Exception:
            pass

    def _safe_error(self, message: str) -> None:
        if self.on_error:
            try:
                self.on_error(message)
            except Exception:
                pass


__all__ = ["HotkeyListener", "HotkeySpec", "HotkeyError", "parse_spec",
           "MOD_ALT", "MOD_CONTROL", "MOD_SHIFT", "MOD_WIN"]
