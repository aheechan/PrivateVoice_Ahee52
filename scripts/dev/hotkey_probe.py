"""热键探针：注册 shift+a，用 SendInput 合成按下/松开，统计可靠性与延迟。

由于 RegisterHotKey 会截走该组合键，合成按键不会泄漏到其它窗口。
"""
from __future__ import annotations

import ctypes
import ctypes.wintypes as wt
import statistics
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core.hotkey import HotkeyListener, parse_spec  # noqa: E402

user32 = ctypes.WinDLL("user32", use_last_error=True)
KEYEVENTF_KEYUP = 0x0002
INPUT_KEYBOARD = 1


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [("wVk", wt.WORD), ("wScan", wt.WORD), ("dwFlags", wt.DWORD),
                ("time", wt.DWORD), ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong))]


class _INPUTunion(ctypes.Union):
    _fields_ = [("ki", KEYBDINPUT), ("padding", ctypes.c_byte * 32)]


class INPUT(ctypes.Structure):
    _fields_ = [("type", wt.DWORD), ("u", _INPUTunion)]


def send(vk: int, up: bool = False) -> None:
    flags = KEYEVENTF_KEYUP if up else 0
    inp = INPUT(type=INPUT_KEYBOARD)
    inp.u.ki = KEYBDINPUT(wVk=vk, wScan=0, dwFlags=flags, time=0, dwExtraInfo=None)
    user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))


VK_SHIFT, VK_A = 0x10, ord("A")


def main() -> int:
    spec = parse_spec("shift+a")
    print(f"spec: {spec.label} mods={spec.mods:#x} vk={spec.vk:#x}")

    presses: list[float] = []
    releases: list[float] = []
    errors: list[str] = []
    done = threading.Event()

    hk = HotkeyListener(
        "shift+a",
        on_press=lambda: presses.append(time.perf_counter()),
        on_release=lambda: (releases.append(time.perf_counter()), done.set()),
        poll_ms=20, min_hold_ms=150,
        on_error=lambda m: (errors.append(m), done.set()),
    )
    hk.start()
    if errors:
        print("REGISTER_FAILED:", errors[0])
        return 1
    print("registered:", hk.registered)
    if not hk.registered:
        print("REGISTER_TIMEOUT")
        return 1

    # 1) 正常长按 300ms
    t_send = time.perf_counter()
    send(VK_SHIFT)
    send(VK_A)
    time.sleep(0.30)
    send(VK_A, up=True)
    send(VK_SHIFT, up=True)
    done.wait(3.0)
    if len(presses) == 1 and len(releases) == 1:
        print(f"long-press  press_detect={1000*(presses[0]-t_send):.1f}ms "
              f"hold={1000*(releases[0]-presses[0]):.1f}ms")
    else:
        print(f"long-press FAILED presses={len(presses)} releases={len(releases)}")
        return 1

    # 2) 误触 60ms（低于 min_hold，应只有 press 没有 release）
    done.clear()
    p0, r0 = len(presses), len(releases)
    send(VK_SHIFT)
    send(VK_A)
    time.sleep(0.06)
    send(VK_A, up=True)
    send(VK_SHIFT, up=True)
    done.wait(1.0)
    short_ok = (len(presses) == p0 + 1 and len(releases) == r0)
    print(f"short-tap   presses={len(presses)-p0} releases={len(releases)-r0} "
          f"-> {'OK（误触被过滤）' if short_ok else 'FAIL'}")

    # 3) 连续 5 次长按，统计抖动
    lat, hold = [], []
    for _ in range(5):
        done.clear()
        p_before, r_before = len(presses), len(releases)
        t0 = time.perf_counter()
        send(VK_SHIFT)
        send(VK_A)
        time.sleep(0.25)
        send(VK_A, up=True)
        send(VK_SHIFT, up=True)
        done.wait(2.0)
        if len(presses) > p_before and len(releases) > r_before:
            lat.append(1000 * (presses[-1] - t0))
            hold.append(1000 * (releases[-1] - presses[-1]))
        time.sleep(0.4)

    print(f"repeat      {len(lat)}/5 ok | press p50={statistics.median(lat):.1f}ms "
          f"max={max(lat):.1f}ms | hold p50={statistics.median(hold):.1f}ms "
          f"(发送 250ms)")

    hk.stop()
    ok = (short_ok and len(lat) == 5 and max(lat) < 120 and statistics.median(hold) < 400)
    print("HOTKEY_PROBE", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
