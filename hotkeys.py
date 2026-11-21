"""Optional global hotkeys (Windows only, no extra dependencies). Works while OBS/game is focused."""
from __future__ import annotations

import os
import threading
from typing import Callable

# action -> (label, modifiers, virtual-key)   MOD_ALT=1, MOD_CONTROL=2
HOTKEYS: dict[str, tuple[str, int, int]] = {
    "next_review": ("Ctrl+Alt+N", 3, ord("N")),
    "stop_current": ("Ctrl+Alt+S", 3, ord("S")),
    "pause_voice": ("Ctrl+Alt+P", 3, ord("P")),
    "skip_next": ("Ctrl+Alt+K", 3, ord("K")),
}
WM_HOTKEY, WM_QUIT = 0x0312, 0x0012


class HotkeyManager:
    def __init__(self, callbacks: dict[str, Callable[[], None]]) -> None:
        self.callbacks = callbacks
        self.thread: threading.Thread | None = None
        self.thread_id = 0
        self.registered: dict[str, bool] = {}
        self.ready = threading.Event()

    @property
    def supported(self) -> bool:
        return os.name == "nt"

    def status(self) -> dict:
        return {
            "supported": self.supported, "active": bool(self.thread and self.thread.is_alive()),
            "keys": {a: {"label": HOTKEYS[a][0], "ok": self.registered.get(a, False)} for a in HOTKEYS},
        }

    def start(self) -> None:
        if not self.supported or (self.thread and self.thread.is_alive()):
            return
        self.ready.clear()
        self.thread = threading.Thread(target=self._loop, name="global-hotkeys", daemon=True)
        self.thread.start()
        self.ready.wait(2.0)

    def stop(self) -> None:
        if self.supported and self.thread and self.thread.is_alive() and self.thread_id:
            try:
                import ctypes
                ctypes.windll.user32.PostThreadMessageW(self.thread_id, WM_QUIT, 0, 0)
            except Exception:
                pass
            self.thread.join(1.5)
        self.thread = None
        self.registered = {}

    def _loop(self) -> None:
        import ctypes
        from ctypes import wintypes
        user32, kernel32 = ctypes.windll.user32, ctypes.windll.kernel32
        self.thread_id = kernel32.GetCurrentThreadId()
        ids: dict[int, str] = {}
        for n, (action, (_label, mods, vk)) in enumerate(HOTKEYS.items(), start=1):
            ok = bool(user32.RegisterHotKey(None, n, mods | 0x4000, vk))   # 0x4000 = MOD_NOREPEAT
            self.registered[action] = ok
            if ok:
                ids[n] = action
        self.ready.set()
        msg = wintypes.MSG()
        try:
            while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
                if msg.message == WM_HOTKEY and msg.wParam in ids:
                    cb = self.callbacks.get(ids[msg.wParam])
                    if cb:
                        try:
                            cb()
                        except Exception:
                            pass
        finally:
            for n in ids:
                user32.UnregisterHotKey(None, n)
