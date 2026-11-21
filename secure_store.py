"""Protect API keys at rest with Windows DPAPI (current-user only). Falls back to plain text
on other systems or if anything goes wrong, so it can never lock the user out."""
from __future__ import annotations

import base64
import os

PREFIX = "dpapi:"


def _crypt(data: bytes, encrypt: bool) -> bytes:
    import ctypes
    from ctypes import wintypes

    class BLOB(ctypes.Structure):
        _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_char))]

    crypt32, kernel32 = ctypes.windll.crypt32, ctypes.windll.kernel32
    fn = crypt32.CryptProtectData if encrypt else crypt32.CryptUnprotectData
    fn.argtypes = [ctypes.POINTER(BLOB), wintypes.LPCWSTR, ctypes.POINTER(BLOB), ctypes.c_void_p,
                   ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(BLOB)]
    fn.restype = wintypes.BOOL
    kernel32.LocalFree.argtypes = [ctypes.c_void_p]
    buf = ctypes.create_string_buffer(data, len(data))
    src = BLOB(len(data), ctypes.cast(buf, ctypes.POINTER(ctypes.c_char)))
    out = BLOB()
    if encrypt:
        ok = fn(ctypes.byref(src), "NinuGamingLiveAssistant", None, None, None, 0, ctypes.byref(out))
    else:
        ok = fn(ctypes.byref(src), None, None, None, None, 0, ctypes.byref(out))
    if not ok:
        raise OSError("Windows DPAPI call failed")
    try:
        return ctypes.string_at(out.pbData, out.cbData)
    finally:
        kernel32.LocalFree(ctypes.cast(out.pbData, ctypes.c_void_p))


def unprotect_secret(value: str) -> str:
    if not value or not value.startswith(PREFIX):
        return value or ""
    try:
        return _crypt(base64.b64decode(value[len(PREFIX):]), False).decode("utf-8")
    except Exception:
        return ""          # saved on another PC/user: treat as not set so the user can re-enter it


def protect_secret(value: str) -> str:
    if not value or value.startswith(PREFIX) or os.name != "nt":
        return value or ""
    try:
        token = PREFIX + base64.b64encode(_crypt(value.encode("utf-8"), True)).decode("ascii")
        return token if unprotect_secret(token) == value else value   # verify round-trip first
    except Exception:
        return value
