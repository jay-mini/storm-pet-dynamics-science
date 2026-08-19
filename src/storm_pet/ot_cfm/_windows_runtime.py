#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Windows runtime setup needed before importing PyTorch."""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path


_VC_RUNTIME_DLLS = ("vcruntime140.dll", "vcruntime140_1.dll", "msvcp140.dll")
_LOADED_HANDLES: list[ctypes.WinDLL] = []
_VC_RUNTIME_LOADED = False


def ensure_windows_vc_runtime_loaded() -> None:
    """
    Preload the system VC runtime before PyTorch loads c10.dll.

    Some Anaconda base installs keep older VC runtime DLLs beside python.exe.
    Windows loads those first, which can make newer PyTorch wheels fail during
    c10.dll initialization. Loading the System32 runtime first pins a compatible
    copy in the process before torch imports its native DLLs.
    """
    global _VC_RUNTIME_LOADED
    if _VC_RUNTIME_LOADED or not sys.platform.startswith("win"):
        return

    system_root = os.environ.get("SystemRoot", r"C:\Windows")
    system32 = Path(system_root) / "System32"
    for dll_name in _VC_RUNTIME_DLLS:
        dll_path = system32 / dll_name
        if dll_path.exists():
            _LOADED_HANDLES.append(ctypes.WinDLL(str(dll_path)))

    _VC_RUNTIME_LOADED = True


def configure_windows_torch_runtime() -> None:
    """Apply Windows-only environment and DLL setup before importing torch."""
    if not sys.platform.startswith("win"):
        return

    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    ensure_windows_vc_runtime_loaded()


