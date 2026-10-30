"""Host CPU-vendor / iGPU probe used to pick a compute device for local document OCR.

Why: marker-pdf/surya choose their torch device as cuda -> mps -> cpu only; an Intel iGPU
(torch "xpu") is never selected on its own, so on Intel client machines OCR/layout run on CPU
cores. This module detects that situation; ``recommend_torch_device`` only suggests ``xpu`` when
every link is verified: Intel CPU, an integrated Intel GPU actually present, and a torch build
whose ``torch.xpu`` is usable. Anything else returns None (= leave the library's own choice).

CPU vendor comes from the OS's CPUID-derived report (registry VendorIdentifier on Windows,
/proc/cpuinfo on Linux) rather than executing CPUID from Python. Never raises; cached.
Override: HERMES_DOC_DEVICE=auto|cpu|xpu|<any torch device>.
"""
from __future__ import annotations

import functools
import os
import platform
import re
import subprocess
import sys
from typing import Any, Optional

_DISCRETE_ARC = re.compile(r"\barc\b.*\b[ab]\d{3}\b", re.I)


def _cpu_info() -> dict:
    vendor = name = ""
    try:
        if sys.platform == "win32":
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
                vendor = winreg.QueryValueEx(k, "VendorIdentifier")[0]
                name = winreg.QueryValueEx(k, "ProcessorNameString")[0]
        elif os.path.exists("/proc/cpuinfo"):
            with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as f:
                for line in f:
                    key, _, val = line.partition(":")
                    key = key.strip()
                    if key == "vendor_id" and not vendor:
                        vendor = val.strip()
                    elif key == "model name" and not name:
                        name = val.strip()
                    if vendor and name:
                        break
        else:
            name = platform.processor()
    except Exception:
        pass
    return {"vendor": vendor.strip(), "name": name.strip(),
            "is_intel": vendor.strip() == "GenuineIntel" or "intel" in name.lower()}


def _is_integrated(name: str) -> bool:
    return not _DISCRETE_ARC.search(name)


def _intel_gpus() -> list[dict]:
    """Intel display adapters present on the host: [{name, integrated}]."""
    found: list[dict] = []
    try:
        if sys.platform == "win32":
            out = subprocess.run(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command",
                 "Get-CimInstance Win32_VideoController | ForEach-Object { $_.Name + '|' + $_.PNPDeviceID }"],
                capture_output=True, text=True, timeout=15).stdout
            for line in out.splitlines():
                name, _, pnp = line.partition("|")
                if "VEN_8086" in pnp.upper():
                    found.append({"name": name.strip(), "integrated": _is_integrated(name)})
        elif sys.platform.startswith("linux"):
            base = "/sys/class/drm"
            for card in sorted(os.listdir(base)) if os.path.isdir(base) else []:
                if not re.fullmatch(r"card\d+", card):
                    continue
                dev = os.path.join(base, card, "device")
                try:
                    with open(os.path.join(dev, "vendor")) as f:
                        if f.read().strip() != "0x8086":
                            continue
                except OSError:
                    continue
                slot = os.path.basename(os.path.realpath(dev))
                # Intel iGPUs sit at PCI 00:02.0 on every client platform; dGPUs sit elsewhere.
                found.append({"name": f"Intel GPU ({slot})", "integrated": slot.endswith(":00:02.0")})
    except Exception:
        pass
    return found


def _openvino_devices() -> list[str]:
    try:
        from openvino import Core
        return list(Core().available_devices)
    except Exception:
        return []


def _torch_xpu() -> bool:
    try:
        import torch
        return bool(getattr(torch, "xpu", None) and torch.xpu.is_available())
    except Exception:
        return False


@functools.lru_cache(maxsize=1)
def probe() -> dict[str, Any]:
    cpu = _cpu_info()
    gpus = _intel_gpus() if cpu["is_intel"] else []
    return {
        "cpu": cpu,
        "intel_gpus": gpus,
        "intel_igpu_present": any(g["integrated"] for g in gpus),
        "openvino_devices": _openvino_devices(),
        "torch_xpu": _torch_xpu(),
    }


def recommend_torch_device() -> Optional[str]:
    """Device name to export as TORCH_DEVICE, or None to leave marker/surya's own auto-detect."""
    choice = os.environ.get("HERMES_DOC_DEVICE", "auto").strip().lower()
    if choice not in ("", "auto"):
        return choice
    try:
        import torch
        if torch.cuda.is_available():  # the library already prefers CUDA
            return None
    except Exception:
        return None
    info = probe()
    if info["cpu"]["is_intel"] and info["intel_igpu_present"] and info["torch_xpu"]:
        return "xpu"
    return None


def apply_document_ocr_device() -> Optional[str]:
    """Set TORCH_DEVICE for marker/surya when a better device is verified; never overrides a
    user-set TORCH_DEVICE. Call BEFORE importing marker. Returns the device it set, if any."""
    if os.environ.get("TORCH_DEVICE"):
        return None
    device = recommend_torch_device()
    if device:
        os.environ["TORCH_DEVICE"] = device
    return device


if __name__ == "__main__":
    import json
    print(json.dumps({**probe(), "recommended_torch_device": recommend_torch_device()}, indent=2))
