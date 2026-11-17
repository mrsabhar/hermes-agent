"""Intel Quick Sync (oneVPL) acceleration for ffmpeg jobs, with a safe CPU fallback.

Why this exists: on an Intel iGPU the media engine can decode, scale and encode video while the CPU
stays free for the model and the agent. The win only appears when the whole job stays on the GPU
(decode -> scale -> encode with no frame copied back), so this module builds those command lines
instead of every skill hand-rolling them.

Measured on Panther Lake (Arc B390), 60 s of 4K H.264 -> 1080p: libx264 medium 11.4 s / 168 CPU-s,
QSV full-GPU 7.4 s / 6 CPU-s. Image sequence -> MP4 (the pixel-art / p5js shape): libx264 crf18
1.8-2.0 s / 20 CPU-s, QSV 1.2 s / 7 CPU-s. See website/docs/developer-guide/ffmpeg-acceleration.md.

What is deliberately NOT accelerated: single-frame jobs (thumbnails, probes). QSV device start-up
costs ~400 ms, more than the whole software thumbnail (117 ms), so those stay on the CPU.

Policy lives in config (``media.ffmpeg_hwaccel``: ``auto`` | ``off``); there is no environment variable.
The gate is a real 1-frame test encode, not a CPU or GPU generation table, so it also catches a missing
driver or an ffmpeg built without oneVPL. Any QSV failure or timeout re-runs the software command.

CLI (the common entry point for skills that are not Python)::

    python -m tools.ffmpeg_accel report
    python -m tools.ffmpeg_accel transcode IN OUT [--height 1080] [--quality 23]
    python -m tools.ffmpeg_accel frames "frame_%04d.png" OUT --fps 15 [--quality 18]
    python -m tools.ffmpeg_accel selftest      # no media needed: synthetic A/B of CPU vs QSV
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# Below these sizes the one-off QSV start-up (~0.4 s) eats the gain.
MIN_VIDEO_SECONDS = 5.0
MIN_FRAMES = 60

QSV_TIMEOUT_S = 900  # a hung QSV call must never stall a job (observed on 4K single-frame decode)
_PROBE_TIMEOUT_S = 30

# PCI device-id high bytes of Intel GPUs that oneVPL supports (Gen12 / Xe and newer). Informational
# only (shown by `report`); the test encode is the real gate.
_GEN12_PLUS_PREFIXES = {
    "9a": "Tiger Lake / Xe", "4c": "Rocket Lake", "46": "Alder Lake", "a7": "Raptor Lake",
    "7d": "Meteor / Arrow Lake", "64": "Lunar Lake", "b0": "Panther Lake", "e2": "Battlemage",
    "56": "Arc Alchemist", "57": "Arc Alchemist", "4f": "Arc Alchemist", "5a": "Arc Alchemist",
}

_HW_IN = ["-hwaccel", "qsv", "-hwaccel_output_format", "qsv"]
_BASE = ["-hide_banner", "-loglevel", "error", "-y"]


# --------------------------------------------------------------------------- policy / capability

def policy() -> str:
    """``media.ffmpeg_hwaccel`` from config: ``"auto"`` (default) or ``"off"``."""
    try:
        from hermes_cli.config import load_config_readonly
        value = (load_config_readonly().get("media") or {}).get("ffmpeg_hwaccel", "auto")
    except Exception:  # config unreadable -> default behaviour
        value = "auto"
    return "off" if str(value).strip().lower() in ("off", "false", "0", "no", "cpu") else "auto"


def _ffmpeg() -> Optional[str]:
    return shutil.which("ffmpeg")


def _run_quiet(cmd: Sequence[str], timeout: float = _PROBE_TIMEOUT_S) -> subprocess.CompletedProcess:
    return subprocess.run(list(cmd), stdin=subprocess.DEVNULL, capture_output=True, text=True,
                          timeout=timeout, encoding="utf-8", errors="replace")


def _cache_file() -> Optional[Path]:
    try:
        from hermes_constants import get_hermes_home
        return Path(get_hermes_home()) / "cache" / "ffmpeg_qsv.json"
    except Exception:
        return None


def _fingerprint(ffmpeg: str) -> str:
    try:
        st = os.stat(ffmpeg)
        return f"{ffmpeg}|{st.st_size}|{int(st.st_mtime)}"
    except OSError:
        return ffmpeg


def _test_encode(ffmpeg: str) -> Tuple[bool, str]:
    """1-frame h264_qsv encode of a synthetic source. Returns (works, last stderr line)."""
    try:
        r = _run_quiet([ffmpeg, *_BASE, "-f", "lavfi", "-i", "color=size=256x256:rate=1:duration=1",
                        "-frames:v", "1", "-c:v", "h264_qsv", "-f", "null", "-"])
    except (OSError, subprocess.SubprocessError) as exc:
        return False, str(exc)
    tail = (r.stderr or "").strip().splitlines()
    return r.returncode == 0, (tail[-1] if tail else "")


@lru_cache(maxsize=1)
def _capability() -> dict:
    ffmpeg = _ffmpeg()
    if not ffmpeg:
        return {"qsv": False, "reason": "ffmpeg not found on PATH"}
    fp = _fingerprint(ffmpeg)
    cache = _cache_file()
    if cache is not None:
        try:
            data = json.loads(cache.read_text(encoding="utf-8"))
            if data.get("fingerprint") == fp:
                return data
        except (OSError, ValueError):
            pass
    try:
        listed = "h264_qsv" in _run_quiet([ffmpeg, "-hide_banner", "-encoders"]).stdout
    except (OSError, subprocess.SubprocessError):
        listed = False
    if not listed:
        cap = {"qsv": False, "reason": "this ffmpeg has no h264_qsv encoder"}
    else:
        ok, why = _test_encode(ffmpeg)
        cap = {"qsv": ok, "reason": "test encode succeeded" if ok else f"test encode failed: {why or 'unknown'}"}
    cap["fingerprint"] = fp
    if cache is not None:
        try:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(cap), encoding="utf-8")
        except OSError:
            pass
    return cap


def available() -> bool:
    """True when policy allows QSV and a real test encode works on this machine."""
    return policy() != "off" and bool(_capability().get("qsv"))


def reset_cache() -> None:
    """Forget the cached capability (tests, or after a driver / ffmpeg change)."""
    _capability.cache_clear()
    cache = _cache_file()
    if cache is not None:
        try:
            cache.unlink()
        except OSError:
            pass


def worth_it(duration_s: Optional[float] = None, frames: Optional[int] = None) -> bool:
    """Whether a job is long enough to beat QSV start-up. Unknown size counts as worth it."""
    if duration_s is not None and duration_s < MIN_VIDEO_SECONDS:
        return False
    if frames is not None and frames < MIN_FRAMES:
        return False
    return True


def use_qsv(duration_s: Optional[float] = None, frames: Optional[int] = None) -> bool:
    return worth_it(duration_s, frames) and available()


# --------------------------------------------------------------------------- command builders

def _even(n: int) -> int:
    return n - (n % 2)


def probe_video(path: str) -> dict:
    """{'width','height','duration'} via ffprobe; empty dict when it cannot be read."""
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return {}
    try:
        r = _run_quiet([ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries",
                        "stream=width,height:format=duration", "-of", "json", path])
        j = json.loads(r.stdout or "{}")
        s = (j.get("streams") or [{}])[0]
        return {"width": int(s["width"]), "height": int(s["height"]),
                "duration": float((j.get("format") or {}).get("duration") or 0) or None}
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, IndexError):
        return {}


def _scaled(info: dict, height: Optional[int]) -> Optional[Tuple[int, int]]:
    """Even (w, h) when a downscale to ``height`` is wanted and possible, else None."""
    if not height or not info or info["height"] <= height:
        return None
    return _even(round(info["width"] * height / info["height"])), _even(height)


def transcode_commands(src: str, dst: str, *, height: Optional[int] = None, quality: int = 23,
                       info: Optional[dict] = None) -> Tuple[Optional[List[str]], List[str]]:
    """(qsv_cmd, cpu_cmd) for video -> H.264 MP4.

    The QSV command keeps every frame on the GPU: ``-hwaccel_output_format qsv`` + ``vpp_qsv`` + ``h264_qsv``.
    Mixing in a CPU filter or a software decoder forces a copy per frame and loses the gain.
    ``quality`` is a CRF-style value; QSV ``-global_quality`` lands within ~6% bitrate of x264 medium at the
    same number (measured), so one value serves both paths. ``qsv_cmd`` is None when QSV is unavailable.
    """
    ffmpeg = _ffmpeg() or "ffmpeg"
    info = info if info is not None else probe_video(src)
    size = _scaled(info, height)
    sw_vf = [] if size is None else ["-vf", f"scale={size[0]}:{size[1]}"]
    cpu = [ffmpeg, *_BASE, "-i", src, *sw_vf, "-c:v", "libx264", "-crf", str(quality), "-c:a", "aac", "-movflags", "+faststart", dst]
    if not use_qsv(info.get("duration") if info else None):
        return None, cpu
    vpp = "vpp_qsv=format=nv12" if size is None else f"vpp_qsv=w={size[0]}:h={size[1]}:format=nv12"
    qsv = [ffmpeg, *_BASE, *_HW_IN, "-i", src, "-vf", vpp, "-c:v", "h264_qsv", "-async_depth", "8",
           "-global_quality", str(quality), "-c:a", "aac", "-movflags", "+faststart", dst]
    return qsv, cpu


def frames_commands(pattern: str, dst: str, *, fps: int = 15, quality: int = 18,
                    frames: Optional[int] = None) -> Tuple[Optional[List[str]], List[str]]:
    """(qsv_cmd, cpu_cmd) for an image sequence (``frame_%04d.png``) -> H.264 MP4.

    PNG decode is CPU work, so there is no zero-copy path to win; the fastest QSV shape is a plain CPU
    format conversion to nv12 feeding ``h264_qsv`` (GPU-side hwupload + vpp_qsv convert was slower: 1.5 s vs 1.2 s).
    """
    ffmpeg = _ffmpeg() or "ffmpeg"
    even = "pad=ceil(iw/2)*2:ceil(ih/2)*2"
    head = [ffmpeg, *_BASE, "-framerate", str(fps), "-i", pattern]
    cpu = [*head, "-vf", f"{even},format=yuv420p", "-c:v", "libx264", "-crf", str(quality), "-movflags", "+faststart", dst]
    if not use_qsv(frames=frames):
        return None, cpu
    qsv = [*head, "-vf", f"{even},format=nv12", "-c:v", "h264_qsv", "-global_quality", str(quality), "-movflags", "+faststart", dst]
    return qsv, cpu


# --------------------------------------------------------------------------- execution

def run(qsv_cmd: Optional[Sequence[str]], cpu_cmd: Sequence[str], *, timeout: float = QSV_TIMEOUT_S) -> dict:
    """Run the QSV command if given, falling back to the CPU command on any failure or timeout.

    Returns {'path': 'qsv'|'cpu', 'seconds': float, 'fell_back': bool, 'returncode': int, 'stderr': str}.
    """
    t0 = time.perf_counter()
    fell_back = False
    if qsv_cmd:
        try:
            r = subprocess.run(list(qsv_cmd), stdin=subprocess.DEVNULL, capture_output=True, text=True,
                               timeout=timeout, encoding="utf-8", errors="replace")
            if r.returncode == 0:
                return {"path": "qsv", "seconds": time.perf_counter() - t0, "fell_back": False,
                        "returncode": 0, "stderr": ""}
            logger.info("QSV encode failed (rc=%s), retrying on CPU: %s", r.returncode,
                        (r.stderr or "").strip().splitlines()[-1:] or "")
        except subprocess.TimeoutExpired:
            logger.warning("QSV encode timed out after %ss, retrying on CPU", timeout)
        except OSError as exc:
            logger.info("QSV encode could not start (%s), retrying on CPU", exc)
        fell_back = True
    r = subprocess.run(list(cpu_cmd), stdin=subprocess.DEVNULL, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    return {"path": "cpu", "seconds": time.perf_counter() - t0, "fell_back": fell_back,
            "returncode": r.returncode, "stderr": (r.stderr or "")[-400:]}


# --------------------------------------------------------------------------- report / CLI

def _intel_gpus() -> List[dict]:
    """Best-effort Intel GPU listing (name, pci device id, generation hint). Informational only."""
    out: List[dict] = []
    try:
        if sys.platform == "win32":
            r = _run_quiet(["powershell", "-NoProfile", "-Command",
                            "Get-CimInstance Win32_VideoController | ForEach-Object { $_.Name + '|' + $_.PNPDeviceID }"])
            rows = [ln.split("|", 1) for ln in r.stdout.splitlines() if "|" in ln]
            ids = [(n.strip(), re.search(r"VEN_8086&DEV_([0-9A-Fa-f]{4})", p or "")) for n, p in rows]
            out = [{"name": n, "device_id": m.group(1).lower()} for n, m in ids if m]
        else:
            for card in sorted(Path("/sys/class/drm").glob("card[0-9]*/device")):
                if (card / "vendor").read_text().strip() == "0x8086":
                    out.append({"name": "Intel GPU", "device_id": (card / "device").read_text().strip()[2:].lower()})
    except (OSError, subprocess.SubprocessError, ValueError):
        pass
    for g in out:
        g["generation_hint"] = _GEN12_PLUS_PREFIXES.get(g["device_id"][:2], "unknown or pre-Gen12 (needs the legacy Media SDK, not oneVPL)")
    return out


def report() -> dict:
    cap = _capability() if _ffmpeg() else {"qsv": False, "reason": "ffmpeg not found on PATH"}
    return {"policy": policy(), "ffmpeg": _ffmpeg(), "qsv_usable": available(), "reason": cap.get("reason"),
            "intel_gpus": _intel_gpus(), "min_video_seconds": MIN_VIDEO_SECONDS, "min_frames": MIN_FRAMES}


def selftest(seconds: int = 20, height: int = 1080) -> dict:
    """Synthetic A/B that needs no media: make a ``seconds``-long noisy 4K clip (noise keeps it as hard to encode as real footage; a clean synthetic
    clip is trivially cheap for libx264 and would mislead), transcode it on the CPU and (if usable)
    with QSV, and compare wall time, CPU seconds, output size and PSNR. A synthetic clip is a smoke test, not a benchmark: use real
    footage (see the docs) for speed claims."""
    import tempfile
    ffmpeg = _ffmpeg()
    if not ffmpeg:
        return {"error": "ffmpeg not found on PATH"}
    out: dict = {"qsv_usable": available(), "reason": _capability().get("reason")}
    with tempfile.TemporaryDirectory(prefix="ffmpeg_accel_selftest_") as d:
        src, w = os.path.join(d, "src.mp4"), _even(round(height * 16 / 9))
        r = _run_quiet([ffmpeg, *_BASE, "-f", "lavfi", "-i", f"testsrc2=s={w * 2}x{height * 2}:r=30:d={seconds},noise=alls=5:allf=t",
                        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "12", "-pix_fmt", "yuv420p", src], timeout=600)
        if r.returncode:
            return {**out, "error": "could not create the test clip: " + (r.stderr or "")[-200:]}
        for name, force_cpu in (("cpu", True), ("qsv", False)):
            if not force_cpu and not out["qsv_usable"]:
                continue
            dst = os.path.join(d, f"{name}.mp4")
            qsv, cpu = transcode_commands(src, dst, height=height, quality=23, info=probe_video(src))
            if not force_cpu and qsv is None:
                continue
            cmd = list(cpu if force_cpu else qsv)
            cmd[cmd.index("-loglevel") + 1] = "info"
            cmd.insert(1, "-benchmark")
            t0 = time.perf_counter()
            r = _run_quiet(cmd, timeout=QSV_TIMEOUT_S)
            row = {"path": "cpu" if force_cpu else "qsv", "seconds": round(time.perf_counter() - t0, 2), "returncode": r.returncode}
            bench = re.search(r"bench: utime=([0-9.]+)s stime=([0-9.]+)s", r.stderr or "")
            if bench:
                row["cpu_seconds"] = round(float(bench.group(1)) + float(bench.group(2)), 1)
            if not r.returncode:
                row["mbytes"] = round(os.path.getsize(dst) / 1e6, 2)
                m = _run_quiet([ffmpeg, "-hide_banner", "-i", dst, "-i", src, "-lavfi",
                                f"[1:v]scale={w}:{height}:flags=lanczos[r];[0:v][r]psnr", "-f", "null", "-"], timeout=600)
                hit = re.search(r"average:([0-9.]+)", m.stderr or "")
                if hit:
                    row["psnr_db"] = float(hit.group(1))
            out[name] = row
    if "cpu" in out and "qsv" in out and not out["cpu"]["returncode"] and not out["qsv"]["returncode"]:
        if out["qsv"]["seconds"]:
            out["speedup_vs_cpu"] = round(out["cpu"]["seconds"] / out["qsv"]["seconds"], 2)
        if out["cpu"].get("cpu_seconds") and out["qsv"].get("cpu_seconds") is not None:
            out["cpu_seconds_saved_pct"] = round(100 * (1 - out["qsv"]["cpu_seconds"] / out["cpu"]["cpu_seconds"]))
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m tools.ffmpeg_accel", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("report", help="print whether QSV is usable here and why")
    st = sub.add_parser("selftest", help="synthetic CPU vs QSV A/B, no media needed")
    st.add_argument("--seconds", type=int, default=20); st.add_argument("--height", type=int, default=1080)
    t = sub.add_parser("transcode", help="video -> H.264 MP4, QSV when worth it")
    t.add_argument("src"); t.add_argument("dst")
    t.add_argument("--height", type=int); t.add_argument("--quality", type=int, default=23)
    f = sub.add_parser("frames", help="image sequence -> H.264 MP4, QSV when worth it")
    f.add_argument("pattern"); f.add_argument("dst")
    f.add_argument("--fps", type=int, default=15); f.add_argument("--quality", type=int, default=18)
    f.add_argument("--frames", type=int, help="frame count, if known (short jobs stay on CPU)")
    for p in (t, f):
        p.add_argument("--cpu", action="store_true", help="force the CPU path (for A/B comparison)")
    a = ap.parse_args(argv)
    if a.cmd == "report":
        print(json.dumps(report(), indent=2))
        return 0
    if a.cmd == "selftest":
        res = selftest(a.seconds, a.height)
        print(json.dumps(res, indent=2))
        return 1 if "error" in res else 0
    if a.cmd == "transcode":
        qsv, cpu = transcode_commands(a.src, a.dst, height=a.height, quality=a.quality)
    else:
        qsv, cpu = frames_commands(a.pattern, a.dst, fps=a.fps, quality=a.quality, frames=a.frames)
    res = run(None if a.cpu else qsv, cpu)
    print(json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in res.items() if k != "stderr"}))
    if res["returncode"]:
        print(res["stderr"], file=sys.stderr)
    return res["returncode"]


if __name__ == "__main__":
    sys.exit(main())
