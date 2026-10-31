"""Intel QSV (oneVPL) acceleration for Hermes' ffmpeg video/image work.

Enabled only when ALL hold (checked lazily, cached per process):
  1. an Intel GPU is present (tools.hw_probe; integrated or Arc) -- or forced with
     HERMES_FFMPEG_HWACCEL=qsv / HERMES_OPT_LIBS=on for A/B runs on other hosts;
  2. this ffmpeg build lists the matching ``*_qsv`` codec;
  3. a tiny real QSV encode succeeds (the encoders are listed even without a working device).
Otherwise every function returns the software choice, so callers keep their default path.
HERMES_FFMPEG_HWACCEL=off disables it outright. Callers should still use ``run`` so a QSV failure
mid-job falls back to software.

``python tools/ffmpeg_accel.py`` prints what this host would use.
"""
from __future__ import annotations

import functools
import os
import shutil
import subprocess
from typing import Optional, Sequence

# software encoder -> QSV encoder. Quality uses -global_quality (ICQ) instead of -crf/-q:v.
ENCODERS = {
    "libx264": "h264_qsv",
    "libx265": "hevc_qsv",
    "libvpx-vp9": "vp9_qsv",
    "libsvtav1": "av1_qsv",
    "libaom-av1": "av1_qsv",
    "mpeg2video": "mpeg2_qsv",
    "mjpeg": "mjpeg_qsv",
}
# ffprobe codec_name -> QSV decoder
DECODERS = {"h264": "h264_qsv", "hevc": "hevc_qsv", "vp9": "vp9_qsv", "av1": "av1_qsv",
            "mpeg2video": "mpeg2_qsv", "vc1": "vc1_qsv", "mjpeg": "mjpeg_qsv"}


def _mode() -> str:
    return os.environ.get("HERMES_FFMPEG_HWACCEL", "auto").strip().lower()


def _quiet(cmd, timeout):
    return subprocess.run(cmd, stdin=subprocess.DEVNULL, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=timeout)


@functools.lru_cache(maxsize=1)
def intel_gpu_present() -> bool:
    """Gate 1. Cheap env decisions first; the hardware query runs only for ``auto``."""
    mode = _mode()
    if mode in ("off", "0", "false", "none"):
        return False
    if mode == "qsv" or os.environ.get("HERMES_OPT_LIBS", "").strip().lower() in ("on", "1", "true"):
        return True
    try:
        from tools.hw_probe import probe
        return bool(probe()["intel_gpus"])
    except Exception:
        return False


@functools.lru_cache(maxsize=1)
def _listed(kind: str) -> frozenset:
    """QSV codec names in this ffmpeg's ``-encoders`` / ``-decoders``."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return frozenset()
    try:
        out = _quiet([ffmpeg, "-hide_banner", f"-{kind}"], 20).stdout
    except (OSError, subprocess.SubprocessError):
        return frozenset()
    return frozenset(tok for line in out.splitlines() for tok in line.split() if tok.endswith("_qsv"))


@functools.lru_cache(maxsize=None)
def _encoder_works(qsv_codec: str) -> bool:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return False
    try:
        return _quiet([ffmpeg, "-hide_banner", "-loglevel", "error", "-f", "lavfi",
                       "-i", "color=size=256x256:rate=1:duration=1", "-frames:v", "1",
                       "-c:v", qsv_codec, "-f", "null", "-"], 30).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def video_encoder(sw_codec: str, quality: Optional[int] = None) -> list:
    """ffmpeg args selecting the encoder: ``["-c:v", "h264_qsv", "-global_quality", "18"]`` when QSV
    is usable for ``sw_codec``, else ``["-c:v", sw_codec]`` (the caller adds its own -crf etc. via
    ``software_quality``). ``quality`` is the CRF-style value the software path would use."""
    qsv = ENCODERS.get(sw_codec)
    if qsv and intel_gpu_present() and qsv in _listed("encoders") and _encoder_works(qsv):
        args = ["-c:v", qsv]
        if quality is not None:
            args += ["-global_quality", str(quality)]
        return args
    return ["-c:v", sw_codec] + (["-crf", str(quality)] if quality is not None else [])


def uses_qsv(args: Sequence[str]) -> bool:
    return any(str(a).endswith("_qsv") for a in args)


def decoder_args(path: str) -> list:
    """Input-side args (``-hwaccel qsv -c:v <codec>_qsv``) when this file's video codec has a usable
    QSV decoder, else []. Needs ffprobe; any doubt -> []."""
    if not intel_gpu_present():
        return []
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return []
    try:
        codec = _quiet([ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries",
                        "stream=codec_name", "-of", "csv=p=0", path], 15).stdout.strip().split(",")[0]
    except (OSError, subprocess.SubprocessError):
        return []
    qsv = DECODERS.get(codec)
    if qsv and qsv in _listed("decoders") and any(_encoder_works(e) for e in _listed("encoders")):
        return ["-hwaccel", "qsv", "-c:v", qsv]
    return []


def run(hw_cmd: Optional[Sequence[str]], sw_cmd: Sequence[str], **kw) -> subprocess.CompletedProcess:
    """Run ``hw_cmd`` when it actually uses QSV; if it fails (or isn't QSV) run ``sw_cmd``.
    ``kw`` goes to subprocess.run (check= applies to the software run only)."""
    check = kw.pop("check", False)
    kw.setdefault("stdin", subprocess.DEVNULL)
    if hw_cmd and uses_qsv(hw_cmd):
        result = subprocess.run(list(hw_cmd), check=False, **kw)
        if result.returncode == 0:
            return result
    return subprocess.run(list(sw_cmd), check=check, **kw)


def report() -> dict:
    return {
        "mode": _mode(),
        "intel_gpu_present": intel_gpu_present(),
        "ffmpeg": shutil.which("ffmpeg"),
        "qsv_encoders_listed": sorted(_listed("encoders")),
        "qsv_decoders_listed": sorted(_listed("decoders")),
        "qsv_encoders_working": sorted(e for e in _listed("encoders")
                                       if intel_gpu_present() and _encoder_works(e)),
    }


if __name__ == "__main__":
    import json
    print(json.dumps(report(), indent=2))
