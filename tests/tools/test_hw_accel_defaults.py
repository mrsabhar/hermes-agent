"""GPU raster flags, local OpenVINO OCR engine choice, QSV encoder fallback."""
import importlib.util
from pathlib import Path

import pytest

from tools import browser_launcher as launcher
from tools import ocr_local


def _argv(**kw):
    return launcher.build_chromium_argv("/c", "/p", packaged=True, headed=False, sandbox_bypass=False, **kw)


def test_gpu_raster_flags_on_by_default(monkeypatch):
    monkeypatch.delenv("HERMES_BROWSER_GPU_RASTER", raising=False)
    assert set(launcher.GPU_RASTER_FLAGS) <= set(_argv())


@pytest.mark.parametrize("value", ["0", "false", "off"])
def test_gpu_raster_env_opt_out(monkeypatch, value):
    monkeypatch.setenv("HERMES_BROWSER_GPU_RASTER", value)
    assert not set(launcher.GPU_RASTER_FLAGS) & set(_argv())


def test_disable_gpu_in_extra_args_wins(monkeypatch):
    monkeypatch.delenv("HERMES_BROWSER_GPU_RASTER", raising=False)
    argv = _argv(extra_args=["--disable-gpu"])
    assert "--enable-gpu-rasterization" not in argv and "--disable-gpu" in argv


def test_ocr_engine_resolution(monkeypatch):
    monkeypatch.delenv("HERMES_LOCAL_OCR_ENGINE", raising=False)
    monkeypatch.setattr(ocr_local, "_openvino_importable", lambda: True)
    assert ocr_local.resolve_engine() == "openvino"
    monkeypatch.setattr(ocr_local, "_openvino_importable", lambda: False)
    assert ocr_local.resolve_engine() == "onnxruntime"
    monkeypatch.setenv("HERMES_LOCAL_OCR_ENGINE", "off")
    assert ocr_local.resolve_engine() == "off"
    assert ocr_local.ocr_pdf_pages("missing.pdf") is None


def _load_pixel_video():
    path = Path(__file__).resolve().parents[2] / "optional-skills/creative/pixel-art/scripts/pixel_art_video.py"
    spec = importlib.util.spec_from_file_location("pixel_art_video", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_qsv_failure_falls_back_to_libx264(monkeypatch):
    mod = _load_pixel_video()
    calls = []

    class R:
        def __init__(self, rc): self.returncode = rc

    def fake_run(cmd, **kw):
        calls.append(cmd)
        return R(1 if "h264_qsv" in cmd else 0)

    monkeypatch.setattr(mod, "_QSV_OK", True)
    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    mod._encode_mp4(10, "f_%04d.png", "o.mp4")
    assert "h264_qsv" in calls[0] and "libx264" in calls[1]


def test_qsv_env_off(monkeypatch):
    mod = _load_pixel_video()
    monkeypatch.setenv("HERMES_FFMPEG_HWACCEL", "off")
    monkeypatch.setattr(mod, "_QSV_OK", True)
    assert mod._qsv_available() is False
