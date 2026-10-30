"""GPU raster flags, local OpenVINO OCR engine choice, QSV encoder fallback."""
import importlib.util
from pathlib import Path

import pytest

from tools import browser_launcher as launcher
from tools import hw_probe


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


def _fake_probe(monkeypatch, *, intel=True, igpu=True, xpu=True):
    info = {"cpu": {"vendor": "GenuineIntel" if intel else "AuthenticAMD", "name": "x", "is_intel": intel},
            "intel_gpus": [], "intel_igpu_present": igpu, "openvino_devices": [], "torch_xpu": xpu}
    monkeypatch.setattr(hw_probe, "probe", lambda: info)
    monkeypatch.delenv("HERMES_DOC_DEVICE", raising=False)
    monkeypatch.delenv("TORCH_DEVICE", raising=False)


@pytest.mark.parametrize("intel,igpu,xpu,expect", [
    (True, True, True, "xpu"), (False, True, True, None), (True, False, True, None), (True, True, False, None)])
def test_xpu_only_when_every_check_passes(monkeypatch, intel, igpu, xpu, expect):
    pytest.importorskip("torch")
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    _fake_probe(monkeypatch, intel=intel, igpu=igpu, xpu=xpu)
    assert hw_probe.recommend_torch_device() == expect


def test_user_torch_device_not_overridden(monkeypatch):
    _fake_probe(monkeypatch)
    monkeypatch.setenv("TORCH_DEVICE", "cpu")
    assert hw_probe.apply_document_ocr_device() is None


def test_explicit_override(monkeypatch):
    _fake_probe(monkeypatch, intel=False, igpu=False, xpu=False)
    monkeypatch.setenv("HERMES_DOC_DEVICE", "cpu")
    assert hw_probe.recommend_torch_device() == "cpu"
