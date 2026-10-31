"""GPU raster flags, local OpenVINO OCR engine choice, QSV encoder fallback."""
import importlib.util
import subprocess
from pathlib import Path

import pytest

from tools import browser_launcher as launcher
from tools import hw_probe


def _argv(**kw):
    return launcher.build_chromium_argv("/c", "/p", packaged=True, headed=False, sandbox_bypass=False, **kw)


def test_gpu_raster_auto_follows_intel_policy(monkeypatch):
    monkeypatch.delenv("HERMES_BROWSER_GPU_RASTER", raising=False)
    monkeypatch.setenv("HERMES_OPT_LIBS", "on")
    assert set(launcher.GPU_RASTER_FLAGS) <= set(_argv())
    monkeypatch.setenv("HERMES_OPT_LIBS", "off")
    assert not set(launcher.GPU_RASTER_FLAGS) & set(_argv())


def test_optimized_libs_switch(monkeypatch):
    monkeypatch.setattr(hw_probe, "is_intel_cpu", lambda: True)
    monkeypatch.delenv("HERMES_OPT_LIBS", raising=False)
    assert hw_probe.optimized_libs_enabled() is True
    monkeypatch.setattr(hw_probe, "is_intel_cpu", lambda: False)
    assert hw_probe.optimized_libs_enabled() is False
    monkeypatch.setenv("HERMES_OPT_LIBS", "on")
    assert hw_probe.optimized_libs_enabled() is True


def test_image_resize_both_paths(monkeypatch):
    from PIL import Image
    from tools import image_accel
    im = Image.new("RGB", (400, 300), (10, 20, 30))
    for mode in ("on", "off"):
        monkeypatch.setenv("HERMES_OPT_LIBS", mode)
        out = image_accel.resize(im, (200, 150), Image.LANCZOS)
        assert out.size == (200, 150) and out.mode == "RGB"
    assert image_accel.resize(im, (800, 600), Image.LANCZOS).size == (800, 600)  # upscale -> Pillow


@pytest.mark.parametrize("value", ["0", "false", "off"])
def test_gpu_raster_env_opt_out(monkeypatch, value):
    monkeypatch.setenv("HERMES_OPT_LIBS", "on")
    monkeypatch.setenv("HERMES_BROWSER_GPU_RASTER", value)
    assert not set(launcher.GPU_RASTER_FLAGS) & set(_argv())


def test_disable_gpu_in_extra_args_wins(monkeypatch):
    monkeypatch.delenv("HERMES_BROWSER_GPU_RASTER", raising=False)
    monkeypatch.setenv("HERMES_OPT_LIBS", "on")
    argv = _argv(extra_args=["--disable-gpu"])
    assert "--enable-gpu-rasterization" not in argv and "--disable-gpu" in argv


def _load_pixel_video():
    path = Path(__file__).resolve().parents[2] / "optional-skills/creative/pixel-art/scripts/pixel_art_video.py"
    spec = importlib.util.spec_from_file_location("pixel_art_video", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_pixel_art_encode_prefers_qsv_then_falls_back(monkeypatch):
    mod = _load_pixel_video()
    from tools import ffmpeg_accel as fa
    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 1 if "h264_qsv" in cmd else 0)

    monkeypatch.setattr(fa, "video_encoder", lambda c, q=None: ["-c:v", "h264_qsv", "-global_quality", "18"])
    monkeypatch.setattr(fa.subprocess, "run", fake_run)
    mod._encode_mp4(10, "f_%04d.png", "o.mp4")
    assert "h264_qsv" in calls[0] and "libx264" in calls[1]


def test_pixel_art_encode_software_without_qsv(monkeypatch):
    mod = _load_pixel_video()
    from tools import ffmpeg_accel as fa
    calls = []
    monkeypatch.setattr(fa, "video_encoder", lambda c, q=None: ["-c:v", "libx264", "-crf", "18"])
    monkeypatch.setattr(fa.subprocess, "run", lambda cmd, **kw: calls.append(cmd) or subprocess.CompletedProcess(cmd, 0))
    mod._encode_mp4(10, "f_%04d.png", "o.mp4")
    assert len(calls) == 1 and "libx264" in calls[0]


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


def test_marker_page_parallel_planning():
    path = Path(__file__).resolve().parents[2] / "skills/productivity/pdf/scripts/extract_marker.py"
    spec = importlib.util.spec_from_file_location("extract_marker", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.split_ranges(10, 3) == [(0, 3), (4, 6), (7, 9)]
    assert mod.plan_workers(0, 100, optimized=False, cpus=24) == 1
    assert mod.plan_workers(0, 100, optimized=True, cpus=24) == 3
    assert mod.plan_workers(0, 6, optimized=True, cpus=24) == 1  # too few pages to split
    assert mod.plan_workers(3, 100, optimized=False) == 3  # explicit request honoured
