"""QSV gating for ffmpeg: Intel GPU + listed codec + working probe, else software."""
import subprocess

import pytest

from tools import ffmpeg_accel as fa


@pytest.fixture(autouse=True)
def _clear(monkeypatch):
    fns = (fa.intel_gpu_present, fa._listed, fa._encoder_works)
    for fn in fns:
        fn.cache_clear()
    monkeypatch.delenv("HERMES_FFMPEG_HWACCEL", raising=False)
    monkeypatch.delenv("HERMES_OPT_LIBS", raising=False)
    yield
    for fn in fns:
        fn.cache_clear()


def _device(monkeypatch, gpu=True, works=True):
    monkeypatch.setattr(fa, "intel_gpu_present", lambda: gpu)
    monkeypatch.setattr(fa, "_listed", lambda kind: frozenset({"h264_qsv", "hevc_qsv"}))
    monkeypatch.setattr(fa, "_encoder_works", lambda codec: works)


def test_encoder_qsv_when_everything_checks_out(monkeypatch):
    _device(monkeypatch)
    assert fa.video_encoder("libx264", 18) == ["-c:v", "h264_qsv", "-global_quality", "18"]
    assert fa.video_encoder("libx265", 20)[:2] == ["-c:v", "hevc_qsv"]


@pytest.mark.parametrize("gpu,works", [(False, True), (True, False)])
def test_encoder_software_without_gpu_or_working_probe(monkeypatch, gpu, works):
    _device(monkeypatch, gpu, works)
    assert fa.video_encoder("libx264", 18) == ["-c:v", "libx264", "-crf", "18"]


def test_unsupported_codec_stays_software(monkeypatch):
    _device(monkeypatch)
    assert fa.video_encoder("libvpx-vp9", 30) == ["-c:v", "libvpx-vp9", "-crf", "30"]  # vp9_qsv not listed


def test_off_switch(monkeypatch):
    monkeypatch.setenv("HERMES_FFMPEG_HWACCEL", "off")
    assert fa.intel_gpu_present() is False


def test_decoder_args_follow_codec(monkeypatch):
    _device(monkeypatch)
    monkeypatch.setattr(fa.shutil, "which", lambda n: n)
    monkeypatch.setattr(fa, "_quiet", lambda cmd, t: subprocess.CompletedProcess(cmd, 0, stdout="hevc\n"))
    assert fa.decoder_args("x.mp4") == ["-hwaccel", "qsv", "-c:v", "hevc_qsv"]
    monkeypatch.setattr(fa, "_quiet", lambda cmd, t: subprocess.CompletedProcess(cmd, 0, stdout="prores\n"))
    assert fa.decoder_args("x.mov") == []


def test_run_falls_back_when_qsv_fails(monkeypatch):
    calls = []

    def fake(cmd, **kw):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 1 if "h264_qsv" in cmd else 0)

    monkeypatch.setattr(fa.subprocess, "run", fake)
    r = fa.run(["ffmpeg", "-c:v", "h264_qsv"], ["ffmpeg", "-c:v", "libx264"])
    assert r.returncode == 0 and len(calls) == 2 and "libx264" in calls[1]
    calls.clear()
    fa.run(None, ["ffmpeg", "-c:v", "libx264"])
    assert len(calls) == 1
