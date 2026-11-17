"""Tests for tools.ffmpeg_accel: gating, command shape, fallback, policy. No real ffmpeg or GPU needed."""

import subprocess
from unittest.mock import patch

import pytest

from tools import ffmpeg_accel as fa


@pytest.fixture(autouse=True)
def _clean(monkeypatch, tmp_path):
    """Isolate the capability cache (process + disk) and default to policy=auto."""
    fa._capability.cache_clear()
    monkeypatch.setattr(fa, "_cache_file", lambda: tmp_path / "ffmpeg_qsv.json")
    monkeypatch.setattr(fa, "policy", lambda: "auto")
    monkeypatch.setattr(fa, "_ffmpeg", lambda: "/usr/bin/ffmpeg")
    yield
    fa._capability.cache_clear()


def _cp(rc=0, out="", err=""):
    return subprocess.CompletedProcess([], rc, stdout=out, stderr=err)


# ------------------------------------------------------------------ capability gating

def test_available_when_encoder_listed_and_test_encode_works():
    with patch.object(fa, "_run_quiet", side_effect=[_cp(out=" V..... h264_qsv"), _cp()]) as rq:
        assert fa.available() is True
    assert rq.call_count == 2  # -encoders listing, then the real test encode


def test_not_available_without_qsv_encoder():
    with patch.object(fa, "_run_quiet", return_value=_cp(out=" V..... libx264")) as rq:
        assert fa.available() is False
    assert rq.call_count == 1  # never tries the test encode
    assert "no h264_qsv" in fa._capability()["reason"]


def test_not_available_when_test_encode_fails():
    with patch.object(fa, "_run_quiet", side_effect=[_cp(out="h264_qsv"), _cp(rc=1, err="Error creating a MFX session")]):
        assert fa.available() is False
    assert "MFX session" in fa._capability()["reason"]


def test_not_available_without_ffmpeg(monkeypatch):
    monkeypatch.setattr(fa, "_ffmpeg", lambda: None)
    assert fa.available() is False


def test_policy_off_never_probes(monkeypatch):
    monkeypatch.setattr(fa, "policy", lambda: "off")
    with patch.object(fa, "_run_quiet") as rq:
        assert fa.available() is False
        assert fa.use_qsv() is False
    rq.assert_not_called()


def test_capability_probe_runs_once_and_is_cached_on_disk(tmp_path):
    with patch.object(fa, "_run_quiet", side_effect=[_cp(out="h264_qsv"), _cp()]) as rq:
        assert fa.available() and fa.available()
        assert rq.call_count == 2
        fa._capability.cache_clear()  # new process: disk cache answers
        assert fa.available()
        assert rq.call_count == 2


def test_probe_exception_is_not_available():
    with patch.object(fa, "_run_quiet", side_effect=[_cp(out="h264_qsv"), OSError("boom")]):
        assert fa.available() is False


# ------------------------------------------------------------------ policy reader

def test_policy_reads_config(monkeypatch):
    monkeypatch.undo()  # drop the autouse policy stub
    for raw, want in [("auto", "auto"), ("off", "off"), ("OFF", "off"), (False, "off"), (None, "auto")]:
        with patch("hermes_cli.config.load_config_readonly", return_value={"media": {"ffmpeg_hwaccel": raw}}):
            assert fa.policy() == (want if raw is not None else "auto")


def test_policy_defaults_to_auto_when_config_unreadable(monkeypatch):
    monkeypatch.undo()
    with patch("hermes_cli.config.load_config_readonly", side_effect=RuntimeError("bad yaml")):
        assert fa.policy() == "auto"


def test_default_config_has_the_key():
    from hermes_cli.config_defaults import DEFAULT_CONFIG
    assert DEFAULT_CONFIG["media"]["ffmpeg_hwaccel"] == "auto"


# ------------------------------------------------------------------ worth_it thresholds

@pytest.mark.parametrize("dur,frames,want", [
    (None, None, True), (2.0, None, False), (5.0, None, True), (None, 10, False), (None, 60, True), (60, 1800, True),
])
def test_worth_it(dur, frames, want):
    assert fa.worth_it(dur, frames) is want


# ------------------------------------------------------------------ command shape

@pytest.fixture
def qsv_ok(monkeypatch):
    monkeypatch.setattr(fa, "available", lambda: True)


def test_transcode_full_gpu_shape(qsv_ok):
    info = {"width": 3840, "height": 1714, "duration": 60.0}
    qsv, cpu = fa.transcode_commands("in.mp4", "out.mp4", height=1080, info=info)
    s = " ".join(qsv)
    assert "-hwaccel qsv -hwaccel_output_format qsv" in s  # decode stays on the GPU
    assert "vpp_qsv=w=2420:h=1080:format=nv12" in s or "vpp_qsv=w=2418:h=1080:format=nv12" in s  # even width, GPU scale
    assert "-c:v h264_qsv" in s and "-global_quality 23" in s
    assert "hwdownload" not in s and "hwupload" not in s  # no copy back to system memory
    assert "libx264" in " ".join(cpu) and "h264_qsv" not in " ".join(cpu)


def test_transcode_dimensions_are_even(qsv_ok):
    info = {"width": 1001, "height": 563, "duration": 30.0}
    qsv, _ = fa.transcode_commands("in.mp4", "o.mp4", height=361, info=info)
    w, h = [int(x) for x in qsv[qsv.index("-vf") + 1].split("w=")[1].split(":format")[0].split(":h=")]
    assert w % 2 == 0 and h % 2 == 0


def test_transcode_no_upscale(qsv_ok):
    qsv, cpu = fa.transcode_commands("in.mp4", "o.mp4", height=2160, info={"width": 1920, "height": 1080, "duration": 30})
    assert qsv[qsv.index("-vf") + 1] == "vpp_qsv=format=nv12"
    assert "-vf" not in cpu


def test_short_transcode_stays_on_cpu(qsv_ok):
    qsv, cpu = fa.transcode_commands("in.mp4", "o.mp4", info={"width": 1920, "height": 1080, "duration": 2.0})
    assert qsv is None and "libx264" in cpu


def test_transcode_unknown_probe_still_builds_both(qsv_ok):
    qsv, cpu = fa.transcode_commands("in.mp4", "o.mp4", height=720, info={})
    assert qsv is not None and cpu and qsv[qsv.index("-vf") + 1] == "vpp_qsv=format=nv12"


def test_frames_shape(qsv_ok):
    qsv, cpu = fa.frames_commands("f_%04d.png", "o.mp4", fps=15, quality=18, frames=300)
    s = " ".join(qsv)
    assert "-framerate 15" in s and "format=nv12" in s and "-c:v h264_qsv" in s and "-global_quality 18" in s
    assert "pad=ceil(iw/2)*2:ceil(ih/2)*2" in s  # odd-sized frames still encode
    assert "hwupload" not in s  # measured slower than CPU convert for frame sequences
    assert "-crf 18" in " ".join(cpu) and "yuv420p" in " ".join(cpu)


def test_short_frame_sequence_stays_on_cpu(qsv_ok):
    qsv, cpu = fa.frames_commands("f_%04d.png", "o.mp4", frames=12)
    assert qsv is None and cpu


def test_no_qsv_when_unavailable(monkeypatch):
    monkeypatch.setattr(fa, "available", lambda: False)
    assert fa.frames_commands("f_%04d.png", "o.mp4", frames=900)[0] is None
    assert fa.transcode_commands("a", "b", info={"duration": 99, "width": 4, "height": 4})[0] is None


# ------------------------------------------------------------------ run / fallback

def test_run_uses_qsv_when_it_works():
    with patch("subprocess.run", return_value=_cp()) as r:
        res = fa.run(["qsv"], ["cpu"])
    assert res["path"] == "qsv" and not res["fell_back"] and r.call_count == 1


def test_run_falls_back_on_failure():
    with patch("subprocess.run", side_effect=[_cp(rc=1, err="Error -40"), _cp()]) as r:
        res = fa.run(["qsv"], ["cpu"])
    assert res["path"] == "cpu" and res["fell_back"] and res["returncode"] == 0
    assert r.call_args_list[1].args[0] == ["cpu"]


def test_run_falls_back_on_timeout():
    with patch("subprocess.run", side_effect=[subprocess.TimeoutExpired("qsv", 1), _cp()]):
        res = fa.run(["qsv"], ["cpu"], timeout=1)
    assert res["path"] == "cpu" and res["fell_back"]


def test_run_without_qsv_goes_straight_to_cpu():
    with patch("subprocess.run", return_value=_cp()) as r:
        res = fa.run(None, ["cpu"])
    assert res["path"] == "cpu" and not res["fell_back"] and r.call_count == 1


def test_run_reports_cpu_failure():
    with patch("subprocess.run", return_value=_cp(rc=2, err="no such file")):
        res = fa.run(None, ["cpu"])
    assert res["returncode"] == 2 and "no such file" in res["stderr"]


# ------------------------------------------------------------------ report / CLI

def test_report_shape(monkeypatch):
    monkeypatch.setattr(fa, "_intel_gpus", lambda: [{"name": "Intel Arc", "device_id": "b080", "generation_hint": "Panther Lake"}])
    with patch.object(fa, "_run_quiet", side_effect=[_cp(out="h264_qsv"), _cp()]):
        rep = fa.report()
    assert rep["qsv_usable"] is True and rep["policy"] == "auto" and rep["intel_gpus"][0]["generation_hint"] == "Panther Lake"


def test_generation_hint_flags_pre_gen12():
    with patch.object(fa, "_run_quiet", return_value=_cp(stdout="") if False else _cp(out="Intel(R) UHD Graphics 630|PCI\\VEN_8086&DEV_3E92&SUBSYS_0")), \
         patch.object(fa.sys, "platform", "win32"):
        gpus = fa._intel_gpus()
    assert gpus and "pre-Gen12" in gpus[0]["generation_hint"]


def test_cli_cpu_flag_forces_software(qsv_ok, capsys):
    with patch("subprocess.run", return_value=_cp()) as r:
        rc = fa.main(["frames", "f_%04d.png", "o.mp4", "--frames", "900", "--cpu"])
    assert rc == 0 and "h264_qsv" not in " ".join(r.call_args.args[0])
    assert '"path": "cpu"' in capsys.readouterr().out


def test_selftest_without_qsv_reports_cpu_only(monkeypatch):
    monkeypatch.setattr(fa, "available", lambda: False)
    monkeypatch.setattr(fa, "_run_quiet", lambda *a, **k: _cp(err="bench: utime=10.0s stime=1.0s rtime=3.0s average:41.5"))
    monkeypatch.setattr(fa, "probe_video", lambda p: {"width": 3840, "height": 2160, "duration": 20.0})
    monkeypatch.setattr(fa.os.path, "getsize", lambda p: 1_000_000)
    res = fa.selftest(seconds=1, height=360)
    assert res["qsv_usable"] is False and res["cpu"]["cpu_seconds"] == 11.0 and "qsv" not in res


def test_selftest_reports_cpu_saving_when_qsv_usable(monkeypatch):
    monkeypatch.setattr(fa, "available", lambda: True)
    errs = iter(["", ""] + ["bench: utime=10.0s stime=0.0s", "average:40", "bench: utime=0.5s stime=0.5s", "average:40"])
    monkeypatch.setattr(fa, "_run_quiet", lambda *a, **k: _cp(err=next(errs)))
    monkeypatch.setattr(fa, "probe_video", lambda p: {"width": 3840, "height": 2160, "duration": 20.0})
    monkeypatch.setattr(fa.os.path, "getsize", lambda p: 1_000_000)
    res = fa.selftest(seconds=1, height=360)
    assert res["qsv"]["path"] == "qsv" and res["cpu_seconds_saved_pct"] == 90
