---
sidebar_position: 12
title: "ffmpeg Acceleration (Intel Quick Sync)"
description: "Use the Intel iGPU media engine for video encodes in skills, how the helper decides, and how to evaluate it on your machine"
---

# ffmpeg Acceleration (Intel Quick Sync)

On an Intel laptop the iGPU has a dedicated media engine (Quick Sync, driven through oneVPL). A video job that
stays on that engine uses a few CPU-seconds instead of a hundred, which leaves the CPU free for the model and the
agent. `tools/ffmpeg_accel.py` builds the right ffmpeg command, checks that it works on this machine, and falls
back to the CPU command if anything goes wrong.

It is on by default (`media.ffmpeg_hwaccel: auto`). Machines without a working Intel GPU behave exactly as before.

## Config

```yaml
media:
  ffmpeg_hwaccel: auto   # auto | off
```

`auto` uses QSV when a real 1-frame test encode succeeds; `off` always uses the CPU. There is no environment
variable. The test result is cached in `~/.hermes/cache/ffmpeg_qsv.json`, keyed by the ffmpeg binary, so it is paid
once per ffmpeg install.

The gate is a test encode, not a CPU or GPU generation table, so it also catches a missing driver or an ffmpeg
built without oneVPL. oneVPL needs Gen12 / Xe or newer (Tiger Lake and later, Arc, Core Ultra). Older parts need the
legacy Media SDK; they fail the test encode and stay on the CPU. `report` prints a generation hint for information.

## What is accelerated, and what is not

| Job | Path | Why |
|---|---|---|
| Video to H.264 MP4 (long) | QSV, whole pipeline on the GPU | Decode, scale and encode never leave GPU memory. |
| Image sequence to MP4 | QSV encode, CPU format convert | PNG decode is CPU work; a GPU-side upload was slower (1.5 s vs 1.2 s). |
| Short jobs (under 5 s of video or 60 frames) | CPU | QSV start-up is about 0.4 s. |
| Single-frame thumbnails and probes | CPU | CPU 117 ms; the only working QSV shape is 511 ms. Branch-style QSV decode fails or hangs on 4K. |

## For skill authors

Do not hand-write QSV flags. Use the helper; it handles the shape, even dimensions, timeouts and fallback.

Python:

```python
from tools import ffmpeg_accel

qsv, cpu = ffmpeg_accel.frames_commands("frame_%04d.png", "out.mp4", fps=30, quality=18, frames=900)
result = ffmpeg_accel.run(qsv, cpu)          # {"path": "qsv"|"cpu", "seconds": ..., "fell_back": ..., "returncode": ...}

qsv, cpu = ffmpeg_accel.transcode_commands("in.mov", "out.mp4", height=1080, quality=23)
result = ffmpeg_accel.run(qsv, cpu)
```

Shell (the helper is the common entry point for non-Python skills; keep a plain ffmpeg fallback for when
`tools` is not importable):

```bash
python -m tools.ffmpeg_accel frames "frame_%04d.png" out.mp4 --fps 30 --quality 18 --frames 900
python -m tools.ffmpeg_accel transcode in.mov out.mp4 --height 1080 --quality 23
```

`quality` is a CRF-style number; the same value is used for libx264 and QSV `-global_quality`. Reference
integrations: `optional-skills/creative/pixel-art/scripts/pixel_art_video.py` and
`skills/creative/p5js/scripts/render.sh`.

Rules the measurements back up:

- Keep frames on the GPU end to end for video input: `-hwaccel qsv -hwaccel_output_format qsv`, `vpp_qsv`, `h264_qsv`.
  Mixing in a software filter or a software decoder forces a copy per frame and loses the gain.
- QSV decode with software filters needs `-hwaccel_output_format nv12`; without it ffmpeg fails with error -40.
- Never use QSV for a single frame.
- Always run with a timeout and a CPU fallback.

## Evaluate it on your machine

Needs ffmpeg with oneVPL (the BtbN win64/linux64 builds have it) and an Intel Gen12+ GPU with a current driver.

1. **Is QSV usable?**

   ```bash
   python -m tools.ffmpeg_accel report
   ```

   Expect `"qsv_usable": true` and your GPU listed. If false, `reason` says why.

   **Quickest check, no media needed** (makes a synthetic noisy 4K clip, transcodes it on CPU and QSV, prints wall time, CPU seconds, size and PSNR):

   ```bash
   python -m tools.ffmpeg_accel selftest
   ```

   Read `cpu_seconds_saved_pct` first: QSV should use about 90% less CPU time. Wall time on a synthetic clip is not
   representative (on this machine QSV was slightly slower in wall time on it, 11.3 s vs 8.3 s, while using 8.4 CPU-s
   against 125). Use real footage for speed claims.

2. **A/B a transcode on your own video** (QSV, then forced CPU):

   ```bash
   python -m tools.ffmpeg_accel transcode my_4k.mp4 out_qsv.mp4 --height 1080
   python -m tools.ffmpeg_accel transcode my_4k.mp4 out_cpu.mp4 --height 1080 --cpu
   ```

3. **A real Hermes skill** (frames to MP4):

   ```bash
   python optional-skills/creative/pixel-art/scripts/pixel_art_video.py base.png out.mp4 --duration 30
   ```

   For the CPU baseline set `media.ffmpeg_hwaccel: off` in `config.yaml` and rerun.

4. **Simulate a weaker CPU** by pinning the job to 2 or 4 cores. PowerShell:

   ```powershell
   Start-Process -Wait -NoNewWindow -FilePath cmd -ArgumentList '/c start /wait /affinity 0xC python -m tools.ffmpeg_accel transcode in.mp4 out.mp4 --height 1080'   # 0xC = 2 cores, 0xF = 4 cores
   ```

   Linux: `taskset -c 0,1 python -m tools.ffmpeg_accel ...`

5. **Measure CPU time and energy.** Windows: `typeperf "\Processor Information(*)\% Processor Utility" "\GPU Engine(*engtype_VideoProcessing)\Utilization Percentage" -si 1`
   during the run; package energy from the `\Energy Meter(rapl_package0_pkg)\Energy` counter where exposed. Linux: `perf stat -e power/energy-pkg/`, `intel_gpu_top`.

6. **Check quality** against a lossless reference (lower-res reference scaled with the same Lanczos filter):

   ```bash
   ffmpeg -t 20 -i in.mp4 -vf scale=1920:858:flags=lanczos,format=yuv420p -an -c:v ffv1 ref.mkv
   ffmpeg -i out_qsv.mp4 -i ref.mkv -lavfi "[0:v][1:v]libvmaf=n_threads=8:feature=name=psnr|name=float_ssim" -f null -
   ```

7. **Unit tests:** `pytest tests/tools/test_ffmpeg_accel.py`

## Reference results

Intel Core Ultra X7 358H (Panther Lake), Arc B390, Windows 11, ffmpeg 8.0.1 BtbN with oneVPL API 2.16. Medians of
3 runs unless stated. These are one machine; treat them as a method to repeat, not a promise.

**60 s of 4K H.264 to 1080p, all cores**

| Config | Wall | CPU-seconds |
|---|---|---|
| libx264 medium (previous default) | 11.4 s | 168 |
| libx264 veryfast | 7.7 s | 107 |
| QSV, whole pipeline on the GPU | 6.6–7.4 s | 5–6 |

**Quality**, 20 s of the same clip against a lossless Lanczos reference

| Config | Bitrate | VMAF | PSNR | SSIM |
|---|---|---|---|---|
| libx264 medium, crf 23 | 2250 kbps | 78.1 | 39.85 dB | 0.9639 |
| QSV default preset, quality 23 | 2114 kbps | 80.6 | 39.87 dB | 0.9635 |
| libx264 veryfast, crf 23 | 1735 kbps | 75.0 | 39.09 dB | 0.9629 |

QSV at the default preset matches libx264 medium on PSNR and SSIM and scores higher on VMAF at slightly lower bitrate.
The faster QSV flags (`veryfast`, no look-ahead, no B-frames) save about 10% wall time and cost about 0.9 VMAF at equal
bitrate, so the helper does not use them.

**Weak CPU: the same 30 s 4K transcode and a 1800-frame 1080p PNG-to-MP4, pinned to fewer cores** (mean of 2)

| Cores | libx264 medium transcode | QSV transcode | libx264 medium frames | QSV frames |
|---|---|---|---|---|
| all | 6.0 s / 111 J | 4.5 s / 17 J | 8.9 s / 156 J | 5.2 s / 59 J |
| 4 P-cores | 14.9 s / 170 J | 4.4 s / 20 J | 22.7 s / 220 J | 9.7 s / 74 J |
| 2 P-cores | 27.2 s / 195 J | 4.5 s / 16 J | 43.0 s / 285 J | 17.6 s / 96 J |
| 4 E-cores | 22.2 s / 116 J | 4.8 s / 12 J | 30.9 s / 157 J | 9.5 s / 52 J |
| 2 E-cores | 42.5 s / 149 J | 4.6 s / 13 J | 59.5 s / 203 J | 18.6 s / 64 J |

Video-to-video QSV time does not depend on the CPU at all (4.4–4.8 s from 16 cores down to 2). Frame-sequence jobs still
need the CPU for PNG decode, so they scale with cores, but stay 1.7x (all cores) to 3.3x (4 E-cores) faster than libx264 medium.

## Known limits

- Tested on one Intel platform. Other Gen12+ parts and Linux (VAAPI-backed oneVPL) are untested here; use the steps above.
- QSV is a hardware encoder: output differs from libx264 at the bit level, and rate control is less configurable.
- Only H.264 output is wired. HEVC (`hevc_qsv`) measured fine, but no skill needs it yet.
- Audio is re-encoded to AAC in `transcode`.
