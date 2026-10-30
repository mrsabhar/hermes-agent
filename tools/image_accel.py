"""Image resize used by the vision tools: OpenCV (SIMD/multithreaded) when Hermes' optimized-library
policy is on (tools.hw_probe.optimized_libs_enabled: Intel CPU, or HERMES_OPT_LIBS=on), else Pillow.

Downscaling uses cv2.INTER_AREA (the antialiased downscale filter), which on a 6000x4000 -> 3000x2000
RGB resize measured ~1.8 ms vs ~157 ms for Pillow LANCZOS on Arrow Lake. Output is not bit-identical
to Pillow's LANCZOS (acceptable for lossy vision re-encodes). Upscales, palette/odd modes and any
cv2 failure fall back to Pillow. JPEG/PNG encode stays on Pillow (measured faster).
"""
from __future__ import annotations

_CV_MODES = {"RGB": 3, "RGBA": 4, "L": 1}


def resize(img, size, pil_filter):
    """Return ``img`` resized to ``size`` (w, h); ``pil_filter`` is the Pillow fallback filter."""
    try:
        from tools.hw_probe import optimized_libs_enabled
        w, h = size
        if (optimized_libs_enabled() and img.mode in _CV_MODES and w <= img.width and h <= img.height):
            import cv2
            import numpy as np
            from PIL import Image
            out = cv2.resize(np.asarray(img), (w, h), interpolation=cv2.INTER_AREA)
            return Image.fromarray(out, img.mode)
    except Exception:
        pass
    return img.resize(size, pil_filter)
