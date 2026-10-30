"""Local OCR for scanned PDF pages / images: RapidOCR on OpenVINO by default.

Engine order (``file_tools.local_ocr_engine``, env ``HERMES_LOCAL_OCR_ENGINE``):
  ``auto`` (default) -> OpenVINO when importable, else onnxruntime
  ``openvino`` / ``onnxruntime`` -> that engine only
  ``off`` -> local OCR disabled

OpenVINO runs on its CPU device (``file_tools.local_ocr_device``, env ``HERMES_LOCAL_OCR_DEVICE``;
CPU | GPU | NPU | AUTO). CPU is the default: the PP-OCR models are dynamic-shape, which the NPU
plugin rejects. Everything is optional: no rapidocr / no renderer -> ``None`` and the caller falls
back to its previous behaviour. Never raises.
"""
from __future__ import annotations

import contextlib
import logging
import os
import threading
from typing import Any, Optional

logger = logging.getLogger(__name__)

MAX_PAGES = 25
RENDER_DPI = 200

_engine_lock = threading.Lock()
_engine: Any = None
_engine_name = ""


def _setting(env: str, key: str, default: str) -> str:
    value = os.environ.get(env)
    if not value:
        with contextlib.suppress(Exception):
            from hermes_cli.config import load_config_readonly
            section = load_config_readonly().get("file_tools")
            if isinstance(section, dict) and section.get(key):
                value = str(section[key])
    return (value or default).strip().lower()


def _openvino_importable() -> bool:
    try:
        import openvino  # noqa: F401
        return True
    except Exception:
        return False


def resolve_engine() -> str:
    """``openvino`` | ``onnxruntime`` | ``off`` — what local OCR will use right now."""
    choice = _setting("HERMES_LOCAL_OCR_ENGINE", "local_ocr_engine", "auto")
    if choice in ("off", "false", "0", "none"):
        return "off"
    if choice == "onnxruntime":
        return "onnxruntime"
    if choice == "openvino":
        return "openvino" if _openvino_importable() else "onnxruntime"
    return "openvino" if _openvino_importable() else "onnxruntime"


def _build_engine():
    from rapidocr import EngineType, RapidOCR
    name = resolve_engine()
    if name == "off":
        return None, "off"
    params: dict = {}
    if name == "openvino":
        device = _setting("HERMES_LOCAL_OCR_DEVICE", "local_ocr_device", "CPU").upper()
        params = {f"{stage}.engine_type": EngineType.OPENVINO for stage in ("Det", "Cls", "Rec")}
        params["EngineConfig.openvino.device"] = device
        try:
            return RapidOCR(params=params), "openvino"
        except Exception as exc:  # unsupported device/model for OpenVINO -> stock engine
            logger.warning("OpenVINO OCR engine unavailable (%s); using onnxruntime", exc)
    return RapidOCR(), "onnxruntime"


def _get_engine():
    global _engine, _engine_name
    with _engine_lock:
        if _engine is None:
            _engine, _engine_name = _build_engine()
        return _engine, _engine_name


def engine_name() -> str:
    """The engine actually loaded ('' until first use)."""
    return _engine_name


def _render_pages(pdf_path: str, pages: list[int]):
    """Yield (1-based page, PIL.Image) via pymupdf, else pypdfium2."""
    from PIL import Image
    try:
        try:
            import pymupdf as fitz
        except ImportError:
            import fitz
        with fitz.open(pdf_path) as doc:
            for p in pages:
                if 1 <= p <= doc.page_count:
                    pix = doc[p - 1].get_pixmap(dpi=RENDER_DPI)
                    yield p, Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        return
    except ImportError:
        pass
    import pypdfium2
    pdf = pypdfium2.PdfDocument(pdf_path)
    try:
        for p in pages:
            if 1 <= p <= len(pdf):
                yield p, pdf[p - 1].render(scale=RENDER_DPI / 72).to_pil().convert("RGB")
    finally:
        pdf.close()


def ocr_image(image: Any) -> Optional[str]:
    """OCR one image (path / ndarray / PIL.Image); text lines joined by newlines, None if unavailable."""
    try:
        engine, _ = _get_engine()
        if engine is None:
            return None
        import numpy as np
        if hasattr(image, "convert"):
            image = np.asarray(image.convert("RGB"))[:, :, ::-1]  # RapidOCR expects BGR arrays
        result = engine(image)
        return "\n".join(result.txts or ())
    except Exception as exc:
        logger.warning("local OCR failed: %s", exc)
        return None


def ocr_pdf_pages(pdf_path: str, pages: Optional[list[int]] = None) -> Optional[str]:
    """OCR scanned PDF pages (1-based; None/[] = first MAX_PAGES). Returns markdown-ish text with a
    ``## Page N`` header per page, or None when local OCR is off/unavailable or found no text."""
    if resolve_engine() == "off":
        return None
    try:
        wanted = list(pages or range(1, MAX_PAGES + 1))[:MAX_PAGES]
        parts = []
        for page, img in _render_pages(pdf_path, wanted):
            text = ocr_image(img)
            if text is None:
                return None
            if text.strip():
                parts.append(f"## Page {page}\n\n{text.strip()}")
        if not parts:
            return None
        return "\n\n".join(parts) + "\n"
    except Exception as exc:
        logger.warning("local OCR of %s failed: %s", pdf_path, exc)
        return None
