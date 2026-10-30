#!/usr/bin/env python3
"""Extract text from documents using marker-pdf. High-quality OCR + layout analysis.

Requires ~3-5GB disk (PyTorch + models downloaded on first use).
Supports: PDF, DOCX, PPTX, XLSX, HTML, EPUB, images.

Usage:
    python extract_marker.py document.pdf
    python extract_marker.py document.pdf --output_dir ./output
    python extract_marker.py presentation.pptx
    python extract_marker.py spreadsheet.xlsx
    python extract_marker.py scanned_doc.pdf           # OCR works here
    python extract_marker.py document.pdf --json        # Structured output
    python extract_marker.py document.pdf --use_llm     # LLM-boosted accuracy
"""
import sys
import os

def _select_device():
    """marker/surya only auto-pick cuda -> mps -> cpu. When tools/hw_probe verifies an Intel CPU,
    an Intel iGPU and a working torch.xpu, export TORCH_DEVICE=xpu (a user-set TORCH_DEVICE wins).
    Silent no-op when the Hermes tree isn't importable from here (e.g. a copied skill)."""
    try:
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), *([".."] * 4)))
        if root not in sys.path:
            sys.path.append(root)
        from tools.hw_probe import apply_document_ocr_device
        device = apply_document_ocr_device()
        if device:
            print(f"marker: using TORCH_DEVICE={device}", file=sys.stderr)
    except Exception:
        pass


def convert(path, output_dir=None, output_format="markdown", use_llm=False):
    _select_device()  # must precede the marker imports below
    from marker.converters.pdf import PdfConverter
    from marker.models import create_model_dict
    from marker.config.parser import ConfigParser

    config_dict = {}
    if use_llm:
        config_dict["use_llm"] = True

    config_parser = ConfigParser(config_dict)
    models = create_model_dict()
    converter = PdfConverter(config=config_parser.generate_config_dict(), artifact_dict=models)
    rendered = converter(path)

    if output_format == "json":
        import json
        print(json.dumps({
            "markdown": rendered.markdown,
            "metadata": rendered.metadata if hasattr(rendered, "metadata") else {},
        }, indent=2, ensure_ascii=False))
    else:
        print(rendered.markdown)

    # Save images if output_dir specified
    if output_dir and hasattr(rendered, "images") and rendered.images:
        from pathlib import Path
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        for name, img_data in rendered.images.items():
            img_path = os.path.join(output_dir, name)
            with open(img_path, "wb") as f:
                f.write(img_data)
        print(f"\nSaved {len(rendered.images)} image(s) to {output_dir}/", file=sys.stderr)


def check_requirements():
    """Check disk space before installing."""
    import shutil
    free_gb = shutil.disk_usage("/").free / (1024**3)
    if free_gb < 5:
        print(f"⚠️  Only {free_gb:.1f}GB free. marker-pdf needs ~5GB for PyTorch + models.")
        print("Use pymupdf instead (scripts/extract_pymupdf.py) or free up disk space.")
        sys.exit(1)
    print(f"✓ {free_gb:.1f}GB free — sufficient for marker-pdf")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract text from documents using marker-pdf (high-quality OCR + layout analysis)."
    )
    parser.add_argument("path", nargs="?", help="Document to convert (PDF, DOCX, PPTX, XLSX, HTML, EPUB, image)")
    parser.add_argument("--output_dir", help="Directory to save extracted images")
    parser.add_argument("--json", action="store_true", help="Structured JSON output instead of markdown")
    parser.add_argument("--use_llm", action="store_true", help="LLM-boosted accuracy")
    parser.add_argument("--check", action="store_true", help="Check disk space requirements and exit")
    args = parser.parse_args()

    if args.check:
        check_requirements()
        sys.exit(0)
    if not args.path:
        parser.error("path is required unless --check is given")

    convert(
        args.path,
        output_dir=args.output_dir,
        output_format="json" if args.json else "markdown",
        use_llm=args.use_llm,
    )
