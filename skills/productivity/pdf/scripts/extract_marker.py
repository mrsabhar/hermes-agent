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


def convert(path, output_dir=None, output_format="markdown", use_llm=False, page_range=None):
    _select_device()  # must precede the marker imports below
    from marker.converters.pdf import PdfConverter
    from marker.models import create_model_dict
    from marker.config.parser import ConfigParser

    config_dict = {}
    if use_llm:
        config_dict["use_llm"] = True
    if page_range:
        config_dict["page_range"] = page_range  # marker syntax, 0-based, e.g. "0-7"

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


def _page_count(path):
    try:
        import pymupdf
        with pymupdf.open(path) as doc:
            return doc.page_count
    except Exception:
        pass
    try:
        import pypdfium2
        pdf = pypdfium2.PdfDocument(path)
        try:
            return len(pdf)
        finally:
            pdf.close()
    except Exception:
        return 0


MIN_PAGES_PER_WORKER = 4


def plan_workers(requested, pages, optimized=None, cpus=None):
    """Worker count for page-parallel extraction. ``requested`` > 0 is honoured (capped by pages);
    0 = auto: 1 unless Hermes' optimized-library policy is on (Intel CPU) AND the models run on CPU,
    then one worker per 8 logical CPUs (max 4). Each worker loads its own models (GBs of RAM)."""
    cpus = cpus or os.cpu_count() or 1
    if requested and requested > 0:
        n = requested
    else:
        if optimized is None:
            try:
                root = os.path.abspath(os.path.join(os.path.dirname(__file__), *([".."] * 4)))
                if root not in sys.path:
                    sys.path.append(root)
                from tools.hw_probe import optimized_libs_enabled
                optimized = optimized_libs_enabled()
            except Exception:
                optimized = False
        on_cpu = os.environ.get("TORCH_DEVICE", "cpu").lower() == "cpu"
        n = min(4, max(1, cpus // 8)) if optimized and on_cpu else 1
    return max(1, min(n, pages // MIN_PAGES_PER_WORKER))


def split_ranges(pages, workers):
    """Contiguous 0-based inclusive (start, end) ranges covering ``pages`` pages."""
    base, extra = divmod(pages, workers)
    out, start = [], 0
    for i in range(workers):
        size = base + (1 if i < extra else 0)
        out.append((start, start + size - 1))
        start += size
    return out


def convert_parallel(path, workers, use_llm=False):
    """Run ``workers`` child processes over disjoint page ranges and join their markdown in order."""
    import subprocess
    pages = _page_count(path)
    workers = plan_workers(workers, pages)
    if workers <= 1:
        return None
    threads = str(max(1, (os.cpu_count() or workers) // workers))
    env = dict(os.environ, OMP_NUM_THREADS=threads, MKL_NUM_THREADS=threads)
    procs = []
    for lo, hi in split_ranges(pages, workers):
        cmd = [sys.executable, os.path.abspath(__file__), path, "--workers", "1", "--page_range", f"{lo}-{hi}"]
        if use_llm:
            cmd.append("--use_llm")
        procs.append(subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env,
                                      text=True, encoding="utf-8"))
    parts = []
    for p in procs:
        out, err = p.communicate()
        if p.returncode != 0:
            raise RuntimeError(f"marker worker failed: {err.strip()[-500:]}")
        parts.append(out.rstrip("\n"))
    return "\n\n".join(parts) + "\n"


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
    parser.add_argument("--workers", type=int, default=0,
                        help="Page-parallel worker processes (0 = auto: Intel CPU + CPU models only)")
    parser.add_argument("--page_range", help="0-based pages, e.g. 0-7 (used by page-parallel workers)")
    parser.add_argument("--check", action="store_true", help="Check disk space requirements and exit")
    args = parser.parse_args()

    if args.check:
        check_requirements()
        sys.exit(0)
    if not args.path:
        parser.error("path is required unless --check is given")

    if (not args.json and not args.page_range and args.path.lower().endswith(".pdf")
            and (args.workers != 1)):
        _select_device()  # decide device first: plan_workers only parallelises CPU runs
        merged = convert_parallel(args.path, args.workers, use_llm=args.use_llm)
        if merged is not None:
            print(merged)
            sys.exit(0)

    convert(
        args.path,
        output_dir=args.output_dir,
        output_format="json" if args.json else "markdown",
        use_llm=args.use_llm,
        page_range=args.page_range,
    )
