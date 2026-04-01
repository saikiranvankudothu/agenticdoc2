"""
Key design decisions:
3-tier backend strategy — the agent auto-selects at runtime:

LayoutParser + PubLayNet (Detectron2) → best for general academic PDFs
Microsoft DiT + DocLayNet (HuggingFace) → 11-class labels, better granularity
Heuristic (zero deps, always available) → 10 rules covering title, caption, equation, reference, header/footer, abstract, etc.

Hybrid fusion (fuse_ml_and_heuristic) — ML gives high precision, heuristics give high specificity. When they overlap on a region, a specificity ranking decides the winner (CAPTION beats PARAGRAPH; EQUATION beats TEXT). Heuristic-only regions (small captions, reference entries the ML model misses) are always preserved.
To run on your machine:
bashpip install pymupdf pytesseract layoutparser[layoutmodels] detectron2
python run_layout.py --pdf your_paper.pdf --print-regions

# Or heuristic-only (no GPU/detectron needed):
python run_layout.py --pdf your_paper.pdf --backend heuristic --print-regions

"""

"""
run_layout.py
--------------
CLI runner for Step 3 — Layout Detection Agent.
Chains: Agent 1 (Extraction) → Agent 2 (Layout Detection)

Usage
-----
    # Full pipeline from PDF:
    python run_layout.py --pdf pdf/paper.pdf

    # Use existing extraction JSON (skip Agent 1):
    python run_layout.py --extraction-json output/doc_id/json/extraction.json

    # Force a specific backend:
    python run_layout.py --pdf paper.pdf --backend heuristic
    python run_layout.py --pdf paper.pdf --backend layoutparser
    python run_layout.py --pdf paper.pdf --backend dit

    # Print region summary:
    python run_layout.py --pdf pdf/paper.pdf --print-regions

Outputs
-------
    output/<doc_id>/json/extraction.json   ← Agent 1 output
    output/<doc_id>/json/layout.json       ← Agent 2 output
"""

import argparse
import json
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))


def load_extraction_from_json(json_path: str):
    """Reconstruct a DocumentExtractionResult from saved JSON."""
    from extractors.models import (
        DocumentExtractionResult, PageExtractionResult,
        TextBlock, FigureBlock, BoundingBox, StyleAttributes,
        BlockType, ExtractionMethod,
    )
    with open(json_path) as f:
        data = json.load(f)

    pages = []
    for pg in data["pages"]:
        text_blocks = []
        for b in pg["text_blocks"]:
            bb  = b["bbox"]
            sty = b.get("style", {})
            text_blocks.append(TextBlock(
                block_id          = b["block_id"],
                text              = b["text"],
                bbox              = BoundingBox(**bb),
                page_index        = b["page_index"],
                style             = StyleAttributes(
                    font_name = sty.get("font_name"),
                    font_size = sty.get("font_size"),
                    is_bold   = sty.get("is_bold"),
                    is_italic = sty.get("is_italic"),
                    color     = sty.get("color"),
                ),
                block_type        = BlockType(b.get("block_type", "text")),
                extraction_method = ExtractionMethod(
                    b.get("extraction_method", "embedded")),
                confidence        = b.get("confidence", 1.0),
                raw_block_no      = b.get("raw_block_no"),
            ))

        figure_blocks = []
        for f in pg.get("figure_blocks", []):
            bb = f["bbox"]
            figure_blocks.append(FigureBlock(
                block_id   = f["block_id"],
                bbox       = BoundingBox(**bb),
                page_index = f["page_index"],
                image_path = f.get("image_path"),
                caption    = f.get("caption"),
            ))

        pages.append(PageExtractionResult(
            page_index     = pg["page_index"],
            width          = pg["width"],
            height         = pg["height"],
            text_blocks    = text_blocks,
            figure_blocks  = figure_blocks,
            ocr_triggered  = pg.get("ocr_triggered", False),
        ))

    return DocumentExtractionResult(
        doc_id      = data["doc_id"],
        source_path = data["source_path"],
        total_pages = data["total_pages"],
        pages       = pages,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Step 3: Layout Detection Agent — classify regions in academic PDFs"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--pdf",
        help="Path to input PDF (runs Agent 1 + Agent 2)")
    source.add_argument("--extraction-json",
        help="Path to existing extraction.json (runs Agent 2 only)")

    parser.add_argument("--output-dir", default="output",
        help="Root directory for outputs (default: ./output)")
    parser.add_argument("--backend", default="auto",
        choices=["auto", "layoutparser", "dit", "heuristic"],
        help="Layout detection backend (default: auto)")
    parser.add_argument("--score-threshold", type=float, default=0.70,
        help="Min ML detection confidence (default: 0.70)")
    parser.add_argument("--dpi", type=int, default=150,
        help="DPI for page rendering (default: 150)")
    parser.add_argument("--verbose", action="store_true",
        help="Enable debug logging")
    parser.add_argument("--print-regions", action="store_true",
        help="Print detected regions per page")

    args = parser.parse_args()

    # ── Step A: Get extraction result ──────────────────────
    if args.pdf:
        print(f"\n[Step 2] Running Document Extraction Agent on: {args.pdf}")
        from agents.document_extraction_agent import DocumentExtractionAgent
        ext_agent  = DocumentExtractionAgent(
            output_dir=args.output_dir, verbose=args.verbose)
        extraction = ext_agent.run(args.pdf)
        pdf_path   = args.pdf
    else:
        print(f"\n[Step 2] Loading extraction from: {args.extraction_json}")
        extraction = load_extraction_from_json(args.extraction_json)
        pdf_path   = extraction.source_path

    # ── Step B: Layout Detection ────────────────────────────
    print(f"\n[Step 3] Running Layout Detection Agent (backend={args.backend})...")
    from agents.layout_detection_agent import LayoutDetectionAgent
    layout_agent = LayoutDetectionAgent(
        backend         = args.backend,
        output_dir      = args.output_dir,
        render_dpi      = args.dpi,
        score_threshold = args.score_threshold,
        verbose         = args.verbose,
    )
    layout = layout_agent.run(extraction, pdf_path=pdf_path)

    # ── Print summary ───────────────────────────────────────
    stats = layout.stats()
    print("\n" + "="*60)
    print("  LAYOUT DETECTION SUMMARY")
    print("="*60)
    print(f"  Document ID     : {layout.doc_id}")
    print(f"  Total Pages     : {stats['total_pages']}")
    print(f"  Total Regions   : {stats['total_regions']}")
    print(f"  Region breakdown:")
    for cls, count in sorted(stats["by_class"].items()):
        bar = "█" * count
        print(f"    {cls:<14} {count:>3}  {bar}")
    print("="*60)

    # ── Print region details ────────────────────────────────
    if args.print_regions:
        print("\n  DETECTED REGIONS (first 8 per page):\n")
        for pg in layout.pages:
            print(f"  ── Page {pg.page_index} ──")
            for r in pg.regions[:8]:
                bb = r.bbox
                preview = (r.text_content or "")[:60].replace("\n", " ")
                print(f"    [{r.region_class.value.upper():<12}] "
                      f"conf={r.confidence:.2f}  "
                      f"bbox=({bb.x0:.0f},{bb.y0:.0f},{bb.x1:.0f},{bb.y1:.0f})")
                if preview:
                    print(f"    text: \"{preview}{'...' if len(r.text_content or '')>60 else ''}\"")
            print()

    print(f"\n  Output saved → {args.output_dir}/{layout.doc_id}/json/layout.json\n")


if __name__ == "__main__":
    main()