"""
run_extraction.py
-----------------
Demo runner for Step 2 — Document Extraction Agent.

Usage
-----
    python run_extraction.py --pdf path/to/paper.pdf
    python run_extraction.py --pdf path/to/paper.pdf --verbose
    python run_extraction.py --pdf path/to/paper.pdf --ocr-threshold 100 --dpi 200

Outputs
-------
  output/<doc_id>/
    figures/         ← PNG crops of all image blocks
    json/
      extraction.json ← Full typed extraction result (passed to Agent 2)
"""

import argparse
import json
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from agents.document_extraction_agent import DocumentExtractionAgent


def main():
    parser = argparse.ArgumentParser(
        description="Step 2: Document Extraction Agent — extract text/figures from academic PDFs"
    )
    parser.add_argument(
        "--pdf", required=True,
        help="Path to the input academic PDF"
    )
    parser.add_argument(
        "--output-dir", default="output",
        help="Root directory for outputs (default: ./output)"
    )
    parser.add_argument(
        "--ocr-threshold", type=int, default=50,
        help="Pages with fewer chars than this trigger OCR (default: 50)"
    )
    parser.add_argument(
        "--dpi", type=int, default=150,
        help="DPI for page rendering (default: 150)"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable debug logging"
    )
    parser.add_argument(
        "--print-blocks", action="store_true",
        help="Print first 5 extracted blocks per page to stdout"
    )

    args = parser.parse_args()

    # ── Run Agent ──────────────────────────────────────────
    agent = DocumentExtractionAgent(
        output_dir    = args.output_dir,
        ocr_threshold = args.ocr_threshold,
        render_dpi    = args.dpi,
        verbose       = args.verbose,
    )

    result = agent.run(pdf_path=args.pdf)

    # ── Print summary ──────────────────────────────────────
    stats = result.stats()
    print("\n" + "="*60)
    print("  EXTRACTION SUMMARY")
    print("="*60)
    print(f"  Document ID   : {result.doc_id}")
    print(f"  Source        : {result.source_path}")
    print(f"  Total Pages   : {stats['total_pages']}")
    print(f"  Text Blocks   : {stats['total_blocks']}")
    print(f"  Figure Blocks : {stats['figure_blocks']}")
    print(f"  OCR Pages     : {stats['ocr_pages']}")
    print(f"  Embedded Pages: {stats['embedded_pages']}")
    print("="*60)

    # ── Print sample blocks ────────────────────────────────
    if args.print_blocks:
        print("\n  SAMPLE TEXT BLOCKS (first 5 per page):\n")
        for page in result.pages:
            print(f"  ── Page {page.page_index} ──")
            for blk in page.text_blocks[:5]:
                bbox = blk.bbox
                print(f"    [{blk.block_type.value.upper()}] "
                      f"bbox=({bbox.x0:.0f},{bbox.y0:.0f},{bbox.x1:.0f},{bbox.y1:.0f}) "
                      f"method={blk.extraction_method.value} "
                      f"conf={blk.confidence:.2f}")
                preview = blk.text[:80].replace("\n", " ")
                print(f"    text: \"{preview}{'...' if len(blk.text)>80 else ''}\"")
                if blk.style.font_name:
                    print(f"    font: {blk.style.font_name} {blk.style.font_size:.1f}pt "
                          f"bold={blk.style.is_bold} italic={blk.style.is_italic}")
                print()

    print(f"\n  Output saved → {args.output_dir}/{result.doc_id}/\n")


if __name__ == "__main__":
    main()