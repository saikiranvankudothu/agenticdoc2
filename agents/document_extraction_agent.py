"""
agents/document_extraction_agent.py
-------------------------------------
Agent 1 — Document Extraction Agent

Orchestrates full extraction pipeline for one PDF:
  1. Open document with PyMuPDF
  2. Per-page: extract text blocks (embedded) OR OCR fallback
  3. Extract figure/image blocks and save crops
  4. Serialize result to JSON → input for Agent 2

Paper reference (Section IV-B-1):
  "The Document Extraction Agent parses academic PDFs to extract
   text blocks, tables, figures, captions, and equations while
   maintaining spatial layout."
"""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Optional

try:
    import fitz
except ImportError:
    raise ImportError("PyMuPDF required. Run: pip install pymupdf")

from extractors.pdf_extractor import PDFTextExtractor
from extractors.models import DocumentExtractionResult, PageExtractionResult, ExtractionMethod

logger = logging.getLogger(__name__)


class DocumentExtractionAgent:
    """
    Agent 1: Document Extraction

    Parameters
    ----------
    output_dir    : root directory for outputs (figures/, json/ sub-dirs)
    ocr_threshold : char count below which OCR fires on a page
    render_dpi    : DPI for rasterising pages
    verbose       : enable debug logging
    """
    AGENT_NAME = "DocumentExtractionAgent"

    def __init__(self, output_dir="output", ocr_threshold=50,
                 render_dpi=150, verbose=False):
        self.output_dir    = Path(output_dir)
        self.ocr_threshold = ocr_threshold
        self.render_dpi    = render_dpi
        logging.basicConfig(
            format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
            level=logging.DEBUG if verbose else logging.INFO)

    # ── Public API ─────────────────────────────────────────

    def run(self, pdf_path: str, doc_id: Optional[str] = None) -> DocumentExtractionResult:
        """
        Process a PDF and return a DocumentExtractionResult.

        Parameters
        ----------
        pdf_path : path to input PDF
        doc_id   : custom identifier (derived from filename if None)

        Returns
        -------
        DocumentExtractionResult — typed contract for downstream agents
        """
        pdf_path = Path(pdf_path).resolve()
        self._validate_input(pdf_path)
        doc_id   = doc_id or self._make_doc_id(pdf_path)
        fig_dir  = self._prepare_output_dirs(doc_id)

        logger.info(f"[{self.AGENT_NAME}] Starting: {pdf_path.name} (id={doc_id})")

        extractor = PDFTextExtractor(
            ocr_threshold=self.ocr_threshold,
            render_dpi=self.render_dpi,
            figure_output_dir=str(fig_dir))

        doc   = fitz.open(str(pdf_path))
        pages = []

        for idx in range(len(doc)):
            page   = doc[idx]
            logger.info(f"  Processing page {idx+1}/{len(doc)}...")
            result = extractor.extract_page(page=page, doc_id=doc_id, page_index=idx)
            pages.append(result)
            method = "OCR" if result.ocr_triggered else "embedded"
            logger.info(f"    → {len(result.text_blocks)} text blocks, "
                        f"{len(result.figure_blocks)} figures [{method}]")

        doc.close()

        extraction = DocumentExtractionResult(
            doc_id=doc_id, source_path=str(pdf_path),
            total_pages=len(pages), pages=pages)

        stats = extraction.stats()
        logger.info(f"[{self.AGENT_NAME}] Done — "
                    f"{stats['total_blocks']} blocks, "
                    f"{stats['figure_blocks']} figures, "
                    f"{stats['ocr_pages']} OCR pages")

        self._save_json(extraction, doc_id)
        return extraction

    # ── Internal helpers ───────────────────────────────────

    def _validate_input(self, path: Path):
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {path}")
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"Expected .pdf, got: {path.suffix}")

    def _make_doc_id(self, path: Path) -> str:
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in path.stem)
        return safe[:40]

    def _prepare_output_dirs(self, doc_id: str) -> Path:
        base = self.output_dir / doc_id
        for sub in ("figures", "json"):
            (base / sub).mkdir(parents=True, exist_ok=True)
        return base / "figures"

    def _save_json(self, result: DocumentExtractionResult, doc_id: str):
        path = self.output_dir / doc_id / "json" / "extraction.json"
        result.save_json(str(path))
        logger.info(f"  Saved → {path}")