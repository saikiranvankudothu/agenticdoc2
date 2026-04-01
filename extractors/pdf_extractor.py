"""
extractors/pdf_extractor.py
---------------------------
Low-level PDF parsing using PyMuPDF (fitz) with Tesseract OCR fallback.

Paper reference: B = {(ti, bi, pi, si)} extraction with OCR fallback
when embedded text coverage is below threshold.
"""
from __future__ import annotations
import hashlib, logging
from pathlib import Path
from typing import Optional

try:
    import fitz
except ImportError:
    raise ImportError("PyMuPDF required. Run: pip install pymupdf")

from PIL import Image
from .models import (TextBlock, FigureBlock, BoundingBox, StyleAttributes,
                     BlockType, ExtractionMethod, PageExtractionResult)
from .figure_detector import detect_all_figures

logger = logging.getLogger(__name__)

OCR_TRIGGER_CHAR_THRESHOLD = 50
RENDER_DPI    = 150
MIN_BLOCK_AREA = 10.0
FITZ_BLOCK_TEXT  = 0
FITZ_BLOCK_IMAGE = 1


def _make_block_id(doc_id: str, page_idx: int, block_no: int) -> str:
    return hashlib.md5(f"{doc_id}::p{page_idx}::b{block_no}".encode()).hexdigest()[:12]


def _extract_style(span: dict) -> StyleAttributes:
    flags = span.get("flags", 0)
    return StyleAttributes(
        font_name=span.get("font"), font_size=span.get("size"),
        is_bold=bool(flags & (1 << 4)), is_italic=bool(flags & (1 << 1)),
        color=span.get("color"))


def _dominant_style(spans: list) -> StyleAttributes:
    if not spans: return StyleAttributes()
    return _extract_style(max(spans, key=lambda s: len(s.get("text", ""))))


def _guess_block_type(text: str, style: StyleAttributes) -> BlockType:
    math_chars = set("∑∫∂∇αβγδεζηθλμπρσφψω=±×÷→∞")
    if sum(1 for c in text if c in math_chars) / max(len(text), 1) > 0.05:
        return BlockType.EQUATION
    return BlockType.TEXT


class PDFTextExtractor:
    """
    Extracts text blocks and image regions from a single PDF page.

    Parameters
    ----------
    ocr_threshold     : pages with fewer chars than this trigger OCR
    render_dpi        : DPI for rasterising pages
    figure_output_dir : directory to save figure crops (None = skip)
    """
    def __init__(self, ocr_threshold=OCR_TRIGGER_CHAR_THRESHOLD,
                 render_dpi=RENDER_DPI, figure_output_dir=None):
        self.ocr_threshold     = ocr_threshold
        self.render_dpi        = render_dpi
        self.figure_output_dir = Path(figure_output_dir) if figure_output_dir else None
        if self.figure_output_dir:
            self.figure_output_dir.mkdir(parents=True, exist_ok=True)

    def extract_page(self, page, doc_id: str, page_index: int) -> PageExtractionResult:
        rect = page.rect
        result = PageExtractionResult(page_index=page_index,
                                      width=rect.width, height=rect.height)
        raw_blocks   = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
        text_blocks  = []
        figure_blocks = []
        total_chars  = 0

        for blk in raw_blocks:
            blk_no = blk.get("number", 0)
            if blk["type"] == FITZ_BLOCK_TEXT:
                tb, cc = self._parse_text_block(blk, doc_id, page_index, blk_no)
                if tb: text_blocks.append(tb); total_chars += cc
            elif blk["type"] == FITZ_BLOCK_IMAGE:
                fb = self._parse_image_block(blk, page, doc_id, page_index, blk_no)
                if fb: figure_blocks.append(fb)

        ocr_triggered = False
        if total_chars < self.ocr_threshold:
            logger.info(f"Page {page_index}: {total_chars} chars < threshold — OCR fallback")
            text_blocks   = self._ocr_page(page, doc_id, page_index)
            ocr_triggered = True

        # ── Enhanced figure detection (vector + XObject + gap strategies) ──
        text_bboxes = [b.bbox for b in text_blocks]
        figure_blocks = detect_all_figures(
            page                    = page,
            doc_id                  = doc_id,
            page_index              = page_index,
            page_width              = rect.width,
            page_height             = rect.height,
            text_bboxes             = text_bboxes,
            existing_raster_figures = figure_blocks,   # type=1 blocks already found
        )

        result.text_blocks       = text_blocks
        result.figure_blocks     = figure_blocks
        result.ocr_triggered     = ocr_triggered
        result.extraction_method = ExtractionMethod.OCR if ocr_triggered else ExtractionMethod.EMBEDDED
        return result

    def _parse_text_block(self, blk, doc_id, page_index, blk_no):
        bbox = BoundingBox(x0=blk["bbox"][0], y0=blk["bbox"][1],
                           x1=blk["bbox"][2], y1=blk["bbox"][3])
        if bbox.area < MIN_BLOCK_AREA: return None, 0

        all_spans, parts = [], []
        for line in blk.get("lines", []):
            for span in line.get("spans", []):
                t = span.get("text", "").strip()
                if t: all_spans.append(span); parts.append(t)

        full_text = " ".join(parts).strip()
        if not full_text: return None, 0

        style = _dominant_style(all_spans)
        return TextBlock(
            block_id=_make_block_id(doc_id, page_index, blk_no),
            text=full_text, bbox=bbox, page_index=page_index, style=style,
            block_type=_guess_block_type(full_text, style),
            extraction_method=ExtractionMethod.EMBEDDED,
            confidence=1.0, raw_block_no=blk_no), len(full_text)

    def _parse_image_block(self, blk, page, doc_id, page_index, blk_no):
        bbox = BoundingBox(x0=blk["bbox"][0], y0=blk["bbox"][1],
                           x1=blk["bbox"][2], y1=blk["bbox"][3])
        if bbox.area < MIN_BLOCK_AREA: return None
        block_id   = _make_block_id(doc_id, page_index, blk_no)
        image_path = None
        if self.figure_output_dir:
            image_path = self._save_figure_crop(page, bbox, doc_id, page_index, blk_no)
        return FigureBlock(block_id=block_id, bbox=bbox, page_index=page_index,
                           image_path=image_path)

    def _save_figure_crop(self, page, bbox, doc_id, page_index, blk_no) -> str:
        clip = fitz.Rect(bbox.x0, bbox.y0, bbox.x1, bbox.y1)
        mat  = fitz.Matrix(self.render_dpi / 72, self.render_dpi / 72)
        pix  = page.get_pixmap(matrix=mat, clip=clip, alpha=False)
        path = self.figure_output_dir / f"{doc_id}_p{page_index}_fig{blk_no}.png"
        pix.save(str(path))
        return str(path)

    def _ocr_page(self, page, doc_id, page_index) -> list:
        try:
            import pytesseract
            from pytesseract import Output
        except ImportError:
            logger.error("pytesseract not installed — OCR unavailable"); return []

        mat = fitz.Matrix(self.render_dpi / 72, self.render_dpi / 72)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples).convert("L")

        data       = pytesseract.image_to_data(img, output_type=Output.DICT, config="--psm 3")
        scale_x    = page.rect.width  / pix.width
        scale_y    = page.rect.height / pix.height
        blk_map: dict = {}

        for i in range(len(data["text"])):
            word = data["text"][i].strip()
            conf = int(data["conf"][i])
            if not word or conf < 0: continue
            bn = data["block_num"][i]
            blk_map.setdefault(bn, []).append({
                "text": word, "conf": conf / 100.0,
                "left": data["left"][i], "top": data["top"][i],
                "width": data["width"][i], "height": data["height"][i]})

        result = []
        for blk_no, words in blk_map.items():
            combined = " ".join(w["text"] for w in words)
            avg_conf = sum(w["conf"] for w in words) / len(words)
            bbox = BoundingBox(
                x0=min(w["left"] for w in words) * scale_x,
                y0=min(w["top"]  for w in words) * scale_y,
                x1=max(w["left"] + w["width"]  for w in words) * scale_x,
                y1=max(w["top"]  + w["height"] for w in words) * scale_y)
            result.append(TextBlock(
                block_id=_make_block_id(doc_id, page_index, blk_no),
                text=combined, bbox=bbox, page_index=page_index,
                style=StyleAttributes(), block_type=BlockType.TEXT,
                extraction_method=ExtractionMethod.OCR,
                confidence=avg_conf, raw_block_no=blk_no))

        logger.info(f"Page {page_index}: OCR extracted {len(result)} blocks")
        return result