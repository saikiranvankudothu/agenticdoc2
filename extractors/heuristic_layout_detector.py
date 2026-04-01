"""
extractors/heuristic_layout_detector.py
-----------------------------------------
Rule-based layout classifier — ZERO ML dependencies.

FIXES APPLIED (v2)
------------------
Fix 1 — Section headings no longer misclassified as REFERENCE or LIST
  Problem A: RE_REFERENCE r"^\d{1,3}\.\s+[A-Z]" matched "2. Related Work".
  Problem B: RE_BULLET    r"^\s*\d+[.)]\s+"      matched "2. Related Work".
  Solution:  Section/title check runs BEFORE both reference AND list checks.
             _is_reference() uses author/year/venue signals to distinguish
             bibliography entries from section headings.

Fix 2 — Table detection heuristic
  New _is_table() detects grid-like numeric blocks (≥3 columns separated
  by 2+ spaces across ≥2 rows) and blocks adjacent to table captions.

Fix 3 — Extended footer catches standalone page numbers
  Footer margin widened 6% → 10%. Additionally, narrow numeric-only blocks
  (≤3 chars, ≤50pts wide) in the bottom 15% of the page → FOOTER.

Fix 4 — Widened header catches running heads
  Header margin widened 6% → 9%, catching running heads slightly below
  the old threshold.

Rule priority order
-------------------
 1. Footer    (Fix 3 — bottom 10% OR page-number pattern)
 2. Header    (Fix 4 — top 9%)
 3. Caption   (Fig./Table/Algorithm + number)
 4. Equation  (math-symbol density)
 5. Abstract  (keyword match, first pages)
 6. Title     (Fix 1 — section/paper heading, BEFORE reference and list)
 7. Table     (Fix 2 — grid content OR table-caption adjacency)
 8. Reference (Fix 1 — bracket OR numbered-author pattern)
 9. Algorithm
10. List      (Fix 1 — checked AFTER title so "2. Intro" is not a list)
11. Paragraph (default)
"""

from __future__ import annotations
import hashlib
import logging
import re
from typing import Optional

from extractors.models import (
    BoundingBox, TextBlock, FigureBlock, PageExtractionResult,
)
from extractors.layout_models import (
    LayoutRegion, PageLayoutResult, RegionClass, DetectionBackend,
)
from extractors.reading_order import sort_reading_order

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Compiled regex patterns
# ─────────────────────────────────────────────────────────────

# Caption: "Fig. 3", "Figure 1.", "Table 2", "Tab. 1", "Algorithm 2"
RE_CAPTION = re.compile(
    r"^(fig(?:ure)?\.?\s*\d+|table\s*\d+|tab\.?\s*\d+|algorithm\s*\d+)",
    re.IGNORECASE,
)

# Section number patterns: "1.", "1.2", "1.2.3", "I.", "IV.", "VIII"
# Used to identify section headings (TITLE class)
RE_SECTION_NUM = re.compile(
    r"^("
    r"[IVX]{1,6}[.)]\s"            # Roman numeral: I. II. IV.
    r"|\d{1,2}(\.\d{1,2}){0,2}[.)]\s"  # Arabic: 1. 1.2. 1.2.3.
    r"|\d{1,2}(\.\d{1,2}){0,2}\s"  # Arabic without punctuation: "1 Intro"
    r")"
)

# FIX 1 — Reference detection (bracket style always wins)
RE_REF_BRACKET = re.compile(r"^\s*\[\d[\d,\s]*\]")

# Numbered reference: must have author/year/venue signal after the number
RE_REF_NUMBERED = re.compile(
    r"^\d{1,3}\.\s+"
    r"(?:"
    r"[A-Z][a-z]+,\s[A-Z]"         # "Smith, J" — Surname, Firstname
    r"|.*\bet\s+al\b"               # "et al."
    r"|.*\b(19|20)\d{2}[,.\s]"     # year 19xx or 20xx
    r"|.*\b(arxiv|proceedings|journal|conference|workshop|preprint|transactions|acl|emnlp|naacl|iclr|icml|neurips|nips|cvpr|iccv|eccv)\b"
    r")",
    re.IGNORECASE,
)

# FIX 1 — List bullet: only genuine bullets (NOT "1. Introduction")
# We require the number to be followed by ) not . to avoid section clash,
# OR use non-numeric bullet symbols
RE_BULLET = re.compile(r"^(\s*[-•◦▪▸*]\s+|\s*\d+[)]\s+)")

# Abstract keyword
RE_ABSTRACT = re.compile(r"^abstract[\s\-—:]*$", re.IGNORECASE)

# Page number: 1–4 digits only, optional surrounding space
RE_PAGE_NUMBER = re.compile(r"^\s*\d{1,4}\s*$")

# Grid-like table row: ≥3 tokens separated by 2+ spaces or tabs
RE_TABLE_ROW = re.compile(r"\S+[ \t]{2,}\S+[ \t]{2,}\S+")

# Math characters
MATH_CHARS = set("∑∫∂∇αβγδεζηθλμπρσφψω±×÷→↔⇒⊕⊗∈∉⊂⊆∪∩≤≥≠≈∞")


# ─────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────

def _region_id(doc_id: str, page_idx: int, seq: int) -> str:
    return hashlib.md5(
        f"{doc_id}::layout::p{page_idx}::r{seq}".encode()
    ).hexdigest()[:12]


def _math_density(text: str) -> float:
    if not text:
        return 0.0
    return sum(1 for c in text if c in MATH_CHARS) / len(text)


# ── Fix 3 & 4: Widened header/footer ─────────────────────────

def _is_header_footer(
    bbox: BoundingBox,
    text: str,
    page_height: float,
    header_frac: float = 0.09,    # FIX 4: widened from 0.06 → 0.09
    footer_frac: float = 0.10,    # FIX 3: widened from 0.06 → 0.10
    page_num_zone: float = 0.15,  # FIX 3: page numbers anywhere in bottom 15%
) -> Optional[RegionClass]:
    """
    Detect header/footer regions.
    Fix 3: Standalone page numbers (narrow, numeric-only) in bottom 15% → FOOTER.
    Fix 4: Header margin widened to 9% to catch running heads.
    """
    header_margin = page_height * header_frac
    footer_margin = page_height * footer_frac

    if bbox.y1 <= header_margin:
        return RegionClass.HEADER
    if bbox.y0 >= page_height - footer_margin:
        return RegionClass.FOOTER

    # FIX 3: narrow page-number block anywhere in bottom 15%
    stripped = text.strip()
    if (bbox.y0 >= page_height * (1.0 - page_num_zone)
            and RE_PAGE_NUMBER.match(stripped)
            and bbox.width <= 50):
        return RegionClass.FOOTER

    return None


# ── Fix 1: Section heading detection ─────────────────────────

def _is_section_heading(
    text: str,
    is_bold: bool,
    font_size: float,
    median_font_size: float,
    page_height: float,
    y0: float,
    page_idx: int,
) -> bool:
    """
    Return True if this block is a section/paper title heading.
    Must run BEFORE reference and list checks (Fix 1).

    Conditions (any one sufficient):
    A) Bold + section-number pattern + short text
    B) Bold + font ≥ 1.15× median + short text
    C) Large font (≥ 1.25×) + section-number pattern (not bold)
    D) Very large bold font near top of first pages (paper title)
    """
    short = len(text) < 150
    numbered = bool(RE_SECTION_NUM.match(text))

    # A: Bold + section number + short
    if is_bold and numbered and short:
        return True

    # B: Bold + significantly larger font + short
    if is_bold and font_size >= median_font_size * 1.15 and short:
        return True

    # C: Large font + section number (even without bold)
    if font_size >= median_font_size * 1.20 and numbered and short:
        return True

    return False


# ── Fix 1: Reference detection ───────────────────────────────

def _is_reference(
    text: str,
    is_bold: bool,
    font_size: float,
    median_font_size: float,
) -> bool:
    """
    Distinguish bibliography entries from section headings.
    Bracket format [N] is always a reference.
    Numbered format only if body contains author/year/venue signals.
    Bold + short → section heading wins over numbered reference.
    """
    # Bracket style — unambiguous
    if RE_REF_BRACKET.match(text):
        return True

    # Numbered style — only if it has bibliographic signals
    if RE_REF_NUMBERED.match(text):
        # Bold short text is a section heading even if it starts with a number
        if is_bold and len(text.split()) <= 8:
            return False
        return True

    return False


# ── Fix 2: Table detection ────────────────────────────────────

def _is_table(
    text: str,
    nearby_table_caption: bool,
) -> bool:
    """
    Detect table content via two signals:
    A) Grid content: ≥2 lines each with ≥3 space-separated columns.
    B) Numeric data rows: ≥3 lines of short tokens including numbers.
    C) Adjacent to a table caption (caption detected on same page above).
    """
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return False

    # Signal A: grid-like rows (3+ columns separated by 2+ spaces)
    grid_lines = sum(1 for ln in lines if RE_TABLE_ROW.search(ln))
    if grid_lines >= 2:
        return True

    # Signal B: multiple rows of short numeric tokens
    def _is_data_row(ln: str) -> bool:
        tokens = ln.split()
        if len(tokens) < 3:
            return False
        has_number = any(re.match(r"[\d.%±\-]+$", t) for t in tokens)
        all_short  = all(len(t) <= 15 for t in tokens)
        return has_number and all_short

    data_rows = sum(1 for ln in lines if _is_data_row(ln))
    if data_rows >= 2:
        return True

    # Signal C: immediately below a table caption (even if sparse)
    if nearby_table_caption and len(lines) >= 1:
        return True

    return False


# ─────────────────────────────────────────────────────────────
# Main classification function
# ─────────────────────────────────────────────────────────────

def _classify_text_block(
    blk: TextBlock,
    page_height: float,
    page_width: float,
    median_font_size: float,
    page_idx_in_doc: int,
    seq: int,
    nearby_table_caption: bool = False,
) -> RegionClass:
    """
    Classify a single TextBlock. Rules applied in strict priority order.
    """
    text  = blk.text.strip()
    style = blk.style
    bbox  = blk.bbox
    fsize = style.font_size or median_font_size
    bold  = style.is_bold or False

    # ── Pre-check: very large bold text is a paper title even if near top ──
    # (Prevents large bold titles from being misclassified as HEADER)
    if (bold and fsize >= median_font_size * 1.4
            and bbox.y0 < page_height * 0.40
            and page_idx_in_doc <= 1
            and len(text) < 150):
        return RegionClass.TITLE

    # ── Rule 1 & 2: Header / Footer (Fixes 3 & 4) ────────────
    hf = _is_header_footer(bbox, text, page_height)
    if hf:
        return hf

    # ── Rule 3: Caption ───────────────────────────────────────
    if RE_CAPTION.match(text):
        return RegionClass.CAPTION

    # ── Rule 4: Equation ──────────────────────────────────────
    if _math_density(text) > 0.06:
        return RegionClass.EQUATION

    # ── Rule 5: Abstract ──────────────────────────────────────
    if page_idx_in_doc <= 1 and RE_ABSTRACT.match(text):
        return RegionClass.ABSTRACT

    # ── Rule 6: Title / Section heading (Fix 1 — BEFORE ref & list) ──
    if _is_section_heading(
            text, bold, fsize, median_font_size,
            page_height, bbox.y0, page_idx_in_doc):
        return RegionClass.TITLE

    # ── Rule 7: Table (Fix 2) ─────────────────────────────────
    if _is_table(text, nearby_table_caption):
        return RegionClass.TABLE

    # ── Rule 8: Reference (Fix 1 — after title check) ─────────
    if _is_reference(text, bold, fsize, median_font_size):
        return RegionClass.REFERENCE

    # ── Rule 9: Algorithm ──────────────────────────────────────
    if text.lower().startswith("algorithm") and len(text) > 30:
        return RegionClass.ALGORITHM

    # ── Rule 10: List (Fix 1 — after title, uses tighter bullet regex) ──
    lines = text.splitlines()
    bullet_lines = sum(1 for ln in lines if RE_BULLET.match(ln))
    if bullet_lines >= 2 or (bullet_lines == 1 and len(lines) == 1):
        return RegionClass.LIST

    # ── Rule 11: Paragraph (default) ──────────────────────────
    return RegionClass.PARAGRAPH


# ─────────────────────────────────────────────────────────────
# HeuristicLayoutDetector
# ─────────────────────────────────────────────────────────────

class HeuristicLayoutDetector:
    """
    Rule-based layout detector (v2 — all 4 fixes applied).

    Parameters
    ----------
    confidence_text  : confidence for heuristically classified text regions
    confidence_image : confidence for figure/image regions
    """

    def __init__(
        self,
        confidence_text:  float = 0.80,
        confidence_image: float = 0.95,
    ):
        self.confidence_text  = confidence_text
        self.confidence_image = confidence_image

    def detect_page(
        self,
        page: PageExtractionResult,
        doc_id: str,
        page_idx_in_doc: int,
    ) -> PageLayoutResult:
        """Classify all blocks on a page into LayoutRegions."""
        regions: list[LayoutRegion] = []
        seq = 0

        # Median font size for relative comparisons
        font_sizes = [
            b.style.font_size for b in page.text_blocks
            if b.style.font_size is not None
        ]
        median_fs = (
            sorted(font_sizes)[len(font_sizes) // 2]
            if font_sizes else 11.0
        )

        # Pre-scan: find table captions on this page (Fix 2)
        # A block is "near a table caption" if any caption containing
        # "table" or "tab." appears on the same page.
        has_table_caption = any(
            RE_CAPTION.match(b.text.strip())
            and re.search(r"\b(table|tab\.?)\s*\d", b.text, re.IGNORECASE)
            for b in page.text_blocks
        )

        # Classify each text block
        for blk in page.text_blocks:
            # Only pass table-caption signal to blocks that appear BELOW
            # a table caption (y0 greater than caption's y0)
            below_table_cap = False
            if has_table_caption:
                for cap_blk in page.text_blocks:
                    cap = cap_blk.text.strip()
                    if (RE_CAPTION.match(cap)
                            and re.search(r"\b(table|tab\.?)\s*\d", cap, re.IGNORECASE)
                            and blk.bbox.y0 > cap_blk.bbox.y0
                            and blk.bbox.y0 - cap_blk.bbox.y1 < 40):
                        below_table_cap = True
                        break

            cls = _classify_text_block(
                blk                  = blk,
                page_height          = page.height,
                page_width           = page.width,
                median_font_size     = median_fs,
                page_idx_in_doc      = page_idx_in_doc,
                seq                  = seq,
                nearby_table_caption = below_table_cap,
            )

            regions.append(LayoutRegion(
                region_id        = _region_id(doc_id, page.page_index, seq),
                region_class     = cls,
                bbox             = blk.bbox,
                page_index       = page.page_index,
                confidence       = self.confidence_text,
                backend          = DetectionBackend.HEURISTIC,
                text_content     = blk.text,
                source_block_ids = [blk.block_id],
            ))
            seq += 1

        # Figure blocks are always FIGURE
        for fig in page.figure_blocks:
            regions.append(LayoutRegion(
                region_id        = _region_id(doc_id, page.page_index, seq),
                region_class     = RegionClass.FIGURE,
                bbox             = fig.bbox,
                page_index       = page.page_index,
                confidence       = self.confidence_image,
                backend          = DetectionBackend.HEURISTIC,
                text_content     = None,
                source_block_ids = [fig.block_id],
            ))
            seq += 1

        # Column-aware reading order sort
        regions = sort_reading_order(
            regions              = regions,
            page_width           = page.width,
            page_height          = page.height,
            full_width_threshold = 0.55,
            column_gap_min       = 15.0,
        )

        return PageLayoutResult(
            page_index = page.page_index,
            width      = page.width,
            height     = page.height,
            regions    = regions,
            backend    = DetectionBackend.HEURISTIC,
        )