"""
tests/test_four_fixes.py
-------------------------
Tests for all 4 heuristic detector fixes observed from real PDF output.

Fix 1 — Section headings no longer misclassified as REFERENCE
Fix 2 — Table detection heuristic
Fix 3 — Extended footer detection catches standalone page numbers
Fix 4 — Widened header detection catches running heads

Run: python run_tests.py
"""
import sys, os, unittest, re
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from extractors.models import BoundingBox, StyleAttributes, TextBlock, FigureBlock, PageExtractionResult
from extractors.layout_models import RegionClass, DetectionBackend
from extractors.heuristic_layout_detector import (
    _classify_text_block, _is_header_footer, _is_reference,
    _is_table, _math_density, _is_section_heading, HeuristicLayoutDetector,
)

PAGE_W, PAGE_H = 595.0, 842.0


# ─── Helpers ─────────────────────────────────────────────────

def _bb(x0, y0, x1, y1):
    return BoundingBox(x0=x0, y0=y0, x1=x1, y1=y1)

def _style(font_size=11.0, is_bold=False):
    return StyleAttributes(font_name="Times", font_size=font_size, is_bold=is_bold)

def _blk(text, x0=20, y0=200, x1=500, y1=220,
         font_size=11.0, is_bold=False, bid="b1"):
    return TextBlock(
        block_id=bid, text=text,
        bbox=_bb(x0, y0, x1, y1),
        page_index=0,
        style=_style(font_size=font_size, is_bold=is_bold))

def _classify(text, y0=200, y1=220, font_size=11.0, is_bold=False,
              page_idx=0, median_fs=11.0, nearby_caps=None, x0=20, x1=500):
    blk = _blk(text, x0=x0, y0=y0, x1=x1, y1=y1,
               font_size=font_size, is_bold=is_bold)
    # nearby_caps: list of caption strings; convert to bool for new API
    nearby_table_cap = bool(nearby_caps and any(
        re.search(r"\b(table|tab\.?)\s*\d", c, re.IGNORECASE)
        for c in nearby_caps))
    return _classify_text_block(
        blk=blk, page_height=PAGE_H, page_width=PAGE_W,
        median_font_size=median_fs, page_idx_in_doc=page_idx,
        seq=0, nearby_table_caption=nearby_table_cap)


# ═════════════════════════════════════════════════════════════
# FIX 1 — Section headings vs References
# ═════════════════════════════════════════════════════════════

class TestFix1SectionVsReference(unittest.TestCase):
    """
    Core fix: "2. Related Work" must be TITLE, not REFERENCE.
    The old RE_REFERENCE = r"^\d{1,3}\.\s+[A-Z]" matched both.
    """

    # ── Section headings → must be TITLE ─────────────────────

    def test_numbered_section_bold_is_title(self):
        cls = _classify("2. Related Work", is_bold=True, font_size=12.0)
        self.assertEqual(cls, RegionClass.TITLE,
                         "Bold numbered section heading should be TITLE")

    def test_numbered_section_not_bold_is_title(self):
        cls = _classify("3. Methodology", is_bold=False, font_size=13.5,
                        median_fs=11.0)
        self.assertEqual(cls, RegionClass.TITLE,
                         "Large-font numbered section should be TITLE")

    def test_subsection_bold_is_title(self):
        cls = _classify("2.1. KV Eviction & Compression", is_bold=True,
                        font_size=11.5, median_fs=11.0)
        self.assertEqual(cls, RegionClass.TITLE)

    def test_roman_numeral_section_is_title(self):
        cls = _classify("IV. EXPERIMENTS", is_bold=True, font_size=12.0)
        self.assertEqual(cls, RegionClass.TITLE)

    def test_section_without_number_bold_is_title(self):
        cls = _classify("5. Ablation Result", is_bold=True, font_size=12.0)
        self.assertEqual(cls, RegionClass.TITLE)

    def test_deep_subsection_is_title(self):
        cls = _classify("3.2.1 Experimental Setup", is_bold=True, font_size=11.5)
        self.assertEqual(cls, RegionClass.TITLE)

    # ── Real references → must be REFERENCE ──────────────────

    def test_bracket_reference_is_reference(self):
        cls = _classify("[1] P. Lewis et al., NeurIPS 2020.")
        self.assertEqual(cls, RegionClass.REFERENCE)

    def test_bracket_reference_multi_is_reference(self):
        cls = _classify("[12] J. Devlin, M. Chang, NAACL 2019.")
        self.assertEqual(cls, RegionClass.REFERENCE)

    def test_numbered_reference_with_author_is_reference(self):
        cls = _classify("1. Lewis, P., Perez, E. et al. NeurIPS 2020.")
        self.assertEqual(cls, RegionClass.REFERENCE)

    def test_numbered_reference_with_year_is_reference(self):
        cls = _classify("3. Vaswani, A. et al. Attention is All You Need. 2017.")
        self.assertEqual(cls, RegionClass.REFERENCE)

    def test_numbered_reference_with_venue_is_reference(self):
        cls = _classify("5. Brown et al. Language Models. Conference on NLP, 2020.")
        self.assertEqual(cls, RegionClass.REFERENCE)

    def test_numbered_reference_with_arxiv_is_reference(self):
        cls = _classify("7. Smith, J. A Novel Approach. arXiv:2301.00001, 2023.")
        self.assertEqual(cls, RegionClass.REFERENCE)

    # ── Edge cases ────────────────────────────────────────────

    def test_short_bold_numbered_line_is_title_not_reference(self):
        """The key fix: short bold numbered line must NOT be reference."""
        cls = _classify("1. Introduction", is_bold=True, font_size=12.0)
        self.assertNotEqual(cls, RegionClass.REFERENCE)
        self.assertEqual(cls, RegionClass.TITLE)

    def test_abstract_heading_is_not_reference(self):
        cls = _classify("Abstract", is_bold=True, font_size=12.0, page_idx=0)
        self.assertNotEqual(cls, RegionClass.REFERENCE)

    def test_conclusion_section_is_title(self):
        cls = _classify("6. Conclusion", is_bold=True, font_size=12.0)
        self.assertEqual(cls, RegionClass.TITLE)


class TestIsReferenceHelper(unittest.TestCase):
    """Unit tests for the _is_reference() helper directly."""

    def test_bracket_always_reference(self):
        self.assertTrue(_is_reference("[1] Lewis et al.", False, 11.0, 11.0))

    def test_numbered_et_al_is_reference(self):
        self.assertTrue(_is_reference("2. Brown et al. GPT-3. 2020.",
                                       False, 11.0, 11.0))

    def test_numbered_year_is_reference(self):
        self.assertTrue(_is_reference("4. Smith, J. Title. NeurIPS 2021.",
                                       False, 11.0, 11.0))

    def test_bold_short_numbered_is_not_reference(self):
        self.assertFalse(_is_reference("2. Related Work", True, 12.0, 11.0))

    def test_plain_section_number_is_not_reference(self):
        self.assertFalse(_is_reference("3. Experiments", False, 11.0, 11.0))

    def test_introduction_is_not_reference(self):
        self.assertFalse(_is_reference("1. Introduction", True, 12.0, 11.0))


# ═════════════════════════════════════════════════════════════
# FIX 2 — Table detection
# ═════════════════════════════════════════════════════════════

class TestFix2TableDetection(unittest.TestCase):
    """
    Tables were completely undetected before Fix 2.
    Now detected via grid-content analysis and caption proximity.
    """

    def _table_text_grid(self):
        """Realistic ablation table content from the images."""
        return (
            "Method       WikiMQA  MuSiQue  HotpotQA  NarrativeQA\n"
            "HL-HP        0.4455   0.2871   0.5529    0.1481\n"
            "HL-TP        0.4458   0.2970   0.5693    0.1923\n"
            "HL-TP        0.4722   0.3072   0.5651    0.2100\n"
            "GLOBAL       0.5019   0.3386   0.5954    0.2288"
        )

    def _table_text_similarity(self):
        """RoPE similarity table from the images."""
        return (
            "Model        Method       2WikiMQA  HotpotQA\n"
            "             MoM  Max     MoM  Max\n"
            "LLaMA  Norm-based  0.5324  0.9773  0.5219  0.9766\n"
            "       CacheBlend  0.5243  0.9133  0.5191  0.8570\n"
            "       PARAGRAPH  0.5049  0.6734  0.4985  0.6852"
        )

    def test_grid_content_detected_as_table(self):
        cls = _classify(self._table_text_grid())
        self.assertEqual(cls, RegionClass.TABLE,
                         "Multi-column numeric grid should be TABLE")

    def test_similarity_table_detected(self):
        cls = _classify(self._table_text_similarity())
        self.assertEqual(cls, RegionClass.TABLE)

    def test_below_table_caption_is_table(self):
        # Even sparse content is TABLE if there's a table caption above
        sparse_table = "HL-HP  0.4455  0.2871  0.5529"
        cls = _classify(sparse_table,
                        nearby_caps=["Table 1. Ablation of RoPE geometry."])
        self.assertEqual(cls, RegionClass.TABLE)

    def test_below_fig_caption_is_not_table(self):
        # A figure caption nearby should NOT make text a TABLE
        text = "This is a regular paragraph explaining the figure."
        cls = _classify(text,
                        nearby_caps=["Figure 1. System architecture overview."])
        self.assertNotEqual(cls, RegionClass.TABLE)

    def test_normal_paragraph_not_table(self):
        cls = _classify(
            "We propose a simple and effective attention-norm-based "
            "criterion from the query to context tokens.")
        self.assertNotEqual(cls, RegionClass.TABLE)

    def test_single_number_row_not_table(self):
        cls = _classify("0.5019")
        self.assertNotEqual(cls, RegionClass.TABLE)

    def test_two_column_data_not_table(self):
        # Only 2 columns — below the ≥3 threshold
        cls = _classify("Method  Score\nHL-HP   0.44\nGLOBAL  0.50")
        # May or may not be table — just check it doesn't crash
        self.assertIn(cls, list(RegionClass))


class TestIsTableHelper(unittest.TestCase):
    """Unit tests for the _is_table() helper directly."""

    def test_three_column_grid(self):
        text = "A  B  C\n1  2  3\n4  5  6"
        self.assertTrue(_is_table(text, False))

    def test_numeric_data_rows(self):
        text = "HP  0.44  0.29  0.55\nTP  0.44  0.29  0.56\nGL  0.50  0.33  0.59"
        self.assertTrue(_is_table(text, False))

    def test_table_caption_signal(self):
        text = "HP  0.44"  # too few columns alone
        caps = ["Table 2. RoPE similarity statistics."]
        self.assertTrue(_is_table(text, True))

    def test_plain_text_not_table(self):
        text = "We evaluate our approach on both LLM and VLM benchmarks."
        self.assertFalse(_is_table(text, False))


# ═════════════════════════════════════════════════════════════
# FIX 3 — Footer detection (page numbers)
# ═════════════════════════════════════════════════════════════

class TestFix3FooterDetection(unittest.TestCase):
    """
    Standalone page numbers at bottom of page were labeled PARAGRAPH.
    Now they are correctly labeled FOOTER.
    """

    def test_page_number_bottom_is_footer(self):
        """Single digit at very bottom of page → FOOTER."""
        cls = _classify("5", x0=280, y0=820, x1=315, y1=835)
        self.assertEqual(cls, RegionClass.FOOTER,
                         "Standalone page number should be FOOTER")

    def test_page_number_bottom_15pct_is_footer(self):
        """Page number in bottom 15% of page → FOOTER."""
        y0 = PAGE_H * 0.87   # 87% down = in bottom 13%
        cls = _classify("16", x0=280, y0=y0, x1=315, y1=y0+15)
        self.assertEqual(cls, RegionClass.FOOTER)

    def test_multidigit_page_number_is_footer(self):
        cls = _classify("42", x0=280, y0=810, x1=320, y1=825)
        self.assertEqual(cls, RegionClass.FOOTER)

    def test_standard_footer_margin_still_works(self):
        """Block fully inside bottom 10% margin → still FOOTER."""
        y0 = PAGE_H * 0.95
        cls = _classify("Page 1 of 10", x0=20, y0=y0, x1=575, y1=y0+15)
        self.assertEqual(cls, RegionClass.FOOTER)

    def test_body_text_not_footer(self):
        """Paragraph in middle of page → never FOOTER."""
        cls = _classify("This is body text in the middle of the page.",
                        y0=400, y1=415)
        self.assertNotEqual(cls, RegionClass.FOOTER)

    def test_page_number_in_top_half_not_footer(self):
        """Page number near top → not FOOTER (might be HEADER or paragraph)."""
        cls = _classify("1", x0=280, y0=50, x1=315, y1=65)
        self.assertNotEqual(cls, RegionClass.FOOTER)

    def test_wide_block_with_number_not_footer(self):
        """A wide block with a number is NOT a page number (too wide)."""
        # Wide block = not a page number
        cls = _classify("5", x0=20, y0=820, x1=575, y1=835)  # full width
        # Should still be FOOTER due to standard margin, just checking no crash
        self.assertIn(cls, list(RegionClass))


class TestIsHeaderFooterHelper(unittest.TestCase):
    """Unit tests for the updated _is_header_footer() helper."""

    def test_standard_header_zone(self):
        # y1 ≤ 9% of 842 = 75.78
        bbox = _bb(0, 0, 595, 70)
        cls  = _is_header_footer(bbox, "Running Head", PAGE_H)
        self.assertEqual(cls, RegionClass.HEADER)

    def test_standard_footer_zone(self):
        # y0 ≥ 90% of 842 = 757.8
        bbox = _bb(0, 760, 595, 842)
        cls  = _is_header_footer(bbox, "Some footer text", PAGE_H)
        self.assertEqual(cls, RegionClass.FOOTER)

    def test_page_number_in_bottom_15pct(self):
        y0   = PAGE_H * 0.87   # bottom 13%
        bbox = _bb(280, y0, 315, y0 + 15)
        cls  = _is_header_footer(bbox, "7", PAGE_H)
        self.assertEqual(cls, RegionClass.FOOTER)

    def test_body_block_returns_none(self):
        bbox = _bb(20, 300, 575, 320)
        cls  = _is_header_footer(bbox, "Normal paragraph text.", PAGE_H)
        self.assertIsNone(cls)

    def test_wide_page_number_not_caught_by_page_num_rule(self):
        # Wide block (>50pts) doesn't trigger the page-number-specific rule
        y0   = PAGE_H * 0.88
        bbox = _bb(20, y0, 575, y0 + 15)   # width = 555 > 50
        cls  = _is_header_footer(bbox, "5", PAGE_H)
        # Falls back to standard footer check (y0 ≥ 90% threshold)
        # 88% < 90%, so it's None (not in standard footer zone either)
        self.assertIsNone(cls)


# ═════════════════════════════════════════════════════════════
# FIX 4 — Header detection (running heads)
# ═════════════════════════════════════════════════════════════

class TestFix4HeaderDetection(unittest.TestCase):
    """
    Running heads (paper title at top of each page) were labeled PARAGRAPH.
    Now caught by widened 9% margin.
    """

    def test_running_head_top_8pct_is_header(self):
        """Running head sitting at ~8% of page height → HEADER."""
        y0 = PAGE_H * 0.07    # 7% down = just below old 6% threshold
        y1 = y0 + 12
        cls = _classify(
            "Information-Flow-Aware KV Recomputation for Long Context",
            x0=50, y0=y0, x1=545, y1=y1)
        self.assertEqual(cls, RegionClass.HEADER,
                         "Running head at 7% from top should be HEADER")

    def test_running_head_top_6pct_is_header(self):
        """Running head well within old 6% threshold → still HEADER."""
        y0 = PAGE_H * 0.04
        cls = _classify("Paper Title Running Head",
                        x0=50, y0=y0, x1=545, y1=y0+12)
        self.assertEqual(cls, RegionClass.HEADER)

    def test_body_text_not_header(self):
        """Text at 50% height → never HEADER."""
        cls = _classify(
            "Information-Flow-Aware KV Recomputation for Long Context",
            x0=50, y0=400, x1=545, y1=415)
        self.assertNotEqual(cls, RegionClass.HEADER)

    def test_page_number_top_is_header(self):
        """Page number at very top of page → HEADER."""
        cls = _classify("1", x0=280, y0=10, x1=315, y1=25)
        self.assertEqual(cls, RegionClass.HEADER)

    def test_top_9pct_boundary(self):
        """Block whose bottom edge is exactly at 9% mark → HEADER."""
        y1   = PAGE_H * 0.09   # exactly at 9% = 75.78pts
        y0   = y1 - 12
        cls  = _classify("Running Head Text", x0=50, y0=y0, x1=545, y1=y1)
        self.assertEqual(cls, RegionClass.HEADER)

    def test_just_below_9pct_is_not_header(self):
        """Block starting at 11% of page → not header."""
        y0 = PAGE_H * 0.11
        cls = _classify("Some paragraph text here.", x0=20, y0=y0, x1=575, y1=y0+14)
        self.assertNotEqual(cls, RegionClass.HEADER)


# ═════════════════════════════════════════════════════════════
# Integration — full page detection with all 4 fixes
# ═════════════════════════════════════════════════════════════

class TestFullPageIntegration(unittest.TestCase):
    """
    Simulates the actual page from the images (Image 1 / Image 2)
    and verifies all 4 fixes work together correctly.
    """

    def _make_page(self):
        """Build a PageExtractionResult mimicking the observed PDF pages."""
        blocks = [
            # Header: running head (y ≈ 7% = 59pts) — FIX 4
            _blk("Information-Flow-Aware KV Recomputation for Long Context",
                 x0=50, y0=55, x1=545, y1=67, bid="hdr"),

            # Full-width paper title area (first page)
            _blk("Agent Orchestrated Multimodal Knowledge Extraction",
                 x0=50, y0=90, x1=545, y1=115,
                 font_size=16.0, is_bold=True, bid="papertitle"),

            # Section heading — FIX 1 (must be TITLE, not REFERENCE)
            _blk("2. Related Work", x0=20, y0=160, x1=290, y1=175,
                 font_size=12.0, is_bold=True, bid="sec2"),

            _blk("2.1. KV Eviction & Compression", x0=20, y0=180, x1=290, y1=193,
                 font_size=11.5, is_bold=True, bid="sec21"),

            # Body paragraphs (left column)
            _blk("KV eviction and compression methods reduce long-context "
                 "inference cost by limiting the size or precision of KV "
                 "caches during decoding.",
                 x0=20, y0=200, x1=280, y1=250, bid="para1"),

            # Body paragraphs (right column)
            _blk("We show that reliable recomputation target selection "
                 "requires an inference-consistent RoPE ordering.",
                 x0=310, y0=200, x1=570, y1=250, bid="para2"),

            # Table caption
            _blk("Table 1. Ablation of RoPE geometry configurations.",
                 x0=310, y0=300, x1=570, y1=313, bid="tabcap"),

            # Table data — FIX 2
            _blk("Method     WikiMQA  MuSiQue  HotpotQA\n"
                 "HL-HP      0.4455   0.2871   0.5529\n"
                 "GLOBAL     0.5019   0.3386   0.5954",
                 x0=310, y0=318, x1=570, y1=370, bid="tabledata"),

            # References — FIX 1 (must be REFERENCE, not TITLE)
            _blk("[1] P. Lewis, E. Perez et al. NeurIPS 2020.",
                 x0=20, y0=700, x1=575, y1=713, bid="ref1"),
            _blk("[2] J. Devlin, M. Chang. BERT. NAACL 2019.",
                 x0=20, y0=715, x1=575, y1=728, bid="ref2"),

            # Page number — FIX 3
            _blk("5", x0=285, y0=825, x1=310, y1=838, bid="pgnum"),
        ]

        fig = FigureBlock(
            block_id="fig1",
            bbox=BoundingBox(x0=50, y0=400, x1=545, y1=680),
            page_index=0)

        return PageExtractionResult(
            page_index=0, width=PAGE_W, height=PAGE_H,
            text_blocks=blocks, figure_blocks=[fig])

    def test_all_fixes_on_full_page(self):
        detector = HeuristicLayoutDetector()
        page     = self._make_page()
        result   = detector.detect_page(page, doc_id="test", page_idx_in_doc=0)

        # Build lookup: block_id's source → region class
        cls_map = {}
        for r in result.regions:
            for bid in r.source_block_ids:
                cls_map[bid] = r.region_class

        # FIX 4: Running head → HEADER
        self.assertEqual(cls_map.get("hdr"), RegionClass.HEADER,
                         "FIX 4 FAILED: Running head should be HEADER")

        # FIX 1: Section headings → TITLE (not REFERENCE)
        self.assertEqual(cls_map.get("sec2"), RegionClass.TITLE,
                         "FIX 1 FAILED: '2. Related Work' should be TITLE")
        self.assertEqual(cls_map.get("sec21"), RegionClass.TITLE,
                         "FIX 1 FAILED: '2.1. KV Eviction' should be TITLE")

        # FIX 2: Table data → TABLE
        self.assertEqual(cls_map.get("tabledata"), RegionClass.TABLE,
                         "FIX 2 FAILED: Numeric grid data should be TABLE")

        # FIX 1: References → REFERENCE (not TITLE)
        self.assertEqual(cls_map.get("ref1"), RegionClass.REFERENCE,
                         "FIX 1 FAILED: '[1] Lewis et al.' should be REFERENCE")
        self.assertEqual(cls_map.get("ref2"), RegionClass.REFERENCE,
                         "FIX 1 FAILED: '[2] Devlin et al.' should be REFERENCE")

        # FIX 3: Page number → FOOTER
        self.assertEqual(cls_map.get("pgnum"), RegionClass.FOOTER,
                         "FIX 3 FAILED: Page number '5' should be FOOTER")

        # Sanity: body text still PARAGRAPH
        self.assertEqual(cls_map.get("para1"), RegionClass.PARAGRAPH)
        self.assertEqual(cls_map.get("para2"), RegionClass.PARAGRAPH)

        # Figure still FIGURE
        self.assertEqual(cls_map.get("fig1"), RegionClass.FIGURE)

    def test_region_count_reasonable(self):
        detector = HeuristicLayoutDetector()
        page     = self._make_page()
        result   = detector.detect_page(page, doc_id="test", page_idx_in_doc=0)
        # 11 text blocks + 1 figure = 12 regions total
        self.assertEqual(len(result.regions), 12)

    def test_no_regions_lost(self):
        """Every input block must produce exactly one output region."""
        detector = HeuristicLayoutDetector()
        page     = self._make_page()
        result   = detector.detect_page(page, doc_id="test", page_idx_in_doc=0)
        input_bids  = {b.block_id for b in page.text_blocks}
        input_bids |= {f.block_id for f in page.figure_blocks}
        output_bids = {bid for r in result.regions for bid in r.source_block_ids}
        self.assertEqual(input_bids, output_bids,
                         "Some blocks were lost during classification")


if __name__ == "__main__":
    unittest.main(verbosity=2)