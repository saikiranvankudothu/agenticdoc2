"""
tests/test_fixes.py
--------------------
Tests for:
  Fix 1 — Enhanced figure detection (vector / XObject / gap strategies)
  Fix 2 — Column-aware reading order (two-column IEEE/ACM layout)

Run: python run_tests.py
"""
import sys, os, unittest
from unittest.mock import MagicMock, patch, PropertyMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from extractors.models import BoundingBox, FigureBlock, TextBlock, StyleAttributes
from extractors.figure_detector import (
    detect_vector_figures, detect_textgap_figures,
    detect_all_figures, _bbox_union, _bboxes_overlap,
)
from extractors.reading_order import (
    detect_columns, is_full_width, sort_reading_order, Column,
)
from extractors.layout_models import LayoutRegion, RegionClass, DetectionBackend


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _bb(x0, y0, x1, y1):
    return BoundingBox(x0=x0, y0=y0, x1=x1, y1=y1)

def _fig(x0, y0, x1, y1, bid="f1", page=0):
    return FigureBlock(block_id=bid, bbox=_bb(x0,y0,x1,y1), page_index=page)

def _region(cls, x0, y0, x1, y1, rid="r1", page=0):
    return LayoutRegion(
        region_id=rid, region_class=cls,
        bbox=_bb(x0,y0,x1,y1),
        page_index=page, confidence=0.9,
        backend=DetectionBackend.HEURISTIC)


# ─────────────────────────────────────────────────────────────
# Fix 1: Figure detection helpers
# ─────────────────────────────────────────────────────────────

class TestBboxHelpers(unittest.TestCase):
    def test_bbox_union(self):
        boxes  = [_bb(0,0,100,50), _bb(50,30,200,100)]
        result = _bbox_union(boxes)
        self.assertEqual(result.x0, 0)
        self.assertEqual(result.y0, 0)
        self.assertEqual(result.x1, 200)
        self.assertEqual(result.y1, 100)

    def test_bboxes_overlap_true(self):
        self.assertTrue(_bboxes_overlap(_bb(0,0,100,100), _bb(50,50,150,150)))

    def test_bboxes_overlap_false(self):
        self.assertFalse(_bboxes_overlap(_bb(0,0,100,100), _bb(110,0,200,100)))

    def test_bboxes_overlap_tolerance(self):
        # Touching but not overlapping — tolerance makes them overlap
        self.assertTrue(_bboxes_overlap(
            _bb(0,0,100,100), _bb(100,0,200,100), tolerance=5.0))


class TestDetectVectorFigures(unittest.TestCase):

    def _mock_page_with_drawings(self, drawing_rects):
        """Build a mock fitz.Page with specified drawing rectangles."""
        page = MagicMock()

        def rect_obj(x0, y0, x1, y1):
            r = MagicMock()
            r.x0, r.y0, r.x1, r.y1 = x0, y0, x1, y1
            return r

        page.get_drawings.return_value = [
            {"rect": rect_obj(*coords)} for coords in drawing_rects
        ]
        return page

    def test_detects_large_vector_cluster(self):
        # A cluster of drawing paths forming a large region = figure
        page = self._mock_page_with_drawings([
            (50,  50,  150, 100),
            (60,  60,  140, 90),
            (50,  100, 150, 200),
            (55,  150, 145, 195),
        ])
        figs = detect_vector_figures(
            page=page, doc_id="d", page_index=0,
            page_width=595, page_height=842,
            text_bboxes=[], min_area=2000)
        self.assertGreater(len(figs), 0)
        # All detections should be FigureBlock
        self.assertIsInstance(figs[0], FigureBlock)

    def test_ignores_tiny_drawing(self):
        # Very small drawing (< MIN_FIGURE_AREA) should be ignored
        page = self._mock_page_with_drawings([
            (50, 50, 60, 60),   # 10×10 = 100 pts² < 2000 threshold
        ])
        figs = detect_vector_figures(
            page=page, doc_id="d", page_index=0,
            page_width=595, page_height=842,
            text_bboxes=[], min_area=2000)
        self.assertEqual(len(figs), 0)

    def test_ignores_text_covered_region(self):
        # Drawing region heavily overlapped by text = table/decoration, not figure
        page = self._mock_page_with_drawings([
            (50, 100, 500, 400),   # large region
        ])
        # Text blocks covering 80% of that region
        text_bboxes = [_bb(50, 100, 500, 400)]
        figs = detect_vector_figures(
            page=page, doc_id="d", page_index=0,
            page_width=595, page_height=842,
            text_bboxes=text_bboxes, min_area=2000)
        self.assertEqual(len(figs), 0)

    def test_empty_page_returns_nothing(self):
        page = MagicMock()
        page.get_drawings.return_value = []
        figs = detect_vector_figures(
            page=page, doc_id="d", page_index=0,
            page_width=595, page_height=842, text_bboxes=[])
        self.assertEqual(figs, [])

    def test_clusters_nearby_paths(self):
        # Two groups of paths far apart should produce 2 separate figures
        page = self._mock_page_with_drawings([
            # Cluster 1: top-left area must be > MIN_FIGURE_PAGE_FRAC*page_area
            # 595*842*0.02 = 10020pts² → need >100×100=10000, use 120×120
            (10, 10, 130, 130), (15, 15, 125, 125),
            # Cluster 2: bottom-right (far away)
            (400, 600, 550, 750), (410, 610, 545, 745),
        ])
        figs = detect_vector_figures(
            page=page, doc_id="d", page_index=0,
            page_width=595, page_height=842,
            text_bboxes=[], min_area=100)   # low threshold to catch both
        self.assertEqual(len(figs), 2)


class TestDetectTextgapFigures(unittest.TestCase):
    def test_detects_gap_between_text_blocks(self):
        # Text at top (y=0-30) then gap (y=30-300) then text at bottom (y=300-320)
        text_bboxes = [
            _bb(0,  0,   595, 30),   # header text
            _bb(0,  300, 595, 320),  # body text
        ]
        figs = detect_textgap_figures(
            text_bboxes=text_bboxes,
            existing_figure_bboxes=[],
            page_width=595, page_height=842,
            doc_id="d", page_index=0,
            min_gap_height=40)
        self.assertEqual(len(figs), 1)
        self.assertAlmostEqual(figs[0].bbox.y0, 30.0)
        self.assertAlmostEqual(figs[0].bbox.y1, 300.0)

    def test_no_gap_no_figures(self):
        # Continuous text, no gaps
        text_bboxes = [_bb(0, i*20, 595, i*20+18) for i in range(20)]
        figs = detect_textgap_figures(
            text_bboxes=text_bboxes, existing_figure_bboxes=[],
            page_width=595, page_height=842,
            doc_id="d", page_index=0, min_gap_height=40)
        self.assertEqual(len(figs), 0)

    def test_existing_figure_suppresses_gap_detection(self):
        text_bboxes = [
            _bb(0, 0,   595, 30),
            _bb(0, 300, 595, 320),
        ]
        existing = [_bb(0, 30, 595, 300)]   # already detected
        figs = detect_textgap_figures(
            text_bboxes=text_bboxes,
            existing_figure_bboxes=existing,
            page_width=595, page_height=842,
            doc_id="d", page_index=0, min_gap_height=40)
        self.assertEqual(len(figs), 0)

    def test_empty_page(self):
        figs = detect_textgap_figures(
            text_bboxes=[], existing_figure_bboxes=[],
            page_width=595, page_height=842,
            doc_id="d", page_index=0)
        self.assertEqual(figs, [])


class TestDetectAllFigures(unittest.TestCase):
    def test_deduplicates_overlapping_figures(self):
        # Two detectors find overlapping regions → only one should survive
        raster = [_fig(50, 50, 400, 350, bid="raster")]
        page   = MagicMock()
        page.get_drawings.return_value = []
        page.get_xobjects.return_value = []

        with patch("extractors.figure_detector.detect_vector_figures",
                   return_value=[_fig(55, 55, 395, 345, bid="vec")]), \
             patch("extractors.figure_detector.detect_xobject_figures",
                   return_value=[]), \
             patch("extractors.figure_detector.detect_textgap_figures",
                   return_value=[]):
            result = detect_all_figures(
                page=page, doc_id="d", page_index=0,
                page_width=595, page_height=842,
                text_bboxes=[], existing_raster_figures=raster)

        # Both overlap by >70% → only 1 kept
        self.assertEqual(len(result), 1)

    def test_non_overlapping_figures_all_kept(self):
        raster = [_fig(0, 0, 200, 200, bid="r1")]
        page   = MagicMock()
        with patch("extractors.figure_detector.detect_vector_figures",
                   return_value=[_fig(300, 400, 500, 600, bid="v1")]), \
             patch("extractors.figure_detector.detect_xobject_figures",
                   return_value=[]), \
             patch("extractors.figure_detector.detect_textgap_figures",
                   return_value=[]):
            result = detect_all_figures(
                page=page, doc_id="d", page_index=0,
                page_width=595, page_height=842,
                text_bboxes=[], existing_raster_figures=raster)

        self.assertEqual(len(result), 2)


# ─────────────────────────────────────────────────────────────
# Fix 2: Column detection
# ─────────────────────────────────────────────────────────────

class TestDetectColumns(unittest.TestCase):
    PAGE_W = 595.0

    def _make_two_col_regions(self):
        """Simulate two-column IEEE layout: left col x=0-270, right col x=310-595."""
        return [
            _region(RegionClass.PARAGRAPH, 20,  100, 260, 130, rid="L1"),
            _region(RegionClass.PARAGRAPH, 20,  140, 260, 170, rid="L2"),
            _region(RegionClass.PARAGRAPH, 310, 100, 570, 130, rid="R1"),
            _region(RegionClass.PARAGRAPH, 310, 140, 570, 170, rid="R2"),
        ]

    def test_two_column_detection(self):
        regions = self._make_two_col_regions()
        cols    = detect_columns(regions, self.PAGE_W, min_gap_width=15)
        self.assertEqual(len(cols), 2)
        # Columns may share a boundary point; verify by centre position
        self.assertLess(cols[0].centre, cols[1].centre)   # left col centre < right col centre

    def test_single_column_detection(self):
        regions = [
            _region(RegionClass.PARAGRAPH, 50, 100, 545, 130, rid="r1"),
            _region(RegionClass.PARAGRAPH, 50, 140, 545, 170, rid="r2"),
        ]
        cols = detect_columns(regions, self.PAGE_W, min_gap_width=15)
        self.assertEqual(len(cols), 1)

    def test_empty_regions_returns_one_column(self):
        cols = detect_columns([], self.PAGE_W)
        self.assertEqual(len(cols), 1)
        self.assertEqual(cols[0].x_min, 0)
        self.assertEqual(cols[0].x_max, self.PAGE_W)

    def test_column_ordering_left_to_right(self):
        regions = self._make_two_col_regions()
        cols    = detect_columns(regions, self.PAGE_W, min_gap_width=15)
        indices = [c.index for c in cols]
        self.assertEqual(indices, sorted(indices))


class TestIsFullWidth(unittest.TestCase):
    def test_full_width_title(self):
        r = _region(RegionClass.TITLE, 20, 50, 575, 70)   # spans 555/595 = 93%
        self.assertTrue(is_full_width(r, 595.0, threshold=0.55))

    def test_narrow_column_block(self):
        r = _region(RegionClass.PARAGRAPH, 20, 100, 260, 130)  # 240/595 = 40%
        self.assertFalse(is_full_width(r, 595.0, threshold=0.55))

    def test_threshold_boundary(self):
        # Block spanning exactly 55% of page
        r = _region(RegionClass.PARAGRAPH, 0, 0, 329, 20)  # 329/595 = 55.3% > 55%
        self.assertTrue(is_full_width(r, 595.0, threshold=0.55))


# ─────────────────────────────────────────────────────────────
# Fix 2: Reading order (the critical test)
# ─────────────────────────────────────────────────────────────

class TestSortReadingOrder(unittest.TestCase):
    PAGE_W = 595.0
    PAGE_H = 842.0

    def _two_col_layout(self):
        """
        Two-column layout:
          Full-width title at top
          Left col: L1(y=120), L2(y=180)
          Right col: R1(y=120), R2(y=180)
          Full-width section heading at y=240
          Left col: L3(y=260), L4(y=320)
          Right col: R3(y=260), R4(y=320)

        Expected reading order:
          title → L1 → L2 → R1 → R2 → section → L3 → L4 → R3 → R4
        """
        return [
            # Full-width title
            _region(RegionClass.TITLE,     20, 50,  575, 70,  rid="title"),
            # First band: two columns
            _region(RegionClass.PARAGRAPH, 20, 120, 260, 150, rid="L1"),
            _region(RegionClass.PARAGRAPH, 20, 180, 260, 210, rid="L2"),
            _region(RegionClass.PARAGRAPH, 310,120, 570, 150, rid="R1"),
            _region(RegionClass.PARAGRAPH, 310,180, 570, 210, rid="R2"),
            # Full-width section heading
            _region(RegionClass.TITLE,     20, 230, 575, 250, rid="section"),
            # Second band: two columns
            _region(RegionClass.PARAGRAPH, 20, 260, 260, 290, rid="L3"),
            _region(RegionClass.PARAGRAPH, 20, 320, 260, 350, rid="L4"),
            _region(RegionClass.PARAGRAPH, 310,260, 570, 290, rid="R3"),
            _region(RegionClass.PARAGRAPH, 310,320, 570, 350, rid="R4"),
        ]

    def test_two_column_reading_order(self):
        regions  = self._two_col_layout()
        sorted_r = sort_reading_order(regions, self.PAGE_W, self.PAGE_H)
        ids      = [r.region_id for r in sorted_r]

        # Title must come first
        self.assertEqual(ids[0], "title")

        # Left column must come entirely before right column
        idx_L1 = ids.index("L1"); idx_L2 = ids.index("L2")
        idx_R1 = ids.index("R1"); idx_R2 = ids.index("R2")
        self.assertLess(idx_L1, idx_R1, "L1 should come before R1")
        self.assertLess(idx_L2, idx_R1, "L2 should come before R1")
        self.assertLess(idx_L1, idx_L2, "L1 should come before L2")
        self.assertLess(idx_R1, idx_R2, "R1 should come before R2")

        # Section heading between first and second bands
        idx_section = ids.index("section")
        self.assertGreater(idx_section, idx_R2, "section after R2")

        # Second band: left before right again
        idx_L3 = ids.index("L3"); idx_R3 = ids.index("R3")
        self.assertLess(idx_L3, idx_R3, "L3 should come before R3")

    def test_naive_sort_would_fail_this_test(self):
        """Demonstrate that naive (y0,x0) sort gives WRONG order."""
        regions = self._two_col_layout()
        naive   = sorted(regions, key=lambda r: (r.bbox.y0, r.bbox.x0))
        naive_ids = [r.region_id for r in naive]

        idx_L2_naive = naive_ids.index("L2")
        idx_R1_naive = naive_ids.index("R1")

        # With naive sort, R1(y=120) comes BEFORE L2(y=180) → WRONG
        # This test documents the bug that our fix corrects.
        self.assertLess(idx_R1_naive, idx_L2_naive,
                        "Naive sort incorrectly puts R1 before L2 — this is the bug we fix")

    def test_single_column_unchanged(self):
        """Single column layout should give same order as naive sort."""
        regions = [
            _region(RegionClass.PARAGRAPH, 50, 100, 545, 130, rid="b1"),
            _region(RegionClass.PARAGRAPH, 50, 140, 545, 170, rid="b2"),
            _region(RegionClass.PARAGRAPH, 50, 180, 545, 210, rid="b3"),
        ]
        sorted_r = sort_reading_order(regions, self.PAGE_W, self.PAGE_H)
        ids = [r.region_id for r in sorted_r]
        self.assertEqual(ids, ["b1", "b2", "b3"])

    def test_full_width_blocks_interrupt_columns(self):
        regions = [
            _region(RegionClass.TITLE,     20, 50,  575, 70,  rid="title"),
            _region(RegionClass.PARAGRAPH, 20, 100, 260, 130, rid="L1"),
            _region(RegionClass.PARAGRAPH, 310,100, 570, 130, rid="R1"),
        ]
        sorted_r = sort_reading_order(regions, self.PAGE_W, self.PAGE_H)
        ids = [r.region_id for r in sorted_r]
        # Title must be first
        self.assertEqual(ids[0], "title")
        # L1 must come before R1
        self.assertLess(ids.index("L1"), ids.index("R1"))

    def test_empty_returns_empty(self):
        self.assertEqual(sort_reading_order([], self.PAGE_W, self.PAGE_H), [])

    def test_single_region(self):
        r = [_region(RegionClass.PARAGRAPH, 50, 100, 545, 130, rid="only")]
        self.assertEqual([x.region_id for x in
                          sort_reading_order(r, self.PAGE_W, self.PAGE_H)], ["only"])

    def test_reading_order_returns_all_regions(self):
        """No regions should be lost."""
        regions  = self._two_col_layout()
        sorted_r = sort_reading_order(regions, self.PAGE_W, self.PAGE_H)
        self.assertEqual(len(sorted_r), len(regions))
        original_ids = {r.region_id for r in regions}
        sorted_ids   = {r.region_id for r in sorted_r}
        self.assertEqual(original_ids, sorted_ids)


# ─────────────────────────────────────────────────────────────
# Integration: pdf_extractor uses enhanced figure detection
# ─────────────────────────────────────────────────────────────

class TestPDFExtractorWithFigureFix(unittest.TestCase):
    def test_vector_figures_detected_on_page(self):
        """
        Simulate a page where a figure exists only as vector drawing commands
        (not as a type=1 image block). The extractor should still find it.
        """
        import sys
        sys.modules['fitz'] = MagicMock()

        from extractors.pdf_extractor import PDFTextExtractor
        from extractors.models import ExtractionMethod

        with __import__('tempfile').TemporaryDirectory() as tmp:
            ext  = PDFTextExtractor(figure_output_dir=tmp)

            # Mock page: has text blocks (enough chars) but NO type=1 image blocks
            page = MagicMock()
            page.rect = MagicMock(width=595, height=842, x0=0, y0=0, x1=595, y1=842)
            page.get_text.return_value = {"blocks": [
                {"type": 0, "number": 0, "bbox": (20,20,500,40),
                 "lines": [{"spans": [{"text": "Some paper text " * 5,
                                        "font": "Arial", "size": 11.0,
                                        "flags": 0, "color": 0}]}]},
            ]}

            # Mock figure_detector to return a detected vector figure
            fake_fig = FigureBlock(
                block_id="vec_fig", bbox=_bb(50, 100, 500, 400), page_index=0)

            with patch("extractors.pdf_extractor.detect_all_figures",
                       return_value=[fake_fig]) as mock_detect:
                result = ext.extract_page(page, "doc", 0)
                mock_detect.assert_called_once()

            self.assertEqual(len(result.figure_blocks), 1)
            self.assertEqual(result.figure_blocks[0].block_id, "vec_fig")


if __name__ == "__main__":
    unittest.main(verbosity=2)