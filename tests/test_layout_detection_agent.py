"""
tests/test_layout_detection_agent.py
--------------------------------------
Unit tests for Step 3 — Layout Detection Agent.
Uses stdlib unittest only — zero extra dependencies.

Run:
    python -m unittest discover tests/ -v
    OR (with fitz mocked):
    python run_tests.py
"""
import sys, os, json, tempfile, unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from extractors.models import (
    BoundingBox, StyleAttributes, TextBlock, FigureBlock,
    BlockType, ExtractionMethod, PageExtractionResult, DocumentExtractionResult,
)
from extractors.layout_models import (
    LayoutRegion, PageLayoutResult, DocumentLayoutResult,
    RegionClass, DetectionBackend,
)
from extractors.heuristic_layout_detector import (
    HeuristicLayoutDetector, _classify_text_block,
    _math_density, _is_header_footer,
)
from extractors.ml_layout_detector import fuse_ml_and_heuristic, _pdf_bbox_from_pixel


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _bbox(x0=10, y0=10, x1=200, y1=30):
    return BoundingBox(x0=x0, y0=y0, x1=x1, y1=y1)

def _style(font_size=11.0, is_bold=False):
    return StyleAttributes(font_name="Times", font_size=font_size, is_bold=is_bold)

def _text_block(text, x0=10, y0=10, x1=500, y1=30,
                font_size=11.0, is_bold=False, block_id="b1"):
    return TextBlock(
        block_id=block_id, text=text,
        bbox=BoundingBox(x0=x0, y0=y0, x1=x1, y1=y1),
        page_index=0, style=StyleAttributes(font_size=font_size, is_bold=is_bold))

def _page(text_blocks=None, figure_blocks=None, page_index=0, w=595, h=842):
    return PageExtractionResult(
        page_index=page_index, width=w, height=h,
        text_blocks=text_blocks or [],
        figure_blocks=figure_blocks or [])

def _region(cls, x0=10, y0=10, x1=200, y1=30, conf=0.9,
            backend=DetectionBackend.HEURISTIC, region_id="r1"):
    return LayoutRegion(
        region_id=region_id, region_class=cls,
        bbox=BoundingBox(x0=x0, y0=y0, x1=x1, y1=y1),
        page_index=0, confidence=conf, backend=backend)


# ─────────────────────────────────────────────────────────────
# layout_models tests
# ─────────────────────────────────────────────────────────────

class TestLayoutRegion(unittest.TestCase):
    def test_to_dict_keys(self):
        r = _region(RegionClass.PARAGRAPH)
        d = r.to_dict()
        for key in ("region_id", "region_class", "bbox", "page_index",
                    "confidence", "backend"):
            self.assertIn(key, d)

    def test_region_class_value(self):
        r = _region(RegionClass.FIGURE)
        self.assertEqual(r.to_dict()["region_class"], "figure")

    def test_default_source_block_ids(self):
        r = _region(RegionClass.TABLE)
        self.assertEqual(r.source_block_ids, [])


class TestPageLayoutResult(unittest.TestCase):
    def test_regions_of_class(self):
        plr = PageLayoutResult(page_index=0, width=595, height=842, regions=[
            _region(RegionClass.FIGURE,    region_id="r1"),
            _region(RegionClass.TABLE,     region_id="r2"),
            _region(RegionClass.PARAGRAPH, region_id="r3"),
            _region(RegionClass.FIGURE,    region_id="r4"),
        ])
        self.assertEqual(len(plr.regions_of_class(RegionClass.FIGURE)), 2)
        self.assertEqual(len(plr.regions_of_class(RegionClass.TABLE)),  1)

    def test_to_dict(self):
        plr = PageLayoutResult(page_index=1, width=595, height=842)
        d   = plr.to_dict()
        self.assertEqual(d["page_index"], 1)
        self.assertIn("regions", d)


class TestDocumentLayoutResult(unittest.TestCase):
    def _make_dlr(self):
        page = PageLayoutResult(page_index=0, width=595, height=842, regions=[
            _region(RegionClass.TITLE,     region_id="r1"),
            _region(RegionClass.PARAGRAPH, region_id="r2"),
            _region(RegionClass.FIGURE,    region_id="r3"),
        ])
        return DocumentLayoutResult(doc_id="d1", source_path="/t.pdf",
                                    total_pages=1, pages=[page])

    def test_all_regions(self):
        self.assertEqual(len(self._make_dlr().all_regions()), 3)

    def test_regions_of_class(self):
        self.assertEqual(len(self._make_dlr().regions_of_class(RegionClass.TITLE)), 1)

    def test_stats(self):
        s = self._make_dlr().stats()
        self.assertEqual(s["total_regions"], 3)
        self.assertEqual(s["by_class"]["title"], 1)
        self.assertEqual(s["by_class"]["figure"], 1)

    def test_save_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "layout.json")
            self._make_dlr().save_json(path)
            with open(path) as f:
                data = json.load(f)
            self.assertEqual(data["doc_id"], "d1")
            self.assertEqual(len(data["pages"]), 1)


# ─────────────────────────────────────────────────────────────
# Heuristic classifier tests
# ─────────────────────────────────────────────────────────────

class TestMathDensity(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(_math_density(""), 0.0)

    def test_high_math(self):
        self.assertGreater(_math_density("∑∫∂∇αβγ = λ × ∞"), 0.06)

    def test_low_math(self):
        self.assertLess(_math_density("This is a regular sentence."), 0.06)


class TestIsHeaderFooter(unittest.TestCase):
    PAGE_H = 842.0

    def test_header(self):
        bbox = BoundingBox(0, 0, 595, 40)   # top 5% of 842
        cls  = _is_header_footer(bbox, "text", self.PAGE_H)
        self.assertEqual(cls, RegionClass.HEADER)

    def test_footer(self):
        bbox = BoundingBox(0, 810, 595, 842)  # bottom 5%
        cls  = _is_header_footer(bbox, "text", self.PAGE_H)
        self.assertEqual(cls, RegionClass.FOOTER)

    def test_body(self):
        bbox = BoundingBox(0, 200, 595, 400)
        cls  = _is_header_footer(bbox, "text", self.PAGE_H)
        self.assertIsNone(cls)


class TestClassifyTextBlock(unittest.TestCase):

    def _classify(self, text, font_size=11.0, is_bold=False,
                  y0=100, y1=120, page_idx=0, median_fs=11.0):
        blk = _text_block(text, y0=y0, y1=y1,
                          font_size=font_size, is_bold=is_bold)
        return _classify_text_block(
            blk=blk, page_height=842.0, page_width=595.0,
            median_font_size=median_fs,
            page_idx_in_doc=page_idx, seq=0, nearby_table_caption=False)

    def test_caption_fig(self):
        self.assertEqual(self._classify("Fig. 3: Results of our experiment."),
                         RegionClass.CAPTION)

    def test_caption_table(self):
        self.assertEqual(self._classify("Table 2: Comparison of methods."),
                         RegionClass.CAPTION)

    def test_caption_algorithm(self):
        self.assertEqual(self._classify("Algorithm 1: Text-Layout Alignment"),
                         RegionClass.CAPTION)

    def test_reference_bracket(self):
        self.assertEqual(self._classify("[1] P. Lewis et al., NeurIPS 2020."),
                         RegionClass.REFERENCE)

    def test_equation_math_chars(self):
        cls = self._classify("Shybrid = λ × Sv + (1-λ) × Sg ∑∫∂∇")
        self.assertEqual(cls, RegionClass.EQUATION)

    def test_abstract_first_page(self):
        cls = self._classify("Abstract",
                              page_idx=0)
        self.assertEqual(cls, RegionClass.ABSTRACT)

    def test_abstract_not_on_later_page(self):
        cls = self._classify("Abstract",
                              page_idx=2)
        # Should not be ABSTRACT on page 2
        self.assertNotEqual(cls, RegionClass.ABSTRACT)

    def test_title_large_bold_top(self):
        cls = self._classify("Agent Orchestrated Multimodal Knowledge Extraction",
                              font_size=18.0, is_bold=True, y0=50, y1=70,
                              page_idx=0, median_fs=11.0)
        self.assertEqual(cls, RegionClass.TITLE)

    def test_paragraph_default(self):
        cls = self._classify("In this work we propose a framework that combines "
                             "vector search with graph-based reasoning.")
        self.assertEqual(cls, RegionClass.PARAGRAPH)

    def test_header_top_of_page(self):
        cls = self._classify("IEEE Conference on Document Intelligence 2024",
                              y0=5, y1=25)   # within top 6% of 842
        self.assertEqual(cls, RegionClass.HEADER)

    def test_footer_bottom_of_page(self):
        cls = self._classify("© 2024 IEEE. All rights reserved.",
                              y0=815, y1=835)
        self.assertEqual(cls, RegionClass.FOOTER)


class TestHeuristicLayoutDetector(unittest.TestCase):

    def test_classify_text_blocks(self):
        blocks = [
            _text_block("Fig. 1: System architecture overview.", block_id="b1",
                        y0=200, y1=215),
            _text_block("In this section we present our approach.", block_id="b2",
                        y0=220, y1=240),
            _text_block("[1] P. Lewis et al., NeurIPS 2020.", block_id="b3",
                        y0=700, y1=715),
        ]
        fig = FigureBlock(block_id="f1", bbox=BoundingBox(50,50,300,180),
                          page_index=0)
        page = _page(text_blocks=blocks, figure_blocks=[fig])

        det    = HeuristicLayoutDetector()
        result = det.detect_page(page, doc_id="doc", page_idx_in_doc=0)

        self.assertIsInstance(result, PageLayoutResult)
        self.assertEqual(len(result.regions), 4)   # 3 text + 1 figure

        classes = {r.source_block_ids[0]: r.region_class for r in result.regions
                   if r.source_block_ids}
        self.assertEqual(classes["b1"], RegionClass.CAPTION)
        self.assertEqual(classes["b2"], RegionClass.PARAGRAPH)
        self.assertEqual(classes["b3"], RegionClass.REFERENCE)

    def test_figure_blocks_become_figure_regions(self):
        fig = FigureBlock(block_id="fig1", bbox=BoundingBox(50,50,300,200),
                          page_index=0)
        page   = _page(figure_blocks=[fig])
        det    = HeuristicLayoutDetector()
        result = det.detect_page(page, doc_id="doc", page_idx_in_doc=0)

        fig_regions = result.regions_of_class(RegionClass.FIGURE)
        self.assertEqual(len(fig_regions), 1)
        self.assertEqual(fig_regions[0].confidence, 0.95)
        self.assertEqual(fig_regions[0].backend, DetectionBackend.HEURISTIC)

    def test_reading_order_top_to_bottom(self):
        # Blocks inserted in reverse order → should be sorted top-to-bottom
        blocks = [
            _text_block("Bottom paragraph.", block_id="b_bot", y0=600, y1=620),
            _text_block("Middle paragraph.", block_id="b_mid", y0=300, y1=320),
            _text_block("Top paragraph.",    block_id="b_top", y0=100, y1=120),
        ]
        page   = _page(text_blocks=blocks)
        det    = HeuristicLayoutDetector()
        result = det.detect_page(page, doc_id="doc", page_idx_in_doc=1)

        y_positions = [r.bbox.y0 for r in result.regions]
        self.assertEqual(y_positions, sorted(y_positions))

    def test_empty_page(self):
        page   = _page()
        det    = HeuristicLayoutDetector()
        result = det.detect_page(page, doc_id="doc", page_idx_in_doc=0)
        self.assertEqual(len(result.regions), 0)

    def test_text_content_propagated(self):
        blocks = [_text_block("Sample abstract text.", block_id="b1")]
        page   = _page(text_blocks=blocks)
        det    = HeuristicLayoutDetector()
        result = det.detect_page(page, doc_id="doc", page_idx_in_doc=0)
        self.assertEqual(result.regions[0].text_content, "Sample abstract text.")


# ─────────────────────────────────────────────────────────────
# Fusion tests
# ─────────────────────────────────────────────────────────────

class TestFuseMLAndHeuristic(unittest.TestCase):

    def _make_region(self, cls, x0, y0, x1, y1, conf, backend, rid):
        return LayoutRegion(
            region_id=rid, region_class=cls,
            bbox=BoundingBox(x0=x0, y0=y0, x1=x1, y1=y1),
            page_index=0, confidence=conf, backend=backend)

    def test_heuristic_wins_on_specificity(self):
        """ML says PARAGRAPH, heuristic says CAPTION → CAPTION wins."""
        ml_r  = [self._make_region(RegionClass.PARAGRAPH, 10,10,200,30,
                                   0.85, DetectionBackend.LAYOUTPARSER, "m1")]
        heur_r = [self._make_region(RegionClass.CAPTION,  12,11,198,29,
                                    0.80, DetectionBackend.HEURISTIC, "h1")]
        fused = fuse_ml_and_heuristic(ml_r, heur_r, iou_threshold=0.3)
        self.assertEqual(len(fused), 1)
        self.assertEqual(fused[0].region_class, RegionClass.CAPTION)
        self.assertEqual(fused[0].backend, DetectionBackend.HYBRID)

    def test_ml_wins_when_equally_specific(self):
        """Both say PARAGRAPH → ML region kept (higher confidence)."""
        ml_r  = [self._make_region(RegionClass.PARAGRAPH, 10,10,200,30,
                                   0.92, DetectionBackend.LAYOUTPARSER, "m1")]
        heur_r = [self._make_region(RegionClass.PARAGRAPH, 12,11,198,29,
                                    0.80, DetectionBackend.HEURISTIC, "h1")]
        fused = fuse_ml_and_heuristic(ml_r, heur_r, iou_threshold=0.3)
        self.assertEqual(len(fused), 1)
        self.assertEqual(fused[0].region_class, RegionClass.PARAGRAPH)

    def test_heuristic_only_regions_added(self):
        """Heuristic finds a caption not detected by ML → should be added."""
        ml_r  = [self._make_region(RegionClass.FIGURE, 50,50,300,300,
                                   0.90, DetectionBackend.LAYOUTPARSER, "m1")]
        # Caption below the figure, no ML overlap
        heur_r = [
            self._make_region(RegionClass.FIGURE,  50,50,300,300,
                              0.80, DetectionBackend.HEURISTIC, "h1"),
            self._make_region(RegionClass.CAPTION, 50,305,300,320,
                              0.80, DetectionBackend.HEURISTIC, "h2"),
        ]
        fused = fuse_ml_and_heuristic(ml_r, heur_r, iou_threshold=0.3)
        classes = [r.region_class for r in fused]
        self.assertIn(RegionClass.CAPTION, classes)

    def test_no_overlap_returns_all(self):
        """Disjoint regions → all kept."""
        ml_r   = [self._make_region(RegionClass.TITLE,  0,0,200,20,
                                    0.9, DetectionBackend.LAYOUTPARSER, "m1")]
        heur_r = [self._make_region(RegionClass.CAPTION,0,700,200,720,
                                    0.8, DetectionBackend.HEURISTIC, "h1")]
        fused  = fuse_ml_and_heuristic(ml_r, heur_r, iou_threshold=0.3)
        self.assertEqual(len(fused), 2)

    def test_reading_order_preserved(self):
        """Fused result should be sorted top-to-bottom."""
        ml_r  = [self._make_region(RegionClass.PARAGRAPH, 0,500,200,520,
                                   0.9, DetectionBackend.LAYOUTPARSER, "m1")]
        heur_r = [self._make_region(RegionClass.TITLE, 0,100,200,120,
                                    0.8, DetectionBackend.HEURISTIC, "h1")]
        fused = fuse_ml_and_heuristic(ml_r, heur_r, iou_threshold=0.1)
        y0s = [r.bbox.y0 for r in fused]
        self.assertEqual(y0s, sorted(y0s))


class TestPdfBboxFromPixel(unittest.TestCase):
    def test_full_page(self):
        bbox = _pdf_bbox_from_pixel((0, 0, 1000, 1500), 1000, 1500, 595, 842)
        self.assertAlmostEqual(bbox.x0, 0)
        self.assertAlmostEqual(bbox.y0, 0)
        self.assertAlmostEqual(bbox.x1, 595)
        self.assertAlmostEqual(bbox.y1, 842)

    def test_half_width(self):
        bbox = _pdf_bbox_from_pixel((0, 0, 500, 1500), 1000, 1500, 595, 842)
        self.assertAlmostEqual(bbox.x1, 297.5)


# ─────────────────────────────────────────────────────────────
# Agent integration test
# ─────────────────────────────────────────────────────────────

class TestLayoutDetectionAgentIntegration(unittest.TestCase):

    def _make_extraction(self):
        blocks = [
            _text_block("Agent Orchestrated Knowledge Extraction",
                        font_size=16.0, is_bold=True, y0=50, y1=70, block_id="b0"),
            _text_block("Abstract—This paper presents a multimodal framework.",
                        y0=90, y1=110, block_id="b1"),
            _text_block("Fig. 1: Proposed system architecture.",
                        y0=350, y1=365, block_id="b2"),
            _text_block("[1] P. Lewis et al., Retrieval-Augmented Generation, NeurIPS 2020.",
                        y0=750, y1=765, block_id="b3"),
        ]
        fig = FigureBlock(block_id="fig0", bbox=BoundingBox(50,120,500,340),
                          page_index=0)
        page = PageExtractionResult(
            page_index=0, width=595, height=842,
            text_blocks=blocks, figure_blocks=[fig])
        return DocumentExtractionResult(
            doc_id="test_doc", source_path="/tmp/test.pdf",
            total_pages=1, pages=[page])

    def test_agent_heuristic_backend(self):
        from agents.layout_detection_agent import LayoutDetectionAgent

        with tempfile.TemporaryDirectory() as tmp:
            agent = LayoutDetectionAgent(
                backend="heuristic", output_dir=tmp, verbose=False)
            result = agent.run(self._make_extraction())

        self.assertIsInstance(result, DocumentLayoutResult)
        self.assertEqual(result.total_pages, 1)

        all_r   = result.all_regions()
        classes = [r.region_class for r in all_r]
        self.assertIn(RegionClass.FIGURE,    classes)
        self.assertIn(RegionClass.CAPTION,   classes)
        self.assertIn(RegionClass.REFERENCE, classes)

    def test_agent_saves_json(self):
        from agents.layout_detection_agent import LayoutDetectionAgent

        with tempfile.TemporaryDirectory() as tmp:
            agent = LayoutDetectionAgent(
                backend="heuristic", output_dir=tmp, verbose=False)
            result = agent.run(self._make_extraction())

            json_path = os.path.join(tmp, "test_doc", "json", "layout.json")
            self.assertTrue(os.path.exists(json_path))

            with open(json_path) as f:
                data = json.load(f)

            self.assertEqual(data["doc_id"], "test_doc")
            self.assertIn("pages", data)
            self.assertGreater(len(data["pages"][0]["regions"]), 0)

    def test_agent_backend_auto_falls_back_to_heuristic(self):
        """When no ML libs are present, auto should select heuristic."""
        from agents.layout_detection_agent import LayoutDetectionAgent

        with tempfile.TemporaryDirectory() as tmp:
            # Patch _try_import to simulate no ML libs
            with patch("agents.layout_detection_agent._try_import", return_value=False):
                agent = LayoutDetectionAgent(backend="auto", output_dir=tmp)
            self.assertEqual(agent._backend_name, "heuristic")


if __name__ == "__main__":
    unittest.main(verbosity=2)