"""
tests/test_document_extraction_agent.py
Unit tests using Python stdlib unittest — zero extra dependencies.
Run: python -m pytest tests/ -v   OR   python -m unittest discover tests/
"""
import sys, os, json, tempfile, unittest
from unittest.mock import MagicMock, patch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from extractors.models import (
    BoundingBox, StyleAttributes, TextBlock, FigureBlock,
    BlockType, ExtractionMethod, PageExtractionResult, DocumentExtractionResult
)
from extractors.pdf_extractor import (
    PDFTextExtractor, _make_block_id, _dominant_style, _guess_block_type
)


class TestBoundingBox(unittest.TestCase):
    def test_dimensions(self):
        bb = BoundingBox(10, 20, 110, 70)
        self.assertEqual(bb.width, 100)
        self.assertEqual(bb.height, 50)
        self.assertEqual(bb.area, 5000)

    def test_zero_area(self):
        self.assertEqual(BoundingBox(5,5,5,5).area, 0)

    def test_to_tuple(self):
        self.assertEqual(BoundingBox(1,2,3,4).to_tuple(), (1,2,3,4))

    def test_to_dict(self):
        d = BoundingBox(1,2,3,4).to_dict()
        self.assertIn("x0", d)
        self.assertEqual(d["x1"], 3)


class TestStyleAttributes(unittest.TestCase):
    def test_defaults_none(self):
        s = StyleAttributes()
        self.assertIsNone(s.font_name)
        self.assertIsNone(s.font_size)

    def test_set_values(self):
        s = StyleAttributes(font_name="Arial", font_size=12.0, is_bold=True)
        self.assertEqual(s.font_name, "Arial")
        self.assertTrue(s.is_bold)

    def test_to_dict(self):
        s = StyleAttributes(font_name="Times", font_size=11.0)
        d = s.to_dict()
        self.assertEqual(d["font_name"], "Times")


class TestTextBlock(unittest.TestCase):
    def _make_block(self):
        return TextBlock(
            block_id="abc123",
            text="Hello world",
            bbox=BoundingBox(0,0,100,20),
            page_index=0)

    def test_construction(self):
        tb = self._make_block()
        self.assertEqual(tb.text, "Hello world")
        self.assertEqual(tb.confidence, 1.0)
        self.assertEqual(tb.extraction_method, ExtractionMethod.EMBEDDED)

    def test_to_dict(self):
        d = self._make_block().to_dict()
        self.assertIn("block_id", d)
        self.assertIn("bbox", d)
        self.assertEqual(d["text"], "Hello world")


class TestDocumentExtractionResult(unittest.TestCase):
    def _make_result(self):
        bb = BoundingBox(0,0,100,20)
        blocks = [TextBlock(block_id=f"b{i}", text=f"Block {i}",
                            bbox=bb, page_index=0) for i in range(3)]
        page = PageExtractionResult(page_index=0, width=595, height=842,
                                    text_blocks=blocks)
        return DocumentExtractionResult(doc_id="test", source_path="/tmp/t.pdf",
                                        total_pages=1, pages=[page])

    def test_all_text_blocks(self):
        self.assertEqual(len(self._make_result().all_text_blocks()), 3)

    def test_blocks_for_page_hit(self):
        self.assertEqual(len(self._make_result().blocks_for_page(0)), 3)

    def test_blocks_for_page_miss(self):
        self.assertEqual(self._make_result().blocks_for_page(99), [])

    def test_stats(self):
        s = self._make_result().stats()
        self.assertEqual(s["total_pages"], 1)
        self.assertEqual(s["total_blocks"], 3)
        self.assertEqual(s["ocr_pages"], 0)

    def test_save_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "out.json")
            self._make_result().save_json(path)
            with open(path) as f:
                data = json.load(f)
            self.assertEqual(data["doc_id"], "test")
            self.assertEqual(len(data["pages"]), 1)


class TestHelpers(unittest.TestCase):
    def test_block_id_deterministic(self):
        self.assertEqual(_make_block_id("d",0,1), _make_block_id("d",0,1))

    def test_block_id_unique(self):
        self.assertNotEqual(_make_block_id("d",0,1), _make_block_id("d",0,2))

    def test_block_id_length(self):
        self.assertEqual(len(_make_block_id("doc",1,2)), 12)

    def test_guess_equation(self):
        self.assertEqual(_guess_block_type("∑∫∂∇αβ = λ × ∞", StyleAttributes()),
                         BlockType.EQUATION)

    def test_guess_text(self):
        self.assertEqual(_guess_block_type("Normal academic paragraph text here.",
                                           StyleAttributes()), BlockType.TEXT)

    def test_dominant_style_empty(self):
        s = _dominant_style([])
        self.assertIsNone(s.font_name)

    def test_dominant_style_picks_longest(self):
        spans = [
            {"text": "hi",          "font": "FontA", "size": 10, "flags": 0, "color": 0},
            {"text": "hello world", "font": "FontB", "size": 12, "flags": 16, "color": 0},
        ]
        s = _dominant_style(spans)
        self.assertEqual(s.font_name, "FontB")
        self.assertTrue(s.is_bold)


class TestPDFTextExtractorMocked(unittest.TestCase):
    """Tests using mocked fitz.Page — no real PDF needed."""

    def _extractor(self, tmpdir):
        return PDFTextExtractor(ocr_threshold=50, render_dpi=72,
                                figure_output_dir=str(tmpdir))

    def _text_blk_dict(self, text, bbox=(10,10,200,30), blk_no=0):
        return {"type": 0, "number": blk_no, "bbox": bbox,
                "lines": [{"spans": [{"text": text, "font": "Arial",
                                       "size": 11.0, "flags": 0, "color": 0}]}]}

    def _mock_page(self, blocks, w=595, h=842):
        page = MagicMock()
        page.rect = MagicMock(width=w, height=h, x0=0, y0=0, x1=w, y1=h)
        page.get_text.return_value = {"blocks": blocks}
        return page

    def test_basic_extraction(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ext  = self._extractor(tmpdir)
            page = self._mock_page([
                self._text_blk_dict("Abstract: this paper presents...", blk_no=0),
                self._text_blk_dict("Introduction section here.",       blk_no=1),
            ])
            result = ext.extract_page(page, "test", 0)
            self.assertEqual(len(result.text_blocks), 2)
            self.assertFalse(result.ocr_triggered)
            self.assertEqual(result.extraction_method, ExtractionMethod.EMBEDDED)

    def test_empty_page_triggers_ocr(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ext  = self._extractor(tmpdir)
            page = self._mock_page([])   # zero chars → triggers OCR
            dummy = TextBlock(block_id="o1", text="OCR text",
                              bbox=BoundingBox(0,0,100,20), page_index=0,
                              extraction_method=ExtractionMethod.OCR, confidence=0.9)
            with patch.object(ext, "_ocr_page", return_value=[dummy]) as mock_ocr:
                result = ext.extract_page(page, "test", 0)
                mock_ocr.assert_called_once()
            self.assertTrue(result.ocr_triggered)
            self.assertEqual(result.extraction_method, ExtractionMethod.OCR)

    def test_image_block_creates_figure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ext = self._extractor(tmpdir)
            img_blk = {"type": 1, "number": 0, "bbox": (50,100,300,400)}
            # text block with enough chars to NOT trigger OCR
            txt_blk = self._text_blk_dict("Word " * 20, blk_no=1)
            page = self._mock_page([txt_blk, img_blk])
            with patch.object(ext, "_save_figure_crop", return_value="/tmp/fig.png"):
                result = ext.extract_page(page, "test", 0)
            self.assertEqual(len(result.figure_blocks), 1)
            self.assertEqual(result.figure_blocks[0].bbox.x0, 50)

    def test_skips_empty_text(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ext  = self._extractor(tmpdir)
            page = self._mock_page([
                self._text_blk_dict("",        blk_no=0),
                self._text_blk_dict("   ",     blk_no=1),
                self._text_blk_dict("Valid " * 15, blk_no=2),
            ])
            result = ext.extract_page(page, "test", 0)
            # only the non-empty block survives
            texts = [b.text for b in result.text_blocks]
            self.assertFalse(any(t.strip() == "" for t in texts))


class TestAgentIntegration(unittest.TestCase):
    """Full agent integration test — mocked fitz, no real PDF."""

    def test_run_saves_json(self):
        from agents.document_extraction_agent import DocumentExtractionAgent

        with tempfile.TemporaryDirectory() as tmpdir:
            agent = DocumentExtractionAgent(
                output_dir=os.path.join(tmpdir, "out"),
                ocr_threshold=50, verbose=False)

            dummy_pdf = os.path.join(tmpdir, "paper.pdf")
            with open(dummy_pdf, "wb") as f:
                f.write(b"%PDF-1.4 dummy")

            # Build a fake PageExtractionResult
            bb   = BoundingBox(0,0,200,20)
            fake_page = PageExtractionResult(
                page_index=0, width=595, height=842,
                text_blocks=[TextBlock(block_id="b1", text="Sample text.",
                                       bbox=bb, page_index=0)])

            with patch("agents.document_extraction_agent.fitz") as mf, \
                 patch("agents.document_extraction_agent.PDFTextExtractor") as ME:

                mock_doc = MagicMock()
                mock_doc.__len__ = MagicMock(return_value=1)
                mock_doc.__getitem__ = MagicMock(return_value=MagicMock())
                mf.open.return_value = mock_doc
                ME.return_value.extract_page.return_value = fake_page

                result = agent.run(dummy_pdf, doc_id="paper")

            json_path = os.path.join(tmpdir, "out", "paper", "json", "extraction.json")
            self.assertTrue(os.path.exists(json_path), "extraction.json not saved")

            with open(json_path) as f:
                data = json.load(f)

            self.assertEqual(data["doc_id"], "paper")
            self.assertEqual(data["total_pages"], 1)
            self.assertEqual(data["pages"][0]["text_blocks"][0]["text"], "Sample text.")


if __name__ == "__main__":
    unittest.main(verbosity=2)