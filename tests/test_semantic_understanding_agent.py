"""
tests/test_semantic_understanding_agent.py
==========================================
Run:  python -m pytest tests/test_semantic_understanding_agent.py -v

Uses a deterministic mock encoder so no internet / GPU is needed.
All tests run against field shapes derived from your real layout.json.
"""

import json
import hashlib
import sys
from collections import Counter
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ── Mock encoder fixture ──────────────────────────────────────────────────────

def _make_mock_encoder():
    """Hash-based deterministic 384-dim unit vectors — no network needed."""
    def mock_encode(texts, **kwargs):
        out = []
        for t in texts:
            h    = hashlib.md5(t.encode()).digest()
            seed = int.from_bytes(h[:4], "little")
            rng  = np.random.default_rng(seed)
            v    = rng.standard_normal(384).astype(np.float32)
            v   /= (np.linalg.norm(v) + 1e-9)
            out.append(v)
        return np.stack(out)

    enc = MagicMock()
    enc.encode.side_effect = mock_encode
    return enc


from agents.semantic_understanding_agent import (
    BETA,
    LayoutRegionInput,
    MultimodalLink,
    MultimodalLinker,
    ScholarlyRoleClassifier,
    SemanticRegion,
    SemanticUnderstandingAgent,
    TEXT_REGION_CLASSES,
    TARGET_CLASSES,
    _extract_typed_refs,
)


@pytest.fixture(scope="module")
def agent():
    """One agent instance per test session, using mock encoder."""
    enc = _make_mock_encoder()
    with patch("agents.semantic_understanding_agent.SentenceTransformer", return_value=enc):
        a = SemanticUnderstandingAgent.__new__(SemanticUnderstandingAgent)
        a._encoder    = enc
        a._classifier = ScholarlyRoleClassifier(enc)
        a._classifier.fit_seed()
        a._linker     = MultimodalLinker(beta=BETA)
    return a


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_regions():
    """
    Minimal region list that mirrors your real JSON field names exactly:
      region_class, text_content, bbox (dict), page_index
    """
    return [
        LayoutRegionInput("r0", "paragraph", "We define the loss as …",           {"x0":0,"y0":0,"x1":1,"y1":0.1},  0),
        LayoutRegionInput("r1", "paragraph", "Our method uses a transformer …",   {"x0":0,"y0":0.1,"x1":1,"y1":0.2},0),
        LayoutRegionInput("r2", "paragraph", "The model achieves 96% accuracy …", {"x0":0,"y0":0.2,"x1":1,"y1":0.3},0),
        LayoutRegionInput("r3", "caption",   "Figure 1. Training loss curves.",   {"x0":0,"y0":0.3,"x1":1,"y1":0.35},0),
        LayoutRegionInput("r4", "figure",    None,                                 {"x0":0,"y0":0.35,"x1":1,"y1":0.6},0),
        LayoutRegionInput("r5", "caption",   "Table 2 shows benchmark results.",  {"x0":0,"y0":0.6,"x1":1,"y1":0.65},0),
        LayoutRegionInput("r6", "table",     "Model | Acc | F1",                  {"x0":0,"y0":0.65,"x1":1,"y1":0.8},0),
    ]


# ── Field-mapping tests ───────────────────────────────────────────────────────

class TestFieldMapping:

    def test_region_class_preserved(self, agent):
        regions = _make_regions()
        results = agent.process(regions)
        for s, r in zip(results["semantic_regions"], regions):
            assert s.region_class == r.region_class

    def test_text_content_preserved(self, agent):
        regions = _make_regions()
        results = agent.process(regions)
        for s, r in zip(results["semantic_regions"], regions):
            assert s.text_content == r.text_content   # including None for figure

    def test_page_index_preserved(self, agent):
        regions = _make_regions()
        results = agent.process(regions)
        for s, r in zip(results["semantic_regions"], regions):
            assert s.page_index == r.page_index

    def test_bbox_is_dict(self, agent):
        regions = _make_regions()
        results = agent.process(regions)
        for s in results["semantic_regions"]:
            assert isinstance(s.bbox, dict)
            assert "x0" in s.bbox

    def test_embedding_shape(self, agent):
        results = agent.process(_make_regions())
        for s in results["semantic_regions"]:
            assert isinstance(s.embedding, np.ndarray)
            assert s.embedding.shape == (384,)

    def test_process_from_dict_field_keys(self, agent):
        """Verify process_from_dict reads real JSON key names."""
        layout_dict = {
            "pages": [{
                "page_index": 0,
                "regions": [
                    {
                        "region_id":        "x0",
                        "region_class":     "paragraph",   # ← NOT "region_type"
                        "text_content":     "We propose a new approach …",  # ← NOT "text"
                        "bbox":             {"x0": 0, "y0": 0, "x1": 1, "y1": 0.1},  # ← dict
                        "page_index":       0,             # ← NOT "page"
                        "confidence":       0.9,
                        "backend":          "heuristic",
                        "source_block_ids": ["b0"],
                    }
                ]
            }]
        }
        results = agent.process_from_dict(layout_dict)
        s = results["semantic_regions"][0]
        assert s.region_class    == "paragraph"
        assert s.text_content    == "We propose a new approach …"
        assert isinstance(s.bbox, dict) and "x0" in s.bbox
        assert s.page_index      == 0


# ── Role classification tests ─────────────────────────────────────────────────

class TestRoleClassification:

    def test_figure_gets_na_role(self, agent):
        """figure with null text_content must get N/A."""
        results = agent.process(_make_regions())
        fig = next(r for r in results["semantic_regions"] if r.region_class == "figure")
        assert fig.scholarly_role == "N/A"
        assert fig.role_confidence == 0.0

    def test_text_regions_get_valid_role(self, agent):
        valid = {"Definition", "Method", "Result", "Observation", "Dataset"}
        results = agent.process(_make_regions())
        for s in results["semantic_regions"]:
            if s.region_class in TEXT_REGION_CLASSES and s.text_content:
                assert s.scholarly_role in valid
                assert 0.0 < s.role_confidence <= 1.0

    def test_empty_text_gets_na(self, agent):
        regions = [LayoutRegionInput("r0", "paragraph", "",
                                     {"x0":0,"y0":0,"x1":1,"y1":1}, 0)]
        results = agent.process(regions)
        assert results["semantic_regions"][0].scholarly_role == "N/A"

    def test_none_text_gets_na(self, agent):
        regions = [LayoutRegionInput("r0", "table", None,
                                     {"x0":0,"y0":0,"x1":1,"y1":1}, 0)]
        results = agent.process(regions)
        assert results["semantic_regions"][0].scholarly_role == "N/A"

    def test_confidence_in_01(self, agent):
        results = agent.process(_make_regions())
        for s in results["semantic_regions"]:
            assert 0.0 <= s.role_confidence <= 1.0


# ── Typed-reference extraction ────────────────────────────────────────────────

class TestTypedRefExtraction:

    def test_figure_ref(self):
        assert ("figure", 1) in _extract_typed_refs("Figure 1. Shows the pipeline.")

    def test_table_ref(self):
        assert ("table", 2) in _extract_typed_refs("Table 2 shows results.")

    def test_fig_dot(self):
        assert ("figure", 3) in _extract_typed_refs("as shown in Fig. 3")

    def test_none_returns_empty(self):
        assert _extract_typed_refs(None) == []

    def test_no_ref_returns_empty(self):
        assert _extract_typed_refs("This is just a paragraph.") == []


# ── Multimodal linker / Equation 4 tests ─────────────────────────────────────

class TestMultimodalLinker:

    def _make_sem(self, region_id, region_class, text_content, page=0):
        enc = _make_mock_encoder()
        text = text_content or "[EMPTY]"
        h    = hashlib.md5(text.encode()).digest()
        rng  = np.random.default_rng(int.from_bytes(h[:4], "little"))
        emb  = (rng.standard_normal(384) / 10).astype(np.float32)
        emb /= (np.linalg.norm(emb) + 1e-9)
        return SemanticRegion(
            region_id=region_id, region_class=region_class,
            text_content=text_content, bbox={"x0":0,"y0":0,"x1":1,"y1":1},
            page_index=page, scholarly_role="N/A", role_confidence=0.0,
            embedding=emb,
        )

    def test_equation_4_formula(self):
        linker = MultimodalLinker(beta=0.7)
        cap = self._make_sem("c0", "caption", "Figure 1. Shows results.")
        fig = self._make_sem("f0", "figure",  None)
        links = linker.link([cap], [fig])
        assert len(links) == 1
        lnk = links[0]
        expected = round(0.7 * lnk.s_ref + 0.3 * lnk.s_emb, 4)
        assert abs(lnk.s_link - expected) < 1e-3

    def test_s_ref_figure_caption_same_page(self):
        linker = MultimodalLinker()
        cap = self._make_sem("c0", "caption", "Figure 1. Main architecture.", page=0)
        fig = self._make_sem("f0", "figure",  None,                           page=0)
        links = linker.link([cap], [fig])
        assert links[0].s_ref == 1.0
        assert ("figure", 1) in links[0].matched_refs

    def test_s_ref_cross_page(self):
        linker = MultimodalLinker()
        cap = self._make_sem("c0", "caption", "Figure 1. Main architecture.", page=0)
        fig = self._make_sem("f0", "figure",  None,                           page=1)
        links = linker.link([cap], [fig])
        assert links[0].s_ref == 0.3

    def test_s_ref_table_match(self):
        linker = MultimodalLinker()
        cap = self._make_sem("c0", "caption", "Table 2 shows results.", page=0)
        tbl = self._make_sem("t0", "table",   "Col1 Col2",              page=0)
        links = linker.link([cap], [tbl])
        assert links[0].s_ref == 1.0

    def test_s_emb_in_01(self):
        linker = MultimodalLinker()
        cap = self._make_sem("c0", "caption", "Some caption text.", page=0)
        fig = self._make_sem("f0", "figure",  None,                 page=0)
        links = linker.link([cap], [fig])
        assert 0.0 <= links[0].s_emb <= 1.0

    def test_s_link_in_01(self):
        linker = MultimodalLinker()
        cap = self._make_sem("c0", "caption", "Figure 1.", page=0)
        fig = self._make_sem("f0", "figure",  None,        page=0)
        links = linker.link([cap], [fig])
        assert 0.0 <= links[0].s_link <= 1.0

    def test_no_links_empty_input(self):
        linker = MultimodalLinker()
        assert linker.link([], []) == []

    def test_top_k_respected(self, agent):
        results = agent.process(_make_regions(), top_k_links=1)
        counts = Counter(lnk.caption_id for lnk in results["multimodal_links"])
        assert all(v <= 1 for v in counts.values())


# ── Integration tests ─────────────────────────────────────────────────────────

class TestIntegration:

    def test_empty_input(self, agent):
        results = agent.process([])
        assert results["semantic_regions"] == []
        assert results["multimodal_links"]  == []

    def test_full_pipeline_region_count(self, agent):
        regions = _make_regions()
        results = agent.process(regions)
        assert len(results["semantic_regions"]) == len(regions)

    def test_serialize_roundtrip(self, agent):
        results = agent.process(_make_regions())
        out  = SemanticUnderstandingAgent.serialize(
            results["semantic_regions"], results["multimodal_links"]
        )
        js     = json.dumps(out)
        loaded = json.loads(js)
        assert len(loaded["semantic_regions"]) == len(results["semantic_regions"])
        assert "embedding"       not in loaded["semantic_regions"][0]
        assert "scholarly_role"  in loaded["semantic_regions"][0]
        assert "s_link"          in loaded["multimodal_links"][0]

    def test_process_from_dict_with_none_text(self, agent):
        layout_dict = {
            "pages": [{
                "page_index": 0,
                "regions": [
                    {"region_id":"r0","region_class":"figure",
                     "text_content": None,
                     "bbox":{"x0":0,"y0":0,"x1":1,"y1":0.5},
                     "page_index":0,"confidence":0.95,
                     "backend":"heuristic","source_block_ids":[]},
                    {"region_id":"r1","region_class":"caption",
                     "text_content": "Figure 1. An overview.",
                     "bbox":{"x0":0,"y0":0.5,"x1":1,"y1":0.6},
                     "page_index":0,"confidence":0.8,
                     "backend":"heuristic","source_block_ids":[]},
                ]
            }]
        }
        results = agent.process_from_dict(layout_dict)
        assert len(results["semantic_regions"]) == 2
        assert len(results["multimodal_links"])  == 1
        lnk = results["multimodal_links"][0]
        assert lnk.s_ref == 1.0   # "Figure 1" in caption → figure on same page
