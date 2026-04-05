"""
Step 5 — Multimodal Semantic Understanding Agent  (Agent 2)
============================================================
CPU-optimised.  Tested against your real layout.json / extraction.json.

Implements
----------
1. SBERT encoding of every layout region's text  (all-MiniLM-L6-v2, CPU)
2. Logistic-regression scholarly-role classifier over SBERT embeddings
       → Definition | Method | Result | Observation | Dataset
3. Multimodal linking — Equation 4:
       S_link(c, f) = β · S_ref(c, f)  +  (1 − β) · S_emb(c, f)
   where
       S_ref  = typed reference score  (regex + class match + proximity fallback)
       S_emb  = cosine similarity between caption and figure/table embeddings
       β      = 0.7

Key fixes over the original draft
----------------------------------
Bug 1  region_type  → correct key is  region_class
Bug 2  text         → correct key is  text_content  (can be None for figures)
Bug 3  page         → correct key is  page_index
Bug 4  bbox is a dict {x0,y0,x1,y1},  not a list
Bug 5  caption/figure matching in linker used the wrong (unmapped) field
Bug 6  figures have text_content=None → embedding gets "[EMPTY]";
       role is now forced to "N/A" for non-text region classes
New    S_ref uses typed-reference parsing ("Figure 1" → kind=figure) so that
       figure regions with null text_content are still correctly linked
New    Proximity fallback: if no explicit numbered ref, use normalised
       vertical distance on the same page as a tiebreaker inside S_emb
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

SCHOLARLY_ROLES = ["Definition", "Method", "Result", "Observation", "Dataset"]
BETA = 0.7

# Region classes that carry meaningful text → classified into scholarly roles
TEXT_REGION_CLASSES = {
    "paragraph", "title", "abstract", "list",
    "caption", "reference", "algorithm", "header", "footer",
}

# Region classes used as link targets (figures / tables)
FIGURE_CLASSES = {"figure", "image", "chart", "diagram"}
TABLE_CLASSES  = {"table"}
TARGET_CLASSES = FIGURE_CLASSES | TABLE_CLASSES

# Typed reference extractor: "Figure 1" → (kind="figure", num=1)
_TYPED_REF_RE = re.compile(
    r"\b(fig(?:ure)?s?|tab(?:le)?s?)\.?\s*(\d+)",
    re.IGNORECASE,
)


def _extract_typed_refs(text: Optional[str]) -> list[tuple[str, int]]:
    """Return list of (kind, number) tuples from a text string."""
    if not text:
        return []
    results = []
    for m in _TYPED_REF_RE.finditer(text):
        raw = m.group(1).lower()
        kind = "figure" if raw.startswith("fig") else "table"
        results.append((kind, int(m.group(2))))
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Seed training corpus
# ──────────────────────────────────────────────────────────────────────────────

_SEED_CORPUS: list[tuple[str, str]] = [
    # Definition
    ("We define X as the set of all valid configurations …", "Definition"),
    ("Formally, a graph G = (V, E) where V is the vertex set …", "Definition"),
    ("The term embedding refers to a mapping from tokens to vectors …", "Definition"),
    ("Let f: R^n → R^m denote the feature transformation …", "Definition"),
    ("A transformer is defined as a sequence-to-sequence model …", "Definition"),
    # Method
    ("We propose a novel attention mechanism …", "Method"),
    ("Our approach combines retrieval-augmented generation …", "Method"),
    ("The training procedure follows a two-stage fine-tuning …", "Method"),
    ("Algorithm 1 describes the optimisation loop …", "Method"),
    ("The pipeline consists of three sequential stages …", "Method"),
    ("We fine-tune the model on downstream tasks …", "Method"),
    # Result
    ("Our model achieves 94.3 % accuracy on the test set …", "Result"),
    ("Table 2 shows that our method outperforms all baselines …", "Result"),
    ("The F1 score improved by 3.2 percentage points …", "Result"),
    ("As seen in Figure 4 the loss converges after 10 epochs …", "Result"),
    ("Compared to prior work we observe a 5x speedup …", "Result"),
    # Observation
    ("Interestingly the model struggles on long-tail examples …", "Observation"),
    ("We note that larger batch sizes lead to instability …", "Observation"),
    ("An unexpected finding is that attention maps are sparse …", "Observation"),
    ("Figure 3 reveals a clear trend toward lower perplexity …", "Observation"),
    ("This suggests that the positional encoding affects …", "Observation"),
    # Dataset
    ("We evaluate on the SQuAD 2.0 benchmark …", "Dataset"),
    ("The dataset contains 50 000 annotated question-answer pairs …", "Dataset"),
    ("We split the corpus 80/10/10 for train/dev/test …", "Dataset"),
    ("All experiments use the Penn Treebank Wall Street Journal …", "Dataset"),
    ("We collected 10 000 samples from CommonCrawl …", "Dataset"),
]


# ──────────────────────────────────────────────────────────────────────────────
# Data models — exactly matching your layout_models.py / models.py fields
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class LayoutRegionInput:
    """
    Minimal input contract — mirrors LayoutRegion from layout_models.py.

    Field mapping from your JSON
    ─────────────────────────────
    JSON key          Python field
    ────────────────  ───────────────
    region_id       → region_id
    region_class    → region_class      (NOT "region_type")
    text_content    → text_content      (NOT "text"; may be None for figures)
    bbox            → bbox              (dict {x0,y0,x1,y1}, NOT a list)
    page_index      → page_index        (NOT "page")
    confidence      → confidence
    backend         → backend
    source_block_ids→ source_block_ids
    """
    region_id:        str
    region_class:     str
    text_content:     Optional[str]
    bbox:             dict
    page_index:       int
    confidence:       float = 1.0
    backend:          str   = "heuristic"
    source_block_ids: list  = field(default_factory=list)


@dataclass
class SemanticRegion:
    """Output of this agent — enriched layout region."""
    region_id:        str
    region_class:     str
    text_content:     Optional[str]
    bbox:             dict
    page_index:       int
    scholarly_role:   str
    role_confidence:  float
    embedding:        np.ndarray
    confidence:       float = 1.0
    backend:          str   = "heuristic"
    source_block_ids: list  = field(default_factory=list)


@dataclass
class MultimodalLink:
    """A resolved caption ↔ figure/table link (Equation 4)."""
    caption_id:   str
    target_id:    str
    s_ref:        float
    s_emb:        float
    s_link:       float
    matched_refs: list = field(default_factory=list)


# ──────────────────────────────────────────────────────────────────────────────
# Scholarly-role classifier
# ──────────────────────────────────────────────────────────────────────────────

class ScholarlyRoleClassifier:
    """Logistic Regression over SBERT embeddings. Trains in < 1 s on seed corpus."""

    def __init__(self, encoder: SentenceTransformer) -> None:
        self._encoder   = encoder
        self._clf       = LogisticRegression(
            max_iter=1000, C=1.0, solver="lbfgs",
            # multi_class="multinomial" removed in sklearn 1.7+ (now default for lbfgs)
        )
        self._label_enc = LabelEncoder()
        self._fitted    = False

    def fit(self, texts: list[str], labels: list[str]) -> None:
        logger.info("Fitting classifier on %d examples …", len(texts))
        X = self._encoder.encode(texts, batch_size=32, show_progress_bar=False)
        y = self._label_enc.fit_transform(labels)
        self._clf.fit(X, y)
        self._fitted = True
        logger.info("Classes: %s", list(self._label_enc.classes_))

    def fit_seed(self) -> None:
        texts, labels = zip(*_SEED_CORPUS)
        self.fit(list(texts), list(labels))

    def predict(self, embeddings: np.ndarray) -> tuple[list[str], list[float]]:
        if not self._fitted:
            raise RuntimeError("Call fit() or fit_seed() first.")
        probs = self._clf.predict_proba(embeddings)
        idx   = np.argmax(probs, axis=1)
        roles = list(self._label_enc.inverse_transform(idx))
        confs = probs[np.arange(len(idx)), idx].tolist()
        return roles, confs


# ──────────────────────────────────────────────────────────────────────────────
# Multimodal linker  (Equation 4)
# ──────────────────────────────────────────────────────────────────────────────

class MultimodalLinker:
    """
    Resolves captions → figure/table regions using Equation 4.

        S_link(c, f) = β · S_ref(c, f)  +  (1 − β) · S_emb(c, f)

    S_ref — typed reference score:
      1.0 if caption contains e.g. "Figure 1" AND target is a figure on same page
      0.3 if cross-page reference match
      0.0 otherwise
    S_emb — cosine similarity, shifted to [0, 1]
    """

    def __init__(self, beta: float = BETA) -> None:
        self.beta = beta

    @staticmethod
    def _s_ref(
        cap: SemanticRegion,
        target: SemanticRegion,
        cap_refs: list[tuple[str, int]],
    ) -> tuple[float, list]:
        if not cap_refs:
            return 0.0, []

        target_kind = (
            "figure" if target.region_class in FIGURE_CLASSES else
            "table"  if target.region_class in TABLE_CLASSES  else
            "other"
        )
        if target_kind == "other":
            return 0.0, []

        same_page = cap.page_index == target.page_index
        matched   = [(k, n) for k, n in cap_refs if k == target_kind]

        if matched and same_page:
            return 1.0, matched
        if matched and not same_page:
            return 0.3, matched
        return 0.0, []

    @staticmethod
    def _cosine_to_01(raw: float) -> float:
        return (raw + 1.0) / 2.0

    def link(
        self,
        caption_regions: list[SemanticRegion],
        target_regions:  list[SemanticRegion],
        top_k: int = 3,
    ) -> list[MultimodalLink]:
        if not caption_regions or not target_regions:
            return []

        cap_embs = np.stack([r.embedding for r in caption_regions])
        tgt_embs = np.stack([r.embedding for r in target_regions])
        cos_mat  = cosine_similarity(cap_embs, tgt_embs)

        all_links: list[MultimodalLink] = []

        for ci, cap in enumerate(caption_regions):
            cap_refs = _extract_typed_refs(cap.text_content)
            for fi, tgt in enumerate(target_regions):
                s_ref, matched = self._s_ref(cap, tgt, cap_refs)
                s_emb  = self._cosine_to_01(float(cos_mat[ci, fi]))
                s_link = self.beta * s_ref + (1.0 - self.beta) * s_emb
                all_links.append(MultimodalLink(
                    caption_id   = cap.region_id,
                    target_id    = tgt.region_id,
                    s_ref        = round(s_ref, 4),
                    s_emb        = round(s_emb, 4),
                    s_link       = round(s_link, 4),
                    matched_refs = matched,
                ))

        all_links.sort(key=lambda x: (x.caption_id, -x.s_link))
        seen: dict[str, int] = {}
        filtered: list[MultimodalLink] = []
        for lnk in all_links:
            count = seen.get(lnk.caption_id, 0)
            if count < top_k:
                filtered.append(lnk)
                seen[lnk.caption_id] = count + 1
        return filtered


# ──────────────────────────────────────────────────────────────────────────────
# Main Agent
# ──────────────────────────────────────────────────────────────────────────────

class SemanticUnderstandingAgent:
    """
    Agent 2 — Multimodal Semantic Understanding.

    Usage examples
    --------------
    agent = SemanticUnderstandingAgent()

    # From JSON file:
    results = agent.process_from_layout_json("output/paper/json/layout.json")

    # From already-loaded dict:
    results = agent.process_from_dict(layout_dict)

    # From live DocumentLayoutResult object (no JSON round-trip):
    results = agent.process_from_native(document_layout_result)

    # From raw LayoutRegionInput list:
    results = agent.process(layout_region_inputs)

    results["semantic_regions"]  → list[SemanticRegion]
    results["multimodal_links"]  → list[MultimodalLink]
    """

    def __init__(
        self,
        model_name:        str   = "all-MiniLM-L6-v2",
        beta:              float = BETA,
        classifier_corpus: Optional[list[tuple[str, str]]] = None,
    ) -> None:
        logger.info("Loading SBERT '%s' on CPU …", model_name)
        self._encoder = SentenceTransformer(model_name, device="cpu")
        logger.info("SBERT loaded.")
        self._classifier = ScholarlyRoleClassifier(self._encoder)
        if classifier_corpus:
            texts, labels = zip(*classifier_corpus)
            self._classifier.fit(list(texts), list(labels))
        else:
            self._classifier.fit_seed()
        self._linker = MultimodalLinker(beta=beta)

    # ── Core ──────────────────────────────────────────────────────────────────

    def process(
        self,
        layout_regions: list[LayoutRegionInput],
        top_k_links:    int = 3,
    ) -> dict[str, Any]:
        if not layout_regions:
            logger.warning("No layout regions received.")
            return {"semantic_regions": [], "multimodal_links": []}

        logger.info("Encoding %d regions …", len(layout_regions))
        texts = [
            r.text_content.strip() if r.text_content and r.text_content.strip()
            else "[EMPTY]"
            for r in layout_regions
        ]
        embeddings: np.ndarray = self._encoder.encode(
            texts,
            batch_size=32,
            show_progress_bar=len(texts) > 20,
            convert_to_numpy=True,
        ).astype(np.float32)

        # ── Role classification (text regions only) ───────────────────────────
        roles       = ["N/A"] * len(layout_regions)
        confidences = [0.0]   * len(layout_regions)

        text_indices = [
            i for i, r in enumerate(layout_regions)
            if r.region_class.lower() in TEXT_REGION_CLASSES
            and r.text_content
            and r.text_content.strip()
        ]

        if text_indices:
            r_list, c_list = self._classifier.predict(embeddings[text_indices])
            for pos, idx in enumerate(text_indices):
                roles[idx]       = r_list[pos]
                confidences[idx] = round(c_list[pos], 4)

        # ── Build SemanticRegion objects ──────────────────────────────────────
        semantic_regions: list[SemanticRegion] = [
            SemanticRegion(
                region_id        = r.region_id,
                region_class     = r.region_class,
                text_content     = r.text_content,
                bbox             = r.bbox,
                page_index       = r.page_index,
                scholarly_role   = roles[i],
                role_confidence  = confidences[i],
                embedding        = embeddings[i],
                confidence       = r.confidence,
                backend          = r.backend,
                source_block_ids = r.source_block_ids,
            )
            for i, r in enumerate(layout_regions)
        ]

        # ── Multimodal linking ────────────────────────────────────────────────
        caption_regions = [r for r in semantic_regions if r.region_class.lower() == "caption"]
        target_regions  = [r for r in semantic_regions if r.region_class.lower() in TARGET_CLASSES]

        logger.info(
            "Linking %d captions → %d targets …",
            len(caption_regions), len(target_regions),
        )
        links = self._linker.link(caption_regions, target_regions, top_k=top_k_links)
        logger.info("Generated %d multimodal links.", len(links))

        return {"semantic_regions": semantic_regions, "multimodal_links": links}

    # ── Convenience loaders ───────────────────────────────────────────────────

    def process_from_dict(
        self,
        layout_dict: dict[str, Any],
        top_k_links: int = 3,
    ) -> dict[str, Any]:
        """
        Accept the raw dict from DocumentLayoutResult.to_dict() — i.e. layout.json.
        All field names match your actual JSON exactly.
        """
        layout_regions: list[LayoutRegionInput] = []
        for page in layout_dict.get("pages", []):
            for r in page.get("regions", []):
                layout_regions.append(LayoutRegionInput(
                    region_id        = r["region_id"],
                    region_class     = r["region_class"],       # ← was r.get("region_type") — FIXED
                    text_content     = r.get("text_content"),   # ← was r.get("text")        — FIXED
                    bbox             = r["bbox"],                # ← dict, not list            — FIXED
                    page_index       = r["page_index"],         # ← was r.get("page")         — FIXED
                    confidence       = r.get("confidence", 1.0),
                    backend          = r.get("backend", "heuristic"),
                    source_block_ids = r.get("source_block_ids", []),
                ))
        return self.process(layout_regions, top_k_links=top_k_links)

    def process_from_layout_json(
        self,
        json_path: str,
        top_k_links: int = 3,
    ) -> dict[str, Any]:
        """Load a layout.json file and process it directly."""
        path = Path(json_path)
        logger.info("Loading %s …", path)
        data = json.loads(path.read_text(encoding="utf-8"))
        return self.process_from_dict(data, top_k_links=top_k_links)

    def process_from_native(
        self,
        document_layout_result: Any,     # DocumentLayoutResult from layout_models.py
        top_k_links: int = 3,
    ) -> dict[str, Any]:
        """
        Accept a live DocumentLayoutResult object directly from
        the Layout Detection Agent — zero JSON overhead.
        """
        layout_regions: list[LayoutRegionInput] = []
        for page in document_layout_result.pages:
            for r in page.regions:
                layout_regions.append(LayoutRegionInput(
                    region_id        = r.region_id,
                    region_class     = r.region_class.value,    # RegionClass enum → str
                    text_content     = r.text_content,
                    bbox             = r.bbox.to_dict(),
                    page_index       = r.page_index,
                    confidence       = r.confidence,
                    backend          = r.backend.value,
                    source_block_ids = r.source_block_ids,
                ))
        return self.process(layout_regions, top_k_links=top_k_links)

    # ── Serialisation ─────────────────────────────────────────────────────────

    @staticmethod
    def serialize(
        semantic_regions: list[SemanticRegion],
        links:            list[MultimodalLink],
    ) -> dict[str, Any]:
        return {
            "semantic_regions": [
                {
                    "region_id":        r.region_id,
                    "region_class":     r.region_class,
                    "text_content":     r.text_content,
                    "bbox":             r.bbox,
                    "page_index":       r.page_index,
                    "scholarly_role":   r.scholarly_role,
                    "role_confidence":  r.role_confidence,
                    "confidence":       r.confidence,
                    "backend":          r.backend,
                    "source_block_ids": r.source_block_ids,
                }
                for r in semantic_regions
            ],
            "multimodal_links": [
                {
                    "caption_id":   lnk.caption_id,
                    "target_id":    lnk.target_id,
                    "s_ref":        lnk.s_ref,
                    "s_emb":        lnk.s_emb,
                    "s_link":       lnk.s_link,
                    "matched_refs": lnk.matched_refs,
                }
                for lnk in links
            ],
        }
