# agents/faiss_agent.py
"""
Step 6 — FAISS Vector Database Agent
======================================
Builds and queries a FAISS index over SemanticRegion embeddings.
Sits between the SemanticUnderstandingAgent (Step 4/5) and the
HybridRetrievalEngine (Step 7), providing the semantic half of hybrid search.

Architecture
────────────
                  semantic.json
                       │
              ┌────────▼────────┐
              │  FAISSAgent     │
              │  .build()       │  ← re-encodes text with all-MiniLM-L6-v2
              │                 │    (embeddings not stored in semantic.json)
              │  FAISS Index    │  ← IndexFlatIP  (exact, ≤ 500 nodes)
              │  + MetadataStore│    IndexIVFFlat (approximate, > 500 nodes)
              └────────┬────────┘
                       │  .search(query, top_k)
                       ▼
              list[RetrievalResult]
              ─────────────────────
              node_id | text | score
              role    | page | section_id
              region_class | confidence

Design decisions
────────────────
1. Two index modes auto-selected by corpus size:
     ≤ 500 nodes → IndexFlatIP   (exact inner-product, no training needed)
     >  500 nodes → IndexIVFFlat (approximate, trained, faster at scale)
   Inner-product on L2-normalised vectors == cosine similarity.

2. Embeddings re-encoded here with the SAME model/device as
   SemanticUnderstandingAgent so that query vectors are in the same space.

3. Noise regions (footer, header, N/A role) are indexed but flagged —
   the caller can filter them via RetrievalResult.is_noise.
   They are NOT excluded at index-time so the graph agent can still
   resolve their node_ids if it needs to.

4. Both paths supported:
     a) JSON file  → .build_from_json(path)
     b) Live list  → .build_from_regions(semantic_regions)   ← no JSON round-trip
        where semantic_regions is list[SemanticRegion] from SemanticUnderstandingAgent

5. Persistence:  .save(dir)  /  .load(dir)  — stores index + metadata as two
   files (faiss_index.bin + metadata.json).  No pickle.

6. Optional role filter on search:  .search(query, top_k, role_filter=["Result","Method"])
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

MODEL_NAME          = "all-MiniLM-L6-v2"   # must match SemanticUnderstandingAgent
EMBEDDING_DIM       = 384                   # all-MiniLM-L6-v2 output dimension
IVF_THRESHOLD       = 500                   # switch from Flat → IVF above this
IVF_NLIST           = 16                    # IVF cluster count  (√n rule of thumb)
IVF_NPROBE          = 4                     # clusters to search at query time
ENCODE_BATCH_SIZE   = 32
TEXT_SNIPPET_LIMIT  = 400                   # chars returned in RetrievalResult.text

# Region classes that never carry meaningful retrievable text
NOISE_CLASSES = {"header", "footer"}
NOISE_ROLES   = {"N/A"}

METADATA_FILE = "metadata.json"
INDEX_FILE    = "faiss_index.bin"


# ──────────────────────────────────────────────────────────────────────────────
# Result dataclass  (returned to the hybrid scorer)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class RetrievalResult:
    """
    Single FAISS hit returned by FAISSAgent.search().

    Fields used by the HybridRetrievalEngine
    ─────────────────────────────────────────
    node_id       — graph node key for KG expansion
    text          — snippet passed to the LLM context window
    score         — cosine similarity in [0, 1]  (higher = more relevant)
    role          — scholarly role for result re-ranking
    page          — page index for provenance
    section_id    — nearest section title node_id (may be None)
    region_class  — paragraph / caption / table / …
    confidence    — layout detection confidence from Step 3
    is_noise      — True for headers / footers / N/A roles
    """
    node_id:      str
    text:         str
    score:        float
    role:         str
    page:         int
    section_id:   Optional[str]
    region_class: str
    confidence:   float
    is_noise:     bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ──────────────────────────────────────────────────────────────────────────────
# Per-node metadata  (stored alongside the FAISS index)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class NodeMeta:
    """
    Everything the retrieval layer needs to know about a node
    without touching the graph.  Stored as a JSON list parallel to the index.
    """
    node_id:      str
    text:         str           # truncated at TEXT_SNIPPET_LIMIT chars
    role:         str
    page:         int
    section_id:   Optional[str]
    region_class: str
    confidence:   float
    is_noise:     bool

    @classmethod
    def from_region_dict(cls, r: dict[str, Any]) -> "NodeMeta":
        rc   = r.get("region_class", "").lower()
        role = r.get("scholarly_role", "N/A")
        text = (r.get("text_content") or "")[:TEXT_SNIPPET_LIMIT]
        return cls(
            node_id      = r["region_id"],
            text         = text,
            role         = role,
            page         = r.get("page_index", 0),
            section_id   = r.get("section_id"),       # populated by KG agent if available
            region_class = rc,
            confidence   = r.get("confidence", 1.0),
            is_noise     = rc in NOISE_CLASSES or role in NOISE_ROLES,
        )


# ──────────────────────────────────────────────────────────────────────────────
# FAISS Agent
# ──────────────────────────────────────────────────────────────────────────────

class FAISSAgent:
    """
    Step 6 — FAISS Vector Database Agent.

    Usage
    ─────
        agent = FAISSAgent()

        # Build from JSON (most common path):
        agent.build_from_json("output/semantic.json")

        # Build from live SemanticRegion objects (no JSON overhead):
        agent.build_from_regions(semantic_regions)   # list[SemanticRegion]

        # Search:
        results = agent.search("KV cache recomputation method", top_k=5)
        results = agent.search("benchmark results", top_k=5, role_filter=["Result"])

        # Persist / reload:
        agent.save("output/faiss/")
        agent2 = FAISSAgent.load("output/faiss/")
    """

    def __init__(self, model_name: str = MODEL_NAME) -> None:
        logger.info("Loading encoder '%s' …", model_name)
        self._encoder: SentenceTransformer = SentenceTransformer(model_name, device="cpu")
        self._index:   Optional[faiss.Index] = None
        self._meta:    list[NodeMeta] = []       # parallel to FAISS internal ids
        self._built:   bool = False
        self._index_type: str = "none"

    # ──────────────────────────────────────────────────────────────────────────
    # BUILD
    # ──────────────────────────────────────────────────────────────────────────

    def build_from_json(self, semantic_json_path: str) -> None:
        """
        Load semantic.json produced by SemanticUnderstandingAgent,
        re-encode all text regions, and build the FAISS index.
        """
        path = Path(semantic_json_path)
        logger.info("Loading %s …", path)
        data = json.loads(path.read_text(encoding="utf-8"))
        regions: list[dict] = data.get("semantic_regions", [])
        if not regions:
            raise ValueError(f"No semantic_regions found in {path}")
        self._build(regions)

    def build_from_regions(self, semantic_regions: list[Any]) -> None:
        """
        Accept live SemanticRegion dataclass objects directly from
        SemanticUnderstandingAgent — avoids JSON serialise/deserialise.

        Expects objects with attributes:
            region_id, region_class, text_content, page_index,
            scholarly_role, confidence, role_confidence
        """
        region_dicts = [
            {
                "region_id":      r.region_id,
                "region_class":   r.region_class,
                "text_content":   r.text_content,
                "page_index":     r.page_index,
                "scholarly_role": r.scholarly_role,
                "confidence":     r.confidence,
            }
            for r in semantic_regions
        ]
        self._build(region_dicts)

    def _build(self, regions: list[dict]) -> None:
        """
        Core build pipeline:
          1. Extract text + build metadata list
          2. Encode with SBERT
          3. L2-normalise  (cosine sim = inner product on unit vectors)
          4. Construct FAISS index  (Flat for ≤ IVF_THRESHOLD, IVF above)
          5. Add vectors
        """
        t0 = time.perf_counter()

        # ── 1. Metadata ───────────────────────────────────────────────────────
        filtered_meta = []

        for r in regions:
            m = NodeMeta.from_region_dict(r)

            if m.is_noise:
                continue

            if not m.text.strip():
                continue

            if len(m.text.strip()) < 30:
                continue

            filtered_meta.append(m)

        self._meta = filtered_meta

        n = len(self._meta)
        logger.info("Building FAISS index over %d regions …", n)

        # ── 2. Encode ─────────────────────────────────────────────────────────
        texts = [
            m.text.strip() if m.text.strip() else "[EMPTY]"
            for m in self._meta
        ]
        embeddings: np.ndarray = self._encoder.encode(
            texts,
            batch_size        = ENCODE_BATCH_SIZE,
            show_progress_bar = n > 20,
            convert_to_numpy  = True,
            normalize_embeddings = True,   # ← L2-norm built into the encoder call
        ).astype(np.float32)

        # Defensive re-normalise in case the model didn't fully normalise
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        embeddings = embeddings / norms

        # ── 3. Validate dimensions ────────────────────────────────────────────
        dim = embeddings.shape[1]
        if dim != EMBEDDING_DIM:
            logger.warning(
                "Unexpected embedding dim %d (expected %d). "
                "Proceeding but downstream scores may be incorrect.",
                dim, EMBEDDING_DIM,
            )

        # ── 4. Build index ────────────────────────────────────────────────────
        if n <= IVF_THRESHOLD:
            # Exact search — no training needed, no approximation error
            self._index = faiss.IndexFlatIP(dim)
            self._index_type = "FlatIP"
            logger.info("Index type: IndexFlatIP (exact cosine, n=%d)", n)
        else:
            # Approximate search — faster at scale
            nlist = min(IVF_NLIST, n // 10) or 1   # guard against tiny splits
            quantiser = faiss.IndexFlatIP(dim)
            self._index = faiss.IndexIVFFlat(quantiser, dim, nlist, faiss.METRIC_INNER_PRODUCT)
            self._index.train(embeddings)
            self._index.nprobe = IVF_NPROBE
            self._index_type = f"IVFFlat(nlist={nlist}, nprobe={IVF_NPROBE})"
            logger.info("Index type: IndexIVFFlat (approximate, n=%d, nlist=%d)", n, nlist)

        # ── 5. Add vectors ────────────────────────────────────────────────────
        self._index.add(embeddings)
        self._built = True

        elapsed = time.perf_counter() - t0
        logger.info(
            "FAISS index built in %.2fs — %d vectors, dim=%d, type=%s",
            elapsed, self._index.ntotal, dim, self._index_type,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # SEARCH
    # ──────────────────────────────────────────────────────────────────────────

    def search(
        self,
        query:       str,
        top_k:       int = 5,
        role_filter: Optional[list[str]] = None,
        include_noise: bool = False,
    ) -> list[RetrievalResult]:
        """
        Encode query, search FAISS, return top-k RetrievalResult objects.

        Parameters
        ──────────
        query         : natural language question or keyword phrase
        top_k         : number of results to return
        role_filter   : if set, only return nodes whose scholarly_role
                        is in this list  e.g. ["Result", "Method"]
        include_noise : if False (default), headers/footers/N/A nodes
                        are excluded from results

        Returns
        ───────
        list[RetrievalResult] sorted by score descending, len ≤ top_k
        """
        self._check_built()

        # Encode + normalise query
        q_vec: np.ndarray = self._encoder.encode(
            [query],
            convert_to_numpy     = True,
            normalize_embeddings = True,
        ).astype(np.float32)

        # Over-fetch to allow for post-filtering without a second search
        fetch_k = top_k * 6 if (role_filter or not include_noise) else top_k

        scores, indices = self._index.search(q_vec, min(fetch_k, self._index.ntotal))

        results: list[RetrievalResult] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:                          # FAISS sentinel for "not enough results"
                continue
            meta = self._meta[idx]

            # ── Filters ───────────────────────────────────────────────────────
            if not include_noise and meta.is_noise:
                continue
            if role_filter and meta.role not in role_filter:
                continue

            results.append(RetrievalResult(
                node_id      = meta.node_id,
                text         = meta.text,
                score        = round(float(score), 4),   # cosine sim ∈ [0, 1]
                role         = meta.role,
                page         = meta.page,
                section_id   = meta.section_id,
                region_class = meta.region_class,
                confidence   = meta.confidence,
                is_noise     = meta.is_noise,
            ))

            if len(results) >= top_k:
                break

        logger.debug(
            "search('%s', top_k=%d) → %d results (role_filter=%s)",
            query[:60], top_k, len(results), role_filter,
        )
        return results

    def search_by_vector(
        self,
        vector:      np.ndarray,
        top_k:       int = 5,
        role_filter: Optional[list[str]] = None,
        include_noise: bool = False,
    ) -> list[RetrievalResult]:
        """
        Search using a pre-computed embedding vector.
        Used by the HybridRetrievalEngine to avoid re-encoding the same query
        when it needs multiple FAISS search rounds.
        """
        self._check_built()

        vec = vector.astype(np.float32).reshape(1, -1)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm

        fetch_k = top_k * 6 if (role_filter or not include_noise) else top_k
        scores, indices = self._index.search(vec, min(fetch_k, self._index.ntotal))

        results: list[RetrievalResult] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            meta = self._meta[idx]
            if not include_noise and meta.is_noise:
                continue
            if role_filter and meta.role not in role_filter:
                continue
            results.append(RetrievalResult(
                node_id      = meta.node_id,
                text         = meta.text,
                score        = round(float(score), 4),
                role         = meta.role,
                page         = meta.page,
                section_id   = meta.section_id,
                region_class = meta.region_class,
                confidence   = meta.confidence,
                is_noise     = meta.is_noise,
            ))
            if len(results) >= top_k:
                break

        return results

    def encode_query(self, query: str) -> np.ndarray:
        """
        Return L2-normalised query embedding.
        Exposed so the HybridRetrievalEngine can reuse it for graph expansion.
        """
        vec = self._encoder.encode(
            [query],
            convert_to_numpy     = True,
            normalize_embeddings = True,
        ).astype(np.float32)
        return vec[0]

    # ──────────────────────────────────────────────────────────────────────────
    # PERSISTENCE  (no pickle)
    # ──────────────────────────────────────────────────────────────────────────

    def save(self, directory: str) -> None:
        """
        Persist the FAISS index and metadata to `directory`.
        Creates two files:
            faiss_index.bin  — binary FAISS index
            metadata.json    — list of NodeMeta dicts (parallel to index)
        """
        self._check_built()
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self._index, str(out / INDEX_FILE))
        logger.info("FAISS index written to %s", out / INDEX_FILE)

        meta_dicts = [asdict(m) for m in self._meta]
        (out / METADATA_FILE).write_text(
            json.dumps(meta_dicts, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info("Metadata written to %s (%d entries)", out / METADATA_FILE, len(meta_dicts))

    @classmethod
    def load(cls, directory: str, model_name: str = MODEL_NAME) -> "FAISSAgent":
        """
        Reload a persisted FAISSAgent from `directory`.
        """
        out = Path(directory)
        index_path = out / INDEX_FILE
        meta_path  = out / METADATA_FILE

        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index not found at {index_path}")
        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata not found at {meta_path}")

        agent = cls(model_name=model_name)
        agent._index = faiss.read_index(str(index_path))

        meta_dicts = json.loads(meta_path.read_text(encoding="utf-8"))
        agent._meta = [
            NodeMeta(
                node_id      = m["node_id"],
                text         = m["text"],
                role         = m["role"],
                page         = m["page"],
                section_id   = m.get("section_id"),
                region_class = m["region_class"],
                confidence   = m["confidence"],
                is_noise     = m["is_noise"],
            )
            for m in meta_dicts
        ]
        agent._built = True
        logger.info(
            "FAISSAgent loaded from %s (%d vectors)",
            directory, agent._index.ntotal,
        )
        return agent

    # ──────────────────────────────────────────────────────────────────────────
    # DIAGNOSTICS
    # ──────────────────────────────────────────────────────────────────────────

    def print_summary(self) -> None:
        self._check_built()
        n = self._index.ntotal
        noise  = sum(1 for m in self._meta if m.is_noise)
        useful = n - noise

        role_counts: dict[str, int] = {}
        for m in self._meta:
            role_counts[m.role] = role_counts.get(m.role, 0) + 1

        print("\n🗂  FAISS AGENT SUMMARY")
        print("=" * 48)
        print(f"  Index type     : {self._index_type}")
        print(f"  Total vectors  : {n}")
        print(f"  Useful nodes   : {useful}  (non-noise)")
        print(f"  Noise nodes    : {noise}   (header/footer/N/A)")
        print(f"  Embedding dim  : {self._index.d}")
        print("\n  Scholarly role distribution:")
        for role, count in sorted(role_counts.items(), key=lambda x: -x[1]):
            print(f"    {role}: {count}")
        print("=" * 48)

    def validate(self) -> list[str]:
        """
        Return a list of warning strings.
        Called before the HybridRetrievalEngine starts.
        """
        warnings: list[str] = []
        if not self._built:
            warnings.append("Index has not been built yet.")
            return warnings
        if self._index.ntotal == 0:
            warnings.append("FAISS index is empty.")
        if len(self._meta) != self._index.ntotal:
            warnings.append(
                f"Metadata length ({len(self._meta)}) != "
                f"index size ({self._index.ntotal}). Index may be corrupt."
            )
        useful = sum(1 for m in self._meta if not m.is_noise)
        if useful == 0:
            warnings.append("All indexed nodes are noise (header/footer/N/A). "
                            "Search results will always be empty with include_noise=False.")
        return warnings

    # ──────────────────────────────────────────────────────────────────────────
    # INTERNAL HELPERS
    # ──────────────────────────────────────────────────────────────────────────

    def _check_built(self) -> None:
        if not self._built or self._index is None:
            raise RuntimeError(
                "FAISSAgent index is not built. "
                "Call build_from_json() or build_from_regions() first."
            )

    # ──────────────────────────────────────────────────────────────────────────
    # PROPERTIES  (read-only, used by HybridRetrievalEngine)
    # ──────────────────────────────────────────────────────────────────────────

    @property
    def total_vectors(self) -> int:
        return self._index.ntotal if self._built else 0

    @property
    def node_ids(self) -> list[str]:
        """All node_ids in index order. Used by the graph agent for alignment."""
        return [m.node_id for m in self._meta]

    @property
    def index_type(self) -> str:
        return self._index_type