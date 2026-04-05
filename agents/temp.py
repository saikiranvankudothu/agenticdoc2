# agents/hybrid_retrieval_engine.py
"""
Step 7 — Hybrid Retrieval Engine
==================================
Combines FAISSAgent (semantic vector search) with KnowledgeGraphAgent
(structured relationship traversal) into a single query interface.

Architecture
────────────
                        query (str)
                            │
               ┌────────────▼────────────┐
               │   QueryIntentClassifier  │
               │   detect intent:         │
               │   method / dataset /     │
               │   result / definition /  │
               │   observation / figure / │
               │   general                │
               └────────────┬────────────┘
                            │
              ┌─────────────▼─────────────┐
              │      FAISSAgent.search()   │
              │      top-k seed nodes      │
              │   + Observation rescue     │  ← v3: second pass for obs nodes
              └─────────────┬─────────────┘
                            │  seed node_ids
              ┌─────────────▼─────────────┐
              │   GraphExpander            │
              │   intent → relations[]     │
              │   BFS 1-hop expansion      │
              │   RELATION_WEIGHTS filter  │
              │   MAX_KG_EXPANSION cap     │
              └─────────────┬─────────────┘
                            │
              ┌─────────────▼─────────────┐
              │   HybridScorer             │
              │   final = α·faiss_score    │
              │         + (1-α)·graph_score│
              │   rank-decayed seed bonus  │  ← v3: replaces flat SEED_BONUS
              │   RELATION_WEIGHTS         │
              │   MIN_KG_SCORE filter      │
              │   dedup + re-rank          │
              └─────────────┬─────────────┘
                            │
               ┌────────────▼────────────┐
               │     HybridResult        │
               │  ranked_results         │  ← flat list for LLM context window
               │  subgraph               │  ← nx.DiGraph for structured reasoning
               │  provenance             │  ← full audit trail
               └─────────────────────────┘

Scoring (v3)
────────────
The core formula is unchanged:
    base = α · faiss_score + (1 − α) · graph_score

Seed bonus is now rank-decayed instead of flat:
    bonus(rank) = SEED_BONUS_BASE × SEED_BONUS_DECAY ** rank
    e.g. rank-0: +0.15, rank-1: +0.09, rank-2: +0.054, rank-3: +0.032, rank-4: +0.019

    final_score (seed)      = base + bonus(rank)
    final_score (expansion) = base + 0.0

Why rank-decay matters vs flat bonus (v2):
    Flat bonus (v2) protected ALL seeds equally. A weak rank-4 seed with
    faiss_score=0.45 scored 0.45×0.7+0.20=0.515, blocking any KG hit from
    entering the top-5 regardless of quality. This made K=5 metrics identical
    to pure FAISS — the KG added nothing.

    Rank-decayed bonus: the same rank-4 seed scores 0.45×0.7+0.019=0.334.
    A KG neighbour with faiss_score=0.60 and graph_score=0.80 (strong edge)
    scores 0.60×0.7+0.80×0.3=0.66 → correctly promoted above the weak seed.
    Meanwhile rank-0 seeds with faiss_score=0.85 score 0.85×0.7+0.15=0.745
    and remain safely at the top.

Changes from v2
───────────────
1. SEED_BONUS_BASE / SEED_BONUS_DECAY  (replaces flat SEED_BONUS=0.20)
   Rank-decayed bonus lets strong KG hits displace weak low-ranked seeds
   while still protecting top-ranked seeds from noise.

2. Observation rescue (step 2c in retrieve())
   observation_retrieval scored 0.000 in both v1 and v2 because FAISS never
   returned Observation-role nodes in its top-5 (embeddings for "findings /
   observations" queries don't cluster near Observation nodes reliably).
   Fix: after primary FAISS search, if zero Observation nodes appear in seeds,
   run a second role-filtered FAISS pass and append unique Observation hits.
   These extra seeds enter the merge with their own rank-decayed bonus so they
   compete fairly against the primary seeds.

3. OBSERVATION_RESCUE_K (default 3)
   How many Observation nodes to fetch in the rescue pass. Small enough not
   to flood the candidate pool; large enough to recover relevant nodes.

4. seed_bonus field added to ScoredNode — visible in print_result() per row.

5. obs_rescue_seeds field added to QueryProvenance for full audit trail.

6. retrieve_observation() convenience wrapper added.

All v2 changes are preserved:
    - MAX_KG_EXPANSION=4 hard cap
    - MIN_KG_SCORE=0.10 filter
    - RELATION_WEIGHTS (refers_to=0.40, etc.)
    - definition / observation intents
    - Role-based intent fallback (CONF_THRESHOLD=0.15)
    - DEFAULT_ALPHA=0.7
"""

from __future__ import annotations

import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Optional

import networkx as nx

from agents.faiss_agent import FAISSAgent, RetrievalResult
from agents.knowledge_graph_agent import KnowledgeGraphAgent

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_ALPHA          = 0.7    # weight on FAISS score; (1-α) on graph score
DEFAULT_TOP_K          = 5      # seeds from FAISS primary search
DEFAULT_EXPAND_K       = 3      # graph neighbours per seed (soft; MAX_KG_EXPANSION is hard)
DEFAULT_FINAL_K        = 10     # results returned to caller
RRF_K                  = 60     # not used — kept for reference

# Rank-decayed seed bonus  (v3 — replaces flat SEED_BONUS=0.20 from v2)
SEED_BONUS_BASE        = 0.15   # bonus applied to rank-0 (top FAISS) seed
SEED_BONUS_DECAY       = 0.60   # multiplied per rank step
#   rank 0 → +0.1500
#   rank 1 → +0.0900
#   rank 2 → +0.0540
#   rank 3 → +0.0324
#   rank 4 → +0.0194
#   rank 5 → +0.0117  (rescue seeds start here — very small bonus)

# Observation rescue  (v3 — new)
OBSERVATION_RESCUE_K   = 3      # Observation-role nodes to fetch in rescue pass
OBS_RESCUE_ROLE        = "Observation"

# Graph expansion controls
MIN_KG_SCORE           = 0.10   # drop expansion nodes with weighted score below this
MAX_KG_EXPANSION       = 4      # hard cap on total KG expansion nodes per query

# Intent classifier fallback threshold
CONF_THRESHOLD         = 0.15   # use role-based fallback if classifier conf < this

# Per-relation score multipliers applied to edge_weight before merge
RELATION_WEIGHTS: dict[str, float] = {
    "produces":      1.0,
    "used_in":       1.0,
    "evaluated_on":  1.0,
    "evaluated_by":  1.0,
    "defines":       0.90,
    "co_section":    0.50,
    "refers_to":     0.40,   # downweighted — caption→figure edges are broad
    "contains":      0.30,   # section containment — weakest signal
}

# Map scholarly_role → query intent (fallback when classifier is weak)
ROLE_TO_INTENT: dict[str, str] = {
    "Method":      "method",
    "Result":      "result",
    "Definition":  "definition",
    "Dataset":     "dataset",
    "Observation": "observation",
}

# Intent → outgoing relations to follow during graph expansion
INTENT_RELATIONS: dict[str, list[str]] = {
    "method":      ["produces", "used_in", "refers_to"],
    "dataset":     ["evaluated_on", "evaluated_by"],
    "result":      ["refers_to"],
    "definition":  ["defines", "used_in", "co_section"],
    "observation": ["produces", "refers_to"],
    "figure":      ["refers_to"],
    "general":     ["produces", "refers_to", "evaluated_on"],
}

# Intent → incoming relations to follow (reverse edges)
INTENT_REVERSE_RELATIONS: dict[str, list[str]] = {
    "result":      ["produces", "evaluated_on"],
    "dataset":     ["evaluated_by"],
    "observation": ["produces"],
}

# Keyword patterns for lightweight intent detection
_INTENT_PATTERNS: list[tuple[str, list[str]]] = [
    ("method",      [r"\bmethod\b", r"\bapproach\b", r"\balgorithm\b", r"\btechnique\b",
                     r"\barchitecture\b", r"\bmodel\b", r"\bpipeline\b", r"\bframework\b",
                     r"\bhow (does|do|did|is)\b", r"\bpropose[ds]?\b", r"\btrain(ing)?\b"]),
    ("dataset",     [r"\bdataset\b", r"\bbenchmark\b", r"\bcorpus\b", r"\bevaluat\w+\b",
                     r"\bbaseline\b", r"\bexperiment\b", r"\btrain(ing)? (set|data)\b"]),
    ("result",      [r"\bresult\b", r"\bperformance\b", r"\baccuracy\b", r"\bscore\b",
                     r"\bimprove\w*\b", r"\boutperform\w*\b", r"\bf1\b", r"\bbleu\b",
                     r"\bprecision\b", r"\brecall\b", r"\bmetric\b"]),
    ("definition",  [r"\bdefin\w+\b", r"\bterm\b", r"\bnotion\b", r"\bconcept\b",
                     r"\bmeaning\b", r"\bwhat is\b", r"\bwhat are\b", r"\bformally\b",
                     r"\bdenot\w+\b", r"\brefer[s]? to\b"]),
    ("observation", [r"\bobservation\b", r"\bfinding[s]?\b", r"\bnotice[d]?\b",
                     r"\breport[s]?\b", r"\bshow[s]?\b", r"\bindicate[s]?\b",
                     r"\bsuggest[s]?\b", r"\bevidence\b", r"\binsight[s]?\b"]),
    ("figure",      [r"\bfigure\b", r"\bfig\b", r"\btable\b", r"\bplot\b", r"\bdiagram\b",
                     r"\bvisuali[sz]\w*\b", r"\billustrat\w*\b", r"\bshown? in\b"]),
]


# ──────────────────────────────────────────────────────────────────────────────
# Seed bonus helper
# ──────────────────────────────────────────────────────────────────────────────

def _seed_bonus(rank: int) -> float:
    """
    Rank-decayed additive bonus for FAISS seed nodes (0-indexed rank).

    Default values (SEED_BONUS_BASE=0.15, SEED_BONUS_DECAY=0.60):
        rank 0 → 0.1500   (top primary seed — strongly protected)
        rank 1 → 0.0900
        rank 2 → 0.0540
        rank 3 → 0.0324
        rank 4 → 0.0194   (5th primary seed — weakly protected, can be displaced)
        rank 5 → 0.0117   (rescue seeds begin here)
    """
    return round(SEED_BONUS_BASE * (SEED_BONUS_DECAY ** rank), 4)


# ──────────────────────────────────────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ScoredNode:
    """
    A single node in the final ranked list returned to the answer generator.
    Carries everything the LLM needs — no further graph or index lookups required.
    """
    node_id:       str
    text:          str
    final_score:   float          # α·faiss + (1-α)·graph + seed_bonus(rank)
    faiss_score:   float          # raw cosine similarity
    graph_score:   float          # weighted edge score (0 if faiss-only seed)
    seed_bonus:    float          # actual bonus added (0.0 for graph expansion nodes)
    role:          str
    page:          int
    section_id:    Optional[str]
    region_class:  str
    confidence:    float
    origin:        str            # "faiss_seed" | "obs_rescue" | "graph_expansion" | "both"
    relation_path: list[str]      # e.g. ["produces"] — empty for pure seeds

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id":       self.node_id,
            "text":          self.text,
            "final_score":   self.final_score,
            "faiss_score":   self.faiss_score,
            "graph_score":   self.graph_score,
            "seed_bonus":    self.seed_bonus,
            "role":          self.role,
            "page":          self.page,
            "section_id":    self.section_id,
            "region_class":  self.region_class,
            "confidence":    self.confidence,
            "origin":        self.origin,
            "relation_path": self.relation_path,
        }


@dataclass
class QueryProvenance:
    """Full audit trail for a single query."""
    query:            str
    intent:           str
    intent_conf:      float
    intent_source:    str        # "classifier" | "role_fallback" | "override"
    alpha:            float
    faiss_seeds:      list[str]  # primary FAISS seed node_ids
    obs_rescue_seeds: list[str]  # node_ids added by observation rescue pass
    expanded_nodes:   list[str]  # node_ids added by graph expansion
    relations_used:   list[str]  # relation types traversed
    total_candidates: int
    final_k:          int


@dataclass
class HybridResult:
    """The complete output of one HybridRetrievalEngine.retrieve() call."""
    ranked_results: list[ScoredNode]
    subgraph:       nx.DiGraph
    provenance:     QueryProvenance

    def top_texts(self, k: int = 5) -> list[str]:
        return [r.text for r in self.ranked_results[:k] if r.text.strip()]

    def top_nodes(self, k: int = 5) -> list[ScoredNode]:
        return self.ranked_results[:k]

    def nodes_by_role(self, role: str) -> list[ScoredNode]:
        return [r for r in self.ranked_results if r.role == role]

    def edge_explanations(self) -> list[str]:
        lines = []
        for src, tgt, data in self.subgraph.edges(data=True):
            rel  = data.get("relation", "?")
            s_rc = self.subgraph.nodes[src].get("type", "?")
            t_rc = self.subgraph.nodes[tgt].get("type", "?")
            lines.append(f"{s_rc}[{src[:8]}] --{rel}--> {t_rc}[{tgt[:8]}]")
        return lines


# ──────────────────────────────────────────────────────────────────────────────
# Query Intent Classifier
# ──────────────────────────────────────────────────────────────────────────────

class QueryIntentClassifier:
    """
    Lightweight regex-based intent detector.
    Returns (intent: str, confidence: float).

    Confidence = matched_patterns / total_patterns_for_winning_intent.
    Tie-break between equally-scoring intents → "general".
    """

    def classify(self, query: str) -> tuple[str, float]:
        q = query.lower()
        scores: dict[str, int] = defaultdict(int)

        for intent, patterns in _INTENT_PATTERNS:
            for pat in patterns:
                if re.search(pat, q):
                    scores[intent] += 1

        if not scores:
            return "general", 0.0

        best_intent = max(scores, key=lambda k: scores[k])

        # Confidence normalised per-intent (v2 fix — was dividing by wrong denominator)
        intent_pattern_count = next(
            len(pats) for name, pats in _INTENT_PATTERNS if name == best_intent
        )
        confidence = min(scores[best_intent] / max(intent_pattern_count, 1), 1.0)

        # Tie-break: two intents matched equally → fall back to "general"
        top_score   = scores[best_intent]
        top_intents = [k for k, v in scores.items() if v == top_score]
        if len(top_intents) > 1:
            return "general", round(confidence, 3)

        return best_intent, round(confidence, 3)


# ──────────────────────────────────────────────────────────────────────────────
# Role-based Intent Fallback
# ──────────────────────────────────────────────────────────────────────────────

def infer_intent_from_seeds(faiss_results: list[RetrievalResult]) -> Optional[str]:
    """
    Infer query intent from the majority scholarly_role of FAISS seed nodes.
    Used as fallback when the regex classifier fires with low confidence.

    E.g. if FAISS returns 3 Definition nodes → intent = "definition",
    bypassing a weak classifier that returned conf=0.00.
    """
    roles = [r.role for r in faiss_results if r.role and r.role != "N/A"]
    if not roles:
        return None
    majority_role = Counter(roles).most_common(1)[0][0]
    return ROLE_TO_INTENT.get(majority_role)


# ──────────────────────────────────────────────────────────────────────────────
# Graph Expander
# ──────────────────────────────────────────────────────────────────────────────

class GraphExpander:
    """
    Given seed node_ids and a query intent, walks the knowledge graph and
    returns (seed_id, neighbour_id, weighted_edge_score, relation) tuples.

    RELATION_WEIGHTS applied at expansion time.
    Total expansion capped at MAX_KG_EXPANSION globally.
    Nodes with weighted_score < MIN_KG_SCORE filtered before returning.
    """

    def __init__(self, graph: nx.DiGraph) -> None:
        self._graph = graph

    def expand(
        self,
        seed_ids: list[str],
        intent:   str,
        expand_k: int = DEFAULT_EXPAND_K,
    ) -> list[tuple[str, str, float, str]]:
        """
        Returns list of (seed_id, neighbour_id, weighted_score, relation).
        neighbour_id is never in seed_ids.
        Total results capped at MAX_KG_EXPANSION.
        """
        relations_out = INTENT_RELATIONS.get(intent, INTENT_RELATIONS["general"])
        relations_in  = INTENT_REVERSE_RELATIONS.get(intent, [])

        seed_set = set(seed_ids)
        results:  list[tuple[str, str, float, str]] = []

        for seed in seed_ids:
            if len(results) >= MAX_KG_EXPANSION:
                break
            if seed not in self._graph:
                continue

            added = 0

            # ── Outgoing edges ─────────────────────────────────────────────
            for _, nbr, data in self._graph.out_edges(seed, data=True):
                if added >= expand_k or len(results) >= MAX_KG_EXPANSION:
                    break
                rel = data.get("relation", "")
                if rel not in relations_out or nbr in seed_set:
                    continue
                raw_w    = float(data.get("weight", 0.5))
                weighted = round(raw_w * RELATION_WEIGHTS.get(rel, 0.5), 4)
                if weighted < MIN_KG_SCORE:
                    continue
                results.append((seed, nbr, weighted, rel))
                added += 1

            # ── Incoming edges (reverse traversal) ────────────────────────
            for src, _, data in self._graph.in_edges(seed, data=True):
                if added >= expand_k or len(results) >= MAX_KG_EXPANSION:
                    break
                rel = data.get("relation", "")
                if rel not in relations_in or src in seed_set:
                    continue
                raw_w    = float(data.get("weight", 0.5))
                weighted = round(raw_w * RELATION_WEIGHTS.get(rel, 0.5), 4)
                if weighted < MIN_KG_SCORE:
                    continue
                results.append((seed, src, weighted, f"←{rel}"))
                added += 1

        return results


# ──────────────────────────────────────────────────────────────────────────────
# Hybrid Scorer
# ──────────────────────────────────────────────────────────────────────────────

class HybridScorer:
    """
    Merges FAISS seeds, observation rescue seeds, and graph expansion hits
    into a single ranked list using rank-decayed seed bonuses.

    Per-node scoring
    ────────────────
    base  = α · faiss_score + (1 − α) · graph_score

    Seeds (origin = "faiss_seed" | "obs_rescue" | "both"):
        final = base + _seed_bonus(seed_rank)

    Pure graph expansion (origin = "graph_expansion"):
        final = base + 0.0

    seed_rank is the position index in the combined seeds list
    (primary FAISS seeds first, rescue seeds appended after).
    Rescue seeds start at rank = len(primary_seeds) so their bonus
    is very small (~0.01) — they compete fairly but don't crowd out
    confirmed primary seeds.
    """

    def __init__(self, alpha: float = DEFAULT_ALPHA) -> None:
        self.alpha = alpha

    def merge(
        self,
        faiss_results:  list[RetrievalResult],
        expansion_hits: list[tuple[str, str, float, str]],
        faiss_lookup:   dict[str, RetrievalResult],
        graph:          nx.DiGraph,
        seed_origins:   dict[str, str],
    ) -> list[ScoredNode]:
        """
        Parameters
        ──────────
        faiss_results  : primary FAISS seeds + observation rescue seeds (in rank order)
        expansion_hits : (seed_id, nbr_id, weighted_score, relation) from GraphExpander
        faiss_lookup   : node_id → RetrievalResult for O(1) score lookup
        graph          : full KG — provides node attributes for expansion-only nodes
        seed_origins   : node_id → "faiss_seed" | "obs_rescue"
        """

        acc: dict[str, dict[str, Any]] = {}

        # 1. Register seed nodes with their rank for bonus decay
        for rank, fr in enumerate(faiss_results):
            acc[fr.node_id] = {
                "faiss_score":   fr.score,
                "graph_score":   0.0,
                "seed_rank":     rank,
                "is_seed":       True,
                "origin":        seed_origins.get(fr.node_id, "faiss_seed"),
                "relation_path": [],
                "result":        fr,
            }

        # 2. Graph expansion neighbours
        for seed_id, nbr_id, weighted_score, relation in expansion_hits:
            nbr_faiss = faiss_lookup[nbr_id].score if nbr_id in faiss_lookup else 0.0

            if nbr_id in acc:
                # Already in acc as a seed — upgrade graph score, keep seed status
                existing = acc[nbr_id]
                existing["faiss_score"] = max(existing["faiss_score"], nbr_faiss)
                existing["graph_score"] = max(existing["graph_score"], weighted_score)
                if existing["origin"] in ("faiss_seed", "obs_rescue"):
                    existing["origin"] = "both"
                existing["relation_path"].append(relation)
            else:
                # New node arriving only via graph expansion
                g_attrs = graph.nodes.get(nbr_id, {})
                pseudo  = RetrievalResult(
                    node_id      = nbr_id,
                    text         = g_attrs.get("text", ""),
                    score        = nbr_faiss,
                    role         = g_attrs.get("role", "N/A"),
                    page         = g_attrs.get("page", -1),
                    section_id   = g_attrs.get("section_id"),
                    region_class = g_attrs.get("type", ""),
                    confidence   = g_attrs.get("confidence", 1.0),
                    is_noise     = g_attrs.get("is_noise", False),
                )
                acc[nbr_id] = {
                    "faiss_score":   nbr_faiss,
                    "graph_score":   weighted_score,
                    "seed_rank":     9999,   # not a seed — no rank bonus
                    "is_seed":       False,
                    "origin":        "graph_expansion",
                    "relation_path": [relation],
                    "result":        pseudo,
                }

        # 3. Compute final scores and build ScoredNode list
        scored: list[ScoredNode] = []
        for nid, entry in acc.items():
            fs      = entry["faiss_score"]
            gs      = entry["graph_score"]
            is_seed = entry["is_seed"]
            rank    = entry["seed_rank"]
            bonus   = _seed_bonus(rank) if is_seed else 0.0
            base    = self.alpha * fs + (1.0 - self.alpha) * gs
            final   = round(base + bonus, 4)

            r: RetrievalResult = entry["result"]
            scored.append(ScoredNode(
                node_id       = nid,
                text          = r.text,
                final_score   = final,
                faiss_score   = round(fs, 4),
                graph_score   = round(gs, 4),
                seed_bonus    = round(bonus, 4),
                role          = r.role,
                page          = r.page,
                section_id    = r.section_id,
                region_class  = r.region_class,
                confidence    = r.confidence,
                origin        = entry["origin"],
                relation_path = entry["relation_path"],
            ))

        scored.sort(key=lambda x: x.final_score, reverse=True)
        return scored


# ──────────────────────────────────────────────────────────────────────────────
# Subgraph Builder
# ──────────────────────────────────────────────────────────────────────────────

class SubgraphBuilder:
    @staticmethod
    def build(node_ids: list[str], graph: nx.DiGraph) -> nx.DiGraph:
        node_set = set(node_ids)
        sub = nx.DiGraph()
        for nid in node_set:
            if nid in graph:
                sub.add_node(nid, **dict(graph.nodes[nid]))
        for src, tgt, data in graph.edges(data=True):
            if src in node_set and tgt in node_set:
                sub.add_edge(src, tgt, **data)
        return sub


# ──────────────────────────────────────────────────────────────────────────────
# Hybrid Retrieval Engine  (public interface)
# ──────────────────────────────────────────────────────────────────────────────

class HybridRetrievalEngine:
    """
    Step 7 — Hybrid Retrieval Engine (v3).

    Usage
    ─────
        faiss_agent = FAISSAgent()
        faiss_agent.build_from_json("output/semantic.json")

        kg_agent = KnowledgeGraphAgent()
        kg_agent.build_from_json("output/semantic.json")

        engine = HybridRetrievalEngine(faiss_agent, kg_agent)

        result = engine.retrieve("How does KV recomputation work?")
        result = engine.retrieve("What benchmarks were used?", top_k=8)
        result = engine.retrieve("What did the paper observe?")
        result = engine.retrieve("What are the key definitions?")
    """

    def __init__(
        self,
        faiss_agent: FAISSAgent,
        kg_agent:    KnowledgeGraphAgent,
        alpha:       float = DEFAULT_ALPHA,
    ) -> None:
        self._faiss    = faiss_agent
        self._kg       = kg_agent
        self._alpha    = alpha
        self._intent   = QueryIntentClassifier()
        self._expander = GraphExpander(kg_agent.graph)
        self._scorer   = HybridScorer(alpha)

        self._validate_agents()

    # ──────────────────────────────────────────────────────────────────────────
    # PRIMARY API
    # ──────────────────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query:           str,
        top_k:           int             = DEFAULT_TOP_K,
        expand_k:        int             = DEFAULT_EXPAND_K,
        final_k:         int             = DEFAULT_FINAL_K,
        alpha:           Optional[float] = None,
        role_filter:     Optional[list[str]] = None,
        intent_override: Optional[str]   = None,
    ) -> HybridResult:
        """
        Full hybrid retrieval pipeline for a single query.

        Parameters
        ──────────
        query           : natural language question
        top_k           : FAISS seeds from primary search (default 5)
        expand_k        : graph neighbours per seed — soft; MAX_KG_EXPANSION is hard cap
        final_k         : nodes to return in ranked_results (default 10)
        alpha           : override instance-level alpha for this call
        role_filter     : restrict primary FAISS seeds to these scholarly roles
        intent_override : skip intent detection, force this intent string
        """
        _alpha = alpha if alpha is not None else self._alpha
        self._scorer.alpha = _alpha

        # ── 1. Intent classification ──────────────────────────────────────────
        intent_source = "classifier"
        if intent_override:
            intent, intent_conf = intent_override, 1.0
            intent_source = "override"
        else:
            intent, intent_conf = self._intent.classify(query)

        logger.info("Query: '%s'  →  intent=%s (conf=%.2f)", query[:80], intent, intent_conf)

        # ── 2. Primary FAISS search ───────────────────────────────────────────
        faiss_results: list[RetrievalResult] = self._faiss.search(
            query,
            top_k         = top_k,
            role_filter   = role_filter,
            include_noise = False,
        )

        if not faiss_results:
            logger.warning("FAISS returned no results for query: '%s'", query)
            return self._empty_result(query, intent, intent_conf, intent_source, _alpha)

        faiss_lookup: dict[str, RetrievalResult] = {r.node_id: r for r in faiss_results}
        seed_origins: dict[str, str]             = {r.node_id: "faiss_seed" for r in faiss_results}

        logger.info(
            "FAISS seeds: %d nodes  %s",
            len(faiss_results), [r.node_id[:8] for r in faiss_results],
        )

        # ── 2b. Role-based intent fallback ────────────────────────────────────
        # When regex classifier confidence is too low, infer intent from
        # majority scholarly_role among the retrieved seeds.
        if intent_conf < CONF_THRESHOLD and not intent_override:
            fallback = infer_intent_from_seeds(faiss_results)
            if fallback and fallback != intent:
                logger.info(
                    "Intent fallback: classifier='%s' (conf=%.2f) → role-inferred='%s'",
                    intent, intent_conf, fallback,
                )
                intent = fallback
                intent_source = "role_fallback"

        # ── 2c. Observation rescue ────────────────────────────────────────────
        # observation_retrieval scores 0.000 when FAISS top-k contains no
        # Observation-role nodes. Fix: run a second role-filtered FAISS pass,
        # append unique Observation hits to the seeds list.
        #
        # Rescue seeds get rank = len(primary_seeds) + i, so their rank-decayed
        # bonus is very small (≈0.01). They can still outrank weak primary seeds
        # if their faiss_score is significantly higher — fair competition.
        obs_rescue_ids: list[str] = []
        primary_roles = {r.role for r in faiss_results}

        if OBS_RESCUE_ROLE not in primary_roles:
            obs_hits: list[RetrievalResult] = self._faiss.search(
                query,
                top_k         = OBSERVATION_RESCUE_K,
                role_filter   = [OBS_RESCUE_ROLE],
                include_noise = False,
            )
            existing_ids = set(faiss_lookup.keys())
            for r in obs_hits:
                if r.node_id not in existing_ids:
                    faiss_results.append(r)
                    faiss_lookup[r.node_id]  = r
                    seed_origins[r.node_id]  = "obs_rescue"
                    obs_rescue_ids.append(r.node_id)

            if obs_rescue_ids:
                logger.info(
                    "Observation rescue: +%d nodes  %s",
                    len(obs_rescue_ids), [n[:8] for n in obs_rescue_ids],
                )

        seed_ids = [r.node_id for r in faiss_results]

        # ── 3. Graph expansion ────────────────────────────────────────────────
        expansion_hits = self._expander.expand(seed_ids, intent, expand_k)
        expanded_ids   = list({nbr for _, nbr, _, _ in expansion_hits})
        relations_used = list({rel for _, _, _, rel in expansion_hits})

        logger.info(
            "Graph expansion: %d neighbours via relations %s",
            len(expanded_ids), relations_used,
        )

        # ── 4. Hybrid scoring & merge ─────────────────────────────────────────
        scored_nodes = self._scorer.merge(
            faiss_results  = faiss_results,
            expansion_hits = expansion_hits,
            faiss_lookup   = faiss_lookup,
            graph          = self._kg.graph,
            seed_origins   = seed_origins,
        )

        # Soft role filter — preferred roles bubble to front, not removed
        if role_filter:
            scored_nodes = (
                [n for n in scored_nodes if n.role in role_filter]
                + [n for n in scored_nodes if n.role not in role_filter]
            )

        final_nodes = scored_nodes[:final_k]

        # ── 5. Build subgraph ─────────────────────────────────────────────────
        subgraph = SubgraphBuilder.build(
            [n.node_id for n in final_nodes], self._kg.graph
        )

        # ── 6. Provenance ─────────────────────────────────────────────────────
        primary_seed_ids = [
            r.node_id for r in faiss_results
            if seed_origins.get(r.node_id) == "faiss_seed"
        ]
        provenance = QueryProvenance(
            query            = query,
            intent           = intent,
            intent_conf      = intent_conf,
            intent_source    = intent_source,
            alpha            = _alpha,
            faiss_seeds      = primary_seed_ids,
            obs_rescue_seeds = obs_rescue_ids,
            expanded_nodes   = expanded_ids,
            relations_used   = relations_used,
            total_candidates = len(scored_nodes),
            final_k          = len(final_nodes),
        )

        logger.info(
            "Retrieval complete: %d final nodes (primary=%d, rescue=%d, expanded=%d)",
            len(final_nodes), len(primary_seed_ids),
            len(obs_rescue_ids), len(expanded_ids),
        )

        return HybridResult(
            ranked_results = final_nodes,
            subgraph       = subgraph,
            provenance     = provenance,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # CONVENIENCE WRAPPERS
    # ──────────────────────────────────────────────────────────────────────────

    def retrieve_method(self, query: str, **kwargs) -> HybridResult:
        """Force method intent — expands produces, used_in, refers_to."""
        return self.retrieve(query, intent_override="method", **kwargs)

    def retrieve_results(self, query: str, **kwargs) -> HybridResult:
        """Force result intent — reverse-traverses produces, evaluated_on."""
        return self.retrieve(query, intent_override="result", **kwargs)

    def retrieve_dataset(self, query: str, **kwargs) -> HybridResult:
        """Force dataset intent — expands evaluated_on, evaluated_by."""
        return self.retrieve(query, intent_override="dataset", **kwargs)

    def retrieve_figure(self, query: str, **kwargs) -> HybridResult:
        """Force figure intent — follows refers_to edges to captions/figures."""
        return self.retrieve(query, intent_override="figure", **kwargs)

    def retrieve_definition(self, query: str, **kwargs) -> HybridResult:
        """Force definition intent — follows defines, used_in, co_section."""
        return self.retrieve(query, intent_override="definition", **kwargs)

    def retrieve_observation(self, query: str, **kwargs) -> HybridResult:
        """Force observation intent — reverse-traverses produces + rescue pass."""
        return self.retrieve(query, intent_override="observation", **kwargs)

    # ──────────────────────────────────────────────────────────────────────────
    # DIAGNOSTICS
    # ──────────────────────────────────────────────────────────────────────────

    def print_result(self, result: HybridResult, show_graph: bool = True) -> None:
        """Pretty-print a HybridResult — shows rank-decayed bonus column per row."""
        p = result.provenance
        print(f"\n{'='*70}")
        print(f"  HYBRID RETRIEVAL RESULT (v3)")
        print(f"{'='*70}")
        print(f"  Query          : {p.query}")
        print(f"  Intent         : {p.intent}  (conf={p.intent_conf:.2f}, source={p.intent_source})")
        print(f"  Alpha          : {p.alpha}  →  α·FAISS + (1-α)·Graph")
        print(f"  Seed bonus     : base={SEED_BONUS_BASE}, decay={SEED_BONUS_DECAY} (rank-decayed)")
        print(f"  Primary seeds  : {len(p.faiss_seeds)}")
        print(f"  Rescue seeds   : {len(p.obs_rescue_seeds)}  {[n[:8] for n in p.obs_rescue_seeds]}")
        print(f"  KG expanded    : {len(p.expanded_nodes)} neighbours (hard cap={MAX_KG_EXPANSION})")
        print(f"  Relations used : {p.relations_used}")
        print(f"  Total cands    : {p.total_candidates}  →  returned: {p.final_k}")
        print(f"\n  Ranked Results:")
        print(f"  {'#':<3} {'node_id':<14} {'role':<12} {'pg':<4} "
              f"{'final':>7} {'faiss':>7} {'graph':>7} {'bonus':>7}  {'origin':<16}  text[:55]")
        print(f"  {'-'*112}")
        for i, n in enumerate(result.ranked_results):
            print(
                f"  {i+1:<3} {n.node_id[:12]:<14} {n.role:<12} {n.page:<4} "
                f"{n.final_score:>7.4f} {n.faiss_score:>7.4f} "
                f"{n.graph_score:>7.4f} {n.seed_bonus:>7.4f}  "
                f"{n.origin:<16}  {n.text[:55].replace(chr(10), ' ')}"
            )
        if show_graph and result.subgraph.number_of_edges() > 0:
            print(f"\n  Subgraph edges ({result.subgraph.number_of_edges()}):")
            for line in result.edge_explanations():
                print(f"    {line}")
        print(f"{'='*70}\n")

    def validate(self) -> list[str]:
        """Pre-flight checks called at construction time."""
        warnings: list[str] = []
        warnings += self._faiss.validate()
        warnings += self._kg.validate()
        if self._kg.graph.number_of_nodes() == 0:
            warnings.append("Knowledge graph is empty — expansion will return 0 neighbours.")
        if not (0.0 <= self._alpha <= 1.0):
            warnings.append(f"Alpha={self._alpha} is outside [0, 1].")
        return warnings

    # ──────────────────────────────────────────────────────────────────────────
    # INTERNAL
    # ──────────────────────────────────────────────────────────────────────────

    def _validate_agents(self) -> None:
        for w in self.validate():
            logger.warning("Validation: %s", w)

    def _empty_result(
        self,
        query:         str,
        intent:        str,
        intent_conf:   float,
        intent_source: str,
        alpha:         float,
    ) -> HybridResult:
        return HybridResult(
            ranked_results = [],
            subgraph       = nx.DiGraph(),
            provenance     = QueryProvenance(
                query=query, intent=intent, intent_conf=intent_conf,
                intent_source=intent_source, alpha=alpha,
                faiss_seeds=[], obs_rescue_seeds=[], expanded_nodes=[],
                relations_used=[], total_candidates=0, final_k=0,
            ),
        )