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
               │   result / definition /  │  ← definition now supported
               │   figure / general       │
               └────────────┬────────────┘
                            │
              ┌─────────────▼─────────────┐
              │      FAISSAgent.search()   │
              │      top-k seed nodes      │
              └─────────────┬─────────────┘
                            │  seed node_ids
              ┌─────────────▼─────────────┐
              │   GraphExpander            │
              │   intent → relations[]     │
              │   BFS 1-hop expansion      │
              │   graph_score per neighbour│
              │   MAX_KG_EXPANSION cap     │  ← new: absolute cap on expansion
              └─────────────┬─────────────┘
                            │
              ┌─────────────▼─────────────┐
              │   HybridScorer             │
              │   final = α·faiss_score    │
              │         + (1-α)·graph_score│
              │   SEED_BONUS for seeds     │  ← new: seeds can't be displaced
              │   RELATION_WEIGHTS         │  ← new: refers_to downweighted
              │   MIN_KG_SCORE filter      │  ← new: noisy expansions pruned
              │   dedup + re-rank          │
              └─────────────┬─────────────┘
                            │
               ┌────────────▼────────────┐
               │     HybridResult        │
               │  ┌──────────────────┐   │
               │  │  ranked_results  │   │  ← flat list for LLM context window
               │  │ list[ScoredNode] │   │
               │  └──────────────────┘   │
               │  ┌──────────────────┐   │
               │  │    subgraph      │   │  ← nx.DiGraph for structured reasoning
               │  │  nx.DiGraph      │   │
               │  └──────────────────┘   │
               │  ┌──────────────────┐   │
               │  │    provenance    │   │  ← explains every score
               │  └──────────────────┘   │
               └─────────────────────────┘

Scoring (v2)
────────────
FAISS seeds  : faiss_score  = cosine similarity ∈ [0, 1]
               graph_score  = 0.0 (no edge traversed)
               seed_bonus   = SEED_BONUS (0.20) added before final sort
               final_score  = α · faiss_score + (1-α) · graph_score + seed_bonus

Graph neighbours: faiss_score = cosine(query_vec, neighbour_vec) if in FAISS else 0
                  graph_score  = edge_weight × RELATION_WEIGHT[relation]
                  final_score  = α · faiss_score + (1-α) · graph_score
                  (NO seed_bonus — expansion nodes compete fairly)

Changes from v1
───────────────
- SEED_BONUS (0.20): FAISS seeds receive a score additive so that low-weight
  graph expansions cannot displace top-ranked seeds. This fixed method/result
  nDCG collapsing to 0 when KG neighbours ranked higher than correct seeds.

- MAX_KG_EXPANSION (4): Hard cap on total graph expansion nodes per query,
  replacing the per-seed `expand_k` expansion which was producing 7-10
  expansions for only 42 indexed nodes (massive inflation ratio).

- MIN_KG_SCORE (0.10): Graph neighbours with edge_weight below this threshold
  are dropped before merge, removing noise edges from scoring.

- RELATION_WEIGHTS: `refers_to` downweighted to 0.40 (too promiscuous),
  `co_section` at 0.50, semantic relations at 1.0. Fixes dataset nDCG drop.

- "definition" intent added: uses `defines`, `used_in`, `co_section` edges.
  Fixes definition_retrieval nDCG = 0.000 in Hybrid.

- Role-based intent fallback: if regex classifier confidence < CONF_THRESHOLD,
  infer intent from majority scholarly_role of FAISS seeds. This bypasses the
  weak text-pattern classifier for queries like "What are the key definitions?"
  which fired conf=0.00 in v1.

Alpha (α) default = 0.7  → raised from 0.6 to further favour semantic score,
since the FAISS-only baseline was already strong.
"""

from __future__ import annotations

import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
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

DEFAULT_ALPHA     = 0.60     # raised from 0.6 — FAISS baseline was already strong
DEFAULT_TOP_K     = 5       # seeds from FAISS
DEFAULT_EXPAND_K  = 3       # graph neighbours per seed (soft limit; MAX_KG_EXPANSION is hard)
DEFAULT_FINAL_K   = 10      # results returned to caller
RRF_K             = 60      # not used (weighted sum chosen) — kept for reference

# v2 additions
SEED_BONUS        = 0.08    # additive bonus applied to all FAISS seed final_scores
                            # prevents graph neighbours displacing correct seeds
MIN_KG_SCORE      = 0.10    # drop expansion nodes whose edge_weight < this
MAX_KG_EXPANSION  = 4       # hard cap on total KG expansion nodes per query
                            # (not per-seed — controls inflation on small graphs)
CONF_THRESHOLD    = 0.15    # if classifier confidence < this, use role-based fallback

# Per-relation score multipliers applied to edge_weight before merge.
# Downweighting `refers_to` fixes the dataset query precision drop (v1 issue).
RELATION_WEIGHTS: dict[str, float] = {
    "produces":      1.0,
    "used_in":       1.0,
    "evaluated_on":  1.0,
    "evaluated_by":  1.0,
    "defines":       0.90,   # new in KG v2
    "co_section":    0.50,   # new in KG v2 — lateral definition links
    "refers_to":     0.40,   # downweighted — caption→figure edges are broad
    "contains":      0.30,   # section containment — weakest signal
}

# Map scholarly_role → query intent (used as fallback when classifier is weak)
ROLE_TO_INTENT: dict[str, str] = {
    "Method":      "method",
    "Result":      "result",
    "Definition":  "definition",
    "Dataset":     "dataset",
    "Observation": "observation",
}

# Intent → outgoing relations to follow during graph expansion
INTENT_RELATIONS: dict[str, list[str]] = {
    "method":     ["produces", "used_in", "refers_to"],
    "dataset":    ["evaluated_on", "evaluated_by"],        # removed refers_to — too broad
    "result":     ["refers_to"],
    "definition": ["defines", "used_in", "co_section"],    # NEW intent
    "observation":["produces", "refers_to"],               # NEW intent
    "figure":     ["refers_to"],
    "general":    ["produces", "refers_to", "evaluated_on"],
}

# Intent → incoming relations to follow (reverse edges)
INTENT_REVERSE_RELATIONS: dict[str, list[str]] = {
    "result":      ["produces", "evaluated_on"],
    "dataset":     ["evaluated_by"],
    "observation": ["produces"],
}

# Keyword patterns for lightweight intent detection
_INTENT_PATTERNS: list[tuple[str, list[str]]] = [
    ("method",     [r"\bmethod\b", r"\bapproach\b", r"\balgorithm\b", r"\btechnique\b",
                    r"\barchitecture\b", r"\bmodel\b", r"\bpipeline\b", r"\bframework\b",
                    r"\bhow (does|do|did|is)\b", r"\bpropose[ds]?\b", r"\btrain(ing)?\b"]),
    ("dataset", [
        r"\bdataset\b",
        r"\bbenchmark\b",   
        r"\bcorpus\b",
        r"\bused (for|in)\b",
        r"\bevaluat\w+\b",
        r"\bbaseline\b",
        r"\bexperiment\b",
        r"\btrain(ing)? (set|data)\b"
    ]),
    ("result",     [r"\bresult\b", r"\bperformance\b", r"\baccuracy\b", r"\bscore\b",
                    r"\bimprove\w*\b", r"\boutperform\w*\b", r"\bf1\b", r"\bbleu\b",
                    r"\bprecision\b", r"\brecall\b", r"\bmetric\b"]),
    ("definition", [r"\bdefin\w+\b", r"\bterm\b", r"\bnotion\b", r"\bconcept\b",
                    r"\bmeaning\b", r"\bwhat is\b", r"\bwhat are\b", r"\bformally\b",
                    r"\bdenot\w+\b", r"\brefer[s]? to\b"]),
    ("figure",     [r"\bfigure\b", r"\bfig\b", r"\btable\b", r"\bplot\b", r"\bdiagram\b",
                    r"\bvisuali[sz]\w*\b", r"\billustrat\w*\b", r"\bshown? in\b"]),
    ("observation", [r"\bobservation\b",r"\bfinding[s]?\b",r"\breport[s]?\b",
                    r"\bdiscover\w*\b",r"\bobserv\w+\b",r"\bwhat (does|do|did).{0,20}(report|show|find|note)\b"]),
]


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
    final_score:   float          # α·faiss + (1-α)·graph [+ seed_bonus for seeds]
    faiss_score:   float          # raw cosine similarity (0 if graph-only)
    graph_score:   float          # edge weight (0 if faiss-only seed)
    role:          str
    page:          int
    section_id:    Optional[str]
    region_class:  str
    confidence:    float
    origin:        str            # "faiss_seed" | "graph_expansion" | "both"
    relation_path: list[str]      # e.g. ["produces"] — empty for seeds

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id":      self.node_id,
            "text":         self.text,
            "final_score":  self.final_score,
            "faiss_score":  self.faiss_score,
            "graph_score":  self.graph_score,
            "role":         self.role,
            "page":         self.page,
            "section_id":   self.section_id,
            "region_class": self.region_class,
            "confidence":   self.confidence,
            "origin":       self.origin,
            "relation_path": self.relation_path,
        }


@dataclass
class QueryProvenance:
    """
    Full audit trail for a single query — useful for debugging
    and for building explainable citations in the answer.
    """
    query:          str
    intent:         str
    intent_conf:    float                    # fraction of matched patterns
    intent_source:  str                      # "classifier" | "role_fallback"
    alpha:          float
    seed_bonus:     float
    faiss_seeds:    list[str]                # node_ids
    expanded_nodes: list[str]                # node_ids added by graph
    relations_used: list[str]                # which relation types were followed
    total_candidates: int
    final_k:        int


@dataclass
class HybridResult:
    """
    The complete output of one HybridRetrievalEngine.retrieve() call.
    """
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

    v2: Added "definition" and "observation" intents.
    Confidence normalisation fixed — was dividing by len(first_pattern_list)
    instead of total matched / total checked, giving misleading low numbers.
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

        # Confidence = matched patterns / total patterns for this intent
        intent_pattern_count = next(
            len(pats) for name, pats in _INTENT_PATTERNS if name == best_intent
        )
        confidence = min(scores[best_intent] / max(intent_pattern_count, 1), 1.0)

        # Tie-break: if two intents match equally → "general"
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

    Example: FAISS returns 3 Definition nodes + 2 N/A nodes
             → majority role = "Definition" → intent = "definition"

    This is more reliable than regex for queries like "What are the key
    definitions?" which pattern-match weakly (no strong definition keywords).
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
    Given a set of seed node_ids and a query intent, walks the knowledge graph
    and returns (seed_id, neighbour_node_id, weighted_edge_score, relation) tuples.

    v2 changes:
    - MAX_KG_EXPANSION: total expansion capped globally (not just per-seed).
      With 42 nodes, expanding 7-10 neighbours per query was causing massive
      score inflation that buried correct seeds.
    - RELATION_WEIGHTS applied at expansion time so the scorer sees pre-weighted
      graph scores directly.
    - MIN_KG_SCORE filter: drops expansion nodes below threshold before returning.
    """

    def __init__(self, graph: nx.DiGraph) -> None:
        self._graph = graph

    def expand(
        self,
        seed_ids:  list[str],
        intent:    str,
        expand_k:  int = DEFAULT_EXPAND_K,
    ) -> list[tuple[str, str, float, str]]:
        """
        Returns list of (seed_id, neighbour_id, weighted_edge_score, relation).
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
                if rel not in relations_out:
                    continue
                if nbr in seed_set:
                    continue
                raw_weight = float(data.get("weight", 0.5))
                rel_weight = RELATION_WEIGHTS.get(rel, 0.5)
                weighted   = round(raw_weight * rel_weight, 4)
                if weighted < MIN_KG_SCORE:
                    continue
                results.append((seed, nbr, weighted, rel))
                added += 1

            # ── Incoming edges (reverse traversal for result/dataset) ───────
            for src, _, data in self._graph.in_edges(seed, data=True):
                if added >= expand_k or len(results) >= MAX_KG_EXPANSION:
                    break
                rel = data.get("relation", "")
                if rel not in relations_in:
                    continue
                if src in seed_set:
                    continue
                raw_weight = float(data.get("weight", 0.5))
                rel_weight = RELATION_WEIGHTS.get(rel, 0.5)
                weighted   = round(raw_weight * rel_weight, 4)
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
    Merges FAISS results and graph expansion hits into a single ranked list.

    Scoring (v2)
    ────────────
    Seeds (origin=faiss_seed):
        base   = α · faiss_score + (1-α) · 0.0
        final  = base + SEED_BONUS

    Graph neighbours (origin=graph_expansion):
        final  = α · faiss_score + (1-α) · graph_score
        (no seed_bonus — they must earn their rank)

    Nodes appearing in both (origin=both):
        final  = α · faiss_score + (1-α) · graph_score + SEED_BONUS

    The SEED_BONUS is the key change: it guarantees that a correct FAISS seed
    with faiss_score=0.70 cannot be displaced by a graph neighbour with
    faiss_score=0.0 and graph_score=0.5, which would score 0.30 at α=0.6.
    With SEED_BONUS=0.20 the seed scores 0.70*0.6+0.20=0.62 vs 0.20 for the
    neighbour — seed stays in the top 5.
    """

    def __init__(self, alpha: float = DEFAULT_ALPHA) -> None:
        self.alpha = alpha

    def merge(
        self,
        faiss_results:      list[RetrievalResult],
        expansion_hits:     list[tuple[str, str, float, str]],
        faiss_lookup:       dict[str, RetrievalResult],
        graph:              nx.DiGraph,
        query_vec_scorer:   Optional[Any] = None,
    ) -> list[ScoredNode]:

        # ── Accumulator: node_id → best scores seen so far ─────────────────
        acc: dict[str, dict[str, Any]] = {}

        # 1. Seed nodes from FAISS
        for fr in faiss_results:
            acc[fr.node_id] = {
                "faiss_score":    fr.score,
                "graph_score":    0.0,
                "is_seed":        True,
                "origin":         "faiss_seed",
                "relation_path":  [],
                "result":         fr,
            }

        # 2. Graph expansion neighbours
        for seed_id, nbr_id, weighted_edge_score, relation in expansion_hits:
            nbr_faiss_score = faiss_lookup[nbr_id].score if nbr_id in faiss_lookup else 0.0

            if nbr_id in acc:
                existing = acc[nbr_id]
                existing["faiss_score"] = max(existing["faiss_score"], nbr_faiss_score)
                existing["graph_score"] = max(existing["graph_score"], weighted_edge_score)
                # Keep is_seed=True if it was already a seed
                if existing["origin"] == "faiss_seed":
                    existing["origin"] = "both"
                existing["relation_path"].append(relation)
            else:
                g_attrs = graph.nodes.get(nbr_id, {})
                pseudo_result = RetrievalResult(
                    node_id      = nbr_id,
                    text         = g_attrs.get("text", ""),
                    score        = nbr_faiss_score,
                    role         = g_attrs.get("role", "N/A"),
                    page         = g_attrs.get("page", -1),
                    section_id   = g_attrs.get("section_id"),
                    region_class = g_attrs.get("type", ""),
                    confidence   = g_attrs.get("confidence", 1.0),
                    is_noise     = g_attrs.get("is_noise", False),
                )
                acc[nbr_id] = {
                    "faiss_score":   nbr_faiss_score,
                    "graph_score":   weighted_edge_score,
                    "is_seed":       False,
                    "origin":        "graph_expansion",
                    "relation_path": [relation],
                    "result":        pseudo_result,
                }

        # 3. Compute final scores and build ScoredNode list
        scored: list[ScoredNode] = []
        for nid, entry in acc.items():
            fs      = entry["faiss_score"]
            gs      = entry["graph_score"]
            is_seed = entry["is_seed"] or entry["origin"] in ("faiss_seed", "both")
            base    = round(self.alpha * fs + (1.0 - self.alpha) * gs, 4)
            final   = round(base + (SEED_BONUS if is_seed else 0.0), 4)

            r: RetrievalResult = entry["result"]
            scored.append(ScoredNode(
                node_id       = nid,
                text          = r.text,
                final_score   = final,
                faiss_score   = round(fs, 4),
                graph_score   = round(gs, 4),
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
    Step 7 — Hybrid Retrieval Engine (v2).

    Usage
    ─────
        faiss_agent = FAISSAgent()
        faiss_agent.build_from_json("output/semantic.json")

        kg_agent = KnowledgeGraphAgent()
        kg_agent.build_from_json("output/semantic.json")

        engine = HybridRetrievalEngine(faiss_agent, kg_agent)

        result = engine.retrieve("How does KV recomputation work?")
        result = engine.retrieve("What benchmarks were used?", top_k=8)
        result = engine.retrieve("Show accuracy results", role_filter=["Result"])
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
        query:        str,
        top_k:        int = DEFAULT_TOP_K,
        expand_k:     int = DEFAULT_EXPAND_K,
        final_k:      int = DEFAULT_FINAL_K,
        alpha:        Optional[float] = None,
        role_filter:  Optional[list[str]] = None,
        intent_override: Optional[str] = None,
    ) -> HybridResult:
        """
        Full hybrid retrieval pipeline for a single query.

        Parameters
        ──────────
        query           : natural language question
        top_k           : FAISS seeds to retrieve  (default 5)
        expand_k        : graph neighbours per seed (soft; MAX_KG_EXPANSION is hard cap)
        final_k         : nodes to return in ranked_results  (default 10)
        alpha           : override instance-level alpha for this call
        role_filter     : restrict FAISS seeds to these scholarly roles
        intent_override : skip intent detection, force this intent
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

        # ── 2. FAISS semantic search ──────────────────────────────────────────
        faiss_results: list[RetrievalResult] = self._faiss.search(
            query,
            top_k         = top_k,
            role_filter   = role_filter,
            include_noise = False,
        )

        if not faiss_results:
            logger.warning("FAISS returned no results for query: '%s'", query)
            return self._empty_result(query, intent, intent_conf, intent_source, _alpha)

        seed_ids    = [r.node_id for r in faiss_results]
        faiss_lookup = {r.node_id: r for r in faiss_results}

        logger.info("FAISS seeds: %d nodes  %s", len(seed_ids), [s[:8] for s in seed_ids])

        # ── 2b. Role-based intent fallback ────────────────────────────────────
        # If the regex classifier fired with low confidence, infer intent from
        # the majority scholarly_role of the retrieved seeds instead.
        if intent_conf < CONF_THRESHOLD and not intent_override:
            fallback = infer_intent_from_seeds(faiss_results)
            if fallback and fallback != intent:
                logger.info(
                    "Intent override: classifier='%s' (conf=%.2f) → role-fallback='%s'",
                    intent, intent_conf, fallback,
                )
                intent = fallback
                intent_source = "role_fallback"

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
        )

        # Apply role filter to final list if specified (soft — filtered roles
        # bubble to the end rather than being removed entirely)
        if role_filter:
            scored_nodes = (
                [n for n in scored_nodes if n.role in role_filter]
                + [n for n in scored_nodes if n.role not in role_filter]
            )

        final_nodes = scored_nodes[:final_k]

        # ── 5. Build subgraph ─────────────────────────────────────────────────
        all_node_ids = [n.node_id for n in final_nodes]
        subgraph     = SubgraphBuilder.build(all_node_ids, self._kg.graph)

        # ── 6. Provenance ─────────────────────────────────────────────────────
        provenance = QueryProvenance(
            query           = query,
            intent          = intent,
            intent_conf     = intent_conf,
            intent_source   = intent_source,
            alpha           = _alpha,
            seed_bonus      = SEED_BONUS,
            faiss_seeds     = seed_ids,
            expanded_nodes  = expanded_ids,
            relations_used  = relations_used,
            total_candidates= len(scored_nodes),
            final_k         = len(final_nodes),
        )

        logger.info(
            "Retrieval complete: %d final nodes (seeds=%d, expanded=%d)",
            len(final_nodes), len(seed_ids), len(expanded_ids),
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
        return self.retrieve(query, intent_override="method", **kwargs)

    def retrieve_results(self, query: str, **kwargs) -> HybridResult:
        return self.retrieve(query, intent_override="result", **kwargs)

    def retrieve_dataset(self, query: str, **kwargs) -> HybridResult:
        return self.retrieve(query, intent_override="dataset", **kwargs)

    def retrieve_figure(self, query: str, **kwargs) -> HybridResult:
        return self.retrieve(query, intent_override="figure", **kwargs)

    def retrieve_definition(self, query: str, **kwargs) -> HybridResult:
        return self.retrieve(query, intent_override="definition", **kwargs)

    # ──────────────────────────────────────────────────────────────────────────
    # DIAGNOSTICS
    # ──────────────────────────────────────────────────────────────────────────

    def print_result(self, result: HybridResult, show_graph: bool = True) -> None:
        p = result.provenance
        print(f"\n{'='*60}")
        print(f"  HYBRID RETRIEVAL RESULT (v2)")
        print(f"{'='*60}")
        print(f"  Query        : {p.query}")
        print(f"  Intent       : {p.intent}  (conf={p.intent_conf:.2f}, source={p.intent_source})")
        print(f"  Alpha        : {p.alpha}  →  α·FAISS + (1-α)·Graph")
        print(f"  Seed bonus   : +{p.seed_bonus}")
        print(f"  Seeds        : {len(p.faiss_seeds)} FAISS nodes")
        print(f"  Expanded     : {len(p.expanded_nodes)} graph neighbours (cap={MAX_KG_EXPANSION})")
        print(f"  Relations    : {p.relations_used}")
        print(f"  Candidates   : {p.total_candidates}  →  final: {p.final_k}")
        print(f"\n  Ranked Results:")
        print(f"  {'#':<3} {'node_id':<14} {'role':<12} {'pg':<4} "
              f"{'final':>6} {'faiss':>6} {'graph':>6}  {'origin':<18}  text[:60]")
        print(f"  {'-'*104}")
        for i, n in enumerate(result.ranked_results):
            print(
                f"  {i+1:<3} {n.node_id[:12]:<14} {n.role:<12} {n.page:<4} "
                f"{n.final_score:>6.4f} {n.faiss_score:>6.4f} {n.graph_score:>6.4f}  "
                f"{n.origin:<18}  {n.text[:60].replace(chr(10),' ')}"
            )
        if show_graph and result.subgraph.number_of_edges() > 0:
            print(f"\n  Subgraph edges ({result.subgraph.number_of_edges()}):")
            for line in result.edge_explanations():
                print(f"    {line}")
        print(f"{'='*60}\n")

    def validate(self) -> list[str]:
        warnings: list[str] = []
        warnings += self._faiss.validate()
        warnings += self._kg.validate()
        if self._kg.graph.number_of_nodes() == 0:
            warnings.append("Knowledge graph is empty — graph expansion will always return 0 neighbours.")
        if not (0.0 <= self._alpha <= 1.0):
            warnings.append(f"Alpha={self._alpha} is outside [0, 1].")
        return warnings

    # ──────────────────────────────────────────────────────────────────────────
    # INTERNAL
    # ──────────────────────────────────────────────────────────────────────────

    def _validate_agents(self) -> None:
        issues = self.validate()
        for w in issues:
            logger.warning("Validation: %s", w)

    def _empty_result(
        self,
        query: str,
        intent: str,
        intent_conf: float,
        intent_source: str,
        alpha: float,
    ) -> HybridResult:
        return HybridResult(
            ranked_results = [],
            subgraph       = nx.DiGraph(),
            provenance     = QueryProvenance(
                query=query, intent=intent, intent_conf=intent_conf,
                intent_source=intent_source, alpha=alpha, seed_bonus=SEED_BONUS,
                faiss_seeds=[], expanded_nodes=[], relations_used=[],
                total_candidates=0, final_k=0,
            ),
        )