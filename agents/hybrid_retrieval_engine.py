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
               │   result / figure /      │
               │   general                │
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
              └─────────────┬─────────────┘
                            │
              ┌─────────────▼─────────────┐
              │   HybridScorer             │
              │   final = α·faiss_score    │
              │         + (1-α)·graph_score│
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

Scoring
───────
FAISS seeds  : faiss_score  = cosine similarity ∈ [0, 1]
               graph_score  = edge_weight of the traversed edge  (0.0 if seed)
               final_score  = α · faiss_score + (1-α) · graph_score

Graph neighbours: faiss_score = cosine(query_vec, neighbour_vec) via FAISS
                  graph_score  = edge_weight (s_link or role_confidence)
                  final_score  = α · faiss_score + (1-α) · graph_score

Alpha (α) default = 0.6  → slightly favours semantic relevance over graph proximity.
Tunable at construction time or per-query.

Query intent → expansion relations
───────────────────────────────────
method    → produces, used_in, refers_to
dataset   → evaluated_on, evaluated_by, refers_to
result    → produces (reverse), evaluated_on (reverse), refers_to
figure    → refers_to
general   → produces, refers_to, evaluated_on        (broad, all high-value)
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
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

DEFAULT_ALPHA     = 0.6     # weight on FAISS score; (1-α) on graph score
DEFAULT_TOP_K     = 5       # seeds from FAISS
DEFAULT_EXPAND_K  = 3       # graph neighbours per seed
DEFAULT_FINAL_K   = 10      # results returned to caller
RRF_K             = 60      # not used (weighted sum chosen) — kept for reference

# Intent → outgoing relations to follow during graph expansion
INTENT_RELATIONS: dict[str, list[str]] = {
    "method":   ["produces", "used_in", "refers_to"],
    "dataset":  ["evaluated_on", "evaluated_by", "refers_to"],
    "result":   ["refers_to"],           # results rarely have outgoing edges;
                                         # see _expand_results() for reverse traversal
    "figure":   ["refers_to"],
    "general":  ["produces", "refers_to", "evaluated_on"],
}

# Intent → incoming relations to follow (reverse edges) — used for "result" queries
INTENT_REVERSE_RELATIONS: dict[str, list[str]] = {
    "result":  ["produces", "evaluated_on"],
    "dataset": ["evaluated_by"],
}

# Keyword patterns for lightweight intent detection
_INTENT_PATTERNS: list[tuple[str, list[str]]] = [
    ("method",  [r"\bmethod\b", r"\bapproach\b", r"\balgorithm\b", r"\btechnique\b",
                 r"\barchitecture\b", r"\bmodel\b", r"\bpipeline\b", r"\bframework\b",
                 r"\bhow (does|do|did|is)\b", r"\bpropose[ds]?\b", r"\btrain(ing)?\b"]),
    ("dataset", [r"\bdataset\b", r"\bbenchmark\b", r"\bcorpus\b", r"\bevaluat\w+\b",
                 r"\bbaseline\b", r"\bexperiment\b", r"\btrain(ing)? (set|data)\b"]),
    ("result",  [r"\bresult\b", r"\bperformance\b", r"\baccuracy\b", r"\bscore\b",
                 r"\bimprove\w*\b", r"\boutperform\w*\b", r"\bf1\b", r"\bbleu\b",
                 r"\bprecision\b", r"\brecall\b", r"\bmetric\b"]),
    ("figure",  [r"\bfigure\b", r"\bfig\b", r"\btable\b", r"\bplot\b", r"\bdiagram\b",
                 r"\bvisuali[sz]\w*\b", r"\billustrat\w*\b", r"\bshown? in\b"]),
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
    final_score:   float          # α·faiss + (1-α)·graph
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
    alpha:          float
    faiss_seeds:    list[str]                # node_ids
    expanded_nodes: list[str]                # node_ids added by graph
    relations_used: list[str]                # which relation types were followed
    total_candidates: int
    final_k:        int


@dataclass
class HybridResult:
    """
    The complete output of one HybridRetrievalEngine.retrieve() call.

    ranked_results  → flat list[ScoredNode] sorted by final_score desc.
                      Pass directly to the LLM as context passages.

    subgraph        → nx.DiGraph containing only the nodes and edges
                      involved in this retrieval.  Use for:
                        • structured reasoning (chain-of-thought over edges)
                        • explainability ("Method X produces Result Y")
                        • follow-up graph queries without re-running FAISS

    provenance      → QueryProvenance — audit trail, scores breakdown,
                      intent classification used.
    """
    ranked_results: list[ScoredNode]
    subgraph:       nx.DiGraph
    provenance:     QueryProvenance

    # ── Convenience accessors used by the answer generator ───────────────────

    def top_texts(self, k: int = 5) -> list[str]:
        """Return the top-k text snippets ready to inject into an LLM prompt."""
        return [r.text for r in self.ranked_results[:k] if r.text.strip()]

    def top_nodes(self, k: int = 5) -> list[ScoredNode]:
        return self.ranked_results[:k]

    def nodes_by_role(self, role: str) -> list[ScoredNode]:
        return [r for r in self.ranked_results if r.role == role]

    def edge_explanations(self) -> list[str]:
        """
        Human-readable strings for every edge in the subgraph.
        e.g. "Method[ec2685] --produces--> Result[3b3ae1]"
        Used to build citation trails in the answer.
        """
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

    Intent drives which graph relations are followed during expansion,
    so it must be fast (no model inference) and conservative —
    ambiguous queries default to "general".
    """

    def classify(self, query: str) -> tuple[str, float]:
        q = query.lower()
        scores: dict[str, int] = defaultdict(int)
        total_patterns = 0

        for intent, patterns in _INTENT_PATTERNS:
            for pat in patterns:
                total_patterns += 1
                if re.search(pat, q):
                    scores[intent] += 1

        if not scores:
            return "general", 0.0

        best_intent = max(scores, key=lambda k: scores[k])
        confidence  = scores[best_intent] / max(len(_INTENT_PATTERNS[0][1]), 1)
        confidence  = min(confidence, 1.0)

        # Tie-break: if two intents match equally, use "general"
        top_score = scores[best_intent]
        top_intents = [k for k, v in scores.items() if v == top_score]
        if len(top_intents) > 1:
            return "general", round(confidence, 3)

        return best_intent, round(confidence, 3)


# ──────────────────────────────────────────────────────────────────────────────
# Graph Expander
# ──────────────────────────────────────────────────────────────────────────────

class GraphExpander:
    """
    Given a set of seed node_ids and a query intent, walks the knowledge graph
    and returns (neighbour_node_id, edge_weight, relation) tuples.

    Follows outgoing edges for most intents.
    For "result" intent also follows *incoming* edges (who produced this result?).
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
        Returns list of (seed_id, neighbour_id, edge_weight, relation).
        neighbour_id is never in seed_ids (no self-loops back to seeds).
        """
        relations_out = INTENT_RELATIONS.get(intent, INTENT_RELATIONS["general"])
        relations_in  = INTENT_REVERSE_RELATIONS.get(intent, [])

        seed_set = set(seed_ids)
        results:  list[tuple[str, str, float, str]] = []

        for seed in seed_ids:
            if seed not in self._graph:
                continue

            added = 0

            # ── Outgoing edges ─────────────────────────────────────────────
            for _, nbr, data in self._graph.out_edges(seed, data=True):
                if added >= expand_k:
                    break
                rel = data.get("relation", "")
                if rel not in relations_out:
                    continue
                if nbr in seed_set:
                    continue
                weight = float(data.get("weight", 0.5))
                results.append((seed, nbr, weight, rel))
                added += 1

            # ── Incoming edges (reverse traversal for result/dataset) ───────
            for src, _, data in self._graph.in_edges(seed, data=True):
                if added >= expand_k:
                    break
                rel = data.get("relation", "")
                if rel not in relations_in:
                    continue
                if src in seed_set:
                    continue
                weight = float(data.get("weight", 0.5))
                results.append((seed, src, weight, f"←{rel}"))
                added += 1

        return results


# ──────────────────────────────────────────────────────────────────────────────
# Hybrid Scorer
# ──────────────────────────────────────────────────────────────────────────────

class HybridScorer:
    """
    Merges FAISS results and graph expansion hits into a single ranked list.

    Scoring
    ───────
    For every candidate node:
        final_score = α · faiss_score + (1 − α) · graph_score

    If a node appears as both a FAISS seed AND a graph neighbour,
    the best faiss_score and best graph_score are kept and merged —
    origin is marked "both".

    Deduplication is by node_id.
    """

    def __init__(self, alpha: float = DEFAULT_ALPHA) -> None:
        self.alpha = alpha

    def merge(
        self,
        faiss_results:      list[RetrievalResult],
        expansion_hits:     list[tuple[str, str, float, str]],     # (seed, nbr, w, rel)
        faiss_lookup:       dict[str, RetrievalResult],            # node_id → result
        graph:              nx.DiGraph,
        query_vec_scorer:   Optional[Any] = None,                  # FAISSAgent for re-scoring
    ) -> list[ScoredNode]:
        """
        Parameters
        ──────────
        faiss_results    : direct FAISS hits
        expansion_hits   : (seed_id, nbr_id, edge_weight, relation) from GraphExpander
        faiss_lookup     : node_id → RetrievalResult for O(1) score access
        graph            : full KG — used to fetch neighbour node attributes
        query_vec_scorer : FAISSAgent — if provided, graph neighbours get a real
                           FAISS cosine score; otherwise graph_score proxies for it
        """

        # ── Accumulator: node_id → best scores seen so far ─────────────────
        acc: dict[str, dict[str, Any]] = {}

        # 1. Seed nodes from FAISS
        for fr in faiss_results:
            acc[fr.node_id] = {
                "faiss_score":    fr.score,
                "graph_score":    0.0,
                "origin":         "faiss_seed",
                "relation_path":  [],
                "result":         fr,
            }

        # 2. Graph expansion neighbours
        for seed_id, nbr_id, edge_weight, relation in expansion_hits:
            nbr_faiss_score = 0.0
            if nbr_id in faiss_lookup:
                nbr_faiss_score = faiss_lookup[nbr_id].score

            if nbr_id in acc:
                # Node already seen — upgrade scores if better
                existing = acc[nbr_id]
                existing["faiss_score"] = max(existing["faiss_score"], nbr_faiss_score)
                existing["graph_score"] = max(existing["graph_score"], edge_weight)
                existing["origin"]      = "both"
                existing["relation_path"].append(relation)
            else:
                # New node arriving via graph — fetch its metadata from the graph
                g_attrs = graph.nodes.get(nbr_id, {})
                # Build a minimal RetrievalResult from graph attrs
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
                    "graph_score":   edge_weight,
                    "origin":        "graph_expansion",
                    "relation_path": [relation],
                    "result":        pseudo_result,
                }

        # 3. Compute final scores and build ScoredNode list
        scored: list[ScoredNode] = []
        for nid, entry in acc.items():
            fs = entry["faiss_score"]
            gs = entry["graph_score"]
            final = round(self.alpha * fs + (1.0 - self.alpha) * gs, 4)

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
    """
    Extracts a minimal subgraph from the full KG containing only the nodes
    and edges that were actually involved in this retrieval.

    This subgraph is what the answer generator uses for structured reasoning —
    it can traverse Method → Result chains directly without querying the full graph.
    """

    @staticmethod
    def build(
        node_ids:  list[str],
        graph:     nx.DiGraph,
    ) -> nx.DiGraph:
        """
        Returns an induced subgraph on node_ids PLUS all edges that connect
        any two nodes in the set.  Preserves all node and edge attributes.
        """
        node_set = set(node_ids)
        sub = nx.DiGraph()

        # Add nodes with full attributes
        for nid in node_set:
            if nid in graph:
                sub.add_node(nid, **dict(graph.nodes[nid]))

        # Add all edges between nodes in the set
        for src, tgt, data in graph.edges(data=True):
            if src in node_set and tgt in node_set:
                sub.add_edge(src, tgt, **data)

        return sub


# ──────────────────────────────────────────────────────────────────────────────
# Hybrid Retrieval Engine  (public interface)
# ──────────────────────────────────────────────────────────────────────────────

class HybridRetrievalEngine:
    """
    Step 7 — Hybrid Retrieval Engine.

    Combines FAISSAgent (semantic similarity) and KnowledgeGraphAgent
    (structured relationships) into a single retrieve() call.

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

        # Access flat list for LLM:
        passages = result.top_texts(k=5)

        # Access subgraph for structured reasoning:
        for explanation in result.edge_explanations():
            print(explanation)

        # Full audit trail:
        print(result.provenance)
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
        expand_k        : graph neighbours per seed  (default 3)
        final_k         : nodes to return in ranked_results  (default 10)
        alpha           : override instance-level alpha for this call
        role_filter     : restrict FAISS seeds to these scholarly roles
                          e.g. ["Method", "Result"]
        intent_override : skip intent detection, force this intent
                          one of: method | dataset | result | figure | general

        Returns
        ───────
        HybridResult with .ranked_results, .subgraph, .provenance
        """
        _alpha = alpha if alpha is not None else self._alpha
        self._scorer.alpha = _alpha

        # ── 1. Intent classification ──────────────────────────────────────────
        if intent_override:
            intent, intent_conf = intent_override, 1.0
        else:
            intent, intent_conf = self._intent.classify(query)

        logger.info("Query: '%s'  →  intent=%s (conf=%.2f)", query[:80], intent, intent_conf)

        # ── 2. FAISS semantic search ──────────────────────────────────────────
        faiss_results: list[RetrievalResult] = self._faiss.search(
            query,
            top_k       = top_k,
            role_filter = role_filter,
            include_noise = False,
        )

        if not faiss_results:
            logger.warning("FAISS returned no results for query: '%s'", query)
            return self._empty_result(query, intent, intent_conf, _alpha)

        seed_ids = [r.node_id for r in faiss_results]
        faiss_lookup = {r.node_id: r for r in faiss_results}

        logger.info("FAISS seeds: %d nodes  %s", len(seed_ids),
                    [s[:8] for s in seed_ids])

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
            faiss_results   = faiss_results,
            expansion_hits  = expansion_hits,
            faiss_lookup    = faiss_lookup,
            graph           = self._kg.graph,
        )

        # Apply role filter to final list if specified
        if role_filter:
            scored_nodes = [n for n in scored_nodes if n.role in role_filter] \
                         + [n for n in scored_nodes if n.role not in role_filter]

        final_nodes = scored_nodes[:final_k]

        # ── 5. Build subgraph ─────────────────────────────────────────────────
        all_node_ids = [n.node_id for n in final_nodes]
        subgraph     = SubgraphBuilder.build(all_node_ids, self._kg.graph)

        # ── 6. Provenance ─────────────────────────────────────────────────────
        provenance = QueryProvenance(
            query           = query,
            intent          = intent,
            intent_conf     = intent_conf,
            alpha           = _alpha,
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

    # ──────────────────────────────────────────────────────────────────────────
    # DIAGNOSTICS
    # ──────────────────────────────────────────────────────────────────────────

    def print_result(self, result: HybridResult, show_graph: bool = True) -> None:
        """Pretty-print a HybridResult for debugging."""
        p = result.provenance
        print(f"\n{'='*56}")
        print(f"  HYBRID RETRIEVAL RESULT")
        print(f"{'='*56}")
        print(f"  Query   : {p.query}")
        print(f"  Intent  : {p.intent}  (conf={p.intent_conf:.2f})")
        print(f"  Alpha   : {p.alpha}  →  α·FAISS + (1-α)·Graph")
        print(f"  Seeds   : {len(p.faiss_seeds)} FAISS nodes")
        print(f"  Expanded: {len(p.expanded_nodes)} graph neighbours")
        print(f"  Relations used: {p.relations_used}")
        print(f"  Total candidates: {p.total_candidates}  →  final: {p.final_k}")
        print(f"\n  Ranked Results:")
        print(f"  {'#':<3} {'node_id':<14} {'role':<12} {'pg':<4} "
              f"{'final':>6} {'faiss':>6} {'graph':>6}  {'origin':<16}  text[:60]")
        print(f"  {'-'*100}")
        for i, n in enumerate(result.ranked_results):
            print(
                f"  {i+1:<3} {n.node_id[:12]:<14} {n.role:<12} {n.page:<4} "
                f"{n.final_score:>6.4f} {n.faiss_score:>6.4f} {n.graph_score:>6.4f}  "
                f"{n.origin:<16}  {n.text[:60].replace(chr(10),' ')}"
            )
        if show_graph and result.subgraph.number_of_edges() > 0:
            print(f"\n  Subgraph edges ({result.subgraph.number_of_edges()}):")
            for line in result.edge_explanations():
                print(f"    {line}")
        print(f"{'='*56}\n")

    def validate(self) -> list[str]:
        """Pre-flight checks before serving queries."""
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
        alpha: float,
    ) -> HybridResult:
        return HybridResult(
            ranked_results = [],
            subgraph       = nx.DiGraph(),
            provenance     = QueryProvenance(
                query=query, intent=intent, intent_conf=intent_conf,
                alpha=alpha, faiss_seeds=[], expanded_nodes=[],
                relations_used=[], total_candidates=0, final_k=0,
            ),
        )