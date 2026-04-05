"""
retrieval_evaluation.py
========================
Retrieval Evaluation Suite for agenticdoc2
Evaluates: FAISS-only  vs  Hybrid (FAISS + KG)

Metrics implemented
────────────────────
  • Precision@K
  • Recall@K
  • MRR  (Mean Reciprocal Rank)
  • nDCG (Normalised Discounted Cumulative Gain)

Usage
─────
    python retrieval_evaluation.py \
        --semantic-json output/paper/json/layout_semantic.json \
        --k 5 10 \
        --report eval_report.json

    # Or in Python:
    from retrieval_evaluation import RetrievalEvaluator
    evaluator = RetrievalEvaluator(faiss_agent, engine)
    report    = evaluator.run(EVAL_QUERIES, k_values=[5, 10])
    evaluator.print_report(report)
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)

# ──────────────────────────────────────────────────────────────────────────────
# 1.  Evaluation query bank
#     Each entry has:
#       query        – natural language question
#       relevant_ids – set of node_ids considered ground-truth relevant
#       relevant_roles – (optional) fallback: any node with this role is relevant
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class EvalQuery:
    """A single labelled query for evaluation."""
    query:          str
    relevant_ids:   set[str]              # exact node_id matches (gold standard)
    relevant_roles: set[str] = field(default_factory=set)  # role-level fallback
    description:    str      = ""         # human label, e.g. "method retrieval"

    # ── Helpers ───────────────────────────────────────────────────────────────

    def is_relevant(self, node_id: str, role: str = "") -> bool:
        """Return True if node is judged relevant for this query."""
        if self.relevant_ids and node_id in self.relevant_ids:
            return True
        if self.relevant_roles and role in self.relevant_roles:
            return True
        return False

    def total_relevant(self) -> int:
        """Number of documents judged relevant (lower-bound estimate)."""
        return max(len(self.relevant_ids), 1)


# ──────────────────────────────────────────────────────────────────────────────
# 2.  Default evaluation query bank
#     In a real evaluation you label these manually against your actual
#     node_ids from layout_semantic.json.  The role-level fallback lets you
#     evaluate even without per-node labels.
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_EVAL_QUERIES: list[EvalQuery] = [
    EvalQuery(
        query           = "What method or approach does this paper propose?",
        relevant_ids    = set(),
        relevant_roles  = {"Method"},
        description     = "method_retrieval",
    ),
    EvalQuery(
        query           = "What are the main results and performance metrics?",
        relevant_ids    = set(),
        relevant_roles  = {"Result"},
        description     = "result_retrieval",
    ),
    EvalQuery(
        query           = "What datasets or benchmarks were used for evaluation?",
        relevant_ids    = set(),
        relevant_roles  = {"Dataset"},
        description     = "dataset_retrieval",
    ),
    EvalQuery(
        query           = "What are the key definitions and terminology?",
        relevant_ids    = set(),
        relevant_roles  = {"Definition"},
        description     = "definition_retrieval",
    ),
    EvalQuery(
        query           = "What observations or findings are reported?",
        relevant_ids    = set(),
        relevant_roles  = {"Observation"},
        description     = "observation_retrieval",
    ),
    EvalQuery(
        query           = "How does the model training procedure work?",
        relevant_ids    = set(),
        relevant_roles  = {"Method"},
        description     = "training_method",
    ),
    EvalQuery(
        query           = "What figures and tables are referenced in the paper?",
        relevant_ids    = set(),
        relevant_roles  = {"Result", "Observation"},
        description     = "figure_table_reference",
    ),
    EvalQuery(
        query           = "What is the baseline comparison for the proposed approach?",
        relevant_ids    = set(),
        relevant_roles  = {"Result", "Method"},
        description     = "baseline_comparison",
    ),
]


# ──────────────────────────────────────────────────────────────────────────────
# 3.  Per-query result
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class QueryResult:
    """Metrics for a single query at a single K."""
    query:        str
    description:  str
    k:            int
    precision_at_k: float
    recall_at_k:    float
    reciprocal_rank: float      # 1/rank of first relevant doc (0 if none)
    ndcg_at_k:      float
    retrieved_ids:  list[str]   # ordered
    relevant_hits:  list[bool]  # parallel to retrieved_ids
    latency_ms:     float = 0.0

    # ── Convenience ───────────────────────────────────────────────────────────

    def is_hit(self) -> bool:
        """True if at least one relevant document was retrieved."""
        return any(self.relevant_hits)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query":           self.query,
            "description":     self.description,
            "k":               self.k,
            "precision@k":     round(self.precision_at_k, 4),
            "recall@k":        round(self.recall_at_k, 4),
            "reciprocal_rank": round(self.reciprocal_rank, 4),
            "ndcg@k":          round(self.ndcg_at_k, 4),
            "hit":             self.is_hit(),
            "latency_ms":      round(self.latency_ms, 2),
        }


@dataclass
class SystemReport:
    """Aggregated report for one retrieval system across all queries and K values."""
    system_name:    str
    k_values:       list[int]
    per_query:      list[QueryResult]   # all queries × all k values

    # ── Aggregated metrics ────────────────────────────────────────────────────

    def aggregate(self, k: int) -> dict[str, float]:
        """Mean metrics for a given K."""
        rows = [r for r in self.per_query if r.k == k]
        if not rows:
            return {}
        return {
            "mean_precision@k": round(np.mean([r.precision_at_k  for r in rows]), 4),
            "mean_recall@k":    round(np.mean([r.recall_at_k     for r in rows]), 4),
            "MRR":              round(np.mean([r.reciprocal_rank  for r in rows]), 4),
            "mean_nDCG@k":      round(np.mean([r.ndcg_at_k       for r in rows]), 4),
            "hit_rate":         round(np.mean([float(r.is_hit()) for r in rows]), 4),
            "mean_latency_ms":  round(np.mean([r.latency_ms      for r in rows]), 2),
            "n_queries":        len(rows),
        }

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"system": self.system_name, "by_k": {}}
        for k in self.k_values:
            out["by_k"][str(k)] = self.aggregate(k)
        out["per_query"] = [r.to_dict() for r in self.per_query]
        return out


@dataclass
class ComparisonReport:
    """Side-by-side comparison: FAISS-only vs Hybrid."""
    faiss_report:  SystemReport
    hybrid_report: SystemReport
    k_values:      list[int]

    def delta(self, k: int) -> dict[str, float]:
        """
        Hybrid − FAISS for each metric.
        Positive delta = Hybrid is better.
        """
        fa = self.faiss_report.aggregate(k)
        hy = self.hybrid_report.aggregate(k)
        metrics = ["mean_precision@k", "mean_recall@k", "MRR", "mean_nDCG@k", "hit_rate"]
        return {
            m: round(hy.get(m, 0.0) - fa.get(m, 0.0), 4)
            for m in metrics
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "faiss_only": self.faiss_report.to_dict(),
            "hybrid":     self.hybrid_report.to_dict(),
            "delta_hybrid_minus_faiss": {
                str(k): self.delta(k) for k in self.k_values
            },
        }


# ──────────────────────────────────────────────────────────────────────────────
# 4.  Core metric functions
# ──────────────────────────────────────────────────────────────────────────────

def precision_at_k(hits: list[bool], k: int) -> float:
    """
    Precision@K  =  |relevant ∩ retrieved[:K]|  /  K

    Parameters
    ──────────
    hits : ordered list of bool — True if doc at rank i is relevant
    k    : cutoff rank

    Returns
    ───────
    float in [0, 1]
    """
    if k <= 0:
        return 0.0
    truncated = hits[:k]
    return sum(truncated) / k


def recall_at_k(hits: list[bool], total_relevant: int, k: int) -> float:
    """
    Recall@K  =  |relevant ∩ retrieved[:K]|  /  |total_relevant|

    Parameters
    ──────────
    hits           : ordered hit list
    total_relevant : ground truth count
    k              : cutoff rank

    Returns
    ───────
    float in [0, 1]
    """
    if total_relevant <= 0:
        return 0.0
    retrieved_relevant = sum(hits[:k])
    return retrieved_relevant / total_relevant


def mean_reciprocal_rank(hits: list[bool]) -> float:
    """
    MRR  =  1 / rank_of_first_relevant_doc
         =  0  if no relevant doc found

    For a single query this is just the Reciprocal Rank (RR).
    Average across queries to get MRR.

    Parameters
    ──────────
    hits : ordered hit list (full retrieved list, no K cutoff)

    Returns
    ───────
    float in [0, 1]
    """
    for rank, hit in enumerate(hits, start=1):
        if hit:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(hits: list[bool], k: int) -> float:
    """
    nDCG@K  =  DCG@K  /  IDCG@K

    DCG@K   =  Σ_{i=1}^{K}  rel_i / log2(i + 1)
    IDCG@K  =  DCG of the ideal (perfect) ranking

    Binary relevance: rel_i ∈ {0, 1}

    Parameters
    ──────────
    hits : ordered hit list
    k    : cutoff rank

    Returns
    ───────
    float in [0, 1]
    """
    if k <= 0:
        return 0.0
    truncated = hits[:k]

    # Actual DCG
    dcg = sum(
        rel / math.log2(rank + 1)
        for rank, rel in enumerate(truncated, start=1)
        if rel
    )

    # Ideal DCG: all relevant docs at the top
    n_relevant = sum(truncated)
    idcg = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(1, min(n_relevant, k) + 1)
    )

    if idcg == 0.0:
        return 0.0
    return dcg / idcg


# ──────────────────────────────────────────────────────────────────────────────
# 5.  Retrieval Evaluator
# ──────────────────────────────────────────────────────────────────────────────

class RetrievalEvaluator:
    """
    Evaluates FAISS-only and Hybrid retrieval against a query bank.

    Typical usage
    ─────────────
        evaluator = RetrievalEvaluator(faiss_agent, hybrid_engine)
        report    = evaluator.run(queries, k_values=[5, 10])
        evaluator.print_report(report)
        evaluator.save_report(report, "eval_report.json")
    """

    def __init__(
        self,
        faiss_agent,            # agents.faiss_agent.FAISSAgent
        hybrid_engine,          # agents.hybrid_retrieval_engine.HybridRetrievalEngine
    ) -> None:
        self._faiss  = faiss_agent
        self._hybrid = hybrid_engine

    # ── Primary API ───────────────────────────────────────────────────────────

    def run(
        self,
        queries:  list[EvalQuery],
        k_values: list[int] = None,
    ) -> ComparisonReport:
        """
        Run full evaluation and return a ComparisonReport.

        Parameters
        ──────────
        queries  : list of EvalQuery with relevance labels
        k_values : list of K cutoffs to evaluate at (default [5, 10])

        Returns
        ───────
        ComparisonReport with both FAISS-only and Hybrid reports
        """
        if k_values is None:
            k_values = [5, 10]

        max_k = max(k_values)

        faiss_results:  list[QueryResult] = []
        hybrid_results: list[QueryResult] = []

        logger.info("Starting evaluation: %d queries × %s k-values", len(queries), k_values)

        for i, eq in enumerate(queries):
            logger.info("[%d/%d] Query: %s", i + 1, len(queries), eq.query[:60])

            # ── FAISS-only retrieval ─────────────────────────────────────────
            t0 = time.perf_counter()
            faiss_hits = self._faiss_retrieve(eq.query, top_k=max_k)
            faiss_lat  = (time.perf_counter() - t0) * 1000

            # ── Hybrid retrieval ─────────────────────────────────────────────
            t0 = time.perf_counter()
            hybrid_hits = self._hybrid_retrieve(eq.query, top_k=max_k)
            hybrid_lat  = (time.perf_counter() - t0) * 1000

            # ── Compute hit vectors ──────────────────────────────────────────
            faiss_relevance  = self._relevance_vector(faiss_hits, eq)
            hybrid_relevance = self._relevance_vector(hybrid_hits, eq)

            total_rel = eq.total_relevant()

            # ── Compute metrics at each K ────────────────────────────────────
            for k in k_values:
                faiss_results.append(QueryResult(
                    query           = eq.query,
                    description     = eq.description,
                    k               = k,
                    precision_at_k  = precision_at_k(faiss_relevance, k),
                    recall_at_k     = recall_at_k(faiss_relevance, total_rel, k),
                    reciprocal_rank = mean_reciprocal_rank(faiss_relevance),
                    ndcg_at_k       = ndcg_at_k(faiss_relevance, k),
                    retrieved_ids   = [n for n, _ in faiss_hits[:k]],
                    relevant_hits   = faiss_relevance[:k],
                    latency_ms      = faiss_lat,
                ))

                hybrid_results.append(QueryResult(
                    query           = eq.query,
                    description     = eq.description,
                    k               = k,
                    precision_at_k  = precision_at_k(hybrid_relevance, k),
                    recall_at_k     = recall_at_k(hybrid_relevance, total_rel, k),
                    reciprocal_rank = mean_reciprocal_rank(hybrid_relevance),
                    ndcg_at_k       = ndcg_at_k(hybrid_relevance, k),
                    retrieved_ids   = [n for n, _ in hybrid_hits[:k]],
                    relevant_hits   = hybrid_relevance[:k],
                    latency_ms      = hybrid_lat,
                ))

        faiss_report  = SystemReport("FAISS-only", k_values, faiss_results)
        hybrid_report = SystemReport("Hybrid (FAISS+KG)", k_values, hybrid_results)
        return ComparisonReport(faiss_report, hybrid_report, k_values)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _faiss_retrieve(
        self, query: str, top_k: int
    ) -> list[tuple[str, str]]:
        """
        Returns list of (node_id, role) tuples from FAISS-only retrieval.
        """
        try:
            results = self._faiss.search(query, top_k=top_k)
            return [(r.node_id, r.role) for r in results]
        except Exception as exc:
            logger.warning("FAISS retrieval failed for query %r: %s", query, exc)
            return []

    def _hybrid_retrieve(
        self, query: str, top_k: int
    ) -> list[tuple[str, str]]:
        """
        Returns list of (node_id, role) tuples from Hybrid retrieval.
        """
        try:
            result = self._hybrid.retrieve(query, final_k=top_k)
            return [(n.node_id, n.role) for n in result.ranked_results]
        except Exception as exc:
            logger.warning("Hybrid retrieval failed for query %r: %s", query, exc)
            return []

    def _relevance_vector(
        self,
        retrieved: list[tuple[str, str]],
        eq: EvalQuery,
    ) -> list[bool]:
        """Binary hit vector: True if node_id is relevant to this query."""
        return [eq.is_relevant(nid, role) for nid, role in retrieved]

    # ── Reporting ─────────────────────────────────────────────────────────────

    @staticmethod
    def print_report(report: ComparisonReport) -> None:
        """Pretty-print the comparison report to stdout."""
        WIDTH = 72

        print()
        print("=" * WIDTH)
        print("  RETRIEVAL EVALUATION REPORT — FAISS-only vs Hybrid (FAISS+KG)")
        print("=" * WIDTH)

        for k in report.k_values:
            fa = report.faiss_report.aggregate(k)
            hy = report.hybrid_report.aggregate(k)
            dl = report.delta(k)

            print(f"\n  ── K = {k} ──────────────────────────────────────────────")
            header = f"  {'Metric':<22} {'FAISS-only':>12} {'Hybrid':>12} {'Δ (Hybrid−FAISS)':>18}"
            print(header)
            print("  " + "─" * (WIDTH - 2))

            def row(label: str, key: str) -> None:
                f_val = fa.get(key, 0.0)
                h_val = hy.get(key, 0.0)
                d_val = dl.get(key, 0.0)
                sign  = "+" if d_val >= 0 else ""
                arrow = "▲" if d_val > 0 else ("▼" if d_val < 0 else "–")
                print(f"  {label:<22} {f_val:>12.4f} {h_val:>12.4f}   "
                      f"{arrow} {sign}{d_val:.4f}")

            row("Precision@K",    "mean_precision@k")
            row("Recall@K",       "mean_recall@k")
            row("MRR",            "MRR")
            row("nDCG@K",         "mean_nDCG@k")
            row("Hit Rate",       "hit_rate")
            row("Latency (ms)",   "mean_latency_ms")

        print()
        print("  ── Per-query breakdown (K=%d) ─────────────────────────────" % report.k_values[0])
        k0 = report.k_values[0]
        faiss_rows  = {r.description: r for r in report.faiss_report.per_query  if r.k == k0}
        hybrid_rows = {r.description: r for r in report.hybrid_report.per_query if r.k == k0}

        for desc in sorted(faiss_rows.keys()):
            fr = faiss_rows[desc]
            hr = hybrid_rows.get(desc)
            f_ndcg = fr.ndcg_at_k
            h_ndcg = hr.ndcg_at_k if hr else 0.0
            delta  = h_ndcg - f_ndcg
            sign   = "+" if delta >= 0 else ""
            arrow  = "▲" if delta > 0 else ("▼" if delta < 0 else "–")
            print(f"  {desc:<30} nDCG  FAISS={f_ndcg:.3f}  Hybrid={h_ndcg:.3f}  "
                  f"{arrow}{sign}{delta:.3f}")

        print()
        print("=" * WIDTH)
        print()

    @staticmethod
    def save_report(report: ComparisonReport, path: str) -> None:
        """Serialise report to JSON."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
        logger.info("Report saved → %s", path)


# ──────────────────────────────────────────────────────────────────────────────
# 6.  Standalone metric unit-tests  (run with: python retrieval_evaluation.py --test)
# ──────────────────────────────────────────────────────────────────────────────

def _run_metric_tests() -> None:
    """Quick sanity-checks of all four metric functions."""
    print("\n── Running metric unit-tests ──\n")
    errors = 0

    # ── Precision@K ──────────────────────────────────────────────────────────
    assert precision_at_k([True, False, True, False, False], k=5) == 0.4, "P@5 failed"
    assert precision_at_k([True, True, True], k=3)  == 1.0, "P@3 perfect failed"
    assert precision_at_k([False, False], k=2)      == 0.0, "P@K zero failed"
    assert precision_at_k([], k=5)                  == 0.0, "P@K empty failed"
    print("  ✓ Precision@K")

    # ── Recall@K ─────────────────────────────────────────────────────────────
    assert recall_at_k([True, False, True, False, True], 3, k=5) == 1.0, "R@5 perfect"
    assert recall_at_k([True, False, False, False], 2, k=1)      == 0.5, "R@1 partial"
    assert recall_at_k([False, False], 2, k=2)                   == 0.0, "R@K zero"
    assert recall_at_k([], 0, k=5)                               == 0.0, "R@K no rel"
    print("  ✓ Recall@K")

    # ── MRR ──────────────────────────────────────────────────────────────────
    assert mean_reciprocal_rank([False, True, False]) == 0.5,  "MRR rank-2"
    assert mean_reciprocal_rank([True,  False, False]) == 1.0, "MRR rank-1"
    assert mean_reciprocal_rank([False, False, False]) == 0.0, "MRR no hit"
    assert mean_reciprocal_rank([False, False, True])  == pytest_approx(1/3)
    print("  ✓ MRR")

    # ── nDCG@K ───────────────────────────────────────────────────────────────
    # rank-1 hit: DCG = 1/log2(2) = 1.0; IDCG = 1.0 → nDCG = 1.0
    assert ndcg_at_k([True, False, False], k=3) == pytest_approx(1.0), "nDCG@3 rank-1"
    assert ndcg_at_k([False, False, False], k=3) == 0.0, "nDCG zero"
    perfect = ndcg_at_k([True, True, True], k=3)
    assert abs(perfect - 1.0) < 1e-6, f"nDCG perfect={perfect}"
    # Rank-2 hit: DCG = 1/log2(3) ≈ 0.631; IDCG = 1/log2(2) = 1.0
    rank2 = ndcg_at_k([False, True, False], k=3)
    expected = (1 / math.log2(3)) / (1 / math.log2(2))
    assert abs(rank2 - expected) < 1e-6, f"nDCG rank-2={rank2} expected={expected}"
    print("  ✓ nDCG@K")

    if errors == 0:
        print("\n  All metric tests passed ✓\n")
    else:
        print(f"\n  {errors} test(s) FAILED ✗\n")


def pytest_approx(x: float, rel: float = 1e-6) -> "_Approx":
    """Tiny helper so tests don't need pytest installed."""
    class _Approx:
        def __init__(self, v): self.v = v
        def __eq__(self, other): return abs(other - self.v) < rel * max(abs(self.v), 1e-9)
        def __repr__(self): return f"≈{self.v}"
    return _Approx(x)


# ──────────────────────────────────────────────────────────────────────────────
# 7.  Demo runner using mock agents (no real index required)
# ──────────────────────────────────────────────────────────────────────────────

class _MockFAISSAgent:
    """
    Mimics FAISSAgent.query().
    Returns results ordered by cosine similarity — simulated here by shuffling.
    In a real evaluation, replace with the actual FAISSAgent.
    """

    def __init__(self, all_nodes: list[dict]) -> None:
        self._nodes = all_nodes

    def query(self, query_text: str, top_k: int = 10):
        from dataclasses import make_dataclass
        import random

        rng    = random.Random(hash(query_text) % (2**31))
        pool   = self._nodes[:]
        rng.shuffle(pool)
        Result = make_dataclass("Result", ["node_id", "role", "score", "text"])

        results = []
        for i, n in enumerate(pool[:top_k]):
            score = max(0.0, 1.0 - i * 0.07 + rng.gauss(0, 0.03))
            results.append(Result(node_id=n["node_id"], role=n["role"],
                                  score=round(score, 4), text=n.get("text", "")))
        return results


class _MockHybridEngine:
    """
    Mimics HybridRetrievalEngine.retrieve().
    Graph expansion biases scores toward role-matched nodes.
    """

    def __init__(self, all_nodes: list[dict]) -> None:
        self._nodes = all_nodes

    def retrieve(self, query_text: str, final_k: int = 10):
        from dataclasses import make_dataclass
        import random

        rng    = random.Random(hash(query_text + "_hybrid") % (2**31))
        pool   = self._nodes[:]

        # Simple bias: boost nodes whose role appears in the query
        query_lower = query_text.lower()
        role_boost  = {
            "method":      "Method",
            "result":      "Result",
            "dataset":     "Dataset",
            "definition":  "Definition",
            "observation": "Observation",
        }
        target_role = next(
            (v for k, v in role_boost.items() if k in query_lower), None
        )

        def sort_key(n):
            base = rng.random()
            # Hybrid gives KG-boosted score for matching roles
            bonus = 0.25 if target_role and n["role"] == target_role else 0.0
            return -(base + bonus)

        pool.sort(key=sort_key)

        ScoredNode = make_dataclass("ScoredNode", ["node_id", "role", "final_score", "text"])
        HybridResult = make_dataclass("HybridResult", ["ranked_results"])

        nodes = [
            ScoredNode(
                node_id     = n["node_id"],
                role        = n["role"],
                final_score = round(max(0.0, 1.0 - i * 0.06), 4),
                text        = n.get("text", ""),
            )
            for i, n in enumerate(pool[:final_k])
        ]
        return HybridResult(ranked_results=nodes)


def _load_nodes_from_semantic_json(json_path: str) -> list[dict]:
    """Load node list from layout_semantic.json."""
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    nodes = []
    for sr in data.get("semantic_regions", []):
        nodes.append({
            "node_id": sr.get("region_id", ""),
            "role":    sr.get("scholarly_role", "N/A"),
            "text":    (sr.get("text_content") or "")[:300],
        })
    logger.info("Loaded %d nodes from %s", len(nodes), json_path)
    return nodes


def _build_queries_from_nodes(nodes: list[dict]) -> list[EvalQuery]:
    """
    Auto-build labeled queries by using role distribution.
    For each role that exists in nodes, create one query and label
    all nodes of that role as relevant.
    """
    role_to_ids: dict[str, list[str]] = {}
    for n in nodes:
        role = n["role"]
        if role != "N/A":
            role_to_ids.setdefault(role, []).append(n["node_id"])

    queries = []
    role_queries = {
        "Method":      "What method or approach does this paper propose?",
        "Result":      "What are the main results and performance metrics?",
        "Dataset":     "What datasets or benchmarks were used?",
        "Definition":  "What are the key definitions in this paper?",
        "Observation": "What observations or findings does the paper report?",
    }

    for role, node_ids in role_to_ids.items():
        if role in role_queries:
            queries.append(EvalQuery(
                query           = role_queries[role],
                relevant_ids    = set(node_ids),
                relevant_roles  = {role},
                description     = role.lower() + "_retrieval",
            ))
    return queries


# ──────────────────────────────────────────────────────────────────────────────
# 8.  CLI entry-point
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retrieval Evaluation — FAISS-only vs Hybrid"
    )
    parser.add_argument(
        "--semantic-json", "-j",
        default="output/paper/json/layout_semantic.json",
        help="Path to layout_semantic.json",
    )
    parser.add_argument(
        "--k", nargs="+", type=int, default=[5, 10],
        help="K values to evaluate at (default: 5 10)",
    )
    parser.add_argument(
        "--report", "-r",
        default="output/eval/retrieval_report.json",
        help="Output path for JSON report",
    )
    parser.add_argument(
        "--test", action="store_true",
        help="Run metric unit-tests only and exit",
    )
    parser.add_argument(
        "--use-real-agents", action="store_true",
        help="Load actual FAISSAgent and HybridRetrievalEngine (requires index built)",
    )
    args = parser.parse_args()

    # ── Unit-test mode ────────────────────────────────────────────────────────
    if args.test:
        _run_metric_tests()
        return

    # ── Load nodes ────────────────────────────────────────────────────────────
    semantic_path = Path(args.semantic_json)
    if not semantic_path.exists():
        logger.error("semantic-json not found: %s", semantic_path)
        logger.error("Run pipeline.py first, then re-run this script.")
        raise SystemExit(1)

    nodes = _load_nodes_from_semantic_json(str(semantic_path))

    if not nodes:
        logger.error("No nodes loaded — check your semantic JSON.")
        raise SystemExit(1)

    # ── Build agents ──────────────────────────────────────────────────────────
    if args.use_real_agents:
        logger.info("Loading real FAISSAgent + HybridRetrievalEngine …")
        from agents.faiss_agent              import FAISSAgent
        from agents.knowledge_graph_agent    import KnowledgeGraphAgent
        from agents.hybrid_retrieval_engine  import HybridRetrievalEngine

        faiss_agent = FAISSAgent()
        faiss_agent.build_from_json(str(semantic_path))

        kg_agent = KnowledgeGraphAgent()
        kg_agent.build_from_json(str(semantic_path))

        engine = HybridRetrievalEngine(faiss_agent, kg_agent)
    else:
        logger.info("Using mock agents (pass --use-real-agents to use real pipeline)")
        faiss_agent = _MockFAISSAgent(nodes)
        engine      = _MockHybridEngine(nodes)

    # ── Build queries ─────────────────────────────────────────────────────────
    queries = _build_queries_from_nodes(nodes)
    if not queries:
        logger.warning("No role-labelled queries built — using DEFAULT_EVAL_QUERIES")
        queries = DEFAULT_EVAL_QUERIES

    logger.info("Evaluating %d queries at K=%s", len(queries), args.k)

    # ── Run evaluation ────────────────────────────────────────────────────────
    evaluator = RetrievalEvaluator(faiss_agent, engine)
    report    = evaluator.run(queries, k_values=args.k)

    # ── Output ────────────────────────────────────────────────────────────────
    evaluator.print_report(report)
    evaluator.save_report(report, args.report)


if __name__ == "__main__":
    main()