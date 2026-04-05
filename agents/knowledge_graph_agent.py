# agents/knowledge_graph_agent.py
"""
Step 5 — Knowledge Graph Construction Agent
============================================
Consumes semantic.json (from SemanticUnderstandingAgent / Step 4).
Produces a NetworkX DiGraph used downstream as a structured retrieval layer.

Graph anatomy
─────────────
Nodes  — one per SemanticRegion  (paragraph, figure, table, caption, …)
Edges  — four distinct sources:
  1. multimodal_links  → caption  ──refers_to──►  figure/table   (from linker)
  2. semantic_roles    → Method   ──produces──►   Result          (rule-based)
                         Definition ──used_in──►  Method
                         Definition ──defines──►  Result/Observation  ← NEW
                         Dataset   ──evaluated_on►Result
  3. section_flow      → title    ──contains──►   paragraph/list  (layout)
  4. co_section        → Definition ──co_section──► Definition     ← NEW
                         (definitions in the same section linked for recall)

Changes from v1
───────────────
- Added `defines` edge: Definition → {Result, Observation, Method}
  This was the root cause of definition_retrieval nDCG = 0.000 in Hybrid —
  no KG edges encoded the "Definition" scholarly role, so expansion had zero signal.
- Added `co_section` edge: Definition → Definition within same section
  Enables recall expansion for definition queries.
- Edge weights now use role_confidence of source node alone (not average with target)
  when target role_confidence is 0 (e.g. figures with no text role).
- MAX_ROLE_EDGES_PER_NODE raised from 3 → 5 for Definition nodes so that
  a definition can link to multiple results/methods it applies to.
- MULTIMODAL_LINK_THRESHOLD lowered from 0.35 → 0.30 for better figure coverage.
- Sections are also assigned using "abstract" region_class (not just "title").
"""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional

import networkx as nx

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

# Roles that can be edge *sources* and their allowed *targets*
SEMANTIC_ROLE_EDGES: dict[str, list[tuple[str, str]]] = {
    #  source role      →  (target role,  relation label)
    "Method":     [("Result",      "produces"),
                   ("Observation", "produces")],
    "Definition": [("Method",      "used_in"),
                   ("Result",      "defines"),       # NEW — Definition clarifies a Result
                   ("Observation", "defines")],      # NEW — Definition scopes an Observation
    "Dataset":    [("Result",      "evaluated_on"),
                   ("Method",      "evaluated_by")],
    "Result":     [("Observation", "supports")],
}

# Max outgoing semantic-role edges per node
# Raised for Definition nodes specifically (see _add_semantic_role_edges).
MAX_ROLE_EDGES_PER_NODE         = 3
MAX_ROLE_EDGES_PER_DEFINITION   = 5   # definitions often apply to many targets

# Section titles regex  (matches "2. Related Work", "3.1 Method", "Abstract", …)
_SECTION_RE = re.compile(r"^\s*(\d+[\.\d]*\.?\s+)?[A-Z][\w\s\-&:]{2,60}$")

# Minimum s_link score to include a multimodal edge (lowered from 0.35)
MULTIMODAL_LINK_THRESHOLD = 0.30

# Roles excluded from being graph nodes useful for traversal
NOISE_ROLES   = {"N/A"}
NOISE_CLASSES = {"header", "footer"}


# ──────────────────────────────────────────────────────────────────────────────
# Node attribute helper
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class NodeAttrs:
    """
    Typed node attribute bag — prevents silent key-name drift between
    construction and retrieval code.
    """
    region_id:       str
    region_class:    str
    scholarly_role:  str
    page:            int
    text:            str            # truncated to TEXT_LIMIT chars
    confidence:      float
    role_confidence: float
    bbox_y0:         float          # top of region — used for proximity scoring
    section_id:      Optional[str]  # region_id of the nearest preceding title
    is_noise:        bool           # footer / header / N/A role

    TEXT_LIMIT: int = 400

    @classmethod
    def from_region(cls, r: dict[str, Any]) -> "NodeAttrs":
        text = (r.get("text_content") or "")
        bbox = r.get("bbox") or {}
        return cls(
            region_id       = r["region_id"],
            region_class    = r["region_class"].lower(),
            scholarly_role  = r.get("scholarly_role", "N/A"),
            page            = r["page_index"],
            text            = text[: cls.TEXT_LIMIT],
            confidence      = r.get("confidence", 1.0),
            role_confidence = r.get("role_confidence", 0.0),
            bbox_y0         = float(bbox.get("y0", 0.0)),
            section_id      = None,           # filled later by _assign_sections
            is_noise        = (
                r["region_class"].lower() in NOISE_CLASSES
                or r.get("scholarly_role", "N/A") in NOISE_ROLES
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "type":            self.region_class,
            "role":            self.scholarly_role,
            "page":            self.page,
            "text":            self.text,
            "confidence":      self.confidence,
            "role_confidence": self.role_confidence,
            "bbox_y0":         self.bbox_y0,
            "section_id":      self.section_id,
            "is_noise":        self.is_noise,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Main Agent
# ──────────────────────────────────────────────────────────────────────────────

class KnowledgeGraphAgent:
    """
    Step 5 — Knowledge Graph Construction Agent

    Input :  semantic.json  produced by SemanticUnderstandingAgent (Step 4)
             Schema: { "semantic_regions": [...], "multimodal_links": [...] }

    Output:  NetworkX DiGraph  G = (V, E)
             Nodes  — one per SemanticRegion
             Edges  — multimodal_links + semantic role-based + section-containment
                    + co_section (new: links Definitions within same section)

    Quick usage
    ───────────
        agent = KnowledgeGraphAgent()
        G = agent.build_from_json("output/semantic.json")
        agent.save_graphml("output/graph.graphml")
        agent.print_summary()
    """

    def __init__(self) -> None:
        self.graph: nx.DiGraph = nx.DiGraph()
        self._node_attrs: dict[str, NodeAttrs] = {}   # region_id → NodeAttrs

    # ──────────────────────────────────────────────────────────────────────────
    # PUBLIC API
    # ──────────────────────────────────────────────────────────────────────────

    def build_from_json(self, semantic_json_path: str) -> nx.DiGraph:
        """Full pipeline: load → build → return graph."""
        path = Path(semantic_json_path)
        logger.info("Loading %s …", path)
        data = json.loads(path.read_text(encoding="utf-8"))
        return self.build_from_dict(data)

    def build_from_dict(self, data: dict[str, Any]) -> nx.DiGraph:
        """
        Accept the raw dict produced by SemanticUnderstandingAgent.serialize().
        Idempotent — resets the graph on each call.
        """
        self.graph = nx.DiGraph()
        self._node_attrs = {}

        regions: list[dict] = data.get("semantic_regions", [])
        links:   list[dict] = data.get("multimodal_links",  [])

        if not regions:
            logger.warning("semantic_regions is empty — graph will be empty.")
            return self.graph

        self._add_nodes(regions)
        self._assign_sections(regions)        # annotates section_id on nodes
        self._add_multimodal_edges(links)
        self._add_semantic_role_edges()
        self._add_co_section_definition_edges()   # NEW
        self._add_section_containment_edges()

        logger.info(
            "Graph built: %d nodes, %d edges",
            self.graph.number_of_nodes(),
            self.graph.number_of_edges(),
        )
        return self.graph

    # ──────────────────────────────────────────────────────────────────────────
    # STEP 1 — NODES
    # ──────────────────────────────────────────────────────────────────────────

    def _add_nodes(self, regions: list[dict]) -> None:
        seen: set[str] = set()
        duplicates = 0

        for r in regions:
            nid = r["region_id"]
            if nid in seen:
                duplicates += 1
                logger.debug("Duplicate region_id skipped: %s", nid)
                continue
            seen.add(nid)

            attrs = NodeAttrs.from_region(r)
            self._node_attrs[nid] = attrs
            self.graph.add_node(nid, **attrs.to_dict())

        if duplicates:
            logger.warning("%d duplicate region_ids were skipped.", duplicates)

    # ──────────────────────────────────────────────────────────────────────────
    # STEP 2 — SECTION ASSIGNMENT  (layout-order traversal, O(n))
    # ──────────────────────────────────────────────────────────────────────────

    def _assign_sections(self, regions: list[dict]) -> None:
        """
        Walk regions in document order (page_index, bbox_y0).
        When a title or abstract region is encountered it becomes the active section.
        Each subsequent non-title region gets section_id = that title's region_id.
        """
        sorted_regions = sorted(
            regions,
            key=lambda r: (r["page_index"], (r.get("bbox") or {}).get("y0", 0.0)),
        )

        current_section: Optional[str] = None

        for r in sorted_regions:
            nid  = r["region_id"]
            rc   = r["region_class"].lower()
            text = r.get("text_content") or ""

            # Accept "abstract" region_class as a section boundary too
            if rc == "title" and _SECTION_RE.match(text):
                current_section = nid
            elif rc == "abstract":
                current_section = nid

            attrs = self._node_attrs.get(nid)
            if attrs:
                attrs.section_id = current_section
                # keep graph node attr in sync
                self.graph.nodes[nid]["section_id"] = current_section

    # ──────────────────────────────────────────────────────────────────────────
    # STEP 3 — MULTIMODAL EDGES  (caption ──refers_to──► figure/table)
    # ──────────────────────────────────────────────────────────────────────────

    def _add_multimodal_edges(self, links: list[dict]) -> None:
        added = skipped = 0
        for lnk in links:
            src = lnk.get("caption_id")
            tgt = lnk.get("target_id")
            score = lnk.get("s_link", 0.0)

            if not src or not tgt:
                continue
            if src not in self.graph or tgt not in self.graph:
                logger.debug("Multimodal link references unknown node(s): %s → %s", src, tgt)
                skipped += 1
                continue
            if score < MULTIMODAL_LINK_THRESHOLD:
                skipped += 1
                continue

            self.graph.add_edge(
                src, tgt,
                relation     = "refers_to",
                weight       = round(score, 4),
                s_ref        = lnk.get("s_ref", 0.0),
                s_emb        = lnk.get("s_emb", 0.0),
                matched_refs = lnk.get("matched_refs", []),
                edge_source  = "multimodal",
            )
            added += 1

        logger.info("Multimodal edges: %d added, %d skipped (below threshold / unknown).", added, skipped)

    # ──────────────────────────────────────────────────────────────────────────
    # STEP 4 — SEMANTIC ROLE EDGES  (O(n), not O(n²))
    # ──────────────────────────────────────────────────────────────────────────

    def _add_semantic_role_edges(self) -> None:
        """
        For each (source_role, target_role, relation) rule, collect all nodes
        of each role into buckets, then connect source→target within the same
        section. Falls back to same-page if section info is unavailable.

        Key improvement over v1:
        - Definition nodes get MAX_ROLE_EDGES_PER_DEFINITION (5) outgoing edges
          instead of 3, so they can point to multiple methods/results.
        - Edge weight uses source role_confidence if target role_confidence is 0
          (avoids always-zero weights on figure/table targets).
        - `defines` relation added: Definition → Result and Definition → Observation.

        Complexity: O(n)  —  bucket look-up, no nested loop over all pairs.
        """
        # Build role → [node_id] buckets  (skip noise nodes)
        role_buckets: dict[str, list[str]] = defaultdict(list)
        for nid, attrs in self._node_attrs.items():
            if not attrs.is_noise:
                role_buckets[attrs.scholarly_role].append(nid)

        total_added = 0

        for src_role, targets in SEMANTIC_ROLE_EDGES.items():
            src_nodes = role_buckets.get(src_role, [])
            if not src_nodes:
                continue

            # Definition nodes are allowed more outgoing edges
            max_edges = (
                MAX_ROLE_EDGES_PER_DEFINITION
                if src_role == "Definition"
                else MAX_ROLE_EDGES_PER_NODE
            )

            for tgt_role, relation in targets:
                tgt_nodes = role_buckets.get(tgt_role, [])
                if not tgt_nodes:
                    continue

                # Index target nodes by (section_id, page) for O(1) lookup
                tgt_by_section: dict[tuple, list[str]] = defaultdict(list)
                for tgt in tgt_nodes:
                    a = self._node_attrs[tgt]
                    key = (a.section_id or f"__page_{a.page}", a.page)
                    tgt_by_section[key].append(tgt)

                for src in src_nodes:
                    a_src = self._node_attrs[src]
                    key   = (a_src.section_id or f"__page_{a_src.page}", a_src.page)

                    candidates = tgt_by_section.get(key, [])

                    # If no same-section match, allow same-page cross-section
                    if not candidates:
                        candidates = [
                            nid for (sec, pg), nodes in tgt_by_section.items()
                            if pg == a_src.page
                            for nid in nodes
                        ]

                    added = 0
                    for tgt in candidates:
                        if added >= max_edges:
                            break
                        if self.graph.has_edge(src, tgt):
                            continue
                        a_tgt = self._node_attrs[tgt]

                        # Use source confidence when target has no role confidence
                        # (avoids always-zero weights e.g. when target is a figure)
                        src_rc = a_src.role_confidence
                        tgt_rc = a_tgt.role_confidence
                        if tgt_rc > 0:
                            weight = round((src_rc + tgt_rc) / 2, 4)
                        else:
                            weight = round(src_rc, 4)

                        self.graph.add_edge(
                            src, tgt,
                            relation    = relation,
                            weight      = max(weight, 0.1),   # floor so edge is never ignored
                            edge_source = "semantic_role",
                        )
                        added += 1
                        total_added += 1

        logger.info("Semantic role edges added: %d", total_added)

    # ──────────────────────────────────────────────────────────────────────────
    # STEP 4b — CO-SECTION DEFINITION EDGES  (NEW)
    # ──────────────────────────────────────────────────────────────────────────

    def _add_co_section_definition_edges(self) -> None:
        """
        Link Definition nodes that share the same section with a `co_section` edge.

        Why this matters:
        - When a definition query's FAISS seed hits one Definition node,
          graph expansion via `co_section` can recover sibling definitions
          that the embedding missed — improving recall for definition queries.
        - Only links within the same section to keep the graph sparse.
        - Max 3 co_section edges per Definition node to avoid hub explosion.
        """
        MAX_CO_SECTION = 3

        # Group Definition nodes by section
        section_defs: dict[str, list[str]] = defaultdict(list)
        for nid, attrs in self._node_attrs.items():
            if attrs.scholarly_role == "Definition" and not attrs.is_noise:
                sec_key = attrs.section_id or f"__page_{attrs.page}"
                section_defs[sec_key].append(nid)

        added = 0
        for sec_key, def_nodes in section_defs.items():
            if len(def_nodes) < 2:
                continue
            for i, src in enumerate(def_nodes):
                connected = 0
                for tgt in def_nodes:
                    if tgt == src:
                        continue
                    if connected >= MAX_CO_SECTION:
                        break
                    if self.graph.has_edge(src, tgt):
                        continue
                    src_rc = self._node_attrs[src].role_confidence
                    self.graph.add_edge(
                        src, tgt,
                        relation    = "co_section",
                        weight      = round(max(src_rc, 0.1), 4),
                        edge_source = "co_section",
                    )
                    connected += 1
                    added += 1

        logger.info("Co-section definition edges added: %d", added)

    # ──────────────────────────────────────────────────────────────────────────
    # STEP 5 — SECTION CONTAINMENT EDGES  (title ──contains──► body)
    # ──────────────────────────────────────────────────────────────────────────

    def _add_section_containment_edges(self) -> None:
        """
        Add lightweight title → content edges so the retrieval layer can
        fetch all content under a section by traversing 'contains' edges.
        """
        added = 0
        for nid, attrs in self._node_attrs.items():
            if attrs.section_id and attrs.region_class != "title":
                if not self.graph.has_edge(attrs.section_id, nid):
                    self.graph.add_edge(
                        attrs.section_id, nid,
                        relation    = "contains",
                        weight      = 1.0,
                        edge_source = "section",
                    )
                    added += 1
        logger.info("Section containment edges added: %d", added)

    # ──────────────────────────────────────────────────────────────────────────
    # RETRIEVAL HELPERS  (used by downstream hybrid scorer)
    # ──────────────────────────────────────────────────────────────────────────

    def neighbors_by_relation(
        self,
        node_id:  str,
        relation: str,
        max_hops: int = 1,
    ) -> list[str]:
        """
        BFS-expand from node_id following only edges with the given relation.
        Returns a list of reached node ids (excluding the start node).
        Useful for the retrieval layer: e.g. expand Method → Result.
        """
        if node_id not in self.graph:
            return []

        visited: set[str] = {node_id}
        frontier: list[str] = [node_id]

        for _ in range(max_hops):
            next_frontier: list[str] = []
            for n in frontier:
                for _, nbr, data in self.graph.out_edges(n, data=True):
                    if data.get("relation") == relation and nbr not in visited:
                        visited.add(nbr)
                        next_frontier.append(nbr)
            frontier = next_frontier

        visited.discard(node_id)
        return list(visited)

    def section_context(self, node_id: str) -> list[str]:
        """
        Return all sibling nodes that share the same section as node_id.
        Helps the LLM get surrounding context without requiring an embedding hit.
        """
        if node_id not in self._node_attrs:
            return []
        sec = self._node_attrs[node_id].section_id
        if not sec:
            return []
        return [
            nid for nid, attrs in self._node_attrs.items()
            if attrs.section_id == sec and nid != node_id
        ]

    def get_node_text(self, node_id: str) -> str:
        """Convenience accessor for the retrieval layer."""
        return self.graph.nodes[node_id].get("text", "") if node_id in self.graph else ""

    # ──────────────────────────────────────────────────────────────────────────
    # SERIALISATION  (GraphML + JSON; no pickle)
    # ──────────────────────────────────────────────────────────────────────────

    def save_graphml(self, path: str = "output/graph.graphml") -> None:
        """
        GraphML is portable (readable by Gephi, Cytoscape, networkx across versions).
        Lists (matched_refs) are serialised as JSON strings to satisfy GraphML's
        single-value constraint.
        """
        G = self._prepare_for_graphml()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        nx.write_graphml(G, path)
        logger.info("Graph saved to %s", path)

    def save_json(self, path: str = "output/graph.json") -> None:
        """
        JSON node-link format — easy to reload or send over HTTP.
        """
        data = nx.node_link_data(self.graph)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        logger.info("Graph saved to %s", path)

    @classmethod
    def load_json(cls, path: str) -> "KnowledgeGraphAgent":
        agent = cls()
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        agent.graph = nx.node_link_graph(data)
        # Rebuild _node_attrs from graph data for retrieval helpers
        for nid, attrs in agent.graph.nodes(data=True):
            agent._node_attrs[nid] = NodeAttrs(
                region_id       = nid,
                region_class    = attrs.get("type", ""),
                scholarly_role  = attrs.get("role", "N/A"),
                page            = attrs.get("page", 0),
                text            = attrs.get("text", ""),
                confidence      = attrs.get("confidence", 1.0),
                role_confidence = attrs.get("role_confidence", 0.0),
                bbox_y0         = attrs.get("bbox_y0", 0.0),
                section_id      = attrs.get("section_id"),
                is_noise        = attrs.get("is_noise", False),
            )
        return agent

    def _prepare_for_graphml(self) -> nx.DiGraph:
        """
        GraphML cannot serialise Python lists or None.
        Shallow-copy the graph and coerce values to strings where needed.
        """
        G = nx.DiGraph()
        for nid, attrs in self.graph.nodes(data=True):
            safe = {k: ("" if v is None else v) for k, v in attrs.items()}
            G.add_node(nid, **safe)
        for src, tgt, attrs in self.graph.edges(data=True):
            safe = {}
            for k, v in attrs.items():
                if isinstance(v, list):
                    safe[k] = json.dumps(v)
                elif v is None:
                    safe[k] = ""
                else:
                    safe[k] = v
            G.add_edge(src, tgt, **safe)
        return G

    # ──────────────────────────────────────────────────────────────────────────
    # DIAGNOSTICS
    # ──────────────────────────────────────────────────────────────────────────

    def print_summary(self) -> None:
        G = self.graph
        print("\n📊 KNOWLEDGE GRAPH SUMMARY")
        print("=" * 48)
        print(f"  Nodes : {G.number_of_nodes()}")
        print(f"  Edges : {G.number_of_edges()}")

        # Edge breakdown by source and relation
        by_source: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for _, _, d in G.edges(data=True):
            src = d.get("edge_source", "unknown")
            rel = d.get("relation",    "unknown")
            by_source[src][rel] += 1

        print("\n  Edge breakdown:")
        for source, rels in sorted(by_source.items()):
            for rel, count in sorted(rels.items()):
                print(f"    [{source}] {rel}: {count}")

        # Node role distribution (non-noise only)
        role_counts: dict[str, int] = defaultdict(int)
        for _, d in G.nodes(data=True):
            if not d.get("is_noise", False):
                role_counts[d.get("role", "N/A")] += 1

        print("\n  Scholarly role distribution (non-noise nodes):")
        for role, count in sorted(role_counts.items(), key=lambda x: -x[1]):
            print(f"    {role}: {count}")

        # Connectivity sanity check
        isolates = list(nx.isolates(G))
        print(f"\n  Isolated nodes (no edges): {len(isolates)}")
        if G.number_of_nodes() > 0:
            density = nx.density(G)
            print(f"  Graph density: {density:.5f}")
        print("=" * 48)

    def validate(self) -> list[str]:
        """
        Return a list of warning strings for common issues.
        Called by tests or a CI health-check before the retrieval layer starts.
        """
        warnings: list[str] = []
        G = self.graph

        if G.number_of_nodes() == 0:
            warnings.append("Graph has no nodes.")
            return warnings

        if G.number_of_edges() == 0:
            warnings.append("Graph has no edges.")

        isolate_count = len(list(nx.isolates(G)))
        if isolate_count > G.number_of_nodes() * 0.3:
            warnings.append(
                f"{isolate_count}/{G.number_of_nodes()} nodes are isolated "
                f"(>{30}% threshold). Edge rules may be too restrictive."
            )

        edge_count = G.number_of_edges()
        if edge_count > 1000:
            warnings.append(
                f"Graph has {edge_count} edges, which may be too dense for "
                "efficient retrieval traversal."
            )

        # Check for dangling edge endpoints
        for src, tgt in G.edges():
            if src not in G.nodes or tgt not in G.nodes:
                warnings.append(f"Dangling edge: {src} → {tgt}")

        return warnings