# agents/answer_generation_agent.py
"""
Step 8 — Answer Generation Agent
==================================
Final step of the pipeline. Consumes a HybridResult from the
HybridRetrievalEngine and produces a structured, grounded answer
using Llama 3 8B served locally via Ollama.

Output format  (three sections, always)
────────────────────────────────────────
  ## Answer
  Direct response to the query in clear prose.

  ## Evidence
  Numbered list of passages that support the answer,
  each tagged with [Node, Role, Page].

  ## Graph Reasoning
  The relationship chain that connected retrieval nodes,
  e.g. Method → produces → Result → evaluated_on → Dataset.

Pipeline position
──────────────────
  HybridRetrievalEngine.retrieve(query)
            │
            ▼  HybridResult
  AnswerGenerationAgent.generate(query, hybrid_result)
            │
            ▼  GeneratedAnswer
  { answer_text, evidence, graph_reasoning, metadata }

Prompt architecture
────────────────────
  [SYSTEM]
    Role definition + strict output format contract.

  [CONTEXT PASSAGES]
    Up to MAX_CONTEXT_PASSAGES top-ranked nodes,
    each formatted as:
      [P1 | Role: Method | Page: 0 | Node: ec2685]
      <text snippet>

  [GRAPH RELATIONSHIPS]
    Edge explanations from HybridResult.edge_explanations():
      Method[ec2685] --produces--> Result[3b3ae1]
      Caption[a08f30] --refers_to--> Figure[0ddc07]

  [QUESTION]
    The original user query.

  [FORMAT INSTRUCTION]
    Explicit output skeleton the model must follow.

Ollama integration
──────────────────
  Uses the Ollama REST API  (http://localhost:11434/api/chat).
  Streaming is supported  —  .generate_stream() yields answer chunks.
  Falls back gracefully if Ollama is unreachable.

  Model: llama3  (maps to llama3:8b on a default Ollama install).
  Configurable at construction time.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Generator, Optional

import requests

# Local pipeline imports
from agents.hybrid_retrieval_engine import HybridResult, ScoredNode

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

OLLAMA_BASE_URL        = "http://localhost:11434"
OLLAMA_CHAT_ENDPOINT   = f"{OLLAMA_BASE_URL}/api/chat"
OLLAMA_TAGS_ENDPOINT   = f"{OLLAMA_BASE_URL}/api/tags"     # health-check
DEFAULT_MODEL          = "llama3"                           # llama3:8b alias
DEFAULT_TEMPERATURE    = 0.2     # low → factual, grounded, less hallucination
DEFAULT_TOP_P          = 0.9
DEFAULT_MAX_TOKENS     = 1024
DEFAULT_TIMEOUT        = 120     # seconds — 8B on CPU can be slow

MAX_CONTEXT_PASSAGES   = 6       # passages injected into the prompt
MAX_GRAPH_EDGES        = 12      # edge lines in [Graph Relationships] section
MAX_PASSAGE_CHARS      = 350     # truncate long passages before injecting
MAX_RETRIES            = 2       # retry once on transient Ollama errors

# Roles ordered by informativeness — used when trimming to MAX_CONTEXT_PASSAGES
ROLE_PRIORITY = ["Result", "Method", "Dataset", "Definition", "Observation", "N/A"]

# ──────────────────────────────────────────────────────────────────────────────
# Output data classes
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class EvidenceItem:
    """A single passage cited in the answer."""
    index:        int              # 1-based, matches [P{index}] in prompt
    node_id:      str
    text:         str
    role:         str
    page:         int
    final_score:  float
    origin:       str              # "faiss_seed" | "graph_expansion" | "both"

    def to_dict(self) -> dict[str, Any]:
        return {
            "index":       self.index,
            "node_id":     self.node_id,
            "text":        self.text,
            "role":        self.role,
            "page":        self.page,
            "final_score": self.final_score,
            "origin":      self.origin,
        }


@dataclass
class GraphReasoningStep:
    """One edge in the graph reasoning chain."""
    source_id:    str
    source_type:  str
    relation:     str
    target_id:    str
    target_type:  str

    def to_natural(self) -> str:
        return (
            f"{self.source_type}[{self.source_id[:8]}] "
            f"--{self.relation}--> "
            f"{self.target_type}[{self.target_id[:8]}]"
        )


@dataclass
class GeneratedAnswer:
    """
    Complete output of AnswerGenerationAgent.generate().

    Fields
    ──────
    answer_text      : the full raw response from Llama 3
    answer_section   : parsed "## Answer" section only
    evidence_section : parsed "## Evidence" section only
    reasoning_section: parsed "## Graph Reasoning" section only
    evidence         : structured EvidenceItem list (for downstream use)
    graph_steps      : structured GraphReasoningStep list
    query            : original question
    model            : model name used
    latency_s        : wall-clock time for the Ollama call
    prompt_tokens    : estimated (Ollama reports this in response)
    metadata         : intent, alpha, node counts, etc.
    """
    answer_text:       str
    answer_section:    str
    evidence_section:  str
    reasoning_section: str
    evidence:          list[EvidenceItem]        = field(default_factory=list)
    graph_steps:       list[GraphReasoningStep]  = field(default_factory=list)
    query:             str                       = ""
    model:             str                       = DEFAULT_MODEL
    latency_s:         float                     = 0.0
    prompt_tokens:     int                       = 0
    metadata:          dict[str, Any]            = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query":            self.query,
            "model":            self.model,
            "answer_text":      self.answer_text,
            "answer_section":   self.answer_section,
            "evidence_section": self.evidence_section,
            "reasoning_section":self.reasoning_section,
            "evidence":         [e.to_dict() for e in self.evidence],
            "graph_steps":      [s.to_natural() for s in self.graph_steps],
            "latency_s":        self.latency_s,
            "prompt_tokens":    self.prompt_tokens,
            "metadata":         self.metadata,
        }

    def print(self) -> None:
        """Pretty terminal output — useful during development."""
        bar = "═" * 60
        print(f"\n{bar}")
        print(f"  QUERY   : {self.query}")
        print(f"  MODEL   : {self.model}   latency={self.latency_s:.1f}s")
        print(bar)
        print(self.answer_text.strip())
        print(f"{bar}\n")


# ──────────────────────────────────────────────────────────────────────────────
# Prompt Builder
# ──────────────────────────────────────────────────────────────────────────────

class PromptBuilder:
    """
    Assembles the full prompt from a HybridResult.

    Prompt structure
    ────────────────
    SYSTEM  — role, rules, strict output format contract
    USER    — [Context Passages] + [Graph Relationships] + [Question] + [Format]
    """

    # ── System prompt ─────────────────────────────────────────────────────────
    SYSTEM_PROMPT = """You are a precise academic document assistant.
Your job is to answer questions about a research paper using ONLY the provided context.

Rules:
- Base every claim strictly on the provided passages. Do not invent or assume.
- If the context does not contain enough information, say so explicitly.
- Reference passages using their [P{n}] label when you cite them.
- Use the Graph Relationships section to explain how concepts connect.
- Be concise and factual. Academic tone.

You MUST respond in exactly this format — no deviations:

## Answer
<your direct answer here, referencing [P1], [P2] etc. where relevant>

## Evidence
<numbered list of the passages you actually used, one per line>
1. [P{n} | Role: {role} | Page: {page}] "{short quote or paraphrase}"
...

## Graph Reasoning
<one sentence per graph edge you found meaningful, describing the relationship>
- {SourceType} → {relation} → {TargetType}: <what this means for the answer>
...
"""

    def build(
        self,
        query:          str,
        hybrid_result:  HybridResult,
        max_passages:   int = MAX_CONTEXT_PASSAGES,
        max_edges:      int = MAX_GRAPH_EDGES,
    ) -> tuple[str, list[EvidenceItem], list[GraphReasoningStep]]:
        """
        Returns (user_prompt_text, evidence_items, graph_steps).
        evidence_items and graph_steps are returned separately so the
        GeneratedAnswer can carry them as structured data.
        """
        passages, evidence_items = self._select_passages(
            hybrid_result.ranked_results, max_passages
        )
        edge_lines, graph_steps = self._select_edges(
            hybrid_result, max_edges
        )

        user_prompt = self._assemble(query, passages, edge_lines)
        return user_prompt, evidence_items, graph_steps

    # ── Passage selection ─────────────────────────────────────────────────────

    def _select_passages(
        self,
        ranked: list[ScoredNode],
        max_n:  int,
    ) -> tuple[list[str], list[EvidenceItem]]:
        """
        Pick the top-max_n non-empty, non-noise passages.
        Re-sort by role priority first so the most informative roles
        appear at the top of the context window.
        """
        # Filter noise and empty text
        usable = [
            n for n in ranked
            if n.text.strip() and not (
                n.region_class in {"header", "footer"} or n.role == "N/A"
            )
        ]

        # Stable sort: final_score desc, then role priority
        role_order = {r: i for i, r in enumerate(ROLE_PRIORITY)}
        usable.sort(
            key=lambda n: (
                -n.final_score,
                role_order.get(n.role, len(ROLE_PRIORITY)),
            )
        )
        usable = usable[:max_n]

        passage_lines: list[str] = []
        evidence_items: list[EvidenceItem] = []

        for i, node in enumerate(usable, start=1):
            text = node.text[:MAX_PASSAGE_CHARS].strip()
            if len(node.text) > MAX_PASSAGE_CHARS:
                text += " …"

            label = (
                f"[P{i} | Role: {node.role} | Page: {node.page} "
                f"| Node: {node.node_id[:8]} | Score: {node.final_score:.3f}]"
            )
            passage_lines.append(f"{label}\n{text}")

            evidence_items.append(EvidenceItem(
                index       = i,
                node_id     = node.node_id,
                text        = text,
                role        = node.role,
                page        = node.page,
                final_score = node.final_score,
                origin      = node.origin,
            ))

        return passage_lines, evidence_items

    # ── Edge selection ────────────────────────────────────────────────────────

    def _select_edges(
        self,
        hybrid_result: HybridResult,
        max_n:         int,
    ) -> tuple[list[str], list[GraphReasoningStep]]:
        """
        Pull edge explanations from the subgraph.
        Prioritise high-value relations; trim to max_n.
        """
        PRIORITY_RELATIONS = {"produces", "evaluated_on", "refers_to",
                               "used_in", "evaluated_by"}

        edge_lines:  list[str]                = []
        graph_steps: list[GraphReasoningStep] = []

        sub = hybrid_result.subgraph
        edges = list(sub.edges(data=True))

        # Sort: priority relations first, then by edge weight desc
        def edge_sort_key(e):
            rel    = e[2].get("relation", "")
            weight = e[2].get("weight", 0.0)
            return (0 if rel in PRIORITY_RELATIONS else 1, -weight)

        edges.sort(key=edge_sort_key)

        for src, tgt, data in edges[:max_n]:
            rel      = data.get("relation", "?")
            src_type = sub.nodes[src].get("type", "node") if src in sub.nodes else "node"
            tgt_type = sub.nodes[tgt].get("type", "node") if tgt in sub.nodes else "node"

            line = (
                f"{src_type.capitalize()}[{src[:8]}] "
                f"--{rel}--> "
                f"{tgt_type.capitalize()}[{tgt[:8]}]"
            )
            edge_lines.append(line)
            graph_steps.append(GraphReasoningStep(
                source_id   = src,
                source_type = src_type,
                relation    = rel,
                target_id   = tgt,
                target_type = tgt_type,
            ))

        return edge_lines, graph_steps

    # ── Assembly ──────────────────────────────────────────────────────────────

    @staticmethod
    def _assemble(
        query:       str,
        passages:    list[str],
        edge_lines:  list[str],
    ) -> str:
        sections: list[str] = []

        # Context passages
        if passages:
            passage_block = "\n\n".join(passages)
            sections.append(f"[Context Passages]\n{passage_block}")
        else:
            sections.append("[Context Passages]\nNo relevant passages found.")

        # Graph relationships
        if edge_lines:
            edge_block = "\n".join(f"  {line}" for line in edge_lines)
            sections.append(f"[Graph Relationships]\n{edge_block}")

        # Question
        sections.append(f"[Question]\n{query}")

        # Format reminder
        sections.append(
            "[Required Format]\n"
            "Respond ONLY using the three-section format:\n"
            "## Answer\n## Evidence\n## Graph Reasoning"
        )

        return "\n\n".join(sections)


# ──────────────────────────────────────────────────────────────────────────────
# Ollama Client
# ──────────────────────────────────────────────────────────────────────────────

class OllamaClient:
    """
    Thin wrapper around the Ollama /api/chat REST endpoint.
    Handles retries, timeout, and health-checking.
    """

    def __init__(
        self,
        model:       str   = DEFAULT_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p:       float = DEFAULT_TOP_P,
        max_tokens:  int   = DEFAULT_MAX_TOKENS,
        timeout:     int   = DEFAULT_TIMEOUT,
        base_url:    str   = OLLAMA_BASE_URL,
    ) -> None:
        self.model       = model
        self.temperature = temperature
        self.top_p       = top_p
        self.max_tokens  = max_tokens
        self.timeout     = timeout
        self.chat_url    = f"{base_url}/api/chat"
        self.tags_url    = f"{base_url}/api/tags"

    def health_check(self) -> bool:
        """Return True if Ollama is reachable and the model is available."""
        try:
            resp = requests.get(self.tags_url, timeout=5)
            if resp.status_code != 200:
                return False
            models = [m["name"] for m in resp.json().get("models", [])]
            # Accept "llama3", "llama3:8b", "llama3:latest" etc.
            return any(self.model.split(":")[0] in m for m in models)
        except requests.RequestException:
            return False

    def chat(
        self,
        system_prompt: str,
        user_prompt:   str,
    ) -> tuple[str, int, float]:
        """
        Single-turn chat completion (non-streaming).

        Returns (response_text, prompt_eval_count, latency_seconds).
        Raises RuntimeError on repeated failure.
        """
        payload = {
            "model":  self.model,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "top_p":       self.top_p,
                "num_predict": self.max_tokens,
            },
            "messages": [
                {"role": "system",  "content": system_prompt},
                {"role": "user",    "content": user_prompt},
            ],
        }

        last_error: Optional[Exception] = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                t0   = time.perf_counter()
                resp = requests.post(
                    self.chat_url,
                    json    = payload,
                    timeout = self.timeout,
                )
                latency = time.perf_counter() - t0
                resp.raise_for_status()

                data          = resp.json()
                text          = data["message"]["content"]
                prompt_tokens = data.get("prompt_eval_count", 0)

                logger.info(
                    "Ollama responded in %.1fs  (model=%s, prompt_tokens=%d)",
                    latency, self.model, prompt_tokens,
                )
                return text, prompt_tokens, latency

            except (requests.RequestException, KeyError, json.JSONDecodeError) as e:
                last_error = e
                logger.warning(
                    "Ollama attempt %d/%d failed: %s",
                    attempt, MAX_RETRIES, e,
                )
                if attempt < MAX_RETRIES:
                    time.sleep(1.5 * attempt)   # brief back-off

        raise RuntimeError(
            f"Ollama call failed after {MAX_RETRIES} attempts. "
            f"Last error: {last_error}. "
            f"Is Ollama running?  Try: ollama serve"
        )

    def stream(
        self,
        system_prompt: str,
        user_prompt:   str,
    ) -> Generator[str, None, None]:
        """
        Streaming variant — yields text chunks as they arrive.
        Caller assembles the full string if needed.
        """
        payload = {
            "model":  self.model,
            "stream": True,
            "options": {
                "temperature": self.temperature,
                "top_p":       self.top_p,
                "num_predict": self.max_tokens,
            },
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
        }

        with requests.post(
            self.chat_url,
            json    = payload,
            stream  = True,
            timeout = self.timeout,
        ) as resp:
            resp.raise_for_status()
            for raw_line in resp.iter_lines():
                if not raw_line:
                    continue
                try:
                    chunk = json.loads(raw_line)
                    token = chunk.get("message", {}).get("content", "")
                    if token:
                        yield token
                    if chunk.get("done"):
                        break
                except json.JSONDecodeError:
                    continue


# ──────────────────────────────────────────────────────────────────────────────
# Response Parser
# ──────────────────────────────────────────────────────────────────────────────

class ResponseParser:
    """
    Splits the raw Llama 3 response into the three structured sections.
    Handles cases where the model doesn't perfectly follow the format.
    """

    _SECTION_RE = re.compile(
        r"##\s*(Answer|Evidence|Graph Reasoning)\s*\n",
        re.IGNORECASE,
    )

    @classmethod
    def parse(cls, raw: str) -> tuple[str, str, str]:
        """
        Returns (answer_section, evidence_section, reasoning_section).
        Each is a stripped string; empty string if section not found.
        """
        parts   = cls._SECTION_RE.split(raw)
        mapping: dict[str, str] = {}

        # parts alternates: [preamble, heading, content, heading, content, ...]
        i = 1
        while i + 1 < len(parts):
            heading = parts[i].strip().lower()
            content = parts[i + 1].strip()
            if "answer" in heading and "evidence" not in heading:
                mapping["answer"]    = content
            elif "evidence" in heading:
                mapping["evidence"]  = content
            elif "graph" in heading or "reasoning" in heading:
                mapping["reasoning"] = content
            i += 2

        answer    = mapping.get("answer",    "")
        evidence  = mapping.get("evidence",  "")
        reasoning = mapping.get("reasoning", "")

        # Fallback: if no sections found, treat entire response as answer
        if not answer and not evidence:
            answer = raw.strip()

        return answer, evidence, reasoning


# ──────────────────────────────────────────────────────────────────────────────
# Answer Generation Agent  (public interface)
# ──────────────────────────────────────────────────────────────────────────────

class AnswerGenerationAgent:
    """
    Step 8 — Answer Generation Agent.

    Consumes a HybridResult and produces a structured GeneratedAnswer
    using Llama 3 8B via Ollama.

    Usage
    ─────
        # Full pipeline:
        faiss_agent  = FAISSAgent()
        faiss_agent.build_from_json("output/semantic.json")

        kg_agent     = KnowledgeGraphAgent()
        kg_agent.build_from_json("output/semantic.json")

        engine       = HybridRetrievalEngine(faiss_agent, kg_agent)
        answer_agent = AnswerGenerationAgent()

        result  = engine.retrieve("How does KV recomputation improve RAG?")
        answer  = answer_agent.generate("How does KV recomputation improve RAG?", result)
        answer.print()

        # Streaming:
        for chunk in answer_agent.generate_stream("What benchmarks were used?", result):
            print(chunk, end="", flush=True)

        # Customise per-call:
        answer = answer_agent.generate(
            query          = "What accuracy was achieved?",
            hybrid_result  = result,
            max_passages   = 4,
            temperature    = 0.1,
        )
    """

    def __init__(
        self,
        model:       str   = DEFAULT_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p:       float = DEFAULT_TOP_P,
        max_tokens:  int   = DEFAULT_MAX_TOKENS,
        timeout:     int   = DEFAULT_TIMEOUT,
        base_url:    str   = OLLAMA_BASE_URL,
    ) -> None:
        self._client  = OllamaClient(
            model       = model,
            temperature = temperature,
            top_p       = top_p,
            max_tokens  = max_tokens,
            timeout     = timeout,
            base_url    = base_url,
        )
        self._builder = PromptBuilder()
        self._parser  = ResponseParser()

        # Warn early if Ollama is not reachable
        if not self._client.health_check():
            logger.warning(
                "Ollama health-check failed. Model '%s' may not be available. "
                "Run: ollama pull %s && ollama serve",
                model, model,
            )

    # ──────────────────────────────────────────────────────────────────────────
    # PRIMARY API
    # ──────────────────────────────────────────────────────────────────────────

    def generate(
        self,
        query:          str,
        hybrid_result:  HybridResult,
        max_passages:   int            = MAX_CONTEXT_PASSAGES,
        max_edges:      int            = MAX_GRAPH_EDGES,
        temperature:    Optional[float] = None,
    ) -> GeneratedAnswer:
        """
        Generate a structured answer for `query` given the HybridResult.

        Parameters
        ──────────
        query         : the original user question
        hybrid_result : output of HybridRetrievalEngine.retrieve()
        max_passages  : number of context passages to inject (default 6)
        max_edges     : number of graph edges to inject (default 12)
        temperature   : override temperature for this call only

        Returns
        ───────
        GeneratedAnswer with answer_text, parsed sections, evidence list,
        graph steps, latency, and full provenance metadata.
        """
        # Override temperature for this call if requested
        if temperature is not None:
            original_temp = self._client.temperature
            self._client.temperature = temperature

        try:
            # ── 1. Build prompt ───────────────────────────────────────────────
            user_prompt, evidence_items, graph_steps = self._builder.build(
                query         = query,
                hybrid_result = hybrid_result,
                max_passages  = max_passages,
                max_edges     = max_edges,
            )

            logger.info(
                "Generating answer for: '%s'  "
                "(passages=%d, edges=%d, intent=%s)",
                query[:70],
                len(evidence_items),
                len(graph_steps),
                hybrid_result.provenance.intent,
            )

            # ── 2. Call Ollama ────────────────────────────────────────────────
            raw_text, prompt_tokens, latency = self._client.chat(
                system_prompt = PromptBuilder.SYSTEM_PROMPT,
                user_prompt   = user_prompt,
            )

            # ── 3. Parse response ─────────────────────────────────────────────
            answer_sec, evidence_sec, reasoning_sec = self._parser.parse(raw_text)

            # ── 4. Build metadata ─────────────────────────────────────────────
            p = hybrid_result.provenance
            metadata = {
                "intent":          p.intent,
                "intent_conf":     p.intent_conf,
                "alpha":           p.alpha,
                "faiss_seeds":     len(p.faiss_seeds),
                "expanded_nodes":  len(p.expanded_nodes),
                "relations_used":  p.relations_used,
                "total_candidates":p.total_candidates,
                "passages_used":   len(evidence_items),
                "edges_used":      len(graph_steps),
            }

            return GeneratedAnswer(
                answer_text       = raw_text,
                answer_section    = answer_sec,
                evidence_section  = evidence_sec,
                reasoning_section = reasoning_sec,
                evidence          = evidence_items,
                graph_steps       = graph_steps,
                query             = query,
                model             = self._client.model,
                latency_s         = round(latency, 2),
                prompt_tokens     = prompt_tokens,
                metadata          = metadata,
            )

        finally:
            if temperature is not None:
                self._client.temperature = original_temp

    def generate_stream(
        self,
        query:         str,
        hybrid_result: HybridResult,
        max_passages:  int = MAX_CONTEXT_PASSAGES,
        max_edges:     int = MAX_GRAPH_EDGES,
    ) -> Generator[str, None, None]:
        """
        Streaming variant — yields text tokens as Ollama produces them.
        Useful for real-time display in a UI or CLI.

        Example:
            for token in agent.generate_stream(query, result):
                print(token, end="", flush=True)
        """
        user_prompt, _, _ = self._builder.build(
            query         = query,
            hybrid_result = hybrid_result,
            max_passages  = max_passages,
            max_edges     = max_edges,
        )
        yield from self._client.stream(
            system_prompt = PromptBuilder.SYSTEM_PROMPT,
            user_prompt   = user_prompt,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # DIAGNOSTICS
    # ──────────────────────────────────────────────────────────────────────────

    def health_check(self) -> bool:
        """Return True if Ollama is reachable and the model is loaded."""
        ok = self._client.health_check()
        if ok:
            logger.info("Ollama health-check passed. Model '%s' is available.", self._client.model)
        else:
            logger.error(
                "Ollama health-check failed. "
                "Run:  ollama pull %s  then  ollama serve",
                self._client.model,
            )
        return ok

    def inspect_prompt(
        self,
        query:         str,
        hybrid_result: HybridResult,
        max_passages:  int = MAX_CONTEXT_PASSAGES,
        max_edges:     int = MAX_GRAPH_EDGES,
    ) -> str:
        """
        Return the fully assembled user prompt without calling Ollama.
        Useful for debugging prompt content and token estimation.
        """
        user_prompt, evidence_items, graph_steps = self._builder.build(
            query, hybrid_result, max_passages, max_edges
        )
        sep = "─" * 60
        header = (
            f"{sep}\n"
            f"SYSTEM PROMPT ({len(PromptBuilder.SYSTEM_PROMPT)} chars)\n"
            f"{sep}\n"
            f"{PromptBuilder.SYSTEM_PROMPT}\n"
            f"{sep}\n"
            f"USER PROMPT ({len(user_prompt)} chars) "
            f"| passages={len(evidence_items)} | edges={len(graph_steps)}\n"
            f"{sep}\n"
        )
        return header + user_prompt