# pipeline.py
"""
End-to-End Intelligent Document Understanding Pipeline
========================================================
Wires all 8 agents into a single callable interface.

Step map
────────
  Step 1–3  Layout detection / extraction        (external — outputs layout.json)
  Step 4    SemanticUnderstandingAgent            → semantic.json
  Step 5    KnowledgeGraphAgent                   → graph (in-memory + graphml)
  Step 6    FAISSAgent                            → FAISS index (in-memory + disk)
  Step 7    HybridRetrievalEngine                 → HybridResult per query
  Step 8    AnswerGenerationAgent  (Llama 3 8B)   → GeneratedAnswer per query

Quick start
───────────
    # Build once from a semantic.json produced by Step 4:
    pipeline = DocumentPipeline()
    pipeline.build("output/semantic.json")

    # Ask questions:
    answer = pipeline.ask("How does KV recomputation improve RAG performance?")
    answer.print()

    # Stream answer tokens:
    for token in pipeline.ask_stream("What benchmarks were evaluated?"):
        print(token, end="", flush=True)

    # Save index to disk and reload later (skip rebuild):
    pipeline.save("output/index/")
    pipeline2 = DocumentPipeline.load("output/index/", "output/semantic.json")
    answer = pipeline2.ask("What is the proposed method?")
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Generator, Optional

from agents.answer_generation_agent import AnswerGenerationAgent, GeneratedAnswer
from agents.faiss_agent import FAISSAgent
from agents.hybrid_retrieval_engine import HybridResult, HybridRetrievalEngine
from agents.knowledge_graph_agent import KnowledgeGraphAgent

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)

# ──────────────────────────────────────────────────────────────────────────────
# Pipeline config dataclass
# ──────────────────────────────────────────────────────────────────────────────

class PipelineConfig:
    """
    Central place for all tunable knobs.
    Pass an instance to DocumentPipeline() to customise behaviour.
    """
    def __init__(
        self,
        # Retrieval
        faiss_top_k:    int   = 5,
        graph_expand_k: int   = 3,
        final_k:        int   = 10,
        alpha:          float = 0.6,    # α·FAISS + (1-α)·Graph
        # Generation
        ollama_model:   str   = "llama3",
        temperature:    float = 0.2,
        max_tokens:     int   = 1024,
        max_passages:   int   = 6,
        max_edges:      int   = 12,
        ollama_base_url:str   = "http://localhost:11434",
    ):
        self.faiss_top_k     = faiss_top_k
        self.graph_expand_k  = graph_expand_k
        self.final_k         = final_k
        self.alpha           = alpha
        self.ollama_model    = ollama_model
        self.temperature     = temperature
        self.max_tokens      = max_tokens
        self.max_passages    = max_passages
        self.max_edges       = max_edges
        self.ollama_base_url = ollama_base_url


# ──────────────────────────────────────────────────────────────────────────────
# Document Pipeline
# ──────────────────────────────────────────────────────────────────────────────

class DocumentPipeline:
    """
    End-to-end document understanding pipeline.
    Single entry point for building indexes and answering queries.
    """

    def __init__(self, config: Optional[PipelineConfig] = None) -> None:
        self.config   = config or PipelineConfig()
        self._built   = False

        # Agents — initialised in build()
        self._faiss:   Optional[FAISSAgent]             = None
        self._kg:      Optional[KnowledgeGraphAgent]    = None
        self._engine:  Optional[HybridRetrievalEngine]  = None
        self._llm:     Optional[AnswerGenerationAgent]  = None

    # ──────────────────────────────────────────────────────────────────────────
    # BUILD
    # ──────────────────────────────────────────────────────────────────────────

    def build(self, semantic_json_path: str) -> "DocumentPipeline":
        """
        Build the full pipeline from a semantic.json file.
        Must be called before ask().
        """
        path = Path(semantic_json_path)
        logger.info("═" * 56)
        logger.info("Building pipeline from: %s", path)
        logger.info("═" * 56)

        # ── Step 5: Knowledge Graph ───────────────────────────────────────────
        logger.info("[Step 5] Building knowledge graph …")
        self._kg = KnowledgeGraphAgent()
        self._kg.build_from_json(semantic_json_path)
        self._kg.print_summary()

        # ── Step 6: FAISS Index ───────────────────────────────────────────────
        logger.info("[Step 6] Building FAISS index …")
        self._faiss = FAISSAgent()
        self._faiss.build_from_json(semantic_json_path)
        self._faiss.print_summary()

        # ── Step 7: Hybrid Retrieval Engine ───────────────────────────────────
        logger.info("[Step 7] Initialising hybrid retrieval engine …")
        self._engine = HybridRetrievalEngine(
            faiss_agent = self._faiss,
            kg_agent    = self._kg,
            alpha       = self.config.alpha,
        )

        # ── Step 8: LLM ───────────────────────────────────────────────────────
        logger.info("[Step 8] Initialising answer generation agent …")
        self._llm = AnswerGenerationAgent(
            model       = self.config.ollama_model,
            temperature = self.config.temperature,
            max_tokens  = self.config.max_tokens,
            base_url    = self.config.ollama_base_url,
        )

        # Validation
        issues = self._engine.validate()
        for w in issues:
            logger.warning("Pipeline validation: %s", w)

        self._built = True
        logger.info("Pipeline ready.")
        return self

    # ──────────────────────────────────────────────────────────────────────────
    # SAVE / LOAD  (skip rebuild on repeated runs)
    # ──────────────────────────────────────────────────────────────────────────

    def save(self, index_dir: str) -> None:
        """
        Persist the FAISS index and KG to `index_dir`.
        Next run can call DocumentPipeline.load() to skip re-encoding.
        """
        self._check_built()
        out = Path(index_dir)
        out.mkdir(parents=True, exist_ok=True)

        self._faiss.save(str(out / "faiss"))
        self._kg.save_graphml(str(out / "graph.graphml"))
        self._kg.save_json(str(out / "graph.json"))

        logger.info("Pipeline saved to %s", index_dir)

    @classmethod
    def load(
        cls,
        index_dir:          str,
        semantic_json_path: str,
        config:             Optional[PipelineConfig] = None,
    ) -> "DocumentPipeline":
        """
        Reload a saved pipeline.  KG is rebuilt from semantic.json
        (fast, no encoding); FAISS index is loaded from disk.
        """
        cfg      = config or PipelineConfig()
        pipeline = cls(config=cfg)

        logger.info("Loading pipeline from %s …", index_dir)
        out = Path(index_dir)

        # FAISS from disk
        pipeline._faiss = FAISSAgent.load(str(out / "faiss"))

        # KG rebuilt from semantic.json (no ML, < 1s for typical papers)
        pipeline._kg = KnowledgeGraphAgent()
        pipeline._kg.build_from_json(semantic_json_path)

        # Retrieval engine + LLM
        pipeline._engine = HybridRetrievalEngine(
            faiss_agent = pipeline._faiss,
            kg_agent    = pipeline._kg,
            alpha       = cfg.alpha,
        )
        pipeline._llm = AnswerGenerationAgent(
            model       = cfg.ollama_model,
            temperature = cfg.temperature,
            max_tokens  = cfg.max_tokens,
            base_url    = cfg.ollama_base_url,
        )
        pipeline._built = True
        logger.info("Pipeline loaded and ready.")
        return pipeline

    # ──────────────────────────────────────────────────────────────────────────
    # QUERY API
    # ──────────────────────────────────────────────────────────────────────────

    def ask(
        self,
        query:           str,
        role_filter:     Optional[list[str]] = None,
        intent_override: Optional[str]       = None,
        temperature:     Optional[float]     = None,
        verbose:         bool                = False,
    ) -> GeneratedAnswer:
        """
        Full pipeline query: retrieve → generate → return structured answer.

        Parameters
        ──────────
        query           : natural language question about the document
        role_filter     : restrict retrieval to certain scholarly roles
                          e.g. ["Result", "Method"]
        intent_override : force a retrieval intent
                          one of: method | dataset | result | figure | general
        temperature     : override LLM temperature for this call
        verbose         : print the HybridResult retrieval debug table

        Returns
        ───────
        GeneratedAnswer — call .print() to display, or .to_dict() to serialise
        """
        self._check_built()

        # Step 7 — retrieve
        hybrid_result: HybridResult = self._engine.retrieve(
            query            = query,
            top_k            = self.config.faiss_top_k,
            expand_k         = self.config.graph_expand_k,
            final_k          = self.config.final_k,
            role_filter      = role_filter,
            intent_override  = intent_override,
        )

        if verbose:
            self._engine.print_result(hybrid_result)

        # Step 8 — generate
        return self._llm.generate(
            query         = query,
            hybrid_result = hybrid_result,
            max_passages  = self.config.max_passages,
            max_edges     = self.config.max_edges,
            temperature   = temperature,
        )

    def ask_stream(
        self,
        query:           str,
        role_filter:     Optional[list[str]] = None,
        intent_override: Optional[str]       = None,
    ) -> Generator[str, None, None]:
        """
        Streaming version of ask() — yields LLM tokens as they arrive.

        Example:
            for token in pipeline.ask_stream("What method was proposed?"):
                print(token, end="", flush=True)
        """
        self._check_built()

        hybrid_result = self._engine.retrieve(
            query           = query,
            top_k           = self.config.faiss_top_k,
            expand_k        = self.config.graph_expand_k,
            final_k         = self.config.final_k,
            role_filter     = role_filter,
            intent_override = intent_override,
        )
        yield from self._llm.generate_stream(
            query         = query,
            hybrid_result = hybrid_result,
            max_passages  = self.config.max_passages,
            max_edges     = self.config.max_edges,
        )

    def retrieve_only(
        self,
        query:           str,
        role_filter:     Optional[list[str]] = None,
        intent_override: Optional[str]       = None,
    ) -> HybridResult:
        """
        Run Steps 5-7 only — retrieval without LLM generation.
        Useful for inspection, testing, or when you want to pass
        the HybridResult to a different LLM.
        """
        self._check_built()
        return self._engine.retrieve(
            query           = query,
            top_k           = self.config.faiss_top_k,
            expand_k        = self.config.graph_expand_k,
            final_k         = self.config.final_k,
            role_filter     = role_filter,
            intent_override = intent_override,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # INTERNAL
    # ──────────────────────────────────────────────────────────────────────────

    def _check_built(self) -> None:
        if not self._built:
            raise RuntimeError(
                "Pipeline not built. Call pipeline.build('output/semantic.json') first."
            )


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Intelligent Document Understanding Pipeline"
    )
    parser.add_argument(
        "--semantic-json",
        default="output/semantic.json",
        help="Path to semantic.json from Step 4",
    )
    parser.add_argument(
        "--query", "-q",
        default="What is the proposed method and how does it work?",
        help="Question to ask about the document",
    )
    parser.add_argument(
        "--stream", action="store_true",
        help="Stream answer tokens to stdout",
    )
    parser.add_argument(
        "--save-index", default=None,
        help="Save FAISS + graph to this directory after building",
    )
    parser.add_argument(
        "--load-index", default=None,
        help="Load from this directory instead of rebuilding",
    )
    parser.add_argument(
        "--intent", default=None,
        choices=["method", "dataset", "result", "figure", "general"],
        help="Force retrieval intent (default: auto-detect)",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.6,
        help="Fusion weight α  (0=graph-only, 1=FAISS-only). Default: 0.6",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print retrieval debug table before the answer",
    )
    args = parser.parse_args()

    cfg = PipelineConfig(alpha=args.alpha)

    if args.load_index:
        pipeline = DocumentPipeline.load(args.load_index, args.semantic_json, config=cfg)
    else:
        pipeline = DocumentPipeline(config=cfg).build(args.semantic_json)
        if args.save_index:
            pipeline.save(args.save_index)

    print(f"\n❓ Query: {args.query}\n")

    if args.stream:
        print("📝 Answer (streaming):\n")
        for token in pipeline.ask_stream(args.query, intent_override=args.intent):
            print(token, end="", flush=True)
        print()
    else:
        answer = pipeline.ask(
            args.query,
            intent_override = args.intent,
            verbose         = args.verbose,
        )
        answer.print()