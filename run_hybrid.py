from agents.faiss_agent import FAISSAgent
from agents.knowledge_graph_agent import KnowledgeGraphAgent
from agents.hybrid_retrieval_engine import HybridRetrievalEngine


if __name__ == "__main__":

    # ─────────────────────────────
    # Step 1: Load FAISS
    # ─────────────────────────────
    faiss_agent = FAISSAgent.load("output/faiss/")

    # ─────────────────────────────
    # Step 2: Load Knowledge Graph
    # ─────────────────────────────
    kg_agent = KnowledgeGraphAgent.load_json("output/graph/graph.json")

    # ─────────────────────────────
    # Step 3: Initialize Hybrid Engine
    # ─────────────────────────────
    engine = HybridRetrievalEngine(
        faiss_agent,
        kg_agent,
        alpha=0.6   # same as your design
    )

    # ─────────────────────────────
    # Step 4: Validate system
    # ─────────────────────────────
    warnings = engine.validate()

    if warnings:
        print("\n⚠️ WARNINGS:")
        for w in warnings:
            print("-", w)
    else:
        print("\n✅ Hybrid engine ready!")

    # ─────────────────────────────
    # Step 5: Query
    # ─────────────────────────────
    query = "How does KV recomputation work?"

    result = engine.retrieve(query)

    # ─────────────────────────────
    # Step 6: Print results
    # ─────────────────────────────
    engine.print_result(result)

    # ─────────────────────────────
    # Step 7: Access for LLM
    # ─────────────────────────────
    print("\n🧠 Top Context for LLM:\n")
    for text in result.top_texts(k=5):
        print("-", text[:120])