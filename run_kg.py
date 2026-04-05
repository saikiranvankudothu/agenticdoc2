from agents.knowledge_graph_agent import KnowledgeGraphAgent

if __name__ == "__main__":

    # ─────────────────────────────
    # Step 1: Initialize agent
    # ─────────────────────────────
    agent = KnowledgeGraphAgent()

    # ─────────────────────────────
    # Step 2: Build graph
    # ─────────────────────────────
    graph = agent.build_from_json(
        "output/paper/json/layout_semantic.json"
    )

    # ─────────────────────────────
    # Step 3: Print summary
    # ─────────────────────────────
    agent.print_summary()

    # ─────────────────────────────
    # Step 4: Validate graph (NEW 🔥)
    # ─────────────────────────────
    warnings = agent.validate()

    if warnings:
        print("\n⚠️ VALIDATION WARNINGS:")
        for w in warnings:
            print(" -", w)
    else:
        print("\n✅ Graph validation passed!")

    # ─────────────────────────────
    # Step 5: Save outputs
    # ─────────────────────────────
    agent.save_graphml("output/graph/graph.graphml")   # for Gephi / Cytoscape
    agent.save_json("output/graph/graph.json")         # for retrieval pipeline

    print("\n🎉 Knowledge Graph pipeline completed successfully!")