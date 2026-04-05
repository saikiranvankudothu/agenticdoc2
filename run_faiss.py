from agents.faiss_agent import FAISSAgent

if __name__ == "__main__":

    # ─────────────────────────────
    # Step 1: Initialize FAISS Agent
    # ─────────────────────────────
    agent = FAISSAgent()

    # ─────────────────────────────
    # Step 2: Build index from semantic output
    # ─────────────────────────────
    agent.build_from_json(
        "output/paper/json/layout_semantic.json"
    )

    # ─────────────────────────────
    # Step 3: Print summary
    # ─────────────────────────────
    agent.print_summary()

    # ─────────────────────────────
    # Step 4: Validate
    # ─────────────────────────────
    warnings = agent.validate()

    if warnings:
        print("\n⚠️ WARNINGS:")
        for w in warnings:
            print("-", w)
    else:
        print("\n✅ FAISS index looks good!")

    # ─────────────────────────────
    # Step 5: Save index
    # ─────────────────────────────
    agent.save("output/faiss/")

    print("\n💾 FAISS index saved!")

    # ─────────────────────────────
    # Step 6: Test search (IMPORTANT)
    # ─────────────────────────────
    query = "What is KV recomputation?"

    results = agent.search(query, top_k=5)

    print("\n🔍 SEARCH RESULTS:")
    for r in results:
        print(f"\nScore: {r.score}")
        print(f"Role : {r.role}")
        print(f"Text : {r.text[:150]}")