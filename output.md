
(agenticdoc) C:\Users\Gopi\Downloads\Kiran\agenticdoc2>uv run python pipeline.py --semantic-json output/paper/json/layout_semantic.json
The cache for model files in Transformers v4.22.0 has been updated. Migrating your old cache. This is a one-time only operation. You can interrupt this and resume the migration later on by calling `transformers.utils.move_cache()`.
0it [00:00, ?it/s]
2026-04-05 17:05:19,715 [INFO] __main__ — ════════════════════════════════════════════════════════
2026-04-05 17:05:19,716 [INFO] __main__ — Building pipeline from: output\paper\json\layout_semantic.json
2026-04-05 17:05:19,716 [INFO] __main__ — ════════════════════════════════════════════════════════
2026-04-05 17:05:19,716 [INFO] __main__ — [Step 5] Building knowledge graph …
2026-04-05 17:05:19,716 [INFO] agents.knowledge_graph_agent — Loading output\paper\json\layout_semantic.json …
2026-04-05 17:05:19,731 [INFO] agents.knowledge_graph_agent — Multimodal edges: 12 added, 0 skipped (below threshold / unknown).
2026-04-05 17:05:19,732 [INFO] agents.knowledge_graph_agent — Semantic role edges added: 116
2026-04-05 17:05:19,732 [INFO] agents.knowledge_graph_agent — Section containment edges added: 41
2026-04-05 17:05:19,732 [INFO] agents.knowledge_graph_agent — Graph built: 70 nodes, 169 edges

📊 KNOWLEDGE GRAPH SUMMARY
================================================
  Nodes : 70
  Edges : 169

  Edge breakdown:
    [multimodal] refers_to: 12
    [section] contains: 41
    [semantic_role] evaluated_by: 3
    [semantic_role] evaluated_on: 3
    [semantic_role] produces: 81
    [semantic_role] used_in: 29

  Scholarly role distribution (non-noise nodes):
    Method: 30
    Definition: 13
    Result: 7
    Observation: 4
    Dataset: 2

  Isolated nodes (no edges): 1
  Graph density: 0.03499
================================================
2026-04-05 17:05:19,734 [INFO] __main__ — [Step 6] Building FAISS index …
2026-04-05 17:05:19,734 [INFO] agents.faiss_agent — Loading encoder 'all-MiniLM-L6-v2' …
2026-04-05 17:05:19,738 [INFO] sentence_transformers.SentenceTransformer — Load pretrained SentenceTransformer: all-MiniLM-L6-v2
C:\Users\Gopi\Downloads\Kiran\agenticdoc2\.venv\Lib\site-packages\huggingface_hub\file_download.py:949: FutureWarning: `resume_download` is deprecated and will to force a new download, use `force_download=True`.
  warnings.warn(
2026-04-05 17:05:25,047 [INFO] agents.faiss_agent — Loading output\paper\json\layout_semantic.json …
2026-04-05 17:05:25,047 [INFO] agents.faiss_agent — Building FAISS index over 42 regions …
Batches: 100%|███████████████████████████████████████████████████████████████████████████| 2/2 [00:01<00:00,  1.47it/s]
2026-04-05 17:05:26,407 [INFO] agents.faiss_agent — Index type: IndexFlatIP (exact cosine, n=42)
2026-04-05 17:05:26,408 [INFO] agents.faiss_agent — FAISS index built in 1.35s — 42 vectors, dim=384, type=FlatIP

🗂  FAISS AGENT SUMMARY
================================================
  Index type     : FlatIP
  Total vectors  : 42
  Useful nodes   : 42  (non-noise)
  Noise nodes    : 0   (header/footer/N/A)
  Embedding dim  : 384

  Scholarly role distribution:
    Method: 23
    Definition: 10
    Observation: 4
    Result: 3
    Dataset: 2
================================================
2026-04-05 17:05:26,409 [INFO] __main__ — [Step 7] Initialising hybrid retrieval engine …
2026-04-05 17:05:26,411 [INFO] __main__ — [Step 8] Initialising answer generation agent …
2026-04-05 17:05:28,525 [INFO] __main__ — Pipeline ready.

❓ Query: What is the proposed method and how does it work?

2026-04-05 17:05:28,527 [INFO] agents.hybrid_retrieval_engine — Query: 'What is the proposed method and how does it work?'  →  intent=method (conf=0.27)
Batches: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  7.24it/s]
2026-04-05 17:05:28,671 [INFO] agents.hybrid_retrieval_engine — FAISS seeds: 5 nodes  ['e1470500', '5ab7c8c2', 'f09ece78', 'a2963931', 'e241fdf9']
2026-04-05 17:05:28,672 [INFO] agents.hybrid_retrieval_engine — Graph expansion: 7 neighbours via relations ['produces', 'used_in']
2026-04-05 17:05:28,672 [INFO] agents.hybrid_retrieval_engine — Retrieval complete: 10 final nodes (seeds=5, expanded=7)
2026-04-05 17:05:28,673 [INFO] agents.answer_generation_agent — Generating answer for: 'What is the proposed method and how does it work?'  (passages=6, edges=12, intent=method)
2026-04-05 17:05:44,702 [INFO] agents.answer_generation_agent — Ollama responded in 16.0s  (model=llama3, prompt_tokens=1015)

════════════════════════════════════════════════════════════
  QUERY   : What is the proposed method and how does it work?
  MODEL   : llama3   latency=16.0s
════════════════════════════════════════════════════════════
## Answer
The proposed method is chunk-wise prefilling, which leverages the freedom in chunk ordering to further align RoPE geometry with information flow in the decoding phase. During this process, each chunk is processed independently to compute its key-value (KV) cache.

## Evidence
1. [P5 | Role: Method | Page: 1 | Node: e241fdf9 | Score: 0.127] "During chunk-wise prefilling, each chunk is processed independently to compute its key–value (KV) cache."
2. [P1 | Role: Definition | Page: 1 | Node: e1470500 | Score: 0.139] "Let the input consist of N tokens, which we partition into K disjoint chunks { C 1 , . . . , C K } ."

## Graph Reasoning
- Paragraph[e241fdf9] → produces → Paragraph[a89ff685]: This shows that the chunk-wise prefilling method (P5) is used to compute the KV cache, which is then used in the observation about attention interactions (P6).
════════════════════════════════════════════════════════════