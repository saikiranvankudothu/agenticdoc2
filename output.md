(agenticdoc) C:\Users\Gopi\Downloads\Kiran\agenticdoc2>uv run python retrieval_evaluation.py --use-real-agents
2026-04-05 21:21:52,415 [INFO] **main** — Loaded 70 nodes from output\paper\json\layout_semantic.json
2026-04-05 21:21:52,415 [INFO] **main** — Loading real FAISSAgent + HybridRetrievalEngine …
2026-04-05 21:21:52,482 [INFO] faiss.loader — Loading faiss with AVX512 support.
2026-04-05 21:21:52,482 [INFO] faiss.loader — Could not load library with AVX512 support due to:
ModuleNotFoundError("No module named 'faiss.swigfaiss_avx512'")
2026-04-05 21:21:52,482 [INFO] faiss.loader — Loading faiss with AVX2 support.
2026-04-05 21:21:52,498 [INFO] faiss.loader — Successfully loaded faiss with AVX2 support.
2026-04-05 21:22:01,591 [INFO] agents.faiss_agent — Loading encoder 'all-MiniLM-L6-v2' …
2026-04-05 21:22:01,591 [INFO] sentence_transformers.SentenceTransformer — Load pretrained SentenceTransformer: all-MiniLM-L6-v2
C:\Users\Gopi\Downloads\Kiran\agenticdoc2\.venv\Lib\site-packages\huggingface_hub\file_download.py:949: FutureWarning: `resume_download` is deprecated and will be removed in version 1.0.0. Downloads always resume when possible. If you want to force a new download, use `force_download=True`.
warnings.warn(
2026-04-05 21:22:05,960 [INFO] agents.faiss_agent — Loading output\paper\json\layout_semantic.json …
2026-04-05 21:22:05,960 [INFO] agents.faiss_agent — Building FAISS index over 42 regions …
Batches: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:01<00:00, 1.76it/s]
2026-04-05 21:22:07,099 [INFO] agents.faiss_agent — Index type: IndexFlatIP (exact cosine, n=42)
2026-04-05 21:22:07,099 [INFO] agents.faiss_agent — FAISS index built in 1.14s — 42 vectors, dim=384, type=FlatIP
2026-04-05 21:22:07,099 [INFO] agents.knowledge_graph_agent — Loading output\paper\json\layout_semantic.json …
2026-04-05 21:22:07,101 [INFO] agents.knowledge_graph_agent — Multimodal edges: 12 added, 0 skipped (below threshold / unknown).
2026-04-05 21:22:07,103 [INFO] agents.knowledge_graph_agent — Semantic role edges added: 156
2026-04-05 21:22:07,103 [INFO] agents.knowledge_graph_agent — Co-section definition edges added: 30
2026-04-05 21:22:07,103 [INFO] agents.knowledge_graph_agent — Section containment edges added: 39
2026-04-05 21:22:07,103 [INFO] agents.knowledge_graph_agent — Graph built: 70 nodes, 237 edges
2026-04-05 21:22:07,103 [INFO] **main** — Evaluating 5 queries at K=[5, 10]
2026-04-05 21:22:07,105 [INFO] **main** — Starting evaluation: 5 queries × [5, 10] k-values
2026-04-05 21:22:07,105 [INFO] **main** — [1/5] Query: What method or approach does this paper propose?
Batches: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 52.42it/s]
2026-04-05 21:22:07,129 [INFO] agents.hybrid_retrieval_engine — Query: 'What method or approach does this paper propose?' → intent=method (conf=0.27)
Batches: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 60.50it/s]
2026-04-05 21:22:07,148 [INFO] agents.hybrid_retrieval_engine — FAISS seeds: 5 nodes ['a2963931', 'e1470500', '6ccea69e', '87213f5a', 'f09ece78']
2026-04-05 21:22:07,150 [INFO] agents.hybrid_retrieval_engine — Graph expansion: 4 neighbours via relations ['refers_to', 'used_in']
2026-04-05 21:22:07,150 [INFO] agents.hybrid_retrieval_engine — Retrieval complete: 9 final nodes (seeds=5, expanded=4)
2026-04-05 21:22:07,150 [INFO] **main** — [2/5] Query: What are the main results and performance metrics?
Batches: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 51.85it/s]
2026-04-05 21:22:07,171 [INFO] agents.hybrid_retrieval_engine — Query: 'What are the main results and performance metrics?' → intent=general (conf=0.09)
Batches: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 66.42it/s]
2026-04-05 21:22:07,192 [INFO] agents.hybrid_retrieval_engine — FAISS seeds: 5 nodes ['3b3ae1e6', 'cee15ba9', '0ffec164', 'e8d32730', '6f9c8d49']
2026-04-05 21:22:07,192 [INFO] agents.hybrid_retrieval_engine — Intent override: classifier='general' (conf=0.09) → role-fallback='result'
2026-04-05 21:22:07,192 [INFO] agents.hybrid_retrieval_engine — Graph expansion: 4 neighbours via relations ['←produces']
2026-04-05 21:22:07,193 [INFO] agents.hybrid_retrieval_engine — Retrieval complete: 9 final nodes (seeds=5, expanded=4)
2026-04-05 21:22:07,193 [INFO] **main** — [3/5] Query: What are the key definitions in this paper?
Batches: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 52.35it/s]
2026-04-05 21:22:07,214 [INFO] agents.hybrid_retrieval_engine — Query: 'What are the key definitions in this paper?' → intent=definition (conf=0.20)
Batches: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 53.57it/s]
2026-04-05 21:22:07,237 [INFO] agents.hybrid_retrieval_engine — FAISS seeds: 5 nodes ['a2963931', '8feaf570', '425202c6', 'e1470500', '6ccea69e']
2026-04-05 21:22:07,238 [INFO] agents.hybrid_retrieval_engine — Graph expansion: 3 neighbours via relations ['defines', 'used_in']
2026-04-05 21:22:07,238 [INFO] agents.hybrid_retrieval_engine — Retrieval complete: 8 final nodes (seeds=5, expanded=3)
2026-04-05 21:22:07,238 [INFO] **main** — [4/5] Query: What observations or findings does the paper report?
Batches: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 54.45it/s]
2026-04-05 21:22:07,261 [INFO] agents.hybrid_retrieval_engine — Query: 'What observations or findings does the paper report?' → intent=general (conf=0.00)  
Batches: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 54.38it/s]
2026-04-05 21:22:07,281 [INFO] agents.hybrid_retrieval_engine — FAISS seeds: 5 nodes ['a2963931', '8feaf570', '94db7a9a', 'ce94567d', '3b3ae1e6']
2026-04-05 21:22:07,281 [INFO] agents.hybrid_retrieval_engine — Intent override: classifier='general' (conf=0.00) → role-fallback='method'
2026-04-05 21:22:07,281 [INFO] agents.hybrid_retrieval_engine — Graph expansion: 4 neighbours via relations ['produces']
2026-04-05 21:22:07,281 [INFO] agents.hybrid_retrieval_engine — Retrieval complete: 9 final nodes (seeds=5, expanded=4)
2026-04-05 21:22:07,281 [INFO] **main** — [5/5] Query: What datasets or benchmarks were used?
Batches: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 69.32it/s]
2026-04-05 21:22:07,302 [INFO] agents.hybrid_retrieval_engine — Query: 'What datasets or benchmarks were used?' → intent=general (conf=0.00)
Batches: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 69.63it/s]
2026-04-05 21:22:07,321 [INFO] agents.hybrid_retrieval_engine — FAISS seeds: 5 nodes ['3b3ae1e6', 'e8d32730', '0ffec164', 'cee15ba9', '6f9c8d49']
2026-04-05 21:22:07,321 [INFO] agents.hybrid_retrieval_engine — Intent override: classifier='general' (conf=0.00) → role-fallback='result'
2026-04-05 21:22:07,321 [INFO] agents.hybrid_retrieval_engine — Graph expansion: 4 neighbours via relations ['refers_to', '←produces']
2026-04-05 21:22:07,321 [INFO] agents.hybrid_retrieval_engine — Retrieval complete: 9 final nodes (seeds=5, expanded=4)

========================================================================
RETRIEVAL EVALUATION REPORT — FAISS-only vs Hybrid (FAISS+KG)
========================================================================

── K = 5 ──────────────────────────────────────────────
Metric FAISS-only Hybrid Δ (Hybrid−FAISS)
──────────────────────────────────────────────────────────────────────
Precision@K 0.2800 0.2800 – +0.0000
Recall@K 0.2122 0.2122 – +0.0000
MRR 0.4167 0.4500 ▲ +0.0333
nDCG@K 0.5158 0.5158 – +0.0000
Hit Rate 0.8000 0.8000 – +0.0000
Latency (ms) 21.5000 21.7000 – +0.0000

── K = 10 ──────────────────────────────────────────────
Metric FAISS-only Hybrid Δ (Hybrid−FAISS)
──────────────────────────────────────────────────────────────────────
Precision@K 0.2000 0.2600 ▲ +0.0600
Recall@K 0.2387 0.3789 ▲ +0.1402
MRR 0.4167 0.4500 ▲ +0.0333
nDCG@K 0.5269 0.6236 ▲ +0.0967
Hit Rate 0.8000 1.0000 ▲ +0.2000
Latency (ms) 21.5000 21.7000 – +0.0000

── Per-query breakdown (K=5) ─────────────────────────────
dataset_retrieval nDCG FAISS=0.631 Hybrid=0.631 –+0.000
definition_retrieval nDCG FAISS=0.431 Hybrid=0.431 –+0.000
method_retrieval nDCG FAISS=0.571 Hybrid=0.571 –+0.000
observation_retrieval nDCG FAISS=0.000 Hybrid=0.000 –+0.000
result_retrieval nDCG FAISS=0.947 Hybrid=0.947 –+0.000

========================================================================

2026-04-05 21:22:07,329 [INFO] **main** — Report saved → output/eval/retrieval_report.json



below is the another output with fined-tuned code 

========================================================================
RETRIEVAL EVALUATION REPORT — FAISS-only vs Hybrid (FAISS+KG)
========================================================================

── K = 5 ──────────────────────────────────────────────
Metric FAISS-only Hybrid Δ (Hybrid−FAISS)
──────────────────────────────────────────────────────────────────────
Precision@K 0.3600 0.3600 – +0.0000
Recall@K 0.2233 0.2233 – +0.0000
MRR 0.5400 0.5733 ▲ +0.0333
nDCG@K 0.5739 0.5739 – +0.0000
Hit Rate 0.8000 0.8000 – +0.0000
Latency (ms) 14.6500 15.0500 – +0.0000

── K = 10 ──────────────────────────────────────────────
Metric FAISS-only Hybrid Δ (Hybrid−FAISS)
──────────────────────────────────────────────────────────────────────
Precision@K 0.2400 0.2200 ▼ -0.0200
Recall@K 0.2400 0.3233 ▲ +0.0833
MRR 0.5400 0.5733 ▲ +0.0333
nDCG@K 0.5719 0.6584 ▲ +0.0865
Hit Rate 0.8000 1.0000 ▲ +0.2000
Latency (ms) 14.6500 15.0500 – +0.0000

── Per-query breakdown (K=5) ─────────────────────────────
dataset_retrieval nDCG FAISS=0.631 Hybrid=0.631 –+0.000
definition_retrieval nDCG FAISS=0.387 Hybrid=0.387 –+0.000
method_retrieval nDCG FAISS=0.905 Hybrid=0.905 –+0.000
observation_retrieval nDCG FAISS=0.000 Hybrid=0.000 –+0.000
result_retrieval nDCG FAISS=0.947 Hybrid=0.947 –+0.000
