"""
app.py  —  AgenticDoc2 Streamlit UI
=====================================
Full-featured interface for the agentic PDF intelligence pipeline.

Run:
    streamlit run app.py

Requires the agenticdoc2 project to be installed / on PYTHONPATH:
    pip install streamlit plotly networkx pymupdf
"""

from __future__ import annotations

import io
import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

import streamlit as st

# ──────────────────────────────────────────────────────────────────────────────
# Page config  (MUST be first Streamlit call)
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AgenticDoc2",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global ────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Sidebar ───────────────────────────────────────────── */
[data-testid="stSidebar"] { background: #0f1117; border-right: 1px solid #1e2130; }
[data-testid="stSidebar"] * { color: #c9d1d9 !important; }
[data-testid="stSidebar"] .sidebar-logo {
    display: flex; align-items: center; gap: 10px;
    padding: 1rem 0 1.5rem; border-bottom: 1px solid #21262d; margin-bottom: 1rem;
}
[data-testid="stSidebar"] .sidebar-logo .logo-icon {
    width: 36px; height: 36px; border-radius: 8px;
    background: #1D9E75; display: flex; align-items: center;
    justify-content: center; font-size: 18px; flex-shrink: 0;
}
[data-testid="stSidebar"] .sidebar-logo .logo-text { font-size: 16px; font-weight: 600; color: #f0f6fc !important; }
[data-testid="stSidebar"] .sidebar-logo .logo-sub  { font-size: 11px; color: #8b949e !important; margin-top: 2px; }

/* ── Metric cards ──────────────────────────────────────── */
.metric-card {
    background: #161b22; border: 1px solid #21262d; border-radius: 10px;
    padding: 1rem 1.25rem; text-align: center;
}
.metric-card .metric-val { font-size: 28px; font-weight: 600; color: #1D9E75; }
.metric-card .metric-lbl { font-size: 12px; color: #8b949e; margin-top: 4px; }
.metric-card .metric-sub { font-size: 11px; color: #6e7681; margin-top: 2px; }

/* ── Section headers ───────────────────────────────────── */
.section-header {
    font-size: 11px; font-weight: 600; letter-spacing: 0.08em;
    text-transform: uppercase; color: #8b949e; margin: 1.5rem 0 0.75rem;
    padding-bottom: 6px; border-bottom: 1px solid #21262d;
}

/* ── Pipeline step cards ───────────────────────────────── */
.step-card {
    border-radius: 8px; padding: 10px 12px; margin-bottom: 6px;
    border: 1px solid #21262d; background: #161b22;
    display: flex; align-items: center; gap: 10px; font-size: 13px;
}
.step-card.done   { border-color: #1D9E75; background: rgba(29,158,117,0.08); }
.step-card.active { border-color: #EF9F27; background: rgba(239,159,39,0.08); animation: pulse 1.5s infinite; }
.step-card.idle   { opacity: 0.45; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.6} }
.step-icon { font-size: 18px; width: 28px; text-align: center; flex-shrink: 0; }
.step-name { font-weight: 500; color: #c9d1d9; flex: 1; }
.step-badge-done   { font-size: 10px; padding: 2px 8px; border-radius: 100px; background: rgba(29,158,117,0.2); color: #1D9E75; font-weight: 600; }
.step-badge-active { font-size: 10px; padding: 2px 8px; border-radius: 100px; background: rgba(239,159,39,0.2); color: #EF9F27; font-weight: 600; }
.step-badge-idle   { font-size: 10px; padding: 2px 8px; border-radius: 100px; background: #21262d; color: #6e7681; }

/* ── Region badges ─────────────────────────────────────── */
.badge { display:inline-block; font-size:10px; padding:2px 8px; border-radius:100px; font-weight:600; }
.badge-title     { background:#3d1f1f; color:#f97583; }
.badge-paragraph { background:#1f2d3d; color:#79b8ff; }
.badge-figure    { background:#1f3d2a; color:#85e89d; }
.badge-caption   { background:#3d2e1f; color:#ffab70; }
.badge-equation  { background:#2d1f3d; color:#c9a0ff; }
.badge-table     { background:#1f3a3d; color:#56d364; }
.badge-reference { background:#21262d; color:#8b949e; }
.badge-abstract  { background:#2d1f2d; color:#d2a8ff; }
.badge-method    { background:#1f2d3d; color:#58a6ff; }
.badge-result    { background:#1f3d2a; color:#85e89d; }
.badge-dataset   { background:#3d2e1f; color:#ffab70; }
.badge-definition{ background:#2d1f3d; color:#d2a8ff; }
.badge-observation { background:#1f3a3d; color:#56d364; }

/* ── Q&A bubbles ───────────────────────────────────────── */
.qbubble {
    background: #21262d; border-radius: 12px 12px 4px 12px;
    padding: 10px 14px; margin-bottom: 8px; font-size: 14px; color: #c9d1d9;
    border: 1px solid #30363d;
}
.abubble {
    background: rgba(29,158,117,0.1); border-radius: 12px 12px 12px 4px;
    padding: 10px 14px; margin-bottom: 8px; font-size: 14px; color: #c9d1d9;
    border: 1px solid rgba(29,158,117,0.3); border-left: 3px solid #1D9E75;
}
.source-chip {
    display:inline-block; font-size:10px; padding:2px 8px;
    border-radius:100px; background:#21262d; border:1px solid #30363d;
    color:#8b949e; margin:3px 2px 0; font-family:'JetBrains Mono',monospace;
}

/* ── Info boxes ────────────────────────────────────────── */
.info-box {
    background: rgba(88,166,255,0.06); border: 1px solid rgba(88,166,255,0.2);
    border-radius: 8px; padding: 12px 16px; font-size: 13px; color: #8b949e;
    margin-bottom: 12px;
}
.warn-box {
    background: rgba(239,159,39,0.06); border: 1px solid rgba(239,159,39,0.25);
    border-radius: 8px; padding: 12px 16px; font-size: 13px; color: #e3b341;
    margin-bottom: 12px;
}

/* ── KG summary table ──────────────────────────────────── */
.kg-table { width:100%; border-collapse:collapse; font-size:13px; }
.kg-table th { color:#8b949e; font-weight:600; font-size:11px; text-transform:uppercase; letter-spacing:0.06em; padding:6px 8px; border-bottom:1px solid #21262d; text-align:left; }
.kg-table td { padding:6px 8px; color:#c9d1d9; border-bottom:1px solid #161b22; }
.kg-table tr:last-child td { border-bottom:none; }

/* ── Scrollable containers ─────────────────────────────── */
.scroll-box { max-height: 380px; overflow-y: auto; padding-right: 4px; }
.scroll-box::-webkit-scrollbar { width: 4px; }
.scroll-box::-webkit-scrollbar-track { background: #161b22; }
.scroll-box::-webkit-scrollbar-thumb { background: #30363d; border-radius: 2px; }

/* ── Evidence items ────────────────────────────────────── */
.evidence-item {
    background:#161b22; border:1px solid #21262d; border-radius:8px;
    padding:10px 14px; margin-bottom:8px; font-size:13px;
}
.evidence-item .ev-meta { font-size:11px; color:#8b949e; margin-bottom:4px; font-family:'JetBrains Mono',monospace; }
.evidence-item .ev-text { color:#c9d1d9; line-height:1.6; }
.evidence-item .ev-score { float:right; font-size:11px; color:#1D9E75; font-weight:600; }

/* ── Graph reasoning ───────────────────────────────────── */
.graph-step {
    background:#161b22; border-left:3px solid #7F77DD;
    padding:8px 12px; margin-bottom:6px; border-radius:0 6px 6px 0;
    font-size:12px; color:#c9d1d9; font-family:'JetBrains Mono',monospace;
}

/* ── Streamlit overrides ───────────────────────────────── */
.stButton > button {
    background: transparent; border: 1px solid #30363d; color: #c9d1d9;
    border-radius: 6px; font-size: 13px; padding: 6px 16px;
    transition: all 0.15s;
}
.stButton > button:hover { background: #21262d; border-color: #8b949e; }
div[data-testid="stForm"] { border: none; padding: 0; }
.stTextInput > div > div > input {
    background: #161b22 !important; border: 1px solid #30363d !important;
    color: #c9d1d9 !important; border-radius: 8px !important;
}
div[data-baseweb="select"] > div { background: #161b22 !important; border-color: #30363d !important; }
.stSlider > div > div > div { background: #1D9E75 !important; }
[data-testid="stExpander"] { border: 1px solid #21262d !important; background: #161b22 !important; border-radius: 8px !important; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Session state init
# ──────────────────────────────────────────────────────────────────────────────
def _init_state():
    defaults = {
        "pipeline":        None,
        "pipeline_built":  False,
        "semantic_path":   None,
        "doc_name":        None,
        "extraction_json": None,
        "layout_json":     None,
        "semantic_json":   None,
        "kg_summary":      None,
        "faiss_summary":   None,
        "chat_history":    [],
        "page":            "upload",
        "ollama_model":    "llama3",
        "ollama_url":      "http://localhost:11434",
        "faiss_top_k":     5,
        "graph_expand_k":  3,
        "final_k":         10,
        "alpha":           0.6,
        "temperature":     0.2,
        "retrieval_mode":  "hybrid",
        "role_filter":     [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
REGION_COLORS = {
    "title":     "#f97583",
    "paragraph": "#79b8ff",
    "figure":    "#85e89d",
    "caption":   "#ffab70",
    "equation":  "#c9a0ff",
    "table":     "#56d364",
    "list":      "#ffd33d",
    "header":    "#8b949e",
    "footer":    "#8b949e",
    "reference": "#8b949e",
    "abstract":  "#d2a8ff",
}

ROLE_COLORS = {
    "Method":      ("badge-method",      "Method"),
    "Result":      ("badge-result",      "Result"),
    "Dataset":     ("badge-dataset",     "Dataset"),
    "Definition":  ("badge-definition",  "Definition"),
    "Observation": ("badge-observation", "Observation"),
    "N/A":         ("badge-reference",   "N/A"),
}


def badge(cls_name: str, label: Optional[str] = None) -> str:
    label = label or cls_name
    css = f"badge-{cls_name.lower()}" if f"badge-{cls_name.lower()}" in [
        "badge-title","badge-paragraph","badge-figure","badge-caption",
        "badge-equation","badge-table","badge-reference","badge-abstract",
        "badge-method","badge-result","badge-dataset","badge-definition","badge-observation",
    ] else "badge-reference"
    return f'<span class="badge {css}">{label}</span>'


def load_json_safe(path: str) -> Optional[dict]:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def format_elapsed(seconds: float) -> str:
    return f"{seconds:.1f}s" if seconds < 60 else f"{int(seconds//60)}m {int(seconds%60)}s"


def try_import_pipeline():
    """Attempt to import DocumentPipeline from the project."""
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from pipeline import DocumentPipeline, PipelineConfig  # noqa: F401
        return DocumentPipeline, PipelineConfig
    except ImportError:
        return None, None


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-logo">
            <div class="logo-icon">📄</div>
            <div>
                <div class="logo-text">AgenticDoc2</div>
                <div class="logo-sub">Agentic PDF Intelligence</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Navigation
        pages = [
            ("📤", "upload",    "Upload PDF"),
            ("⚙️",  "pipeline",  "Pipeline"),
            ("🗺",  "layout",    "Layout"),
            ("🧠",  "semantic",  "Semantics"),
            ("🕸",  "graph",     "Knowledge Graph"),
            ("💬",  "qa",        "Q&A"),
            ("⚙️",  "settings",  "Settings"),
        ]
        st.markdown('<div class="section-header">Navigation</div>', unsafe_allow_html=True)
        for icon, key, label in pages:
            is_active = st.session_state.page == key
            if st.button(
                f"{icon}  {label}",
                key=f"nav_{key}",
                use_container_width=True,
                type="primary" if is_active else "secondary",
            ):
                st.session_state.page = key
                st.rerun()

        # Pipeline status
        st.markdown('<div class="section-header" style="margin-top:2rem;">Pipeline Status</div>', unsafe_allow_html=True)

        steps = [
            ("📄", "Extraction",   bool(st.session_state.extraction_json)),
            ("🗺",  "Layout",       bool(st.session_state.layout_json)),
            ("🔗", "Alignment",    bool(st.session_state.layout_json)),
            ("🧠", "Semantics",    bool(st.session_state.semantic_json)),
            ("⚡", "FAISS",        st.session_state.pipeline_built),
            ("🕸",  "Knowledge Graph", st.session_state.pipeline_built),
            ("🔍", "Hybrid RAG",   st.session_state.pipeline_built),
        ]
        for icon, name, done in steps:
            css = "done" if done else "idle"
            badge_html = (
                '<span class="step-badge-done">done</span>' if done
                else '<span class="step-badge-idle">waiting</span>'
            )
            st.markdown(f"""
            <div class="step-card {css}">
                <span class="step-icon">{icon}</span>
                <span class="step-name">{name}</span>
                {badge_html}
            </div>
            """, unsafe_allow_html=True)

        # Doc info
        if st.session_state.doc_name:
            st.markdown('<div class="section-header" style="margin-top:1.5rem;">Active Document</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div style="font-size:13px;color:#c9d1d9;background:#161b22;border:1px solid #21262d;
                        border-radius:8px;padding:10px 12px;">
                📄 <b>{st.session_state.doc_name}</b>
            </div>
            """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Page: Upload
# ──────────────────────────────────────────────────────────────────────────────
def page_upload():
    st.markdown("## 📤 Upload Document")
    st.markdown("Upload a PDF or load existing pipeline outputs to get started.")

    tab1, tab2 = st.tabs(["Upload PDF", "Load Existing Outputs"])

    with tab1:
        st.markdown('<div class="section-header">PDF Upload</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Drop your academic PDF here",
            type=["pdf"],
            help="Upload a research paper, technical document, or report",
        )

        if uploaded:
            tmp_dir = Path(tempfile.mkdtemp())
            pdf_path = tmp_dir / uploaded.name
            pdf_path.write_bytes(uploaded.read())
            st.session_state.doc_name = uploaded.name

            st.markdown(f"""
            <div class="info-box">
                ✅ <b>{uploaded.name}</b> uploaded — {uploaded.size // 1024} KB
            </div>
            """, unsafe_allow_html=True)

            col1, col2, col3 = st.columns(3)
            output_dir = col1.text_input("Output directory", value="output")
            ocr_threshold = col2.number_input("OCR threshold (chars)", value=50, min_value=0)
            dpi = col3.number_input("Render DPI", value=150, min_value=72)
            backend = st.selectbox("Layout backend", ["auto", "yolo", "heuristic", "layoutparser", "dit"])

            if st.button("🚀 Run Full Pipeline", type="primary", use_container_width=True):
                _run_full_pipeline(str(pdf_path), output_dir, ocr_threshold, dpi, backend)

            st.markdown("---")
            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("Step 1: Extract only", use_container_width=True):
                    _run_extraction(str(pdf_path), output_dir, ocr_threshold, dpi)
            with col_b:
                if st.button("Step 2: Layout only (needs extraction)", use_container_width=True):
                    st.warning("Run extraction first, then load the extraction.json below.")

    with tab2:
        st.markdown('<div class="section-header">Load Existing JSON Outputs</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-box">
            If you've already run the pipeline CLI tools, point to the JSON outputs here
            to load them directly into the UI.
        </div>
        """, unsafe_allow_html=True)

        extraction_path = st.text_input("extraction.json path", placeholder="output/paper/json/extraction.json")
        layout_path     = st.text_input("layout.json path",     placeholder="output/paper/json/layout.json")
        semantic_path   = st.text_input("semantic.json path",   placeholder="output/paper/json/layout_semantic.json")

        if st.button("Load JSON outputs", use_container_width=True):
            loaded = 0
            if extraction_path and Path(extraction_path).exists():
                st.session_state.extraction_json = load_json_safe(extraction_path)
                loaded += 1
            if layout_path and Path(layout_path).exists():
                st.session_state.layout_json = load_json_safe(layout_path)
                loaded += 1
            if semantic_path and Path(semantic_path).exists():
                st.session_state.semantic_json  = load_json_safe(semantic_path)
                st.session_state.semantic_path  = semantic_path
                loaded += 1

            if loaded:
                st.success(f"Loaded {loaded} JSON file(s) successfully.")
                if not st.session_state.doc_name and st.session_state.extraction_json:
                    st.session_state.doc_name = st.session_state.extraction_json.get("doc_id", "document")
            else:
                st.error("No valid files found. Check the paths above.")

        st.markdown("---")
        st.markdown('<div class="section-header">Build RAG Pipeline (Steps 5–8)</div>', unsafe_allow_html=True)
        sem_p = st.text_input(
            "Semantic JSON for pipeline build",
            value=st.session_state.semantic_path or "",
            placeholder="output/paper/json/layout_semantic.json",
        )
        if st.button("Build KG + FAISS + Hybrid RAG", type="primary", use_container_width=True):
            if sem_p and Path(sem_p).exists():
                _build_pipeline(sem_p)
            else:
                st.error("Semantic JSON path is required and must exist.")


# ──────────────────────────────────────────────────────────────────────────────
# Page: Pipeline
# ──────────────────────────────────────────────────────────────────────────────
def page_pipeline():
    st.markdown("## ⚙️ Pipeline")

    # Metrics row
    ext  = st.session_state.extraction_json
    lay  = st.session_state.layout_json
    sem  = st.session_state.semantic_json
    kg   = st.session_state.kg_summary
    faiss = st.session_state.faiss_summary

    total_pages   = ext["total_pages"]  if ext  else "—"
    total_regions = sum(len(pg["regions"]) for pg in lay["pages"]) if lay else "—"
    sem_regions   = len(sem.get("semantic_regions", [])) if sem else "—"
    kg_nodes      = kg["nodes"] if kg else "—"
    kg_edges      = kg["edges"] if kg else "—"
    faiss_vecs    = faiss["vectors"] if faiss else "—"

    cols = st.columns(6)
    cards = [
        ("Pages",    total_pages,   "document pages"),
        ("Regions",  total_regions, "layout regions"),
        ("Semantic", sem_regions,   "embedded regions"),
        ("KG nodes", kg_nodes,      "entities"),
        ("KG edges", kg_edges,      "relations"),
        ("FAISS",    faiss_vecs,    "index vectors"),
    ]
    for col, (lbl, val, sub) in zip(cols, cards):
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-val">{val}</div>
            <div class="metric-lbl">{lbl}</div>
            <div class="metric-sub">{sub}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Step details
    st.markdown('<div class="section-header">Agent Steps</div>', unsafe_allow_html=True)

    step_info = [
        ("📄", "Step 1 — Document Extraction",
         "Uses PyMuPDF (fitz) to extract text blocks, figure crops, and style metadata per page. "
         "Falls back to Tesseract OCR for low-text pages. Output: `extraction.json`.",
         bool(ext)),
        ("🗺", "Step 2 — Layout Detection",
         "Three-tier hybrid: YOLOv8 (primary), LayoutParser/DiT (ML), Heuristic (fallback). "
         "Classifies regions as title, paragraph, figure, table, equation, caption, etc. "
         "Output: `layout.json`.",
         bool(lay)),
        ("🔗", "Step 3 — Text-Layout Alignment",
         "Matches extracted text blocks to layout regions via IoU scoring. "
         "Each region gets assigned source block IDs. Output: `alignment.json`.",
         bool(lay)),
        ("🧠", "Step 4 — Semantic Understanding",
         "Generates 384-dim embeddings (all-MiniLM-L6-v2). Classifies scholarly roles "
         "(Method/Result/Dataset/Definition/Observation). Links figure-caption pairs via "
         "s_link = 0.7·s_ref + 0.3·s_emb. Output: `layout_semantic.json`.",
         bool(sem)),
        ("⚡", "Step 5 — FAISS Vector Index",
         "Builds a FlatIP FAISS index over all semantic embeddings for dense nearest-neighbour "
         "retrieval. Skips noise nodes (header/footer). Saved to `output/faiss/`.",
         st.session_state.pipeline_built),
        ("🕸", "Step 6 — Knowledge Graph",
         "Extracts entities and relations from semantic regions. Adds multimodal edges "
         "(refers_to), section containment edges (contains), and semantic role edges "
         "(produces/used_in/evaluated_on). Saved as `graph.graphml` + `graph.json`.",
         st.session_state.pipeline_built),
        ("🔍", "Step 7 — Hybrid Retrieval + Generation",
         "HybridRetrievalEngine: α·FAISS + (1-α)·Graph expansion. AnswerGenerationAgent "
         "formats context and calls Ollama (llama3) with structured prompt. "
         "Returns answer + evidence + graph reasoning.",
         st.session_state.pipeline_built),
    ]

    for icon, title, desc, done in step_info:
        status = "done" if done else "idle"
        badge_html = (
            '<span class="step-badge-done">✓ complete</span>' if done
            else '<span class="step-badge-idle">waiting</span>'
        )
        with st.expander(f"{icon}  {title}"):
            st.markdown(f"""
            <div style="display:flex;justify-content:flex-end;margin-bottom:8px;">{badge_html}</div>
            <div style="font-size:13px;color:#c9d1d9;line-height:1.7;">{desc}</div>
            """, unsafe_allow_html=True)

    # KG summary
    if kg:
        st.markdown('<div class="section-header">Knowledge Graph Summary</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <table class="kg-table">
              <thead><tr><th>Metric</th><th>Value</th></tr></thead>
              <tbody>
                <tr><td>Total nodes</td><td><b>{kg['nodes']}</b></td></tr>
                <tr><td>Total edges</td><td><b>{kg['edges']}</b></td></tr>
                <tr><td>Graph density</td><td>{kg.get('density','—')}</td></tr>
                <tr><td>Isolated nodes</td><td>{kg.get('isolated',0)}</td></tr>
              </tbody>
            </table>
            """, unsafe_allow_html=True)
        with col2:
            if "role_dist" in kg:
                st.markdown("**Scholarly role distribution**")
                for role, count in kg["role_dist"].items():
                    pct = int(count / max(kg["nodes"], 1) * 100)
                    css, lbl = ROLE_COLORS.get(role, ("badge-reference", role))
                    st.markdown(f"""
                    <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">
                        <span class="badge {css}">{lbl}</span>
                        <div style="flex:1;height:6px;border-radius:3px;background:#21262d;overflow:hidden;">
                            <div style="height:100%;width:{pct}%;background:#1D9E75;border-radius:3px;"></div>
                        </div>
                        <span style="font-size:12px;color:#8b949e;width:28px;text-align:right;">{count}</span>
                    </div>
                    """, unsafe_allow_html=True)

    # FAISS summary
    if faiss:
        st.markdown('<div class="section-header">FAISS Index Summary</div>', unsafe_allow_html=True)
        fcols = st.columns(4)
        fcols[0].metric("Index type",     faiss.get("type",    "FlatIP"))
        fcols[1].metric("Total vectors",  faiss.get("vectors", "—"))
        fcols[2].metric("Embedding dim",  faiss.get("dim",     384))
        fcols[3].metric("Build time",     faiss.get("build_time", "—"))


# ──────────────────────────────────────────────────────────────────────────────
# Page: Layout
# ──────────────────────────────────────────────────────────────────────────────
def page_layout():
    st.markdown("## 🗺 Layout Detection")

    lay = st.session_state.layout_json
    if not lay:
        st.markdown("""
        <div class="warn-box">
            ⚠️ No layout data loaded. Upload a PDF and run the pipeline, or load a
            <code>layout.json</code> from the Upload page.
        </div>
        """, unsafe_allow_html=True)
        return

    pages = lay["pages"]
    total_regions = sum(len(p["regions"]) for p in pages)

    # Page selector
    col1, col2 = st.columns([1, 3])
    page_idx = col1.selectbox("Select page", range(len(pages)), format_func=lambda i: f"Page {i+1}")
    page_data = pages[page_idx]
    regions   = page_data["regions"]

    # Class filter
    all_classes = sorted({r["region_class"] for r in regions})
    selected_classes = col2.multiselect("Filter by class", all_classes, default=all_classes)

    filtered = [r for r in regions if r["region_class"] in selected_classes]

    st.markdown(f"""
    <div class="info-box">
        Page {page_idx+1} of {len(pages)} — <b>{len(filtered)}</b> regions shown
        (total across document: <b>{total_regions}</b>)
    </div>
    """, unsafe_allow_html=True)

    # Regions list
    st.markdown('<div class="section-header">Detected Regions</div>', unsafe_allow_html=True)
    st.markdown('<div class="scroll-box">', unsafe_allow_html=True)
    for r in filtered:
        cls   = r["region_class"]
        conf  = r.get("confidence", 1.0)
        bb    = r["bbox"]
        text  = (r.get("text_content") or "")[:120]
        text  = text.replace("<","&lt;").replace(">","&gt;")
        color = REGION_COLORS.get(cls, "#8b949e")
        st.markdown(f"""
        <div style="display:flex;align-items:flex-start;gap:10px;padding:8px 10px;
                    margin-bottom:6px;border-radius:8px;border:1px solid #21262d;background:#161b22;">
            <div style="width:4px;min-height:40px;border-radius:2px;background:{color};flex-shrink:0;"></div>
            <div style="flex:1;">
                <div style="display:flex;align-items:center;gap:6px;margin-bottom:4px;">
                    {badge(cls)}
                    <span style="font-size:11px;color:#6e7681;font-family:'JetBrains Mono',monospace;">
                        ({bb['x0']:.0f},{bb['y0']:.0f})→({bb['x1']:.0f},{bb['y1']:.0f})
                    </span>
                    <span style="margin-left:auto;font-size:11px;color:#1D9E75;">conf: {conf:.2f}</span>
                </div>
                <div style="font-size:12px;color:#8b949e;line-height:1.5;">{text or '<em style="color:#6e7681;">no text content</em>'}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Class distribution chart
    st.markdown('<div class="section-header">Class Distribution (document-wide)</div>', unsafe_allow_html=True)
    from collections import Counter
    import plotly.graph_objects as go

    all_cls_counts = Counter(
        r["region_class"] for pg in pages for r in pg["regions"]
    )
    sorted_cls = sorted(all_cls_counts.items(), key=lambda x: -x[1])
    labels, values = zip(*sorted_cls) if sorted_cls else ([], [])
    colors = [REGION_COLORS.get(l, "#8b949e") for l in labels]

    fig = go.Figure(go.Bar(
        x=list(labels), y=list(values),
        marker_color=colors, marker_line_width=0,
        text=list(values), textposition="outside",
        textfont=dict(color="#c9d1d9", size=11),
    ))
    fig.update_layout(
        plot_bgcolor="#0f1117", paper_bgcolor="#0f1117",
        font=dict(color="#8b949e", family="Inter"),
        xaxis=dict(gridcolor="#21262d", tickfont=dict(size=12, color="#c9d1d9")),
        yaxis=dict(gridcolor="#21262d", tickfont=dict(size=11, color="#8b949e")),
        margin=dict(t=20, b=20, l=20, r=20),
        height=260,
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────────
# Page: Semantics
# ──────────────────────────────────────────────────────────────────────────────
def page_semantic():
    st.markdown("## 🧠 Semantic Understanding")

    sem = st.session_state.semantic_json
    if not sem:
        st.markdown("""
        <div class="warn-box">
            ⚠️ No semantic data loaded. Run the full pipeline or load a
            <code>layout_semantic.json</code> from the Upload page.
        </div>
        """, unsafe_allow_html=True)
        return

    regions = sem.get("semantic_regions", [])
    links   = sem.get("multimodal_links",  [])

    tab1, tab2, tab3 = st.tabs([
        f"Semantic Regions ({len(regions)})",
        f"Multimodal Links ({len(links)})",
        "Role Distribution",
    ])

    with tab1:
        col1, col2 = st.columns([2, 1])
        all_roles   = sorted({r.get("scholarly_role","N/A") for r in regions})
        all_classes = sorted({r.get("region_class","") for r in regions})
        role_filter = col1.multiselect("Filter by role", all_roles, default=all_roles)
        cls_filter  = col2.multiselect("Filter by class", all_classes, default=all_classes)

        filtered = [
            r for r in regions
            if r.get("scholarly_role","N/A") in role_filter
            and r.get("region_class","") in cls_filter
        ]
        st.caption(f"Showing {len(filtered)} of {len(regions)} regions")

        st.markdown('<div class="scroll-box">', unsafe_allow_html=True)
        for r in filtered:
            role     = r.get("scholarly_role", "N/A")
            conf     = r.get("role_confidence", 0.0)
            cls_name = r.get("region_class", "")
            text     = (r.get("text_content") or "")[:200]
            text     = text.replace("<","&lt;").replace(">","&gt;")
            page_i   = r.get("page_index", 0)
            rid      = r.get("region_id","")[:8]

            role_css, role_lbl = ROLE_COLORS.get(role, ("badge-reference", role))

            st.markdown(f"""
            <div class="evidence-item">
                <div class="ev-meta">
                    <span class="ev-score">conf: {conf:.2f}</span>
                    {badge(cls_name)} &nbsp;
                    <span class="badge {role_css}">{role_lbl}</span> &nbsp;
                    <span style="color:#6e7681;">page {page_i+1} · {rid}…</span>
                </div>
                <div class="ev-text">{text or '<em style="color:#6e7681;">no text</em>'}</div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        if not links:
            st.info("No multimodal links found.")
        else:
            for lnk in links:
                fig_id  = lnk.get("figure_id","—")
                cap_id  = lnk.get("caption_id","—")
                s_link  = lnk.get("s_link", 0.0)
                s_ref   = lnk.get("s_ref",  0.0)
                s_emb   = lnk.get("s_emb",  0.0)
                refs    = lnk.get("matched_refs", [])
                st.markdown(f"""
                <div class="evidence-item">
                    <div class="ev-meta">
                        <span class="ev-score">s_link: {s_link:.3f}</span>
                        <span style="font-family:'JetBrains Mono',monospace;">{fig_id[:8]}… → {cap_id[:8]}…</span>
                    </div>
                    <div style="display:flex;gap:12px;margin-top:6px;font-size:12px;color:#8b949e;">
                        <span>s_ref: <b style="color:#c9d1d9;">{s_ref:.2f}</b></span>
                        <span>s_emb: <b style="color:#c9d1d9;">{s_emb:.2f}</b></span>
                        <span>refs matched: <b style="color:#c9d1d9;">{len(refs)}</b></span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    with tab3:
        import plotly.graph_objects as go
        from collections import Counter
        role_counts = Counter(r.get("scholarly_role","N/A") for r in regions)
        role_counts.pop("N/A", None)

        if role_counts:
            labels = list(role_counts.keys())
            values = list(role_counts.values())
            role_palette = {
                "Method":      "#58a6ff",
                "Result":      "#85e89d",
                "Dataset":     "#ffab70",
                "Definition":  "#d2a8ff",
                "Observation": "#56d364",
            }
            colors = [role_palette.get(l,"#8b949e") for l in labels]

            fig = go.Figure(go.Pie(
                labels=labels, values=values,
                marker=dict(colors=colors, line=dict(color="#0f1117", width=2)),
                textfont=dict(color="#0f1117", size=12),
                hole=0.5,
            ))
            fig.update_layout(
                plot_bgcolor="#0f1117", paper_bgcolor="#0f1117",
                font=dict(color="#8b949e", family="Inter"),
                legend=dict(font=dict(color="#c9d1d9")),
                margin=dict(t=20, b=20, l=20, r=20),
                height=300,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No scholarly role data available.")


# ──────────────────────────────────────────────────────────────────────────────
# Page: Knowledge Graph
# ──────────────────────────────────────────────────────────────────────────────
def page_graph():
    st.markdown("## 🕸 Knowledge Graph")

    kg = st.session_state.kg_summary
    sem = st.session_state.semantic_json

    if not (kg or sem):
        st.markdown("""
        <div class="warn-box">
            ⚠️ Knowledge graph not built yet. Build the full pipeline or run
            <code>python run_kg.py</code> first.
        </div>
        """, unsafe_allow_html=True)
        return

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Nodes",   kg.get("nodes","—") if kg else "—")
    col2.metric("Edges",   kg.get("edges","—") if kg else "—")
    col3.metric("Density", f"{kg.get('density',0):.4f}" if kg else "—")
    col4.metric("Isolated",kg.get("isolated",0) if kg else "—")

    # Edge breakdown
    if kg and "edge_breakdown" in kg:
        st.markdown('<div class="section-header">Edge Breakdown</div>', unsafe_allow_html=True)
        eb = kg["edge_breakdown"]
        total_e = sum(eb.values())
        for rel, cnt in sorted(eb.items(), key=lambda x: -x[1]):
            pct = int(cnt / max(total_e, 1) * 100)
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:10px;margin-bottom:6px;">
                <span style="font-size:12px;color:#c9d1d9;width:200px;font-family:'JetBrains Mono',monospace;">{rel}</span>
                <div style="flex:1;height:6px;border-radius:3px;background:#21262d;overflow:hidden;">
                    <div style="height:100%;width:{pct}%;background:#7F77DD;border-radius:3px;"></div>
                </div>
                <span style="font-size:12px;color:#8b949e;width:36px;text-align:right;">{cnt}</span>
            </div>
            """, unsafe_allow_html=True)

    # Interactive graph (if networkx available + graphml loaded)
    if sem:
        st.markdown('<div class="section-header">Graph Visualization (semantic network)</div>', unsafe_allow_html=True)
        try:
            import plotly.graph_objects as go

            regions = sem.get("semantic_regions", [])[:60]  # limit for perf
            # Build a simple graph from the semantic regions
            import math, random
            random.seed(42)
            n = len(regions)
            angle_step = 2 * math.pi / max(n, 1)

            node_x, node_y, node_text, node_color, node_size = [], [], [], [], []
            role_palette = {
                "Method":      "#58a6ff",
                "Result":      "#85e89d",
                "Dataset":     "#ffab70",
                "Definition":  "#d2a8ff",
                "Observation": "#56d364",
                "N/A":         "#8b949e",
            }

            for i, r in enumerate(regions):
                angle = i * angle_step + random.uniform(-0.3, 0.3)
                radius = 0.4 + random.uniform(0, 0.5)
                node_x.append(radius * math.cos(angle))
                node_y.append(radius * math.sin(angle))
                role = r.get("scholarly_role", "N/A")
                node_color.append(role_palette.get(role, "#8b949e"))
                node_size.append(16 if role != "N/A" else 8)
                short_text = (r.get("text_content") or "")[:60]
                node_text.append(f"[{role}] {short_text}")

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=node_x, y=node_y, mode="markers",
                marker=dict(color=node_color, size=node_size,
                            line=dict(width=1, color="#0f1117")),
                text=node_text, hoverinfo="text",
                name="Regions",
            ))
            fig.update_layout(
                plot_bgcolor="#0f1117", paper_bgcolor="#0f1117",
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                margin=dict(t=20, b=20, l=20, r=20),
                height=400,
                showlegend=False,
                hovermode="closest",
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Each dot is a semantic region. Color = scholarly role. Hover for text preview.")

        except Exception as e:
            st.warning(f"Visualization unavailable: {e}")

    # Role distribution table
    if kg and "role_dist" in kg:
        st.markdown('<div class="section-header">Scholarly Role Distribution</div>', unsafe_allow_html=True)
        role_dist = kg["role_dist"]
        total = sum(role_dist.values())
        tbl_rows = ""
        for role, count in sorted(role_dist.items(), key=lambda x: -x[1]):
            pct = count / max(total, 1) * 100
            css, lbl = ROLE_COLORS.get(role, ("badge-reference", role))
            tbl_rows += f"""
            <tr>
                <td><span class="badge {css}">{lbl}</span></td>
                <td style="text-align:right;color:#c9d1d9;">{count}</td>
                <td style="text-align:right;color:#8b949e;">{pct:.1f}%</td>
            </tr>"""
        st.markdown(f"""
        <table class="kg-table">
            <thead><tr><th>Role</th><th style="text-align:right;">Count</th><th style="text-align:right;">%</th></tr></thead>
            <tbody>{tbl_rows}</tbody>
        </table>
        """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Page: Q&A
# ──────────────────────────────────────────────────────────────────────────────
def page_qa():
    st.markdown("## 💬 Document Q&A")

    if not st.session_state.pipeline_built:
        st.markdown("""
        <div class="warn-box">
            ⚠️ Pipeline not built yet. Go to <b>Upload</b> and build the KG + FAISS + Hybrid RAG
            pipeline, or run <code>python pipeline.py</code> first.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="info-box">
            💡 <b>Demo mode</b>: You can still try the interface below — responses will be
            placeholders. Build the pipeline for real answers via Ollama.
        </div>
        """, unsafe_allow_html=True)

    # Config row
    col1, col2, col3 = st.columns([2, 1, 1])
    retrieval_mode = col1.radio(
        "Retrieval mode",
        ["Hybrid (FAISS + KG)", "Dense (FAISS only)", "Graph-aware"],
        horizontal=True,
        index=0,
    )
    role_filter_opts = ["Method", "Result", "Dataset", "Definition", "Observation"]
    role_filter = col2.multiselect("Role filter", role_filter_opts, default=[])
    intent_override = col3.selectbox(
        "Intent override",
        ["auto", "method", "dataset", "result", "figure", "general"],
    )

    st.markdown("---")

    # Chat history
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f'<div class="qbubble">🧑 {msg["content"]}</div>', unsafe_allow_html=True)
        else:
            answer = msg["content"]
            answer_text  = answer.get("answer_text", "")
            evidence     = answer.get("evidence", [])
            graph_steps  = answer.get("graph_steps", [])
            model        = answer.get("model", "—")
            latency      = answer.get("latency_s", 0)
            prompt_tokens= answer.get("prompt_tokens", 0)

            # Main answer bubble
            st.markdown(f'<div class="abubble">🤖 {answer_text}</div>', unsafe_allow_html=True)

            # Evidence
            if evidence:
                with st.expander(f"📚 Evidence ({len(evidence)} passages)"):
                    for ev in evidence:
                        rank  = ev.get("rank", "—")
                        role  = ev.get("role","—")
                        page  = ev.get("page",0)
                        score = ev.get("score",0.0)
                        text  = ev.get("text","")[:300]
                        text  = text.replace("<","&lt;").replace(">","&gt;")
                        css, lbl = ROLE_COLORS.get(role, ("badge-reference", role))
                        st.markdown(f"""
                        <div class="evidence-item">
                            <div class="ev-meta">
                                <span class="ev-score">score: {score:.3f}</span>
                                P{rank} &nbsp; <span class="badge {css}">{lbl}</span>
                                &nbsp; page {page+1}
                            </div>
                            <div class="ev-text">{text}</div>
                        </div>
                        """, unsafe_allow_html=True)

            # Graph reasoning
            if graph_steps:
                with st.expander(f"🕸 Graph reasoning ({len(graph_steps)} steps)"):
                    for step in graph_steps:
                        step_text = step.replace("<","&lt;").replace(">","&gt;")
                        st.markdown(f'<div class="graph-step">{step_text}</div>', unsafe_allow_html=True)

            # Meta
            st.markdown(f"""
            <div style="font-size:11px;color:#6e7681;margin-bottom:12px;">
                model: {model} · latency: {latency}s · tokens: {prompt_tokens}
            </div>
            """, unsafe_allow_html=True)

    # Query input
    st.markdown("---")
    col_q, col_btn = st.columns([5, 1])
    with col_q:
        query = st.text_input(
            "Ask a question about the document",
            key="query_input",
            label_visibility="collapsed",
            placeholder="e.g. What is the proposed method and how does it work?",
        )
    with col_btn:
        ask_clicked = st.button("Ask →", type="primary", use_container_width=True)

    # Quick questions
    st.markdown("**Quick questions:**")
    qcols = st.columns(4)
    quick_qs = [
        "What is the proposed method?",
        "What datasets were used?",
        "What are the main results?",
        "How does the model architecture work?",
    ]
    for col, q in zip(qcols, quick_qs):
        if col.button(q, use_container_width=True, key=f"qq_{q[:20]}"):
            query = q
            ask_clicked = True

    # Handle query
    if ask_clicked and query.strip():
        st.session_state.chat_history.append({"role": "user", "content": query})

        with st.spinner("Retrieving context and generating answer..."):
            answer = _run_query(
                query,
                role_filter=role_filter if role_filter else None,
                intent_override=None if intent_override == "auto" else intent_override,
            )

        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        st.rerun()

    # Clear chat
    if st.session_state.chat_history:
        if st.button("Clear conversation", key="clear_chat"):
            st.session_state.chat_history = []
            st.rerun()


# ──────────────────────────────────────────────────────────────────────────────
# Page: Settings
# ──────────────────────────────────────────────────────────────────────────────
def page_settings():
    st.markdown("## ⚙️ Settings")

    st.markdown('<div class="section-header">Ollama / LLM</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    st.session_state.ollama_model = col1.text_input(
        "Model name", value=st.session_state.ollama_model,
        help="e.g. llama3, mistral, gemma2"
    )
    st.session_state.ollama_url = col2.text_input(
        "Ollama base URL", value=st.session_state.ollama_url
    )

    st.markdown('<div class="section-header">Retrieval Config</div>', unsafe_allow_html=True)
    col3, col4, col5, col6 = st.columns(4)
    st.session_state.faiss_top_k    = col3.number_input("FAISS top-k",     value=st.session_state.faiss_top_k,    min_value=1, max_value=20)
    st.session_state.graph_expand_k = col4.number_input("Graph expand-k",  value=st.session_state.graph_expand_k, min_value=1, max_value=20)
    st.session_state.final_k        = col5.number_input("Final top-k",     value=st.session_state.final_k,        min_value=1, max_value=30)
    st.session_state.alpha          = col6.slider("α (FAISS weight)", 0.0, 1.0, float(st.session_state.alpha), 0.05,
                                                   help="α·FAISS + (1-α)·Graph")

    st.markdown('<div class="section-header">Generation Config</div>', unsafe_allow_html=True)
    st.session_state.temperature = st.slider(
        "LLM temperature", 0.0, 1.0, float(st.session_state.temperature), 0.05
    )

    st.markdown("---")
    st.markdown('<div class="section-header">Diagnostics</div>', unsafe_allow_html=True)

    if st.button("Check Ollama health"):
        pipeline = st.session_state.pipeline
        if pipeline and hasattr(pipeline, "_llm") and pipeline._llm:
            ok = pipeline._llm.health_check()
            if ok:
                st.success("✅ Ollama is reachable and model is loaded.")
            else:
                st.error(f"❌ Ollama health check failed. Run: ollama pull {st.session_state.ollama_model} && ollama serve")
        else:
            st.warning("Pipeline not built yet. Build pipeline first.")

    st.markdown("---")
    st.markdown('<div class="section-header">Session</div>', unsafe_allow_html=True)
    if st.button("🔄 Reset entire session", type="secondary"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline runners
# ──────────────────────────────────────────────────────────────────────────────
def _run_extraction(pdf_path: str, output_dir: str, ocr_threshold: int, dpi: int):
    try:
        from agents.document_extraction_agent import DocumentExtractionAgent
        with st.spinner("Running document extraction..."):
            agent = DocumentExtractionAgent(
                output_dir=output_dir, ocr_threshold=ocr_threshold,
                render_dpi=dpi, verbose=False,
            )
            result = agent.run(pdf_path)
            doc_id = result.doc_id
            ext_path = Path(output_dir) / doc_id / "json" / "extraction.json"
            st.session_state.extraction_json = load_json_safe(str(ext_path))
            st.session_state.doc_name = result.doc_id
            stats = result.stats()
        st.success(
            f"✅ Extraction done — {stats['total_blocks']} text blocks, "
            f"{stats['figure_blocks']} figures, {stats['total_pages']} pages"
        )
    except ImportError:
        st.error("Could not import DocumentExtractionAgent. Ensure agenticdoc2 is on PYTHONPATH.")
    except Exception as e:
        st.error(f"Extraction failed: {e}")


def _run_full_pipeline(pdf_path: str, output_dir: str, ocr_threshold: int, dpi: int, backend: str):
    DocumentPipeline, PipelineConfig = try_import_pipeline()
    if not DocumentPipeline:
        st.error(
            "Could not import pipeline. Make sure agenticdoc2 is installed and "
            "PYTHONPATH includes the project root."
        )
        return

    progress = st.progress(0, text="Starting pipeline...")
    try:
        # Step 1: Extraction
        progress.progress(10, text="Step 1/7 — Document extraction...")
        from agents.document_extraction_agent import DocumentExtractionAgent
        ext_agent = DocumentExtractionAgent(output_dir=output_dir, ocr_threshold=ocr_threshold, render_dpi=dpi)
        extraction = ext_agent.run(pdf_path)
        doc_id = extraction.doc_id
        ext_path = Path(output_dir) / doc_id / "json" / "extraction.json"
        st.session_state.extraction_json = load_json_safe(str(ext_path))
        st.session_state.doc_name = doc_id

        # Step 2: Layout
        progress.progress(25, text="Step 2/7 — Layout detection...")
        from agents.layout_detection_agent import LayoutDetectionAgent
        lay_agent = LayoutDetectionAgent(backend=backend, output_dir=output_dir, render_dpi=dpi)
        layout = lay_agent.run(extraction, pdf_path=pdf_path)
        lay_path = Path(output_dir) / doc_id / "json" / "layout.json"
        st.session_state.layout_json = load_json_safe(str(lay_path))

        # Step 3: Alignment
        progress.progress(40, text="Step 3/7 — Text-layout alignment...")
        from agents.semantic_understanding_agent import SemanticUnderstandingAgent
        sem_agent = SemanticUnderstandingAgent()
        sem_result = sem_agent.process_from_dict(st.session_state.layout_json)
        sem_path = Path(output_dir) / doc_id / "json" / "layout_semantic.json"
        sem_path.parent.mkdir(parents=True, exist_ok=True)
        with open(sem_path, "w") as f:
            json.dump(SemanticUnderstandingAgent.serialize(
                sem_result["semantic_regions"], sem_result["multimodal_links"]
            ), f)
        st.session_state.semantic_json = load_json_safe(str(sem_path))
        st.session_state.semantic_path = str(sem_path)

        progress.progress(60, text="Steps 4-7 — Building KG + FAISS + Hybrid RAG...")
        _build_pipeline(str(sem_path))

        progress.progress(100, text="Pipeline complete!")
        st.success("✅ Full pipeline complete. Use the tabs to explore results and ask questions.")

    except Exception as e:
        st.error(f"Pipeline failed: {e}")
        logging.exception("Pipeline error")
    finally:
        progress.empty()


def _build_pipeline(semantic_json_path: str):
    DocumentPipeline, PipelineConfig = try_import_pipeline()
    if not DocumentPipeline:
        st.error("Cannot import DocumentPipeline. Check PYTHONPATH.")
        return

    try:
        with st.spinner("Building Knowledge Graph, FAISS index, and Hybrid Retrieval Engine..."):
            cfg = PipelineConfig(
                faiss_top_k     = st.session_state.faiss_top_k,
                graph_expand_k  = st.session_state.graph_expand_k,
                final_k         = st.session_state.final_k,
                alpha           = st.session_state.alpha,
                ollama_model    = st.session_state.ollama_model,
                temperature     = st.session_state.temperature,
                ollama_base_url = st.session_state.ollama_url,
            )
            pipeline = DocumentPipeline(config=cfg)
            pipeline.build(semantic_json_path)
            st.session_state.pipeline = pipeline
            st.session_state.pipeline_built = True
            st.session_state.semantic_path  = semantic_json_path

            # Capture KG summary
            if pipeline._kg:
                g = pipeline._kg.graph
                from collections import Counter
                roles = [d.get("scholarly_role","N/A") for _, d in g.nodes(data=True)]
                edge_breakdown = Counter(
                    d.get("relation","unknown")
                    for _, _, d in g.edges(data=True)
                )
                isolated = sum(1 for n in g.nodes() if g.degree(n) == 0)
                n = g.number_of_nodes()
                e = g.number_of_edges()
                density = 2*e / (n*(n-1)) if n > 1 else 0
                st.session_state.kg_summary = {
                    "nodes": n, "edges": e, "density": round(density, 5),
                    "isolated": isolated,
                    "role_dist": dict(Counter(r for r in roles if r != "N/A")),
                    "edge_breakdown": {f"[{t}] {r}": c for (t,r),c in
                                       Counter((d.get("edge_type",""),d.get("relation",""))
                                               for _,_,d in g.edges(data=True)).items()
                                       } if True else dict(edge_breakdown),
                }

            # Capture FAISS summary
            if pipeline._faiss:
                st.session_state.faiss_summary = {
                    "type":    "FlatIP",
                    "vectors": pipeline._faiss.index.ntotal if hasattr(pipeline._faiss,"index") else "—",
                    "dim":     384,
                    "build_time": "—",
                }

        st.success("✅ Pipeline built. Go to Q&A to start asking questions.")

    except Exception as e:
        st.error(f"Pipeline build failed: {e}")
        logging.exception("Build error")


def _run_query(
    query: str,
    role_filter: Optional[list] = None,
    intent_override: Optional[str] = None,
) -> dict:
    pipeline = st.session_state.pipeline
    if pipeline and st.session_state.pipeline_built:
        try:
            t0 = time.time()
            answer = pipeline.ask(
                query            = query,
                role_filter      = role_filter,
                intent_override  = intent_override,
                temperature      = st.session_state.temperature,
            )
            return answer.to_dict() if hasattr(answer, "to_dict") else {
                "answer_text":  getattr(answer, "answer_text",  str(answer)),
                "evidence":     getattr(answer, "evidence",     []),
                "graph_steps":  getattr(answer, "graph_steps",  []),
                "model":        getattr(answer, "model",        st.session_state.ollama_model),
                "latency_s":    getattr(answer, "latency_s",    round(time.time()-t0, 1)),
                "prompt_tokens":getattr(answer, "prompt_tokens", 0),
            }
        except Exception as e:
            return {
                "answer_text": f"❌ Error during generation: {e}\n\nCheck that Ollama is running "
                               f"(`ollama serve`) and the model is pulled (`ollama pull {st.session_state.ollama_model}`).",
                "evidence": [], "graph_steps": [],
                "model": st.session_state.ollama_model, "latency_s": 0, "prompt_tokens": 0,
            }
    else:
        # Demo placeholder
        return {
            "answer_text": (
                f"**[Demo mode — pipeline not built]**\n\n"
                f"Your query was: *{query}*\n\n"
                f"Build the pipeline (Steps 5–8) to get real answers via Ollama. "
                f"The hybrid retrieval engine will use FAISS dense search + KG graph expansion "
                f"to find relevant passages, then generate a structured answer with evidence."
            ),
            "evidence": [
                {"rank":1,"role":"Method","page":0,"score":0.89,
                 "text":"[Demo] This is where retrieved passage 1 would appear..."},
                {"rank":2,"role":"Result","page":2,"score":0.74,
                 "text":"[Demo] This is where retrieved passage 2 would appear..."},
            ],
            "graph_steps": [
                "[Demo] Node_A → produces → Node_B: graph reasoning step would appear here."
            ],
            "model": "demo", "latency_s": 0.0, "prompt_tokens": 0,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Main app router
# ──────────────────────────────────────────────────────────────────────────────
def main():
    render_sidebar()

    page = st.session_state.page
    if   page == "upload":   page_upload()
    elif page == "pipeline": page_pipeline()
    elif page == "layout":   page_layout()
    elif page == "semantic": page_semantic()
    elif page == "graph":    page_graph()
    elif page == "qa":       page_qa()
    elif page == "settings": page_settings()
    else:
        page_upload()


if __name__ == "__main__":
    main()