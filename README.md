# AgenticDoc – Agentic Document Understanding Pipeline

## Overview

**AgenticDoc** is a modular document intelligence system designed to convert complex academic PDFs into **structured, layout-aware representations** that can later be used by Large Language Models (LLMs).

The project follows an **agent-based architecture**, where each stage of document understanding is handled by a dedicated agent.

Current implemented stages:

1. **Document Extraction Agent**
2. **Layout Detection Agent**
3. **Figure Detection**
4. **Reading Order Reconstruction**
5. **Layout Visualization**

The system transforms raw PDFs into **structured JSON with spatial metadata**, enabling downstream tasks like semantic chunking, retrieval, and LLM reasoning.

---

# Pipeline Architecture

Current pipeline:

```
PDF Document
     ↓
Document Extraction Agent
     ↓
Layout Detection Agent
     ↓
Figure Detection
     ↓
Reading Order Reconstruction
     ↓
Structured JSON Output
     ↓
Visualization / Debugging
```

Future pipeline:

```
PDF
 ↓
Extraction
 ↓
Layout Detection
 ↓
Semantic Chunking
 ↓
Embedding + Retrieval
 ↓
LLM Reasoning
 ↓
Knowledge Graph
```

---

# Repository Structure

```
agenticdoc/
│
├── agents/
│   ├── document_extraction_agent.py
│   └── layout_detection_agent.py
│
├── extractors/
│   ├── pdf_extractor.py
│   ├── heuristic_layout_detector.py
│   ├── ml_layout_detector.py
│   ├── figure_detector.py
│   ├── reading_order.py
│   ├── layout_models.py
│   └── models.py
│
├── output/
│   └── <doc_id>/
│       ├── figures/
│       └── json/
│           ├── extraction.json
│           └── layout.json
│
├── plots/
│
├── tests/
│
├── run_extraction.py
├── run_layout.py
├── draw.py
├── tempdraw.py
└── README.md
```

---

# Agent 1 – Document Extraction Agent

File:

```
agents/document_extraction_agent.py
```

This agent parses the input PDF and extracts structured content.

## Responsibilities

- Open PDF using **PyMuPDF**
- Extract text blocks
- Extract embedded images
- Run OCR fallback when text is missing
- Record bounding boxes and font metadata
- Save structured JSON output

## Output

```
output/<doc_id>/json/extraction.json
```

Example structure:

```json
{
  "doc_id": "paper",
  "total_pages": 3,
  "pages": [
    {
      "page_index": 0,
      "text_blocks": [...],
      "figure_blocks": [...]
    }
  ]
}
```

---

# Agent 2 – Layout Detection Agent

File:

```
agents/layout_detection_agent.py
```

This agent converts extracted blocks into **semantic layout regions**.

## Supported Backends

| Backend      | Description                                  |
| ------------ | -------------------------------------------- |
| heuristic    | Rule-based classifier (no ML dependencies)   |
| layoutparser | Detectron2-based PubLayNet model             |
| dit          | Transformer-based Document Image Transformer |

The system automatically selects the best backend available.

Priority order:

```
LayoutParser → DiT → Heuristic
```

---

# Layout Classes

The system detects the following region types:

| Class     | Description                |
| --------- | -------------------------- |
| title     | Document or section titles |
| paragraph | Body text                  |
| figure    | Figure regions             |
| caption   | Figure/table captions      |
| table     | Table content              |
| equation  | Mathematical expressions   |
| list      | Bullet lists               |
| reference | Bibliography entries       |
| header    | Running headers            |
| footer    | Page numbers / footers     |

---

# Figure Detection

File:

```
extractors/figure_detector.py
```

Many research PDFs contain figures that are **not raster images**.

This module detects figures using **three complementary strategies**:

### 1. Vector Drawing Detection

Uses:

```
page.get_drawings()
```

to detect diagrams drawn using vector graphics.

### 2. XObject Detection

Detects figures embedded as **Form XObjects**.

### 3. Text Gap Detection

Detects large empty areas surrounded by text blocks.

Final figures are **merged and deduplicated**.

---

# Reading Order Reconstruction

File:

```
extractors/reading_order.py
```

Academic papers often use **multi-column layouts**.

This module reconstructs the correct reading order using:

- column detection
- bounding box sorting
- spatial clustering

Example:

```
Correct order:

Title
Figure
Caption
Left column text
Right column text
```

---

# Visualization Tools

Visualization scripts allow debugging of detected regions.

## Layout Visualization

Script:

```
draw.py
```

Command:

```
python draw.py
```

Produces:

```
plots/papermod_layout_visualized.pdf
```

Example visualization:

- **Red** → title
- **Blue** → paragraph
- **Green** → figure
- **Orange** → caption

---

# Running the System

## Install Dependencies

```
pip install -r requirements.txt
```

Core dependencies:

```
pymupdf
pytesseract
opencv-python
numpy
pandas
pydantic
loguru
```

Optional ML backends:

```
layoutparser
torch
transformers
```

---

# Run Extraction

```
python run_extraction.py --pdf pdf/paper.pdf
```

Output:

```
output/paper/json/extraction.json
```

---

# Run Layout Detection

```
python run_layout.py --pdf pdf/paper.pdf
```

Force a specific backend:

```
python run_layout.py --pdf pdf/paper.pdf --backend heuristic
python run_layout.py --pdf pdf/paper.pdf --backend dit
python run_layout.py --pdf pdf/paper.pdf --backend layoutparser
```

Print detected regions:

```
python run_layout.py --pdf pdf/paper.pdf --print-regions
```

Output:

```
output/paper/json/layout.json
```

---

# Example Layout Output

```
LAYOUT DETECTION SUMMARY

Document ID  : paper
Total Pages  : 3
Total Regions: 66

paragraph  █████████████████████████
caption    ████
figure     █
list       ████
reference  ████
equation   █
```

---

# Example Visualization

Detected layout:

```
+--------------------------------+
| TITLE                          |
+--------------------------------+

+--------------------------------+
| FIGURE                         |
+--------------------------------+
| CAPTION                        |
+--------------------------------+

| LEFT COLUMN TEXT | RIGHT TEXT |
| LEFT COLUMN TEXT | RIGHT TEXT |
```

---

# Testing

Unit tests are available in:

```
tests/
```

Run tests:

```
pytest tests
```

---

# Current Capabilities

| Feature               | Status |
| --------------------- | ------ |
| Text extraction       | ✅     |
| OCR fallback          | ✅     |
| Figure detection      | ✅     |
| Caption detection     | ✅     |
| Table detection       | ✅     |
| Layout classification | ✅     |
| Reading order         | ✅     |
| Visualization         | ✅     |

---

# Limitations

Current system does not yet implement:

- semantic chunking
- figure–caption linking
- table structure extraction
- document section hierarchy
- LLM reasoning

These will be implemented in future stages.

---

# Next Development Steps

Upcoming components:

### Semantic Chunking Agent

Groups layout regions into logical document chunks.

Example:

```
Chunk 1
Title + Abstract

Chunk 2
Figure + Caption

Chunk 3
Body Paragraphs
```

### Retrieval Layer

Embedding-based retrieval for RAG pipelines.

### LLM Reasoning Agent

Use structured layout information to answer document queries.

---

# Long-Term Goal

The final system will act as a **Document Intelligence Engine** capable of understanding:

- research papers
- financial reports
- scientific articles
- technical documentation

The goal is to transform documents into **structured knowledge usable by AI systems**.

---

# Summary

AgenticDoc currently implements:

- modular document parsing
- layout-aware region detection
- multi-strategy figure detection
- reading order reconstruction
- structured JSON outputs
- layout visualization tools

This forms the foundation for a **full multimodal document understanding system**.
