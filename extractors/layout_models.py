"""
extractors/layout_models.py
----------------------------
Data models for Step 3 — Layout Detection Agent output.

Paper reference: R = {(cj, rj, pj, γj)}
  cj = region class (title, paragraph, figure, table, caption, equation)
  rj = bounding box
  pj = page index
  γj = confidence score

These models are the OUTPUT CONTRACT of Agent 2 (Layout Detection)
and the INPUT CONTRACT for Agent 3 (Multimodal Semantic Understanding).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import json


# ─────────────────────────────────────────────
# Layout Region Classes
# ─────────────────────────────────────────────

class RegionClass(str, Enum):
    """
    Scholarly region types — maps to PubLayNet / DocLayNet label sets.
    These are refined from BlockType (Step 2's coarse guesses).

    PubLayNet  : text, title, list, table, figure
    DocLayNet  : caption, footnote, formula, list-item, page-footer,
                 page-header, picture, section-header, table, text, title
    We use a unified academic superset:
    """
    TITLE        = "title"         # Section/paper heading
    PARAGRAPH    = "paragraph"     # Body text
    FIGURE       = "figure"        # Image / plot region
    TABLE        = "table"         # Tabular data region
    CAPTION      = "caption"       # Figure or table caption
    EQUATION     = "equation"      # Mathematical expression
    LIST         = "list"          # Bullet / numbered list
    HEADER       = "header"        # Page header (running head)
    FOOTER       = "footer"        # Page footer / footnote
    REFERENCE    = "reference"     # Bibliography entry
    ALGORITHM    = "algorithm"     # Algorithm block
    ABSTRACT     = "abstract"      # Abstract section
    UNKNOWN      = "unknown"       # Fallback


# class DetectionBackend(str, Enum):
#     """Which detection strategy produced this region."""
#     LAYOUTPARSER   = "layoutparser"    # LayoutParser + PubLayNet/Detectron2
#     DIT            = "dit"             # Microsoft DiT (Document Image Transformer)
#     DOCLAYNET      = "doclaynet"       # IBM DocLayNet model
#     HEURISTIC      = "heuristic"       # Rule-based fallback (no ML model)
#     HYBRID         = "hybrid"          # Heuristic + ML fusion


# In extractors/layout_models.py, update DetectionBackend:

class DetectionBackend(str, Enum):
    """Which detection strategy produced this region."""
    LAYOUTPARSER = "layoutparser"    # DEPRECATED - keep for backwards compat
    DIT = "dit"                      # Microsoft DiT
    DOCLAYNET = "doclaynet"          # IBM DocLayNet model
    HEURISTIC = "heuristic"          # Rule-based fallback
    HYBRID = "hybrid"                # Heuristic + ML fusion
    YOLO = "yolo"                    # NEW: YOLOv8
    RTDETR = "rtdetr"                # NEW: RT-DETR

# ─────────────────────────────────────────────
# Core Region Model  →  R_j = (cj, rj, pj, γj)
# ─────────────────────────────────────────────

@dataclass
class LayoutRegion:
    """
    One detected layout region — R_j in the paper.

    Fields
    ------
    region_id    : unique id within the document
    region_class : cj — semantic label (title, figure, table, …)
    bbox         : rj — bounding box in PDF points (x0,y0,x1,y1)
    page_index   : pj — 0-based page number
    confidence   : γj — detection confidence [0, 1]
    backend      : which detector produced this region
    text_content : text falling inside this region (filled during alignment)
    source_block_ids : TextBlock ids that map to this region (filled in Step 4)
    """
    region_id:        str
    region_class:     RegionClass
    bbox:             "BoundingBox"          # imported from extractors.models
    page_index:       int
    confidence:       float                  = 1.0
    backend:          DetectionBackend       = DetectionBackend.HEURISTIC
    text_content:     Optional[str]          = None
    source_block_ids: list                   = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "region_id":        self.region_id,
            "region_class":     self.region_class.value,
            "bbox":             self.bbox.to_dict(),
            "page_index":       self.page_index,
            "confidence":       self.confidence,
            "backend":          self.backend.value,
            "text_content":     self.text_content,
            "source_block_ids": self.source_block_ids,
        }


# ─────────────────────────────────────────────
# Page Layout Result
# ─────────────────────────────────────────────

@dataclass
class PageLayoutResult:
    """All detected layout regions for one page."""
    page_index:  int
    width:       float
    height:      float
    regions:     list = field(default_factory=list)   # list[LayoutRegion]
    backend:     DetectionBackend = DetectionBackend.HEURISTIC

    def regions_of_class(self, cls: RegionClass) -> list:
        return [r for r in self.regions if r.region_class == cls]

    def to_dict(self) -> dict:
        return {
            "page_index": self.page_index,
            "width":      self.width,
            "height":     self.height,
            "regions":    [r.to_dict() for r in self.regions],
            "backend":    self.backend.value,
        }


# ─────────────────────────────────────────────
# Document Layout Result  (top-level output)
# ─────────────────────────────────────────────

@dataclass
class DocumentLayoutResult:
    """
    Top-level output of the Layout Detection Agent.
    Consumed by Agent 3 — Multimodal Semantic Understanding Agent.
    """
    doc_id:      str
    source_path: str
    total_pages: int
    pages:       list = field(default_factory=list)   # list[PageLayoutResult]

    def all_regions(self) -> list:
        return [r for p in self.pages for r in p.regions]

    def regions_of_class(self, cls: RegionClass) -> list:
        return [r for p in self.pages for r in p.regions
                if r.region_class == cls]

    def stats(self) -> dict:
        all_r  = self.all_regions()
        counts = {}
        for r in all_r:
            counts[r.region_class.value] = counts.get(r.region_class.value, 0) + 1
        return {
            "total_pages":   self.total_pages,
            "total_regions": len(all_r),
            "by_class":      counts,
        }

    def to_dict(self) -> dict:
        return {
            "doc_id":      self.doc_id,
            "source_path": self.source_path,
            "total_pages": self.total_pages,
            "pages":       [p.to_dict() for p in self.pages],
        }

    def save_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)