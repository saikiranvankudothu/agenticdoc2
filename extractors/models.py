"""
extractors/models.py
--------------------
Typed data models for all extraction outputs.
Uses Python stdlib dataclasses — zero external dependencies.

Paper reference: B = {(ti, bi, pi, si)}
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import json


class ExtractionMethod(str, Enum):
    EMBEDDED = "embedded"
    OCR      = "ocr"
    HYBRID   = "hybrid"


class BlockType(str, Enum):
    TEXT     = "text"
    IMAGE    = "image"
    TABLE    = "table"
    EQUATION = "equation"
    UNKNOWN  = "unknown"


@dataclass
class BoundingBox:
    x0: float; y0: float; x1: float; y1: float

    @property
    def width(self):  return self.x1 - self.x0
    @property
    def height(self): return self.y1 - self.y0
    @property
    def area(self):   return max(0.0, self.width) * max(0.0, self.height)
    def to_tuple(self): return (self.x0, self.y0, self.x1, self.y1)
    def to_dict(self):  return {"x0": self.x0, "y0": self.y0, "x1": self.x1, "y1": self.y1}


@dataclass
class StyleAttributes:
    font_name:  Optional[str]   = None
    font_size:  Optional[float] = None
    is_bold:    Optional[bool]  = None
    is_italic:  Optional[bool]  = None
    color:      Optional[int]   = None
    def to_dict(self):
        return {"font_name": self.font_name, "font_size": self.font_size,
                "is_bold": self.is_bold, "is_italic": self.is_italic, "color": self.color}


@dataclass
class TextBlock:
    block_id:           str
    text:               str
    bbox:               BoundingBox
    page_index:         int
    style:              StyleAttributes  = field(default_factory=StyleAttributes)
    block_type:         BlockType        = BlockType.TEXT
    extraction_method:  ExtractionMethod = ExtractionMethod.EMBEDDED
    confidence:         float            = 1.0
    raw_block_no:       Optional[int]    = None
    def to_dict(self):
        return {"block_id": self.block_id, "text": self.text, "bbox": self.bbox.to_dict(),
                "page_index": self.page_index, "style": self.style.to_dict(),
                "block_type": self.block_type.value,
                "extraction_method": self.extraction_method.value,
                "confidence": self.confidence, "raw_block_no": self.raw_block_no}


@dataclass
class FigureBlock:
    block_id: str; bbox: BoundingBox; page_index: int
    image_path: Optional[str] = None; caption: Optional[str] = None
    def to_dict(self):
        return {"block_id": self.block_id, "bbox": self.bbox.to_dict(),
                "page_index": self.page_index, "image_path": self.image_path,
                "caption": self.caption}


@dataclass
class PageExtractionResult:
    page_index: int; width: float; height: float
    text_blocks:   list = field(default_factory=list)
    figure_blocks: list = field(default_factory=list)
    extraction_method: ExtractionMethod = ExtractionMethod.EMBEDDED
    ocr_triggered: bool = False
    def to_dict(self):
        return {"page_index": self.page_index, "width": self.width, "height": self.height,
                "text_blocks": [b.to_dict() for b in self.text_blocks],
                "figure_blocks": [f.to_dict() for f in self.figure_blocks],
                "extraction_method": self.extraction_method.value,
                "ocr_triggered": self.ocr_triggered}


@dataclass
class DocumentExtractionResult:
    doc_id: str; source_path: str; total_pages: int
    pages: list = field(default_factory=list)

    def all_text_blocks(self):   return [b for p in self.pages for b in p.text_blocks]
    def all_figure_blocks(self): return [f for p in self.pages for f in p.figure_blocks]
    def blocks_for_page(self, idx):
        for p in self.pages:
            if p.page_index == idx: return p.text_blocks
        return []

    def stats(self):
        ocr = sum(1 for p in self.pages if p.ocr_triggered)
        return {"total_pages": self.total_pages, "total_blocks": len(self.all_text_blocks()),
                "ocr_pages": ocr, "embedded_pages": self.total_pages - ocr,
                "figure_blocks": len(self.all_figure_blocks())}

    def to_dict(self):
        return {"doc_id": self.doc_id, "source_path": self.source_path,
                "total_pages": self.total_pages, "pages": [p.to_dict() for p in self.pages]}

    def save_json(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)