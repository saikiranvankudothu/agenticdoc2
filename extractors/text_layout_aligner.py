"""
extractors/text_layout_aligner.py
-----------------------------------
Step 4 — Text–Layout Alignment

Implements Smap(i,j) = α·SIoU + (1-α)·Scontain
to assign each TextBlock from extraction.json
to the best matching LayoutRegion from layout.json.

Paper: Equation 3, Section IV-B-3
"""
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Optional

from extractors.models import (
    DocumentExtractionResult, TextBlock, BoundingBox
)
from extractors.layout_models import (
    DocumentLayoutResult, LayoutRegion, RegionClass
)

logger = logging.getLogger(__name__)

ALPHA = 0.6   # weight for IoU vs containment
TAU   = 0.3   # minimum score to accept assignment


@dataclass
class AlignmentResult:
    """One text block aligned to one layout region."""
    block_id:     str
    region_id:    str
    region_class: RegionClass
    score:        float
    page_index:   int
    flagged:      bool = False   # True if score < TAU (unassigned)


@dataclass
class DocumentAlignmentResult:
    """Full alignment output — input for Agent 3 (SBERT encoding)."""
    doc_id:      str
    alignments:  list[AlignmentResult] = field(default_factory=list)
    unassigned:  list[str]             = field(default_factory=list)  # block_ids

    def stats(self) -> dict:
        return {
            "total_aligned":   len(self.alignments),
            "total_unassigned": len(self.unassigned),
            "flagged":         sum(1 for a in self.alignments if a.flagged),
        }


def _iou(b1: BoundingBox, b2: BoundingBox) -> float:
    """Intersection over Union between two bounding boxes."""
    ix0 = max(b1.x0, b2.x0)
    iy0 = max(b1.y0, b2.y0)
    ix1 = min(b1.x1, b2.x1)
    iy1 = min(b1.y1, b2.y1)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    union = b1.area + b2.area - inter
    return inter / union if union > 0 else 0.0


def _containment(block_bbox: BoundingBox, region_bbox: BoundingBox) -> float:
    """Fraction of text block area contained within region bbox."""
    ix0 = max(block_bbox.x0, region_bbox.x0)
    iy0 = max(block_bbox.y0, region_bbox.y0)
    ix1 = min(block_bbox.x1, region_bbox.x1)
    iy1 = min(block_bbox.y1, region_bbox.y1)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    return inter / block_bbox.area if block_bbox.area > 0 else 0.0


def _smap(block_bbox: BoundingBox, region_bbox: BoundingBox,
          alpha: float = ALPHA) -> float:
    """
    Smap(i,j) = α·SIoU + (1-α)·Scontain
    Equation 3 from the paper.
    """
    return alpha * _iou(block_bbox, region_bbox) + \
           (1 - alpha) * _containment(block_bbox, region_bbox)


def align(
    extraction: DocumentExtractionResult,
    layout:     DocumentLayoutResult,
    alpha:      float = ALPHA,
    tau:        float = TAU,
) -> DocumentAlignmentResult:
    """
    Align every TextBlock in extraction to the best matching
    LayoutRegion in layout using Smap scoring.
    """
    result = DocumentAlignmentResult(doc_id=extraction.doc_id)

    for page in extraction.pages:
        # Get all regions on this page
        page_layout = next(
            (p for p in layout.pages if p.page_index == page.page_index), None
        )
        if not page_layout:
            # No layout for this page — flag all blocks
            for blk in page.text_blocks:
                result.unassigned.append(blk.block_id)
            continue

        regions = page_layout.regions

        for blk in page.text_blocks:
            best_score  = 0.0
            best_region: Optional[LayoutRegion] = None

            for region in regions:
                # Only compare blocks on same page
                if region.page_index != blk.page_index:
                    continue
                score = _smap(blk.bbox, region.bbox, alpha)
                if score > best_score:
                    best_score  = score
                    best_region = region

            if best_region and best_score >= tau:
                result.alignments.append(AlignmentResult(
                    block_id     = blk.block_id,
                    region_id    = best_region.region_id,
                    region_class = best_region.region_class,
                    score        = best_score,
                    page_index   = blk.page_index,
                    flagged      = False,
                ))
                # Enrich region with text if empty
                if not best_region.text_content:
                    best_region.text_content = blk.text
                if blk.block_id not in best_region.source_block_ids:
                    best_region.source_block_ids.append(blk.block_id)
            else:
                # Score below threshold — unassigned
                result.unassigned.append(blk.block_id)
                result.alignments.append(AlignmentResult(
                    block_id     = blk.block_id,
                    region_id    = "",
                    region_class = RegionClass.UNKNOWN,
                    score        = best_score,
                    page_index   = blk.page_index,
                    flagged      = True,
                ))

    stats = result.stats()
    logger.info(
        f"[Alignment] aligned={stats['total_aligned']} "
        f"unassigned={stats['total_unassigned']} "
        f"flagged={stats['flagged']}"
    )
    return result