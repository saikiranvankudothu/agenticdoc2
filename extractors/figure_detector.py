"""
extractors/figure_detector.py
-------------------------------
Detects figures that PyMuPDF misses because they are NOT embedded images.

ROOT CAUSE — why figures are missed
-------------------------------------
Academic PDFs contain figures in two fundamentally different ways:

  Type A — Raster images (JPEG/PNG embedded in PDF)
    PyMuPDF block type = 1 (FITZ_BLOCK_IMAGE) ← we already catch these ✓

  Type B — Vector graphics (PDF drawing commands: lines, curves, rectangles)
    PyMuPDF block type = 0 (FITZ_BLOCK_TEXT) or not emitted at all ← MISSED ✗
    Examples: matplotlib plots, TikZ diagrams, chart.js exports, Inkscape SVG

  Type C — XObject / Form XObjects (reusable vector groups)
    Referenced via /XObject in page resources ← also often missed ✗

  Type D — Annotation-based figures (rare in academic PDFs)

This module detects ALL figure types using three complementary strategies:

Strategy 1 — Drawing path analysis (catches Type B)
  PyMuPDF's page.get_drawings() returns all vector drawing commands.
  We cluster spatially proximate drawing paths into "drawing regions".
  A large drawing region with no text inside it = vector figure.

Strategy 2 — XObject scan (catches Type C)
  Inspect page.get_xobjects() and page.resources for form XObjects.
  Map each XObject's bounding box to a FigureBlock.

Strategy 3 — Text-gap analysis (catches any missed figure)
  On pages with figures, there are often large rectangular regions with
  NO text. We find these "empty rectangles" bounded by caption blocks
  and existing figure blocks — these are likely figure areas.

All strategies produce FigureBlock objects with reliable bounding boxes.
"""

from __future__ import annotations
import hashlib
import logging
from dataclasses import dataclass, field
from typing import Optional

from extractors.models import BoundingBox, FigureBlock

logger = logging.getLogger(__name__)

# Minimum area (PDF pts²) to count as a real figure region
MIN_FIGURE_AREA   = 2000.0   # ~45×45 pts minimum
# Drawing paths within this distance (pts) are clustered together
CLUSTER_TOLERANCE = 8.0
# A "drawing region" covering this fraction of page area is considered a figure
MIN_FIGURE_PAGE_FRAC = 0.02


def _make_fig_id(doc_id: str, page_idx: int, strategy: str, seq: int) -> str:
    raw = f"{doc_id}::fig::{strategy}::p{page_idx}::s{seq}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def _bbox_union(boxes: list[BoundingBox]) -> BoundingBox:
    return BoundingBox(
        x0=min(b.x0 for b in boxes),
        y0=min(b.y0 for b in boxes),
        x1=max(b.x1 for b in boxes),
        y1=max(b.y1 for b in boxes),
    )


def _bboxes_overlap(a: BoundingBox, b: BoundingBox, tolerance: float = 0.0) -> bool:
    return (a.x0 - tolerance < b.x1 and a.x1 + tolerance > b.x0 and
            a.y0 - tolerance < b.y1 and a.y1 + tolerance > b.y0)


# ─────────────────────────────────────────────────────────────
# Strategy 1: Drawing-path clustering
# ─────────────────────────────────────────────────────────────

def detect_vector_figures(
    page,                    # fitz.Page
    doc_id: str,
    page_index: int,
    page_width: float,
    page_height: float,
    text_bboxes: list[BoundingBox],
    min_area: float = MIN_FIGURE_AREA,
    cluster_tol: float = CLUSTER_TOLERANCE,
) -> list[FigureBlock]:
    """
    Detect vector figure regions from PDF drawing commands.

    Steps
    -----
    1. Collect all drawing path bounding boxes from page.get_drawings()
    2. Cluster nearby paths (union-find style proximity clustering)
    3. Filter clusters: must be large enough AND mostly text-free
    4. Return each qualifying cluster as a FigureBlock
    """
    try:
        drawings = page.get_drawings()
    except Exception as e:
        logger.warning(f"Could not get drawings for page {page_index}: {e}")
        return []

    if not drawings:
        return []

    # Collect raw bounding boxes from drawing paths
    raw_boxes: list[BoundingBox] = []
    for d in drawings:
        rect = d.get("rect")
        if rect is None:
            continue
        bb = BoundingBox(x0=rect.x0, y0=rect.y0, x1=rect.x1, y1=rect.y1)
        if bb.area < 1.0:   # skip hairlines/dots
            continue
        raw_boxes.append(bb)

    if not raw_boxes:
        return []

    # Greedy proximity clustering
    clusters: list[list[BoundingBox]] = []

    for bb in raw_boxes:
        merged = False
        for cluster in clusters:
            cluster_union = _bbox_union(cluster)
            expanded = BoundingBox(
                x0=cluster_union.x0 - cluster_tol,
                y0=cluster_union.y0 - cluster_tol,
                x1=cluster_union.x1 + cluster_tol,
                y1=cluster_union.y1 + cluster_tol,
            )
            if _bboxes_overlap(expanded, bb):
                cluster.append(bb)
                merged = True
                break
        if not merged:
            clusters.append([bb])

    # Second-pass merge: after initial clustering, expand again
    # (handles cases where intermediate paths weren't present to bridge two clusters)
    changed = True
    while changed:
        changed = False
        new_clusters: list[list[BoundingBox]] = []
        used = set()
        for i, ci in enumerate(clusters):
            if i in used:
                continue
            merged_group = list(ci)
            ui = _bbox_union(merged_group)
            for j, cj in enumerate(clusters):
                if j <= i or j in used:
                    continue
                uj = _bbox_union(cj)
                exp = BoundingBox(
                    x0=ui.x0 - cluster_tol, y0=ui.y0 - cluster_tol,
                    x1=ui.x1 + cluster_tol, y1=ui.y1 + cluster_tol,
                )
                if _bboxes_overlap(exp, uj):
                    merged_group.extend(cj)
                    used.add(j)
                    ui = _bbox_union(merged_group)
                    changed = True
            new_clusters.append(merged_group)
            used.add(i)
        clusters = new_clusters

    figures: list[FigureBlock] = []
    page_area = page_width * page_height

    for seq, cluster in enumerate(clusters):
        union_bb = _bbox_union(cluster)

        # Filter: too small
        if union_bb.area < min_area:
            continue

        # Filter: too small relative to page
        if union_bb.area / page_area < MIN_FIGURE_PAGE_FRAC:
            continue

        # Filter: if this region is mostly covered by text blocks, it's not a figure
        # (e.g., table borders, decorative lines in text areas)
        text_overlap_area = 0.0
        for tb in text_bboxes:
            ix0 = max(union_bb.x0, tb.x0); iy0 = max(union_bb.y0, tb.y0)
            ix1 = min(union_bb.x1, tb.x1); iy1 = min(union_bb.y1, tb.y1)
            inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
            text_overlap_area += inter

        text_coverage = text_overlap_area / max(union_bb.area, 1.0)
        if text_coverage > 0.40:   # >40% covered by text → probably a table or decoration
            continue

        fig_id = _make_fig_id(doc_id, page_index, "vector", seq)
        figures.append(FigureBlock(
            block_id   = fig_id,
            bbox       = union_bb,
            page_index = page_index,
        ))
        logger.debug(
            f"  Vector figure detected: p{page_index} "
            f"bbox=({union_bb.x0:.0f},{union_bb.y0:.0f},"
            f"{union_bb.x1:.0f},{union_bb.y1:.0f}) "
            f"area={union_bb.area:.0f}pts²"
        )

    return figures


# ─────────────────────────────────────────────────────────────
# Strategy 2: XObject scan
# ─────────────────────────────────────────────────────────────

def detect_xobject_figures(
    page,
    doc_id: str,
    page_index: int,
    min_area: float = MIN_FIGURE_AREA,
) -> list[FigureBlock]:
    """
    Detect figures embedded as PDF Form XObjects.

    Many vector graphics tools (Inkscape, Adobe Illustrator) export as
    XObjects rather than inline drawing commands. PyMuPDF's get_images()
    only returns raster images; this function catches form XObjects.
    """
    figures: list[FigureBlock] = []
    try:
        # get_xobjects returns list of (xref, name, invoker, bbox)
        xobjs = page.get_xobjects()
    except Exception:
        return []

    for seq, xobj in enumerate(xobjs):
        try:
            # xobj may be (xref, name) or (xref, name, ...)
            if len(xobj) < 2:
                continue
            xref = xobj[0]
            # Try to get the bounding box of this XObject
            # via the /BBox entry in its stream dictionary
            doc  = page.parent
            bbox_val = doc.xref_get_key(xref, "BBox")
            if not bbox_val or bbox_val[0] != "array":
                continue

            coords = [float(v[1]) for v in bbox_val[1] if v[0] in ("int", "real")]
            if len(coords) < 4:
                continue

            bb = BoundingBox(x0=coords[0], y0=coords[1],
                             x1=coords[2], y1=coords[3])
            if bb.area < min_area:
                continue

            fig_id = _make_fig_id(doc_id, page_index, "xobj", seq)
            figures.append(FigureBlock(
                block_id=fig_id, bbox=bb, page_index=page_index))
        except Exception:
            continue

    return figures


# ─────────────────────────────────────────────────────────────
# Strategy 3: Text-gap analysis
# ─────────────────────────────────────────────────────────────

def detect_textgap_figures(
    text_bboxes: list[BoundingBox],
    existing_figure_bboxes: list[BoundingBox],
    page_width: float,
    page_height: float,
    doc_id: str,
    page_index: int,
    min_gap_height: float = 40.0,   # minimum height of empty region
    min_gap_width:  float = 100.0,  # minimum width of empty region
) -> list[FigureBlock]:
    """
    Find large text-free rectangular regions likely to be figures.

    This is a last-resort strategy for PDFs where figures have no
    drawing commands (e.g., the PDF was created by cropping/masking).

    Algorithm
    ---------
    Sweep y-axis top-to-bottom. Track the current "occupied" y positions
    from text blocks. Large y-gaps between text blocks, combined with
    x-spans that have no text, indicate figure regions.
    """
    if not text_bboxes:
        return []

    # Sort text blocks by y
    sorted_text = sorted(text_bboxes, key=lambda b: b.y0)
    all_bboxes  = sorted_text + existing_figure_bboxes

    figures     = []
    seq         = 0

    # Find vertical gaps
    prev_y1 = 0.0
    for bb in sorted_text:
        gap_top    = prev_y1
        gap_bottom = bb.y0
        gap_height = gap_bottom - gap_top

        if gap_height >= min_gap_height:
            # Find x-extent of this gap (where no text exists in this y-band)
            gap_x0 = 0.0
            gap_x1 = page_width

            # Shrink x-range based on text blocks that partially overlap the y-band
            for tb in sorted_text:
                if tb.y1 < gap_top or tb.y0 > gap_bottom:
                    continue
                # This text block overlaps the gap y-band — it narrows the gap
                # (This handles partial-column figures)

            gap_width = gap_x1 - gap_x0
            if gap_width < min_gap_width:
                prev_y1 = max(prev_y1, bb.y1)
                continue

            candidate = BoundingBox(
                x0=gap_x0, y0=gap_top, x1=gap_x1, y1=gap_bottom
            )

            # Don't create duplicate with already-found figures
            already_covered = any(
                _bboxes_overlap(candidate, fb, tolerance=10.0)
                for fb in existing_figure_bboxes
            )
            if not already_covered:
                fig_id = _make_fig_id(doc_id, page_index, "gap", seq)
                figures.append(FigureBlock(
                    block_id=fig_id, bbox=candidate, page_index=page_index))
                seq += 1
                logger.debug(
                    f"  Gap figure detected: p{page_index} "
                    f"y={gap_top:.0f}→{gap_bottom:.0f} h={gap_height:.0f}pts"
                )

        prev_y1 = max(prev_y1, bb.y1)

    return figures


# ─────────────────────────────────────────────────────────────
# Combined figure detection (all 3 strategies)
# ─────────────────────────────────────────────────────────────

def detect_all_figures(
    page,                         # fitz.Page
    doc_id: str,
    page_index: int,
    page_width: float,
    page_height: float,
    text_bboxes: list[BoundingBox],
    existing_raster_figures: list[FigureBlock],
    use_vector:   bool = True,
    use_xobject:  bool = True,
    use_gap:      bool = True,
    dedup_tolerance: float = 20.0,
) -> list[FigureBlock]:
    """
    Run all figure detection strategies and deduplicate results.

    Parameters
    ----------
    page                    : fitz.Page object
    existing_raster_figures : FigureBlocks already found by PyMuPDF (type=1 blocks)
    use_vector / use_xobject / use_gap : toggle individual strategies

    Returns
    -------
    Merged, deduplicated list of FigureBlocks from all strategies.
    """
    all_figures = list(existing_raster_figures)

    if use_vector:
        vector_figs = detect_vector_figures(
            page=page, doc_id=doc_id, page_index=page_index,
            page_width=page_width, page_height=page_height,
            text_bboxes=text_bboxes,
        )
        all_figures.extend(vector_figs)

    if use_xobject:
        xobj_figs = detect_xobject_figures(
            page=page, doc_id=doc_id, page_index=page_index)
        all_figures.extend(xobj_figs)

    if use_gap:
        existing_bboxes = [f.bbox for f in all_figures]
        gap_figs = detect_textgap_figures(
            text_bboxes=text_bboxes,
            existing_figure_bboxes=existing_bboxes,
            page_width=page_width, page_height=page_height,
            doc_id=doc_id, page_index=page_index,
        )
        all_figures.extend(gap_figs)

    # Deduplicate: remove figures whose bbox heavily overlaps an existing one
    deduped: list[FigureBlock] = []
    for fig in all_figures:
        duplicate = False
        for kept in deduped:
            ix0 = max(fig.bbox.x0, kept.bbox.x0)
            iy0 = max(fig.bbox.y0, kept.bbox.y0)
            ix1 = min(fig.bbox.x1, kept.bbox.x1)
            iy1 = min(fig.bbox.y1, kept.bbox.y1)
            inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
            smaller_area = min(fig.bbox.area, kept.bbox.area)
            if smaller_area > 0 and (inter / smaller_area) > 0.70:
                duplicate = True
                break
        if not duplicate:
            deduped.append(fig)

    n_new = len(deduped) - len(existing_raster_figures)
    if n_new > 0:
        logger.info(
            f"  Figure detection: page {page_index} — "
            f"{len(existing_raster_figures)} raster + {n_new} new "
            f"(vector/xobj/gap) = {len(deduped)} total"
        )

    return deduped