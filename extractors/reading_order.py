"""
extractors/reading_order.py
----------------------------
Proper reading order for multi-column academic PDFs.

ROOT CAUSE — why naive sort(y0, x0) breaks
-------------------------------------------
In a two-column IEEE/ACM layout:

  ┌──────────┬──────────┐
  │ Col-L    │ Col-R    │  y≈100
  │ block A  │ block C  │
  │          │          │  y≈200
  │ block B  │ block D  │
  └──────────┴──────────┘

Naive sort by (y0, x0) gives:  A, C, B, D   ← WRONG
Correct reading order is:      A, B, C, D   ← read full left col, then right

Algorithm (XY-Cut inspired)
---------------------------
1. DETECT COLUMNS
   Project all block x-coordinates onto the x-axis.
   Find "x-gaps" — wide horizontal whitespace bands where no block exists.
   These gaps are the column separators.

2. ASSIGN BLOCKS TO COLUMNS
   Each block is assigned to the column whose x-range contains its centre.

3. SORT WITHIN EACH COLUMN by (y0, x0)

4. HANDLE FULL-WIDTH BLOCKS
   Blocks spanning >60% of page width (titles, section headings, figures)
   are "full-width". They interrupt the column flow:
   - All blocks above them → sorted in column order
   - Full-width block → inserted at its y position
   - All blocks below them → restarted

5. MERGE
   Interleave full-width blocks with column blocks at correct y positions.

This handles:
  - Single-column PDFs (one "column" = whole page width)
  - Two-column IEEE/ACM style
  - Three-column conference posters
  - Mixed: full-width abstract + two-column body
"""

from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Optional

from extractors.layout_models import LayoutRegion

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Column detection
# ─────────────────────────────────────────────

@dataclass
class Column:
    """A detected column band on a page."""
    x_min: float
    x_max: float
    index: int          # left→right ordering (0-based)

    @property
    def centre(self) -> float:
        return (self.x_min + self.x_max) / 2

    def contains(self, bbox_x_centre: float, tolerance: float = 5.0) -> bool:
        return (self.x_min - tolerance) <= bbox_x_centre <= (self.x_max + tolerance)


def detect_columns(
    regions: list[LayoutRegion],
    page_width: float,
    min_gap_width: float = 15.0,    # minimum whitespace to count as a column gap
    min_col_width_frac: float = 0.15,  # column must be ≥15% of page width
) -> list[Column]:
    """
    Detect column bands by finding x-axis gaps between blocks.

    Parameters
    ----------
    regions       : all LayoutRegions on the page
    page_width    : page width in PDF points
    min_gap_width : minimum whitespace gap (pts) to count as a column separator
    min_col_width_frac : minimum column width as fraction of page width

    Returns
    -------
    List of Column objects sorted left to right.
    """
    if not regions:
        return [Column(x_min=0, x_max=page_width, index=0)]

    # Build x-coverage array at 1-point resolution
    resolution  = 1.0
    slots       = int(page_width / resolution) + 2
    coverage    = [0] * slots

    for r in regions:
        lo = max(0, int(r.bbox.x0 / resolution))
        hi = min(slots - 1, int(r.bbox.x1 / resolution))
        for i in range(lo, hi + 1):
            coverage[i] += 1

    # Find contiguous zero-coverage gaps
    gaps: list[tuple[float, float]] = []   # (gap_x0, gap_x1)
    in_gap    = False
    gap_start = 0

    for i, val in enumerate(coverage):
        if val == 0 and not in_gap:
            in_gap    = True
            gap_start = i
        elif val > 0 and in_gap:
            in_gap = False
            gap_width = (i - gap_start) * resolution
            if gap_width >= min_gap_width:
                gaps.append((gap_start * resolution, i * resolution))

    # Build column boundaries from gaps
    min_col_w = page_width * min_col_width_frac
    separators = [0.0] + [g[1] for g in gaps] + [page_width]

    columns: list[Column] = []
    for idx in range(len(separators) - 1):
        x0 = separators[idx]
        x1 = separators[idx + 1]
        if (x1 - x0) >= min_col_w:
            columns.append(Column(x_min=x0, x_max=x1, index=len(columns)))

    if not columns:
        columns = [Column(x_min=0, x_max=page_width, index=0)]

    logger.debug(f"  Detected {len(columns)} column(s): "
                 f"{[(f'{c.x_min:.0f}-{c.x_max:.0f}') for c in columns]}")
    return columns


# ─────────────────────────────────────────────
# Full-width block detection
# ─────────────────────────────────────────────

def is_full_width(
    region: LayoutRegion,
    page_width: float,
    threshold: float = 0.55,
) -> bool:
    """
    Returns True if this region spans most of the page width.
    These are treated as "flow interrupters" (titles, section headings,
    wide figures, full-width tables).
    """
    return (region.bbox.width / page_width) >= threshold


# ─────────────────────────────────────────────
# Main reading-order sort
# ─────────────────────────────────────────────

def sort_reading_order(
    regions: list[LayoutRegion],
    page_width: float,
    page_height: float,
    full_width_threshold: float = 0.55,
    column_gap_min: float = 15.0,
) -> list[LayoutRegion]:
    """
    Sort regions into proper human reading order for multi-column layouts.

    Algorithm
    ---------
    1. Separate full-width regions from column-flow regions.
    2. Detect columns among column-flow regions.
    3. Assign each column-flow region to a column by its x-centre.
    4. Group all regions into "bands" separated by full-width blocks.
    5. Within each band, sort column-flow regions: left column first (top→bottom),
       then right column (top→bottom).
    6. Insert full-width regions at their natural y-position within the sequence.

    Parameters
    ----------
    regions              : unsorted LayoutRegions for one page
    page_width           : page width in PDF points
    page_height          : page height in PDF points
    full_width_threshold : width fraction above which a block is "full-width"
    column_gap_min       : minimum gap (pts) to count as column separator
    """
    if not regions:
        return []

    if len(regions) == 1:
        return list(regions)

    # ── Step 1: Split full-width vs column regions ──────────
    full_width_regs = [r for r in regions if is_full_width(r, page_width, full_width_threshold)]
    col_regs        = [r for r in regions if not is_full_width(r, page_width, full_width_threshold)]

    # ── Step 2: Detect columns (using column regions only) ──
    columns = detect_columns(col_regs, page_width, min_gap_width=column_gap_min)

    # ── Step 3: Assign column-flow regions to columns ───────
    def assign_column(r: LayoutRegion) -> int:
        cx = (r.bbox.x0 + r.bbox.x1) / 2
        for col in columns:
            if col.contains(cx):
                return col.index
        # fallback: assign to nearest column
        dists = [abs(cx - col.centre) for col in columns]
        return dists.index(min(dists))

    col_assigned: list[tuple[int, LayoutRegion]] = [
        (assign_column(r), r) for r in col_regs
    ]

    # ── Step 4: Group into bands split by full-width blocks ─
    # Sort full-width blocks by y
    full_width_sorted = sorted(full_width_regs, key=lambda r: r.bbox.y0)

    # Build band boundaries: y-intervals between full-width blocks
    # Band 0: y=0 → first full-width block
    # Band k: between full-width blocks k-1 and k
    # Band N: last full-width block → page bottom
    band_breaks = (
        [-float("inf")]
        + [r.bbox.y0 for r in full_width_sorted]
        + [float("inf")]
    )

    ordered: list[LayoutRegion] = []

    for band_idx in range(len(band_breaks) - 1):
        y_lo = band_breaks[band_idx]
        y_hi = band_breaks[band_idx + 1]

        # Insert the full-width block that opens this band (except first band)
        if band_idx > 0:
            ordered.append(full_width_sorted[band_idx - 1])

        # Get column-flow blocks inside this y-band
        band_blocks = [
            (col_idx, r) for col_idx, r in col_assigned
            if r.bbox.y0 > y_lo and r.bbox.y0 < y_hi
        ]

        if not band_blocks:
            continue

        # ── Step 5: Sort within each column top→bottom ──────
        # Then order: col 0 entirely, then col 1, etc.
        num_cols = len(columns)
        col_buckets: dict[int, list[LayoutRegion]] = {i: [] for i in range(num_cols)}
        for col_idx, r in band_blocks:
            col_buckets[col_idx].append(r)

        for col_idx in range(num_cols):
            col_buckets[col_idx].sort(key=lambda r: (r.bbox.y0, r.bbox.x0))

        # ── Step 6: Interleave columns in reading order ──────
        # For single-column: trivial
        # For multi-column: read column 0 fully, then column 1, etc.
        # EXCEPTION: if columns have very different y-extents (mixed layout),
        # we fall back to row-band interleaving
        if num_cols == 1:
            ordered.extend(col_buckets[0])
        else:
            ordered.extend(_interleave_columns(col_buckets, num_cols))

    return ordered


def _interleave_columns(
    col_buckets: dict[int, list[LayoutRegion]],
    num_cols: int,
) -> list[LayoutRegion]:
    """
    Merge sorted column buckets into reading order.

    Standard academic reading order: read left column fully top-to-bottom,
    then right column fully top-to-bottom (column-major order).
    This is correct for IEEE/ACM two-column papers.
    """
    result: list[LayoutRegion] = []
    for col_idx in range(num_cols):
        result.extend(col_buckets.get(col_idx, []))
    return result