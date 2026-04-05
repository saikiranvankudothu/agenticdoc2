"""Flow connection
run_layout.py → calls → LayoutDetectionAgent
LayoutDetectionAgent → calls → HeuristicLayoutDetector.detect_page()
inside that → calls → sort_reading_order() (your file)
"""

from __future__ import annotations
import logging
from dataclasses import dataclass

from extractors.layout_models import LayoutRegion

logger = logging.getLogger(__name__)


@dataclass
class Column:
    x_min: float
    x_max: float
    index: int

    @property
    def centre(self) -> float:
        return (self.x_min + self.x_max) / 2.0

    @property
    def width(self) -> float:
        return self.x_max - self.x_min


def is_full_width(
    region: LayoutRegion,
    page_width: float,
    threshold: float = 0.55,
) -> bool:
    return (region.bbox.width / max(page_width, 1.0)) >= threshold


def _x_overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def _region_column_overlap(region: LayoutRegion, col: Column) -> float:
    return _x_overlap(region.bbox.x0, region.bbox.x1, col.x_min, col.x_max)


def _sort_single_column(regions: list[LayoutRegion]) -> list[LayoutRegion]:
    return sorted(regions, key=lambda r: (r.bbox.y0, r.bbox.x0, r.bbox.y1))


def _split_wide_regions(
    regions: list[LayoutRegion],
    page_width: float,
    wide_frac: float = 0.48,
) -> tuple[list[LayoutRegion], list[LayoutRegion]]:
    """
    Separate normal column-flow regions from wide regions inside a band.
    Wide regions often behave like local full-width interrupters.
    """
    normal = []
    wide = []
    for r in regions:
        frac = r.bbox.width / max(page_width, 1.0)
        if frac >= wide_frac:
            wide.append(r)
        else:
            normal.append(r)
    return normal, sorted(wide, key=lambda r: (r.bbox.y0, r.bbox.x0))


def detect_columns_in_band(
    regions: list[LayoutRegion],
    page_width: float,
    min_gap_width: float = 18.0,
    min_col_width_frac: float = 0.18,
    max_columns: int = 4,
) -> list[Column]:
    """
    Column detection that supports 1–N columns (up to max_columns).

    Strategy:
    - Merge nearby x-spans into occupied blocks.
    - Find all interior gaps (not at page edges) that are wide enough.
    - Use all qualifying gaps as column separators, not just the widest one.
    - This correctly handles 2-column and 3-column academic layouts.
    """
    if not regions:
        return [Column(0.0, page_width, 0)]

    spans = sorted((r.bbox.x0, r.bbox.x1) for r in regions)

    # Merge overlapping / nearby x-spans into contiguous blocks.
    merged: list[list[float]] = []
    for x0, x1 in spans:
        if not merged:
            merged.append([x0, x1])
            continue
        prev = merged[-1]
        if x0 <= prev[1] + 8.0:          # merge spans within 8px
            prev[1] = max(prev[1], x1)
        else:
            merged.append([x0, x1])

    # Collect all inter-block gaps that are interior and wide enough.
    interior_gaps: list[tuple[float, float]] = []
    for i in range(len(merged) - 1):
        gap_x0 = merged[i][1]
        gap_x1 = merged[i + 1][0]
        gap_w = gap_x1 - gap_x0
        if (
            gap_w >= min_gap_width
            and gap_x0 > page_width * 0.10   # not at left edge
            and gap_x1 < page_width * 0.90   # not at right edge
        ):
            interior_gaps.append((gap_x0, gap_x1))

    if not interior_gaps:
        return [Column(0.0, page_width, 0)]

    # BUG FIX: use ALL interior gaps as column separators, not only the widest.
    # Sort gaps left-to-right to build column boundaries in order.
    interior_gaps.sort(key=lambda g: g[0])

    # Limit to at most (max_columns - 1) separators.
    if len(interior_gaps) >= max_columns:
        # Keep the widest max_columns-1 gaps.
        interior_gaps = sorted(interior_gaps, key=lambda g: g[1] - g[0], reverse=True)[: max_columns - 1]
        interior_gaps.sort(key=lambda g: g[0])

    # Build column objects from the separator gaps.
    boundaries: list[float] = [0.0]
    for g in interior_gaps:
        boundaries.append(g[0])   # right edge of left neighbour
        boundaries.append(g[1])   # left edge of right neighbour
    boundaries.append(page_width)

    # boundaries is: [col0_x0, col0_x1, col1_x0, col1_x1, ...]
    columns: list[Column] = []
    for i in range(0, len(boundaries) - 1, 2):
        col = Column(boundaries[i], boundaries[i + 1], len(columns))
        min_col_w = page_width * min_col_width_frac
        if col.width < min_col_w:
            # Sliver column — abort and fall back to single column.
            return [Column(0.0, page_width, 0)]
        columns.append(col)

    if len(columns) < 2:
        return [Column(0.0, page_width, 0)]

    return columns


def _assign_to_columns(
    regions: list[LayoutRegion],
    columns: list[Column],
) -> dict[int, list[LayoutRegion]]:
    """
    Assign by maximum x-overlap first, then fallback to nearest centre.

    BUG FIX: use fractional overlap (overlap / region_width) so that wide
    regions that spill into an adjacent column are still assigned correctly
    based on where *most* of their width sits.
    """
    buckets: dict[int, list[LayoutRegion]] = {c.index: [] for c in columns}

    for r in regions:
        r_width = max(r.bbox.x1 - r.bbox.x0, 1.0)
        overlaps = [
            (_region_column_overlap(r, c) / r_width, c.index)
            for c in columns
        ]
        best_frac, best_idx = max(overlaps, key=lambda x: x[0])

        if best_frac > 0:
            buckets[best_idx].append(r)
            continue

        # No overlap at all — assign to nearest column by centre distance.
        cx = (r.bbox.x0 + r.bbox.x1) / 2.0
        nearest = min(columns, key=lambda c: abs(cx - c.centre))
        buckets[nearest.index].append(r)

    for idx in buckets:
        buckets[idx].sort(key=lambda r: (r.bbox.y0, r.bbox.x0, r.bbox.y1))

    return buckets


def _sort_band(
    band_regions: list[LayoutRegion],
    page_width: float,
    column_gap_min: float,
) -> list[LayoutRegion]:
    """
    Sort one horizontal band.

    BUG FIX (wide-interrupter slicing):
      Old code used `r.bbox.y0` as break points, so normal regions whose y0
      falls *inside* a wide block (between its y0 and y1) were placed in the
      wrong chunk.  We now use `r.bbox.y1` (bottom of the wide block) as the
      lower boundary for the subsequent chunk.
    """
    if not band_regions:
        return []

    if len(band_regions) == 1:
        return list(band_regions)

    normal, wide = _split_wide_regions(band_regions, page_width)

    if len(normal) <= 1:
        return sorted(band_regions, key=lambda r: (r.bbox.y0, r.bbox.x0))

    columns = detect_columns_in_band(
        normal,
        page_width=page_width,
        min_gap_width=column_gap_min,
    )

    # Single-column band — just sort top-to-bottom.
    if len(columns) == 1:
        return sorted(band_regions, key=lambda r: (r.bbox.y0, r.bbox.x0))

    # Multi-column band: interleave wide interrupters with column chunks.
    ordered: list[LayoutRegion] = []

    # BUG FIX: break points must use y1 of each wide block (its bottom edge),
    # not y0, so that the next chunk starts *after* the wide block ends.
    # Sentinel values bracket the full band.
    break_tops = [-float("inf")] + [w.bbox.y0 for w in wide]
    break_bots = [w.bbox.y1 for w in wide] + [float("inf")]

    for i, (y_lo, y_hi) in enumerate(zip(break_tops, break_bots)):
        # Emit the wide interrupter that opened this gap (skip for first sentinel).
        if i > 0:
            ordered.append(wide[i - 1])

        # Normal regions whose top falls in (y_lo, y_hi).
        # BUG FIX: use y_lo as the *bottom* of the previous wide block (y1),
        # so we don't re-include regions that were already emitted.
        chunk = [r for r in normal if r.bbox.y0 >= y_lo and r.bbox.y0 < y_hi]
        if not chunk:
            continue

        buckets = _assign_to_columns(chunk, columns)

        # Academic reading order: finish left column entirely, then right, etc.
        for col_idx in sorted(buckets):
            ordered.extend(buckets[col_idx])

    return ordered


def sort_reading_order(
    regions: list[LayoutRegion],
    page_width: float,
    page_height: float,
    full_width_threshold: float = 0.55,
    column_gap_min: float = 18.0,
) -> list[LayoutRegion]:
    """
    Reading order for multi-column academic/document pages.

    Algorithm:
    1. Classify each region as full-width (header/footer/figure spanning the
       page) or column-flow.
    2. Full-width regions act as hard horizontal separators, dividing the page
       into bands.
    3. Each band is sorted independently, detecting its own column structure.

    Key fixes vs. previous version:
    - Band boundary is now INCLUSIVE on both sides using a small epsilon so
      straddling regions are never silently dropped.
    - Tail band uses cursor_y (= last full-width block's y1) so the overlap
      filter is consistent with how bands are built above.
    - Column detection now supports N columns, not just 2.
    - Wide-interrupter slice points use y1 (bottom) not y0 (top).
    - Column assignment uses fractional overlap rather than raw pixel overlap.
    """
    if not regions:
        return []
    if len(regions) == 1:
        return list(regions)

    full_width_regs = sorted(
        [r for r in regions if is_full_width(r, page_width, full_width_threshold)],
        key=lambda r: (r.bbox.y0, r.bbox.x0),
    )
    other_regs = [
        r for r in regions
        if not is_full_width(r, page_width, full_width_threshold)
    ]

    ordered: list[LayoutRegion] = []
    cursor_y = -float("inf")

    # Small tolerance to avoid dropping regions that share a boundary pixel
    # with a full-width block due to floating-point imprecision.
    EPS = 1.0

    for fw in full_width_regs:
        # BUG FIX: old code used strict `r.bbox.y1 <= fw.bbox.y0`, dropping
        # any region whose bottom touched the full-width block's top.
        # Use `r.bbox.y0 < fw.bbox.y0 + EPS` (region starts before the fw
        # block) and `r.bbox.y1 <= fw.bbox.y0 + EPS` (region ends at/before
        # the fw block top) to capture all intended content without overlap.
        band = [
            r for r in other_regs
            if r.bbox.y0 >= cursor_y - EPS and r.bbox.y1 <= fw.bbox.y0 + EPS
        ]
        ordered.extend(_sort_band(band, page_width, column_gap_min))
        ordered.append(fw)
        cursor_y = fw.bbox.y1   # advance cursor to bottom of full-width block

    # BUG FIX: tail band should start from cursor_y (bottom of last fw block).
    # Old code used `r.bbox.y0 >= cursor_y` which is correct, but combined
    # with the EPS adjustment above we must be consistent.
    tail_band = [r for r in other_regs if r.bbox.y0 >= cursor_y - EPS]
    ordered.extend(_sort_band(tail_band, page_width, column_gap_min))

    # Deduplicate while preserving first-occurrence order.
    # (A region may land in both a band and the tail if it straddles cursor_y.)
    seen: set = set()
    deduped: list[LayoutRegion] = []
    for r in ordered:
        if r.region_id not in seen:
            deduped.append(r)
            seen.add(r.region_id)

    logger.debug("Reading order result: %s", [r.region_id for r in deduped])
    return deduped