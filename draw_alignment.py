"""
draw_alignment.py
------------------
Visualize Step 4 alignment output.

Shows:
  - Layout regions (colored by class)
  - Text block bounding boxes (thin dashed overlay)
  - Alignment score printed on each matched block
  - Flagged (unassigned) blocks in red
  - Score legend per page

Output: plots/paper_alignment.pdf
"""
import fitz
import json

# ── Load files ────────────────────────────────────────────
pdf        = fitz.open("pdf/paper.pdf")

with open("output/paper/json/layout.json", encoding="utf-8") as f:
    layout = json.load(f)

with open("output/paper/json/extraction.json", encoding="utf-8") as f:
    extraction = json.load(f)

with open("output/paper/json/alignment.json", encoding="utf-8") as f:
    alignment = json.load(f)

# ── Color map (region class → RGB) ───────────────────────
REGION_COLORS = {
    "title":     (0.8, 0.1, 0.1),   # red
    "paragraph": (0.1, 0.3, 0.9),   # blue
    "figure":    (0.1, 0.7, 0.2),   # green
    "caption":   (1.0, 0.5, 0.0),   # orange
    "equation":  (0.5, 0.0, 0.7),   # purple
    "table":     (0.0, 0.7, 0.7),   # teal
    "list":      (0.9, 0.7, 0.0),   # amber
    "header":    (0.5, 0.5, 0.5),   # gray
    "footer":    (0.5, 0.5, 0.5),   # gray
    "reference": (0.2, 0.6, 0.5),   # muted teal
    "abstract":  (0.6, 0.2, 0.4),   # mauve
}

# ── Build lookup maps ─────────────────────────────────────
# region_id → region dict
region_map = {
    r["region_id"]: r
    for pg in layout["pages"]
    for r in pg["regions"]
}

# block_id → alignment record
align_map = {
    a["block_id"]: a
    for a in alignment["alignments"]
}

# page_index → list of text blocks
blocks_by_page = {}
for pg in extraction["pages"]:
    blocks_by_page[pg["page_index"]] = pg["text_blocks"]

# ── Draw ──────────────────────────────────────────────────
for page_data in layout["pages"]:
    page_idx = page_data["page_index"]
    page     = pdf[page_idx]

    # Step 1: Draw filled region boxes (semi-transparent feel via thin border)
    for region in page_data["regions"]:
        bb    = region["bbox"]
        cls   = region["region_class"]
        color = REGION_COLORS.get(cls, (0.3, 0.3, 0.3))
        rect  = fitz.Rect(bb["x0"], bb["y0"], bb["x1"], bb["y1"])

        # Filled region background (very light)
        page.draw_rect(rect, color=color, fill=color, fill_opacity=0.06, width=0)
        # Region border (solid, 1.5px)
        page.draw_rect(rect, color=color, width=1.5)
        # Region class label
        page.insert_text(
            (bb["x0"] + 2, bb["y0"] + 8),
            cls.upper(),
            fontsize=6,
            color=color,
        )

    # Step 2: Draw text block boxes with alignment score
    for blk in blocks_by_page.get(page_idx, []):
        bb      = blk["bbox"]
        rect    = fitz.Rect(bb["x0"], bb["y0"], bb["x1"], bb["y1"])
        a_rec   = align_map.get(blk["block_id"])

        if a_rec and not a_rec["flagged"]:
            score = a_rec["score"]
            # Green tint for high score, yellow for medium
            if score >= 0.7:
                box_color = (0.0, 0.6, 0.0)
            elif score >= 0.45:
                box_color = (0.7, 0.5, 0.0)
            else:
                box_color = (0.9, 0.3, 0.0)

            # Thin dashed-style overlay (draw twice for dash effect)
            page.draw_rect(rect, color=box_color, width=0.6)

            # Score label at bottom-right of block
            score_label = f"{score:.2f}"
            page.insert_text(
                (bb["x1"] - 18, bb["y1"] - 1),
                score_label,
                fontsize=5,
                color=box_color,
            )
        else:
            # Flagged / unassigned — bright red
            page.draw_rect(rect, color=(1, 0, 0), width=1.2)
            page.insert_text(
                (bb["x0"] + 1, bb["y1"] - 1),
                "UNASSIGNED",
                fontsize=5,
                color=(1, 0, 0),
            )

    # Step 3: Page legend (bottom-left)
    legend_items = list(REGION_COLORS.items())
    lx, ly = 5, page.rect.height - 5
    for cls_name, col in legend_items:
        page.draw_rect(
            fitz.Rect(lx, ly - 7, lx + 8, ly),
            color=col, fill=col, width=0
        )
        page.insert_text((lx + 10, ly - 1), cls_name, fontsize=5, color=col)
        lx += 52
        if lx > page.rect.width - 60:
            lx  = 5
            ly -= 10

# ── Save ─────────────────────────────────────────────────
pdf.save("plots/paper_alignment.pdf")
print("Saved → plots/paper_alignment.pdf")