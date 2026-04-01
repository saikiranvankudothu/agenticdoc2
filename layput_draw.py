# extraction visualization
import fitz
import json

# load PDF
pdf = fitz.open("pdf/paper.pdf")

# load json output
with open("output\paper\json\extraction.json") as f:
    data = json.load(f)

for page_data in data["pages"]:
    page_index = page_data["page_index"]
    page = pdf[page_index]

    for block in page_data["text_blocks"]:
        bbox = block["bbox"]

        rect = fitz.Rect(
            bbox["x0"],
            bbox["y0"],
            bbox["x1"],
            bbox["y1"]
        )

        page.draw_rect(rect, color=(1,0,0), width=1)

pdf.save("plots/kimi_output.pdf")



# ==================================================================================>
# visualize_layout_overlay.py
# import fitz
# import json

# pdf = fitz.open("pdf/paper.pdf")

# with open("output/paper/json/layout.json",encoding="utf-8") as f:
#     layout = json.load(f)

# colors = {
#     "title": (1,0,0),
#     "paragraph": (0,0,1),
#     "figure": (0,1,0),
#     "caption": (1,0.5,0)
# }

# for page_data in layout["pages"]:
#     page = pdf[page_data["page_index"]]

#     for region in page_data["regions"]:
#         bbox = region["bbox"]
#         rect = fitz.Rect(
#             bbox["x0"],bbox["y0"],bbox["x1"],bbox["y1"]
#         )

#         color = colors.get(region["region_class"],(0,0,0))
#         page.draw_rect(rect,color=color,width=2)

# pdf.save("plots/kimi_visualized.pdf")


# ======================================================================
