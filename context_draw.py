# import fitz
# import json

# pdf = fitz.open("pdf/paper.pdf")

# with open("output/paper/json/layout.json", "r", encoding="utf-8") as f:
#     layout = json.load(f)

# colors = {
#     "title": (1,0,0),
#     "paragraph": (0,0,1),
#     "figure": (0,1,0),
#     "caption": (1,0.5,0),
#     "equation": (0.5,0,0.5),
#     "table": (0,0.7,0.7),
# }

# for page_data in layout["pages"]:

#     page = pdf[page_data["page_index"]]

#     for region in page_data["regions"]:

#         bbox = region["bbox"]

#         rect = fitz.Rect(
#             bbox["x0"],
#             bbox["y0"],
#             bbox["x1"],
#             bbox["y1"]
#         )

#         cls = region["region_class"]
#         color = colors.get(cls,(0,0,0))

#         page.draw_rect(rect,color=color,width=2)

#         page.insert_text(
#             (bbox["x0"],bbox["y0"]-5),
#             cls.upper(),
#             fontsize=8
#         )

# pdf.save("plots/paper_yolo_dit.pdf")
# =====================================================
import fitz
import json

pdf = fitz.open("pdf/major_paper.pdf")

with open("output/major_paper/json/layout.json",encoding="utf-8") as f:
    layout = json.load(f)

for page_data in layout["pages"]:

    page = pdf[page_data["page_index"]]

    regions = sorted(
        page_data["regions"],
        key=lambda r:(r["bbox"]["y0"],r["bbox"]["x0"])
    )

    for i, region in enumerate(regions):

        bbox = region["bbox"]

        x = (bbox["x0"] + bbox["x1"]) / 2
        y = (bbox["y0"] + bbox["y1"]) / 2

        page.insert_text(
            (x,y),
            str(i+1),
            fontsize=10
        )

pdf.save("plots/major_paper_reading_order.pdf")