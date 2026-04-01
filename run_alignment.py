"""run_alignment.py — Step 4: Text–Layout Alignment"""
import json, sys, os
sys.path.insert(0, os.path.dirname(__file__))

from run_layout import load_extraction_from_json
from extractors.text_layout_aligner import align
from agents.layout_detection_agent import LayoutDetectionAgent
from extractors.layout_models import DocumentLayoutResult

def load_layout_from_json(path: str) -> DocumentLayoutResult:
    from extractors.layout_models import (
        DocumentLayoutResult, PageLayoutResult, LayoutRegion,
        RegionClass, DetectionBackend
    )
    from extractors.models import BoundingBox
    with open(path) as f:
        data = json.load(f)
    pages = []
    for pg in data["pages"]:
        regions = []
        for r in pg["regions"]:
            bb = r["bbox"]
            regions.append(LayoutRegion(
                region_id        = r["region_id"],
                region_class     = RegionClass(r["region_class"]),
                bbox             = BoundingBox(**bb),
                page_index       = r["page_index"],
                confidence       = r["confidence"],
                backend          = DetectionBackend(r["backend"]),
                text_content     = r.get("text_content"),
                source_block_ids = r.get("source_block_ids", []),
            ))
        pages.append(PageLayoutResult(
            page_index = pg["page_index"],
            width      = pg["width"],
            height     = pg["height"],
            regions    = regions,
        ))
    return DocumentLayoutResult(
        doc_id      = data["doc_id"],
        source_path = data["source_path"],
        total_pages = data["total_pages"],
        pages       = pages,
    )

if __name__ == "__main__":
    extraction = load_extraction_from_json("output/paper/json/extraction.json")
    layout     = load_layout_from_json("output/paper/json/layout.json")

    result = align(extraction, layout)

    stats = result.stats()
    print(f"\nAlignment complete:")
    print(f"  Aligned   : {stats['total_aligned']}")
    print(f"  Unassigned: {stats['total_unassigned']}")
    print(f"  Flagged   : {stats['flagged']}")

    # Save
    import json as _json
    out = [{"block_id": a.block_id, "region_id": a.region_id,
            "region_class": a.region_class.value, "score": round(a.score, 4),
            "page_index": a.page_index, "flagged": a.flagged}
           for a in result.alignments]
    with open("output/paper/json/alignment.json", "w") as f:
        _json.dump({"doc_id": result.doc_id, "alignments": out,
                    "unassigned": result.unassigned}, f, indent=2)
    print("  Saved → output/paper/json/alignment.json")