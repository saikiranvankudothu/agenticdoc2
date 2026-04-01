"""
extractors/ml_layout_detector.py
----------------------------------
ML-based layout region detector supporting three backends:

  Backend A — LayoutParser + PubLayNet (Detectron2)
    install: pip install layoutparser[layoutmodels] detectron2
    model  : lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config

  Backend B — Microsoft DiT (via HuggingFace transformers)
    install: pip install transformers torch torchvision
    model  : microsoft/dit-base-finetuned-doclaynet

  Backend C — Heuristic fallback (always available, zero deps)

Selection logic
---------------
  1. Try LayoutParser → if unavailable, try DiT → if unavailable, use heuristic
  You can also force a specific backend via the `backend` constructor arg.

All backends produce the same LayoutRegion output — the agent
is backend-agnostic from the caller's perspective.

Paper reference (Section III / V-E):
  "LayoutLM-style features for layout-aware encoding"
  "Donut or other vision-language encoders can be integrated
   without necessitating modifications to the orchestration
   or retrieval components."
"""

from __future__ import annotations
import logging
import os
import torch
import hashlib
from pathlib import Path
from typing import Optional, Literal
from PIL import Image
import sys
from extractors.models import BoundingBox, PageExtractionResult
from extractors.layout_models import (
    LayoutRegion, PageLayoutResult, RegionClass, DetectionBackend,
)
from extractors.heuristic_layout_detector import HeuristicLayoutDetector


torch.set_num_threads(os.cpu_count() or 4)  # Use all available CPU cores
torch.set_num_interop_threads(2)  # Reduce inter-op parallelism for better CPU efficiency


logger = logging.getLogger(__name__)

# ── Label maps for each backend ────────────────────────────

# PubLayNet 5-class labels (LayoutParser default)
PUBLAYNET_LABEL_MAP: dict[int, RegionClass] = {
    0: RegionClass.PARAGRAPH,
    1: RegionClass.TITLE,
    2: RegionClass.LIST,
    3: RegionClass.TABLE,
    4: RegionClass.FIGURE,
}

# DocLayNet 11-class labels (DiT / DocLayNet models)
DOCLAYNET_LABEL_MAP: dict[str, RegionClass] = {
    "caption":        RegionClass.CAPTION,
    "footnote":       RegionClass.FOOTER,
    "formula":        RegionClass.EQUATION,
    "list-item":      RegionClass.LIST,
    "page-footer":    RegionClass.FOOTER,
    "page-header":    RegionClass.HEADER,
    "picture":        RegionClass.FIGURE,
    "section-header": RegionClass.TITLE,
    "table":          RegionClass.TABLE,
    "text":           RegionClass.PARAGRAPH,
    "title":          RegionClass.TITLE,
}


def _region_id(doc_id: str, page_idx: int, seq: int) -> str:
    return hashlib.md5(
        f"{doc_id}::ml::p{page_idx}::r{seq}".encode()
    ).hexdigest()[:12]


def _pdf_bbox_from_pixel(
    px_box: tuple,
    img_w: int,
    img_h: int,
    page_w: float,
    page_h: float,
) -> BoundingBox:
    """
    Convert pixel-space bounding box (from ML model) to PDF-point space.
    px_box = (x1, y1, x2, y2) in pixels.
    """
    sx = page_w / img_w
    sy = page_h / img_h
    return BoundingBox(
        x0=px_box[0] * sx,
        y0=px_box[1] * sy,
        x1=px_box[2] * sx,
        y1=px_box[3] * sy,
    )


# ─────────────────────────────────────────────────────────────
# Backend A: LayoutParser + PubLayNet / Detectron2
# ─────────────────────────────────────────────────────────────

class LayoutParserDetector:
    """
    Wraps LayoutParser with a pretrained PubLayNet model.

    Installation
    ------------
    pip install layoutparser[layoutmodels]
    pip install 'git+https://github.com/facebookresearch/detectron2.git'

    Model options (set via model_path)
    -----------------------------------
    "lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config"     ← default, fast
    "lp://PubLayNet/mask_rcnn_R_50_FPN_3x/config"       ← better masks
    "lp://PubLayNet/mask_rcnn_X_101_32x8d_FPN_3x/config"← best accuracy
    """

    MODEL_PATH = "lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config"
    SCORE_THR  = 0.70   # minimum detection confidence to keep a region

    def __init__(self, model_path: Optional[str] = None, score_threshold: float = 0.70):
        self.model_path      = model_path or self.MODEL_PATH
        self.score_threshold = score_threshold
        self._model          = None

    def _load(self):
        """Lazy-load the model on first use."""
        if self._model is not None:
            return
        try:
            import layoutparser as lp  # type: ignore
            self._model = lp.Detectron2LayoutModel(
                config_path  = self.model_path,
                extra_config = ["MODEL.ROI_HEADS.SCORE_THRESH_TEST", self.score_threshold],
                label_map    = PUBLAYNET_LABEL_MAP,
            )
            logger.info("LayoutParser model loaded: %s", self.model_path)
        except Exception as e:
            raise RuntimeError(
                f"Could not load LayoutParser model: {e}\n"
                "Install with: pip install layoutparser[layoutmodels] detectron2"
            )

    def detect(
        self,
        page_image,          # PIL.Image of the rendered page
        page: PageExtractionResult,
        doc_id: str,
    ) -> list[LayoutRegion]:
        """
        Run LayoutParser on a rendered page image.

        Parameters
        ----------
        page_image : PIL.Image — full-page raster at 150+ DPI
        page       : PageExtractionResult — provides page dimensions
        doc_id     : document identifier

        Returns
        -------
        list[LayoutRegion] in PDF-point coordinate space
        """
        self._load()
        import layoutparser as lp  # type: ignore

        layout   = self._model.detect(page_image)
        img_w, img_h = page_image.size
        regions  = []

        for seq, block in enumerate(layout):
            score = float(block.score)
            if score < self.score_threshold:
                continue

            # block.type is already mapped via label_map to RegionClass values
            try:
                cls = RegionClass(block.type.lower().replace(" ", "_"))
            except ValueError:
                cls = RegionClass.UNKNOWN

            # Convert pixel bbox → PDF points
            px = block.coordinates   # (x1, y1, x2, y2)
            bbox = _pdf_bbox_from_pixel(
                px, img_w, img_h, page.width, page.height
            )

            regions.append(LayoutRegion(
                region_id    = _region_id(doc_id, page.page_index, seq),
                region_class = cls,
                bbox         = bbox,
                page_index   = page.page_index,
                confidence   = score,
                backend      = DetectionBackend.LAYOUTPARSER,
            ))

        logger.debug(
            "LayoutParser: page %d → %d regions", page.page_index, len(regions)
        )
        return regions

# ─────────────────────────────────────────────────────────────
# Backend B: Microsoft DiT (HuggingFace Transformers)
# ─────────────────────────────────────────────────────────────

class DiTLayoutDetector:
    """
    Wraps Microsoft's Document Image Transformer (DiT) fine-tuned on DocLayNet.

    Installation
    ------------
    pip install transformers torch torchvision

    The model classifies image patches and produces bounding boxes
    via an object-detection head. It uses DocLayNet's 11-class label set
    which gives better granularity for academic PDFs than PubLayNet.

    HuggingFace model: microsoft/dit-base-finetuned-doclaynet
    """

    MODEL_ID   = "cmarkea/detr-layout-detection"
    SCORE_THR  = 0.60

    def __init__(self, model_id: Optional[str] = None, score_threshold: float = 0.60):
        self.model_id        = model_id or self.MODEL_ID
        self.score_threshold = score_threshold
        self._processor      = None
        self._model          = None
        self._device         = "cpu"  # Force CPU for consistency

    def _load(self):
        if self._model is not None:
            return
        try:
            from transformers import AutoImageProcessor, AutoModelForObjectDetection
            
            self._processor = AutoImageProcessor.from_pretrained(self.model_id)
            self._model     = AutoModelForObjectDetection.from_pretrained(
                self.model_id,
                torch_dtype=torch.float32,  # Use float32 for CPU compatibility
            )
            
            # CPU Optimization: Set model to eval mode and optimize
            self._model.eval()
            self._model.to(self._device)
            
            # Optional: Enable torch.compile for Python 3.11+ (further speedup)
            if hasattr(torch, 'compile') and sys.platform != 'win32':
                try:
                    self._model = torch.compile(self._model, mode="reduce-overhead")
                    logger.info("DiT model compiled for CPU optimization")
                except Exception as e:
                    logger.warning(f"Could not compile model: {e}")
            else:
                logger.info("DiT model loaded without compilation (Windows or compile unavailable)")
                        
            logger.info(
                "DiT model loaded: %s (device=%s, threads=%d)",
                self.model_id, 
                self._device,
                torch.get_num_threads()
            )
            
        except Exception as e:
            raise RuntimeError(
                f"Could not load DiT model: {e}\n"
                "Install: pip install transformers torch torchvision"
            )

    def detect(
        self,
        page_image,
        page: PageExtractionResult,
        doc_id: str,
    ) -> list[LayoutRegion]:
        """Run DiT object detection on a rendered page image - CPU optimized."""
        self._load()
        
        # CPU Optimization: Process at slightly lower resolution if image is large
        max_size = 1200  # Limit max dimension for CPU speed
        img_w, img_h = page_image.size
        
        if max(img_w, img_h) > max_size:
            # Resize proportionally to reduce CPU load
            scale = max_size / max(img_w, img_h)
            new_size = (int(img_w * scale), int(img_h * scale))
            page_image = page_image.resize(new_size, Image.Resampling.LANCZOS)
            img_w, img_h = new_size
            logger.debug(f"Resized image for CPU processing: {new_size}")

        inputs = self._processor(images=page_image, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        # CPU Optimization: Use torch.no_grad and disable gradient computation
        with torch.no_grad():
            outputs = self._model(**inputs)

        # Post-process
        target_sizes = torch.tensor([(img_h, img_w)])
        results = self._processor.post_process_object_detection(
            outputs, threshold=self.score_threshold, target_sizes=target_sizes
        )[0]

        regions = []

        for seq, (score, label, box) in enumerate(
            zip(results["scores"], results["labels"], results["boxes"])
        ):
            score = float(score)
            label_str = self._model.config.id2label.get(int(label), "text").lower()
            cls = DOCLAYNET_LABEL_MAP.get(label_str, RegionClass.UNKNOWN)

            if cls == RegionClass.UNKNOWN:
                continue

            # Convert box from pixels to PDF coordinates
            px = [float(b) for b in box]
            
            # Scale back up if image was resized
            if max(img_w, img_h) > max_size:
                scale = max(page_image.size) / max_size
                px = [p * scale for p in px]
                img_w, img_h = page_image.size
            
            bbox = _pdf_bbox_from_pixel(
                tuple(px), img_w, img_h, page.width, page.height
            )

            regions.append(LayoutRegion(
                region_id=_region_id(doc_id, page.page_index, seq),
                region_class=cls,
                bbox=bbox,
                page_index=page.page_index,
                confidence=score,
                backend=DetectionBackend.DIT,
            ))

        logger.debug(
            "DiT: page %d → %d regions (CPU optimized)", page.page_index, len(regions)
        )
        return regions


    def detect_batch(
        self,
        page_images: list,  # List of PIL.Image
        pages: list,        # List of PageExtractionResult
        doc_id: str,
    ) -> list[list[LayoutRegion]]:
        """
        CPU-Optimized: Process multiple pages in a single batch.
        
        Benefits:
        - Model stays loaded (amortize load cost)
        - Parallel image preprocessing
        - Sequential inference (CPU-friendly, avoids memory spikes)
        """
        self._load()
        
        all_regions = []
        
        # Process in mini-batches of 4 pages (CPU memory friendly)
        batch_size = 4
        
        for i in range(0, len(page_images), batch_size):
            batch_images = page_images[i:i + batch_size]
            batch_pages = pages[i:i + batch_size]
            
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(page_images)-1)//batch_size + 1}")
            
            # Process each page in mini-batch (sequential for CPU to avoid memory spike)
            for img, page in zip(batch_images, batch_pages):
                regions = self.detect(img, page, doc_id)
                all_regions.append(regions)
        
        return all_regions

# ─────────────────────────────────────────────────────────────
# Hybrid Fusion: merge ML regions with heuristic refinement
# ─────────────────────────────────────────────────────────────

def fuse_ml_and_heuristic(
    ml_regions:  list[LayoutRegion],
    heur_regions: list[LayoutRegion],
    iou_threshold: float = 0.3,
) -> list[LayoutRegion]:
    """
    Merge ML-detected regions with heuristic regions.

    Strategy
    --------
    1. Start with ML regions (higher precision).
    2. For each ML region, if a heuristic region has high IoU AND
       different class, defer to heuristic IF the heuristic is more
       specific (e.g., heuristic says CAPTION where ML says PARAGRAPH).
    3. Keep heuristic-only regions that don't overlap any ML region
       (ML model may miss small captions, references, equations).

    This implements a "trust but verify" fusion aligned with the paper's
    Agent Independence principle (Section IV-D): each agent makes
    decisions independently; the orchestration layer reconciles them.
    """

    def _iou(a: LayoutRegion, b: LayoutRegion) -> float:
        ax0, ay0, ax1, ay1 = a.bbox.x0, a.bbox.y0, a.bbox.x1, a.bbox.y1
        bx0, by0, bx1, by1 = b.bbox.x0, b.bbox.y0, b.bbox.x1, b.bbox.y1
        ix0 = max(ax0, bx0); iy0 = max(ay0, by0)
        ix1 = min(ax1, bx1); iy1 = min(ay1, by1)
        inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
        union = a.bbox.area + b.bbox.area - inter
        return inter / union if union > 0 else 0.0

    # Specificity ranking: more specific class wins over generic
    SPECIFICITY: dict[RegionClass, int] = {
        RegionClass.CAPTION:    10,
        RegionClass.EQUATION:   10,
        RegionClass.REFERENCE:  10,
        RegionClass.ALGORITHM:  10,
        RegionClass.ABSTRACT:   9,
        RegionClass.HEADER:     8,
        RegionClass.FOOTER:     8,
        RegionClass.TABLE:      7,
        RegionClass.FIGURE:     7,
        RegionClass.TITLE:      6,
        RegionClass.LIST:       5,
        RegionClass.PARAGRAPH:  3,
        RegionClass.UNKNOWN:    0,
    }

    fused         = []
    heur_matched  = set()

    for ml_r in ml_regions:
        best_match   = None
        best_iou     = 0.0

        for idx, h_r in enumerate(heur_regions):
            iou = _iou(ml_r, h_r)
            if iou >= iou_threshold and iou > best_iou:
                best_iou   = iou
                best_match = (idx, h_r)

        if best_match is not None:
            h_idx, h_r = best_match
            heur_matched.add(h_idx)

            # Defer to heuristic if it's more specific
            if SPECIFICITY.get(h_r.region_class, 0) > SPECIFICITY.get(ml_r.region_class, 0):
                # keep heuristic class, but adopt ML confidence & source_block_ids
                from dataclasses import replace
                fused.append(LayoutRegion(
                    region_id        = ml_r.region_id,
                    region_class     = h_r.region_class,
                    bbox             = ml_r.bbox,
                    page_index       = ml_r.page_index,
                    confidence       = (ml_r.confidence + h_r.confidence) / 2,
                    backend          = DetectionBackend.HYBRID,
                    text_content     = h_r.text_content,
                    source_block_ids = h_r.source_block_ids,
                ))
            else:
                # Keep ML region, enrich with text from heuristic
                fused.append(LayoutRegion(
                    region_id        = ml_r.region_id,
                    region_class     = ml_r.region_class,
                    bbox             = ml_r.bbox,
                    page_index       = ml_r.page_index,
                    confidence       = ml_r.confidence,
                    backend          = DetectionBackend.HYBRID,
                    text_content     = h_r.text_content,
                    source_block_ids = h_r.source_block_ids,
                ))
        else:
            fused.append(ml_r)

    # Add heuristic-only regions (not matched by any ML region)
    for idx, h_r in enumerate(heur_regions):
        if idx not in heur_matched:
            fused.append(h_r)

    # Sort top-to-bottom
    fused.sort(key=lambda r: (r.bbox.y0, r.bbox.x0))
    return fused