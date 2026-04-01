# extractors/yolo_layout_detector.py
import onnxruntime as ort
from PIL import Image
import numpy as np

class YOLOLayoutDetector:
    """CPU-optimized YOLOv8 for layout detection - Windows friendly"""
    
    def __init__(self, model_path: str = "yolov8n-doclaynet.onnx"):
        # ONNX Runtime with CPU provider - no PyTorch/CUDA needed
        self.session = ort.InferenceSession(
            model_path,
            providers=['CPUExecutionProvider']  # Windows CPU
        )
        self.input_name = self.session.get_inputs()[0].name
        
    def detect(self, page_image: Image.Image, page, doc_id: str) -> list[LayoutRegion]:
        # Preprocess
        img = page_image.resize((640, 640))
        img_array = np.array(img).astype(np.float32) / 255.0
        img_array = np.transpose(img_array, (2, 0, 1))  # HWC to CHW
        img_array = np.expand_dims(img_array, axis=0)
        
        # Inference
        outputs = self.session.run(None, {self.input_name: img_array})
        
        # Post-process to LayoutRegion objects...