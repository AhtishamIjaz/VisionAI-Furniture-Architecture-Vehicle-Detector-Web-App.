import os
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

CATEGORY_MAP = {
    # Furniture
    "chair": "Furniture",
    "table": "Furniture",
    "dining table": "Furniture",
    "bed": "Furniture",
    "couch": "Furniture",
    "sofa": "Furniture",
    "desk": "Furniture",
    "wardrobe": "Furniture",
    "cabinet": "Furniture",
    "bookshelf": "Furniture",
    "bench": "Furniture",
    "stool": "Furniture",

    # Architecture & Interior
    "window": "Architecture / Interior",
    "curtain": "Architecture / Interior",
    "curtains": "Architecture / Interior",
    "house": "Architecture / Interior",
    "building": "Architecture / Interior",
    "glass": "Architecture / Interior",
    "door": "Architecture / Interior",
    "stairs": "Architecture / Interior",
    "mirror": "Architecture / Interior",
    "lamp": "Architecture / Interior",
    "wall": "Architecture / Interior",
    "roof": "Architecture / Interior",

    # Vehicles
    "car": "Vehicles",
    "truck": "Vehicles",
    "bus": "Vehicles",
    "motorcycle": "Vehicles",
    "bicycle": "Vehicles",
    "van": "Vehicles",
    "vehicle": "Vehicles",
    "airplane": "Vehicles",
    "boat": "Vehicles",
    "train": "Vehicles",

    # Household & Decor
    "potted plant": "Household & Decor",
    "tv": "Electronics",
    "laptop": "Electronics",
    "refrigerator": "Appliances",
    "sink": "Fixtures",
    "clock": "Household & Decor",
    "vase": "Household & Decor",
    "person": "People",
    "dog": "Animals",
    "cat": "Animals"
}

CATEGORY_COLORS = {
    "Furniture": (230, 126, 34),          # Amber / Orange (BGR: 34, 126, 230)
    "Architecture / Interior": (46, 204, 113), # Emerald Green (BGR: 113, 204, 46)
    "Vehicles": (52, 152, 219),            # Bright Blue (BGR: 219, 152, 52)
    "Household & Decor": (155, 89, 182),   # Purple (BGR: 182, 89, 155)
    "Electronics": (241, 196, 15),         # Yellow (BGR: 15, 196, 241)
    "Appliances": (231, 76, 60),           # Coral Red (BGR: 60, 76, 231)
    "Fixtures": (26, 188, 156),            # Turquoise (BGR: 156, 188, 26)
    "Other": (149, 165, 166)               # Slate Gray (BGR: 166, 165, 149)
}

DEFAULT_WORLD_CLASSES = [
    "chair", "dining table", "table", "bed", "couch", "sofa", "wardrobe", "desk", "bookshelf",
    "window", "curtain", "house", "building", "glass", "door", "lamp", "mirror",
    "car", "truck", "bus", "motorcycle", "bicycle", "van", "vehicle",
    "potted plant", "tv", "refrigerator", "sink"
]

class VisionDetector:
    def __init__(self, models_dir: str = "models", default_model: str = "coco"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.active_model_name = default_model
        self.world_classes = list(DEFAULT_WORLD_CLASSES)
        
        self.world_path = self._find_model_file(["yolov8s-worldv2.pt", "yolov8s-world.pt"])
        self.coco_path = self._find_model_file(["yolov8m.pt", "yolov8n.pt"])
        
        self.coco_model = None
        self.world_model = None
        
        # Load default model
        self._init_models()

    def _find_model_file(self, candidates):
        for name in candidates:
            p = self.models_dir / name
            if p.exists():
                return str(p)
            if Path(name).exists():
                return str(Path(name))
        return candidates[0]

    def _init_models(self):
        # Load COCO model immediately for guaranteed fast startup
        coco_file = self.coco_path if Path(self.coco_path).exists() else "yolov8m.pt"
        print(f"[*] Loading standard YOLOv8 model: {coco_file}")
        self.coco_model = YOLO(coco_file)
        print("[OK] YOLOv8 model loaded successfully!")

    def get_active_model(self):
        if self.active_model_name == "world":
            if self.world_model is None:
                world_file = self.world_path if Path(self.world_path).exists() else "yolov8s-worldv2.pt"
                print(f"[*] Loading YOLO-World model: {world_file}...")
                self.world_model = YOLO(world_file)
                try:
                    self.world_model.set_classes(self.world_classes)
                    print("[OK] YOLO-World classes configured!")
                except Exception as e:
                    print(f"[!] Warning setting custom classes: {e}")
            return self.world_model
        return self.coco_model

    def switch_model(self, model_type: str):
        if model_type in ["world", "coco"]:
            self.active_model_name = model_type
            return True
        return False

    def predict(self, image_input, conf_thresh: float = 0.25):
        if isinstance(image_input, (str, Path)):
            img = cv2.imread(str(image_input))
            if img is None:
                raise ValueError(f"Could not read image from {image_input}")
        else:
            img = image_input.copy()

        h, w = img.shape[:2]
        model = self.get_active_model()
        results = model.predict(img, conf=conf_thresh, verbose=False)
        result = results[0]

        detections = []
        category_counts = {
            "Furniture": 0,
            "Architecture / Interior": 0,
            "Vehicles": 0,
            "Other": 0
        }
        annotated = img.copy()

        boxes = result.boxes
        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = [int(v) for v in xyxy]
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                
                label = result.names[cls_id] if cls_id in result.names else f"class_{cls_id}"
                label_lower = label.lower()

                # Category determination
                category = CATEGORY_MAP.get(label_lower, "Other")
                if category not in category_counts:
                    category_counts[category] = 0
                category_counts[category] += 1

                # Color selection (BGR)
                rgb_col = CATEGORY_COLORS.get(category, (149, 165, 166))
                bgr_col = (rgb_col[2], rgb_col[1], rgb_col[0])

                # Draw bounding box
                thickness = max(2, int(min(w, h) / 350))
                cv2.rectangle(annotated, (x1, y1), (x2, y2), bgr_col, thickness)

                # Draw label badge
                conf_pct = int(conf * 100)
                badge_text = f"{label.title()} {conf_pct}%"
                font = cv2.FONT_HERSHEY_DUPLEX
                font_scale = max(0.5, min(w, h) / 900.0)
                font_thick = 1
                (text_w, text_h), baseline = cv2.getTextSize(badge_text, font, font_scale, font_thick)

                badge_y1 = max(0, y1 - text_h - 10)
                badge_y2 = y1
                badge_x1 = x1
                badge_x2 = min(w, x1 + text_w + 14)

                cv2.rectangle(annotated, (badge_x1, badge_y1), (badge_x2, badge_y2), bgr_col, -1)
                cv2.putText(annotated, badge_text, (badge_x1 + 6, badge_y2 - 5), font, font_scale, (255, 255, 255), font_thick, cv2.LINE_AA)

                detections.append({
                    "label": label.title(),
                    "confidence": round(conf, 4),
                    "confidence_pct": conf_pct,
                    "category": category,
                    "bbox": [x1, y1, x2, y2],
                    "color_hex": f"#{rgb_col[0]:02x}{rgb_col[1]:02x}{rgb_col[2]:02x}"
                })

        # Sort detections by confidence descending
        detections.sort(key=lambda x: x["confidence"], reverse=True)

        return {
            "total_objects": len(detections),
            "category_counts": category_counts,
            "detections": detections,
            "annotated_image": annotated,
            "image_dims": {"width": w, "height": h},
            "model_used": "YOLO-World (Open-Vocabulary)" if self.active_model_name == "world" else "YOLOv8 (COCO Standard)"
        }
