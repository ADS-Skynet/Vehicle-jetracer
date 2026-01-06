import cv2
import numpy as np
from ultralytics import YOLO
import time
from typing import List, Dict, Tuple, Optional


class YOLODetector:
    """
    YOLO object detector using Ultralytics YOLOv8/YOLOv11.
    Loads model and performs inference on images.
    """

    def __init__(self, model_path: str = "yolo11n.pt", conf_threshold: float = 0.5, device: str = None):
        """
        Initialize YOLO detector.

        Args:
            model_path: Path to YOLO model file (.pt)
            conf_threshold: Confidence threshold for detections
            device: Device to run inference on ('cpu', 'cuda', 'cuda:0', etc.)
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.device = device or 'cpu'  # Default to cpu
        self.model = None
        self.class_names = []
        self.load_model()

    def load_model(self):
        """Load YOLO model and extract class names."""
        try:
            print(f"[YOLO] Loading model from {self.model_path}")
            self.model = YOLO(self.model_path)
            # Get class names from model
            if hasattr(self.model, 'names'):
                self.class_names = list(self.model.names.values())
            else:
                # Fallback class names for common models
                self.class_names = [
                    'car', 'crosswalk', 'highway_entry', 'highway_exit', 'no_entry',
                    'onewayroad', 'parking', 'pedestrian', 'priority', 'roadblock',
                    'roundabout', 'stop', 'trafficlight'
                ]
            print(f"[YOLO] ✓ Model loaded with {len(self.class_names)} classes: {self.class_names}")
            print(f"[YOLO] Inference device: {self.device}")
        except Exception as e:
            print(f"[YOLO] ✗ Failed to load model: {e}")
            self.model = None

    def detect(self, image: np.ndarray) -> List[Dict]:
        """
        Perform object detection on an image.

        Args:
            image: Input image as numpy array (BGR format)

        Returns:
            List of detection dictionaries with keys:
            - class_id: int
            - class_name: str
            - confidence: float
            - bbox: tuple (x1, y1, x2, y2)
        """
        if self.model is None:
            return []

        try:
            # Run inference
            results = self.model.predict(image, conf=self.conf_threshold, device=self.device, verbose=False)

            detections = []
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        # Extract bounding box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0].cpu().numpy())
                        class_id = int(box.cls[0].cpu().numpy())

                        detection = {
                            'class_id': class_id,
                            'class_name': self.class_names[class_id] if class_id < len(self.class_names) else f'class_{class_id}',
                            'confidence': conf,
                            'bbox': (int(x1), int(y1), int(x2), int(y2))
                        }
                        detections.append(detection)

            return detections

        except Exception as e:
            print(f"[YOLO] Detection error: {e}")
            return []

    def draw_detections(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        Draw detection results on image.

        Args:
            image: Input image
            detections: List of detection dictionaries

        Returns:
            Image with detections drawn
        """
        img_copy = image.copy()

        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            class_name = det['class_name']
            conf = det['confidence']

            # Draw bounding box
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw label
            label = f"{class_name}: {conf:.2f}"
            cv2.putText(img_copy, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


        # cv2.imshow("YOLO Real-Time Detections", img_copy)
        # cv2.waitKey(1)
        return img_copy

    def get_class_counts(self, detections: List[Dict]) -> Dict[str, int]:
        """
        Count detections by class.

        Args:
            detections: List of detection dictionaries

        Returns:
            Dictionary with class names as keys and counts as values
        """
        counts = {}
        for det in detections:
            class_name = det['class_name']
            counts[class_name] = counts.get(class_name, 0) + 1
        return counts


def create_detector_from_config(model_path: str = None, conf_threshold: float = 0.5, device: str = None) -> YOLODetector:
    """
    Factory function to create YOLO detector with default model path.
    """
    if model_path is None:
        # Try to find model in Yolo_object_detection folder
        import os
        base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        yolo_dir = os.path.join(base_path, "Yolo_object_detection", "BFMC.v1i.yolov8")

        # Check for trained best.pt first (from yolov8s training)
        best_model_path = os.path.join(yolo_dir, "runs", "detect", "traffic_signs_full_ads", "weights", "best.pt")
        if os.path.exists(best_model_path):
            model_path = best_model_path
        else:
            # Fallback to pre-trained models
            for model_name in ["yolo11n.pt", "yolov8n.pt"]:
                candidate = os.path.join(yolo_dir, model_name)
                if os.path.exists(candidate):
                    model_path = candidate
                    break

            if model_path is None:
                print("[YOLO] Warning: No YOLO model found, using default")
                model_path = "yolo11n.pt"

    return YOLODetector(model_path, conf_threshold, device)
