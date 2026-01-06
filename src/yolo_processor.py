#!/usr/bin/env python3
"""
YOLO Image Processor
Reads images and provides labeled detection output.
Can process single images or batch process directories.
"""

import cv2
import numpy as np
import argparse
import os
import json
import sys
from pathlib import Path
from typing import List, Dict, Optional, Any
from camera import Camera

sys.path.append(str(Path(__file__).parent))

from detector import YOLODetector, create_detector_from_config

try:
    from ultralytics import YOLO as UltralyticsYOLO
except Exception:
    UltralyticsYOLO = None


def check_gpu_availability() -> Dict[str, Any]:
    """Check GPU availability and return detailed information."""
    gpu_info = {
        'available': False,
        'device': 'cpu',
        'cuda_available': False,
        'cuda_version': 'N/A',
        'device_name': 'N/A',
        'device_count': 0,
        'current_device': -1,
        'torch_version': 'N/A'
    }
    
    try:
        import torch
        gpu_info['torch_version'] = torch.__version__
        gpu_info['cuda_available'] = torch.cuda.is_available()
        
        if gpu_info['cuda_available']:
            gpu_info['available'] = True
            gpu_info['device'] = 'cuda'
            gpu_info['cuda_version'] = torch.version.cuda or 'Unknown'
            gpu_info['device_count'] = torch.cuda.device_count()
            gpu_info['current_device'] = torch.cuda.current_device()
            gpu_info['device_name'] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    except Exception as e:
        print(f"[WARNING] Error checking GPU: {e}")
    
    return gpu_info


def print_gpu_info(gpu_info: Dict[str, Any] = None):
    """Print formatted GPU information to console."""
    if gpu_info is None:
        gpu_info = check_gpu_availability()
    
    print("\n" + "=" * 60)
    print("GPU AVAILABILITY CHECK")
    print("=" * 60)
    print(f"GPU Available:       {gpu_info['available']}")
    print(f"Device:              {gpu_info['device'].upper()}")
    print(f"CUDA Available:      {gpu_info['cuda_available']}")
    print(f"CUDA Version:        {gpu_info['cuda_version']}")
    print(f"PyTorch Version:     {gpu_info['torch_version']}")
    if gpu_info['cuda_available']:
        print(f"GPU Device Name:     {gpu_info['device_name']}")
        print(f"Device Count:        {gpu_info['device_count']}")
        print(f"Current Device:      {gpu_info['current_device']}")
    print("=" * 60 + "\n")


def get_optimal_device(preferred_device: Optional[str] = None) -> str:
    """Get the optimal device for inference, with fallback logic."""
    gpu_info = check_gpu_availability()
    
    if preferred_device:
        if 'cuda' in preferred_device.lower():
            if gpu_info['cuda_available']:
                return preferred_device
            else:
                print(f"[WARNING] CUDA requested but not available. Falling back to CPU.")
                return 'cpu'
        else:
            return preferred_device
    else:
        if gpu_info['cuda_available']:
            return 'cuda'
        else:
            return 'cpu'


class YOLOv8Detector:
    """Minimal wrapper around ultralytics YOLO for detection + drawing."""
    def __init__(self, weights: str, conf_thres: float = 0.5, device: Optional[str] = None):
        if UltralyticsYOLO is None:
            raise ImportError("ultralytics package not available")
        self.model = UltralyticsYOLO(str(weights))
        self.conf = float(conf_thres)
        if device:
            try:
                self.model.to(device)
            except Exception:
                pass

    def detect(self, image: np.ndarray) -> List[Dict]:
        results = self.model(image, conf=self.conf, verbose=False)
        if not results:
            return []
        res = results[0]
        detections = []
        names = getattr(self.model, "names", {}) or {}
        boxes = getattr(res, "boxes", None)
        if boxes is None:
            return []
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            cls_name = str(names.get(cls_id, f"class_{cls_id}"))
            detections.append({
                'class_id': cls_id,
                'class_name': cls_name,
                'confidence': conf,
                'bbox': (int(x1), int(y1), int(x2), int(y2))
            })
        return detections

    def draw_detections(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        annotated = image.copy()
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']
            cls_name = det['class_name']
            color = (0, 255, 0)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = f"{cls_name}: {conf:.2f}"
            cv2.putText(annotated, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return annotated

    def get_class_counts(self, detections: List[Dict]) -> Dict[str, int]:
        counts = {}
        for det in detections:
            cls_name = det['class_name']
            counts[cls_name] = counts.get(cls_name, 0) + 1
        return counts


def process_image(image_path: str, detector, output_dir: Optional[str] = None,
                 save_annotated: bool = False, verbose: bool = True) -> Dict:
    """Process a single image file and return results."""
    image_path = Path(image_path)
    if not image_path.exists():
        if verbose:
            print(f"[ERROR] Image not found: {image_path}")
        return {'file': str(image_path), 'exists': False, 'num_detections': 0}

    try:
        image = cv2.imread(str(image_path))
        if image is None:
            if verbose:
                print(f"[ERROR] Failed to read image: {image_path}")
            return {'file': str(image_path), 'exists': True, 'num_detections': 0, 'readable': False}

        detections = detector.detect(image)
        result = {
            'file': str(image_path),
            'exists': True,
            'readable': True,
            'num_detections': len(detections),
            'detections': detections
        }

        if save_annotated and detections:
            annotated = detector.draw_detections(image, detections)
            if output_dir:
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / f"annotated_{image_path.name}"
                cv2.imwrite(str(output_path), annotated)
                result['annotated_path'] = str(output_path)
                if verbose:
                    print(f"[OK] {image_path.name}: {len(detections)} detections -> {output_path}")
            else:
                if verbose:
                    print(f"[OK] {image_path.name}: {len(detections)} detections (no output dir)")
        else:
            if verbose:
                print(f"[OK] {image_path.name}: {len(detections)} detections")

        return result

    except Exception as e:
        if verbose:
            print(f"[ERROR] Processing {image_path}: {e}")
        return {'file': str(image_path), 'error': str(e), 'num_detections': 0}


def process_directory(dir_path: str, detector, output_dir: Optional[str] = None,
                     save_annotated: bool = False, extensions: List[str] = None,
                     verbose: bool = True) -> List[Dict]:
    """Process all images in a directory."""
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.bmp']

    dir_path = Path(dir_path)
    if not dir_path.is_dir():
        if verbose:
            print(f"[ERROR] Directory not found: {dir_path}")
        return []

    image_files = []
    for ext in extensions:
        image_files.extend(dir_path.glob(f"*{ext}"))
        image_files.extend(dir_path.glob(f"*{ext.upper()}"))

    if verbose:
        print(f"Found {len(image_files)} images to process")

    results = []
    for i, image_path in enumerate(image_files):
        if verbose:
            print(f"\n[{i+1}/{len(image_files)}] Processing {image_path}")

        result = process_image(str(image_path), detector, output_dir, save_annotated, verbose=False)
        results.append(result)

    return results


def save_results_json(results: List[Dict], output_path: str):
    """Save results to JSON file."""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="YOLO Image Processor")
    parser.add_argument('input', nargs='?', help="Input image file or directory (not needed with --camera)")
    parser.add_argument('--camera', action='store_true', help="Use camera for real-time processing")
    parser.add_argument('--device', default='/dev/video4', help="Camera device path (default: /dev/video4)")
    parser.add_argument('--gpu', help="GPU device for YOLO inference (e.g., 'cuda', 'cuda:0', 'cpu')")
    parser.add_argument('--model', '-m', help="Path to YOLO model file")
    parser.add_argument('--conf', '-c', type=float, default=0.5,
                       help="Confidence threshold (default: 0.5)")
    parser.add_argument('--output-dir', '-o', help="Output directory for results")
    parser.add_argument('--save-annotated', '-a', action='store_true',
                       help="Save annotated images with detections")
    parser.add_argument('--json-output', '-j', help="Save results to JSON file")
    parser.add_argument('--quiet', '-q', action='store_true',
                       help="Suppress verbose output")
    parser.add_argument('--show-gpu', action='store_true',
                       help="Show GPU information and exit")
    parser.add_argument('--extensions', nargs='+',
                       default=['.jpg', '.jpeg', '.png', '.bmp'],
                       help="Image extensions to process (default: jpg jpeg png bmp)")

    parser.add_argument('--yolov8', action='store_true', help="Use ultralytics YOLOv8 model")
    parser.add_argument('--weights', default=str(Path(__file__).parent / "weights" / "yolov8s_best.pt"),
                        help="Path to weights file for YOLOv8 (default: ./weights/yolov8s_best.pt)")

    args = parser.parse_args()

    if args.show_gpu:
        gpu_info = check_gpu_availability()
        print_gpu_info(gpu_info)
        return 0

    if not args.quiet:
        gpu_info = check_gpu_availability()
        print_gpu_info(gpu_info)

    detector = None
    trained_weights = Path(__file__).parent.parent.parent / "Yolo_object_detection" / "BFMC.v1i.yolov8" / "runs" / "detect" / "traffic_signs_full_ads" / "weights" / "best.pt"
    
    if trained_weights.exists() or args.yolov8:
        if args.yolov8 or not args.model:
            weights_path = Path(args.weights) if args.yolov8 else trained_weights
            if weights_path.exists():
                try:
                    device = get_optimal_device(args.gpu)
                    detector = YOLOv8Detector(weights_path, conf_thres=args.conf, device=device)
                    print(f"[INFO] Using YOLOv8Detector with trained weights (device: {device})")
                except Exception as e:
                    print(f"Failed to create YOLOv8 detector: {e}")
                    detector = None
    
    if detector is None:
        device = get_optimal_device(args.gpu)
        detector = create_detector_from_config(args.model, args.conf, device)
        print(f"[INFO] Using standard YOLODetector from detector.py (device: {device})")

    if args.camera:
        print("Starting real-time YOLO detection from camera...")
        print(f"Camera device: {args.device}")
        print("Press 'q' to quit")
        
        camera = Camera(device_path=args.device)
        display_available = os.environ.get('DISPLAY') is not None
        print(f"Display available: {display_available}")
        
        try:
            frame_count = 0
            while True:
                frame = camera.read_image()
                if frame is None:
                    print("Warning: Failed to read frame from camera")
                    continue
                
                detections = detector.detect(frame)
                annotated = detector.draw_detections(frame, detections)
                
                if detections and not args.quiet:
                    print(f"Frame {frame_count}: {len(detections)} detections found")
                    for det in detections:
                        print(f"  - {det['class_name']}: {det['confidence']:.2f}")
                elif not args.quiet and frame_count % 30 == 0:
                    print(f"Frame {frame_count}: Processing...")
                
                if display_available:
                    cv2.imshow("YOLO Camera Detection", annotated)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("Exiting...")
                        break
                else:
                    if detections:
                        output_path = f"/tmp/yolo_detection_{frame_count:04d}.jpg"
                        cv2.imwrite(output_path, annotated)
                        print(f"Saved detection frame: {output_path}")
                
                frame_count += 1
                    
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            if display_available:
                cv2.destroyAllWindows()
            camera.release()
        
        return 0

    if not args.input:
        print("Error: Input file/directory required when not using --camera")
        return 1

    input_path = Path(args.input)
    if input_path.is_file():
        result = process_image(str(input_path), detector, args.output_dir,
                              args.save_annotated, not args.quiet)

        if args.json_output:
            save_results_json([result], args.json_output)

    elif input_path.is_dir():
        results = process_directory(str(input_path), detector, args.output_dir,
                                   args.save_annotated, args.extensions, not args.quiet)

        if args.json_output:
            save_results_json(results, args.json_output)

        if not args.quiet:
            total_detections = sum(r.get('num_detections', 0) for r in results)
            print("\nSummary:")
            print(f"  Images processed: {len(results)}")
            print(f"  Total detections: {total_detections}")

    else:
        print(f"Error: Input path {input_path} does not exist")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
