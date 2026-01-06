#!/usr/bin/env python3
"""
Test script to diagnose camera and YOLO detection issues
"""
import cv2
import sys
from pathlib import Path
from camera import Camera
from detector import create_detector_from_config
from yolo_processor import YOLOv8Detector

def test_camera():
    """Test camera is working"""
    print("[TEST] Starting camera test...")
    try:
        camera = Camera(device_path='/dev/video4')
        frame = camera.read_image()
        if frame is not None:
            print(f"[✓] Camera OK - Frame shape: {frame.shape}")
            return True
        else:
            print("[✗] Camera returned None")
            return False
    except Exception as e:
        print(f"[✗] Camera error: {e}")
        return False

def test_detector(use_trained=True):
    """Test detector is working"""
    print("\n[TEST] Starting detector test...")
    try:
        if use_trained:
            trained_weights = Path(__file__).parent.parent.parent / "Yolo_object_detection" / "BFMC.v1i.yolov8" / "runs" / "detect" / "traffic_signs_full_ads" / "weights" / "best.pt"
            if trained_weights.exists():
                print(f"[INFO] Using trained weights: {trained_weights}")
                detector = YOLOv8Detector(str(trained_weights), conf_thres=0.5, device='cpu')
            else:
                print(f"[WARNING] Trained weights not found: {trained_weights}")
                detector = create_detector_from_config()
        else:
            detector = create_detector_from_config()
        
        print("[✓] Detector loaded successfully")
        return detector
    except Exception as e:
        print(f"[✗] Detector error: {e}")
        return None

def test_detection_on_frame(detector, frame):
    """Test detection on a single frame"""
    print("\n[TEST] Running detection on frame...")
    try:
        detections = detector.detect(frame)
        print(f"[✓] Detection OK - Found {len(detections)} objects")
        
        if detections:
            print("  Detections:")
            for det in detections:
                print(f"    - {det['class_name']}: {det['confidence']:.2f}")
        
        # Try to draw annotations
        annotated = detector.draw_detections(frame, detections)
        print(f"[✓] Annotations drawn - Output shape: {annotated.shape}")
        
        return annotated
    except Exception as e:
        print(f"[✗] Detection error: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    print("=" * 50)
    print("YOLO Camera & Detection Diagnostics")
    print("=" * 50)
    
    # Test 1: Camera
    camera_ok = test_camera()
    if not camera_ok:
        print("\n[ERROR] Camera test failed. Cannot proceed.")
        return 1
    
    # Test 2: Detector
    camera = Camera(device_path='/dev/video4')
    frame = camera.read_image()
    
    detector = test_detector(use_trained=True)
    if detector is None:
        print("\n[ERROR] Detector test failed.")
        return 1
    
    # Test 3: Detection
    if frame is not None:
        annotated = test_detection_on_frame(detector, frame)
        if annotated is not None:
            # Save test frame
            test_output = "/tmp/yolo_test_frame.jpg"
            cv2.imwrite(test_output, annotated)
            print(f"\n[✓] Test frame saved to: {test_output}")
    
    print("\n" + "=" * 50)
    print("[✓] All tests completed successfully!")
    print("=" * 50)
    return 0

if __name__ == "__main__":
    sys.exit(main())
