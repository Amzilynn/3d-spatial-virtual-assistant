"""
Real Object Detection Module
Uses YOLO to detect objects in RGB frames and map them to 3D coordinates
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from ultralytics import YOLO
import warnings
warnings.filterwarnings('ignore')


class RealObjectDetector:
    """
    YOLO-based object detection with 3D coordinate mapping
    """
    
    # COCO dataset classes we care about (furniture)
    RELEVANT_CLASSES = {
        56: 'Chair',
        57: 'Sofa',  # 'couch' in COCO
        60: 'Table',  # 'dining table' in COCO
        59: 'Bed',
        62: 'TV',
        # Add more as needed
    }
    
    def __init__(self, model_name: str = "yolov8n.pt", confidence_threshold: float = 0.3):
        """
        Initialize YOLO detector
        
        Args:
            model_name: YOLO model variant (n=nano, s=small, m=medium, l=large, x=extra-large)
            confidence_threshold: Minimum confidence for detections
        """
        self.confidence_threshold = confidence_threshold
        
        print(f"[RealObjectDetector] Loading YOLO model {model_name}...")
        self.model = YOLO(model_name)
        print("[RealObjectDetector] Model loaded successfully")
    
    def detect_objects(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect objects in frame
        
        Args:
            frame: Input frame (BGR)
            
        Returns:
            List of detections with {label, confidence, bbox, center_2d}
        """
        # Run inference
        results = self.model(frame, verbose=False)
        
        detections = []
        
        for result in results:
            boxes = result.boxes
            
            for box in boxes:
                # Get class ID and confidence
                cls_id = int(box.cls[0])
                confidence = float(box.conf[0])
                
                # Filter by confidence and relevant classes
                if confidence < self.confidence_threshold:
                    continue
                
                if cls_id not in self.RELEVANT_CLASSES:
                    continue
                
                # Get bounding box (x1, y1, x2, y2)
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                # Calculate center
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                
                label = self.RELEVANT_CLASSES[cls_id]
                
                detection = {
                    'label': label,
                    'confidence': confidence,
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'center_2d': [center_x, center_y],
                    'class_id': cls_id
                }
                
                detections.append(detection)
        
        print(f"[RealObjectDetector] Detected {len(detections)} objects: {[d['label'] for d in detections]}")
        
        return detections
    
    def project_to_3d(
        self, 
        detection: Dict, 
        depth_map: np.ndarray,
        focal_length: float,
        max_depth: float = 10.0
    ) -> np.ndarray:
        """
        Project 2D detection to 3D coordinates using depth map
        
        Args:
            detection: Detection dict with bbox and center_2d
            depth_map: Normalized depth map [0, 1]
            focal_length: Camera focal length in pixels
            max_depth: Maximum depth in meters
            
        Returns:
            3D centroid [x, y, z] in camera coordinates
        """
        x1, y1, x2, y2 = detection['bbox']
        h, w = depth_map.shape
        
        # Ensure bbox is within image bounds
        x1 = max(0, min(x1, w-1))
        y1 = max(0, min(y1, h-1))
        x2 = max(0, min(x2, w-1))
        y2 = max(0, min(y2, h-1))
        
        # Get depth in the bounding box region
        depth_roi = depth_map[y1:y2, x1:x2]
        
        if depth_roi.size == 0:
            # Fallback to center point
            cx, cy = detection['center_2d']
            cx, cy = int(cx), int(cy)
            z = depth_map[cy, cx] * max_depth
        else:
            # Use median depth in ROI (more robust than mean)
            z = np.median(depth_roi) * max_depth
        
        # Get 2D center
        cx, cy = detection['center_2d']
        
        # Convert to 3D camera coordinates
        x = (cx - w / 2) * z / focal_length
        y = (cy - h / 2) * z / focal_length
        
        return np.array([x, y, z])
    
    def detect_and_project(
        self,
        frame: np.ndarray,
        depth_map: np.ndarray,
        focal_length: Optional[float] = None,
        max_depth: float = 10.0
    ) -> List[Dict]:
        """
        End-to-end: detect objects and map to 3D
        
        Args:
            frame: Input frame (BGR)
            depth_map: Depth map (same size as frame)
            focal_length: Focal length (auto-estimated if None)
            max_depth: Maximum depth in meters
            
        Returns:
            List of detections with 3D coordinates added
        """
        # Detect objects in 2D
        detections = self.detect_objects(frame)
        
        if not detections:
            return []
        
        # Estimate focal length if needed
        if focal_length is None:
            focal_length = frame.shape[1] * 0.8
        
        # Add 3D coordinates
        for detection in detections:
            center_3d = self.project_to_3d(detection, depth_map, focal_length, max_depth)
            detection['center'] = center_3d.tolist()
            
            print(f"  - {detection['label']} at 3D: {center_3d} (confidence: {detection['confidence']:.2f})")
        
        return detections
    
    def draw_detections(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        Draw bounding boxes and labels on frame (for debugging)
        
        Args:
            frame: Input frame
            detections: List of detections
            
        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            label = det['label']
            conf = det['confidence']
            
            # Draw box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            text = f"{label} {conf:.2f}"
            cv2.putText(annotated, text, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return annotated


if __name__ == "__main__":
    # Test
    detector = RealObjectDetector()
    
    # Create dummy frame
    test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    test_depth = np.random.rand(480, 640)
    
    detections = detector.detect_and_project(test_frame, test_depth)
    print(f"Detections: {detections}")
