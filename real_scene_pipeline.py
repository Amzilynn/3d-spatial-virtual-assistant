import numpy as np
import open3d as o3d
from typing import Tuple, List, Dict, Optional
from video_processor import VideoProcessor
from depth_estimator import DepthEstimator
from object_detector_real import RealObjectDetector


class RealScenePipeline:
    
    def __init__(self, use_gpu: bool = True):
        """
        Initialize all components
        
        Args:
            use_gpu: Whether to use GPU for depth estimation
        """
        print("\n" + "="*60)
        print("   REAL SCENE PIPELINE - Initializing...")
        print("="*60)
        
        self.video_processor = VideoProcessor()
        
        # Use smaller/faster models for MVP
        device = "cuda" if use_gpu else "cpu"
        self.depth_estimator = DepthEstimator(model_type="small", device=device)
        
        self.object_detector = RealObjectDetector(model_name="yolov8n.pt")
        
        print("[Pipeline] All components loaded!\n")
    
    def process_video(
        self, 
        video_path: str,
        focal_length: Optional[float] = None,
        max_depth: float = 10.0
    ) -> Tuple[o3d.geometry.PointCloud, List[Dict]]:
        """
        Full pipeline: video → point cloud + objects
        
        Args:
            video_path: Path to video file
            focal_length: Camera focal length (auto-estimated if None)
            max_depth: Maximum depth in meters
            
        Returns:
            Tuple of (point_cloud, detected_objects)
        """
        print(f"\n[Pipeline] Processing video: {video_path}")
        print("-" * 60)
        
        # Step 1: Extract best frame
        print("\n[1/3] Extracting video frames...")
        original_frame, preprocessed_frame = self.video_processor.process_video(video_path)
        
        # Step 2: Estimate depth and create point cloud
        print("\n[2/3] Estimating depth and creating 3D point cloud...")
        depth_map, point_cloud = self.depth_estimator.process_frame(
            original_frame, 
            focal_length=focal_length,
            max_depth=max_depth
        )
        
        # Step 3: Detect objects and map to 3D
        print("\n[3/3] Detecting objects and mapping to 3D coordinates...")
        detections = self.object_detector.detect_and_project(
            original_frame,
            depth_map,
            focal_length=focal_length,
            max_depth=max_depth
        )
        
        # Prepare objects for spatial assistant (match format)
        objects = []
        for det in detections:
            # Estimate rough dimensions (not accurate, but good enough for MVP)
            bbox_2d = det['bbox']
            width_px = bbox_2d[2] - bbox_2d[0]
            height_px = bbox_2d[3] - bbox_2d[1]
            
            # Scale to meters (very rough estimation)
            z = det['center'][2]
            focal = focal_length if focal_length else original_frame.shape[1] * 0.8
            width_m = width_px * z / focal
            height_m = height_px * z / focal
            depth_m = min(width_m, height_m)  # Rough depth estimate
            
            obj = {
                'label': det['label'],
                'center': det['center'],
                'dimensions': [width_m, depth_m, height_m],
                'confidence': det['confidence'],
                'bounds': None  # Will be created in visualizer
            }
            objects.append(obj)
        
        print("\n" + "="*60)
        print(f"[Pipeline] COMPLETE!")
        print(f"  - Point Cloud: {len(point_cloud.points)} points")
        print(f"  - Detected Objects: {len(objects)}")
        print("="*60 + "\n")
        
        return point_cloud, objects


if __name__ == "__main__":
    # Test
    pipeline = RealScenePipeline(use_gpu=False)
    
    try:
        pcd, objects = pipeline.process_video("sample_room.mp4")
        print(f"\nResults: {len(pcd.points)} points, {len(objects)} objects")
        
        # Visualize
        o3d.visualization.draw_geometries([pcd])
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
