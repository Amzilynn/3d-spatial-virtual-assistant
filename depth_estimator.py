import torch
import cv2
import numpy as np
import open3d as o3d
from typing import Tuple, Optional
from transformers import pipeline
import warnings
warnings.filterwarnings('ignore')


class DepthEstimator:
    """
    Depth Anything-based depth estimation for creating 3D point clouds
    """
    
    def __init__(self, model_type: str = "small", device: Optional[str] = None):
        """
        Initialize depth estimator
        
        Args:
            model_type: 'small', 'base', or 'large'
            device: 'cuda' or 'cpu', auto-detected if None
        """
        self.model_type = model_type
        
        if device is None:
            self.device = 0 if torch.cuda.is_available() else -1  # pipeline uses 0 for GPU, -1 for CPU
        else:
            self.device = 0 if device == "cuda" else -1
        
        device_name = "GPU" if self.device == 0 else "CPU"
        print(f"[DepthEstimator] Loading Depth-Anything-{model_type} on {device_name}...")
        
        try:
            # Use LiheYoung/depth-anything models from Hugging Face
            model_id = f"LiheYoung/depth-anything-{model_type}-hf"
            
            self.pipe = pipeline(
                task="depth-estimation",
                model=model_id,
                device=self.device
            )
            
            print("[DepthEstimator] Model loaded successfully")
        except Exception as e:
            print(f"[DepthEstimator] Error loading model: {e}")
            raise
    
    def estimate_depth(self, frame: np.ndarray) -> np.ndarray:
        """
        Estimate depth map from RGB frame
        
        Args:
            frame: Input frame (BGR format from OpenCV)
            
        Returns:
            Depth map (normalized, float32)
        """
        # Convert BGR to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Predict depth
        result = self.pipe(rgb)
        
        # Extract depth map (PIL Image)
        depth_pil = result["depth"]
        
        # Convert to numpy
        depth = np.array(depth_pil).astype(np.float32)
        
        # Normalize to [0, 1]
        depth_min = depth.min()
        depth_max = depth.max()
        depth_normalized = (depth - depth_min) / (depth_max - depth_min + 1e-8)
        
        # Resize to match original frame size
        if depth.shape[:2] != frame.shape[:2]:
            depth_normalized = cv2.resize(
                depth_normalized, 
                (frame.shape[1], frame.shape[0]),
                interpolation=cv2.INTER_LINEAR
            )
        
        return depth_normalized
    
    def depth_to_pointcloud(
        self, 
        depth: np.ndarray, 
        rgb: np.ndarray,
        focal_length: Optional[float] = None,
        max_depth: float = 10.0,
        downsample: int = 4
    ) -> o3d.geometry.PointCloud:
        """
        Convert depth map and RGB to Open3D point cloud
        
        Args:
            depth: Normalized depth map [0, 1]
            rgb: RGB image (same size as depth)
            focal_length: Camera focal length in pixels (auto-estimated if None)
            max_depth: Maximum depth in meters
            downsample: Downsample factor to reduce point count
            
        Returns:
            Open3D PointCloud
        """
        h, w = depth.shape
        
        # Estimate focal length if not provided (assuming standard camera)
        if focal_length is None:
            focal_length = w * 0.8  # Rough estimate
        
        # Scale depth to meters
        depth_meters = depth * max_depth
        
        # Create meshgrid
        u = np.arange(0, w, downsample)
        v = np.arange(0, h, downsample)
        u, v = np.meshgrid(u, v)
        
        # Convert to camera coordinates
        z = depth_meters[v, u]
        x = (u - w / 2) * z / focal_length
        y = (v - h / 2) * z / focal_length
        
        # Stack to get points
        points = np.stack([x, y, z], axis=-1).reshape(-1, 3)
        
        # Get colors (normalize to [0, 1])
        if rgb.shape[:2] != depth.shape:
            rgb_resized = cv2.resize(rgb, (w, h))
        else:
            rgb_resized = rgb
        
        colors = rgb_resized[v, u].reshape(-1, 3) / 255.0
        
        # Filter invalid points
        valid = (z > 0.1) & (z < max_depth)  # Remove too close/far
        valid = valid.flatten()
        
        points = points[valid]
        colors = colors[valid]
        
        # Create point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # Statistical outlier removal
        if len(pcd.points) > 100:
            pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        
        print(f"[DepthEstimator] Generated point cloud with {len(pcd.points)} points")
        
        return pcd
    
    def process_frame(
        self, 
        frame: np.ndarray,
        focal_length: Optional[float] = None,
        max_depth: float = 10.0
    ) -> Tuple[np.ndarray, o3d.geometry.PointCloud]:
        """
        End-to-end: estimate depth and create point cloud
        
        Args:
            frame: Input frame (BGR)
            focal_length: Camera focal length
            max_depth: Maximum depth in meters
            
        Returns:
            Tuple of (depth_map, point_cloud)
        """
        print("[DepthEstimator] Estimating depth...")
        depth = self.estimate_depth(frame)
        
        print("[DepthEstimator] Converting to point cloud...")
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pcd = self.depth_to_pointcloud(depth, rgb, focal_length, max_depth)
        
        return depth, pcd


if __name__ == "__main__":
    # Test
    estimator = DepthEstimator(model_type="small")
    
    # Create dummy frame
    test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    depth, pcd = estimator.process_frame(test_frame)
    
    print(f"Depth shape: {depth.shape}, Point cloud points: {len(pcd.points)}")
