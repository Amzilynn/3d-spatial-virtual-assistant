import cv2
import numpy as np
from typing import List, Tuple, Optional
from pathlib import Path


class VideoProcessor:
    """
    Extracts and preprocesses frames from video files for depth estimation
    """
    
    def __init__(self):
        self.target_size = (512, 512)  # Optimal size for MiDaS
    
    def extract_frames(self, video_path: str, max_frames: int = 10) -> List[np.ndarray]:
        """
        Extract evenly spaced frames from video
        
        Args:
            video_path: Path to video file
            max_frames: Maximum number of frames to extract
            
        Returns:
            List of frames as numpy arrays (BGR format)
        """
        if not Path(video_path).exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0
        
        print(f"[VideoProcessor] Video info: {total_frames} frames, {fps:.1f} FPS, {duration:.1f}s")
        
        # Calculate frame indices to extract (evenly spaced)
        if total_frames <= max_frames:
            frame_indices = list(range(total_frames))
        else:
            step = total_frames // max_frames
            frame_indices = [i * step for i in range(max_frames)]
        
        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
        
        cap.release()
        
        print(f"[VideoProcessor] Extracted {len(frames)} frames")
        return frames
    
    def select_best_frame(self, frames: List[np.ndarray]) -> Tuple[np.ndarray, int]:
        """
        Select the best frame based on contrast and sharpness
        
        Args:
            frames: List of video frames
            
        Returns:
            Tuple of (best_frame, frame_index)
        """
        if not frames:
            raise ValueError("No frames provided")
        
        best_score = -1
        best_idx = 0
        
        for idx, frame in enumerate(frames):
            # Convert to grayscale for analysis
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Calculate sharpness using Laplacian variance
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Calculate contrast using standard deviation
            contrast = gray.std()
            
            # Combined score (weighted)
            score = laplacian_var * 0.7 + contrast * 0.3
            
            if score > best_score:
                best_score = score
                best_idx = idx
        
        print(f"[VideoProcessor] Selected frame {best_idx} (score: {best_score:.1f})")
        return frames[best_idx], best_idx
    
    def preprocess_frame(self, frame: np.ndarray, target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Resize and normalize frame for model input
        
        Args:
            frame: Input frame (BGR)
            target_size: Target size (width, height), uses self.target_size if None
            
        Returns:
            Preprocessed frame
        """
        if target_size is None:
            target_size = self.target_size
        
        # Resize while maintaining aspect ratio
        h, w = frame.shape[:2]
        target_w, target_h = target_size
        
        # Calculate scaling factor
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
        # Pad to target size
        pad_w = target_w - new_w
        pad_h = target_h - new_h
        pad_left = pad_w // 2
        pad_top = pad_h // 2
        pad_right = pad_w - pad_left
        pad_bottom = pad_h - pad_top
        
        padded = cv2.copyMakeBorder(
            resized, 
            pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT, 
            value=[0, 0, 0]
        )
        
        return padded
    
    def process_video(self, video_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        End-to-end processing: extract, select, and preprocess
        
        Args:
            video_path: Path to video file
            
        Returns:
            Tuple of (original_frame, preprocessed_frame)
        """
        frames = self.extract_frames(video_path, max_frames=5)  # Fewer frames for speed
        best_frame, _ = self.select_best_frame(frames)
        preprocessed = self.preprocess_frame(best_frame)
        
        return best_frame, preprocessed


if __name__ == "__main__":
    # Test
    processor = VideoProcessor()
    try:
        original, processed = processor.process_video("sample_room.mp4")
        print(f"Original shape: {original.shape}, Processed shape: {processed.shape}")
    except Exception as e:
        print(f"Error: {e}")
