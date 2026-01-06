import open3d as o3d
import numpy as np
import sys
import os
import math
from typing import List, Dict, Optional, Tuple, NamedTuple

# ==============================================================================
# 1. SCENE GENERATION & LOADING
# ==============================================================================

class SceneObject(NamedTuple):
    name: str
    color: List[float]
    dimensions: List[float] # w, k, h
    position: List[float]   # x, y, z

class SceneGenerator:
    """
    Handles loading a point cloud or generating a synthetic 3D room.
    """
    
    def __init__(self):
        # Define standard objects for synthetic generation
        self.synthetic_objects = [
            SceneObject("Sofa",  [1.0, 0.0, 0.0], [2.0, 1.0, 0.8], [0.0, 2.0, 0.4]), # Red Sofa in front/center
            SceneObject("Table", [0.0, 0.0, 1.0], [1.2, 0.8, 0.6], [1.5, 0.0, 0.3]), # Blue Table to the right
            SceneObject("Chair", [0.0, 1.0, 0.0], [0.5, 0.5, 0.9], [-1.0, 1.0, 0.45]) # Green Chair to the left
        ]

    def create_synthetic_room(self) -> Tuple[o3d.geometry.PointCloud, List[Dict]]:
        """
        Creates a simple point cloud room with Walls, Floor, and 3 Objects.
        Returns:
            pcd: The combined point cloud
            gt_objects: Ground truth object metadata (for the detector)
        """
        print("[SceneGenerator] Generating synthetic room (Walls, Floor, Sofa, Table, Chair)...")
        
        geometries = []
        gt_objects = []

        # 1. Floor (Gray) - 5x5 meters
        floor = o3d.geometry.TriangleMesh.create_box(width=5.0, height=0.1, depth=5.0)
        floor.compute_vertex_normals()
        floor.paint_uniform_color([0.7, 0.7, 0.7])
        floor.translate([-2.5, -0.05, -2.5]) # Center at origin
        geometries.append(floor)

        # 2. Walls (White)
        # Back Wall
        wall_back = o3d.geometry.TriangleMesh.create_box(width=5.0, height=3.0, depth=0.1)
        wall_back.compute_vertex_normals()
        wall_back.paint_uniform_color([0.9, 0.9, 0.9])
        wall_back.translate([-2.5, 0.0, -2.55])
        geometries.append(wall_back)

        # 3. Objects
        for obj in self.synthetic_objects:
            # Create a box for the object
            mesh = o3d.geometry.TriangleMesh.create_box(
                width=obj.dimensions[0], 
                height=obj.dimensions[2], 
                depth=obj.dimensions[1]
            )
            mesh.compute_vertex_normals()
            mesh.paint_uniform_color(obj.color)
            
            # Centering logic: create_box puts corner at (0,0,0). 
            # We want center at obj.position.
            # Center of the box is (w/2, h/2, d/2)
            center_offset = np.array([obj.dimensions[0]/2, obj.dimensions[2]/2, obj.dimensions[1]/2])
            target_pos = np.array(obj.position)
            
            # Shift to target position (target - center_offset)
            mesh.translate(target_pos - center_offset)
            
            geometries.append(mesh)

            # Store Ground Truth for "Detection"
            gt_objects.append({
                "label": obj.name,
                "center": obj.position,
                "dimensions": obj.dimensions,
                "color": obj.color,
                "bounds": mesh.get_axis_aligned_bounding_box()
            })

        # Merge all into a Single Point Cloud (Sample points from meshes)
        combined_pcd = o3d.geometry.PointCloud()
        for mesh in geometries:
            # Sample points to make it a point cloud
            pcd = mesh.sample_points_poisson_disk(number_of_points=5000)
            pcd.paint_uniform_color(mesh.paint_uniform_color(mesh.vertex_colors[0]).vertex_colors[0])
            combined_pcd += pcd

        print(f"[SceneGenerator] Room generated with {len(combined_pcd.points)} points.")
        return combined_pcd, gt_objects

    def load_scene(self, filepath: str = "scene.ply"):
        """
        Attempts to load a scene, otherwise creates synthetic.
        """
        # Check explicit existence to avoid Open3D 'RPly' warning logs
        if not os.path.exists(filepath):
            print(f"[SceneGenerator] No local '{filepath}' found. Using Synthetic Scene Generator.")
            return self.create_synthetic_room()
            
        try:
            pcd = o3d.io.read_point_cloud(filepath)
            if pcd.is_empty():
                print(f"[SceneGenerator] '{filepath}' is empty.")
                raise ValueError("Empty point cloud")
            print(f"[SceneGenerator] Successfully loaded {filepath}")
            # NOTE: For a real loaded scene, we'd run real detection. 
            # For this MVP, if we load a file, we might not have labels.
            # We will fallback to synthetic if file load fails or is requested.
            return pcd, None 
        except Exception as e:
            print(f"[SceneGenerator] Error loading file: {e}. Fallback to Synthetic.")
            return self.create_synthetic_room()

# ==============================================================================
# 2. OBJECT DETECTION
# ==============================================================================

class ObjectDetector:
    """
    Detects objects in the scene. 
    For MVP, this relies on Ground Truth from the synthetic generator 
    to ensure perfect reliability during the demo.
    """
    def detect(self, pcd, known_objects: Optional[List[Dict]] = None) -> List[Dict]:
        """
        Returns a list of detected objects dicts: {label, center, bounds...}
        """
        if known_objects:
            print(f"[ObjectDetector] Using {len(known_objects)} known objects (Synthetic Ground Truth).")
            return known_objects
        
        # Fallback: Simple clustering if we loaded a raw file (Not used in default synthetic path)
        print("[ObjectDetector] Running DBSCAN clustering...")
        # (Simplified for MVP - assuming synthetic path is primary)
        return []

# ==============================================================================
# 3. SPATIAL REASONING ENGINE (The "Brain")
# ==============================================================================

class SpatialReasoningEngine:
    """
    "SpatialLM" replacement for Offline MVP.
    Converts 3D coordinates into natural language descriptions.
    """
    def __init__(self, objects: List[Dict]):
        self.objects = {obj['label'].lower(): obj for obj in objects}
        # Assume User is at origin (0,0,0) facing +Y (or +Z depending on coord sys). 
        # In our Gen: Floor is XZ plane, Height is Y.
        # Let's assume User is at (0, 1.0, -2.0) looking towards center (0,1,0)
        self.user_pos = np.array([0.0, 1.6, -2.5]) # User standing near the "door"
        self.user_heading = np.array([0.0, 0.0, 1.0]) # Facing +Z (into the room)

    def get_object_center(self, name: str) -> np.ndarray:
        name = name.lower()
        for k, v in self.objects.items():
            if k in name:
                return np.array(v['center'])
        return None

    def calculate_relations(self, target_pos: np.ndarray) -> str:
        """
        Calculates distance and relative direction from User.
        """
        # 1. Distance
        vec_to_target = target_pos - self.user_pos
        dist = np.linalg.norm(vec_to_target)
        dist_str = f"{dist:.1f} meters"

        # 2. Direction (Relative to User Facing)
        # Project to 2D (XZ plane) for "left/right/front"
        flat_vec = np.array([vec_to_target[0], 0, vec_to_target[2]])
        flat_vec = flat_vec / (np.linalg.norm(flat_vec) + 1e-6)
        
        flat_heading = np.array([self.user_heading[0], 0, self.user_heading[2]])
        
        # Dot product for Front/Back
        dot = np.dot(flat_vec, flat_heading)
        # Cross product for Left/Right (Y-up)
        cross = np.cross(flat_heading, flat_vec)[1] 

        direction = ""
        if dot > 0.5:
            direction = "directly in front of you"
        elif dot > 0.0:
            if cross > 0: direction = "ahead to your right"
            else: direction = "ahead to your left"
        else:
            direction = "behind you" # Should not happen in this setup

        # Refine left/right
        if abs(cross) > 0.5:
             if cross > 0: direction = "to your right"
             else: direction = "to your left"

        return f"{direction}, about {dist_str} away"

    def find_nearest_object(self, target_name: str) -> str:
        """
        Finds the object closest to the target object.
        """
        target_center = self.get_object_center(target_name)
        if target_center is None: return ""

        min_dist = float('inf')
        nearest_name = ""

        for name, obj in self.objects.items():
            if name in target_name.lower(): continue # Skip self
            
            c = np.array(obj['center'])
            d = np.linalg.norm(c - target_center)
            if d < min_dist:
                min_dist = d
                nearest_name = obj['label']
        
        if nearest_name:
            return f"It is near the {nearest_name}."
        return ""

    def process_query(self, query: str) -> Tuple[str, Optional[str]]:
        """
        Parses query and returns (Answer, ObjectName_to_Highlight).
        """
        query = query.lower()
        
        # Identify subject
        found_obj = None
        for name in self.objects.keys():
            if name in query:
                found_obj = name
                break
        
        if not found_obj:
            # Check for general queries
            if "free space" in query:
                return "There is plenty of free space in the center of the room, between the Sofa and the Table.", None
            return "I understood the query, but I didn't recognize a specific object name (Sofa, Table, Chair).", None

        # Build response
        obj_data = self.objects[found_obj]
        center = np.array(obj_data['center'])
        
        location_desc = self.calculate_relations(center)
        context_desc = self.find_nearest_object(found_obj)

        answer = f"The {obj_data['label']} is {location_desc}. {context_desc}"
        return answer, obj_data['label']

# ==============================================================================
# 4. VISUALIZATION
# ==============================================================================

def visualize_scene(pcd, objects: List[Dict], highlight_name: str = None):
    """
    Shows the scene. If highlight_name is provided:
      1. Paints the target object's BBox bright RED.
      2. Adds a floating sphere above the target object.
      3. Dims/Greys out other BBoxes.
    """
    to_draw = [pcd]
    
    # 1. Coordinate Frame (for orientation reference)
    to_draw.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0,0,0]))

    detected_name = None

    for obj in objects:
        # Create a fresh bbox to avoid mutating the original repeatedly
        center = obj['center']
        extent = obj['dimensions']
        # Re-create AxisAlignedBoundingBox from center/extent logic
        # Min = center - extent/2, Max = center + extent/2
        min_bound = np.array(center) - np.array(extent)/2
        max_bound = np.array(center) + np.array(extent)/2
        bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
        
        is_target = highlight_name and (obj['label'].lower() == highlight_name.lower())
        
        if is_target:
            detected_name = obj['label']
            # Highlight: Bright Red Box
            bbox.color = [1.0, 0.0, 0.0] 
            
            # Add a "Pointer" Sphere bouncing above
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.15)
            sphere.paint_uniform_color([1.0, 0.0, 0.0]) # Red
            # Float it 0.5m above the object
            sphere.translate(np.array(center) + np.array([0, extent[1]/2 + 0.5, 0]))
            to_draw.append(sphere)
            
            # Print for debug
            print(f"[Visualizer] HIGHLIGHTING: {obj['label']} (Red Box + Sphere)")
        else:
            # Others: Dim Grey
            bbox.color = [0.3, 0.3, 0.3]
        
        to_draw.append(bbox)

    win_name = f"Assistant: {highlight_name if highlight_name else 'Room View'}"
    print(f"[Visualizer] Opening '{win_name}'. CLOSE WINDOW to continue.")
    o3d.visualization.draw_geometries(to_draw, window_name=win_name)

# ==============================================================================
# 5. MAIN APP
# ==============================================================================

def interactive_cli(use_video: bool = False, video_path: str = "sample_room.mp4"):
    print("\n" + "="*60)
    print("   INTERACTIVE SPATIAL ASSISTANT (MVP)")
    print("   Microsoft Imagine Cup 2026")
    print("="*60)
    
    # 1. Init - Choose between synthetic or real scene
    if use_video:
        print(f"\n[Mode] REAL 3D SCENE from video: {video_path}")
        try:
            from real_scene_pipeline import RealScenePipeline
            import torch
            
            use_gpu = torch.cuda.is_available()
            pipeline = RealScenePipeline(use_gpu=use_gpu)
            pcd, objects = pipeline.process_video(video_path)
            
        except Exception as e:
            print(f"\n[ERROR] Failed to process video: {e}")
            print("[Fallback] Using Synthetic Scene instead...\n")
            import traceback
            traceback.print_exc()
            
            gen = SceneGenerator()
            pcd, gt_objects = gen.load_scene()
            detector = ObjectDetector()
            objects = detector.detect(pcd, known_objects=gt_objects)
    else:
        print("\n[Mode] SYNTHETIC SCENE (demo)")
        gen = SceneGenerator()
        pcd, gt_objects = gen.load_scene() # Defaults to synthetic if no file
        
        detector = ObjectDetector()
        objects = detector.detect(pcd, known_objects=gt_objects)
    
    # 2. Initialize Spatial Reasoning
    if not objects:
        print("\n[ERROR] No objects detected! Cannot proceed.")
        return
    
    engine = SpatialReasoningEngine(objects)
    
    print(f"\n[System] Scene Loaded. I can see: {', '.join([o['label'] for o in objects])}")
    print("\n" + "-"*60)
    print("VISUAL GUIDE - What you'll see in the 3D window:")
    print("-"*60)
    print("   POINT CLOUD: Room (grey floor, white walls, 3 colored objects)")
    print("   RGB ARROWS: Coordinate frame (Red=X, Green=Y, Blue=Z)")
    print("   GREY BOXES: Objects not being queried")
    print("  🔴 RED BOX + SPHERE: The object you asked about! ← LOOK HERE")
    print("-"*60)
    print("\n[System] Try asking:")
    print("  - 'Where is the chair?'")
    print("  - 'How far is the sofa?'")
    print("  - 'Show free space.'")
    print("  - 'exit' to quit.\n")

    # Initial Vis
    visualize_scene(pcd, objects)

    # 3. Loop
    while True:
        try:
            query = input("\nUSER >> ").strip()
        except EOFError:
            break

        if not query or query.lower() in ["exit", "quit"]:
            print("Goodbye.")
            break
            
        print(f"[Thinking] Processing '{query}'...")
        
        answer, highlight_name = engine.process_query(query)
        
        print(f"\nASSISTANT >> {answer}")
        
        if highlight_name:
            print(f"\n💡 LOOK FOR: Red box + Red sphere above the {highlight_name}")
            visualize_scene(pcd, objects, highlight_name)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Interactive Spatial Assistant with 3D Awareness")
    parser.add_argument("--video", type=str, default=None, 
                       help="Path to video file for real 3D reconstruction (e.g., sample_room.mp4)")
    parser.add_argument("--synthetic", action="store_true", 
                       help="Force synthetic scene (default if no --video)")
    
    args = parser.parse_args()
    
    if args.video:
        interactive_cli(use_video=True, video_path=args.video)
    else:
        interactive_cli(use_video=False)
