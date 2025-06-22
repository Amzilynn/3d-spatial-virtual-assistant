# app/scene_processing.py

import open3d as o3d
import numpy as np
from sklearn.cluster import DBSCAN

def segment_floor_and_walls(pcd, dist_thresh=0.02, ransac_n=3, num_iter=1000):
    """
    Returns:
      floor_model, floor_inliers,
      wall_models (list), wall_inliers (list),
      remaining_cloud (Open3D point cloud without floor+walls)
    """
    remaining = pcd
    # — 1) Extract the largest plane (assumed floor) —
    floor_model, floor_inliers = remaining.segment_plane(
        distance_threshold=dist_thresh,
        ransac_n=ransac_n,
        num_iterations=num_iter
    )
    floor = remaining.select_by_index(floor_inliers)
    remaining = remaining.select_by_index(floor_inliers, invert=True)

    # — 2) Extract up to two vertical planes (walls) —
    wall_models, wall_inliers = [], []
    for _ in range(2):
        if len(np.asarray(remaining.points)) < 100:
            break
        model, inliers = remaining.segment_plane(
            distance_threshold=dist_thresh,
            ransac_n=ransac_n,
            num_iterations=num_iter
        )
        # model = [a,b,c,d] plane normal (a,b,c)
        # check near-vertical by |normal·z| small
        if abs(model[2]) < 0.1:
            wall_models.append(model)
            wall_inliers.append(inliers)
            remaining = remaining.select_by_index(inliers, invert=True)
        else:
            break

    return floor_model, floor_inliers, wall_models, wall_inliers, remaining

def cluster_objects(pcd, eps=0.05, min_samples=30):
    """
    DBSCAN clusters the points,
    returns a dict[label] = centroid_xyz
    """
    pts = np.asarray(pcd.points)
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(pts)
    clusters = {}
    for idx, lbl in enumerate(labels):
        if lbl < 0: continue
        clusters.setdefault(lbl, []).append(idx)
    # compute centroids
    centroids = {
        lbl: pts[idxs].mean(axis=0).tolist()
        for lbl, idxs in clusters.items()
    }
    return centroids

def build_scene_description_from_pcd(ply_path: str) -> str:
    """
    High-level helper: load ply, segment, cluster, then serialize.
    """
    pcd = o3d.io.read_point_cloud(ply_path)
    # 1) Planes
    floor_model, floor_inliers, wall_models, _, remaining = segment_floor_and_walls(pcd)
    # 2) Objects
    centroids = cluster_objects(remaining)

    # — Format text —
    lines = []
    # Floor plane, if you want to include:
    # lines.append(f"FLOOR plane: {list(floor_model)}")
    lines.append("WALLS:")
    for m in wall_models:
        lines.append(f"- plane: {np.round(m,3).tolist()}")
    lines.append("")
    lines.append("OBJECTS:")
    # name objects numerically or by heuristic
    for i, (lbl, coord) in enumerate(centroids.items()):
        lines.append(f"- object_{i} at {np.round(coord,3).tolist()}")

    return "\n".join(lines)
