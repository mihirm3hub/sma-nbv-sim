# view_methods_frustums.py
# Visualize camera frustums for multiple methods (SMA/GREEDY/RANDOM) in Open3D.
# Usage examples at the end.

import argparse, os, csv
import numpy as np
import open3d as o3d

from utils import (
    load_cfg, load_mesh_glb, default_intr_from_cfg,
    build_clean_tensor_scene, look_at_cv,
    generate_base_centered_fib_cameras,
    pca_object_frame, extents_in_object_frame, generate_objectframe_base_fib_cameras,
)

COLORS = {
    "SMA":     (0.10, 0.80, 0.20),  # green
    "GREEDY":  (0.10, 0.45, 0.95),  # blue
    "RANDOM":  (1.00, 0.55, 0.10),  # orange
}

def read_cameras_csv(path):
    cams = []
    with open(path, "r", newline="") as f:
        r = csv.DictReader(f)
        cols = [c.lower() for c in r.fieldnames]
        if not {"idx","x","y","z"}.issubset(cols):
            raise ValueError(f"{path} must have header idx,x,y,z")
        for row in r:
            cams.append(np.array([float(row["x"]), float(row["y"]), float(row["z"])], dtype=np.float32))
    if not cams:
        raise ValueError(f"No cameras found in {path}")
    return np.vstack(cams)

def make_spheres(points, radius=0.008, color=(0.7,0.7,0.7)):
    geoms = []
    for p in points:
        s = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        s.paint_uniform_color(color); s.translate(p)
        geoms.append(s)
    return geoms

def make_rays(points, target, stride=4, color=(0.5,0.5,0.5)):
    # Draw a subset of rays to keep the view clean
    pts, lines = [], []
    idx = list(range(0, len(points), max(1, stride)))
    for i, k in enumerate(idx):
        pts.extend([points[k], target])
        lines.append([2*i, 2*i+1])
    ls = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(np.asarray(pts)),
        lines=o3d.utility.Vector2iVector(np.asarray(lines, dtype=np.int32)),
    )
    ls.paint_uniform_color(color)
    return ls

def make_frustum(intr, cam_T, near=0.05, far=0.30, color=(0.2,0.8,0.2)):
    fx, fy, cx, cy = float(intr["fx"]), float(intr["fy"]), float(intr["cx"]), float(intr["cy"])
    W, H = int(intr["width"]), int(intr["height"])

    def corners_at(z):
        return np.array([
            [(0 - cx) * z / fx, (0 - cy) * z / fy, z],
            [(W - cx) * z / fx, (0 - cy) * z / fy, z],
            [(W - cx) * z / fx, (H - cy) * z / fy, z],
            [(0 - cx) * z / fx, (H - cy) * z / fy, z],
        ], dtype=float)

    Cn, Cf = corners_at(near), corners_at(far)
    origin = np.zeros((1,3))
    P = np.vstack([Cn, Cf, origin])
    Pw = (cam_T @ np.hstack([P, np.ones((len(P),1))]).T).T[:, :3]

    lines = []
    lines += [[0,1],[1,2],[2,3],[3,0]]
    lines += [[4,5],[5,6],[6,7],[7,4]]
    lines += [[0,4],[1,5],[2,6],[3,7]]
    lines += [[8,0],[8,1],[8,2],[8,3]]

    ls = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(Pw),
        lines=o3d.utility.Vector2iVector(np.asarray(lines, dtype=np.int32))
    )
    ls.paint_uniform_color(color)
    return ls

def add_method_geoms(method_name, pts, intr, target_point, frustum_stride=1):
    color = COLORS.get(method_name.upper(), (0.9,0.9,0.9))
    geoms = []
    # Spheres at camera centers
    geoms += make_spheres(pts, radius=0.010, color=color)
    # A few rays for directionality
    geoms.append(make_rays(pts, target_point, stride=max(2, len(pts)//24), color=color))
    # Frustums (downsample if many)
    step = max(1, frustum_stride)
    for p in pts[::step]:
        T = look_at_cv(p, target_point)
        dist = float(np.linalg.norm(target_point - p))
        far = max(0.18, 0.35 * dist)
        geoms.append(make_frustum(intr, T, near=0.05, far=far, color=color))
    return geoms

def main(cfg_path,
         frame_mode="object",
         unit_autoscale=True,
         overlay=True,
         frustum_stride=1,
         samples=256,
         safety_scale=1.3,
         sma_csv="",
         greedy_csv="",
         random_csv=""):

    cfg = load_cfg(cfg_path)
    intr = default_intr_from_cfg(cfg)

    # Load mesh + decide target point
    mesh, center_w, extent_w = load_mesh_glb(cfg["mesh_path"], unit_autoscale=unit_autoscale)

    if frame_mode.lower() == "object":
        # object center via PCA frame (consistent with your other scripts)
        center_o, x_o, y_o, z_o = pca_object_frame(mesh)
        target_point = center_o
    else:
        target_point = center_w

    aabb = mesh.get_axis_aligned_bounding_box(); aabb.color = (1.0, 0.2, 0.2)

    # Collect methods present
    method_paths = []
    if sma_csv:    method_paths.append(("SMA", sma_csv))
    if greedy_csv: method_paths.append(("GREEDY", greedy_csv))
    if random_csv: method_paths.append(("RANDOM", random_csv))
    if not method_paths:
        raise ValueError("Provide at least one of --sma-csv/--greedy-csv/--random-csv.")

    if overlay:
        # One combined window
        geoms = [mesh, aabb]
        for name, path in method_paths:
            pts = read_cameras_csv(path)
            geoms += add_method_geoms(name, pts, intr, target_point, frustum_stride=frustum_stride)

        o3d.visualization.draw_geometries(
            geoms,
            window_name="Camera frustums (overlay): SMA(green) / GREEDY(blue) / RANDOM(orange)",
            width=1500, height=950,
            lookat=target_point.tolist(), front=[0,-1,0], up=[0,0,1], zoom=0.7
        )
    else:
        # Separate windows per method
        for name, path in method_paths:
            geoms = [mesh, aabb]
            pts = read_cameras_csv(path)
            geoms += add_method_geoms(name, pts, intr, target_point, frustum_stride=frustum_stride)
            o3d.visualization.draw_geometries(
                geoms,
                window_name=f"Frustums — {name}",
                width=1400, height=900,
                lookat=target_point.tolist(), front=[0,-1,0], up=[0,0,1], zoom=0.7
            )

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True, help="experiments/configs/can_hemisphere.yaml")
    ap.add_argument("--frame", choices=["world","object"], default="object")
    ap.add_argument("--no-unit-autoscale", action="store_true")
    # Which CSVs to show (provide any subset)
    ap.add_argument("--sma-csv", default="experiments/results/sma_selected.csv")
    ap.add_argument("--greedy-csv", default="")
    ap.add_argument("--random-csv", default="")
    # Viz options
    ap.add_argument("--overlay", action="store_true", help="Show all methods together in a single window")
    ap.add_argument("--frustum-stride", type=int, default=1, help="Draw every Nth frustum to declutter")
    # (kept for compatibility; not used directly here)
    ap.add_argument("--samples", type=int, default=256)
    ap.add_argument("--safety-scale", type=float, default=1.3)
    args = ap.parse_args()

    main(
        args.cfg,
        frame_mode=args.frame,
        unit_autoscale=not args.no_unit_autoscale,
        overlay=args.overlay,
        frustum_stride=max(1, args.frustum_stride),
        samples=args.samples,
        safety_scale=args.safety_scale,
        sma_csv=args.sma_csv,
        greedy_csv=args.greedy_csv,
        random_csv=args.random_csv,
    )
