# render_multi_depth.py — render depth maps for "red" (skipped) cameras from cam_stats.csv
import argparse, os, csv, numpy as np, imageio.v2 as imageio, open3d as o3d
from utils import (
    load_cfg, load_mesh_glb, default_intr_from_cfg,
    build_clean_tensor_scene, render_depth_z_cv, look_at_cv,
    pca_object_frame,  # for object-frame target
    validate_look_at,  # shared pointing check
)

# --- Local helpers (file IO + frustum viz) ---

def read_cam_stats(csv_path):
    rows = []
    with open(csv_path, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append({
                "idx": int(row["idx"]),
                "pos": np.array([float(row["x"]), float(row["y"]), float(row["z"])]),
                "valid_px": int(float(row["valid_px"])),
                "valid_frac": float(row["valid_frac"]),
                "used": (row["used"].strip().lower() in ("1","true","t","yes","y")),
            })
    return rows

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

def save_depth_16u(path_png, depth_m, depth_trunc):
    depth_mm = (np.clip(depth_m, 0, depth_trunc) * 1000.0).astype(np.uint16)
    imageio.imwrite(path_png, depth_mm)

def colorize_depth(depth_m, depth_trunc):
    d = np.clip(depth_m / max(1e-6, depth_trunc), 0.0, 1.0)
    d = (255.0 * (1.0 - d)).astype(np.uint8)
    return np.stack([d, d, d], axis=-1)

def make_camera_frustum(intr, cam_T, near=0.05, far=0.3, color=(0.9, 0.1, 0.1)):
    """Create frustum lines in CV convention (+Z forward)."""
    fx, fy, cx, cy = float(intr["fx"]), float(intr["fy"]), float(intr["cx"]), float(intr["cy"])
    W, H = int(intr["width"]), int(intr["height"])

    def corners_at(z):
        return np.array([
            [(0 - cx) * z / fx, (0 - cy) * z / fy, z],
            [(W - cx) * z / fx, (0 - cy) * z / fy, z],
            [(W - cx) * z / fx, (H - cy) * z / fy, z],
            [(0 - cx) * z / fx, (H - cy) * z / fy, z],
        ], dtype=float)

    Cn = corners_at(near)
    Cf = corners_at(far)
    cam_origin = np.zeros((1, 3), dtype=float)

    P = np.vstack([Cn, Cf, cam_origin])                         # 9 points
    Pw = (cam_T @ np.hstack([P, np.ones((len(P), 1))]).T).T[:, :3]

    # indices based on P order above
    lines = []
    lines += [[0,1],[1,2],[2,3],[3,0]]   # near
    lines += [[4,5],[5,6],[6,7],[7,4]]   # far
    lines += [[0,4],[1,5],[2,6],[3,7]]   # connect
    lines += [[8,0],[8,1],[8,2],[8,3]]   # origin->near

    ls = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(Pw),
        lines=o3d.utility.Vector2iVector(np.asarray(lines, dtype=np.int32))
    )
    ls.paint_uniform_color(color)
    return ls

# --- Main ---

def main(cfg_path,
         cam_stats_csv="experiments/results/cam_stats.csv",
         frame="object",
         out_dir="experiments/results/red_depth",
         max_views=64,
         depth_trunc=None,
         unit_autoscale=True,
         save_npz=False,
         visualize=True):

    ensure_dir(out_dir)
    cfg = load_cfg(cfg_path)
    mesh, center_w, extent_w = load_mesh_glb(cfg["mesh_path"], unit_autoscale=unit_autoscale)
    scene = build_clean_tensor_scene(mesh)
    intr  = default_intr_from_cfg(cfg)

    if frame.lower() == "object":
        center_o, *_ = pca_object_frame(mesh)
        target_point = center_o
        frame_name = "OBJECT"
    else:
        target_point = center_w
        frame_name = "WORLD"

    stats = read_cam_stats(cam_stats_csv)
    red = [s for s in stats if not s["used"]]
    red = sorted(red, key=lambda r: r["idx"])[:max_views]

    if not red:
        print("No red (skipped) cameras found.")
        return

    # Validate camera pointing before rendering (shared util)
    validate_look_at(np.array([r["pos"] for r in red]), target_point)

    print(f"Rendering {len(red)} red views ({frame_name}-frame) → {out_dir}")

    default_trunc = float(cfg.get("tsdf", {}).get("trunc", 0.012)) if depth_trunc is None else float(depth_trunc)
    manifest_rows = []
    frustum_geoms = []

    for k, r in enumerate(red):
        pos = r["pos"]
        dist = float(np.linalg.norm(target_point - pos))
        dtrunc = max(1.5 * default_trunc, 1.2 * dist, 0.6)

        cam_T = look_at_cv(pos, target_point)
        depth_z = render_depth_z_cv(scene, cam_T, intr, depth_trunc=dtrunc)

        valid = int((depth_z > 0).sum())
        frac = valid / (intr["width"] * intr["height"])
        print(f"[{k+1:03d}/{len(red)}] idx={r['idx']:03d} valid={valid} ({frac:.1%}) dist={dist:.3f} trunc={dtrunc:.3f}")

        base = os.path.join(out_dir, f"red_{r['idx']:03d}")
        png_path = base + ".png"
        jpg_path = base + "_viz.jpg"
        npz_path = base + ".npz"

        save_depth_16u(png_path, depth_z, dtrunc)
        imageio.imwrite(jpg_path, colorize_depth(depth_z, dtrunc))

        if save_npz:
            np.savez_compressed(
                npz_path,
                depth=depth_z.astype(np.float32),
                cam_T_world=cam_T.astype(np.float32),
                intr=np.array([intr["fx"], intr["fy"], intr["cx"], intr["cy"], intr["width"], intr["height"]],
                              dtype=np.float32),
                pos=pos.astype(np.float32),
                target=target_point.astype(np.float32),
                depth_trunc=np.array(dtrunc, dtype=np.float32),
            )

        manifest_rows.append([r["idx"], png_path, jpg_path, valid, frac, dist, dtrunc, pos[0], pos[1], pos[2]])

        if visualize:
            frustum_geoms.append(make_camera_frustum(intr, cam_T, near=0.05, far=max(0.15, dist * 0.3), color=(0.9, 0.1, 0.1)))

    manifest_csv = os.path.join(out_dir, "manifest.csv")
    with open(manifest_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idx", "depth_png", "preview_jpg", "valid_px", "valid_frac", "cam_dist", "depth_trunc",
                    "cam_x", "cam_y", "cam_z"])
        w.writerows(manifest_rows)
    print(f"Saved manifest: {manifest_csv}")

    if visualize:
        print("Launching Open3D viewer with skipped camera frustums...")
        geoms = [mesh] + frustum_geoms
        o3d.visualization.draw_geometries(
            geoms,
            window_name="Skipped (red) cameras",
            lookat=target_point.tolist(),
            front=[0, -1, 0],
            up=[0, 0, 1],
            zoom=0.7
        )

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--cam-stats", default="experiments/results/cam_stats.csv")
    ap.add_argument("--frame", choices=["world", "object"], default="object")
    ap.add_argument("--out-dir", default="experiments/results/red_depth")
    ap.add_argument("--max-views", type=int, default=64)
    ap.add_argument("--depth-trunc", type=float, default=None)
    ap.add_argument("--no-unit-autoscale", action="store_true")
    ap.add_argument("--save-npz", action="store_true")
    ap.add_argument("--no-viz", action="store_true", help="Disable Open3D frustum viewer")
    args = ap.parse_args()

    main(args.cfg,
         cam_stats_csv=args.cam_stats,
         frame=args.frame,
         out_dir=args.out_dir,
         max_views=args.max_views,
         depth_trunc=args.depth_trunc,
         unit_autoscale=not args.no_unit_autoscale,
         save_npz=args.save_npz,
         visualize=not args.no_viz)
