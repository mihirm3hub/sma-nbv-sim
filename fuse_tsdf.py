# fuse_tsdf.py — TSDF fusion (CV/+Z camera convention) + camera-usage viz + per-view refocus
import argparse, os, copy, csv, numpy as np, open3d as o3d
from utils import (
    load_cfg, load_mesh_glb, default_intr_from_cfg,
    build_clean_tensor_scene, render_depth_z_cv, look_at_cv,
    generate_base_centered_fib_cameras,
    pca_object_frame, extents_in_object_frame, generate_objectframe_base_fib_cameras,
    TSDF,
    make_spheres, make_rays, create_grid, validate_look_at
)

# ---------- main ----------
def main(cfg_path,
         frame_mode="object",
         samples=64,
         unit_autoscale=True,
         safety_scale=1.3,
         voxel=0.004,
         trunc=0.012,
         min_valid_frac=0.03,
         out_mesh="experiments/results/tsdf_mesh.ply",
         refocus=False,
         refocus_scales="1.0,0.85,0.70",
         save_diag=False,
         integrate_all=False):

    os.makedirs(os.path.dirname(out_mesh), exist_ok=True)
    stats_csv = os.path.join(os.path.dirname(out_mesh), "cam_stats.csv")

    cfg = load_cfg(cfg_path)

    # 1) Load mesh
    mesh, center_w, extent_w = load_mesh_glb(cfg["mesh_path"], unit_autoscale=unit_autoscale)
    scene = build_clean_tensor_scene(mesh)
    intr  = default_intr_from_cfg(cfg)
    print(f"Intrinsics: {intr}")

    # 2) Build hemisphere
    hemi = cfg["hemisphere"]
    phi_min, phi_max = float(hemi["phi_deg_min"]), float(hemi["phi_deg_max"])
    z_margin, r0_m   = float(hemi["z_margin_m"]), float(hemi["r0_m"])

    if frame_mode.lower() == "object":
        center_o, x_o, y_o, z_o = pca_object_frame(mesh)
        ext_obj, _, _ = extents_in_object_frame(mesh, center_o, (x_o, y_o, z_o))
        cams = generate_objectframe_base_fib_cameras(
            mesh, center_o, x_o, y_o, z_o, ext_obj,
            phi_min_deg=phi_min, phi_max_deg=phi_max,
            z_margin=z_margin, r0_m=r0_m,
            samples=samples, safety_scale=safety_scale
        )
        target_point = center_o
        print("Using OBJECT-frame hemisphere.")
    else:
        cams = generate_base_centered_fib_cameras(
            mesh, center_w, extent_w,
            phi_min_deg=phi_min, phi_max_deg=phi_max,
            z_margin=z_margin, r0_m=r0_m,
            samples=samples, safety_scale=safety_scale
        )
        target_point = center_w
        print("Using WORLD-frame hemisphere.")

    print(f"#views: {len(cams)}, phi:[{phi_min},{phi_max}], z_margin:{z_margin}, r0_m:{r0_m}")

    # sanity check
    validate_look_at(cams, target_point)

    # 3) TSDF volume
    tsdf = TSDF(voxel=voxel, trunc=trunc)

    # 4) Integrate views
    H, W = intr["height"], intr["width"]
    total_px = H * W
    used_mask = np.zeros(len(cams), dtype=bool)
    valid_counts = np.zeros(len(cams), dtype=int)

    scales = [float(s) for s in refocus_scales.split(",") if s.strip()] or [1.0]

    # optional diagnostics
    if save_diag:
        try:
            import imageio
        except Exception:
            imageio = None
            print("note: imageio not available; skipping depth saves (pip install imageio)")
        diag_dir = os.path.join(os.path.dirname(out_mesh), "diag")
        os.makedirs(diag_dir, exist_ok=True)
        diag_stride = max(1, len(cams) // 10)

    def render_once(pos):
        cam_T = look_at_cv(pos, target_point)  # +Z forward
        dtrunc = max(1.5 * trunc,
                     1.2 * float(np.linalg.norm(target_point - pos)),
                     0.6)
        depth_z = render_depth_z_cv(scene, cam_T, intr, depth_trunc=dtrunc)
        valid = int((depth_z > 0).sum())
        frac  = valid / total_px
        return depth_z, cam_T, dtrunc, valid, frac

    for i, cam_pos0 in enumerate(cams):
        vec = target_point - cam_pos0
        d0 = float(np.linalg.norm(vec))
        if d0 < 1e-6:
            vec = np.array([0,0,1.0], float); d0 = 1.0
        dir_to_target = vec / d0

        used = False
        best_valid, best_frac = 0, 0.0
        last_depth = None

        for s in (scales if refocus else [1.0]):
            cam_pos = target_point - dir_to_target * (d0 * s)
            depth_z, cam_T, dtrunc, valid, frac = render_once(cam_pos)
            last_depth = depth_z
            if valid > best_valid:
                best_valid, best_frac = valid, frac

            if integrate_all or frac >= min_valid_frac:
                tsdf.integrate(depth_z, intr, cam_T, depth_scale=1000.0, depth_trunc=dtrunc)
                used_mask[i] = True
                valid_counts[i] = valid
                tag = "USED(all)" if integrate_all else "USED"
                print(f"[{i+1:03d}/{len(cams)}] {tag:8s} valid:{valid:6d} ({frac:.1%})  dist:{d0*s:.3f}  trunc:{dtrunc:.3f}")
                used = True
                break

        if not used:
            valid_counts[i] = best_valid
            print(f"[{i+1:03d}/{len(cams)}] skip      valid:{best_valid:6d} ({best_frac:.1%})")

        if save_diag and ('imageio' in globals() and imageio is not None) and (i % diag_stride == 0) and (last_depth is not None):
            png = (np.clip(last_depth * 1000.0, 0, 65535).astype(np.uint16))
            imageio.imwrite(os.path.join(diag_dir, f"depth_{i:03d}.png"), png)

    # Save per-view stats
    with open(stats_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idx","x","y","z","valid_px","valid_frac","used"])
        for i, p in enumerate(cams):
            frac = valid_counts[i] / total_px
            w.writerow([i, p[0], p[1], p[2], valid_counts[i], frac, bool(used_mask[i])])
    print(f"Saved camera stats: {stats_csv}")

    # 5) Extract + save
    rec = tsdf.extract_mesh()
    o3d.io.write_triangle_mesh(out_mesh, rec)
    v = np.asarray(rec.vertices).shape[0]
    f = np.asarray(rec.triangles).shape[0]
    print(f"Saved: {out_mesh}  V={v}, F={f}")
    assert f > 2000, "Reconstruction too sparse — increase samples or reduce voxel size."
    print("PASS: TSDF fusion produced a reasonable mesh.")

    # 6) Side-by-side preview
    orig_vis = copy.deepcopy(mesh); orig_vis.paint_uniform_color([0.8, 0.8, 0.8])
    rec_vis  = copy.deepcopy(rec);  rec_vis.paint_uniform_color([1.0, 0.6, 0.0])

    gap_scale = 0.6
    max_extent = float(np.max(extent_w))
    shift = gap_scale * max_extent
    orig_vis.translate([-shift, 0, 0], relative=True)
    rec_vis.translate([ shift, 0, 0], relative=True)

    bbox_orig = orig_vis.get_axis_aligned_bounding_box(); bbox_orig.color = (0, 1, 0)
    bbox_rec  = rec_vis.get_axis_aligned_bounding_box();  bbox_rec.color  = (1, 0, 0)

    combined_min = np.minimum(bbox_orig.get_min_bound(), bbox_rec.get_min_bound())
    combined_max = np.maximum(bbox_orig.get_max_bound(), bbox_rec.get_max_bound())
    center_xy = 0.5 * (combined_min[:2] + combined_max[:2])
    base_z = min(bbox_orig.get_min_bound()[2], bbox_rec.get_min_bound()[2]) - 1e-3

    span_x = combined_max[0] - combined_min[0]
    span_y = combined_max[1] - combined_min[1]
    grid_size = max(span_x, span_y) * 0.75

    grid_floor = create_grid(size=float(grid_size), step=max_extent * 0.05, z=float(base_z))
    grid_floor.translate([center_xy[0], center_xy[1], 0], relative=True)

    o3d.visualization.draw_geometries(
        [grid_floor, orig_vis, bbox_orig, rec_vis, bbox_rec],
        window_name="Original (left) vs TSDF Reconstruction (right)",
        width=1400, height=800
    )

    # 7) Camera usage visualization
    aabb = mesh.get_axis_aligned_bounding_box(); aabb.color = (1.0, 0.2, 0.2)
    used_pts   = cams[used_mask]
    unused_pts = cams[~used_mask]

    geoms = [mesh, aabb]
    geoms += make_spheres(used_pts,   radius=0.01, color=(0.2, 0.8, 0.2))
    geoms += make_spheres(unused_pts, radius=0.01, color=(0.9, 0.2, 0.2))
    geoms.append(make_rays(used_pts, target_point, every=max(1, len(used_pts)//24), color=(0.2,0.7,0.2)))

    print(f"Camera usage: {used_pts.shape[0]} used / {cams.shape[0]} total "
          f"({100.0*used_pts.shape[0]/max(1,cams.shape[0]):.1f}%)")
    o3d.visualization.draw_geometries(
        geoms,
        window_name="Hemisphere cameras: green=used, red=skipped",
        width=1280, height=800,
        lookat=target_point.tolist(), front=[0,-1,0], up=[0,0,1], zoom=0.7
    )

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True, help="experiments/configs/can_hemisphere.yaml")
    ap.add_argument("--frame", choices=["world","object"], default="object")
    ap.add_argument("--samples", type=int, default=64)
    ap.add_argument("--no-unit-autoscale", action="store_true")
    ap.add_argument("--safety-scale", type=float, default=1.3)
    ap.add_argument("--voxel", type=float, default=0.004)
    ap.add_argument("--trunc", type=float, default=0.012)
    ap.add_argument("--min-valid-frac", type=float, default=0.03)
    ap.add_argument("--refocus", action="store_true")
    ap.add_argument("--refocus-scales", type=str, default="1.0,0.85,0.70")
    ap.add_argument("--save-diag", action="store_true")
    ap.add_argument("--integrate-all", action="store_true")
    ap.add_argument("--out", type=str, default="experiments/results/tsdf_mesh.ply")
    args = ap.parse_args()

    main(args.cfg,
         frame_mode=args.frame,
         samples=args.samples,
         unit_autoscale=not args.no_unit_autoscale,
         safety_scale=args.safety_scale,
         voxel=args.voxel,
         trunc=args.trunc,
         min_valid_frac=args.min_valid_frac,
         out_mesh=args.out,
         refocus=args.refocus,
         refocus_scales=args.refocus_scales,
         save_diag=args.save_diag,
         integrate_all=args.integrate_all)
