# sma_nbv.py — SMA-guided NBV over hemisphere candidates + TSDF fusion
# Adds optional 3-way viewer: GT | Baseline (fuse_tsdf.py) | SMA
# NEW: --use-3d-coverage uses surface-area coverage (front-facing & visible) for scoring.
# NEW: A second viewer shows camera viewpoints (all, selected, actually used) + frustums/rays.

import argparse, os, copy, math, csv
import numpy as np
import open3d as o3d

from utils import (
    load_cfg, load_mesh_glb, default_intr_from_cfg,
    build_clean_tensor_scene, render_depth_z_cv, look_at_cv,
    generate_base_centered_fib_cameras,
    pca_object_frame, extents_in_object_frame, generate_objectframe_base_fib_cameras,
    TSDF,
)

# -------------------- scoring helpers (2D) --------------------

def binary_mask(depth_z: np.ndarray) -> np.ndarray:
    return (depth_z > 0).astype(np.uint8)

def iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union + 1e-9)

def coverage_gain(mask: np.ndarray, covered_mask: np.ndarray) -> float:
    newly = np.logical_and(mask == 1, covered_mask == 0).sum()
    total = mask.size
    return float(newly) / float(total)

def overlap_penalty(mask: np.ndarray, selected_masks: list) -> float:
    if not selected_masks:
        return 0.0
    return max(iou(mask, m) for m in selected_masks)

def view_quality_from_depth(depth_z: np.ndarray) -> float:
    nz = depth_z[depth_z > 0]
    if nz.size < 100:
        return 0.0
    med = float(np.median(nz))
    q25, q75 = np.percentile(nz, [25, 75])
    spread = float(q75 - q25 + 1e-6)
    norm = med / (q75 + 1e-6)
    score = math.exp(-((norm - 0.6) ** 2) / 0.15)
    score *= 1.0 / (1.0 + 2.0 * spread)
    return float(np.clip(score, 0.0, 1.0))

def detection_bonus_center_hit(scene, cam_T, intr) -> float:
    H, W = intr["height"], intr["width"]
    cx, cy = int(intr["cx"]), int(intr["cy"])
    patch = np.zeros((H, W), dtype=np.uint8)
    patch[max(0,cy-1):min(H,cy+2), max(0,cx-1):min(W,cx+2)] = 1
    d = render_depth_z_cv(scene, cam_T, intr, depth_trunc=3.0)
    return 1.0 if (d[patch == 1] > 0).mean() > 0.6 else 0.0

def compute_view_score_2d(scene,
                          cam_pos,
                          target_point,
                          intr,
                          covered_mask,
                          selected_masks,
                          depth_trunc,
                          weights):
    cam_T = look_at_cv(cam_pos, target_point)
    depth_z = render_depth_z_cv(scene, cam_T, intr, depth_trunc=depth_trunc)
    mask = (depth_z > 0).astype(np.uint8)

    cov = coverage_gain(mask, covered_mask)
    qual = view_quality_from_depth(depth_z)
    det  = detection_bonus_center_hit(scene, cam_T, intr)
    ovp  = overlap_penalty(mask, selected_masks)

    w1, w2, w3, w4 = weights
    score = w1 * cov + w2 * qual + w3 * det - w4 * ovp
    return score, mask, depth_z

# -------------------- NEW: 3D surface-coverage helpers --------------------

def _normalize(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / (n + 1e-9)

def sample_surface_points(mesh, n=6000, seed=0):
    """Area-weighted random sampling of triangle surface points + normals + per-sample area weight."""
    rng = np.random.default_rng(seed)
    V = np.asarray(mesh.vertices); F = np.asarray(mesh.triangles)
    if V.size == 0 or F.size == 0:
        raise ValueError("Empty mesh for sampling.")
    a = V[F[:,0]]; b = V[F[:,1]]; c = V[F[:,2]]
    tri_norm = np.cross(b - a, c - a)
    tri_area = 0.5 * np.linalg.norm(tri_norm, axis=1)
    total_area = float(tri_area.sum())
    if total_area <= 1e-12:
        raise ValueError("Degenerate mesh area.")
    prob = tri_area / total_area
    prob = prob / (prob.sum() + 1e-12)  # explicit renorm

    face_idx = rng.choice(len(F), size=n, p=prob)
    u = np.sqrt(rng.random(n)); v = rng.random(n)
    Aa = V[F[face_idx,0]]; Bb = V[F[face_idx,1]]; Cc = V[F[face_idx,2]]
    P = (1-u)[:,None]*Aa + (u*(1-v))[:,None]*Bb + (u*v)[:,None]*Cc

    FN = np.cross(Bb - Aa, Cc - Aa)
    FN = _normalize(FN)
    N = FN

    # approximate per-sample weight
    W = tri_area[face_idx] / (prob[face_idx] * n + 1e-9)
    return P.astype(np.float32), N.astype(np.float32), W.astype(np.float32)

def visible_area_gain(scene, cam_pos, target, P, N, W, front_thresh=0.2, eps=1e-4):
    cam_T = look_at_cv(cam_pos, target)
    C = cam_T[:3, 3]
    VEC = P - C
    dist = np.linalg.norm(VEC, axis=1) + 1e-9
    DIRc = VEC / dist[:,None]  # from cam to point
    # front-facing wrt camera (normal points roughly toward camera)
    ff = (np.sum(N * (-DIRc), axis=1) > front_thresh)

    # occlusion check: from just above surface back toward camera
    origins = P + N * eps
    rays = np.concatenate([origins, -DIRc], axis=1).astype(np.float32)
    out = scene.cast_rays(o3d.core.Tensor(rays))
    t_hit = out["t_hit"].numpy()
    vis = ff & (np.isfinite(t_hit) & (t_hit >= (dist - 2*eps)))
    return vis, W, cam_T

def compute_view_score_3d(scene, cam_pos, target_point, intr,
                          P_surf, N_surf, W_surf, covered_pts,
                          selected_dirs, depth_trunc, weights,
                          div_weight=0.15, front_thresh=0.2):
    # 3D visibility
    vis, w, cam_T = visible_area_gain(scene, cam_pos, target_point, P_surf, N_surf, W_surf,
                                      front_thresh=front_thresh)
    newly = vis & (~covered_pts)
    cov_area = float(w[newly].sum()) / float(w.sum() + 1e-9)

    # keep some of the 2D terms (quality & soft redundancy via mask mean)
    depth_z = render_depth_z_cv(scene, cam_T, intr, depth_trunc=depth_trunc)
    mask = (depth_z > 0).astype(np.uint8)
    qual = view_quality_from_depth(depth_z)
    det  = detection_bonus_center_hit(scene, cam_T, intr)

    # angular diversity reward
    dir_w = target_point - cam_pos
    dir_w = dir_w / (np.linalg.norm(dir_w) + 1e-9)
    if selected_dirs:
        sims = [abs(np.dot(dir_w, d)) for d in selected_dirs]
        diversity = 1.0 - max(sims)
    else:
        diversity = 1.0

    w1, w2, w3, w4 = weights
    score = w1*cov_area + w2*qual + w3*det + div_weight*diversity - w4*float(mask.mean())
    return score, mask, depth_z, vis, dir_w

# -------------------- discrete SMA (over candidate indices) --------------------

def sma_optimize(scene, cams, target_point, intr,
                 depth_trunc, weights,
                 budget=16, pop=40, iters=30, seed=0, verbose=True,
                 use_3d=False, P_surf=None, N_surf=None, W_surf=None,
                 div_weight=0.15, front_thresh=0.2):
    rng = np.random.default_rng(seed)
    N = len(cams)
    H, W = intr["height"], intr["width"]

    # state for 2D score
    covered_mask = np.zeros((H, W), dtype=np.uint8)
    selected_masks = []

    # state for 3D score
    if use_3d:
        assert P_surf is not None and N_surf is not None and W_surf is not None
        covered_pts = np.zeros(len(P_surf), dtype=bool)
        selected_dirs = []

    selected = []
    remaining = set(range(N))

    for step in range(budget):
        if not remaining:
            if verbose: print("No more candidates remaining.")
            break
        if verbose:
            print(f"\n=== SMA NBV step {step+1}/{budget} ===")

        init_size = min(pop, len(remaining))
        pop_idx = rng.choice(list(remaining), size=init_size, replace=False)
        fitness = np.zeros(init_size, dtype=np.float32)

        # caches
        cache_2d = {}      # idx -> (score, mask)
        cache_3d = {}      # idx -> (score, mask, vis, dir)

        # evaluate initial
        for i, idx in enumerate(pop_idx):
            if use_3d:
                s, m, _, vis, dvec = compute_view_score_3d(
                    scene, cams[idx], target_point, intr,
                    P_surf, N_surf, W_surf, covered_pts,
                    selected_dirs, depth_trunc, weights,
                    div_weight=div_weight, front_thresh=front_thresh
                )
                cache_3d[idx] = (s, m, vis, dvec)
            else:
                s, m, _ = compute_view_score_2d(
                    scene, cams[idx], target_point, intr,
                    covered_mask=covered_mask, selected_masks=selected_masks,
                    depth_trunc=depth_trunc, weights=weights
                )
                cache_2d[idx] = (s, m)
            fitness[i] = s

        gbest_idx = int(pop_idx[np.argmax(fitness)])
        gbest_fit = float(fitness.max())

        # SMA-esque updates
        for _ in range(iters):
            order = np.argsort(-fitness)
            for rank_pos, p_i in enumerate(order):
                idx = int(pop_idx[p_i])
                step_span = max(1, int(0.05 * N * (1.0 - rank_pos / max(1, len(order)-1))))
                if rng.random() < 0.65:
                    direction = np.sign(gbest_idx - idx)
                    new_idx = int(idx + direction * rng.integers(1, step_span+1))
                else:
                    new_idx = int(idx + rng.integers(-step_span, step_span+1))
                new_idx = int(np.clip(new_idx, 0, N-1))

                if new_idx not in remaining:
                    # try to snap to an available candidate
                    for _try in range(2):
                        trial = int(rng.choice(list(remaining)))
                        if trial != idx:
                            new_idx = trial
                            break

                if use_3d:
                    if new_idx in cache_3d:
                        s, _, _, _ = cache_3d[new_idx]
                    else:
                        s, m, _, vis, dvec = compute_view_score_3d(
                            scene, cams[new_idx], target_point, intr,
                            P_surf, N_surf, W_surf, covered_pts,
                            selected_dirs, depth_trunc, weights,
                            div_weight=div_weight, front_thresh=front_thresh
                        )
                        cache_3d[new_idx] = (s, m, vis, dvec)
                else:
                    if new_idx in cache_2d:
                        s, _ = cache_2d[new_idx]
                    else:
                        s, m, _ = compute_view_score_2d(
                            scene, cams[new_idx], target_point, intr,
                            covered_mask=covered_mask, selected_masks=selected_masks,
                            depth_trunc=depth_trunc, weights=weights
                        )
                        cache_2d[new_idx] = (s, m)

                if s >= fitness[p_i]:
                    pop_idx[p_i] = new_idx
                    fitness[p_i] = s
                    if s > gbest_fit:
                        gbest_fit = s
                        gbest_idx = new_idx

        # commit the best of this step
        choose_idx = int(gbest_idx)
        selected.append(choose_idx)
        remaining.discard(choose_idx)

        if use_3d:
            s, m, vis, dvec = cache_3d[choose_idx]
            covered_pts |= vis
            selected_dirs.append(dvec)
            covered_pct = 100.0 * covered_pts.mean()
            if verbose:
                print(f"Picked idx {choose_idx}  score={s:.4f}  3D covered≈{covered_pct:.2f}%")
        else:
            s, m = cache_2d[choose_idx]
            selected_masks.append(m.copy())
            covered_mask = np.logical_or(covered_mask == 1, m == 1).astype(np.uint8)
            covered_pct = 100.0 * (covered_mask.sum() / covered_mask.size)
            if verbose:
                print(f"Picked idx {choose_idx}  score={s:.4f}  2D covered≈{covered_pct:.2f}%")

    return selected

# -------------------- TSDF fuse (selected only) --------------------

def fuse_selected(scene, intr, cams, target_point, sel_idx, voxel=0.004, trunc=0.012):
    tsdf = TSDF(voxel=voxel, trunc=trunc)
    H, W = intr["height"], intr["width"]
    total_px = H * W
    used = []
    for k, i in enumerate(sel_idx):
        cam_pos = cams[i]
        cam_T = look_at_cv(cam_pos, target_point)
        dist = float(np.linalg.norm(target_point - cam_pos))
        dtrunc = max(1.5*trunc, 1.2*dist, 0.6)
        depth_z = render_depth_z_cv(scene, cam_T, intr, depth_trunc=dtrunc)
        valid = int((depth_z > 0).sum())
        frac = valid / total_px
        print(f"[{k+1:02d}/{len(sel_idx)}] idx={i:03d} valid={valid} ({frac:.1%}) dist={dist:.3f} trunc={dtrunc:.3f}")
        if valid > 500:  # can lower to 200 if sides are still too sparse
            tsdf.integrate(depth_z, intr, cam_T, depth_scale=1000.0, depth_trunc=dtrunc)
            used.append(i)
    rec = tsdf.extract_mesh()
    return rec, used

# -------------------- visualization --------------------

def make_grid(size=1.0, step=0.05, z=0.0):
    pts, lines = [], []
    n = int(size / step)
    for i in range(-n, n + 1):
        pts.extend([[-size, i*step, z], [size, i*step, z]]); lines.append([len(pts)-2, len(pts)-1])
        pts.extend([[i*step, -size, z], [i*step,  size, z]]); lines.append([len(pts)-2, len(pts)-1])
    grid = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(np.asarray(pts)),
        lines=o3d.utility.Vector2iVector(np.asarray(lines))
    )
    grid.paint_uniform_color([0.4, 0.4, 0.4])
    return grid

def tri_view(gt_mesh, baseline_mesh, sma_mesh, title="GT | Baseline | SMA"):
    gt = copy.deepcopy(gt_mesh); gt.paint_uniform_color([0.82, 0.82, 0.82])
    bl = copy.deepcopy(baseline_mesh) if baseline_mesh is not None else None
    if bl is not None:
        bl.paint_uniform_color([1.0, 0.6, 0.0])
    sm = copy.deepcopy(sma_mesh); sm.paint_uniform_color([0.2, 0.8, 0.3])

    ext = float(np.max(gt_mesh.get_axis_aligned_bounding_box().get_extent()))
    shift = 0.6 * ext

    if bl is None:
        gt.translate([-shift, 0, 0], relative=True)
        sm.translate([ shift, 0, 0], relative=True)
        bgt = gt.get_axis_aligned_bounding_box(); bgt.color = (0,1,0)
        bsm = sm.get_axis_aligned_bounding_box(); bsm.color = (1,0,0)

        combined_min = np.minimum(bgt.get_min_bound(), bsm.get_min_bound())
        combined_max = np.maximum(bgt.get_max_bound(), bsm.get_max_bound())
        center_xy = 0.5 * (combined_min[:2] + combined_max[:2])
        base_z = min(bgt.get_min_bound()[2], bsm.get_min_bound()[2]) - 1e-3
        grid_size = max(combined_max[0]-combined_min[0], combined_max[1]-combined_min[1]) * 0.75
        grid = make_grid(size=float(grid_size), step=ext*0.05, z=float(base_z))
        grid.translate([center_xy[0], center_xy[1], 0], relative=True)

        o3d.visualization.draw_geometries([grid, gt, bgt, sm, bsm],
                                          window_name=title, width=1400, height=800)
    else:
        gt.translate([-2*shift, 0, 0], relative=True)
        bl.translate([      0, 0, 0], relative=True)
        sm.translate([ 2*shift, 0, 0], relative=True)

        bgt = gt.get_axis_aligned_bounding_box(); bgt.color = (0,1,0)
        bbl = bl.get_axis_aligned_bounding_box(); bbl.color = (0,0,1)
        bsm = sm.get_axis_aligned_bounding_box(); bsm.color = (1,0,0)

        mins = np.minimum.reduce([bgt.get_min_bound(), bbl.get_min_bound(), bsm.get_min_bound()])
        maxs = np.maximum.reduce([bgt.get_max_bound(), bbl.get_max_bound(), bsm.get_max_bound()])
        center_xy = 0.5 * (mins[:2] + maxs[:2])
        base_z = min(bgt.get_min_bound()[2], bbl.get_min_bound()[2], bsm.get_min_bound()[2]) - 1e-3
        grid_size = max(maxs[0]-mins[0], maxs[1]-mins[1]) * 0.75
        grid = make_grid(size=float(grid_size), step=ext*0.05, z=float(base_z))
        grid.translate([center_xy[0], center_xy[1], 0], relative=True)

        o3d.visualization.draw_geometries([grid, gt, bgt, bl, bbl, sm, bsm],
                                          window_name=title, width=1600, height=850)

# ---- Camera viewpoint window helpers ----

def make_spheres(points, radius=0.008, color=(0.6, 0.6, 0.6)):
    geoms = []
    for p in points:
        s = o3d.geometry.TriangleMesh.create_sphere(radius)
        s.translate(p)
        s.paint_uniform_color(color)
        geoms.append(s)
    return geoms

def make_rays(points, target, every=6, color=(0.5, 0.5, 0.5)):
    pts, lines = [], []
    idx = list(range(0, len(points), max(1, every)))
    for i, k in enumerate(idx):
        pts.extend([points[k], target])
        lines.append([2 * i, 2 * i + 1])
    ls = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(np.asarray(pts)),
        lines=o3d.utility.Vector2iVector(np.asarray(lines)),
    )
    ls.paint_uniform_color(color)
    return ls

def make_frustum(intr, cam_T, near=0.05, far=0.25, color=(0.2, 0.8, 0.2)):
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
    O  = np.zeros((1,3))
    P = np.vstack([Cn, Cf, O])
    Pw = (cam_T @ np.hstack([P, np.ones((len(P), 1))]).T).T[:, :3]

    lines = []
    lines += [[0,1],[1,2],[2,3],[3,0]]  # near
    lines += [[4,5],[5,6],[6,7],[7,4]]  # far
    lines += [[0,4],[1,5],[2,6],[3,7]]
    lines += [[8,0],[8,1],[8,2],[8,3]]  # rays to near

    ls = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(Pw),
        lines=o3d.utility.Vector2iVector(np.asarray(lines, dtype=np.int32))
    )
    ls.paint_uniform_color(color)
    return ls

def camera_view_window(mesh, intr, cams, target_point, selected_idx, used_idx):
    aabb = mesh.get_axis_aligned_bounding_box(); aabb.color = (1.0, 0.2, 0.2)
    geoms = [mesh, aabb]

    # all candidates (light gray)
    geoms += make_spheres(cams, radius=0.006, color=(0.7, 0.7, 0.7))

    # selected by SMA (blue)
    if len(selected_idx):
        geoms += make_spheres(cams[selected_idx], radius=0.009, color=(0.2, 0.4, 1.0))

    # actually used (integrated) (green) + frustums and some rays
    used_pts = cams[used_idx] if len(used_idx) else np.zeros((0,3))
    geoms += make_spheres(used_pts, radius=0.011, color=(0.1, 0.8, 0.2))
    geoms.append(make_rays(used_pts, target_point, every=max(1, len(used_pts)//24), color=(0.2,0.7,0.2)))

    # frustums for used
    for p in used_pts:
        T = look_at_cv(p, target_point)
        dist = float(np.linalg.norm(target_point - p))
        geoms.append(make_frustum(intr, T, near=0.05, far=max(0.15, 0.3*dist), color=(0.1, 0.8, 0.2)))

    o3d.visualization.draw_geometries(
        geoms,
        window_name="Camera viewpoints — all (gray), selected (blue), used (green)",
        width=1400, height=850,
        lookat=target_point.tolist(), front=[0,-1,0], up=[0,0,1], zoom=0.7
    )

# -------------------- main --------------------

def main(cfg_path,
         frame_mode="object",
         samples=256,
         unit_autoscale=True,
         safety_scale=1.3,
         budget=16,
         sma_pop=40,
         sma_iters=30,
         w1=0.5, w2=0.3, w3=0.15, w4=0.2,
         voxel=0.004, trunc=0.012,
         out_mesh="experiments/results/sma_tsdf_mesh.ply",
         out_csv="experiments/results/sma_selected.csv",
         baseline_mesh_path=None,
         seed=0,
         use_3d_coverage=False,
         surf_samples=6000,
         div_weight=0.15,
         front_thresh=0.2):

    os.makedirs(os.path.dirname(out_mesh), exist_ok=True)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    cfg   = load_cfg(cfg_path)
    gt_mesh, center_w, extent_w = load_mesh_glb(cfg["mesh_path"], unit_autoscale=unit_autoscale)
    scene = build_clean_tensor_scene(gt_mesh)
    intr  = default_intr_from_cfg(cfg)
    print(f"Intrinsics: {intr}")

    hemi = cfg["hemisphere"]
    phi_min, phi_max = float(hemi["phi_deg_min"]), float(hemi["phi_deg_max"])
    z_margin, r0_m   = float(hemi["z_margin_m"]), float(hemi["r0_m"])

    if frame_mode.lower() == "object":
        center_o, x_o, y_o, z_o = pca_object_frame(gt_mesh)
        ext_obj, _, _ = extents_in_object_frame(gt_mesh, center_o, (x_o, y_o, z_o))
        cams = generate_objectframe_base_fib_cameras(
            gt_mesh, center_o, x_o, y_o, z_o, ext_obj,
            phi_min_deg=phi_min, phi_max_deg=phi_max,
            z_margin=z_margin, r0_m=r0_m,
            samples=samples, safety_scale=safety_scale
        )
        target_point = center_o
        print("Using OBJECT-frame hemisphere.")
    else:
        cams = generate_base_centered_fib_cameras(
            gt_mesh, center_w, extent_w,
            phi_min_deg=phi_min, phi_max_deg=phi_max,
            z_margin=z_margin, r0_m=r0_m,
            samples=samples, safety_scale=safety_scale
        )
        target_point = center_w
        print("Using WORLD-frame hemisphere.")

    print(f"Candidates: {len(cams)} | Budget: {budget} | SMA pop:{sma_pop} iters:{sma_iters}")
    weights = (float(w1), float(w2), float(w3), float(w4))
    depth_trunc = max(1.0, 1.5 * trunc)

    # surface sampling if needed
    P_surf = N_surf = W_surf = None
    if use_3d_coverage:
        print(f"Sampling {surf_samples} surface points for 3D coverage scoring…")
        P_surf, N_surf, W_surf = sample_surface_points(gt_mesh, n=int(surf_samples), seed=seed)

    # SMA selection
    sel_idx = sma_optimize(
        scene, cams, target_point, intr,
        depth_trunc=depth_trunc, weights=weights,
        budget=budget, pop=sma_pop, iters=sma_iters, seed=seed, verbose=True,
        use_3d=use_3d_coverage, P_surf=P_surf, N_surf=N_surf, W_surf=W_surf,
        div_weight=div_weight, front_thresh=front_thresh
    )
    print("Selected indices:", sel_idx)

    # Save chosen cameras
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idx","x","y","z"])
        for i in sel_idx:
            p = cams[i]
            w.writerow([i, p[0], p[1], p[2]])
    print(f"Saved selected cameras: {out_csv}")

    # Fuse only selected
    sma_mesh, used = fuse_selected(scene, intr, cams, target_point, sel_idx, voxel=voxel, trunc=trunc)
    o3d.io.write_triangle_mesh(out_mesh, sma_mesh)
    print(f"SMA mesh saved: {out_mesh} | used {len(used)}/{len(sel_idx)} selected")

    # (Optional) load baseline mesh from fuse_tsdf.py
    baseline_mesh = None
    if baseline_mesh_path and os.path.isfile(baseline_mesh_path):
        baseline_mesh = o3d.io.read_triangle_mesh(baseline_mesh_path)
        if baseline_mesh.is_empty():
            print(f"Warning: failed to read baseline mesh at {baseline_mesh_path} (empty).")
            baseline_mesh = None
        else:
            baseline_mesh.compute_vertex_normals()
            print(f"Loaded baseline mesh: {baseline_mesh_path}")

    # Tri- (or bi-) view viz
    tri_view(
        gt_mesh, baseline_mesh, sma_mesh,
        title="GT (left) | Baseline (center) | SMA (right)" if baseline_mesh is not None
              else "GT (left) | SMA (right)"
    )

    # Second viewer: camera viewpoints (all, selected, used) + frustums/rays
    sel_idx_arr  = np.array(sel_idx, dtype=int)
    used_idx_arr = np.array(used,    dtype=int) if len(used) else np.zeros((0,), dtype=int)
    camera_view_window(gt_mesh, intr, cams, target_point, sel_idx_arr, used_idx_arr)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True, help="experiments/configs/can_hemisphere.yaml")
    ap.add_argument("--frame", choices=["world","object"], default="object")
    ap.add_argument("--samples", type=int, default=256)
    ap.add_argument("--no-unit-autoscale", action="store_true")
    ap.add_argument("--safety-scale", type=float, default=1.3)

    ap.add_argument("--budget", type=int, default=16)
    ap.add_argument("--sma-pop", type=int, default=40)
    ap.add_argument("--sma-iters", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--w1", type=float, default=0.5)
    ap.add_argument("--w2", type=float, default=0.3)
    ap.add_argument("--w3", type=float, default=0.15)
    ap.add_argument("--w4", type=float, default=0.2)

    ap.add_argument("--voxel", type=float, default=0.004)
    ap.add_argument("--trunc", type=float, default=0.012)
    ap.add_argument("--out", type=str, default="experiments/results/sma_tsdf_mesh.ply")
    ap.add_argument("--out-csv", type=str, default="experiments/results/sma_selected.csv")
    ap.add_argument("--baseline-mesh", type=str, default="", help="Path to baseline mesh from fuse_tsdf.py")

    # NEW flags (added earlier; unchanged)
    ap.add_argument("--use-3d-coverage", action="store_true",
                    help="Score views by newly revealed surface area (front-facing & visible) instead of 2D depth mask.")
    ap.add_argument("--surf-samples", type=int, default=6000,
                    help="Number of surface samples for 3D coverage.")
    ap.add_argument("--div-weight", type=float, default=0.15,
                    help="Weight for angular diversity bonus in 3D scoring.")
    ap.add_argument("--front-thresh", type=float, default=0.2,
                    help="Front-facing threshold (dot(normal, view)>thr) for 3D coverage.")

    args = ap.parse_args()

    main(args.cfg,
         frame_mode=args.frame,
         samples=args.samples,
         unit_autoscale=not args.no_unit_autoscale,
         safety_scale=args.safety_scale,
         budget=args.budget,
         sma_pop=args.sma_pop,
         sma_iters=args.sma_iters,
         w1=args.w1, w2=args.w2, w3=args.w3, w4=args.w4,
         voxel=args.voxel, trunc=args.trunc,
         out_mesh=args.out, out_csv=args.out_csv,
         baseline_mesh_path=args.baseline_mesh,
         seed=args.seed,
         use_3d_coverage=args.use_3d_coverage,
         surf_samples=args.surf_samples,
         div_weight=args.div_weight,
         front_thresh=args.front_thresh)
