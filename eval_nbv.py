# eval_nbv.py — select (SMA/Greedy/Random) and/or evaluate 3D surface coverage
# - If --method none  (default): evaluate an existing cams CSV (idx,x,y,z)
# - If --method {sma,greedy,random}: generate candidates from cfg, select, save selection,
#   compute coverage curve, and append per-step logs for plotting across methods.
#
# Examples:
#   # 1) Evaluate already-saved cameras
#   python eval_nbv.py --cfg experiments/configs/can_hemisphere.yaml \
#                      --cams experiments/results/sma_selected.csv \
#                      --plot experiments/results/sma_coverage.png \
#                      --out-csv experiments/results/sma_coverage.csv
#
#   # 2) Run SMA selection with 3D coverage, log per-step curve, and save selected CSV
#   python eval_nbv.py --cfg experiments/configs/can_hemisphere.yaml \
#                      --method sma --use-3d-coverage --budget 24 --seed 0 \
#                      --out-selected experiments/results/sma_selected.csv \
#                      --covlog-csv experiments/results/coverage_by_method.csv \
#                      --plot experiments/results/sma_coverage.png
#
#   # 3) Greedy / Random baselines (2D or 3D coverage by flag)
#   python eval_nbv.py --cfg ... --method greedy --budget 24 --seed 1
#   python eval_nbv.py --cfg ... --method random --use-3d-coverage --budget 24 --seed 2
#
# Requires your utils.py (look_at_cv, build_clean_tensor_scene, hemisphere generators, etc).

import argparse, os, csv, math
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

from utils import (
    load_cfg, load_mesh_glb, default_intr_from_cfg, look_at_cv,
    build_clean_tensor_scene,
    generate_base_centered_fib_cameras,
    pca_object_frame, extents_in_object_frame, generate_objectframe_base_fib_cameras,
)

# --------------------------- small utils ---------------------------

def _normalize(v, eps=1e-9):
    v = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / (n + eps)

def ensure_dir_for(path):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)

# -------------------- surface sampling + visibility --------------------

def sample_surface_points(mesh, n=20000, seed=0):
    """
    Area-weighted sampling of surface points with per-sample normals and weights.
    Returns:
      P : (n, 3) sampled points
      N : (n, 3) unit normals at those points
      W : (n,)   per-sample area weights (sum≈surface area)
    """
    rng = np.random.default_rng(seed)
    V = np.asarray(mesh.vertices, dtype=np.float32)
    F = np.asarray(mesh.triangles, dtype=np.int32)
    if V.size == 0 or F.size == 0:
        raise ValueError("Empty mesh in sample_surface_points.")

    a = V[F[:, 0]]; b = V[F[:, 1]]; c = V[F[:, 2]]
    tri_norm = np.cross(b - a, c - a)
    tri_area = 0.5 * np.linalg.norm(tri_norm, axis=1)
    total_area = float(tri_area.sum())
    if total_area <= 0:
        raise ValueError("Mesh has zero total area; check geometry.")

    prob = tri_area / total_area
    prob = prob / prob.sum()  # exact renorm

    face_idx = rng.choice(len(F), size=n, p=prob)
    u = np.sqrt(rng.random(n)).astype(np.float32)
    v = rng.random(n).astype(np.float32)

    Aa = V[F[face_idx, 0]]
    Bb = V[F[face_idx, 1]]
    Cc = V[F[face_idx, 2]]

    P = (1.0 - u)[:, None] * Aa + (u * (1.0 - v))[:, None] * Bb + (u * v)[:, None] * Cc

    # per-sample face normal -> unit
    FN = np.cross(Bb - Aa, Cc - Aa)
    N  = _normalize(FN)

    # unbiased per-sample weight
    W = tri_area[face_idx] / (prob[face_idx] * n)
    return P.astype(np.float32), N.astype(np.float32), W.astype(np.float32)

def visible_from_camera(scene, cam_T, P, N, front_thresh=0.2, rel_tol=1e-3, abs_tol=1e-4):
    """
    Cast rays FROM the camera origin TO the points. A sample is visible if:
      (i)  front-facing: -dot(N, dir_cam_to_point) > front_thresh
      (ii) first hit distance matches point distance within tolerance.
    """
    C   = cam_T[:3, 3]
    vec = P - C[None, :]
    dist = np.linalg.norm(vec, axis=1)
    dirw = _normalize(vec)

    ff = (-np.sum(N * dirw, axis=1) > float(front_thresh))

    origins = C[None, :] + dirw * 1e-6
    rays = np.concatenate([origins.astype(np.float32), dirw.astype(np.float32)], axis=1)
    out = scene.cast_rays(o3d.core.Tensor(rays))
    t_hit = out["t_hit"].numpy()

    ok  = np.isfinite(t_hit)
    tol = np.maximum(abs_tol, rel_tol * dist)
    hit_match = np.abs(t_hit - dist) <= tol
    return ff & ok & hit_match

# -------------------- 2D depth/quality terms (for SMA/Greedy scoring) --------------------

def render_depth_mask(scene, cam_pos, target_point, intr, depth_trunc=2.0):
    cam_T = look_at_cv(cam_pos, target_point)
    # RaycastingScene returns range; convert to z with cos factor not needed here for mask
    # We just need non-zero hit: cast camera->point rays via a coarse screen proxy.
    # For simplicity, reuse 3D visibility on surface points for coverage terms; here mask is only for redundancy penalty.
    # We'll approximate a dense mask via projecting the surface samples if needed; to keep this light, we use 3D only.
    # Return a tiny placeholder to avoid heavy per-view image rendering in eval script.
    return None  # Not used for coverage here.

def detection_bonus_center_hit(scene, cam_T, intr) -> float:
    # Lightweight center-line check using raycast at optical axis
    C = cam_T[:3, 3]
    z_axis = cam_T[:3, 2]  # +Z forward
    origins = C[None, :]
    dirs    = z_axis[None, :]
    rays = np.concatenate([origins.astype(np.float32), dirs.astype(np.float32)], axis=1)
    out  = scene.cast_rays(o3d.core.Tensor(rays))
    t_hit = out["t_hit"].numpy()[0]
    return 1.0 if np.isfinite(t_hit) else 0.0

# -------------------- Scoring (2D-lite + 3D coverage) --------------------

def view_quality_depth_proxy(scene, cam_T, intr):
    # Simple proxy: distance along optical axis should be finite and not extremely close/far
    C = cam_T[:3, 3]
    z_axis = cam_T[:3, 2]
    origins = C[None, :]
    dirs    = z_axis[None, :]
    out = scene.cast_rays(o3d.core.Tensor(np.concatenate([origins, dirs], axis=1).astype(np.float32)))
    t = out["t_hit"].numpy()[0]
    if not np.isfinite(t):
        return 0.0
    # Favor mid-range distances
    score = math.exp(-((t - 0.6) ** 2) / 0.25)  # tune if needed
    return float(np.clip(score, 0.0, 1.0))

def compute_view_score_3d(scene, cam_pos, target_point, intr,
                          P_surf, N_surf, W_surf, covered_pts,
                          selected_dirs, weights,
                          div_weight=0.15, front_thresh=0.2):
    cam_T = look_at_cv(cam_pos, target_point)

    vis = visible_from_camera(scene, cam_T, P_surf, N_surf,
                              front_thresh=front_thresh, rel_tol=1e-3, abs_tol=1e-4)
    newly = vis & (~covered_pts)
    cov_area = float(W_surf[newly].sum()) / float(W_surf.sum() + 1e-12)

    qual = view_quality_depth_proxy(scene, cam_T, intr)
    det  = detection_bonus_center_hit(scene, cam_T, intr)

    # angular diversity
    dir_w = target_point - cam_pos
    dir_w = dir_w / (np.linalg.norm(dir_w) + 1e-9)
    if selected_dirs:
        sims = [abs(np.dot(dir_w, d)) for d in selected_dirs]
        diversity = 1.0 - max(sims)
    else:
        diversity = 1.0

    w1, w2, w3, w4 = weights
    # Here w4 is unused (no 2D overlap term in this eval tool)
    score = w1*cov_area + w2*qual + w3*det + div_weight*diversity
    return score, vis, dir_w, cov_area

# -------------------- Selection methods (return indices + coverage curve) --------------------

def sma_optimize(scene, cams, target_point, intr,
                 weights, budget=16, pop=40, iters=30, seed=0, verbose=True,
                 P_surf=None, N_surf=None, W_surf=None, div_weight=0.15, front_thresh=0.2):
    assert P_surf is not None
    rng = np.random.default_rng(seed)
    N = len(cams)

    covered_pts = np.zeros(len(P_surf), dtype=bool)
    selected_dirs = []
    selected = []
    curve = []

    remaining = set(range(N))

    for step in range(budget):
        if not remaining:
            break
        if verbose:
            print(f"\n=== SMA step {step+1}/{budget} ===")

        init_size = min(pop, len(remaining))
        pop_idx = rng.choice(list(remaining), size=init_size, replace=False)
        fitness = np.zeros(init_size, dtype=np.float32)
        cache = {}

        # init evaluate
        for i, idx in enumerate(pop_idx):
            s, vis, dvec, _ = compute_view_score_3d(
                scene, cams[idx], target_point, intr,
                P_surf, N_surf, W_surf, covered_pts, selected_dirs,
                weights, div_weight=div_weight, front_thresh=front_thresh
            )
            fitness[i] = s
            cache[idx] = (s, vis, dvec)

        gbest_idx = int(pop_idx[np.argmax(fitness)])
        gbest_fit = float(fitness.max())

        # SMA-like updates
        for _ in range(iters):
            order = np.argsort(-fitness)
            for rank_pos, p_i in enumerate(order):
                idx = int(pop_idx[p_i])
                step_span = max(1, int(0.05 * N * (1.0 - rank_pos / max(1, len(order)-1))))
                if rng.random() < 0.65:
                    direction = np.sign(gbest_idx - idx)
                    new_idx = int(np.clip(idx + direction * rng.integers(1, step_span+1), 0, N-1))
                else:
                    new_idx = int(np.clip(idx + rng.integers(-step_span, step_span+1), 0, N-1))

                if new_idx not in remaining:
                    # snap to available
                    for _try in range(2):
                        trial = int(rng.choice(list(remaining)))
                        if trial != idx:
                            new_idx = trial
                            break

                if new_idx in cache:
                    s = cache[new_idx][0]
                else:
                    s, vis, dvec, _ = compute_view_score_3d(
                        scene, cams[new_idx], target_point, intr,
                        P_surf, N_surf, W_surf, covered_pts, selected_dirs,
                        weights, div_weight=div_weight, front_thresh=front_thresh
                    )
                    cache[new_idx] = (s, vis, dvec)

                if s >= fitness[p_i]:
                    pop_idx[p_i] = new_idx
                    fitness[p_i] = s
                    if s > gbest_fit:
                        gbest_fit = s
                        gbest_idx = new_idx

        # commit
        chosen = int(gbest_idx)
        s, vis, dvec = cache[chosen]
        selected.append(chosen)
        remaining.discard(chosen)
        covered_pts |= vis
        selected_dirs.append(dvec)
        covered_frac = float(W_surf[covered_pts].sum() / (W_surf.sum() + 1e-12))
        curve.append(covered_frac)
        if verbose:
            print(f"Pick {chosen}  score={s:.4f}  covered≈{covered_frac*100:.2f}%")

    return selected, curve

def greedy_optimize(scene, cams, target_point, intr,
                    weights, budget=16, seed=0, verbose=True,
                    P_surf=None, N_surf=None, W_surf=None, div_weight=0.15, front_thresh=0.2):
    assert P_surf is not None
    N = len(cams)
    covered_pts = np.zeros(len(P_surf), dtype=bool)
    selected_dirs = []
    selected = []
    curve = []
    remaining = set(range(N))

    for step in range(budget):
        best_idx, best_s, best_vis, best_dir = -1, -1e9, None, None
        for idx in list(remaining):
            s, vis, dvec, _ = compute_view_score_3d(
                scene, cams[idx], target_point, intr,
                P_surf, N_surf, W_surf, covered_pts, selected_dirs,
                weights, div_weight=div_weight, front_thresh=front_thresh
            )
            if s > best_s:
                best_idx, best_s, best_vis, best_dir = idx, s, vis, dvec

        selected.append(best_idx)
        remaining.discard(best_idx)
        covered_pts |= best_vis
        selected_dirs.append(best_dir)
        covered_frac = float(W_surf[covered_pts].sum() / (W_surf.sum() + 1e-12))
        curve.append(covered_frac)
        if verbose:
            print(f"[Greedy {step+1}/{budget}] pick {best_idx}  score={best_s:.4f}  covered≈{covered_frac*100:.2f}%")

    return selected, curve

def random_optimize(cams, budget=16, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(cams))
    rng.shuffle(idx)
    return idx[:budget].tolist()

# -------------------- I/O helpers --------------------

def read_cameras_csv(path):
    """Expect header: idx,x,y,z (as written by sma_nbv.py)."""
    cams = []
    with open(path, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            cols = {k.lower(): v for k, v in row.items()}
            if not {"x","y","z"}.issubset(cols.keys()):
                raise ValueError("Camera CSV must have columns idx,x,y,z")
            cams.append(np.array([float(cols["x"]), float(cols["y"]), float(cols["z"])],
                                 dtype=np.float32))
    if not cams:
        raise ValueError("No cameras found in CSV.")
    return np.vstack(cams)

def write_selected_csv(path, cams, sel_idx):
    ensure_dir_for(path)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idx", "x", "y", "z"])
        for i in sel_idx:
            p = cams[i]
            w.writerow([int(i), float(p[0]), float(p[1]), float(p[2])])
    print(f"Saved selected cameras: {path}")

def append_coverage_log(path, method, curve, seed, budget, use_3d, obj_name="unknown"):
    ensure_dir_for(path)
    new_file = not os.path.isfile(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["method", "step", "coverage_fraction", "seed", "budget", "use_3d", "object"])
        for step, cov in enumerate(curve, start=1):
            w.writerow([method, step, float(cov), int(seed), int(budget), int(use_3d), obj_name])
    print(f"Appended coverage to: {path}")

# -------------------- frustum viz --------------------

def make_camera_frustum_cv(intr, cam_T, near=0.05, far=0.25, color=(0.1, 0.8, 0.1)):
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
    origin = np.zeros((1, 3))
    P = np.vstack([Cn, Cf, origin])
    Pw = (cam_T @ np.hstack([P, np.ones((len(P), 1))]).T).T[:, :3]

    lines = []
    lines += [[0,1],[1,2],[2,3],[3,0]]
    lines += [[4,5],[5,6],[6,7],[7,4]]
    lines += [[0,4],[1,5],[2,6],[3,7]]
    lines += [[8,0],[8,1],[8,2],[8,3]]

    ls = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(Pw),
        lines=o3d.utility.Vector2iVector(np.array(lines, dtype=np.int32))
    )
    ls.paint_uniform_color(color)
    return ls

# --------------------------- main ---------------------------

def main(cfg_path,
         cams_csv=None,
         frame_mode="object",
         gt_mesh_path=None,
         unit_autoscale=True,
         surf_samples=20000,
         front_thresh=0.2,
         out_csv="experiments/results/coverage_curve.csv",
         plot_png="experiments/results/coverage_curve.png",
         viz=False,
         viz_frustums=12,
         seed=0,
         # selection flags
         method="none",                    # "none", "sma", "greedy", "random"
         budget=16,
         sma_pop=40,
         sma_iters=30,
         w1=0.5, w2=0.3, w3=0.15, w4=0.0,  # w4 unused here
         use_3d_coverage=True,
         div_weight=0.15,
         out_selected="experiments/results/selected.csv",
         covlog_csv="experiments/results/coverage_by_method.csv"):

    # cfg / intrinsics
    cfg  = load_cfg(cfg_path)
    intr = default_intr_from_cfg(cfg)

    # mesh (from cfg unless overridden)
    mesh_path = gt_mesh_path or cfg["mesh_path"]
    gt_mesh, center_w, extent_w = load_mesh_glb(mesh_path, unit_autoscale=unit_autoscale)
    scene = build_clean_tensor_scene(gt_mesh)

    # target (must match how candidates were produced)
    if frame_mode.lower() == "object":
        # Use PCA center as in other scripts
        V = np.asarray(gt_mesh.vertices)
        target_point = V.mean(axis=0).astype(np.float32)
        center_o, x_o, y_o, z_o = pca_object_frame(gt_mesh)
        ext_obj, _, _ = extents_in_object_frame(gt_mesh, center_o, (x_o, y_o, z_o))
        cams_all = generate_objectframe_base_fib_cameras(
            gt_mesh, center_o, x_o, y_o, z_o, ext_obj,
            phi_min_deg=float(cfg["hemisphere"]["phi_deg_min"]),
            phi_max_deg=float(cfg["hemisphere"]["phi_deg_max"]),
            z_margin=float(cfg["hemisphere"]["z_margin_m"]),
            r0_m=float(cfg["hemisphere"]["r0_m"]),
            samples=int(cfg.get("hemisphere", {}).get("samples", 256)),
            safety_scale=float(cfg.get("hemisphere", {}).get("safety_scale", 1.3))
        )
    else:
        target_point = center_w.astype(np.float32)
        cams_all = generate_base_centered_fib_cameras(
            gt_mesh, center_w, extent_w,
            phi_min_deg=float(cfg["hemisphere"]["phi_deg_min"]),
            phi_max_deg=float(cfg["hemisphere"]["phi_deg_max"]),
            z_margin=float(cfg["hemisphere"]["z_margin_m"]),
            r0_m=float(cfg["hemisphere"]["r0_m"]),
            samples=int(cfg.get("hemisphere", {}).get("samples", 256)),
            safety_scale=float(cfg.get("hemisphere", {}).get("safety_scale", 1.3))
        )

    # Surface samples (always used for coverage evaluation; also for 3D-aware selection)
    print(f"Sampling {surf_samples} surface points for coverage…")
    P, N, W = sample_surface_points(gt_mesh, n=int(surf_samples), seed=seed)
    Wn = W / (np.sum(W) + 1e-12)

    # If we are only evaluating a given CSV of cameras:
    if method == "none":
        if cams_csv is None:
            raise ValueError("--cams is required when --method none")
        cam_pos = read_cameras_csv(cams_csv)

        covered = np.zeros(len(P), dtype=bool)
        curve = []
        for k, p in enumerate(cam_pos, start=1):
            cam_T = look_at_cv(p, target_point)
            vis = visible_from_camera(scene, cam_T, P, N, front_thresh=front_thresh,
                                      rel_tol=1e-3, abs_tol=1e-4)
            covered |= vis
            cov_frac = float(np.sum(Wn[covered]))
            curve.append((k, cov_frac))
            print(f"[{k:03d}/{len(cam_pos)}] cumulative coverage = {cov_frac*100:.2f}%")

        # save curve csv + plot
        ensure_dir_for(out_csv); ensure_dir_for(plot_png)
        with open(out_csv, "w", newline="") as f:
            wcsv = csv.writer(f)
            wcsv.writerow(["views", "coverage_fraction"])
            wcsv.writerows(curve)
        print(f"Saved coverage curve: {out_csv}")

        if plot_png:
            xs = [c[0] for c in curve]; ys = [100.0*c[1] for c in curve]
            plt.figure(figsize=(6,4), dpi=140)
            plt.plot(xs, ys, marker="o", linewidth=2)
            plt.xlabel("# views"); plt.ylabel("Surface coverage (%)")
            plt.title("Coverage vs. number of views")
            plt.grid(True, alpha=0.3)
            plt.tight_layout(); plt.savefig(plot_png); plt.close()
            print(f"Saved plot: {plot_png}")
        return

    # Else: select cameras by method, then log/evaluate
    weights = (float(w1), float(w2), float(w3), float(w4))
    print(f"Candidates: {len(cams_all)} | Method: {method} | Budget: {budget}")

    if method == "sma":
        sel_idx, curve = sma_optimize(
            scene, cams_all, target_point, intr,
            weights, budget=budget, pop=sma_pop, iters=sma_iters, seed=seed, verbose=True,
            P_surf=P, N_surf=N, W_surf=W, div_weight=div_weight, front_thresh=front_thresh
        )
    elif method == "greedy":
        sel_idx, curve = greedy_optimize(
            scene, cams_all, target_point, intr,
            weights, budget=budget, seed=seed, verbose=True,
            P_surf=P, N_surf=N, W_surf=W, div_weight=div_weight, front_thresh=front_thresh
        )
    else:  # random
        sel_idx = random_optimize(cams_all, budget=budget, seed=seed)
        # build curve for random using visibility
        covered = np.zeros(len(P), dtype=bool)
        curve = []
        for k, i in enumerate(sel_idx, start=1):
            cam_T = look_at_cv(cams_all[i], target_point)
            vis = visible_from_camera(scene, cam_T, P, N, front_thresh=front_thresh,
                                      rel_tol=1e-3, abs_tol=1e-4)
            covered |= vis
            curve.append(float(W[covered].sum() / (W.sum() + 1e-12)))
            print(f"[Random {k:02d}/{budget}] covered≈{curve[-1]*100:.2f}%")

    print("Selected indices:", sel_idx)

    # Save selected CSV
    write_selected_csv(out_selected, cams_all, sel_idx)

    # Save a per-run coverage curve CSV (views vs. fraction)
    ensure_dir_for(out_csv); ensure_dir_for(plot_png)
    with open(out_csv, "w", newline="") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["views", "coverage_fraction"])
        for k, cov in enumerate(curve, start=1):
            wcsv.writerow([k, float(cov)])
    print(f"Saved coverage curve: {out_csv}")

    # Append to shared coverage_by_method.csv
    obj_name = os.path.splitext(os.path.basename(cfg["mesh_path"]))[0]
    append_coverage_log(covlog_csv, method, curve, seed, budget, True, obj_name=obj_name)

    # Plot
    if plot_png:
        xs = list(range(1, len(curve)+1))
        ys = [100.0*c for c in curve]
        plt.figure(figsize=(6,4), dpi=140)
        plt.plot(xs, ys, marker="o", linewidth=2)
        plt.xlabel("# views"); plt.ylabel("Surface coverage (%)")
        plt.title(f"{method.upper()} coverage vs. views")
        plt.grid(True, alpha=0.3)
        plt.tight_layout(); plt.savefig(plot_png); plt.close()
        print(f"Saved plot: {plot_png}")

    # Optional quick viz of first few frustums
    if viz:
        k_show = min(viz_frustums, len(sel_idx))
        geoms = [gt_mesh]
        covered_k = np.zeros(len(P), dtype=bool)
        for j in range(k_show):
            cam_T = look_at_cv(cams_all[sel_idx[j]], target_point)
            vis = visible_from_camera(scene, cam_T, P, N, front_thresh=front_thresh,
                                      rel_tol=1e-3, abs_tol=1e-4)
            covered_k |= vis
            geoms.append(make_camera_frustum_cv(intr, cam_T, near=0.05, far=0.3, color=(0.1, 0.8, 0.1)))

        pts = o3d.geometry.PointCloud()
        pts.points = o3d.utility.Vector3dVector(P)
        colors = np.tile(np.array([[0.7, 0.7, 0.7]]), (len(P), 1))
        colors[covered_k] = np.array([0.1, 0.8, 0.2])
        pts.colors = o3d.utility.Vector3dVector(colors)
        geoms.append(pts)

        aabb = gt_mesh.get_axis_aligned_bounding_box(); aabb.color = (1.0, 0.2, 0.2)
        geoms.append(aabb)

        o3d.visualization.draw_geometries(
            geoms,
            window_name=f"{method.upper()} (first {k_show} views) — green samples = covered",
            width=1280, height=800,
            lookat=target_point.tolist(), front=[0,-1,0], up=[0,0,1], zoom=0.7
        )

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True, help="Path to config (intrinsics & mesh_path)")
    ap.add_argument("--cams", default=None, help="CSV with idx,x,y,z (used only when --method none)")
    ap.add_argument("--frame", choices=["world", "object"], default="object",
                    help="Hemisphere frame (must match intended look-at target)")
    ap.add_argument("--gt-mesh", default=None, help="Override GT mesh path (else cfg['mesh_path'])")
    ap.add_argument("--no-unit-autoscale", action="store_true")
    ap.add_argument("--surf-samples", type=int, default=20000)
    ap.add_argument("--front-thresh", type=float, default=0.2,
                    help="dot(-normal, dir_cam_to_point) threshold for front-facing")
    ap.add_argument("--out-csv", default="experiments/results/coverage_curve.csv")
    ap.add_argument("--plot", dest="plot_png", default="experiments/results/coverage_curve.png")
    ap.add_argument("--viz", action="store_true", help="Open Open3D window with first N frustums")
    ap.add_argument("--viz-frustums", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)

    # Selection flags (active when --method != none)
    ap.add_argument("--method", choices=["none", "sma", "greedy", "random"], default="none",
                    help="Selection strategy; 'none' means just evaluate --cams CSV")
    ap.add_argument("--budget", type=int, default=16)
    ap.add_argument("--sma-pop", type=int, default=40)
    ap.add_argument("--sma-iters", type=int, default=30)
    ap.add_argument("--w1", type=float, default=0.5)
    ap.add_argument("--w2", type=float, default=0.3)
    ap.add_argument("--w3", type=float, default=0.15)
    ap.add_argument("--w4", type=float, default=0.0, help="(unused in this eval tool)")
    ap.add_argument("--use-3d-coverage", action="store_true", default=True,
                    help="(kept for API parity; this tool always evaluates 3D coverage)")
    ap.add_argument("--div-weight", type=float, default=0.15,
                    help="Weight for angular diversity in 3D scoring")
    ap.add_argument("--out-selected", default="experiments/results/selected.csv",
                    help="Where to save selected cameras when method != none")
    ap.add_argument("--covlog-csv", default="experiments/results/coverage_by_method.csv",
                    help="Append per-step coverage here (method,step,coverage_fraction,seed,budget,use_3d,object)")

    args = ap.parse_args()

    main(args.cfg,
         cams_csv=args.cams,
         frame_mode=args.frame,
         gt_mesh_path=args.gt_mesh,
         unit_autoscale=not args.no_unit_autoscale,
         surf_samples=args.surf_samples,
         front_thresh=args.front_thresh,
         out_csv=args.out_csv,
         plot_png=args.plot_png,
         viz=args.viz,
         viz_frustums=args.viz_frustums,
         seed=args.seed,
         method=args.method,
         budget=args.budget,
         sma_pop=args.sma_pop,
         sma_iters=args.sma_iters,
         w1=args.w1, w2=args.w2, w3=args.w3, w4=args.w4,
         use_3d_coverage=args.use_3d_coverage,
         div_weight=args.div_weight,
         out_selected=args.out_selected,
         covlog_csv=args.covlog_csv)
