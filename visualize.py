# visualize.py — uses utils.py (world-frame vs object-frame hemispheres)
import argparse
import numpy as np
import open3d as o3d
from utils import (
    load_cfg, load_mesh_glb,
    generate_base_centered_fib_cameras,
    pca_object_frame, extents_in_object_frame, generate_objectframe_base_fib_cameras,
    validate_look_at
)

# ---------- small viz helpers ----------
def make_camera_markers(positions, color=(0.1, 0.6, 1.0), radius=0.006):
    geoms = []
    for p in positions:
        s = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        s.translate(p)
        s.paint_uniform_color(color)
        geoms.append(s)
    return geoms

def make_rays_to_point(positions, point, stride=6, color=(0.2, 0.8, 0.2)):
    pts, lines = [], []
    k = max(1, stride)
    for i, p in enumerate(positions[::k]):
        pts += [p, point]
        lines.append([2*i, 2*i+1])
    ls = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(np.array(pts)),
        lines=o3d.utility.Vector2iVector(np.array(lines))
    )
    ls.colors = o3d.utility.Vector3dVector(np.tile(color, (len(lines), 1)))
    return ls

def make_axes_at(center, R=None, size=0.05):
    """Coordinate frame at center; if R is given, rotate it."""
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
    T = np.eye(4); T[:3, 3] = center
    if R is not None:
        T[:3, :3] = R
    frame.transform(T)
    return frame

# ---------- main ----------
def main(cfg_path, frame_mode="object", samples=192, unit_autoscale=True, safety_scale=1.3):
    cfg = load_cfg(cfg_path)

    # Load mesh (optionally fix mm/cm -> m in-memory only)
    mesh, center_w, extent_w = load_mesh_glb(cfg["mesh_path"], unit_autoscale=unit_autoscale)
    aabb = mesh.get_axis_aligned_bounding_box(); aabb.color = (1.0, 0.3, 0.1)

    hemi = cfg["hemisphere"]
    phi_min  = float(hemi["phi_deg_min"])
    phi_max  = float(hemi["phi_deg_max"])
    z_margin = float(hemi["z_margin_m"])
    r0_m     = float(hemi["r0_m"])

    geoms = [mesh, aabb]

    if frame_mode.lower() == "object":
        # PCA object frame
        center_o, x_o, y_o, z_o = pca_object_frame(mesh)
        ext_obj, _, _ = extents_in_object_frame(mesh, center_o, (x_o, y_o, z_o))
        cams = generate_objectframe_base_fib_cameras(
            mesh, center_o, x_o, y_o, z_o, ext_obj,
            phi_min_deg=phi_min, phi_max_deg=phi_max,
            z_margin=z_margin, r0_m=r0_m,
            samples=samples, safety_scale=safety_scale
        )
        validate_look_at(cams, center_o)

        # Frames for reference
        R_obj = np.stack([x_o, y_o, z_o], axis=1)
        geoms.append(make_axes_at(center_o, R=R_obj, size=0.06))  # object axes
        geoms.append(make_axes_at(center_w, R=np.eye(3), size=0.05))  # world axes at AABB center
        # Rays target the object-frame center
        rays = make_rays_to_point(cams, center_o, stride=6)
        geoms += make_camera_markers(cams, radius=0.006)
        geoms.append(rays)

        print("\n=== Object-frame hemisphere ===")
        print(f"samples: {len(cams)}, phi:[{phi_min},{phi_max}] deg, safety_scale:{safety_scale}")
        print("object extents (x_o,y_o,z_o):", ext_obj)

        front = (-y_o).tolist()
        up    = z_o.tolist()
        look  = center_o.tolist()

    else:
        # World-frame hemisphere
        cams = generate_base_centered_fib_cameras(
            mesh, center_w, extent_w,
            phi_min_deg=phi_min, phi_max_deg=phi_max,
            z_margin=z_margin, r0_m=r0_m,
            samples=samples, safety_scale=safety_scale
        )
        validate_look_at(cams, center_w)

        geoms.append(make_axes_at(center_w, R=np.eye(3), size=0.06))  # world axes
        rays = make_rays_to_point(cams, center_w, stride=6)
        geoms += make_camera_markers(cams, radius=0.006)
        geoms.append(rays)

        print("\n=== World-frame hemisphere ===")
        print(f"samples: {len(cams)}, phi:[{phi_min},{phi_max}] deg, safety_scale:{safety_scale}")
        print("world extents (x,y,z):", extent_w)

        front = [0, -1, 0]
        up    = [0, 0, 1]
        look  = center_w.tolist()

    # Basic stats
    min_z = float(np.min(cams[:, 2]))
    base_z = float(aabb.get_min_bound()[2])
    print(f"min cam z: {min_z:.3f}  (base_z + margin ≈ {base_z + max(0.01, z_margin):.3f})")

    # Show
    o3d.visualization.draw_geometries(
        geoms,
        window_name=f"Fibonacci Hemisphere ({frame_mode}-frame)",
        width=1280, height=800,
        lookat=look, front=front, up=up, zoom=0.7
    )

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True, help="experiments/configs/can_hemisphere.yaml")
    ap.add_argument("--frame", choices=["world","object"], default="object")
    ap.add_argument("--samples", type=int, default=192)
    ap.add_argument("--no-unit-autoscale", action="store_true")
    ap.add_argument("--safety-scale", type=float, default=1.3)
    args = ap.parse_args()
    main(
        args.cfg,
        frame_mode=args.frame,
        samples=args.samples,
        unit_autoscale=not args.no_unit_autoscale,
        safety_scale=args.safety_scale
    )
