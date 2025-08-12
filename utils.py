# utils.py — core utilities for SMA-NBV (Windows-friendly)
# CV/+Z camera convention (OpenCV/Open3D RGB-D) for TSDF fusion.
# Adds intrinsics-aware camera radius selection to avoid empty/zero-depth views.

from __future__ import annotations
from typing import Dict, Tuple, Optional, List
import numpy as np
import open3d as o3d
import yaml

# -------------------------
# Config / I/O
# -------------------------


def load_cfg(path: str) -> Dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def load_mesh_glb(path: str, unit_autoscale: bool = False) -> Tuple[o3d.geometry.TriangleMesh, np.ndarray, np.ndarray]:
    mesh = o3d.io.read_triangle_mesh(path)
    if mesh.is_empty():
        raise ValueError(f"Failed to load mesh: {path}")
    mesh.compute_vertex_normals()

    aabb = mesh.get_axis_aligned_bounding_box()
    center = np.asarray(aabb.get_center())
    extent = np.asarray(aabb.get_extent())

    if unit_autoscale:
        scale = 1.0
        mx = float(np.max(extent))
        if mx > 5.0:       # likely millimeters -> meters
            scale = 0.001
        elif mx < 0.01:    # likely centimeters -> meters
            scale = 0.01
        if scale != 1.0:
            mesh.scale(scale, center=aabb.get_center())
            aabb = mesh.get_axis_aligned_bounding_box()
            center = np.asarray(aabb.get_center())
            extent = np.asarray(aabb.get_extent())

    return mesh, center, extent

# -------------------------
# Math helpers
# -------------------------

def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / (n + 1e-9)

# -------------------------
# Camera intrinsics / pose
# -------------------------

def pinhole_intrinsics(width: int = 640, height: int = 480, fov_deg: float = 60.0) -> Dict[str, float]:
    fx = fy = 0.5 * width / np.tan(np.radians(fov_deg / 2))
    return dict(width=width, height=height, fx=fx, fy=fy, cx=width / 2, cy=height / 2)

def default_intr_from_cfg(cfg: Dict) -> Dict[str, float]:
    i = cfg["intrinsics"]
    return pinhole_intrinsics(i["width"], i["height"], i["fov_deg"])

def look_at_cv(cam_pos, target, up=np.array([0, 0, 1.0], dtype=float)):
    """
    Camera-to-world transform for **CV convention**:
      - x: right, y: down, z: forward
      - Forward axis is **+Z** pointing from camera to target.
    Returns 4x4 world transform.
    """
    cam_pos = np.asarray(cam_pos, dtype=float)
    target  = np.asarray(target,  dtype=float)
    up      = _normalize(np.asarray(up, dtype=float))

    f = _normalize(target - cam_pos)          # forward (+Z)
    if abs(np.dot(f, up)) > 0.999:
        up = np.array([0, 1, 0], float) if abs(f[2]) > 0.9 else np.array([0, 0, 1], float)
        up = _normalize(up)

    r = _normalize(np.cross(f, up))           # right  (+X)
    u = _normalize(np.cross(r, f))            # true up(+Y)

    R = np.stack([r, u, f], axis=1)           # columns = camera axes in world
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3,  3] = cam_pos
    return T

# -------------------------
# Visualization helpers
# -------------------------

def make_spheres(points: np.ndarray, radius=0.01, color=(0.1, 0.6, 1.0)) -> List[o3d.geometry.TriangleMesh]:
    geoms = []
    for p in points:
        s = o3d.geometry.TriangleMesh.create_sphere(radius)
        s.translate(p)
        s.paint_uniform_color(color)
        geoms.append(s)
    return geoms

def make_rays(points: np.ndarray, target: np.ndarray, every=8, color=(0.6, 0.6, 0.6)) -> o3d.geometry.LineSet:
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

def create_grid(size=1.0, step=0.05, z=0.0) -> o3d.geometry.LineSet:
    pts, lines = [], []
    n = int(size / step)
    for i in range(-n, n + 1):
        pts.extend([[-size, i*step, z], [ size, i*step, z]]); lines.append([len(pts)-2, len(pts)-1])
        pts.extend([[i*step, -size, z], [i*step,  size, z]]); lines.append([len(pts)-2, len(pts)-1])
    grid = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(np.asarray(pts)),
        lines=o3d.utility.Vector2iVector(np.asarray(lines))
    )
    grid.paint_uniform_color([0.4, 0.4, 0.4])
    return grid

def validate_look_at(cams: np.ndarray, target: np.ndarray) -> None:
    """Print angular error between camera +Z axis and (target - cam)."""
    errs = []
    for p in cams:
        T = look_at_cv(p, target)
        z_world = T[:3, 2]               # camera +Z in world
        to_tgt  = _normalize(target - p)
        ang = np.degrees(np.arccos(np.clip(np.dot(z_world, to_tgt), -1.0, 1.0)))
        errs.append(ang)
    if errs:
        print(f"[look_at_cv] pointing error — mean:{np.mean(errs):.3f}°, max:{np.max(errs):.3f}°")

# -------------------------
# Raycasting (+Z forward) and depth rendering
# -------------------------

def rays_from_camera_cv(cam_T_world: np.ndarray, intr: Dict[str, float]) -> np.ndarray:
    w, h = int(intr["width"]), int(intr["height"])
    fx, fy, cx, cy = float(intr["fx"]), float(intr["fy"]), float(intr["cx"]), float(intr["cy"])
    i, j = np.meshgrid(np.arange(w), np.arange(h))
    x = (i - cx) / fx
    y = (j - cy) / fy
    dirs_cam = np.stack([x, y, np.ones_like(x)], axis=-1)  # +Z forward
    R = cam_T_world[:3, :3]; t = cam_T_world[:3, 3]
    dirs_w = dirs_cam @ R.T
    dirs_w /= (np.linalg.norm(dirs_w, axis=-1, keepdims=True) + 1e-9)
    orig_w = np.broadcast_to(t, dirs_w.shape)
    return np.concatenate([orig_w, dirs_w], axis=-1).astype(np.float32).reshape(-1, 6)

def build_clean_tensor_scene(mesh_legacy: o3d.geometry.TriangleMesh) -> o3d.t.geometry.RaycastingScene:
    V = np.asarray(mesh_legacy.vertices)
    F = np.asarray(mesh_legacy.triangles)
    if V.size == 0 or F.size == 0:
        raise ValueError("Empty mesh geometry.")
    tmesh = o3d.t.geometry.TriangleMesh(
        vertex_positions=o3d.core.Tensor(V.astype(np.float32)),
        triangle_indices=o3d.core.Tensor(F.astype(np.int32))
    )
    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(tmesh)
    return scene

def render_depth_z_cv(scene, cam_T_world: np.ndarray, intr: Dict[str, float], depth_trunc: float = 3.0) -> np.ndarray:
    """
    z-depth (meters along camera +Z) for TSDF/RGB-D.
    """
    H, W = int(intr["height"]), int(intr["width"])
    fx, fy, cx, cy = float(intr["fx"]), float(intr["fy"]), float(intr["cx"]), float(intr["cy"])
    i, j = np.meshgrid(np.arange(W), np.arange(H))
    x = (i - cx) / fx
    y = (j - cy) / fy
    ray_norm = np.sqrt(x*x + y*y + 1.0)     # ||[x,y,1]||
    cos_theta = 1.0 / (ray_norm + 1e-9)

    rays = rays_from_camera_cv(cam_T_world, intr)
    t_hit = scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy().reshape(H, W)
    depth_z = np.where(np.isfinite(t_hit), t_hit * cos_theta, 0.0)
    return np.clip(depth_z, 0.0, float(depth_trunc))

# Handy: quick center-ray test to avoid wasted renders
def center_ray_hits(scene, cam_T_world: np.ndarray, intr: Dict[str, float]) -> bool:
    dir_cam = np.array([0.0, 0.0, 1.0], dtype=np.float32)  # +Z forward
    R = cam_T_world[:3, :3]; t = cam_T_world[:3, 3]
    dir_w = dir_cam @ R.T
    ray = np.concatenate([t, dir_w]).astype(np.float32)[None, ...]
    t_hit = scene.cast_rays(o3d.core.Tensor(ray))["t_hit"].numpy()[0]
    return np.isfinite(t_hit)

# -------------------------
# TSDF (Open3D legacy integration)
# -------------------------

class TSDF:
    def __init__(self, voxel: float = 0.008, trunc: float = 0.024):
        self.vol = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=float(voxel),
            sdf_trunc=float(trunc),
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.NoColor
        )

    def integrate(self, depth_m: np.ndarray, intr: Dict[str, float], cam_T_world: np.ndarray,
                  depth_scale: float = 1000.0, depth_trunc: float = 3.0) -> None:
        H, W = int(intr["height"]), int(intr["width"])
        depth_u16 = (np.clip(depth_m, 0, depth_trunc) * depth_scale).astype(np.uint16)
        depth_img = o3d.geometry.Image(depth_u16)
        rgb_img = o3d.geometry.Image(np.zeros((H, W, 3), dtype=np.uint8))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb_img, depth_img, depth_scale=float(depth_scale), depth_trunc=float(depth_trunc),
            convert_rgb_to_intensity=False
        )
        intrinsic = o3d.camera.PinholeCameraIntrinsic(W, H, float(intr["fx"]), float(intr["fy"]),
                                                      float(intr["cx"]), float(intr["cy"]))
        ext = np.linalg.inv(cam_T_world)  # expects world_T_cam^-1
        self.vol.integrate(rgbd, intrinsic, ext)

    def extract_mesh(self) -> o3d.geometry.TriangleMesh:
        mesh = self.vol.extract_triangle_mesh()
        mesh.compute_vertex_normals()
        return mesh

# -------------------------
# Intrinsics-aware camera radius (prevents empty views)
# -------------------------

def radius_for_image_fill(extent_xy_max: float,
                          intr: Dict[str, float],
                          fill_frac: float = 0.7,
                          min_clearance: float = 0.02) -> float:
    """
    Choose camera distance so the object's lateral size fills ~fill_frac of image width.
    Simple pinhole approx: d ≈ fx * W_object / (fill_frac * W_pixels)
    Ensures at least min_clearance around the object.
    """
    fx = float(intr["fx"]); Wpx = float(intr["width"])
    W_obj = float(extent_xy_max) + 2.0 * float(min_clearance)
    d_desired = (fx * W_obj) / max(1e-6, fill_frac * Wpx)
    return max(d_desired, W_obj)  # never inside the object

# -------------------------
# Hemisphere sampling (Fibonacci) — world frame
# -------------------------

def fib_hemisphere_dirs(num_pts: int, phi_min_deg: float, phi_max_deg: float) -> np.ndarray:
    assert num_pts > 0
    ga = np.pi * (3.0 - np.sqrt(5.0))  # golden angle
    i = np.arange(num_pts)
    z0, z1 = np.sin(np.radians(phi_min_deg)), np.sin(np.radians(phi_max_deg))
    z = z0 + (z1 - z0) * ((i + 0.5) / num_pts)   # uniform in z within band
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, 1.0))
    th = ga * i
    d = np.stack([r * np.cos(th), r * np.sin(th), z], axis=1)
    d /= (np.linalg.norm(d, axis=1, keepdims=True) + 1e-9)
    return d

def base_point_from_aabb(aabb: o3d.geometry.AxisAlignedBoundingBox, center: np.ndarray) -> np.ndarray:
    return np.array([center[0], center[1], float(aabb.get_min_bound()[2])], dtype=float)

def generate_base_centered_fib_cameras(
    mesh: o3d.geometry.TriangleMesh,
    center: np.ndarray,
    extent: np.ndarray,
    phi_min_deg: float, phi_max_deg: float,
    z_margin: float,
    r0_m: float,
    samples: int = 192,
    safety_scale: float = 1.3,
    intr: Optional[Dict[str, float]] = None,
    fill_frac: float = 0.7,
    min_clearance: float = 0.02,
) -> np.ndarray:
    """
    World-frame hemisphere centered at the object's base.
    If intr is provided, radius is increased to achieve ~fill_frac of image width.
    """
    aabb = mesh.get_axis_aligned_bounding_box()
    base = base_point_from_aabb(aabb, center)
    dirs = fib_hemisphere_dirs(samples, phi_min_deg, phi_max_deg)

    half_r_xy = 0.5 * max(extent[0], extent[1])
    base_r = max(r0_m, safety_scale * max(half_r_xy, 1e-6))
    if intr is not None:
        base_r = max(base_r, radius_for_image_fill(2.0 * half_r_xy, intr, fill_frac, min_clearance))

    cams = base + base_r * dirs
    cams[:, 2] = np.maximum(cams[:, 2], base[2] + max(0.01, z_margin))
    return cams

# -------------------------
# PCA object-frame helpers
# -------------------------

def pca_object_frame(mesh: o3d.geometry.TriangleMesh) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimate an object-centric orthonormal basis (x_o, y_o, z_o) via PCA of vertices.
    z_o = longest principal axis; ensure right-handed.
    """
    V = np.asarray(mesh.vertices)
    c = V.mean(axis=0)
    X = V - c
    C = (X.T @ X) / max(len(X) - 1, 1)
    vals, vecs = np.linalg.eigh(C)  # ascending
    z_o = vecs[:, -1]
    x_o = vecs[:, -2]
    x_o = x_o - z_o * (x_o @ z_o)
    x_o /= (np.linalg.norm(x_o) + 1e-9)
    y_o = np.cross(z_o, x_o); y_o /= (np.linalg.norm(y_o) + 1e-9)
    if np.dot(np.cross(x_o, y_o), z_o) < 0:
        y_o = -y_o
    return c, x_o, y_o, z_o

def extents_in_object_frame(mesh: o3d.geometry.TriangleMesh,
                            center: np.ndarray,
                            axes: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    V = np.asarray(mesh.vertices) - center
    A = np.stack(axes, axis=1)
    L = V @ A
    mins = L.min(axis=0)
    maxs = L.max(axis=0)
    ext = maxs - mins
    return ext, mins, maxs

def object_frame_matrix(x_o: np.ndarray, y_o: np.ndarray, z_o: np.ndarray) -> np.ndarray:
    return np.stack([x_o, y_o, z_o], axis=1)

# -------------------------
# Object-frame hemisphere (base-centered)
# -------------------------

def generate_objectframe_base_fib_cameras(
    mesh: o3d.geometry.TriangleMesh,
    center: np.ndarray,
    x_o: np.ndarray, y_o: np.ndarray, z_o: np.ndarray,
    extents_obj: np.ndarray,
    phi_min_deg: float, phi_max_deg: float,
    z_margin: float,
    r0_m: float,
    samples: int = 192,
    safety_scale: float = 1.3,
    intr: Optional[Dict[str, float]] = None,
    fill_frac: float = 0.7,
    min_clearance: float = 0.02,
) -> np.ndarray:
    """
    Fibonacci hemisphere in the object frame (z_o up), centered at base plane.
    If intr is provided, radius is increased to achieve ~fill_frac of image width.
    Returns camera positions in WORLD coordinates.
    """
    R_ow = object_frame_matrix(x_o, y_o, z_o)  # columns = axes
    R_wo = R_ow.T

    V = np.asarray(mesh.vertices) - center
    L = V @ R_wo
    min_z_o = float(L[:, 2].min())
    base_o = np.array([0.0, 0.0, min_z_o], dtype=float)

    half_r_xy = 0.5 * max(extents_obj[0], extents_obj[1])
    base_r = max(r0_m, safety_scale * max(half_r_xy, 1e-6))
    if intr is not None:
        base_r = max(base_r, radius_for_image_fill(2.0 * half_r_xy, intr, fill_frac, min_clearance))

    # sample directions directly in object frame
    ga = np.pi * (3.0 - np.sqrt(5.0))
    i = np.arange(samples)
    z0, z1 = np.sin(np.radians(phi_min_deg)), np.sin(np.radians(phi_max_deg))
    z = z0 + (z1 - z0) * ((i + 0.5) / samples)
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, 1.0))
    th = ga * i
    dirs_o = np.stack([r * np.cos(th), r * np.sin(th), z], axis=1)
    dirs_o /= (np.linalg.norm(dirs_o, axis=1, keepdims=True) + 1e-9)

    cams_o = base_o + base_r * dirs_o
    cams_o[:, 2] = np.maximum(cams_o[:, 2], base_o[2] + max(0.01, z_margin))
    cams_w = center + cams_o @ R_ow.T
    return cams_w

# -------------------------
# Exported names
# -------------------------

__all__ = [
    # config / io
    "load_cfg", "load_mesh_glb",
    # camera
    "pinhole_intrinsics", "default_intr_from_cfg", "look_at_cv",
    # math
    "_normalize",
    # viz
    "make_spheres", "make_rays", "create_grid", "validate_look_at",
    # raycasting / depth
    "build_clean_tensor_scene", "rays_from_camera", "render_depth",
    "render_depth_z", "rays_from_camera_cv", "render_depth_z_cv",
    # tsdf
    "TSDF",
    # world-frame hemisphere
    "fib_hemisphere_dirs", "base_point_from_aabb", "generate_base_centered_fib_cameras",
    # object-frame helpers
    "pca_object_frame", "extents_in_object_frame", "object_frame_matrix",
    # object-frame hemisphere
    "generate_objectframe_base_fib_cameras",
]
