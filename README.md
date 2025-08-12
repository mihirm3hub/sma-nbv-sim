# 📸 SMA-NBV: Slime Mould Algorithm for Next-Best-View Selection

This repository implements a **Slime Mould Algorithm (SMA)**-based Next-Best-View (NBV) planner over hemisphere camera candidates, with **TSDF fusion** to reconstruct a 3D mesh.
It supports both **2D depth-mask coverage** and **3D surface-area coverage** scoring, with optional diversity penalties to reduce redundant views.

<!-- ![Tri-View Example](docs/tri_view_example.png) <sub>GT (left) | Baseline (center) | SMA (right)</sub> -->

---

## 🚀 Features

* **SMA optimization** for selecting best camera viewpoints from hemisphere candidates.
* **2D coverage** (depth masks) or **3D coverage** (front-facing visible surface area) scoring.
* **TSDF fusion** to create meshes from selected viewpoints.
* **Tri-View visualization**: Ground Truth | Baseline | SMA result.
* **Candidate camera visualization** with connections and coverage.
* Flexible YAML configs for geometry, hemisphere bounds, intrinsics.

---

## 📂 Repository Structure

```
SMA-NBV/
├─ assets/                     # Small demo meshes (or leave empty, see below)
│   └─ soda-can.glb
├─ experiments/
│   ├─ configs/                 # Hemisphere + camera setup configs
│   │   └─ can_hemisphere.yaml
│   └─ results/                 # Ignored: meshes, depth maps, CSVs, plots
├─ fuse_tsdf.py                 # TSDF fusion from a sequence of depth maps
├─ render_multi_depth.py        # Render depth from multiple camera poses
├─ sma_nbv.py                   # SMA-guided NBV selection + TSDF fusion
├─ utils.py                     # Geometry, mesh, camera, and TSDF helpers
├─ visualize.py                 # Camera candidate & mesh visualization
├─ run_sma.py                   # Example batch runner for SMA
├─ run_baselines.py             # Random / baseline view selectors
├─ env.yaml / requirements.txt  # Environment spec
├─ README.md
└─ .gitignore
```

---

## 📦 Installation

Clone the repo:

```bash
git clone https://github.com/mihirm3hub/sma-nbv-sim.git
cd SMA-NBV
```

Install dependencies:

```bash
# Option 1: Conda
conda env create -f env.yaml
conda activate sma-nbv

# Option 2: pip
pip install -r requirements.txt
```

---

## 🛠 Quickstart

### 1️⃣ Visualize candidate camera poses

```bash
python visualize.py --cfg experiments/configs/can_hemisphere.yaml
```

### 2️⃣ Fuse all views as baseline

```bash
python fuse_tsdf.py --cfg experiments/configs/can_hemisphere.yaml --voxel 0.002 --trunc 0.008 --out experiments/results/tsdf_mesh.ply
```
 
### 3️⃣ Run SMA NBV with 3D coverage scoring

```bash
python sma_nbv.py --cfg experiments/configs/can_hemisphere.yaml --budget 16 --use-3d-coverage --surf-samples 6000--voxel 0.002 --trunc 0.008 --baseline-mesh experiments/results/tsdf_mesh.ply
```

Outputs:

* `experiments/results/sma_selected.csv` – selected camera indices & positions
* `experiments/results/sma_tsdf_mesh.ply` – SMA fused mesh
* Tri-view window showing GT, baseline, and SMA reconstructions

---

## ⚙ Config

Edit `experiments/configs/can_hemisphere.yaml` to change:

* **Mesh path** (`mesh_path`)
* **Hemisphere bounds** (`phi_deg_min`, `phi_deg_max`, `z_margin_m`)
* **Camera intrinsics** (`width`, `height`, `fx`, `fy`, etc.)

---

## 📊 Evaluation (optional)

If you have ground truth point clouds, you can extend with **Chamfer Distance** or **Completeness** metrics.
(Current version only compares visually.)

---

## 📁 Data

* **Small demo mesh**: The included `assets/soda-can.glb` is for quick tests.
* For large meshes or datasets:

  * Use your own in `mesh_path` in the config
  * Or add download scripts in `scripts/`

---

## 🪝 Git Hygiene

The `.gitignore` is set to ignore:

* Generated meshes, depth maps, CSVs in `experiments/results/`
* Caches (`__pycache__`, `.vscode/`, `.idea/`)
* Large binary assets unless placed in `/assets` and explicitly whitelisted.

---

## 📜 License

MIT License (adjust as needed).

---

## 🙌 Citation

If you use this code in research:

```
@misc{sma-nbv,
  author = {Your Name},
  title = {SMA-NBV: Slime Mould Algorithm for Next-Best-View Planning},
  year = {2025},
  howpublished = {\url{https://github.com/<your-username>/SMA-NBV}}
}
```
