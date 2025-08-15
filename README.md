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
│   └─ sodacan.glb
├─ experiments/
│   ├─ configs/                 # Hemisphere + camera setup configs
│   │   └─ can_hemisphere.yaml
│   └─ results/                 # Ignored: meshes, depth maps, CSVs, plots
├─ eval_nbv.py                  # Compare sma, greedy, random methods
├─ fuse_tsdf.py                 # TSDF fusion from a sequence of depth maps
├─ make_figs.py                 # Coverage vs views plotting
├─ render_multi_depth.py        # Render depth from multiple camera poses
├─ sma_nbv.py                   # SMA-guided NBV selection + TSDF fusion
├─ utils.py                     # Geometry, mesh, camera, and TSDF helpers
├─ view_methods_frustums.py     # View camera frustums by methods 
├─ visualize.py                 # Camera candidate & mesh visualization

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

<!-- ## 📊 Evaluation (optional)

If you have ground truth point clouds, you can extend with **Chamfer Distance** or **Completeness** metrics.
(Current version only compares visually.) -->

## 📊 Metrics
Generate Multi-Panel Summary
Produces one figure containing:
Coverage vs views (mean ± std),
Final coverage @ budget,
Views to reach 80%,
Coverage gain per step (mean ± std).

```bash
python make_figs_gain.py \
    --csv experiments/coverage_by_method.csv \
    --object ALL \
    --use-3d any \
    --target 0.80 \
    --out plots/summary_all_panels.png
```

**Final Coverage @ Budget** — Coverage fraction at the last view within the given budget.

**Views-to-Target** — Number of views required to reach a target coverage (e.g., 80%).

**Coverage Gain per Step** — Incremental coverage improvement per additional view.

All metrics are saved in `coverage_by_method.csv` with the following columns:

| Column              | Description                            |
| ------------------- | -------------------------------------- |
| `method`            | NBV method (`SMA`, `GREEDY`, `RANDOM`) |
| `step`              | View index (1-based)                   |
| `coverage_fraction` | Fractional surface coverage (0–1)      |
| `seed`              | Random seed used for reproducibility   |
| `budget`            | Maximum number of views in the run     |
| `use_3d`            | 3D fusion flag (1/0)                   |
| `object`            | Object name                            |

---

## 📈 Results

From experiments/coverage_by_method.csv:
| Method | Coverage (%) | Views to 80% | Success Rate |
| ------ | ------------ | ------------ | ------------ |
| SMA    | 89.47 ± 0.3  | 3.0 ± 0.0    | 100%         |
| Greedy | 88.92 ± 0.5  | 4.0 ± 0.0    | 100%         |
| Random | 89.77 ± 0.6  | 12.0 ± 1.2   | 100%         |


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

Apache 2.0. © Mihir Milind Mainkar

---

## 🙌 Citation

If you use this code in research, please cite:

```
@misc{sma-nbv,
  author = {Mihir Milind Mainkar},
  title = {SMA-NBV: Slime Mould Algorithm for Next-Best-View Planning},
  year = {2025},
  url = {\url{https://github.com/mihirm3hub/sma-nbv-sim.git}}
}
```
