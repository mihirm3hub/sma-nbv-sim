from utils import load_cfg, load_mesh_glb, default_intr_from_cfg
import argparse

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', required=True)
    args = ap.parse_args()
    cfg = load_cfg(args.cfg)
    mesh, center, extent = load_mesh_glb(cfg['mesh_path'])
    intr = default_intr_from_cfg(cfg)
    print('OK: cfg + mesh + intr loaded.')
    print('Mesh center:', center, 'extent:', extent)