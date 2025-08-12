from utils import load_cfg
import argparse
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', required=True)
    args = ap.parse_args()
    cfg = load_cfg(args.cfg)
    print('SMA runner stub. Using cfg:', args.cfg)