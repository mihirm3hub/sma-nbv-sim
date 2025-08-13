# make_figs_gain.py — One figure with:
#   (left)  Coverage vs views (mean ± std)
#   (top-right)  Final coverage @ budget (per method, mean ± std over seeds)
#   (mid-right)  Views to reach 80% (per method, mean ± std over seeds + success rate)
#   (bottom-right)  Coverage gain per step (mean ± std)

import os, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# --------------------------- IO / filtering ---------------------------

def load_curves(csv_path, obj_filter=None, use3d_filter=None):
    df = pd.read_csv(csv_path)

    # normalize columns that sometimes arrive as strings
    if "coverage_fraction" in df.columns:
        df["coverage_fraction"] = pd.to_numeric(df["coverage_fraction"], errors="coerce")
    if "step" in df.columns:
        df["step"] = pd.to_numeric(df["step"], errors="coerce").astype("Int64")
    if "seed" in df.columns:
        df["seed"] = pd.to_numeric(df["seed"], errors="coerce").astype("Int64")
    if "budget" in df.columns:
        df["budget"] = pd.to_numeric(df["budget"], errors="coerce").astype("Int64")
    if "use_3d" in df.columns:
        # ensure numeric flag {0,1}
        df["use_3d"] = pd.to_numeric(df["use_3d"], errors="coerce").fillna(0).astype(int)

    # drop rows with NaN steps or coverage
    df = df.dropna(subset=["coverage_fraction", "step"])

    # Optional filters
    if obj_filter is not None and str(obj_filter).lower() != "all":
        if "object" not in df.columns:
            raise ValueError("CSV has no 'object' column to filter on.")
        df = df[df["object"].astype(str).str.lower() == str(obj_filter).lower()]

    if use3d_filter is not None and str(use3d_filter).lower() != "any":
        flag = 1 if str(use3d_filter).lower() in ("1","true","t","yes","y") else 0
        if "use_3d" not in df.columns:
            raise ValueError("CSV has no 'use_3d' column to filter on.")
        df = df[df["use_3d"] == flag]

    # Normalize method labels
    if "method" not in df.columns:
        raise ValueError("CSV must contain a 'method' column.")
    df["method"] = df["method"].astype(str).str.upper()

    # keep only expected columns if present
    keep = [c for c in ["method","step","coverage_fraction","seed","budget","use_3d","object"] if c in df.columns]
    df = df[keep].copy()

    # ensure step ordering within each (method, seed)
    df = df.sort_values(["method","seed","step"]).reset_index(drop=True)
    return df

# --------------------------- aggregations ---------------------------

def mean_std_by_method_step(df):
    g = df.groupby(["method","step"])["coverage_fraction"]
    mu = g.mean().rename("mean").reset_index()
    sd = g.std(ddof=0).fillna(0).rename("std").reset_index()
    return mu.merge(sd, on=["method","step"])

def final_coverage_at_budget(df):
    """
    Take the final (last step) coverage per (method, seed), then mean±std over seeds.
    Budget shown = max budget observed for that method (for display only).
    """
    # last row per (method, seed)
    last = df.sort_values(["method","seed","step"]).groupby(["method","seed"], as_index=False).tail(1)
    finals = last.groupby("method")["coverage_fraction"].agg(["mean","std"]).reset_index()
    finals["std"] = finals["std"].fillna(0)
    # show max budget per method if present, else use max step
    if "budget" in last.columns and last["budget"].notna().any():
        budgets = last.groupby("method")["budget"].max().rename("budget").reset_index()
    else:
        budgets = last.groupby("method")["step"].max().rename("budget").reset_index()
    return finals.merge(budgets, on="method")

def views_to_reach(df, target=0.80):
    """
    First step where coverage >= target, per (method, seed).
    Returns mean±std of views and success rate across seeds for each method.
    """
    rows = []
    for (m, s), grp in df.groupby(["method","seed"], dropna=False):
        grp = grp.sort_values("step")
        hit = grp.loc[grp["coverage_fraction"] >= float(target)]
        if len(hit):
            rows.append({"method": m, "seed": s, "views": int(hit.iloc[0]["step"])})
    if not rows:
        return pd.DataFrame(columns=["method","mean","std","succ_rate"])

    vt = pd.DataFrame(rows)
    summary = vt.groupby("method")["views"].agg(["mean","std"]).reset_index()
    summary["std"] = summary["std"].fillna(0)

    total_seeds = df.groupby("method")["seed"].nunique().rename("total_seeds")
    succ = vt.groupby("method")["seed"].nunique().rename("succ")
    summary = summary.merge(total_seeds, on="method").merge(succ, on="method")
    summary["succ_rate"] = 100.0 * summary["succ"] / summary["total_seeds"].clip(lower=1)
    return summary[["method","mean","std","succ_rate"]]

def per_step_gain(df):
    """
    Gain at step k = cov(k) - cov(k-1), with cov(0)=0, for each (method, seed).
    Aggregate mean±std by (method, step).
    """
    parts = []
    for (m, s), grp in df.groupby(["method","seed"], dropna=False):
        grp = grp.sort_values("step")
        cov = grp["coverage_fraction"].to_numpy(dtype=float)
        steps = grp["step"].to_numpy()
        gain = np.diff(np.concatenate([[0.0], cov]))
        parts.append(pd.DataFrame({"method": m, "seed": s, "step": steps, "gain": gain}))
    if not parts:
        return pd.DataFrame(columns=["method","step","mean","std"])
    gains = pd.concat(parts, ignore_index=True)
    agg = gains.groupby(["method","step"])["gain"].agg(["mean","std"]).reset_index()
    agg["std"] = agg["std"].fillna(0)
    return agg

# --------------------------- plotting ---------------------------

def plot_one_figure(df, out_png, title_suffix="object=ALL, use_3d=any", target=0.80):
    if df.empty:
        raise ValueError("Filtered DataFrame is empty — check --object / --use-3d filters and CSV path.")

    methods_order = ["SMA","GREEDY","RANDOM"]
    colors = {"SMA":"#2ca02c", "GREEDY":"#1f77b4", "RANDOM":"#ff7f0e"}

    curves = mean_std_by_method_step(df)
    finals = final_coverage_at_budget(df)
    v2t    = views_to_reach(df, target=target)
    gains  = per_step_gain(df)

    fig = plt.figure(figsize=(14,7.5), dpi=140)
    gs  = GridSpec(nrows=3, ncols=2, width_ratios=[2.2, 1], height_ratios=[1,1,1], hspace=0.35, wspace=0.25)
    ax_curve = fig.add_subplot(gs[:,0])
    ax_final = fig.add_subplot(gs[0,1])
    ax_v2t   = fig.add_subplot(gs[1,1])
    ax_gain  = fig.add_subplot(gs[2,1])

    # --- Left: coverage vs views (mean ± std)
    for m in methods_order:
        sub = curves[curves["method"] == m].sort_values("step")
        if sub.empty:
            continue
        x  = sub["step"].to_numpy()
        y  = 100.0 * sub["mean"].to_numpy()
        sd = 100.0 * sub["std"].to_numpy()
        ax_curve.plot(x, y, label=m, lw=2, marker="o", ms=3, color=colors[m])
        ax_curve.fill_between(x, y - sd, y + sd, alpha=0.15, color=colors[m])
    ax_curve.set_xlabel("# Views")
    ax_curve.set_ylabel("Coverage (%)")
    ax_curve.set_title("SMA-guided NBV vs Greedy & Random (mean ± std)")
    ax_curve.grid(True, alpha=0.3)
    ax_curve.set_ylim(0, 100)
    ax_curve.legend(loc="lower right", frameon=True)

    # --- Top-right: Final coverage @ budget
    finals_plot = finals.set_index("method").reindex(methods_order).dropna(subset=["mean"])
    if not finals_plot.empty:
        y    = 100.0 * finals_plot["mean"].to_numpy()
        yerr = 100.0 * finals_plot["std"].to_numpy()
        ax_final.bar(finals_plot.index, y, yerr=yerr, capsize=4,
                     color=[colors[m] for m in finals_plot.index])
        ax_final.set_ylim(0, 100)
        ax_final.set_title("Final Coverage @ Budget")
        ax_final.set_ylabel("Coverage (%)")
        # annotate bars with value and (budget)
        for i, (val, m) in enumerate(zip(y, finals_plot.index.tolist())):
            b = int(finals_plot.loc[m, "budget"]) if "budget" in finals_plot.columns else None
            label = f"{val:.1f}%"
            if b is not None:
                label += f"\n@{b}"
            ax_final.text(i, val + 1.5, label, ha="center", va="bottom", fontsize=9)
    else:
        ax_final.set_title("Final Coverage @ Budget")
        ax_final.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax_final.transAxes)
        ax_final.set_axis_off()

    # --- Mid-right: Views to reach target
    v2t_plot = v2t.set_index("method").reindex(methods_order).dropna(subset=["mean"], how="any")
    if not v2t_plot.empty:
        y    = v2t_plot["mean"].to_numpy()
        yerr = v2t_plot["std"].to_numpy()
        ax_v2t.bar(v2t_plot.index, y, yerr=yerr, capsize=4,
                   color=[colors[m] for m in v2t_plot.index])
        ax_v2t.set_title(f"Views to reach {int(target*100)}%")
        ax_v2t.set_ylabel("# Views")
        for i, m in enumerate(v2t_plot.index.tolist()):
            val = y[i]
            sr  = float(v2t_plot.loc[m, "succ_rate"])
            ax_v2t.text(i, val + 0.6, f"{val:.1f}\n{sr:.0f}%", ha="center", va="bottom", fontsize=9)
    else:
        ax_v2t.set_title(f"Views to reach {int(target*100)}%")
        ax_v2t.text(0.5, 0.5, "No method reached target", ha="center", va="center", transform=ax_v2t.transAxes)
        ax_v2t.set_axis_off()

    # --- Bottom-right: Coverage gain per step (mean ± std)
    for m in methods_order:
        sub = gains[gains["method"] == m].sort_values("step")
        if sub.empty:
            continue
        x  = sub["step"].to_numpy()
        y  = 100.0 * sub["mean"].to_numpy()
        sd = 100.0 * sub["std"].to_numpy()
        ax_gain.plot(x, y, label=m, lw=2, marker="o", ms=3, color=colors[m])
        ax_gain.fill_between(x, y - sd, y + sd, alpha=0.15, color=colors[m])
    ax_gain.set_xlabel("# Views")
    ax_gain.set_ylabel("Gain (%)")
    ax_gain.set_title("Coverage gain per step (mean ± std)")
    ax_gain.grid(True, alpha=0.3)

    fig.suptitle(f"Coverage vs. views — {title_suffix}", y=0.98, fontsize=14)
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    print(f"Saved: {out_png}")

# --------------------------- CLI ---------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="coverage_by_method.csv")
    ap.add_argument("--object", default="ALL", help="Filter object name (or 'ALL')")
    ap.add_argument("--use-3d", default="any", help="'1'/'0' or 'any' to mix")
    ap.add_argument("--target", type=float, default=0.80, help="Target coverage fraction (e.g., 0.80)")
    ap.add_argument("--out", default="experiments/results/summary_all_panels.png")
    args = ap.parse_args()

    df = load_curves(args.csv, obj_filter=args.object, use3d_filter=args.use_3d)
    suffix = f"object={args.object}, use_3d={args.use_3d}"
    plot_one_figure(df, args.out, title_suffix=suffix, target=args.target)
