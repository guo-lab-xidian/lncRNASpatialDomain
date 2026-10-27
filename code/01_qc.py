# -*- coding: utf-8 -*-
"""
阶段 2：细胞级 + 空间 QC（流式，内存友好）。

产出（analysis/01_qc/）：
  cell_qc_<panel>.csv       每细胞 QC 指标 + obs 元数据
  feature_qc_<panel>.csv    每特征检出率
  sample_summary.csv        按 sample 汇总
  fig_cell_qc_<panel>.png   细胞级分布（按 sample 分面）
  fig_spatial_<panel>.png   空间覆盖与密度

用法：
  python 01_qc.py WT        只跑 WT
  python 01_qc.py ISO       只跑 ISO
  python 01_qc.py           两个都跑
"""
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import spatial_io as sio

OUT = r"E:\研究工作\agent长非与空间域\analysis\01_qc"
os.makedirs(OUT, exist_ok=True)

PANELS = {
    "WT": "cellxgene_WT",
    "ISO": "cellxgene_ISO",
}
LAYER = "X"          # README: .X 为原始计数
BLOCK = 15000


def run(panel):
    key = PANELS[panel]
    t0 = time.time()
    print(f"\n===== {panel} ({key}) =====", flush=True)
    n_obs, n_var, nnz = sio.shape_of(key, layer=LAYER)
    print(f"  shape: {n_obs:,} cells x {n_var:,} features | nnz={nnz:,} "
          f"| 稀疏度={100 * (1 - nnz / (n_obs * n_var)):.3f}% 空", flush=True)

    obs = sio.read_obs(key)
    print(f"  obs 列: {list(obs.columns)}", flush=True)

    print("  [1/2] 计算细胞级指标 ...", flush=True)
    cm = sio.cell_metrics(key, layer=LAYER, block_rows=BLOCK, verbose=False)
    print(f"        完成，用时 {time.time() - t0:.0f}s", flush=True)

    # 行顺序与 obs 一致，按位置合并（obs 索引是 barcode 字符串，不能直接 join）
    df = obs.copy()
    for col in cm.columns:
        df[col] = cm[col].to_numpy()
    df.to_csv(os.path.join(OUT, f"cell_qc_{panel}.csv"))

    print("  [2/2] 计算特征级检出率 ...", flush=True)
    fm = sio.feature_metrics(key, layer=LAYER, block_rows=BLOCK)
    fm.to_csv(os.path.join(OUT, f"feature_qc_{panel}.csv"))
    print(f"        完成，总用时 {time.time() - t0:.0f}s", flush=True)

    # ---- 按 sample 汇总 ----
    agg_cols = ["total_counts", "n_detected", "volume", "num_z", "top_gene_frac"]
    agg_cols = [c for c in agg_cols if c in df.columns]
    summ = df.groupby("sample", observed=True)[agg_cols].agg(
        ["count", "median", "mean", "std"])
    summ.to_csv(os.path.join(OUT, f"sample_summary_{panel}.csv"))

    print("\n  --- 按 sample 汇总（中位数）---")
    med = df.groupby("sample", observed=True)[agg_cols].median()
    med["n_cells"] = df.groupby("sample", observed=True).size()
    print(med.to_string())

    print("\n  --- 细胞级指标分位数（全部 sample）---")
    q = df[agg_cols].quantile([0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0])
    print(q.to_string())

    print("\n  --- 特征检出率分位数 ---")
    print(fm["detect_frac"].quantile([0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0]).to_string())
    for thr in [0.01, 0.05, 0.1, 0.2, 0.5]:
        n = int((fm["detect_frac"] >= thr).sum())
        print(f"    检出率 >= {thr:>4}: {n:>6,} 个特征 ({100 * n / len(fm):.1f}%)")

    plot_cell_qc(panel, df, agg_cols)
    plot_spatial(panel, df)
    return df, fm


def plot_cell_qc(panel, df, cols):
    samples = list(df["sample"].cat.categories) if hasattr(df["sample"], "cat") else sorted(df["sample"].unique())
    metrics = [c for c in ["total_counts", "n_detected", "volume", "num_z"] if c in cols]
    fig, axes = plt.subplots(len(metrics), len(samples),
                             figsize=(3.6 * len(samples), 2.6 * len(metrics)),
                             squeeze=False)
    for i, m in enumerate(metrics):
        for j, s in enumerate(samples):
            ax = axes[i][j]
            v = df.loc[df["sample"] == s, m].astype(float)
            v = v[v > 0]
            if m in ("total_counts", "volume"):
                ax.hist(np.log10(v), bins=60, color="#4C78A8")
                ax.set_xlabel(f"log10({m})")
            else:
                vmax = np.percentile(v, 99.5) if len(v) else 1
                ax.hist(v[v <= vmax], bins=60, color="#4C78A8")
                ax.set_xlabel(m)
            ax.set_title(f"sample {s} (n={len(v):,})", fontsize=9)
            ax.tick_params(labelsize=7)
            if j == 0:
                ax.set_ylabel("cells", fontsize=8)
    fig.suptitle(f"{panel}: per-cell QC by sample", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"fig_cell_qc_{panel}.png"), dpi=140)
    plt.close(fig)


def plot_spatial(panel, df):
    if "global_x" not in df.columns:
        return
    samples = list(df["sample"].cat.categories) if hasattr(df["sample"], "cat") else sorted(df["sample"].unique())
    n = len(samples)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4.2), squeeze=False)
    for j, s in enumerate(samples):
        ax = axes[0][j]
        sub = df[df["sample"] == s]
        ax.scatter(sub["global_x"], sub["global_y"], s=0.05, c="#2F4B7C", alpha=0.25, linewidths=0)
        ax.set_title(f"sample {s} (n={len(sub):,})", fontsize=9)
        ax.set_aspect("equal")
        ax.set_xlabel("global_x (um)", fontsize=8)
        ax.set_ylabel("global_y (um)", fontsize=8)
        ax.tick_params(labelsize=7)
    fig.suptitle(f"{panel}: spatial coverage of segmented cells", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"fig_spatial_{panel}.png"), dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    which = sys.argv[1].upper() if len(sys.argv) > 1 else "BOTH"
    if which in ("WT", "BOTH"):
        run("WT")
    if which in ("ISO", "BOTH"):
        run("ISO")
    print("\n=== QC 完成，输出目录:", OUT, "===")
