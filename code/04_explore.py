# -*- coding: utf-8 -*-
"""
阶段 5：探索性分析（子样本，避免 OOM）。

每张切片取连续 8,000 个细胞 → 归一化 → HVG → PCA → 邻居图 → UMAP → Leiden，
并与官方 cell_label 比对，检查数据可用性与聚类结构。

产出（analysis/04_explore/）：
  fig_umap_wt.png / fig_umap_iso.png
  fig_spatial_leiden_wt.png
  cluster_check_wt.csv
  explore_summary.txt
"""
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from scipy.sparse import vstack

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import spatial_io as sio

OUT = r"E:\研究工作\agent长非与空间域\analysis\04_explore"
os.makedirs(OUT, exist_ok=True)

N_PER_SAMPLE = 8000
rng = np.random.default_rng(0)


def build_subset(key, n_per_sample=N_PER_SAMPLE):
    obs = sio.read_obs(key)
    var = sio.read_var(key)
    samples = list(obs["sample"].cat.categories) if hasattr(obs["sample"], "cat") else sorted(obs["sample"].unique())

    # 检查样本是否连续排列
    contig = True
    pos = {}
    for s in samples:
        idx = np.where(obs["sample"].astype(str).to_numpy() == str(s))[0]
        pos[s] = (idx.min(), idx.max(), len(idx))
        if idx.max() - idx.min() + 1 != len(idx):
            contig = False
    print("  样本索引区间:", {k: v for k, v in pos.items()}, "| 连续:", contig)

    blocks, obs_parts = [], []
    for s in samples:
        lo, hi, n = pos[s]
        take = min(n_per_sample, n)
        start = lo + (n - take) // 2          # 取中段，避免切片边缘
        x = sio.read_rows(key, start, start + take)
        blocks.append(x)
        obs_parts.append(obs.iloc[start:start + take])
        print(f"  sample {s}: 取 [{start}, {start + take}) 共 {take:,} 细胞")

    X = vstack(blocks, format="csr")
    o = pd.concat(obs_parts)
    a = ad.AnnData(X=X.astype(np.float32), obs=o.copy(), var=var.copy())
    a.obs_names_make_unique()
    return a


def run_panel(panel, key, marker_csv=None):
    t0 = time.time()
    print(f"\n===== 探索性分析: {panel} =====", flush=True)
    a = build_subset(key)
    print(f"  子集矩阵: {a.shape}, nnz={a.X.nnz:,}", flush=True)

    # 预处理
    sc.pp.filter_cells(a, min_counts=50)
    sc.pp.normalize_total(a, target_sum=100.0)   # MERFISH 每细胞计数少，用较小 target
    sc.pp.log1p(a)
    sc.pp.highly_variable_genes(a, n_top_genes=2000, flavor="seurat")
    a.raw = a
    a = a[:, a.var["highly_variable"]].copy()

    sc.pp.scale(a, max_value=10)
    sc.tl.pca(a, n_comps=30, svd_solver="arpack")
    print(f"  PCA 完成 ({time.time()-t0:.0f}s) | 前5个PC方差比: "
          f"{np.round(a.uns['pca']['variance_ratio'][:5], 3).tolist()}", flush=True)

    sc.pp.neighbors(a, n_neighbors=15, n_pcs=20)
    sc.tl.umap(a, min_dist=0.3)
    sc.tl.leiden(a, resolution=1.0, key_added="leiden", flavor="igraph", n_iterations=2,
                 directed=False)
    print(f"  UMAP + Leiden 完成 ({time.time()-t0:.0f}s) | 簇数: {a.obs['leiden'].nunique()}", flush=True)

    # 与官方 cell_label 的一致性
    ct = pd.crosstab(a.obs["leiden"], a.obs["cell_label"])
    ct.to_csv(os.path.join(OUT, f"cluster_vs_celllabel_{panel}.csv"))
    purity = (ct.max(axis=1) / ct.sum(axis=1))
    print(f"  Leiden 簇对 cell_label 的纯度中位数: {purity.median():.3f}", flush=True)

    # 每簇最可能的细胞类型
    top_type = ct.idxmax(axis=1)
    check = pd.DataFrame({
        "leiden": ct.index,
        "n_cells": ct.sum(axis=1).to_numpy(),
        "top_cell_label": top_type.to_numpy(),
        "purity": purity.to_numpy(),
    })
    check.to_csv(os.path.join(OUT, f"cluster_check_{panel}.csv"), index=False)
    print(check.sort_values("n_cells", ascending=False).head(12).to_string(index=False))

    # 图
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))
    cats = list(a.obs["cell_label"].cat.categories) if hasattr(a.obs["cell_label"], "cat") \
        else sorted(a.obs["cell_label"].unique())
    colors = plt.cm.tab20(np.linspace(0, 1, len(cats)))
    cmap_dict = dict(zip(cats, colors))
    for c in cats:
        m = (a.obs["cell_label"] == c).to_numpy()
        axes[0].scatter(a.obsm["X_umap"][m, 0], a.obsm["X_umap"][m, 1], s=1,
                        color=cmap_dict[c], label=str(c), linewidths=0)
    axes[0].set_title("UMAP: official cell_label (legend omitted, 36+ types)")
    axes[0].set_xlabel("UMAP1"); axes[0].set_ylabel("UMAP2")
    axes[0].legend(fontsize=3.5, markerscale=4, ncol=2, loc="best", frameon=False)

    lei = a.obs["leiden"].to_numpy()
    lc = plt.cm.tab20(np.linspace(0, 1, len(np.unique(lei))))
    for i, cl in enumerate(np.unique(lei)):
        m = lei == cl
        axes[1].scatter(a.obsm["X_umap"][m, 0], a.obsm["X_umap"][m, 1], s=1, color=lc[i],
                        label=cl, linewidths=0)
    axes[1].set_title(f"UMAP: Leiden clusters (n={len(np.unique(lei))})")
    axes[1].legend(fontsize=6, markerscale=5, ncol=2, frameon=False)
    axes[1].set_xlabel("UMAP1")

    # 空间图：只画第一个样本，避免重叠
    s0 = a.obs["sample"].astype(str).iloc[0]
    m0 = (a.obs["sample"].astype(str) == s0).to_numpy()
    for i, c in enumerate(cats):
        m = m0 & (a.obs["cell_label"] == c).to_numpy()
        if m.sum() == 0:
            continue
        axes[2].scatter(a.obs.loc[m, "global_x"], a.obs.loc[m, "global_y"], s=0.6,
                        color=cmap_dict[c], linewidths=0)
    axes[2].set_title(f"Spatial: cell_label (sample {s0}, n={m0.sum():,})")
    axes[2].set_aspect("equal")
    axes[2].set_xlabel("global_x (um)"); axes[2].set_ylabel("global_y (um)")

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"fig_umap_{panel.lower()}.png"), dpi=150)
    plt.close(fig)

    # 保存子集嵌入供后续用
    emb = pd.DataFrame(a.obsm["X_umap"], columns=["UMAP1", "UMAP2"], index=a.obs_names)
    emb = emb.join(a.obs[["sample", "cell_label", "leiden", "global_x", "global_y", "annot"]])
    emb.to_csv(os.path.join(OUT, f"embedding_{panel}.csv"))

    print(f"  {panel} 用时 {time.time()-t0:.0f}s", flush=True)
    return a


if __name__ == "__main__":
    which = (sys.argv[1] if len(sys.argv) > 1 else "WT").upper()
    if which in ("WT", "BOTH"):
        run_panel("WT", "cellxgene_WT")
    if which in ("ISO", "BOTH"):
        run_panel("ISO", "cellxgene_ISO")
    print("\n=== 探索性分析完成:", OUT, "===")
