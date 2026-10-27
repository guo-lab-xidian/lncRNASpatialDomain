# -*- coding: utf-8 -*-
"""
课题三 / L-domain：构造"真·空间域"标签（空间感知聚类）。

方法（BANKSY 思路的轻量实现）：
  1. 每张切片取中段 15,000 个细胞（共 45,000），归一化 + log1p
  2. 选 1,500 个高变基因（在子集上按方差选）
  3. TruncatedSVD 取 50 维
  4. **空间平滑**：在每张切片内部用 kNN(k=8) 对 PCA 坐标做邻域平均，Z = (1-λ)·PC + λ·PC_avg
     （等价于 BANKSY 的"自身 + 邻域"拼接，但不需要把邻域表达矩阵落内存）
  5. 在 Z 上建 kNN 图（k=15）→ Leiden → 空间域

空间邻居**只在切片内部**计算（三张切片坐标系互不相同）。

产出（lncRNA空间域/results/Ldomain/）：
  subset_rows.json      子集细胞的行区间（供后续脚本复现同一批细胞）
  domain_labels.csv     每个细胞的 L-domain / cell_label / sample / 坐标
  domain_summary.csv    域大小与切片构成
  fig_domain_wt.png     域的空间分布与 UMAP
"""
import json
import os
import sys

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

PANEL = (sys.argv[1] if len(sys.argv) > 1 else "ISO").upper()
KEY = "cellxgene_" + PANEL
SUFFIX = "" if PANEL == "ISO" else "_" + PANEL
OUT = r"E:\研究工作\agent长非与空间域\lncRNA空间域\results\Ldomain" + SUFFIX
os.makedirs(OUT, exist_ok=True)

N_PER_SAMPLE = 12000 if PANEL == "WT" else 15000   # WT 有 4 张切片，控制总细胞数以防 OOM
K_SMOOTH = 8
LAMBDA = 0.5
N_HVG = 1500
N_PCS = 50
K_GRAPH = 15
RESOLUTION = 1.0
SEED = 0


def subset_ranges(obs):
    samples = list(obs["sample"].cat.categories)
    arr = obs["sample"].astype(str).to_numpy()
    ranges = []
    for s in samples:
        idx = np.where(arr == str(s))[0]
        lo, n = int(idx.min()), len(idx)
        take = min(N_PER_SAMPLE, n)
        start = lo + (n - take) // 2
        ranges.append({"sample": str(s), "start": start, "stop": start + take})
    return ranges


def main():
    rng = np.random.default_rng(SEED)
    obs = sio.read_obs(KEY)
    var = sio.read_var(KEY)
    ranges = subset_ranges(obs)
    print("子集区间:", ranges)
    json.dump(ranges, open(os.path.join(OUT, "subset_rows.json"), "w"), indent=2)

    blocks, obs_parts = [], []
    for r in ranges:
        blocks.append(sio.read_rows(KEY, r["start"], r["stop"]))
        obs_parts.append(obs.iloc[r["start"]:r["stop"]])
    X = vstack(blocks, format="csr").astype(np.float32)
    o = pd.concat(obs_parts)
    print(f"子集矩阵: {X.shape} | nnz={X.nnz:,}")

    a = ad.AnnData(X=X, obs=o.copy(), var=var.copy())
    a.obs_names_make_unique()

    print("\n[1/5] 归一化 + log1p ...")
    sc.pp.normalize_total(a, target_sum=100.0)
    sc.pp.log1p(a)

    print("[2/5] 选高变基因 ...")
    sc.pp.highly_variable_genes(a, n_top_genes=N_HVG, flavor="seurat")
    a = a[:, a.var["highly_variable"]].copy()
    print(f"  HVG 矩阵: {a.shape}")

    print("[3/5] TruncatedSVD ...")
    from sklearn.decomposition import TruncatedSVD
    svd = TruncatedSVD(n_components=N_PCS, random_state=SEED)
    Z = svd.fit_transform(a.X)                    # 45k × 50
    print(f"  前 5 个成分解释方差比: {np.round(svd.explained_variance_ratio_[:5], 3).tolist()}")

    print("[4/5] 切片内空间平滑 ...")
    from sklearn.neighbors import NearestNeighbors
    sample_arr = a.obs["sample"].astype(str).to_numpy()
    coords = a.obs[["global_x", "global_y"]].to_numpy()
    Zs = Z.copy()
    for s in np.unique(sample_arr):
        m = sample_arr == s
        c = coords[m]
        k = min(K_SMOOTH + 1, len(c))
        nn = NearestNeighbors(n_neighbors=k, algorithm="kd_tree").fit(c)
        _, ind = nn.kneighbors(c)
        ind = ind[:, 1:]
        Zs[m] = (1 - LAMBDA) * Z[m] + LAMBDA * Z[m][ind].mean(axis=1)
        print(f"  sample {s}: {m.sum():,} 细胞，k={k-1}")

    print("[5/5] Leiden 聚类 ...")
    b = ad.AnnData(X=np.zeros((len(Zs), 1), dtype=np.float32), obs=a.obs.copy())
    b.obsm["X_domain"] = Zs.astype(np.float32)
    sc.pp.neighbors(b, use_rep="X_domain", n_neighbors=K_GRAPH, random_state=SEED)
    sc.tl.leiden(b, resolution=RESOLUTION, key_added="L_domain", flavor="igraph",
                 n_iterations=2, directed=False)
    sc.tl.umap(b, random_state=SEED)
    print(f"  域数: {b.obs['L_domain'].nunique()}")

    out = b.obs[["sample", "cell_label", "L_domain", "global_x", "global_y", "annot"]].copy()
    out.index.name = "cell"
    out.to_csv(os.path.join(OUT, "domain_labels.csv"))

    summ = (out.groupby("L_domain", observed=True)
            .agg(n_cells=("sample", "size"),
                 n_samples=("sample", "nunique"),
                 top_celltype=("cell_label", lambda s: s.value_counts().index[0]),
                 celltype_purity=("cell_label", lambda s: s.value_counts(normalize=True).iloc[0])))
    summ["frac"] = summ["n_cells"] / len(out)
    summ = summ.sort_values("n_cells", ascending=False)
    summ.to_csv(os.path.join(OUT, "domain_summary.csv"))
    print("\n=== 域汇总 ===")
    print(summ.head(25).to_string())
    print(f"\n域数 {len(summ)} | 细胞 {len(out):,}")
    print(f"域内细胞类型纯度中位: {summ['celltype_purity'].median():.3f}")
    print(f"跨全部切片出现的域: {int((summ['n_samples'] == out['sample'].nunique()).sum())} / {len(summ)}")

    # 图
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.4))
    doms = out["L_domain"].astype("category").cat.categories
    cmap = plt.cm.tab20(np.linspace(0, 1, len(doms)))
    for i, s in enumerate(sorted(out["sample"].astype(str).unique())):
        ax = axes[i]
        sub = out[out["sample"].astype(str) == s]
        for j, d in enumerate(doms):
            m = sub["L_domain"] == d
            if m.sum() == 0:
                continue
            ax.scatter(sub.loc[m, "global_x"], sub.loc[m, "global_y"], s=0.5,
                       color=cmap[j], linewidths=0, rasterized=True)
        ax.set_title(f"sample {s}: L-domain", fontsize=9)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
    ax = axes[3]
    for j, d in enumerate(doms):
        m = out["L_domain"] == d
        ax.scatter(b.obsm["X_umap"][m, 0], b.obsm["X_umap"][m, 1], s=0.5,
                   color=cmap[j], linewidths=0, rasterized=True)
    ax.set_title("UMAP (spatially smoothed)", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_domain_wt.png"), dpi=150)
    plt.close(fig)

    print(f"\n=== L-domain 完成，输出: {OUT} ===")


if __name__ == "__main__":
    main()
