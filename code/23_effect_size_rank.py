# -*- coding: utf-8 -*-
"""
以跨面板可重复性评价效应量估计方法。

背景（实测）：真实数据中每个基因都"显著"——45,000 细胞下，脑区异质性让
"检出率与域无关"这个零假设对任何基因都不成立。因此 p 值无法区分优劣，
必须比较**效应量的排序可靠性**。

评价方案：两套独立设计的探针面板（WT / ISO）互为外部验证，
比较各方法给出的域特异性排序在两面版之间的 Spearman 相关。

比较的方法：
  1. 原始 DSI = max_k f_k − median_k f_k
  2. 收缩 DSI（固定 κ 的伪计数平滑）
  3. Pearson 卡方统计量
  4. 域间检出率的变异系数 CV

产出：results/Method/rank_compare.csv、fig_rank_compare.png
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import h5py
from scipy.sparse import csr_matrix

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import spatial_io as sio
import bb_dsi as bb

RES = r"E:\研究工作\agent长非与空间域\lncRNA空间域\results"
CLS = r"E:\研究工作\agent长非与空间域\analysis\03_lncrna\panel_features_class.csv"
OUT = os.path.join(RES, "Method")
os.makedirs(OUT, exist_ok=True)

BLOCK = 15000
KAPPA = 100.0
MIN_DETECT = 0.01


def counts_by_domain(panel):
    """流式累积每个域内每个特征的检出细胞数（K × n_features）。"""
    suffix = "" if panel == "ISO" else "_" + panel
    ldir = os.path.join(RES, "Ldomain" + suffix)
    lab = pd.read_csv(os.path.join(ldir, "domain_labels.csv"), index_col=0)
    ranges = json.load(open(os.path.join(ldir, "subset_rows.json")))
    dom_codes, dom_names = pd.factorize(lab["L_domain"].astype(str))
    K = len(dom_names)
    n_k = np.bincount(dom_codes, minlength=K)

    path = sio._resolve("cellxgene_" + panel)
    with h5py.File(path, "r") as f:
        n_var = f["var/_index"].shape[0]
        indptr = f["X/indptr"][:]
        dset, iset = f["X/data"], f["X/indices"]
        cnt = np.zeros((K, n_var), dtype=np.int32)
        off = 0
        for r in ranges:
            for start in range(r["start"], r["stop"], BLOCK):
                end = min(start + BLOCK, r["stop"])
                lo, hi = int(indptr[start]), int(indptr[end])
                blk = csr_matrix(((dset[lo:hi] > 0).astype(np.float32), iset[lo:hi],
                                  indptr[start:end + 1] - lo),
                                 shape=(end - start, n_var))
                m = end - start
                gd = dom_codes[off:off + m]
                G = csr_matrix((np.ones(m, dtype=np.float32), (np.arange(m), gd)),
                               shape=(m, K))
                cnt += np.asarray((G.T @ blk).todense()).astype(np.int32)
                off += m
    print(f"  {panel}: 累积完成 {cnt.shape}，细胞 {off:,}")
    return cnt, n_k, off


def compute_stats(cnt, n_k, n_cells):
    """由域×特征计数矩阵计算各效应量。"""
    n_k_f = n_k.astype(float)[:, None]
    f = cnt / n_k_f
    total = cnt.sum(axis=0)
    det_frac = total / n_cells

    dsi_raw = f.max(axis=0) - np.median(f, axis=0)

    # 收缩 DSI（固定 κ）
    p = np.clip(det_frac, 1e-9, 1 - 1e-9)
    post = (cnt + p[None, :] * KAPPA) / (n_k_f + KAPPA)
    dsi_shrunk = post.max(axis=0) - np.median(post, axis=0)

    # Pearson 卡方统计量
    mu = n_k_f * p[None, :]
    chi2 = np.where(mu > 0, (cnt - mu) ** 2 / mu, 0).sum(axis=0)
    chi2 += np.where((n_k_f - mu) > 0,
                     ((n_k_f - cnt) - (n_k_f - mu)) ** 2 / (n_k_f - mu), 0).sum(axis=0)

    # 域间变异系数
    cv = f.std(axis=0) / np.maximum(f.mean(axis=0), 1e-12)

    return pd.DataFrame({
        "detect_frac": det_frac,
        "dsi_raw": dsi_raw,
        "dsi_shrunk": dsi_shrunk,
        "chi2": chi2,
        "cv": cv,
    })


def main():
    frames = {}
    for panel in ["ISO", "WT"]:
        print(f"=== {panel} ===")
        cnt, n_k, n_cells = counts_by_domain(panel)
        var = sio.read_var("cellxgene_" + panel)
        st = compute_stats(cnt, n_k, n_cells)
        st.index = var.index
        frames[panel] = st
        st.to_csv(os.path.join(OUT, f"effectsize_{panel}.csv"))

    # 注释表含 ISO 与 WT 两份同名特征，按特征去重
    spec = pd.read_csv(CLS, index_col=0)
    spec = spec[~spec.index.duplicated(keep="first")]
    # 只比较在两套面板中检出率都 ≥1% 的基因（保证可比）
    common = frames["ISO"].join(frames["WT"], lsuffix="_ISO", rsuffix="_WT", how="inner")
    common = common[(common["detect_frac_ISO"] >= MIN_DETECT) &
                    (common["detect_frac_WT"] >= MIN_DETECT)]
    common["gene_class"] = spec["gene_class"].reindex(common.index)
    common["gene_name"] = spec["gene_name"].reindex(common.index)
    print(f"\n两面板共同分析基因（检出率均≥{MIN_DETECT}）: {len(common):,}")

    rows = []
    for grp, sub in [("全部", common),
                     ("lncRNA", common[common["gene_class"] == "lncRNA"]),
                     ("protein_coding", common[common["gene_class"] == "protein_coding"])]:
        if len(sub) < 10:
            continue
        for m in ["dsi_raw", "dsi_shrunk", "chi2", "cv"]:
            r = stats.spearmanr(sub[f"{m}_ISO"], sub[f"{m}_WT"]).statistic
            rows.append({"group": grp, "method": m, "n": len(sub),
                         "spearman_cross_panel": r})
    cmp = pd.DataFrame(rows)
    cmp.to_csv(os.path.join(OUT, "rank_compare.csv"), index=False)
    print("\n=== 跨面板排序一致性（Spearman ρ，越高越好）===")
    print(cmp.pivot(index="method", columns="group", values="spearman_cross_panel")
          .round(4).to_string())

    # 只在高检出率子集上：方法差异应更明显
    hi = common[(common["detect_frac_ISO"] >= 0.05) & (common["detect_frac_WT"] >= 0.05)]
    print(f"\n高检出率子集（两面板均≥5%）: {len(hi):,}")
    rows = []
    for grp, sub in [("全部", hi),
                     ("lncRNA", hi[hi["gene_class"] == "lncRNA"]),
                     ("protein_coding", hi[hi["gene_class"] == "protein_coding"])]:
        if len(sub) < 10:
            continue
        for m in ["dsi_raw", "dsi_shrunk", "chi2", "cv"]:
            r = stats.spearmanr(sub[f"{m}_ISO"], sub[f"{m}_WT"]).statistic
            rows.append({"group": grp, "method": m, "n": len(sub),
                         "spearman_cross_panel": r})
    cmp2 = pd.DataFrame(rows)
    cmp2.to_csv(os.path.join(OUT, "rank_compare_hidetect.csv"), index=False)
    print(cmp2.pivot(index="method", columns="group", values="spearman_cross_panel")
          .round(4).to_string())

    # 图
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, df, title in [(axes[0], cmp, "all genes (detect≥1%)"),
                          (axes[1], cmp2, "high-detection genes (≥5%)")]:
        piv = df.pivot(index="method", columns="group", values="spearman_cross_panel")
        piv.plot(kind="bar", ax=ax, rot=0)
        ax.set_ylabel("cross-panel Spearman ρ")
        ax.set_title(title)
        ax.legend(fontsize=7)
        ax.set_xlabel("")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_rank_compare.png"), dpi=150)
    plt.close(fig)
    print(f"\n=== 完成，输出: {OUT} ===")


if __name__ == "__main__":
    main()
