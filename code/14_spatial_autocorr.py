# -*- coding: utf-8 -*-
"""
课题三 / H1b：空间自相关（Moran's I）对比——不依赖域标签的独立指标。

对 lncRNA 及其检出率匹配的蛋白编码对照，计算二值检出模式的空间自相关：
  在每个样本内部构造 kNN(k=8) 空间权重 W（按行列和标准化），
  I = (n / S0) * (x^T W x) / (x^T x)，x 为该基因的检出指示向量（已中心化）。
三张切片分别计算后按细胞数加权平均。

产出（lncRNA空间域/results/H1H2/）：
  moransI.csv        每个基因的 I 与匹配对照比较
  fig_moransI.png
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy import stats
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import spatial_io as sio

PANEL = (sys.argv[1] if len(sys.argv) > 1 else "ISO").upper()
SUFFIX = "" if PANEL == "ISO" else "_" + PANEL
KEY = "cellxgene_" + PANEL
RES = r"E:\研究工作\agent长非与空间域\lncRNA空间域\results"
LDIR = os.path.join(RES, "Ldomain" + SUFFIX)
SPEC = os.path.join(RES, "H1H2" + SUFFIX, "gene_specificity.csv")
OUT = os.path.join(RES, "H1H2" + SUFFIX)
LNC_MIN_DETECT = 0.05
K = 8
SEED = 0


def build_W(coords, k=K):
    n = len(coords)
    kk = min(k + 1, n)
    nn = NearestNeighbors(n_neighbors=kk, algorithm="kd_tree").fit(coords)
    _, ind = nn.kneighbors(coords)
    ind = ind[:, 1:]
    rows = np.repeat(np.arange(n), ind.shape[1])
    cols = ind.ravel()
    W = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    W = W + W.T
    d = np.asarray(W.sum(axis=1)).ravel()
    d[d == 0] = 1
    from scipy.sparse import diags
    return diags(1.0 / d) @ W          # 行标准化


def main():
    rng = np.random.default_rng(SEED)
    spec = pd.read_csv(SPEC, index_col=0)
    lab = pd.read_csv(os.path.join(LDIR, "domain_labels.csv"), index_col=0)
    ranges = json.load(open(os.path.join(LDIR, "subset_rows.json")))

    lnc = spec[(spec["gene_class"] == "lncRNA") & (spec["panel_detect"] >= LNC_MIN_DETECT)]
    pc = spec[(spec["gene_class"] == "protein_coding") & (spec["panel_detect"] >= LNC_MIN_DETECT)]

    # 目标基因：全部 lncRNA + 其匹配对照
    matched = {}
    for r in lnc.itertuples():
        sel = pc[(pc["panel_detect"] >= r.panel_detect * 0.8) &
                 (pc["panel_detect"] <= r.panel_detect * 1.2)]
        if len(sel) < 5:
            sel = pc.iloc[np.argsort(np.abs(pc["panel_detect"] - r.panel_detect))[:5]]
        matched[r.Index] = list(sel.sample(min(5, len(sel)),
                                           random_state=int(rng.integers(1e9))).index)
    genes = list(lnc.index) + list(dict.fromkeys([g for v in matched.values() for g in v]))
    print(f"lncRNA {len(lnc)} | 匹配对照 {len(genes) - len(lnc)} | 合计 {len(genes)}")

    var = sio.read_var(KEY)
    pos = var.index.get_indexer(genes)
    assert (pos >= 0).all()

    sample_arr = lab["sample"].astype(str).to_numpy()
    coords_all = lab[["global_x", "global_y"]].to_numpy()

    I_sum = np.zeros(len(genes))
    I_w = np.zeros(len(genes))
    n_by_sample = {}
    for r in ranges:
        s = r["sample"]
        m = sample_arr == s
        n = int(r["stop"] - r["start"])
        assert m.sum() == n, f"样本 {s} 细胞数与区间不符"
        idx = np.where(m)[0]
        X = sio.read_rows(KEY, r["start"], r["stop"])[:, pos].astype(np.float32)
        Xb = (X > 0).astype(np.float32).toarray()
        coords = coords_all[idx]
        W = build_W(coords)
        S0 = W.sum()
        Xc = Xb - Xb.mean(axis=0, keepdims=True)
        num = np.einsum("ij,ij->j", Xc, W @ Xc)
        den = np.einsum("ij,ij->j", Xc, Xc)
        with np.errstate(divide="ignore", invalid="ignore"):
            I = (n / S0) * np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0)
        I_sum += np.nan_to_num(I) * n
        I_w += (np.isfinite(I)).astype(float) * n
        n_by_sample[s] = n
        print(f"  sample {s}: {n:,} 细胞，I 中位 {np.nanmedian(I):.4f}", flush=True)

    I_mean = np.where(I_w > 0, I_sum / np.maximum(I_w, 1), np.nan)
    res = pd.DataFrame({"gene": genes, "moransI": I_mean}).set_index("gene")
    res["gene_class"] = spec.loc[genes, "gene_class"]
    res["panel_detect"] = spec.loc[genes, "panel_detect"]
    res["DSI_domain"] = spec.loc[genes, "DSI_domain"]
    res["DSI_cell"] = spec.loc[genes, "DSI_cell"]
    res.to_csv(os.path.join(OUT, "moransI.csv"))

    rows = []
    for g in lnc.index:
        ctrl = matched[g]
        rows.append({"gene": g, "gene_name": lnc.loc[g, "gene_name"],
                     "detect": lnc.loc[g, "panel_detect"],
                     "I_lnc": res.loc[g, "moransI"],
                     "I_ctrl": res.loc[ctrl, "moransI"].median()})
    df = pd.DataFrame(rows)
    df["delta"] = df["I_lnc"] - df["I_ctrl"]

    print("\n=== Moran's I 对比 ===")
    print(f"  lncRNA I 中位: {df['I_lnc'].median():.4f}")
    print(f"  匹配对照 I 中位: {df['I_ctrl'].median():.4f}")
    print(f"  delta 中位: {df['delta'].median():+.4f} | 为正 {int((df['delta'] > 0).sum())}/{len(df)}")
    w = stats.wilcoxon(df["delta"].dropna())
    print(f"  Wilcoxon p = {w.pvalue:.3e}")
    print("\n  delta 最大的 12 个:")
    print(df.nlargest(12, "delta")[["gene_name", "detect", "I_lnc", "I_ctrl", "delta"]]
          .round(4).to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    axes[0].hist(df["I_ctrl"].dropna(), bins=30, alpha=0.6, color="#4C78A8", label="matched control")
    axes[0].hist(df["I_lnc"].dropna(), bins=30, alpha=0.7, color="#D1495B", label="lncRNA")
    axes[0].set_xlabel("Moran's I (binarized detection)")
    axes[0].set_ylabel("genes")
    axes[0].legend(fontsize=8)
    axes[0].set_title("Spatial autocorrelation")
    axes[1].hist(df["delta"].dropna(), bins=30, color="#59A14F")
    axes[1].axvline(0, color="grey", ls="--", lw=1)
    axes[1].set_xlabel("I(lncRNA) − median I(matched control)")
    axes[1].set_ylabel("lncRNAs")
    axes[1].set_title("Paired difference")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_moransI.png"), dpi=150)
    plt.close(fig)
    print(f"\n=== 完成，输出: {OUT} ===")


if __name__ == "__main__":
    main()
