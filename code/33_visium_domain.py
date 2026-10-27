# -*- coding: utf-8 -*-
"""
在 10x Visium（全转录组）上检验"面板选择偏差"假说。

核心问题：在没有预设探针限制的全转录组数据里，
         lncRNA 的域特异性是否高于检出率匹配的蛋白编码基因？

与 MERFISH 分析的差别：
  - Visium 每 spot 中位检出 5,451 个基因（MERFISH 仅 592），信噪比高得多
  - 因此可以用归一化表达量而非二值检出

产出：lncRNA空间域/results/Visium/
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import scipy.sparse as sp
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CACHE = r"E:\研究工作\agent长非与空间域\data\visium\cache"
COORD = r"E:\研究工作\agent长非与空间域\lncRNA空间域\results\gene_coords_m25.tsv"
CLS = r"E:\研究工作\agent长非与空间域\analysis\03_lncrna\panel_features_class.csv"
OUT = r"E:\研究工作\agent长非与空间域\lncRNA空间域\results\Visium"
os.makedirs(OUT, exist_ok=True)

SEED = 0
N_HVG = 2000
K_SMOOTH = 6
LAMBDA = 0.5
RESOLUTION = 1.0
MIN_DETECT = 0.01


def main():
    rng = np.random.default_rng(SEED)
    X = sp.load_npz(os.path.join(CACHE, "visium_counts.npz"))
    meta = pd.read_parquet(os.path.join(CACHE, "visium_meta.parquet"))
    genes = pd.read_csv(os.path.join(CACHE, "visium_genes.txt"), header=None)[0].tolist()
    print(f"加载: {X.shape[0]:,} spots × {X.shape[1]:,} genes")

    a = ad.AnnData(X=X.astype(np.float32),
                   obs=meta.set_index("barcode").copy(),
                   var=pd.DataFrame(index=genes))
    a.var_names_make_unique()
    a.obs_names_make_unique()

    print("[1/4] QC ...")
    sc.pp.filter_cells(a, min_counts=500)
    sc.pp.filter_genes(a, min_cells=10)
    print(f"  过滤后: {a.shape}")

    print("[2/4] 归一化 + 空间域聚类 ...")
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    sc.pp.highly_variable_genes(a, n_top_genes=N_HVG, flavor="seurat")
    hv = a.var["highly_variable"].to_numpy()
    sub = a[:, hv].copy()
    sc.pp.scale(sub, max_value=10)          # 原地操作
    Zmat = np.asarray(sub.X)
    from sklearn.decomposition import PCA
    pcs = PCA(n_components=30, random_state=SEED).fit_transform(Zmat)

    # 切片内空间平滑（用像素坐标）
    from sklearn.neighbors import NearestNeighbors
    coords = a.obs[["pxl_row_in_fullres", "pxl_col_in_fullres"]].to_numpy()
    sample_arr = a.obs["sample"].astype(str).to_numpy()
    Zs = pcs.copy()
    for s in np.unique(sample_arr):
        m = sample_arr == s
        k = min(K_SMOOTH + 1, m.sum())
        nn = NearestNeighbors(n_neighbors=k).fit(coords[m])
        _, ind = nn.kneighbors(coords[m])
        ind = ind[:, 1:]
        Zs[m] = (1 - LAMBDA) * pcs[m] + LAMBDA * pcs[m][ind].mean(axis=1)
        print(f"  {s}: {m.sum()} spots，k={k-1}")

    b = ad.AnnData(X=np.zeros((len(Zs), 1), dtype=np.float32), obs=a.obs.copy())
    b.obsm["X_dom"] = Zs
    sc.pp.neighbors(b, use_rep="X_dom", n_neighbors=15, random_state=SEED)
    sc.tl.leiden(b, resolution=RESOLUTION, key_added="L_domain", flavor="igraph",
                 n_iterations=2, directed=False)
    dom = b.obs["L_domain"].astype(str).to_numpy()
    dom_codes, dom_names = pd.factorize(dom)
    K = len(dom_names)
    print(f"  空间域: {K} 个")
    print(pd.Series(dom).value_counts().to_string())

    meta_out = b.obs[["sample", "L_domain", "pxl_row_in_fullres", "pxl_col_in_fullres"]].copy()
    meta_out.to_csv(os.path.join(OUT, "domain_labels.csv"))

    print("[3/4] 计算域特异性 ...")
    G = sp.csr_matrix((np.ones(a.n_obs), (np.arange(a.n_obs), dom_codes)),
                      shape=(a.n_obs, K))
    n_k = np.asarray(G.sum(axis=0)).ravel()
    Xn = a.X.tocsr()
    dom_sum = np.asarray((G.T @ Xn).todense())        # K × genes（求和）
    dom_mean = dom_sum / n_k[:, None]                  # 域内均值
    dsi = dom_mean.max(axis=0) - np.median(dom_mean, axis=0)

    det_rate = np.asarray((Xn > 0).mean(axis=0)).ravel()
    mean_expr = np.asarray(Xn.mean(axis=0)).ravel()

    st = pd.DataFrame({
        "gene": a.var_names,
        "detect_frac": det_rate,
        "mean_expr": mean_expr,
        "DSI": dsi,
        "peak_domain": [dom_names[i] for i in dom_mean.argmax(axis=0)],
    }).set_index("gene")
    st.to_csv(os.path.join(OUT, "gene_specificity.csv"))
    print(f"  基因数: {len(st):,}")

    print("[4/4] 注释 lncRNA 并做匹配比较 ...")
    pf = pd.read_csv(CLS, index_col=0)
    pf = pf[~pf.index.duplicated(keep="first")]
    cls = pf["gene_class"].reindex(st.index)
    # 未在本面板注释表里的基因，用 GENCODE M25 补
    coords_df = pd.read_csv(COORD, sep="\t")
    name2type = coords_df.drop_duplicates("gene_name").set_index("gene_name")["gene_type"]
    LNC_TYPES = {"lncRNA", "lincRNA", "antisense", "processed_transcript",
                 "sense_intronic", "sense_overlapping", "macro_lncRNA"}
    fallback = name2type.reindex(st.index).map(
        lambda t: "lncRNA" if isinstance(t, str) and t in LNC_TYPES
        else ("protein_coding" if t == "protein_coding" else "other"))
    st["gene_class"] = cls.fillna(fallback)

    lnc = st[(st["gene_class"] == "lncRNA") & (st["detect_frac"] >= MIN_DETECT)]
    pc = st[(st["gene_class"] == "protein_coding") & (st["detect_frac"] >= MIN_DETECT)]
    print(f"  lncRNA（检出率≥{MIN_DETECT}）: {len(lnc):,} | 蛋白编码: {len(pc):,}")

    rows = []
    for g, r in lnc.iterrows():
        sel = pc[(pc["detect_frac"] >= r.detect_frac * 0.8) &
                 (pc["detect_frac"] <= r.detect_frac * 1.2)]
        if len(sel) < 5:
            sel = pc.iloc[np.argsort(np.abs(pc["detect_frac"] - r.detect_frac))[:5]]
        ctrl = sel.sample(min(5, len(sel)), random_state=int(rng.integers(1e9)))
        rows.append({"gene": g, "detect": r.detect_frac, "DSI_lnc": r.DSI,
                     "DSI_ctrl": ctrl["DSI"].median()})
    comp = pd.DataFrame(rows)
    comp["delta"] = comp["DSI_lnc"] - comp["DSI_ctrl"]
    comp.to_csv(os.path.join(OUT, "matched_comparison.csv"))

    d = comp["delta"].dropna()
    w = stats.wilcoxon(d)
    print("\n" + "=" * 70)
    print("=== Visium 全转录组：lncRNA vs 匹配蛋白编码 ===")
    print(f"  lncRNA 数: {len(d):,}")
    print(f"  DSI 差值中位: {d.median():+.4f} | 均值: {d.mean():+.4f}")
    print(f"  为正比例: {int((d > 0).sum())}/{len(d)} = {(d > 0).mean():.1%}")
    print(f"  Wilcoxon p = {w.pvalue:.3e}")
    print(f"  lncRNA DSI 中位 {comp['DSI_lnc'].median():.4f} vs 对照 {comp['DSI_ctrl'].median():.4f}")

    tab = st.groupby("gene_class").agg(
        n=("DSI", "size"), DSI_median=("DSI", "median"),
        detect_median=("detect_frac", "median"))
    print("\n=== 按类别 ===")
    print(tab.round(4).to_string())
    tab.to_csv(os.path.join(OUT, "class_summary.csv"))

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    ax[0].hist(comp["DSI_ctrl"].dropna(), bins=40, alpha=0.6, color="#4C78A8",
               label="matched protein-coding")
    ax[0].hist(comp["DSI_lnc"].dropna(), bins=40, alpha=0.7, color="#D1495B", label="lncRNA")
    ax[0].set_xlabel("DSI (domain specificity)")
    ax[0].set_ylabel("genes")
    ax[0].legend(fontsize=8)
    ax[0].set_title("Visium whole transcriptome")
    ax[1].hist(d, bins=40, color="#59A14F")
    ax[1].axvline(0, color="grey", ls="--", lw=1)
    ax[1].set_xlabel("DSI(lncRNA) − DSI(matched control)")
    ax[1].set_ylabel("lncRNAs")
    ax[1].set_title(f"paired difference (p={w.pvalue:.2e})")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_visium_dsi.png"), dpi=150)
    plt.close(fig)
    print(f"\n=== 完成，输出: {OUT} ===")


if __name__ == "__main__":
    main()
