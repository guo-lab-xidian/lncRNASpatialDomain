# -*- coding: utf-8 -*-
"""
Visium HD（8 μm bin）上的 lncRNA 域特异性验证。

8 μm bin 接近单细胞尺度，且为全转录组探针。
因总量达 39 万个 bin、7,900 万非零元，本机内存有限，采取：
  1. 载入后先做 QC 过滤
  2. 随机抽样 N_BINS 个 bin 做域聚类与比较（保留空间结构）

产出：lncRNA空间域/results/VisiumHD/
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import scipy.sparse as sp
from scipy.sparse import csc_matrix
from scipy import stats
import h5py

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

HD = r"E:\研究工作\agent长非与空间域\data\visium\binned_outputs\square_008um"
COORD = r"E:\研究工作\agent长非与空间域\lncRNA空间域\results\gene_coords_m25.tsv"
CLS = r"E:\研究工作\agent长非与空间域\analysis\03_lncrna\panel_features_class.csv"
OUT = r"E:\研究工作\agent长非与空间域\lncRNA空间域\results\VisiumHD"
os.makedirs(OUT, exist_ok=True)

N_BINS = 60000
SEED = 0
N_HVG = 1500
K_SMOOTH = 6
LAMBDA = 0.5
MIN_DETECT = 0.01


def load_hd():
    p = os.path.join(HD, "filtered_feature_bc_matrix.h5")
    with h5py.File(p, "r") as f:
        m = f["matrix"]
        shape = tuple(m["shape"][:])
        data = m["data"][:]
        indices = m["indices"][:]
        indptr = m["indptr"][:]
        names = [x.decode() for x in m["features"]["name"][:]]
        barcodes = [x.decode() for x in m["barcodes"][:]]
    X = csc_matrix((data, indices, indptr), shape=shape).T.tocsr()  # bins × genes
    return X, names, barcodes


def main():
    rng = np.random.default_rng(SEED)
    print("加载 Visium HD 8um ...")
    X, genes, barcodes = load_hd()
    print(f"  原始: {X.shape[0]:,} bins × {X.shape[1]:,} genes | nnz={X.nnz:,}")

    pos = pd.read_parquet(os.path.join(HD, "spatial", "tissue_positions.parquet"))
    print(f"  位置表: {pos.shape} 列={list(pos.columns)[:8]}")
    if "barcode" in pos.columns:
        pos = pos.set_index("barcode")
    pos.index = pos.index.astype(str)
    bdf = pd.DataFrame({"barcode": barcodes}).set_index("barcode")
    meta = bdf.join(pos, how="left")
    if "in_tissue" in meta.columns:
        keep_tissue = meta["in_tissue"].fillna(0).astype(int).to_numpy() == 1
    else:
        keep_tissue = np.ones(len(meta), dtype=bool)
    print(f"  组织内 bin: {keep_tissue.sum():,}")

    n_counts = np.asarray(X.sum(axis=1)).ravel()
    ok = keep_tissue & (n_counts >= 20)
    print(f"  计数≥20 且组织内: {ok.sum():,}")

    idx = np.where(ok)[0]
    if len(idx) > N_BINS:
        idx = np.sort(rng.choice(idx, N_BINS, replace=False))
    Xs = X[idx]
    meta = meta.iloc[idx].copy()
    print(f"  分析子集: {Xs.shape[0]:,} bins | nnz={Xs.nnz:,}")
    del X

    a = ad.AnnData(X=Xs.astype(np.float32), obs=meta.copy(),
                   var=pd.DataFrame(index=genes))
    a.var_names_make_unique()
    a.obs_names_make_unique()

    print("归一化 + 聚类 ...")
    sc.pp.filter_genes(a, min_cells=10)
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    sc.pp.highly_variable_genes(a, n_top_genes=N_HVG, flavor="seurat")
    hv = a.var["highly_variable"].to_numpy()
    sub = a[:, hv].copy()
    sc.pp.scale(sub, max_value=10)
    from sklearn.decomposition import PCA
    n_comp = min(30, sub.n_vars - 1, sub.n_obs - 1)
    pcs = PCA(n_components=n_comp, random_state=SEED).fit_transform(np.asarray(sub.X))

    from sklearn.neighbors import NearestNeighbors
    coord_cols = [c for c in ["pxl_row_in_fullres", "pxl_col_in_fullres"] if c in a.obs.columns]
    coords = a.obs[coord_cols].to_numpy(dtype=float)
    nn = NearestNeighbors(n_neighbors=K_SMOOTH + 1).fit(coords)
    _, ind = nn.kneighbors(coords)
    ind = ind[:, 1:]
    Zs = (1 - LAMBDA) * pcs + LAMBDA * pcs[ind].mean(axis=1)

    b = ad.AnnData(X=np.zeros((len(Zs), 1), dtype=np.float32), obs=a.obs.copy())
    b.obsm["X_dom"] = Zs
    sc.pp.neighbors(b, use_rep="X_dom", n_neighbors=15, random_state=SEED)
    sc.tl.leiden(b, resolution=1.0, key_added="L_domain", flavor="igraph",
                 n_iterations=2, directed=False)
    dom_codes, dom_names = pd.factorize(b.obs["L_domain"].astype(str).to_numpy())
    K = len(dom_names)
    print(f"  空间域: {K} 个")

    G = sp.csr_matrix((np.ones(a.n_obs), (np.arange(a.n_obs), dom_codes)),
                      shape=(a.n_obs, K))
    n_k = np.asarray(G.sum(axis=0)).ravel()
    Xn = a.X.tocsr()
    dom_mean = np.asarray((G.T @ Xn).todense()) / n_k[:, None]
    dsi = dom_mean.max(axis=0) - np.median(dom_mean, axis=0)
    det = np.asarray((Xn > 0).mean(axis=0)).ravel()

    st = pd.DataFrame({"gene": a.var_names, "detect_frac": det, "DSI": dsi}).set_index("gene")
    st.to_csv(os.path.join(OUT, "gene_specificity.csv"))

    pf = pd.read_csv(CLS, index_col=0)
    pf = pf[~pf.index.duplicated(keep="first")]
    cls = pf["gene_class"].reindex(st.index)
    coords_df = pd.read_csv(COORD, sep="\t")
    name2type = coords_df.drop_duplicates("gene_name").set_index("gene_name")["gene_type"]
    LNC = {"lncRNA", "lincRNA", "antisense", "processed_transcript",
           "sense_intronic", "sense_overlapping", "macro_lncRNA"}
    fb = name2type.reindex(st.index).map(
        lambda t: "lncRNA" if isinstance(t, str) and t in LNC
        else ("protein_coding" if t == "protein_coding" else "other"))
    st["gene_class"] = cls.fillna(fb)

    lnc = st[(st["gene_class"] == "lncRNA") & (st["detect_frac"] >= MIN_DETECT)]
    pc = st[(st["gene_class"] == "protein_coding") & (st["detect_frac"] >= MIN_DETECT)]
    print(f"  lncRNA {len(lnc):,} | 蛋白编码 {len(pc):,}")

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
    print("=== Visium HD 8um：lncRNA vs 匹配蛋白编码 ===")
    print(f"  lncRNA 数: {len(d):,}")
    print(f"  DSI 差值中位: {d.median():+.4f} | 为正比例 {(d>0).mean():.1%}")
    print(f"  Wilcoxon p = {w.pvalue:.3e}")
    print(f"  lncRNA DSI 中位 {comp['DSI_lnc'].median():.4f} vs 对照 {comp['DSI_ctrl'].median():.4f}")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    ax[0].hist(comp["DSI_ctrl"].dropna(), bins=40, alpha=0.6, color="#4C78A8",
               label="matched protein-coding")
    ax[0].hist(comp["DSI_lnc"].dropna(), bins=40, alpha=0.7, color="#D1495B", label="lncRNA")
    ax[0].set_xlabel("DSI")
    ax[0].legend(fontsize=8)
    ax[0].set_title("Visium HD 8um")
    ax[1].hist(d, bins=40, color="#59A14F")
    ax[1].axvline(0, color="grey", ls="--", lw=1)
    ax[1].set_xlabel("DSI(lncRNA) − DSI(matched control)")
    ax[1].set_title(f"p={w.pvalue:.2e}")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_hd_dsi.png"), dpi=150)
    plt.close(fig)
    print(f"\n=== 完成，输出: {OUT} ===")


if __name__ == "__main__":
    main()
