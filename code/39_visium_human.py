# -*- coding: utf-8 -*-
"""
人类脑 10x Visium：跨物种验证 lncRNA 的域特异性。

数据：10x V1 Human Brain Section 1 / 2（全转录组，polyA 捕获）
注释：Ensembl 人类基因 biotype（BioMart）

产出：lncRNA空间域/results/VisiumHuman/
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
from scipy.sparse import csc_matrix
from scipy import stats
import h5py

DIR = r"E:\研究工作\agent长非与空间域\data\visium_human"
ANN = r"E:\研究工作\agent长非与空间域\data\ensembl_human_gene.tsv"
OUT = r"E:\研究工作\agent长非与空间域\lncRNA空间域\results\VisiumHuman"
os.makedirs(OUT, exist_ok=True)

SEED = 0
N_HVG = 1500
K_SMOOTH = 6
LAMBDA = 0.5
MIN_DETECT = 0.01
SAMPLES = [("S1", "S1.h5", "spatial_s1"), ("S2", "S2.h5", "spatial_s2")]


def load_h5(path):
    with h5py.File(path, "r") as f:
        m = f["matrix"]
        shape = tuple(m["shape"][:])
        data = m["data"][:]
        indices = m["indices"][:]
        indptr = m["indptr"][:]
        names = [x.decode() for x in m["features"]["name"][:]]
        bcs = [x.decode() for x in m["barcodes"][:]]
    X = csc_matrix((data, indices, indptr), shape=shape).T.tocsr()
    return X, names, bcs


def load_positions(d):
    p = os.path.join(d, "tissue_positions_list.csv")
    df = pd.read_csv(p, header=None,
                     names=["barcode", "in_tissue", "array_row", "array_col",
                            "pxl_row_in_fullres", "pxl_col_in_fullres"])
    return df.set_index("barcode")


def main():
    rng = np.random.default_rng(SEED)
    mats, metas, genes = [], [], None
    for name, h5, spat in SAMPLES:
        X, names, bcs = load_h5(os.path.join(DIR, h5))
        pos = load_positions(os.path.join(DIR, spat))
        if genes is None:
            genes = names
        else:
            assert genes == names
        it = pos.reindex(bcs)["in_tissue"].fillna(0).astype(int).to_numpy()
        idx = np.where(it == 1)[0]
        print(f"{name}: {len(bcs):,} spots（组织内 {len(idx):,}）")
        mats.append(X[idx])
        m = pos.reindex([bcs[i] for i in idx]).copy()
        m["sample"] = name
        m["barcode"] = m.index
        metas.append(m)
    from scipy.sparse import vstack
    Xall = vstack(mats, format="csr")
    meta = pd.concat(metas)
    print(f"合并: {Xall.shape[0]:,} spots × {Xall.shape[1]:,} genes | nnz={Xall.nnz:,}")

    a = ad.AnnData(X=Xall.astype(np.float32), obs=meta.copy(),
                   var=pd.DataFrame(index=genes))
    a.var_names_make_unique()
    a.obs_names_make_unique()
    sc.pp.filter_cells(a, min_counts=500)
    sc.pp.filter_genes(a, min_cells=10)
    print(f"QC 后: {a.shape}")

    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    sc.pp.highly_variable_genes(a, n_top_genes=N_HVG, flavor="seurat")
    hv = a.var["highly_variable"].to_numpy()
    sub = a[:, hv].copy()
    sc.pp.scale(sub, max_value=10)
    from sklearn.decomposition import PCA
    pcs = PCA(n_components=min(30, sub.n_vars - 1), random_state=SEED).fit_transform(np.asarray(sub.X))

    from sklearn.neighbors import NearestNeighbors
    coords = a.obs[["pxl_row_in_fullres", "pxl_col_in_fullres"]].to_numpy(dtype=float)
    sample_arr = a.obs["sample"].astype(str).to_numpy()
    Zs = pcs.copy()
    for s in np.unique(sample_arr):
        m = sample_arr == s
        nn = NearestNeighbors(n_neighbors=K_SMOOTH + 1).fit(coords[m])
        _, ind = nn.kneighbors(coords[m])
        Zs[m] = (1 - LAMBDA) * pcs[m] + LAMBDA * pcs[m][ind[:, 1:]].mean(axis=1)

    b = ad.AnnData(X=np.zeros((len(Zs), 1), dtype=np.float32), obs=a.obs.copy())
    b.obsm["X_dom"] = Zs
    sc.pp.neighbors(b, use_rep="X_dom", n_neighbors=15, random_state=SEED)
    sc.tl.leiden(b, resolution=1.0, key_added="L_domain", flavor="igraph",
                 n_iterations=2, directed=False)
    dom_codes, dom_names = pd.factorize(b.obs["L_domain"].astype(str).to_numpy())
    K = len(dom_names)
    print(f"空间域: {K} 个")

    G = sp.csr_matrix((np.ones(a.n_obs), (np.arange(a.n_obs), dom_codes)),
                      shape=(a.n_obs, K))
    n_k = np.asarray(G.sum(axis=0)).ravel()
    dom_mean = np.asarray((G.T @ a.X.tocsr()).todense()) / n_k[:, None]
    dsi = dom_mean.max(axis=0) - np.median(dom_mean, axis=0)
    det = np.asarray((a.X.tocsr() > 0).mean(axis=0)).ravel()

    st = pd.DataFrame({"gene": a.var_names, "detect_frac": det, "DSI": dsi}).set_index("gene")

    # 人类注释
    ann = pd.read_csv(ANN, sep="\t")
    ann.columns = ["gene_name", "gene_id", "gene_biotype"]
    ann = ann.dropna(subset=["gene_name"]).drop_duplicates("gene_name").set_index("gene_name")
    LNC = {"lncRNA", "lincRNA", "antisense", "processed_transcript",
           "sense_intronic", "sense_overlapping", "macro_lncRNA", "3prime_overlapping_ncRNA"}
    bt = ann["gene_biotype"].reindex(st.index)
    st["gene_class"] = np.where(bt.isin(LNC), "lncRNA",
                                np.where(bt == "protein_coding", "protein_coding", "other"))
    st.to_csv(os.path.join(OUT, "gene_specificity.csv"))
    print(st["gene_class"].value_counts().to_string())

    lnc = st[(st["gene_class"] == "lncRNA") & (st["detect_frac"] >= MIN_DETECT)]
    pc = st[(st["gene_class"] == "protein_coding") & (st["detect_frac"] >= MIN_DETECT)]
    print(f"lncRNA {len(lnc):,} | 蛋白编码 {len(pc):,}")
    print(f"检出率中位: lncRNA {lnc.detect_frac.median():.3f} | PC {pc.detect_frac.median():.3f}")

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
    comp.to_csv(os.path.join(OUT, "matched_comparison.csv"), index=False)

    if len(comp):
        d = comp["delta"].dropna()
        w = stats.wilcoxon(d)
        print("\n" + "=" * 70)
        print("=== 人类脑 Visium：lncRNA vs 匹配蛋白编码 ===")
        print(f"  lncRNA {len(comp):,} | DSI 差值中位 {d.median():+.5f} | "
              f"为正 {(d>0).mean():.1%} | p={w.pvalue:.2e}")
        # 分层
        s2 = st[st["gene_class"].isin(["lncRNA", "protein_coding"]) & (st["detect_frac"] >= 0.01)].copy()
        s2["bin"] = pd.qcut(s2["detect_frac"], 10, labels=False, duplicates="drop")
        print("\n  分层（检出率十分位）:")
        for b_, sub in s2.groupby("bin"):
            aa = sub.loc[sub.gene_class == "lncRNA", "DSI"]
            cc = sub.loc[sub.gene_class == "protein_coding", "DSI"]
            if len(aa) < 5:
                continue
            print(f"    bin {b_}: n_lnc={len(aa):>4} delta={aa.median()-cc.median():+.5f} "
                  f"(p={stats.mannwhitneyu(aa, cc).pvalue:.2g})")
    print(f"\n=== 完成，输出: {OUT} ===")


if __name__ == "__main__":
    main()
