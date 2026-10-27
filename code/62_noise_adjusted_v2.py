# -*- coding: utf-8 -*-
"""
噪声校正后的类别比较（修正版）。

修正一：零模型必须与 DSI 的定义匹配
------------------------------------
  MERFISH : DSI = max_k(域内检出率) − median_k(域内检出率)   → 二项零模型，精确
  Visium  : DSI = max_k(域内归一化表达均值) − median_k(...)  → 连续量

旧版（57_）对所有数据集都用二项零模型，在 Visium 上不成立
（症状：观测/零模型比值低至 0.67）。本版对 Visium 改用置换零模型：
打乱 spot 与域标签的对应（域大小固定），域均值的渐近分布为
    m_k ~ N(μ_g, σ_g²/n_k · (N−n_k)/(N−1))
DSI = σ_g · (max u_k − median u_k)，故只需模拟一次标准化零分布再按基因缩放。

修正二：匹配的随机数流必须与主分析一致
--------------------------------------
旧版把零模型模拟与对照抽样共用一个 rng，导致抽到的对照基因与
13_domain_specificity.py / 33_visium_domain.py 不同，raw 列无法复现正文。
本版为每个数据集新建 default_rng(0)，且只用于对照抽样，与主分析逐位一致。

输出
----
results/Robustness/noise_adjusted_comparison.csv   （覆盖旧版）
results/Robustness/null_model_params.csv
"""
import os
import sys

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

RES = r"E:\研究工作\agent长非与空间域\lncRNA空间域\results"
CACHE = r"E:\研究工作\agent长非与空间域\data\visium\cache"
LNC = {"lncRNA", "lincRNA", "antisense", "processed_transcript",
       "sense_intronic", "sense_overlapping", "macro_lncRNA"}
SEED = 0
N_CTRL = 5
N_SIM = 4000
GRID = np.array([0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.18,
                 0.25, 0.35, 0.5, 0.65, 0.8, 0.95])


# --------------------------------------------------------------- MERFISH
def load_merfish(name):
    if name == "MERFISH ISO":
        s = pd.read_csv(os.path.join(RES, "H1H2", "gene_specificity.csv"), index_col=0)
        dom = pd.read_csv(os.path.join(RES, "Ldomain", "domain_labels.csv"), index_col=0)
    else:
        s = pd.read_csv(os.path.join(RES, "H1H2_WT", "gene_specificity.csv"), index_col=0)
        dom = pd.read_csv(os.path.join(RES, "Ldomain_WT", "domain_labels.csv"), index_col=0)
    s = s[~s.index.duplicated(keep="first")].rename(
        columns={"panel_detect": "detect", "DSI_domain": "DSI"})
    n_k = np.bincount(pd.factorize(dom["L_domain"].astype(str))[0])
    s = s[["gene_class", "detect", "DSI"]].dropna()
    return s, n_k


def binomial_null(n_k, rng):
    mu, sd = [], []
    for p in GRID:
        sims = rng.binomial(n_k[None, :], p, size=(N_SIM, len(n_k))).astype(float)
        r = sims / n_k[None, :]
        d = r.max(axis=1) - np.median(r, axis=1)
        mu.append(d.mean()); sd.append(d.std())
    return np.array(mu), np.array(sd)


# ------------------------------------------------------------------ Visium
def load_visium():
    import anndata as ad
    import scanpy as sc

    X = sp.load_npz(os.path.join(CACHE, "visium_counts.npz"))
    meta = pd.read_parquet(os.path.join(CACHE, "visium_meta.parquet"))
    genes = pd.read_csv(os.path.join(CACHE, "visium_genes.txt"), header=None)[0].tolist()
    a = ad.AnnData(X=X.astype(np.float32),
                   obs=meta.set_index("barcode").copy(),
                   var=pd.DataFrame(index=genes))
    a.var_names_make_unique(); a.obs_names_make_unique()
    sc.pp.filter_cells(a, min_counts=500)
    sc.pp.filter_genes(a, min_cells=10)
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    A = a.X.toarray() if sp.issparse(a.X) else np.asarray(a.X)
    sigma = A.std(axis=0, ddof=0)
    del A
    sig = pd.Series(sigma, index=a.var_names)

    s = pd.read_csv(os.path.join(RES, "Visium", "gene_specificity.csv"), index_col=0)
    s = s[~s.index.duplicated(keep="first")].rename(columns={"detect_frac": "detect"})
    pf = pd.read_csv(r"E:\研究工作\agent长非与空间域\analysis\03_lncrna\panel_features_class.csv",
                     index_col=0)
    pf = pf[~pf.index.duplicated(keep="first")]
    coords = pd.read_csv(os.path.join(RES, "gene_coords_m25.tsv"), sep="\t")
    n2t = coords.drop_duplicates("gene_name").set_index("gene_name")["gene_type"]
    fb = n2t.reindex(s.index).map(lambda t: "lncRNA" if isinstance(t, str) and t in LNC
                                  else ("protein_coding" if t == "protein_coding" else "other"))
    s["gene_class"] = pf["gene_class"].reindex(s.index).fillna(fb)
    s["sigma"] = sig.reindex(s.index)

    dom = pd.read_csv(os.path.join(RES, "Visium", "domain_labels.csv"), index_col=0)
    n_k = np.bincount(pd.factorize(dom["L_domain"].astype(str))[0])
    s = s[["gene_class", "detect", "DSI", "sigma"]].dropna(subset=["gene_class", "detect", "DSI"])
    s = s[s.sigma > 0]
    return s, n_k, len(dom)


def permutation_null_std(n_k, N, rng):
    v = (1.0 / n_k) * (N - n_k) / (N - 1)
    u = rng.normal(0.0, 1.0, size=(N_SIM, len(n_k))) * np.sqrt(v)[None, :]
    d = u.max(axis=1) - np.median(u, axis=1)
    return d.mean(), d.std()


# ------------------------------------------------------------ 匹配比较
def matched_delta(s, window):
    """与主分析逐位一致：每个数据集新建 default_rng(SEED)，只用于对照抽样。"""
    rng = np.random.default_rng(SEED)
    lnc = s[(s.gene_class == "lncRNA") & (s.detect >= 0.01)]
    pc = s[(s.gene_class == "protein_coding") & (s.detect >= 0.01)]
    d_raw, d_z = [], []
    for _, r in lnc.iterrows():
        sel = pc[(pc.detect >= r.detect * (1 - window)) & (pc.detect <= r.detect * (1 + window))]
        if len(sel) < 5:
            sel = pc.iloc[np.argsort(np.abs(pc.detect - r.detect))[:5]]
        ctrl = sel.sample(min(N_CTRL, len(sel)), random_state=int(rng.integers(1e9)))
        d_raw.append(r.DSI - ctrl.DSI.median())
        d_z.append(r.z - ctrl.z.median())
    return pd.Series(d_raw).dropna(), pd.Series(d_z).dropna()


rows, params, gene_rows = [], [], []
rng_null = np.random.default_rng(12345)


def stash(name, s):
    """逐基因导出零模型参数，供绘图脚本直接使用。"""
    t = s[["gene_class", "detect", "DSI", "null_mean", "null_sd", "z"]].copy()
    t["dataset"] = name
    t.index.name = "gene"
    gene_rows.append(t.reset_index())


for name in ["MERFISH ISO", "MERFISH WT"]:
    s, n_k = load_merfish(name)
    mu, sd = binomial_null(n_k, rng_null)
    params.append({"dataset": name, "null_model": "binomial detection",
                   "n_domains": len(n_k), "null_mean_p0.01": mu[2],
                   "null_mean_p0.40": mu[6],
                   "null_sd_p0.01": sd[2], "null_sd_p0.40": sd[6]})
    logp = np.log10(s["detect"].clip(1e-4, 0.999))
    s["null_mean"] = np.interp(logp, np.log10(GRID), mu)
    s["null_sd"] = np.maximum(np.interp(logp, np.log10(GRID), sd), 1e-9)
    s["z"] = (s["DSI"] - s["null_mean"]) / s["null_sd"]
    stash(name, s)
    for w in [0.20, 0.50]:
        dr, dz = matched_delta(s, w)
        rows.append({"dataset": name, "window": f"±{int(w*100)}%",
                     "raw_median": dr.median(), "raw_p": stats.wilcoxon(dr).pvalue,
                     "z_median": dz.median(), "z_p": stats.wilcoxon(dz).pvalue,
                     "z_frac_pos": float((dz > 0).mean())})
        print(f"{name:14} ±{int(w*100):>2}% | raw {dr.median():+.5f} "
              f"(p={stats.wilcoxon(dr).pvalue:.2g}) | z {dz.median():+.4f} "
              f"(p={stats.wilcoxon(dz).pvalue:.2g})")

name = "Visium mouse"
s, n_k, N = load_visium()
mD, sD = permutation_null_std(n_k, N, rng_null)
params.append({"dataset": name, "null_model": "spot permutation (finite pop.)",
               "n_domains": len(n_k), "null_mean_p0.01": np.nan,
               "null_mean_p0.40": np.nan, "null_sd_p0.01": np.nan, "null_sd_p0.40": np.nan})
print(f"\nVisium 置换零分布: mean_D={mD:.4f}, sd_D={sD:.4f} "
      f"(N={N}, K={len(n_k)}, n_k={n_k.min()}–{n_k.max()})")
s["null_mean"] = s["sigma"] * mD
s["null_sd"] = s["sigma"] * sD
s["z"] = (s["DSI"] - s["null_mean"]) / s["null_sd"]
stash(name, s)
for w in [0.20, 0.50]:
    dr, dz = matched_delta(s, w)
    rows.append({"dataset": name, "window": f"±{int(w*100)}%",
                 "raw_median": dr.median(), "raw_p": stats.wilcoxon(dr).pvalue,
                 "z_median": dz.median(), "z_p": stats.wilcoxon(dz).pvalue,
                 "z_frac_pos": float((dz > 0).mean())})
    print(f"{name:14} ±{int(w*100):>2}% | raw {dr.median():+.5f} "
          f"(p={stats.wilcoxon(dr).pvalue:.2g}) | z {dz.median():+.4f} "
          f"(p={stats.wilcoxon(dz).pvalue:.2g})")

out = pd.DataFrame(rows)
out.to_csv(os.path.join(RES, "Robustness", "noise_adjusted_comparison.csv"), index=False)
pd.DataFrame(params).to_csv(os.path.join(RES, "Robustness", "null_model_params.csv"), index=False)
pd.concat(gene_rows, ignore_index=True).to_csv(
    os.path.join(RES, "Robustness", "gene_null_scores.csv"), index=False)
print(f"\n=== 输出: {RES}\\Robustness\\noise_adjusted_comparison.csv ===")
