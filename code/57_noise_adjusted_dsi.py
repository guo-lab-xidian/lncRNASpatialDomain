# -*- coding: utf-8 -*-
"""
!! DEPRECATED — 不要使用本脚本的结果 !!

本文件保留仅作记录，正确版本是 62_noise_adjusted_v2.py。它有两处错误：

  1. 对 Visium 套用了二项零模型。Visium 的 DSI 基于归一化表达均值，
     而非域内检出率，二项零模型不成立（症状：观测/零模型比值 < 1）。
     62_ 版对 Visium 改用带有限总体校正的置换零模型。
  2. 零模型模拟与对照抽样共用一个随机数流，导致抽到的对照基因与主分析
     (13_domain_specificity.py / 33_visium_domain.py) 不一致，raw 列无法
     复现正文表格。62_ 版为每个数据集新建独立随机流。

--------------------------------------------------------------------

用"噪声校正后的 DSI"重做类别比较。

问题：DSI = max−median 在低检出时存在噪声地板（零模型期望随 p 上升），
      因此原始 DSI 的跨检出率比较本身就不公平。

做法：
  1. 对每个数据集，按其真实域结构模拟"完全无域结构"的基因，
     得到每个检出率 p 下 DSI 的零分布（均值与标准差）
  2. 校正指标：z = (DSI_obs − mean_null(p)) / sd_null(p)
  3. 用 z 重做匹配比较（±20% 与 ±50% 两个带宽）
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

RES = r"E:\研究工作\agent长非与空间域\lncRNA空间域\results"
LNC = {"lncRNA", "lincRNA", "antisense", "processed_transcript",
       "sense_intronic", "sense_overlapping", "macro_lncRNA"}
rng = np.random.default_rng(0)
GRID = np.array([0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.18,
                 0.25, 0.35, 0.5, 0.65, 0.8, 0.95])


def null_table(n_k, n_sim=3000):
    """每个网格 p 下，无域结构基因的 DSI 零分布均值与标准差。"""
    means, sds = [], []
    for p in GRID:
        sims = rng.binomial(n_k[None, :], p, size=(n_sim, len(n_k))).astype(float)
        r = sims / n_k[None, :]
        d = r.max(axis=1) - np.median(r, axis=1)
        means.append(d.mean()); sds.append(d.std())
    return np.array(means), np.array(sds)


def load(name):
    if name == "MERFISH ISO":
        s = pd.read_csv(os.path.join(RES, "H1H2", "gene_specificity.csv"), index_col=0)
        s = s[~s.index.duplicated(keep="first")].rename(
            columns={"panel_detect": "detect", "DSI_domain": "DSI"})
        dom = pd.read_csv(os.path.join(RES, "Ldomain", "domain_labels.csv"), index_col=0)
    elif name == "MERFISH WT":
        s = pd.read_csv(os.path.join(RES, "H1H2_WT", "gene_specificity.csv"), index_col=0)
        s = s[~s.index.duplicated(keep="first")].rename(
            columns={"panel_detect": "detect", "DSI_domain": "DSI"})
        dom = pd.read_csv(os.path.join(RES, "Ldomain_WT", "domain_labels.csv"), index_col=0)
    elif name == "Visium mouse":
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
        dom = pd.read_csv(os.path.join(RES, "Visium", "domain_labels.csv"), index_col=0)
    else:
        s = pd.read_csv(os.path.join(RES, "VisiumHuman", "gene_specificity.csv"), index_col=0)
        s = s[~s.index.duplicated(keep="first")].rename(columns={"detect_frac": "detect"})
        dom = None
    return s[["gene_class", "detect", "DSI"]].dropna(), dom


rows = []
for name in ["MERFISH ISO", "MERFISH WT", "Visium mouse"]:
    s, dom = load(name)
    codes = pd.factorize(dom["L_domain"].astype(str))[0]
    n_k = np.bincount(codes)
    mu, sd = null_table(n_k)
    # 插值到每个基因的检出率
    logp = np.log10(s["detect"].clip(1e-4, 0.999))
    mu_g = np.interp(logp, np.log10(GRID), mu)
    sd_g = np.interp(logp, np.log10(GRID), sd)
    s = s.assign(z=(s["DSI"] - mu_g) / np.maximum(sd_g, 1e-9))

    lnc = s[(s.gene_class == "lncRNA") & (s.detect >= 0.01)]
    pc = s[(s.gene_class == "protein_coding") & (s.detect >= 0.01)]
    for w in [0.20, 0.50]:
        d_raw, d_z = [], []
        for _, r in lnc.iterrows():
            sel = pc[(pc.detect >= r.detect * (1 - w)) & (pc.detect <= r.detect * (1 + w))]
            if len(sel) < 5:
                sel = pc.iloc[np.argsort(np.abs(pc.detect - r.detect))[:5]]
            ctrl = sel.sample(min(5, len(sel)), random_state=int(rng.integers(1e9)))
            d_raw.append(r.DSI - ctrl.DSI.median())
            d_z.append(r.z - ctrl.z.median())
        d_raw = pd.Series(d_raw).dropna(); d_z = pd.Series(d_z).dropna()
        rows.append({
            "dataset": name, "window": f"±{int(w*100)}%",
            "raw_median": d_raw.median(), "raw_p": stats.wilcoxon(d_raw).pvalue,
            "z_median": d_z.median(), "z_p": stats.wilcoxon(d_z).pvalue,
            "z_frac_pos": (d_z > 0).mean(),
        })
        print(f"{name:14} ±{int(w*100):>2}% | 原始 ΔDSI {d_raw.median():+.5f} "
              f"(p={stats.wilcoxon(d_raw).pvalue:.2g}) | 噪声校正 z 差 "
              f"{d_z.median():+.4f} (p={stats.wilcoxon(d_z).pvalue:.2g})")

out = pd.DataFrame(rows)
out.to_csv(os.path.join(RES, "Robustness", "noise_adjusted_comparison.csv"), index=False)
print(f"\n=== 输出: {RES}/Robustness/noise_adjusted_comparison.csv ===")
