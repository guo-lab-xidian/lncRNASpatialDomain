# -*- coding: utf-8 -*-
"""
稳健性检验：把类别比较换到 Moran's I 上重做。

动机
----
主结论基于自建的 DSI（依赖域划分）。审稿人可能质疑：
"结论只是这个指标的产物"。Moran's I 不依赖任何域划分，
是空间统计的标准量，因此是一次独立的交叉检验。

数据
----
results/H1H2/moransI.csv 与 H1H2_WT/moransI.csv 已含
lncRNA 与其检出率匹配的蛋白编码对照（ISO 99+439，WT 140+641）。

做法
----
在该表内、按 ±20% 与 ±50% 检出率窗口重新匹配，
计算 Moran's I 的配对差（lncRNA − 5 个对照的中位数）。
"""
import os

import numpy as np
import pandas as pd
from scipy import stats

RES = r"E:\研究工作\agent长非与空间域\lncRNA空间域\results"
SEED, N_CTRL = 0, 5
rows = []

for name, sub in [("MERFISH ISO", "H1H2"), ("MERFISH WT", "H1H2_WT")]:
    d = pd.read_csv(os.path.join(RES, sub, "moransI.csv"), index_col=0)
    d = d[~d.index.duplicated(keep="first")].rename(columns={"panel_detect": "detect"})
    d = d.dropna(subset=["moransI", "detect", "gene_class"])
    lnc = d[(d.gene_class == "lncRNA") & (d.detect >= 0.01)]
    pc = d[(d.gene_class == "protein_coding") & (d.detect >= 0.01)]

    for w in (0.20, 0.50):
        rng = np.random.default_rng(SEED)
        diffs = []
        for _, r in lnc.iterrows():
            sel = pc[(pc.detect >= r.detect * (1 - w)) & (pc.detect <= r.detect * (1 + w))]
            if len(sel) < N_CTRL:
                sel = pc.iloc[np.argsort(np.abs(pc.detect - r.detect))[:N_CTRL]]
            ctrl = sel.sample(min(N_CTRL, len(sel)), random_state=int(rng.integers(1e9)))
            diffs.append(r.moransI - ctrl.moransI.median())
        dv = pd.Series(diffs).dropna()
        p = stats.wilcoxon(dv).pvalue
        rows.append({"dataset": name, "window": f"±{int(w*100)}%",
                     "n_lncRNA": len(dv), "median_delta_I": dv.median(),
                     "frac_positive": float((dv > 0).mean()), "p": p})
        print(f"{name:14} ±{int(w*100):>2}% | n={len(dv):3d} | "
              f"Δ Moran's I 中位 {dv.median():+.4f} | 为正 {(dv>0).mean():.1%} | p={p:.3g}")

out = pd.DataFrame(rows)
p = os.path.join(RES, "Robustness", "moran_class_comparison.csv")
out.to_csv(p, index=False)
print("\n=== 输出:", p, "===")

# ---- 检出率分层核对：残余混杂是否会伪造这个差异？
strat = []
for name, sub in [("MERFISH ISO", "H1H2"), ("MERFISH WT", "H1H2_WT")]:
    d = pd.read_csv(os.path.join(RES, sub, "moransI.csv"), index_col=0)
    d = d.rename(columns={"panel_detect": "detect"}).dropna(subset=["moransI", "detect"])
    d = d[d.detect >= 0.05].copy()
    d["bin"] = pd.qcut(d.detect, 8, labels=False, duplicates="drop")
    for b, s in d.groupby("bin"):
        a = s.loc[s.gene_class == "lncRNA", "moransI"]
        c = s.loc[s.gene_class == "protein_coding", "moransI"]
        if len(a) >= 5 and len(c) >= 5:
            strat.append({"dataset": name, "bin": int(b), "n_lncRNA": len(a),
                          "median_I_lncRNA": a.median(),
                          "median_I_protein_coding": c.median(),
                          "delta": a.median() - c.median()})
st = pd.DataFrame(strat)
st.to_csv(os.path.join(RES, "Robustness", "moran_stratified.csv"), index=False)
print("\n分层核对（Δ = lncRNA 中位 − 蛋白编码中位）：")
for ds, g in st.groupby("dataset"):
    print(f"  {ds}: {int((g.delta < 0).sum())}/{len(g)} 层为负")
