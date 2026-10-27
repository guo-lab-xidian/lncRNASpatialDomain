# -*- coding: utf-8 -*-
"""核对"检出率混杂"在三个数据集上的具体表现：匹配 vs 不匹配的比较差异。"""
import pandas as pd
import numpy as np
from scipy import stats

R = r"E:\研究工作\agent长非与空间域\lncRNA空间域\results"
rng = np.random.default_rng(0)
MIN_D = 0.01

sets = {
    "MERFISH ISO": (R + r"\H1H2\gene_specificity.csv", "DSI_domain"),
    "MERFISH WT": (R + r"\H1H2_WT\gene_specificity.csv", "DSI_domain"),
    "Visium V1": (R + r"\Visium\gene_specificity.csv", "DSI"),
}

for name, (path, col) in sets.items():
    s = pd.read_csv(path, index_col=0)
    s = s[~s.index.duplicated(keep="first")]
    lnc = s[(s["gene_class"] == "lncRNA") & (s["panel_detect" if "panel_detect" in s else "detect_frac"] >= MIN_D)]
    detcol = "panel_detect" if "panel_detect" in s else "detect_frac"
    pc = s[(s["gene_class"] == "protein_coding") & (s[detcol] >= MIN_D)]
    if len(lnc) < 20:
        continue

    # 不匹配
    u = stats.mannwhitneyu(lnc[col], pc[col], alternative="two-sided")
    # 匹配
    deltas = []
    for g, r in lnc.iterrows():
        sel = pc[(pc[detcol] >= r[detcol] * 0.8) & (pc[detcol] <= r[detcol] * 1.2)]
        if len(sel) < 5:
            sel = pc.iloc[np.argsort(np.abs(pc[detcol] - r[detcol]))[:5]]
        ctrl = sel.sample(min(5, len(sel)), random_state=int(rng.integers(1e9)))
        deltas.append(r[col] - ctrl[col].median())
    d = pd.Series(deltas).dropna()
    w = stats.wilcoxon(d)

    print(f"=== {name} ===")
    print(f"  基因数: lncRNA {len(lnc):,} | 蛋白编码 {len(pc):,}")
    print(f"  检出率中位: lncRNA {lnc[detcol].median():.4f} | 蛋白编码 {pc[detcol].median():.4f}")
    print(f"  【不匹配】lncRNA {col} 中位 {lnc[col].median():.4f} vs PC {pc[col].median():.4f} "
          f"| 倍数 {lnc[col].median()/max(pc[col].median(),1e-9):.2f}× | MWU p={u.pvalue:.2e}")
    print(f"  【匹配】  差值中位 {d.median():+.4f} | 为正 {(d>0).mean():.1%} | Wilcoxon p={w.pvalue:.2e}")
    print()


