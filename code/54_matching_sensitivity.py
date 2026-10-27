# -*- coding: utf-8 -*-
"""
匹配窗口敏感性分析：±10% / ±15% / ±20% / ±30% / ±50%。
检验"匹配后无差异"这一结论是否依赖于窗口宽度的选择。
"""
import os

import numpy as np
import pandas as pd
from scipy import stats

R = r"E:\研究工作\agent长非与空间域\lncRNA空间域\results"
OUT = os.path.join(R, "Robustness")
os.makedirs(OUT, exist_ok=True)
LNC = {"lncRNA", "lincRNA", "antisense", "processed_transcript",
       "sense_intronic", "sense_overlapping", "macro_lncRNA"}
WINDOWS = [0.10, 0.15, 0.20, 0.30, 0.50]
N_CTRL = 5
SEED = 0


def load(name):
    if name == "MERFISH ISO":
        s = pd.read_csv(os.path.join(R, "H1H2", "gene_specificity.csv"), index_col=0)
        s = s[~s.index.duplicated(keep="first")].rename(
            columns={"panel_detect": "detect", "DSI_domain": "DSI"})
    elif name == "MERFISH WT":
        s = pd.read_csv(os.path.join(R, "H1H2_WT", "gene_specificity.csv"), index_col=0)
        s = s[~s.index.duplicated(keep="first")].rename(
            columns={"panel_detect": "detect", "DSI_domain": "DSI"})
    elif name == "Visium mouse":
        s = pd.read_csv(os.path.join(R, "Visium", "gene_specificity.csv"), index_col=0)
        s = s[~s.index.duplicated(keep="first")].rename(columns={"detect_frac": "detect"})
        pf = pd.read_csv(r"E:\研究工作\agent长非与空间域\analysis\03_lncrna\panel_features_class.csv",
                         index_col=0)
        pf = pf[~pf.index.duplicated(keep="first")]
        coords = pd.read_csv(os.path.join(R, "gene_coords_m25.tsv"), sep="\t")
        n2t = coords.drop_duplicates("gene_name").set_index("gene_name")["gene_type"]
        fb = n2t.reindex(s.index).map(lambda t: "lncRNA" if isinstance(t, str) and t in LNC
                                      else ("protein_coding" if t == "protein_coding" else "other"))
        s["gene_class"] = pf["gene_class"].reindex(s.index).fillna(fb)
    else:
        s = pd.read_csv(os.path.join(R, "VisiumHuman", "gene_specificity.csv"), index_col=0)
        s = s[~s.index.duplicated(keep="first")].rename(columns={"detect_frac": "detect"})
    return s[["gene_class", "detect", "DSI"]].dropna()


rows = []
for name in ["MERFISH ISO", "MERFISH WT", "Visium mouse", "Visium human"]:
    s = load(name)
    lnc = s[(s.gene_class == "lncRNA") & (s.detect >= 0.01)]
    pc = s[(s.gene_class == "protein_coding") & (s.detect >= 0.01)]
    for w in WINDOWS:
        rng = np.random.default_rng(SEED)
        d = []
        n_fallback = 0
        for _, r in lnc.iterrows():
            lo, hi = r.detect * (1 - w), r.detect * (1 + w)
            sel = pc[(pc.detect >= lo) & (pc.detect <= hi)]
            if len(sel) < N_CTRL:
                sel = pc.iloc[np.argsort(np.abs(pc.detect - r.detect))[:N_CTRL]]
                n_fallback += 1
            ctrl = sel.sample(min(N_CTRL, len(sel)), random_state=int(rng.integers(1e9)))
            d.append(r.DSI - ctrl.DSI.median())
        d = pd.Series(d).dropna()
        p = stats.wilcoxon(d).pvalue if len(d) > 10 else np.nan
        rows.append({"dataset": name, "window": f"±{int(w*100)}%", "n_lncRNA": len(d),
                     "median_delta": d.median(), "frac_positive": (d > 0).mean(),
                     "p": p, "n_fallback": n_fallback})
        print(f"{name:15} ±{int(w*100):>2}% | n={len(d):>4} | Δ中位 {d.median():+.5f} | "
              f"为正 {(d>0).mean():.1%} | p={p:.3g} | 回退 {n_fallback}")

res = pd.DataFrame(rows)
res.to_csv(os.path.join(OUT, "matching_window_sensitivity.csv"), index=False)
print(f"\n=== 输出: {OUT}/matching_window_sensitivity.csv ===")
