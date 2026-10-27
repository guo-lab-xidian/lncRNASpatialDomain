# -*- coding: utf-8 -*-
"""检出率混杂的稳健性核对：不同阈值下，匹配 vs 不匹配的结论是否稳定。"""
import pandas as pd
import numpy as np
from scipy import stats

R = r"E:\研究工作\agent长非与空间域\lncRNA空间域\results"
rng = np.random.default_rng(0)

sets = {
    "MERFISH ISO": (R + r"\H1H2\gene_specificity.csv", "DSI_domain", "panel_detect"),
    "MERFISH WT": (R + r"\H1H2_WT\gene_specificity.csv", "DSI_domain", "panel_detect"),
}

for name, (path, col, detcol) in sets.items():
    s = pd.read_csv(path, index_col=0)
    s = s[~s.index.duplicated(keep="first")]
    print(f"\n{'='*80}\n=== {name} ===")
    for MIN_D in [0.01, 0.05]:
        lnc = s[(s["gene_class"] == "lncRNA") & (s[detcol] >= MIN_D)]
        pc = s[(s["gene_class"] == "protein_coding") & (s[detcol] >= MIN_D)]
        u = stats.mannwhitneyu(lnc[col], pc[col], alternative="two-sided")
        deltas = []
        for g, r in lnc.iterrows():
            sel = pc[(pc[detcol] >= r[detcol] * 0.8) & (pc[detcol] <= r[detcol] * 1.2)]
            if len(sel) < 5:
                sel = pc.iloc[np.argsort(np.abs(pc[detcol] - r[detcol]))[:5]]
            ctrl = sel.sample(min(5, len(sel)), random_state=int(rng.integers(1e9)))
            deltas.append(r[col] - ctrl[col].median())
        d = pd.Series(deltas).dropna()
        w = stats.wilcoxon(d)
        print(f"\n  阈值检出率 ≥{MIN_D:.0%}: lncRNA {len(lnc)} | PC {len(pc):,}")
        print(f"    检出率中位: lncRNA {lnc[detcol].median():.3f} | PC {pc[detcol].median():.3f}")
        print(f"    【不匹配】lncRNA {lnc[col].median():.4f} vs PC {pc[col].median():.4f} "
              f"→ {'lncRNA更高' if lnc[col].median()>pc[col].median() else 'lncRNA更低'} "
              f"({lnc[col].median()/pc[col].median():.2f}×), p={u.pvalue:.2e}")
        print(f"    【匹配】  差值 {d.median():+.5f}, 为正 {(d>0).mean():.0%}, p={w.pvalue:.2e}")

# Visium：需要重新叠加 gene_class
print(f"\n{'='*80}\n=== Visium V1 ===")
v = pd.read_csv(R + r"\Visium\gene_specificity.csv", index_col=0)
pf = pd.read_csv(r"E:\研究工作\agent长非与空间域\analysis\03_lncrna\panel_features_class.csv", index_col=0)
pf = pf[~pf.index.duplicated(keep="first")]
coords = pd.read_csv(R + r"\gene_coords_m25.tsv", sep="\t")
n2t = coords.drop_duplicates("gene_name").set_index("gene_name")["gene_type"]
LNC = {"lncRNA", "lincRNA", "antisense", "processed_transcript",
       "sense_intronic", "sense_overlapping", "macro_lncRNA"}
cls = pf["gene_class"].reindex(v.index)
fb = n2t.reindex(v.index).map(lambda t: "lncRNA" if isinstance(t, str) and t in LNC
                              else ("protein_coding" if t == "protein_coding" else "other"))
v["gene_class"] = cls.fillna(fb)
for MIN_D in [0.01, 0.05]:
    lnc = v[(v["gene_class"] == "lncRNA") & (v["detect_frac"] >= MIN_D)]
    pc = v[(v["gene_class"] == "protein_coding") & (v["detect_frac"] >= MIN_D)]
    u = stats.mannwhitneyu(lnc["DSI"], pc["DSI"], alternative="two-sided")
    deltas = []
    for g, r in lnc.iterrows():
        sel = pc[(pc["detect_frac"] >= r.detect_frac * 0.8) & (pc["detect_frac"] <= r.detect_frac * 1.2)]
        if len(sel) < 5:
            sel = pc.iloc[np.argsort(np.abs(pc["detect_frac"] - r.detect_frac))[:5]]
        ctrl = sel.sample(min(5, len(sel)), random_state=int(rng.integers(1e9)))
        deltas.append(r["DSI"] - ctrl["DSI"].median())
    d = pd.Series(deltas).dropna()
    w = stats.wilcoxon(d)
    print(f"\n  阈值检出率 ≥{MIN_D:.0%}: lncRNA {len(lnc)} | PC {len(pc):,}")
    print(f"    检出率中位: lncRNA {lnc['detect_frac'].median():.3f} | PC {pc['detect_frac'].median():.3f}")
    print(f"    【不匹配】lncRNA {lnc['DSI'].median():.4f} vs PC {pc['DSI'].median():.4f} "
          f"({lnc['DSI'].median()/pc['DSI'].median():.2f}×), p={u.pvalue:.2e}")
    print(f"    【匹配】  差值 {d.median():+.5f}, 为正 {(d>0).mean():.0%}, p={w.pvalue:.2e}")
