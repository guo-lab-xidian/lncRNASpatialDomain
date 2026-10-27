# -*- coding: utf-8 -*-
"""核对报告中引用的基因层检出率数字（避免把转录本特征混入）。"""
import pandas as pd

CLS = r"E:\研究工作\agent长非与空间域\analysis\03_lncrna\panel_features_class.csv"
pf = pd.read_csv(CLS, index_col=0)
print(pf.groupby(["panel", "feature_type"]).size().to_string())
print()
for panel in ["WT", "ISO"]:
    g = pf[(pf["panel"] == panel) & (pf["feature_type"] == "gene")]
    print(f"=== {panel} 基因层 (n={len(g):,}) ===")
    for cls in ["protein_coding", "lncRNA"]:
        s = g[g["gene_class"] == cls]["detect_frac"]
        print(f"  {cls:15} n={len(s):>6,} 中位={s.median():.5f} "
              f">=1%:{int((s>=0.01).sum()):>6,} >=5%:{int((s>=0.05).sum()):>5,} "
              f">=10%:{int((s>=0.10).sum()):>5,} >=20%:{int((s>=0.20).sum()):>4,}")
