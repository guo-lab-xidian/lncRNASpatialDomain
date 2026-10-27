# -*- coding: utf-8 -*-
"""
课题三可行性核算：异构体层还剩多少可用材料。

回答三个问题：
  Q1 9,754 个异构体特征的基因分布（每个基因几个异构体）
  Q2 被检出的异构体里，有多少基因有 ≥2 个异构体可比较
  Q3 lncRNA 与蛋白编码在"多异构体可比性"上的差异
"""
import os

import pandas as pd

CLS = r"E:\研究工作\agent长非与空间域\analysis\03_lncrna\panel_features_class.csv"
OUT = r"E:\研究工作\agent长非与空间域\analysis\03_lncrna"

pf = pd.read_csv(CLS, index_col=0)
tx = pf[(pf["panel"] == "ISO") & (pf["feature_type"] == "transcript")].copy()
print(f"ISO 异构体特征总数: {len(tx):,}")

print("\n=== Q1: 每个基因的异构体探针数 ===")
per_gene = tx.groupby("gene_id").agg(
    gene_name=("gene_name", "first"),
    gene_class=("gene_class", "first"),
    n_isoforms=("detect_frac", "size"),
    max_detect=("detect_frac", "max"),
    min_detect=("detect_frac", "min"),
    max_counts=("total_counts", "max"),
).reset_index()
print(per_gene["n_isoforms"].value_counts().sort_index().to_string())
print("\n按基因类别统计:")
print(per_gene.groupby("gene_class").agg(
    n_genes=("gene_id", "size"),
    multi_iso_genes=("n_isoforms", lambda s: int((s >= 2).sum())),
    med_iso=("n_isoforms", "median"),
).to_string())

print("\n=== Q2: 可用材料（按检出率阈值）===")
print(f"{'阈值':>8} {'异构体数':>10} {'涉及基因':>10} {'≥2异构体检出':>14} {'其中lncRNA基因':>16}")
for thr in [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]:
    det = tx[tx["detect_frac"] >= thr]
    g = det.groupby("gene_id").size()
    multi = g[g >= 2]
    lnc_multi = per_gene.set_index("gene_id").loc[multi.index]
    lnc_multi = lnc_multi[lnc_multi["gene_class"] == "lncRNA"]
    print(f"{thr:>8} {len(det):>10,} {len(g):>10,} {len(multi):>14,} {len(lnc_multi):>16,}")

print("\n=== Q3: 多异构体基因的类别人数 ===")
for cls in ["lncRNA", "protein_coding"]:
    sub = per_gene[(per_gene["gene_class"] == cls) & (per_gene["n_isoforms"] >= 2)]
    usable = sub[sub["max_detect"] >= 0.01]
    print(f"  {cls:15}: 多异构体基因 {len(sub):>4}，其中至少一个异构体检出≥1% 的 {len(usable):>4}")

print("\n=== lncRNA 多异构体基因明细（按最大检出率排序，取 25 个）===")
sub = per_gene[(per_gene["gene_class"] == "lncRNA") & (per_gene["n_isoforms"] >= 2)]
sub = sub.sort_values("max_detect", ascending=False).head(25)
sub["iso_spread"] = sub["max_detect"] - sub["min_detect"]
print(sub[["gene_name", "n_isoforms", "max_detect", "min_detect", "iso_spread", "max_counts"]].to_string(index=False))

sub.to_csv(os.path.join(OUT, "lncrna_multi_isoform_genes.csv"), index=False)
per_gene.to_csv(os.path.join(OUT, "isoform_per_gene.csv"), index=False)
print("\n已写出: isoform_per_gene.csv / lncrna_multi_isoform_genes.csv")
