# -*- coding: utf-8 -*-
"""检查 16 个 lncRNA 的染色体位置与共表达模块重叠，排除性别/染色体混杂。"""
import pandas as pd
from collections import Counter

RES = r"E:\研究工作\agent长非与空间域\lncRNA空间域\results"
coords = pd.read_csv(RES + r"\gene_coords_m25.tsv", sep="\t").drop_duplicates("gene_id").set_index("gene_id")
ann = pd.read_csv(RES + r"\annotation\lncrna_annotation.csv", index_col=0)
co = pd.read_csv(RES + r"\annotation\coexpression_top200.csv", index_col=0)

print("=== 16 个 lncRNA 的染色体位置 ===")
chrom_count = Counter()
for g, r in ann.iterrows():
    gid = r["gene_id"]
    if gid in coords.index:
        c = coords.loc[gid]
        chrom_count[c["chrom"]] += 1
        print(f"  {g:16} {c['chrom']:>6} {int(c['start']):>10}-{int(c['end']):>10}  {c['gene_type']}")
print("\n染色体分布:", dict(chrom_count))

print("\n=== 共表达 Top5 基因的出现频次 ===")
cnt = Counter()
for g, r in co.iterrows():
    for x in str(r["top5"]).split(";"):
        if x and x != "nan":
            cnt[x] += 1
for x, n in cnt.most_common(15):
    gid = None
    row = ann[ann["gene_id"] == x]
    print(f"  {x:14} 出现在 {n:2d} 个 lncRNA 的 Top5 中")

print("\n=== 这些高频共表达基因的染色体 ===")
gi = pd.read_csv(r"E:\研究工作\agent长非与空间域\analysis\03_lncrna\panel_features_class.csv", index_col=0)
gi = gi[~gi.index.duplicated()]
for x, n in cnt.most_common(15):
    if x in gi.index:
        gid = gi.loc[x, "gene_id"]
        if gid in coords.index:
            c = coords.loc[gid]
            print(f"  {x:14} {c['chrom']:>6}  出现 {n:2d} 次")
