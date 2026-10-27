# -*- coding: utf-8 -*-
"""
课题三 / 阶段 1：从 GENCODE M25 GTF 提取基因坐标，用于定义顺式邻域。

产出（lncRNA空间域/results/）：
  gene_coords_m25.tsv   gene_id, gene_name, chrom, start, end, strand, gene_type, length
"""
import gzip
import os
import re

import pandas as pd

GTF = r"E:\研究工作\agent长非与空间域\data\gencode.vM25.annotation.gtf.gz"
OUT = r"E:\研究工作\agent长非与空间域\lncRNA空间域\results"
os.makedirs(OUT, exist_ok=True)

ATTR = re.compile(r'(\w+) "([^"]*)"')


def parse():
    rows = []
    with gzip.open(GTF, "rt", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                continue
            p = line.rstrip("\n").split("\t")
            if len(p) < 9 or p[2] != "gene":
                continue
            a = dict(ATTR.findall(p[8]))
            rows.append({
                "gene_id": a.get("gene_id", "").split(".")[0],
                "gene_name": a.get("gene_name", ""),
                "chrom": p[0],
                "start": int(p[3]),
                "end": int(p[4]),
                "strand": p[6],
                "gene_type": a.get("gene_type", ""),
            })
    df = pd.DataFrame(rows)
    df["length"] = df["end"] - df["start"] + 1
    return df


if __name__ == "__main__":
    df = parse()
    df.to_csv(os.path.join(OUT, "gene_coords_m25.tsv"), sep="\t", index=False)
    print(f"基因数: {len(df):,}")
    print("\n基因类型 top10:")
    print(df["gene_type"].value_counts().head(10).to_string())
    print("\n基因长度（lncRNA vs protein_coding）:")
    for t in ["lncRNA", "lincRNA", "antisense", "protein_coding"]:
        s = df.loc[df["gene_type"] == t, "length"]
        if len(s):
            print(f"  {t:15} n={len(s):>6,} 中位={s.median()/1000:>8.1f} kb 均值={s.mean()/1000:>8.1f} kb")
    print("\n染色体分布（前 25 条染色体上的基因数）:")
    print(df["chrom"].value_counts().head(25).to_string())
    print(f"\n已写出: {os.path.join(OUT, 'gene_coords_m25.tsv')}")
