# -*- coding: utf-8 -*-
"""
检查性别相关混杂：X 连锁基因在各切片间的检出率差异。
若某张切片系统性不同，说明样本间存在性别/个体差异，共表达分析需按切片分层或校正。
"""
import json
import sys
import os

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import spatial_io as sio

KEY = "cellxgene_ISO"
RES = r"E:\研究工作\agent长非与空间域\lncRNA空间域\results"

lab = pd.read_csv(os.path.join(RES, "Ldomain", "domain_labels.csv"), index_col=0)
ranges = json.load(open(os.path.join(RES, "Ldomain", "subset_rows.json")))

genes = ["Bex1", "Bex3", "Tceal3", "Tceal5", "Cend1", "Miat", "Cdr1",
         "Xist", "Ddx3x", "Kdm5c"]     # Xist/Ddx3x/Kdm5c 为 X 连锁对照
var = sio.read_var(KEY)
pos = var.index.get_indexer(genes)
found = [(g, p) for g, p in zip(genes, pos) if p >= 0]
print("可查基因:", [g for g, _ in found])

cols = [p for _, p in found]
names = [g for g, _ in found]
sample_arr = lab["sample"].astype(str).to_numpy()

X = []
for r in ranges:
    X.append((sio.read_rows(KEY, r["start"], r["stop"])[:, cols].tocsr() > 0))
from scipy.sparse import vstack
Xb = vstack(X, format="csr").toarray()
print(f"矩阵: {Xb.shape}")

df = pd.DataFrame(Xb, columns=names)
df["sample"] = sample_arr
tab = df.groupby("sample").mean().T
print("\n=== 各切片检出率 ===")
print(tab.round(4).to_string())

print("\n=== 每张切片内部的共表达（Bex1 vs 各 lncRNA）===")
for s in sorted(set(sample_arr)):
    sub = df[df["sample"] == s]
    vals = []
    for g in ["Miat", "Cdr1"]:
        if g in sub:
            r = np.corrcoef(sub["Bex1"], sub[g])[0, 1]
            vals.append(f"{g}={r:.3f}")
    print(f"  sample {s}: Bex1 检出率 {sub['Bex1'].mean():.3f} | " + " | ".join(vals))

print("\n=== 合并所有切片后（当前做法）===")
for g in ["Miat", "Cdr1"]:
    if g in df:
        print(f"  Bex1 vs {g}: r={np.corrcoef(df['Bex1'], df[g])[0,1]:.3f}")

if "Xist" in df.columns:
    print("\n=== Xist（X 失活标志）各切片检出率 ===")
    print(df.groupby("sample")["Xist"].mean().round(4).to_string())
