# -*- coding: utf-8 -*-
"""检查 obs/annot 是否真的是 Allen CCF 结构 ID：用已知基因的细胞类型分布来验证。"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.sparse import vstack

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import spatial_io as sio

KEY = "cellxgene_ISO"
RES = r"E:\研究工作\agent长非与空间域\lncRNA空间域\results"
LDIR = os.path.join(RES, "Ldomain")
OUT = os.path.join(RES, "AllenISH")

lab = pd.read_csv(os.path.join(LDIR, "domain_labels.csv"), index_col=0)
ranges = json.load(open(os.path.join(LDIR, "subset_rows.json")))
var = sio.read_var(KEY)
st = pd.read_csv(os.path.join(OUT, "allen_structures.csv"))
st["id"] = st["id"].astype(int)
name = st.set_index("id")["name"]
acro = st.set_index("id")["acronym"]

annot = lab["annot"].astype(int).to_numpy()
vc = pd.Series(annot).value_counts()
print("=== annot 取值最多的 15 个，及其在 Allen 结构表里的名称 ===")
for k, n in vc.head(15).items():
    print(f"  annot={int(k):>5} n={n:>5} | {acro.get(int(k), '未匹配'):<12} {str(name.get(int(k), ''))[:50]}")

print("\n=== 用已知细胞类型检验映射 ===")
# 若 annot 是真实脑区，则同一 annot 内的细胞类型应高度一致
ct = lab["cell_label"].astype(str).to_numpy()
for k in vc.head(8).index:
    m = annot == k
    top = pd.Series(ct[m]).value_counts(normalize=True).head(3)
    print(f"  annot={int(k)} (n={m.sum()}): " +
          ", ".join(f"{a}={b:.0%}" for a, b in top.items()))

print("\n=== 用已知标志基因检验：每个 annot 内标志基因的检出率 ===")
markers = {"Gfap": "星形胶质", "Mbp": "少突胶质", "Slc17a7": "兴奋性神经元",
           "Gad1": "抑制性神经元", "Pecam1": "内皮", "Lamp5": "Lamp5中间神经元"}
mk = [g for g in markers if g in var.index]
pos = var.index.get_indexer(mk)
blocks = [sio.read_rows(KEY, r["start"], r["stop"])[:, pos].tocsr() for r in ranges]
X = (vstack(blocks, format="csr") > 0).toarray().astype(float)
print("  " + "annot".ljust(8) + " ".join(f"{g:>10}" for g in mk))
for k in vc.head(10).index:
    m = annot == k
    vals = [X[m, j].mean() for j in range(len(mk))]
    print("  " + str(int(k)).ljust(8) + " ".join(f"{v:>10.3f}" for v in vals))

print("\n=== 若映射正确，Mbp 高的 annot 应对应白质结构 ===")
mbp_i = mk.index("Mbp") if "Mbp" in mk else None
if mbp_i is not None:
    rates = {int(k): X[annot == k, mbp_i].mean() for k in vc.head(30).index}
    top = sorted(rates.items(), key=lambda x: -x[1])[:6]
    for k, v in top:
        print(f"  annot={k} Mbp检出率={v:.3f} → {acro.get(k,'?')} {str(name.get(k,''))[:45]}")
