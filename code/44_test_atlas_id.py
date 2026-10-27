# -*- coding: utf-8 -*-
"""检验 annot 是否对应 Allen 的 atlas_id 而非 id。用细胞类型组成做判据。"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.sparse import vstack

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import spatial_io as sio

RES = r"E:\研究工作\agent长非与空间域\lncRNA空间域\results"
LDIR = os.path.join(RES, "Ldomain")
OUT = os.path.join(RES, "AllenISH")

lab = pd.read_csv(os.path.join(LDIR, "domain_labels.csv"), index_col=0)
st = pd.read_csv(os.path.join(OUT, "allen_structures.csv"))
annot = lab["annot"].astype(int).to_numpy()
ct = lab["cell_label"].astype(str).to_numpy()
vc = pd.Series(annot).value_counts()

# 用细胞类型的"主导类型"作为金标准，检查哪种 id 映射能得到匹配的名称
kw = {
    "Hippocampus_DG": ["dentate", "DG"],
    "Hippocampus_CA1": ["CA1", "field CA1"],
    "Hippocampus_CA2-CA3": ["CA2", "CA3", "field CA"],
    "Thalamus": ["thalam"],
    "Hypothalamus": ["hypothalam"],
    "Oligodendrocytes": ["white matter", "tract", "corpus callosum", "fiber"],
    "Astrocytes": [],
    "Endothelial": [],
    "Cortex L4 RSP": ["retrosplenial"],
    "Cortex L5 ET": ["cortex", "cortical"],
}

for id_col in ["id", "atlas_id", "graph_id"]:
    if id_col not in st.columns:
        print(f"[{id_col}] 表中无该列")
        continue
    m = st.dropna(subset=[id_col]).copy()
    m[id_col] = m[id_col].astype(int)
    n_hit = n_tot = 0
    examples = []
    for k in vc.head(30).index:
        sub = m[m[id_col] == int(k)]
        if sub.empty:
            continue
        nm = (str(sub["name"].iloc[0]) + " " + str(sub["acronym"].iloc[0])).lower()
        top_ct = pd.Series(ct[annot == k]).value_counts(normalize=True).index[0]
        keys = kw.get(top_ct, [])
        if not keys:
            continue
        n_tot += 1
        ok = any(x.lower() in nm for x in keys)
        n_hit += int(ok)
        if len(examples) < 6:
            examples.append(f"annot={k} 主细胞类型={top_ct} → {sub['acronym'].iloc[0]} ({sub['name'].iloc[0]}) {'✓' if ok else '✗'}")
    print(f"\n=== 映射字段: {id_col} ===  匹配 {n_hit}/{n_tot}")
    for e in examples:
        print("   ", e)
