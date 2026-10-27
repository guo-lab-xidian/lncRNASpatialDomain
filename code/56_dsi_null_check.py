# -*- coding: utf-8 -*-
"""
关键检验：DSI 随检出率上升，是生物学效应还是"低检出时噪声抬高 DSI"的伪影？

做法：构造**真实的零模型**——基因在各域检出率完全相同（p 恒定），
观察纯噪声会产生多大的 DSI，并看它随 p 如何变化。

若零模型的 DSI 随 p 上升 → 我们观察到的关系可能是噪声伪影；
若零模型的 DSI 随 p 下降或持平 → 观察到的上升是真实生物学。
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import spatial_io as sio

RES = r"E:\研究工作\agent长非与空间域\lncRNA空间域\results"
LDIR = os.path.join(RES, "Ldomain")
rng = np.random.default_rng(0)

# 用真实的域大小
lab = pd.read_csv(os.path.join(LDIR, "domain_labels.csv"), index_col=0)
dom = pd.factorize(lab["L_domain"].astype(str))[0]
K = dom.max() + 1
n_k = np.bincount(dom, minlength=K)
print(f"真实域结构: {K} 个域，大小 {n_k.min()}–{n_k.max()}（中位 {int(np.median(n_k))}）")

print("\n=== 零模型：基因在各域检出率完全相同 ===")
print(f"{'检出率 p':>10} {'期望 DSI':>10} {'95% 分位':>10}")
rows = []
for p in [0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.40, 0.60]:
    n_sim = 2000
    sims = rng.binomial(n_k[None, :], p, size=(n_sim, K)).astype(float)
    rates = sims / n_k[None, :]
    dsi_null = rates.max(axis=1) - np.median(rates, axis=1)
    rows.append({"p": p, "mean_dsi": dsi_null.mean(),
                 "p95": np.percentile(dsi_null, 95)})
    print(f"{p:>10.3f} {dsi_null.mean():>10.4f} {np.percentile(dsi_null, 95):>10.4f}")

sim = pd.DataFrame(rows)
sim.to_csv(os.path.join(RES, "Robustness", "dsi_null_vs_detection.csv"), index=False)

print("\n=== 解读 ===")
first, last = sim.iloc[1], sim.iloc[-2]
if last["mean_dsi"] < first["mean_dsi"]:
    print(f"  零模型 DSI 随检出率**下降**（{first['mean_dsi']:.4f} @p={first['p']:.2f} "
          f"→ {last['mean_dsi']:.4f} @p={last['p']:.2f}）")
    print("  ⇒ 观察到的 DSI 上升**不是噪声伪影**，而是真实生物学差异")
    print("  ⇒ 匹配去除的是真实的生物学差异，因此比较的是")
    print("     '表达水平相当时，两类基因是否一样特异'，而非'去除技术噪声'")
else:
    print("  零模型 DSI 随检出率上升 ⇒ 观察到的关系可能部分来自噪声抬高")

# 对照：真实数据中的关系
spec = pd.read_csv(os.path.join(RES, "H1H2", "gene_specificity.csv"), index_col=0)
spec = spec[~spec.index.duplicated(keep="first")]
pc = spec[(spec["gene_class"] == "protein_coding") & (spec["panel_detect"] >= 0.005)]
print(f"\n=== 真实数据（MERFISH ISO，蛋白编码基因 n={len(pc):,}）===")
print(f"{'检出率区间':>16} {'n':>6} {'DSI 中位':>10} {'零模型期望':>12}")
bins = [0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.40, 1.01]
for lo, hi in zip(bins[:-1], bins[1:]):
    sub = pc[(pc["panel_detect"] >= lo) & (pc["panel_detect"] < hi)]
    if len(sub) < 10:
        continue
    mid = np.sqrt(lo * min(hi, 0.6))
    null = np.interp(np.log10(mid), np.log10(sim["p"]), sim["mean_dsi"])
    print(f"{lo:>7.3f}–{hi:<7.3f} {len(sub):>6} {sub['DSI_domain'].median():>10.4f} {null:>12.4f}")
