# -*- coding: utf-8 -*-
"""绘制 Fig 4：域特异性指标的检出率依赖噪声地板。

所有零模型参数直接读 62_noise_adjusted_v2.py 的输出，避免重复实现口径。

A  零模型 DSI（无空间结构）随检出率上升；叠加真实基因的中位 DSI
B  实测 / 噪声地板 比值（每个数据集用各自正确的零模型）
C  噪声校正后的类别差异（Δz）
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

R = r"E:\研究工作\agent长非与空间域\lncRNA空间域\results"
FIG = os.path.join(R, "figures")
os.makedirs(FIG, exist_ok=True)

PAL = {"MERFISH ISO": "#4C78A8", "MERFISH WT": "#59A14F", "Visium mouse": "#E49444"}
NAMES = ["MERFISH ISO", "MERFISH WT", "Visium mouse"]

gz = pd.read_csv(os.path.join(R, "Robustness", "gene_null_scores.csv"))
gz = gz[~gz.duplicated(["dataset", "gene"])]
na = pd.read_csv(os.path.join(R, "Robustness", "noise_adjusted_comparison.csv"))

fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.7))

# ---------------- A: 零模型曲线 + 实测（MERFISH ISO）
ax = axes[0]
iso = gz[(gz.dataset == "MERFISH ISO") & (gz.detect >= 0.004)].copy()
iso = iso.sort_values("detect")
# 零模型是检出率的函数，取唯一对应关系
curve = iso.drop_duplicates("null_mean").sort_values("detect")
ax.fill_between(curve["detect"], curve["null_mean"] - 1.96 * curve["null_sd"],
                curve["null_mean"] + 1.96 * curve["null_sd"],
                color="#BBBBBB", alpha=0.45, lw=0, label="null, 95% interval")
ax.plot(curve["detect"], curve["null_mean"], "-", color="#333333", lw=1.8,
        label="null, mean")
pc = iso[iso.gene_class == "protein_coding"]
bins = np.logspace(np.log10(0.005), 0, 16)
idx = np.digitize(pc["detect"], bins)
ctr = [np.sqrt(bins[k - 1] * bins[k]) for k in range(1, len(bins))]
med = [pc.loc[idx == k, "DSI"].median() for k in range(1, len(bins))]
ax.plot(ctr, med, "o-", color=PAL["MERFISH ISO"], ms=4, lw=1.6,
        label="observed (protein-coding)")
ax.set_xscale("log")
ax.set_xlabel("detection rate $p$")
ax.set_ylabel("domain specificity index (DSI)")
ax.set_title("A | A uniform gene is not scored zero", fontsize=9)
ax.legend(fontsize=7, loc="upper left")

# ---------------- B: 观测 / 零模型
ax = axes[1]
edges = [0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.40, 1.01]
for name in NAMES:
    d = gz[(gz.dataset == name) & (gz.gene_class == "protein_coding")]
    xs, ys = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        sub = d[(d.detect >= a) & (d.detect < b)]
        if len(sub) < 10:
            continue
        xs.append(np.sqrt(a * min(b, 0.6)))
        ys.append(sub["DSI"].median() / sub["null_mean"].median())
    ax.plot(xs, ys, "o-", color=PAL[name], ms=4, label=name)
ax.axhline(1, color="k", ls="--", lw=1)
ax.set_xscale("log")
ax.set_xlabel("detection rate $p$")
ax.set_ylabel("observed DSI / null DSI")
ax.set_title("B | Low-detection genes sit on the floor", fontsize=9)
ax.legend(fontsize=7)

# ---------------- C: 噪声校正后的类别差异
ax = axes[2]
na["ds"] = na["dataset"].str.strip()
wins = ["±20%", "±50%"]
w = 0.36
xpos = np.arange(len(NAMES))
for i, win in enumerate(wins):
    vals, ps = [], []
    for nm in NAMES:
        row = na[(na.ds == nm) & (na.window == win)].iloc[0]
        vals.append(row["z_median"]); ps.append(row["z_p"])
    bars = ax.bar(xpos + (i - 0.5) * w, vals, width=w,
                  color=["#9ECAE1", "#2171B5"][i], label=f"window {win}")
    for b, v, pv in zip(bars, vals, ps):
        star = "***" if pv < 1e-3 else ("**" if pv < 1e-2 else ("*" if pv < 0.05 else "n.s."))
        ax.text(b.get_x() + b.get_width() / 2,
                v + (0.02 if v >= 0 else -0.02), star,
                ha="center", va="bottom" if v >= 0 else "top", fontsize=7)
ax.axhline(0, color="k", ls="--", lw=1)
ax.set_xticks(xpos)
ax.set_xticklabels([n.replace(" ", "\n") for n in NAMES], fontsize=8)
ax.set_ylabel("Δ$z$ (lncRNA − matched protein-coding)")
ax.set_title("C | Noise-corrected class difference", fontsize=9)
ax.legend(fontsize=7)

fig.suptitle("Fig 4 | Domain specificity indices have a detection-dependent noise floor",
             fontsize=12)
fig.tight_layout()
out = os.path.join(FIG, "fig4_noise_floor.png")
fig.savefig(out, dpi=200)
plt.close(fig)
print("Fig4 完成:", out)

# 图注所需数字
for name in NAMES:
    d = gz[(gz.dataset == name) & (gz.gene_class == "protein_coding")]
    r = (d["DSI"] / d["null_mean"]).replace([np.inf, -np.inf], np.nan).dropna()
    lo = d[d.detect < 0.02]; hi = d[d.detect > 0.15]
    print(f"{name:14} 比值中位 {r.median():.2f} | 低检出(<2%) {lo['DSI'].median()/lo['null_mean'].median():.2f} "
          f"| 高检出(>15%) {hi['DSI'].median()/hi['null_mean'].median():.2f}")
