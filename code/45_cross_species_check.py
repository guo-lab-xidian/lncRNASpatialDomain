# -*- coding: utf-8 -*-
"""
替代性正交验证：在人类脑 Visium（独立技术、独立物种）中检查小鼠候选基因的人类同源基因。

对每个小鼠 lncRNA，取其人类同源符号（如 Miat→MIAT），
检查在人类数据中是否被检出、其域特异性处于同检出率基因的什么分位。
"""
import os

import numpy as np
import pandas as pd

R = r"E:\研究工作\agent长非与空间域\lncRNA空间域\results"
HUM = os.path.join(R, "VisiumHuman", "gene_specificity.csv")
OUT = os.path.join(R, "CatalogValidation")
os.makedirs(OUT, exist_ok=True)

# 小鼠 → 人类同源（仅对已知同源基因；Gm*/Rik 通常无人类同源）
ORTHOLOG = {
    "Miat": "MIAT", "Cdr1": "CDR1", "Ftx": "FTX", "Pantr1": "PANO1",
    "Dlx1as": "DLX1AS", "Neat1": "NEAT1", "Xist": "XIST", "H19": "H19",
}

TARGETS = ["Miat", "Cdr1", "Gm27000", "Gm49698", "Gm48623", "Gm49655",
           "Gm35240", "Gm47221", "Gm42428", "Gm45163", "Gm45895", "Gm47408",
           "Gm49124", "AY512931", "B230220B15Rik", "A830036E02Rik"]

h = pd.read_csv(HUM, index_col=0)
h = h[~h.index.duplicated(keep="first")]
print(f"人类数据基因数: {len(h):,} | lncRNA {int((h['gene_class']=='lncRNA').sum()):,}")

rows = []
for g in TARGETS:
    human = ORTHOLOG.get(g)
    if human is None:
        rows.append({"mouse": g, "human_ortholog": "", "status": "无已知同源"})
        continue
    if human not in h.index:
        rows.append({"mouse": g, "human_ortholog": human, "status": "人类数据中未检出"})
        continue
    d = h.loc[human, "detect_frac"]
    dsi = h.loc[human, "DSI"]
    peers = h[(h["detect_frac"] >= d * 0.8) & (h["detect_frac"] <= d * 1.2)]["DSI"]
    pct = float((peers < dsi).mean())
    rows.append({"mouse": g, "human_ortholog": human, "status": "已检出",
                 "human_detect": d, "human_DSI": dsi,
                 "percentile_vs_matched": pct, "n_peers": len(peers)})
    print(f"  {g:16} → {human:10} 检出率 {d:.3f} | DSI {dsi:.4f} | "
          f"同检出率分位 {pct:.1%} (n={len(peers)})")

out = pd.DataFrame(rows)
out.to_csv(os.path.join(OUT, "human_ortholog_check.csv"), index=False)

done = out[out["status"] == "已检出"]
if len(done):
    print(f"\n有小鼠-人类同源且可评估的: {len(done)} 个")
    print(f"  期望若随机：中位分位应为 50%；实测中位 {done['percentile_vs_matched'].median():.1%}")

# 对照：随机 lncRNA 的分位分布
lnc = h[(h["gene_class"] == "lncRNA") & (h["detect_frac"] >= 0.01)]
print(f"\n参考：人类数据中 lncRNA 共 {len(lnc):,} 个，DSI 中位 {lnc['DSI'].median():.4f}")
