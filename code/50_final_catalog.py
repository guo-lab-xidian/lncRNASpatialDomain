# -*- coding: utf-8 -*-
"""
固化论文用的 15 个候选 lncRNA 终表。

筛选逻辑（预先确定，避免事后挑选）：
  1. 在两套独立 MERFISH 面板中 Moran's I 均排名前 20（得 16 个）
  2. 剔除 bootstrap 入选率 < 50% 的（AY512931 入选率 10%）
  → 15 个

产出：results/Catalog/final_catalog.csv + .md
"""
import os

import numpy as np
import pandas as pd

R = r"E:\研究工作\agent长非与空间域\lncRNA空间域\results"
OUT = os.path.join(R, "Catalog")
os.makedirs(OUT, exist_ok=True)

TARGETS16 = ["Miat", "Cdr1", "Gm27000", "Gm49698", "Gm48623", "Gm49655",
             "Gm35240", "Gm47221", "Gm42428", "Gm45163", "Gm45895", "Gm47408",
             "Gm49124", "AY512931", "B230220B15Rik", "A830036E02Rik"]

spec = pd.read_csv(os.path.join(R, "H1H2", "gene_specificity.csv"), index_col=0)
spec = spec[~spec.index.duplicated(keep="first")]
moran = pd.read_csv(os.path.join(R, "H1H2", "moransI.csv"), index_col=0)
moran = moran[~moran.index.duplicated(keep="first")]
moran_wt = pd.read_csv(os.path.join(R, "H1H2_WT", "moransI.csv"), index_col=0)
moran_wt = moran_wt[~moran_wt.index.duplicated(keep="first")]
boot = pd.read_csv(os.path.join(R, "CatalogValidation", "bootstrap_stability.csv"))
go = pd.read_csv(os.path.join(R, "annotation", "go_enrichment.csv"), index_col=0)
co = pd.read_csv(os.path.join(R, "annotation", "coexpression_depth_adjusted.csv"), index_col=0)
coords = pd.read_csv(os.path.join(R, "gene_coords_m25.tsv"), sep="\t")
c2 = coords.drop_duplicates("gene_name").set_index("gene_name")

rows = []
for g in TARGETS16:
    hit = boot.loc[boot["gene"] == g, "bootstrap_hit_rate"]
    rows.append({
        "gene": g,
        "chrom": c2.loc[g, "chrom"] if g in c2.index else "",
        "detect_frac": spec.loc[g, "panel_detect"] if g in spec.index else np.nan,
        "DSI_domain": spec.loc[g, "DSI_domain"] if g in spec.index else np.nan,
        "DSI_cell": spec.loc[g, "DSI_cell"] if g in spec.index else np.nan,
        "peak_celltype": spec.loc[g, "peak_celltype"] if g in spec.index else "",
        "moransI_ISO": moran.loc[g, "moransI"] if g in moran.index else np.nan,
        "moransI_WT": moran_wt.loc[g, "moransI"] if g in moran_wt.index else np.nan,
        "bootstrap_hit": float(hit.iloc[0]) if len(hit) else np.nan,
        "top_coexpressed": co.loc[g, "top5"] if g in co.index else "",
        "go_top": go.loc[g, "top_term"] if g in go.index else "",
        "go_adj_p": go.loc[g, "adj_p"] if g in go.index else np.nan,
    })

df = pd.DataFrame(rows).set_index("gene")
df["kept"] = df["bootstrap_hit"] >= 0.5
final = df[df["kept"]].drop(columns=["kept"]).sort_values("moransI_ISO", ascending=False)
final = final.round(4)
final.to_csv(os.path.join(OUT, "final_catalog.csv"))

print(f"16 个候选中保留 {len(final)} 个")
print(f"剔除: {', '.join(df.index[~df['kept']])}\n")
print(final[["chrom", "detect_frac", "moransI_ISO", "moransI_WT",
             "bootstrap_hit", "peak_celltype"]].to_string())

lines = ["# 最终候选目录（15 个 lncRNA）", "",
         "> 筛选：两套独立 MERFISH 面板 Moran's I 均前 20；bootstrap 入选率 ≥50%",
         f"> 保留 {len(final)} 个；剔除 1 个（AY512931，bootstrap 入选率 10%）", "",
         "| lncRNA | 染色体 | 检出率 | Moran's I (ISO) | Moran's I (WT) | bootstrap 入选率 | 峰值细胞类型 | 共表达 Top1 | GO 首位条目 |",
         "|---|---|---|---|---|---|---|---|---|"]
for g, r in final.iterrows():
    top1 = str(r["top_coexpressed"]).split(";")[0] if pd.notna(r["top_coexpressed"]) else ""
    go_txt = str(r["go_top"]).split(" (GO")[0] if pd.notna(r["go_top"]) else "—"
    lines.append(f"| {g} | {r['chrom']} | {r['detect_frac']:.1%} | {r['moransI_ISO']:.3f} | "
                 f"{r['moransI_WT']:.3f} | {r['bootstrap_hit']:.0%} | {r['peak_celltype']} | "
                 f"{top1} | {go_txt} |")
lines += ["", "## 说明", "",
          "- 15 个中 13 个为 `Gm*`/`*Rik` 编号基因，文献中几乎无功能报道",
          "- 2 个已知基因（Miat、Cdr1）的注释结果符合文献预期，可作方法学阳性对照",
          "- 跨平台验证：Miat、A830036E02Rik 在小鼠 Visium 中亦达同检出率基因的 89–93 分位",
          "- **定位**：数据驱动候选，尚未经独立实验方法验证"]
open(os.path.join(OUT, "final_catalog.md"), "w", encoding="utf-8").write("\n".join(lines))
print(f"\n=== 完成，输出: {OUT} ===")
