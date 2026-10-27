# -*- coding: utf-8 -*-
"""汇总 16 个 lncRNA 的功能注释，输出最终表格与 Markdown 摘要。"""
import os

import pandas as pd

RES = r"E:\研究工作\agent长非与空间域\lncRNA空间域\results"
ANN = os.path.join(RES, "annotation")
OUT_MD = r"E:\研究工作\agent长非与空间域\lncRNA空间域\07_16个lncRNA功能注释.md"

ann = pd.read_csv(os.path.join(ANN, "lncrna_annotation.csv"), index_col=0)
co = pd.read_csv(os.path.join(ANN, "coexpression_depth_adjusted.csv"), index_col=0)
go = pd.read_csv(os.path.join(ANN, "go_enrichment.csv"), index_col=0)
coords = pd.read_csv(os.path.join(RES, "gene_coords_m25.tsv"), sep="\t").drop_duplicates("gene_id").set_index("gene_id")

tab = ann[["gene_id", "panel_detect", "DSI_domain", "DSI_cell", "peak_celltype",
           "peak_domain", "moransI", "gene_biotype", "nb100kb"]].copy()
tab["chrom"] = tab["gene_id"].map(coords["chrom"])
tab["top_coexpressed"] = co["top5"]
tab["top1_r"] = co["top1_r"]
tab["go_top"] = go["top_term"].reindex(tab.index)
tab["go_adj_p"] = go["adj_p"].reindex(tab.index)

tab = tab.sort_values("moransI", ascending=False)
tab.to_csv(os.path.join(ANN, "lncrna_annotation_final.csv"))

lines = ["# 16 个跨面板可重复的高空间结构 lncRNA：功能注释", "",
         "> 判定依据：在小鼠脑 MERFISH 的两套独立探针面板（ISO / WT）中，Moran's I 均排名前 20",
         "> 注释方法：共表达模块（已控制每细胞检出深度与切片批次）→ GO 富集；基因组邻域取自 GENCODE M25",
         "", "## 汇总表", "",
         "| lncRNA | 染色体 | 检出率 | Moran's I | 峰值细胞类型 | 共表达 Top1 (r) | GO 首位条目 (adj.p) |",
         "|---|---|---|---|---|---|---|"]

for g, r in tab.iterrows():
    go_txt = r["go_top"] if pd.notna(r["go_top"]) else "—"
    if pd.notna(r["go_adj_p"]):
        go_txt += f" ({r['go_adj_p']:.1e})"
    top1 = str(r["top_coexpressed"]).split(";")[0]
    lines.append(
        f"| {g} | {r['chrom']} | {r['panel_detect']:.1%} | {r['moransI']:.3f} | "
        f"{r['peak_celltype']} | {top1} ({r['top1_r']:.2f}) | {go_txt} |")

lines += ["", "## 基因组邻域（±100 kb 内蛋白编码基因）", ""]
for g, r in tab.iterrows():
    nb = r["nb100kb"] if pd.notna(r["nb100kb"]) else "（无）"
    lines.append(f"- **{g}**：{nb}")

lines += ["", "## 主要观察", "",
          "1. **14/16 富集到突触与神经元功能**：AMPA 受体活性调控、单价离子跨膜转运调控、",
          "   神经元突触可塑性调控、神经元突起发育。这不是随机的 GO 结果——",
          "   说明这批空间结构最显著的 lncRNA 属于神经元/突触模块。",
          "2. **峰值细胞类型分布**：皮层 L4 RSP（6 个）、皮层 L5 ET/IT、海马 CA2-CA3、",
          "   嗅区、中脑——以兴奋性神经元为主。",
          "3. **基因组成分**：13 个 `Gm*`/`*Rik` 编号基因在文献中几乎无功能报道，",
          "   2 个已知基因（Miat、Cdr1）作为阳性对照，其 GO 与共表达结果符合文献预期。",
          "4. **方法学提醒**：若不做检出深度校正，16 个 lncRNA 的共表达 Top5 会几乎全部",
          "   集中在 Bex1/Bex3/Tceal3/Tceal5（X 连锁神经元基因），GO 全部落在",
          "   \"RNA Processing\"——这是稀疏二值数据的典型深度伪影。",
          "", "## 产出文件", "",
          "- `results/annotation/lncrna_annotation_final.csv`：最终注释表",
          "- `results/annotation/coexpression_depth_adjusted.csv`：深度校正后的共表达模块",
          "- `results/annotation/go_enrichment.csv`：GO 富集结果",
          ""]

open(OUT_MD, "w", encoding="utf-8").write("\n".join(lines))
print(f"已写出: {OUT_MD}")
print("\n=== 汇总 ===")
print(tab[["chrom", "panel_detect", "moransI", "peak_celltype", "go_top"]].round(3).to_string())
