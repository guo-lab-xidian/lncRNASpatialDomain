# -*- coding: utf-8 -*-
"""
阶段 4b：统一基因类别口径并输出 lncRNA 覆盖的最终结论表。

问题：GENCODE M25（GRCm38）用 lincRNA / antisense / processed_transcript 等旧命名，
当前 Ensembl（GRCm39）统一为 lncRNA。直接按 biotype 字符串统计会漏掉 M25 兜底的那批。
本脚本把各种非编码亚类归并成 gene_class，再重新统计。

产出（analysis/03_lncrna/）：
  panel_features_class.csv    带 gene_class 的最终表
  class_summary.csv           面板 / 码本 × 类别 计数
  class_detection.csv         各类别检出率对比
  lncrna_final_top.csv        检出率最高的 lncRNA（宽口径）
  lncrna_isoforms.csv         来自 lncRNA 基因的转录本特征
"""
import os
import sys

import numpy as np
import pandas as pd

ANN = r"E:\研究工作\agent长非与空间域\analysis\02_annotation"
OUT = r"E:\研究工作\agent长非与空间域\analysis\03_lncrna"
os.makedirs(OUT, exist_ok=True)

# GENCODE 各版本里属于 lncRNA 之下的 biotype
LNC_TYPES = {
    "lncRNA", "lincRNA", "antisense", "processed_transcript", "sense_intronic",
    "sense_overlapping", "macro_lncRNA", "bidirectional_promoter_lncRNA",
    "3prime_overlapping_ncRNA", "non_coding", "non_coding_transcript",
}
OTHER_NC = {
    "miRNA", "snoRNA", "snRNA", "misc_RNA", "rRNA", "Mt_tRNA", "Mt_rRNA",
    "TEC", "scaRNA", "sRNA", "vault_RNA", "Y_RNA", "RNase_MRP_RNA", "RNase_P_RNA",
    "telomerase_RNA", "guide_RNA", "lncRNA_so", "pseudogenic_transcript",
}


def classify(bt):
    if pd.isna(bt):
        return "UNMATCHED"
    bt = str(bt)
    if bt == "protein_coding":
        return "protein_coding"
    if bt in LNC_TYPES:
        return "lncRNA"
    if "pseudogene" in bt:
        return "pseudogene"
    if bt in OTHER_NC:
        return "other_nc"
    return "other"


def main():
    pf = pd.read_csv(os.path.join(ANN, "panel_features_final.csv"), index_col=0)
    pf["gene_class"] = pf["gene_biotype"].map(classify)
    pf.to_csv(os.path.join(OUT, "panel_features_class.csv"))

    print("=== 各类别计数（按面板 / 码本）===")
    rows = []
    for (p, cb), sub in pf.groupby(["panel", "codebook"], observed=True):
        vc = sub["gene_class"].value_counts()
        for c, n in vc.items():
            rows.append({"panel": p, "codebook": cb, "gene_class": c, "n": int(n)})
    cs = pd.DataFrame(rows)
    cs.to_csv(os.path.join(OUT, "class_summary.csv"), index=False)
    for (p, cb), sub in cs.groupby(["panel", "codebook"], observed=True):
        tot = sub["n"].sum()
        print(f"  {p}/{cb} (n={tot:,}): " +
              ", ".join(f"{r.gene_class}={r.n}({100*r.n/tot:.1f}%)" for r in sub.itertuples()))

    # ---- 检出率对比 ----
    print("\n=== 各类别检出率 ===")
    rows = []
    for p in ["WT", "ISO"]:
        sub = pf[pf["panel"] == p]
        for cls in ["protein_coding", "lncRNA", "pseudogene", "other_nc", "other", "UNMATCHED"]:
            s = sub[sub["gene_class"] == cls]
            if len(s) == 0:
                continue
            d = s["detect_frac"]
            rows.append({
                "panel": p, "gene_class": cls, "n": len(s),
                "detect_median": d.median(), "detect_mean": d.mean(),
                "n_ge_1pct": int((d >= 0.01).sum()),
                "n_ge_5pct": int((d >= 0.05).sum()),
                "n_ge_10pct": int((d >= 0.10).sum()),
                "n_ge_20pct": int((d >= 0.20).sum()),
            })
    cd = pd.DataFrame(rows)
    cd.to_csv(os.path.join(OUT, "class_detection.csv"), index=False)
    print(cd.to_string(index=False))

    # ---- lncRNA 明细 ----
    for p in ["WT", "ISO"]:
        g = pf[(pf["panel"] == p) & (pf["feature_type"] == "gene") & (pf["gene_class"] == "lncRNA")]
        print(f"\n[{p}] lncRNA 基因 {len(g):,} / 22,930 = {100*len(g)/22930:.2f}%")
        print("  亚型构成:")
        print(g["gene_biotype"].value_counts().to_string())
        for thr in [0.01, 0.05, 0.1, 0.2]:
            n = int((g["detect_frac"] >= thr).sum())
            print(f"  检出率 >= {thr:>4}: {n:>5,} ({100*n/len(g):.1f}% of lncRNA)")

    iso = pf[(pf["panel"] == "ISO") & (pf["feature_type"] == "gene") & (pf["gene_class"] == "lncRNA")]
    top = iso.nlargest(60, "detect_frac")[
        ["gene_id", "gene_name", "gene_biotype", "ann_source", "detect_frac",
         "n_cells_detected", "total_counts", "mean_positive"]]
    top.to_csv(os.path.join(OUT, "lncrna_final_top.csv"))
    print("\n=== ISO 面板检出率 Top 20 lncRNA（宽口径）===")
    print(top.head(20).to_string(index=False))

    # ---- lncRNA 来源的转录本特征 ----
    tx = pf[(pf["panel"] == "ISO") & (pf["feature_type"] == "transcript")]
    txl = tx[tx["gene_class"] == "lncRNA"].copy()
    txl = txl.sort_values("detect_frac", ascending=False)
    txl[["gene_id", "gene_name", "gene_biotype", "detect_frac", "n_cells_detected",
         "total_counts", "mean_positive"]].to_csv(os.path.join(OUT, "lncrna_isoforms.csv"))
    print(f"\n=== lncRNA 来源的转录本特征: {len(txl):,} 个 ===")
    print(f"  基因类型构成: {txl['gene_biotype'].value_counts().to_dict()}")
    print(f"  检出率中位数: {txl['detect_frac'].median():.5f}")
    for thr in [0.01, 0.05, 0.1]:
        print(f"  检出率 >= {thr:>4}: {int((txl['detect_frac'] >= thr).sum())} 个")
    print("\n  Top 10:")
    print(txl[["gene_name", "detect_frac", "total_counts"]].head(10).to_string(index=False))

    # 同一基因的多转录本：有多少基因有 ≥2 个转录本特征
    multi = txl.groupby("gene_id").size()
    print(f"\n  涉及 lncRNA 基因 {txl['gene_id'].nunique()} 个；"
          f"其中 {int((multi >= 2).sum())} 个基因有 ≥2 个异构体特征")
    if (multi >= 2).any():
        print("  多异构体的 lncRNA 基因示例:",
              multi[multi >= 2].nlargest(10).to_dict())

    print("\n=== 阶段 4b 完成:", OUT, "===")


if __name__ == "__main__":
    main()
