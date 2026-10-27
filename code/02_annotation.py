# -*- coding: utf-8 -*-
"""
阶段 3：把探针面板映射到 Ensembl 注释，统计基因类型构成。

输入：
  data/ensembl_mouse_gene_transcript.tsv   (BioMart: transcript/gene/name/gene_type/transcript_type)
  analysis/01_qc/feature_qc_{WT,ISO}.csv   (阶段 2 的检出率)
  h5ad 的 var/codebook

产出（analysis/02_annotation/）：
  gene_annotation.csv      基因符号 -> gene_id / biotype（去重后）
  panel_features.csv       WT+ISO 全部特征的统一注释表（含检出率）
  biotype_summary.csv      按面板/codebook 的 biotype 构成
  transcript_match.csv     9,754 个 ENSMUST 的匹配明细
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import spatial_io as sio

DATA = r"E:\研究工作\agent长非与空间域\data"
QC = r"E:\研究工作\agent长非与空间域\analysis\01_qc"
OUT = r"E:\研究工作\agent长非与空间域\analysis\02_annotation"
os.makedirs(OUT, exist_ok=True)

TSV = os.path.join(DATA, "ensembl_mouse_gene_transcript.tsv")


def load_annotation():
    ann = pd.read_csv(TSV, sep="\t", dtype=str)
    ann.columns = ["transcript_id", "gene_id", "gene_name", "gene_biotype", "transcript_biotype"]
    ann = ann.dropna(subset=["transcript_id"])
    print(f"注释表: {len(ann):,} 条转录本 | {ann['gene_id'].nunique():,} 个基因")
    print("  基因类型分布（按转录本计）:")
    print(ann["gene_biotype"].value_counts().head(12).to_string())
    return ann


def build_gene_table(ann):
    """基因符号 -> gene_id / biotype。同名多基因时保留转录本最多的那个。"""
    g = ann.groupby(["gene_name", "gene_id", "gene_biotype"]).size().reset_index(name="n_tx")
    g = g.sort_values("n_tx", ascending=False)
    dup = g["gene_name"].duplicated(keep=False)
    n_dup = g.loc[dup, "gene_name"].nunique()
    g_dedup = g.drop_duplicates("gene_name", keep="first").set_index("gene_name")
    print(f"\n基因符号表: {len(g_dedup):,} 个唯一符号（其中 {n_dup} 个符号存在多基因歧义，取转录本最多的）")
    return g_dedup, g


def main():
    ann = load_annotation()
    gene_tab, gene_all = build_gene_table(ann)
    gene_tab.reset_index().to_csv(os.path.join(OUT, "gene_annotation.csv"), index=False)

    tx_tab = ann.set_index("transcript_id")

    rows = []
    for panel in ["WT", "ISO"]:
        key = "cellxgene_" + panel
        var = sio.read_var(key)
        fq = pd.read_csv(os.path.join(QC, f"feature_qc_{panel}.csv"), index_col=0)
        v = var.join(fq, how="left")
        v["panel"] = panel
        v["codebook"] = v["codebook"].astype(str)
        v["feature_type"] = np.where(v.index.str.startswith("ENSMUS"), "transcript", "gene")

        # 基因型特征：按符号匹配
        is_gene = v["feature_type"] == "gene"
        v.loc[is_gene, "gene_name"] = v.index[is_gene]
        v.loc[is_gene, "gene_id"] = gene_tab["gene_id"].reindex(v.index[is_gene]).to_numpy()
        v.loc[is_gene, "gene_biotype"] = gene_tab["gene_biotype"].reindex(v.index[is_gene]).to_numpy()
        # 转录本型特征：按 ENSMUST 匹配
        is_tx = v["feature_type"] == "transcript"
        v.loc[is_tx, "transcript_id"] = v.index[is_tx]
        v.loc[is_tx, "gene_id"] = tx_tab["gene_id"].reindex(v.index[is_tx]).to_numpy()
        v.loc[is_tx, "gene_name"] = tx_tab["gene_name"].reindex(v.index[is_tx]).to_numpy()
        v.loc[is_tx, "gene_biotype"] = tx_tab["gene_biotype"].reindex(v.index[is_tx]).to_numpy()
        v.loc[is_tx, "transcript_biotype"] = tx_tab["transcript_biotype"].reindex(v.index[is_tx]).to_numpy()
        rows.append(v)

    panel = pd.concat(rows)
    panel.index.name = "feature"
    panel.to_csv(os.path.join(OUT, "panel_features.csv"))

    # ---- 匹配率 ----
    print("\n=== 注释匹配率 ===")
    for p in ["WT", "ISO"]:
        sub = panel[panel["panel"] == p]
        for ft in ["gene", "transcript"]:
            s = sub[sub["feature_type"] == ft]
            if len(s) == 0:
                continue
            matched = s["gene_id"].notna().sum()
            print(f"  {p:4} {ft:10}: {len(s):>6,} 个，匹配 {matched:>6,} ({100*matched/len(s):.1f}%)")

    # ---- biotype 构成 ----
    rows = []
    for (p, cb), sub in panel.groupby(["panel", "codebook"], observed=True):
        vc = sub["gene_biotype"].fillna("UNMATCHED").value_counts()
        for bt, n in vc.items():
            rows.append({"panel": p, "codebook": cb, "biotype": bt, "n": int(n)})
    bs = pd.DataFrame(rows)
    bs.to_csv(os.path.join(OUT, "biotype_summary.csv"), index=False)

    print("\n=== 各面板/码本的基因类型构成（前 8 类）===")
    for (p, cb), sub in bs.groupby(["panel", "codebook"], observed=True):
        tot = sub["n"].sum()
        top = sub.nlargest(8, "n")
        txt = ", ".join(f"{r.biotype}={r.n}({100*r.n/tot:.1f}%)" for r in top.itertuples())
        print(f"  {p}/{cb} (n={tot:,}): {txt}")

    # ---- lncRNA 汇总 ----
    print("\n" + "=" * 70)
    print("=== lncRNA 覆盖情况 ===")
    report_lines = []
    for p in ["WT", "ISO"]:
        sub = panel[panel["panel"] == p]
        g = sub[sub["feature_type"] == "gene"]
        t = sub[sub["feature_type"] == "transcript"]
        g_lnc = g[g["gene_biotype"] == "lncRNA"]
        t_lnc = t[t["gene_biotype"] == "lncRNA"]
        line = (f"\n[{p}] 面板 {len(sub):,} 个特征 = 基因 {len(g):,} + 转录本 {len(t):,}\n"
                f"   lncRNA 基因: {len(g_lnc):,} ({100*len(g_lnc)/max(len(g),1):.2f}% of 基因)\n"
                f"   lncRNA 来源的转录本: {len(t_lnc):,} ({100*len(t_lnc)/max(len(t),1):.2f}% of 转录本)\n"
                f"   涉及 lncRNA 基因总数(去重): {pd.concat([g_lnc[['gene_id']], t_lnc[['gene_id']]])['gene_id'].nunique():,}")
        print(line)
        report_lines.append(line)

        if len(g_lnc):
            for thr in [0.01, 0.05, 0.1, 0.2]:
                n = int((g_lnc["detect_frac"] >= thr).sum())
                print(f"      lncRNA 基因检出率 >= {thr:>4}: {n:>5,} ({100*n/len(g_lnc):.1f}%)")

    # ---- 蛋白编码 vs lncRNA 检出率对比 ----
    print("\n=== 各类别检出率中位数对比 ===")
    cmp_rows = []
    for p in ["WT", "ISO"]:
        sub = panel[(panel["panel"] == p) & (panel["feature_type"] == "gene")]
        for bt in ["protein_coding", "lncRNA", "pseudogene", "processed_pseudogene", "miRNA", "snRNA"]:
            s = sub[sub["gene_biotype"] == bt]["detect_frac"].dropna()
            if len(s) == 0:
                continue
            cmp_rows.append({
                "panel": p, "biotype": bt, "n_features": len(s),
                "detect_frac_median": s.median(), "detect_frac_mean": s.mean(),
                "n_detect_ge_10pct": int((s >= 0.1).sum()),
                "n_detect_ge_20pct": int((s >= 0.2).sum()),
            })
    cmp = pd.DataFrame(cmp_rows)
    print(cmp.to_string(index=False))
    cmp.to_csv(os.path.join(OUT, "biotype_detection_compare.csv"), index=False)

    # ---- 高表达 lncRNA Top50 ----
    iso_g = panel[(panel["panel"] == "ISO") & (panel["feature_type"] == "gene") &
                  (panel["gene_biotype"] == "lncRNA")]
    top = iso_g.nlargest(50, "detect_frac")[
        ["gene_id", "gene_name", "detect_frac", "n_cells_detected", "total_counts", "mean_positive"]]
    top.to_csv(os.path.join(OUT, "lncrna_top50_ISO.csv"))
    print("\n=== ISO 面板检出率最高的 15 个 lncRNA ===")
    print(top.head(15).to_string(index=False))

    # ---- 转录本匹配明细 ----
    tx = panel[panel["feature_type"] == "transcript"]
    tx_out = tx[["codebook", "gene_id", "gene_name", "gene_biotype", "transcript_biotype",
                 "detect_frac", "n_cells_detected", "total_counts"]].copy()
    tx_out.to_csv(os.path.join(OUT, "transcript_match.csv"))
    print(f"\n转录本匹配明细已写入 transcript_match.csv（{len(tx_out):,} 行）")
    if len(tx):
        print("  转录本来源基因类型:")
        print(tx["gene_biotype"].fillna("UNMATCHED").value_counts().head(10).to_string())
        print(f"  转录本检出率中位数: {tx['detect_frac'].median():.5f}"
              f" | >=5% 检出: {int((tx['detect_frac'] >= 0.05).sum())} 个"
              f" | >=10% 检出: {int((tx['detect_frac'] >= 0.1).sum())} 个")

    # ---- 未匹配特征 ----
    unmatched = panel[panel["gene_id"].isna()]
    print(f"\n未匹配到注释的特征: {len(unmatched):,} 个")
    print("  示例:", unmatched.index[:20].tolist())
    unmatched.index.to_series().to_csv(os.path.join(OUT, "unmatched_features.csv"), index_label="feature")

    print("\n=== 阶段 3 完成，输出目录:", OUT, "===")


if __name__ == "__main__":
    main()
