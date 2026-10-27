# -*- coding: utf-8 -*-
"""
阶段 3b：用 GENCODE M25(GRCm38) 注释补齐当前 Ensembl(GRCm39) 匹配不到的符号。

背景：WT/ISO 面板里有 2,065 个基因符号（多为 *Rik / Gm*）在当前 Ensembl 里查不到，
说明探针面板基于较早期的注释（GRCm38 时代）。这里用 GENCODE M25 做二次匹配，
并顺带检验 9,754 个 ENSMUST 更接近哪个注释版本。

产出（analysis/02_annotation/）：
  gencode_m25_genes.tsv        M25 基因表
  panel_features_final.csv     最终注释表（current + M25 兜底）
  match_rate_compare.csv       两个注释版本的匹配率对比
  biotype_summary_final.csv    最终 biotype 构成
"""
import gzip
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import spatial_io as sio

DATA = r"E:\研究工作\agent长非与空间域\data"
QC = r"E:\研究工作\agent长非与空间域\analysis\01_qc"
ANN = r"E:\研究工作\agent长非与空间域\analysis\02_annotation"
GTF = os.path.join(DATA, "gencode.vM25.annotation.gtf.gz")
TSV = os.path.join(DATA, "ensembl_mouse_gene_transcript.tsv")

ATTR = re.compile(r'(\w+) "([^"]*)"')


def parse_gtf(path):
    genes, txs = [], []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9:
                continue
            feat = parts[2]
            if feat not in ("gene", "transcript"):
                continue
            attrs = dict(ATTR.findall(parts[8]))
            gid = attrs.get("gene_id", "").split(".")[0]
            if feat == "gene":
                genes.append({
                    "gene_id": gid,
                    "gene_name": attrs.get("gene_name", ""),
                    "gene_biotype": attrs.get("gene_type", ""),
                    "chrom": parts[0],
                })
            else:
                txs.append({
                    "transcript_id": attrs.get("transcript_id", "").split(".")[0],
                    "gene_id": gid,
                })
    return pd.DataFrame(genes), pd.DataFrame(txs)


def main():
    if not os.path.exists(GTF):
        print("缺少 M25 GTF:", GTF)
        sys.exit(1)

    print("解析 GENCODE M25 ...", flush=True)
    g25, t25 = parse_gtf(GTF)
    print(f"  M25 基因 {len(g25):,} | 转录本 {len(t25):,}")
    print("  M25 基因类型 top10:")
    print(g25["gene_biotype"].value_counts().head(10).to_string())
    g25.to_csv(os.path.join(ANN, "gencode_m25_genes.tsv"), sep="\t", index=False)

    # ---- 当前 Ensembl ----
    cur = pd.read_csv(TSV, sep="\t", dtype=str)
    cur.columns = ["transcript_id", "gene_id", "gene_name", "gene_biotype", "transcript_biotype"]
    cur_tx = cur.dropna(subset=["transcript_id"]).drop_duplicates("transcript_id").set_index("transcript_id")
    cur_gene = (cur.dropna(subset=["gene_name"])
                .groupby(["gene_name", "gene_id", "gene_biotype"]).size()
                .reset_index(name="n").sort_values("n", ascending=False)
                .drop_duplicates("gene_name").set_index("gene_name"))

    g25_by_name = g25.drop_duplicates("gene_name").set_index("gene_name")
    t25_by_id = t25.drop_duplicates("transcript_id").set_index("transcript_id")

    # ---- 匹配率对比 ----
    var_wt = sio.read_var("cellxgene_WT")
    var_iso = sio.read_var("cellxgene_ISO")
    names = var_wt.index.to_numpy()
    tx_ids = np.array([n for n in var_iso.index if n.startswith("ENSMUS")])

    rows = []
    for label, tab in [("Ensembl current (GRCm39)", cur_gene), ("GENCODE M25 (GRCm38)", g25_by_name)]:
        hit = tab.index.isin(names).sum()
        rows.append({"source": label, "feature": "gene_symbol", "n": len(names),
                     "matched": int(hit), "rate": hit / len(names)})
    for label, tab in [("Ensembl current (GRCm39)", cur_tx), ("GENCODE M25 (GRCm38)", t25_by_id)]:
        hit = tab.index.isin(tx_ids).sum()
        rows.append({"source": label, "feature": "transcript_id", "n": len(tx_ids),
                     "matched": int(hit), "rate": hit / len(tx_ids)})
    mr = pd.DataFrame(rows)
    mr.to_csv(os.path.join(ANN, "match_rate_compare.csv"), index=False)
    print("\n=== 注释版本匹配率对比 ===")
    print(mr.to_string(index=False))

    # ---- 合并注释（current 优先，M25 兜底）----
    def annotate_features(panel):
        var = sio.read_var("cellxgene_" + panel)
        fq = pd.read_csv(os.path.join(QC, f"feature_qc_{panel}.csv"), index_col=0)
        v = var.join(fq, how="left")
        v["panel"] = panel
        v["codebook"] = v["codebook"].astype(str)
        v["feature_type"] = np.where(v.index.str.startswith("ENSMUS"), "transcript", "gene")
        # 预先建列，供后续逐特征回填
        for col in ["gene_id", "gene_name", "gene_biotype", "transcript_biotype", "ann_source"]:
            v[col] = pd.Series(pd.NA, index=v.index, dtype="object")

        idx = v.index
        for i, feat in enumerate(idx):
            if v.at[feat, "feature_type"] == "transcript":
                if feat in cur_tx.index:
                    r = cur_tx.loc[feat]
                    v.at[feat, "gene_id"] = r["gene_id"]
                    v.at[feat, "gene_name"] = r["gene_name"]
                    v.at[feat, "gene_biotype"] = r["gene_biotype"]
                    v.at[feat, "transcript_biotype"] = r["transcript_biotype"]
                    v.at[feat, "ann_source"] = "ensembl_current"
                elif feat in t25_by_id.index:
                    gid = t25_by_id.at[feat, "gene_id"]
                    g = g25[g25["gene_id"] == gid]
                    v.at[feat, "gene_id"] = gid
                    v.at[feat, "gene_biotype"] = g["gene_biotype"].iloc[0] if len(g) else pd.NA
                    v.at[feat, "gene_name"] = g["gene_name"].iloc[0] if len(g) else pd.NA
                    v.at[feat, "ann_source"] = "gencode_m25"
            else:
                if feat in cur_gene.index:
                    r = cur_gene.loc[feat]
                    v.at[feat, "gene_name"] = feat
                    v.at[feat, "gene_id"] = r["gene_id"]
                    v.at[feat, "gene_biotype"] = r["gene_biotype"]
                    v.at[feat, "ann_source"] = "ensembl_current"
                elif feat in g25_by_name.index:
                    r = g25_by_name.loc[feat]
                    v.at[feat, "gene_name"] = feat
                    v.at[feat, "gene_id"] = r["gene_id"]
                    v.at[feat, "gene_biotype"] = r["gene_biotype"]
                    v.at[feat, "ann_source"] = "gencode_m25"
        return v

    panel_df = pd.concat([annotate_features("WT"), annotate_features("ISO")])
    panel_df.index.name = "feature"
    panel_df.to_csv(os.path.join(ANN, "panel_features_final.csv"))
    print(f"\n最终注释表: {len(panel_df):,} 行 -> panel_features_final.csv")

    print("\n=== 注释来源分布 ===")
    print(panel_df.groupby(["panel", "ann_source"], dropna=False).size().to_string())

    # ---- 最终 biotype 构成 ----
    rows = []
    for (p, cb), sub in panel_df.groupby(["panel", "codebook"], observed=True):
        vc = sub["gene_biotype"].fillna("UNMATCHED").value_counts()
        for bt, n in vc.items():
            rows.append({"panel": p, "codebook": cb, "biotype": bt, "n": int(n)})
    bs = pd.DataFrame(rows)
    bs.to_csv(os.path.join(ANN, "biotype_summary_final.csv"), index=False)
    print("\n=== 最终构成 ===")
    for (p, cb), sub in bs.groupby(["panel", "codebook"], observed=True):
        tot = sub["n"].sum()
        top = sub.nlargest(6, "n")
        print(f"  {p}/{cb} (n={tot:,}): " +
              ", ".join(f"{r.biotype}={r.n}({100*r.n/tot:.1f}%)" for r in top.itertuples()))

    # ---- lncRNA 最终数字 ----
    print("\n" + "=" * 70)
    for p in ["WT", "ISO"]:
        sub = panel_df[panel_df["panel"] == p]
        g = sub[sub["feature_type"] == "gene"]
        t = sub[sub["feature_type"] == "transcript"]
        gl = g[g["gene_biotype"] == "lncRNA"]
        tl = t[t["gene_biotype"] == "lncRNA"]
        print(f"[{p}] 基因 {len(g):,} 中 lncRNA {len(gl):,} ({100*len(gl)/len(g):.2f}%) | "
              f"转录本 {len(t):,} 中 lncRNA 来源 {len(tl):,}"
              f" | 未匹配 {int(sub['gene_biotype'].isna().sum()):,}")
        for thr in [0.01, 0.05, 0.1, 0.2]:
            n = int((gl["detect_frac"] >= thr).sum())
            print(f"    lncRNA 检出率 >= {thr:>4}: {n:>5,} ({100*n/max(len(gl),1):.1f}%)")

    # ---- 非编码总量 ----
    noncoding = ["lncRNA", "processed_pseudogene", "unprocessed_pseudogene",
                 "transcribed_unprocessed_pseudogene", "transcribed_processed_pseudogene",
                 "pseudogene", "TEC", "miRNA", "snoRNA", "snRNA", "misc_RNA", "rRNA", "Mt_tRNA", "Mt_rRNA"]
    wt = panel_df[panel_df["panel"] == "WT"]
    n_nc = wt["gene_biotype"].isin(noncoding).sum()
    print(f"\nWT 面板非编码类基因合计: {n_nc:,} ({100*n_nc/len(wt):.2f}%)")


if __name__ == "__main__":
    main()
