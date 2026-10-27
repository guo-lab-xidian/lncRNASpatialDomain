# -*- coding: utf-8 -*-
"""
用 gseapy 对 16 个 lncRNA 的共表达模块做 GO 富集（重试版，替代 25 脚本里的手写 HTTP 调用）。
"""
import os
import time

import pandas as pd
import gseapy as gp

RES = r"E:\研究工作\agent长非与空间域\lncRNA空间域\results"
ANN = os.path.join(RES, "annotation")

coex = pd.read_csv(os.path.join(ANN, "coexpression_depth_adjusted.csv"), index_col=0)
print(f"待富集基因数: {len(coex)}")

rows = []
for g, r in coex.iterrows():
    genes = str(r["top200"]).split(";") if pd.notna(r["top200"]) else []
    genes = [x for x in genes if x and not x.startswith("Gm")]
    if len(genes) < 15:
        print(f"  [{g}] 有效基因太少 ({len(genes)})，跳过")
        continue
    try:
        enr = gp.enrichr(gene_list=genes,
                         gene_sets=["GO_Biological_Process_2023"],
                         organism="mouse",
                         outdir=None, no_plot=True)
        df = enr.results
        df = df[df["Adjusted P-value"] < 0.05].sort_values("Adjusted P-value")
        if len(df):
            rows.append({
                "gene_name": g, "n_query": len(genes),
                "top_term": df.iloc[0]["Term"],
                "adj_p": df.iloc[0]["Adjusted P-value"],
                "top5_terms": " | ".join(df["Term"].head(5)),
            })
            print(f"  [{g}] {len(df)} 个显著条目 | 首位: {df.iloc[0]['Term']}")
        else:
            rows.append({"gene_name": g, "n_query": len(genes), "top_term": "",
                         "adj_p": float("nan"), "top5_terms": ""})
            print(f"  [{g}] 无显著条目")
    except Exception as e:
        print(f"  [{g}] 失败: {e}")
    time.sleep(0.5)

if rows:
    go = pd.DataFrame(rows).set_index("gene_name")
    go.to_csv(os.path.join(ANN, "go_enrichment.csv"))
    print(f"\n=== GO 富集完成，{len(go)} 个基因有结果 ===")
    print(go[["n_query", "top_term", "adj_p"]].to_string())
else:
    print("无结果")



