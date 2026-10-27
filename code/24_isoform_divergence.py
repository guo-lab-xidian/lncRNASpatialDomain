# -*- coding: utf-8 -*-
"""
H3：lncRNA 异构体的空间分化分析。

问题：同一 lncRNA 基因的不同异构体，是否定位于不同的空间域/细胞类型？

数据：ISO 面板 Codebook_3，9,754 个转录本特征；其中 307 个来自 142 个 lncRNA 基因
     （每个 lncRNA 基因恰好 2 个异构体探针），蛋白编码基因作为阳性对照。

指标：
  1. 异构体间共检出（Phi 相关系数）
  2. 域偏好差异（两个异构体在 K 个域上检出率向量的 1 − cosine 相似度）
  3. 检出率比值 log2(f_iso1 / f_iso2)

核心比较：lncRNA 双异构体的"分歧程度"是否高于蛋白编码双异构体（后者应高度一致）。

产出：results/H3/ 下的表与图
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import vstack
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import spatial_io as sio

KEY = "cellxgene_ISO"
RES = r"E:\研究工作\agent长非与空间域\lncRNA空间域\results"
LDIR = os.path.join(RES, "Ldomain")
CLS = r"E:\研究工作\agent长非与空间域\analysis\03_lncrna\panel_features_class.csv"
OUT = os.path.join(RES, "H3")
os.makedirs(OUT, exist_ok=True)

MIN_ISO_DETECT = 0.01      # 至少一个异构体检出率 ≥1%


def phi(a, b):
    """两个二值向量的 Phi 相关系数（等价 Pearson）。"""
    a = a.astype(bool)
    b = b.astype(bool)
    n11 = (a & b).sum()
    n10 = (a & ~b).sum()
    n01 = (~a & b).sum()
    n00 = (~a & ~b).sum()
    den = np.sqrt((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00))
    return float((n11 * n00 - n10 * n01) / den) if den > 0 else np.nan


def cosine_sim(u, v):
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu == 0 or nv == 0:
        return np.nan
    return float(np.dot(u, v) / (nu * nv))


def main():
    pf = pd.read_csv(CLS, index_col=0)
    iso = pf[(pf["panel"] == "ISO") & (pf["feature_type"] == "transcript")].copy()
    print(f"转录本特征: {len(iso):,}，来自 {iso['gene_id'].nunique():,} 个基因")

    # 每个基因的异构体数（本面板中）
    per_gene = iso.groupby("gene_id").agg(
        n_iso=("detect_frac", "size"),
        gene_name=("gene_name", "first"),
        gene_class=("gene_class", "first"),
        max_det=("detect_frac", "max"),
    )
    pairs = per_gene[(per_gene["n_iso"] == 2) & (per_gene["max_det"] >= MIN_ISO_DETECT)]
    print(f"双异构体且至少一个检出率≥{MIN_ISO_DETECT}: {len(pairs)} 个基因 "
          f"（lncRNA {int((pairs['gene_class']=='lncRNA').sum())}，"
          f"蛋白编码 {int((pairs['gene_class']=='protein_coding').sum())}）")

    genes = list(pairs.index)
    feats = iso[iso["gene_id"].isin(genes)].index.tolist()
    print(f"需读取的转录本特征: {len(feats)}")

    # 读取子集表达
    lab = pd.read_csv(os.path.join(LDIR, "domain_labels.csv"), index_col=0)
    ranges = json.load(open(os.path.join(LDIR, "subset_rows.json")))
    var = sio.read_var(KEY)
    pos = var.index.get_indexer(feats)
    assert (pos >= 0).all()
    blocks = [sio.read_rows(KEY, r["start"], r["stop"])[:, pos].tocsr() for r in ranges]
    X = (vstack(blocks, format="csr") > 0).astype(np.float32).toarray()
    print(f"表达矩阵: {X.shape} | 阳性比例 {X.mean():.4f}")

    dom_codes, dom_names = pd.factorize(lab["L_domain"].astype(str))
    K = len(dom_names)
    n_dom = np.bincount(dom_codes, minlength=K)

    rows = []
    feat_pos = {f: i for i, f in enumerate(feats)}
    for gid, r in pairs.iterrows():
        sub = iso[iso["gene_id"] == gid].sort_values("detect_frac", ascending=False)
        if len(sub) != 2:
            continue
        f1, f2 = sub.index[0], sub.index[1]
        a, b = X[:, feat_pos[f1]], X[:, feat_pos[f2]]
        if a.sum() < 3 or b.sum() < 3:
            continue
        f1_dom = np.bincount(dom_codes, weights=a, minlength=K) / n_dom
        f2_dom = np.bincount(dom_codes, weights=b, minlength=K) / n_dom
        rows.append({
            "gene_id": gid, "gene_name": r["gene_name"], "gene_class": r["gene_class"],
            "iso1": f1, "iso2": f2,
            "det1": float(a.mean()), "det2": float(b.mean()),
            "log2_ratio": float(np.log2((a.mean() + 1e-6) / (b.mean() + 1e-6))),
            "phi": phi(a, b),
            "domain_cosine": cosine_sim(f1_dom, f2_dom),
            "domain_bias": 1 - cosine_sim(f1_dom, f2_dom)
            if np.isfinite(cosine_sim(f1_dom, f2_dom)) else np.nan,
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "isoform_pairs.csv"), index=False)
    print(f"\n可分析的双异构体基因: {len(df)}")

    print("\n=== 按基因类别 ===")
    summ = df.groupby("gene_class").agg(
        n=("gene_id", "size"),
        phi_median=("phi", "median"),
        cosine_median=("domain_cosine", "median"),
        abs_log2_median=("log2_ratio", lambda s: s.abs().median()),
    )
    print(summ.round(3).to_string())
    summ.to_csv(os.path.join(OUT, "isoform_summary_by_class.csv"))

    if {"lncRNA", "protein_coding"} <= set(df["gene_class"]):
        a = df.loc[df["gene_class"] == "lncRNA", "domain_cosine"].dropna()
        b = df.loc[df["gene_class"] == "protein_coding", "domain_cosine"].dropna()
        u = stats.mannwhitneyu(a, b, alternative="less")
        print(f"\n域偏好一致性: lncRNA 中位 {a.median():.3f} (n={len(a)}) vs "
              f"蛋白编码 {b.median():.3f} (n={len(b)})")
        print(f"Mann-Whitney（单侧 lncRNA 更不一致）p = {u.pvalue:.3e}")

    print("\n=== lncRNA 双异构体中分歧最大的 15 个 ===")
    l = df[df["gene_class"] == "lncRNA"].sort_values("domain_cosine")
    print(l.head(15)[["gene_name", "det1", "det2", "log2_ratio", "phi",
                      "domain_cosine"]].round(4).to_string(index=False))

    # 主案例：检出率差异最大的 lncRNA
    if len(l):
        case = l.reindex(l["log2_ratio"].abs().sort_values(ascending=False).index).iloc[0]
        print(f"\n=== 主案例 {case['gene_name']} ===")
        print(f"  异构体1 {case['iso1']}: 检出率 {case['det1']:.4f}")
        print(f"  异构体2 {case['iso2']}: 检出率 {case['det2']:.4f}")
        print(f"  log2 比值 {case['log2_ratio']:.2f} | Phi {case['phi']:.4f} "
              f"| 域偏好 cosine {case['domain_cosine']:.4f}")

        # 空间图 + 域分布
        i1, i2 = feat_pos[case["iso1"]], feat_pos[case["iso2"]]
        fig, axes = plt.subplots(2, 3, figsize=(16, 8))
        coords = lab[["global_x", "global_y"]].to_numpy()
        sample_arr = lab["sample"].astype(str).to_numpy()
        for row, (idx, label) in enumerate([(i1, "isoform 1"), (i2, "isoform 2")]):
            v = X[:, idx]
            for col, s in enumerate(sorted(set(sample_arr))):
                m = sample_arr == s
                ax = axes[row][col]
                ax.scatter(coords[m, 0], coords[m, 1], s=0.15, c="lightgrey", linewidths=0,
                           rasterized=True)
                hit = m & (v > 0)
                ax.scatter(coords[hit, 0], coords[hit, 1], s=1.0, c="#D1495B",
                           linewidths=0, rasterized=True)
                ax.set_aspect("equal")
                ax.set_xticks([]); ax.set_yticks([])
                ax.set_title(f"{label} | sample {s} (n={hit.sum():,})", fontsize=9)
        fig.suptitle(f"{case['gene_name']}: two isoforms of the same lncRNA gene")
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "fig_case_isoforms.png"), dpi=150)
        plt.close(fig)

    print(f"\n=== H3 完成，输出: {OUT} ===")


if __name__ == "__main__":
    main()
