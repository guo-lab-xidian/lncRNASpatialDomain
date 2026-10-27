# -*- coding: utf-8 -*-
"""生成 Supplementary Figures S1-S6（英文标注）。"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

R = r"E:\研究工作\agent长非与空间域\lncRNA空间域\results"
FIG = os.path.join(R, "figures", "supplementary")
os.makedirs(FIG, exist_ok=True)


def _dom_map(ax, labels_path, title, xcol=None, ycol=None):
    d = pd.read_csv(labels_path, index_col=0)
    d = d[d["sample"].astype(str) == d["sample"].astype(str).iloc[0]]
    xc = xcol or ("global_x" if "global_x" in d.columns else "pxl_row_in_fullres")
    yc = ycol or ("global_y" if "global_y" in d.columns else "pxl_col_in_fullres")
    doms = d["L_domain"].astype("category").cat.categories
    # 用 tab20 + tab20b 的深色变体拼出饱和调色板（tab20 的浅色一半太淡）
    palette = list(plt.cm.tab20(np.linspace(0, 1, 20))[::2]) + \
              list(plt.cm.tab20b(np.linspace(0, 1, 20))[::2]) + \
              list(plt.cm.tab20c(np.linspace(0, 1, 20))[::2])
    for j, dm in enumerate(doms):
        m = d["L_domain"] == dm
        ax.scatter(d.loc[m, xc], d.loc[m, yc], s=1.0, color=palette[j % len(palette)],
                   linewidths=0, rasterized=True)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=9)


# ------------------------------------------------------------------ S1
def s1_qc():
    qc = r"E:\研究工作\agent长非与空间域\analysis\01_qc"
    files = [("cell_qc_WT.csv", "MERFISH WT"), ("cell_qc_ISO.csv", "MERFISH ISO")]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for i, (fn, name) in enumerate(files):
        p = os.path.join(qc, fn)
        if not os.path.exists(p):
            continue
        d = pd.read_csv(p, index_col=0, low_memory=False)
        for j, (col, xl) in enumerate([("total_counts", "total counts per cell"),
                                       ("n_detected", "detected genes per cell")]):
            ax = axes[i][j]
            v = pd.to_numeric(d[col], errors="coerce").dropna()
            ax.hist(v, bins=80, color="#4C78A8")
            ax.set_yscale("log")
            ax.set_xlabel(xl, fontsize=8)
            ax.set_ylabel("cells", fontsize=8)
            ax.set_title(f"{name} (n={len(v):,}; median {v.median():.0f})", fontsize=9)
    fig.suptitle("Figure S1 | Per-cell quality control metrics", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "figS1_qc.png"), dpi=180)
    plt.close(fig)
    print("S1 done")


# ------------------------------------------------------------------ S2
def s2_domains():
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))
    _dom_map(axes[0], os.path.join(R, "Ldomain", "domain_labels.csv"),
             "MERFISH ISO (25 domains)")
    _dom_map(axes[1], os.path.join(R, "Ldomain_WT", "domain_labels.csv"),
             "MERFISH WT (21 domains)")
    _dom_map(axes[2], os.path.join(R, "Visium", "domain_labels.csv"),
             "Visium mouse (30 domains)")
    fig.suptitle("Figure S2 | Spatial domains (L-domain) per dataset", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "figS2_domains.png"), dpi=180)
    plt.close(fig)
    print("S2 done")


# ------------------------------------------------------------------ S3
def s3_cis():
    """H2 顺式邻域耦合的负结果。"""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    ax = axes[0]
    for tag, color, label in [("lncRNA", "#D1495B", "lncRNA"),
                              ("PC", "#4C78A8", "protein-coding (control)")]:
        p = os.path.join(R, "H2", f"cis_pairs_{tag}_250kb.csv")
        if not os.path.exists(p):
            continue
        d = pd.read_csv(p)["delta_raw"].dropna()
        ax.hist(d, bins=40, alpha=0.6, color=color, label=f"{label} (median {d.median():+.4f})")
    ax.axvline(0, color="k", ls="--", lw=1)
    ax.set_xlabel("Δ correlation (cis neighbour − detection-matched null)")
    ax.set_ylabel("gene pairs")
    ax.set_title("(a) Pairwise cis coupling at ±250 kb", fontsize=9)
    ax.legend(fontsize=7)

    ax = axes[1]
    p = os.path.join(R, "Method", "rank_compare.csv")
    if os.path.exists(p):
        d = pd.read_csv(p)
        sub = d[d["group"] == "lncRNA"]
        x = np.arange(len(sub))
        ax.bar(x, sub["spearman_cross_panel"], color="#59A14F")
        ax.set_xticks(x)
        ax.set_xticklabels(sub["method"], rotation=20, fontsize=8)
        ax.set_ylabel("cross-panel Spearman ρ")
        ax.set_title("(b) Effect-size estimators vs Pearson statistic", fontsize=9)
        ax.set_ylim(0, 1)
    fig.suptitle("Figure S3 | Supporting negative results", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "figS3_negative.png"), dpi=180)
    plt.close(fig)
    print("S3 done")


# ------------------------------------------------------------------ S4
def s4_isoform():
    p = os.path.join(R, "H3", "isoform_pairs.csv")
    if not os.path.exists(p):
        print("S4 skipped (no file)")
        return
    d = pd.read_csv(p)
    fig, ax = plt.subplots(figsize=(6.4, 4.3))
    for cls, color in [("protein_coding", "#4C78A8"), ("lncRNA", "#D1495B")]:
        sub = d[d["gene_class"] == cls]
        if not len(sub):
            continue
        ax.hist(sub["domain_cosine"].dropna(), bins=40, alpha=0.6, color=color,
                label=f"{cls} (n={len(sub)}, median {sub['domain_cosine'].median():.3f})")
    ax.set_xlabel("cosine similarity of isoform domain profiles")
    ax.set_ylabel("genes")
    ax.set_title("Figure S4 | Isoform pairs show no lncRNA-specific divergence", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "figS4_isoform.png"), dpi=180)
    plt.close(fig)
    print("S4 done")


# ------------------------------------------------------------------ S5
def s5_method():
    p = os.path.join(R, "Method", "simulation_summary.csv")
    if not os.path.exists(p):
        print("S5 skipped")
        return
    d = pd.read_csv(p)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for m, color in [("mcpearson", "#4C78A8"), ("chisq", "#E49444")]:
        col = f"{m}_FPR_null"
        if col in d:
            axes[0].plot(d["pi"] * 100, d[col], "o-", color=color, label=m)
        col = f"{m}_power"
        if col in d:
            axes[1].plot(d["pi"] * 100, d[col], "o-", color=color, label=m)
    axes[0].axhline(0.05, color="grey", ls="--", lw=1, label="nominal 5%")
    axes[0].set_xscale("log"); axes[0].set_xlabel("detection rate π (%)")
    axes[0].set_ylabel("empirical FPR (null genes)")
    axes[0].set_title("(a) Calibration", fontsize=9); axes[0].legend(fontsize=7)
    axes[1].set_xscale("log"); axes[1].set_xlabel("detection rate π (%)")
    axes[1].set_ylabel("power (FDR = 5%)")
    axes[1].set_title("(b) Power", fontsize=9); axes[1].legend(fontsize=7)
    fig.suptitle("Figure S5 | Calibration of domain-specificity tests under sparsity", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "figS5_simulation.png"), dpi=180)
    plt.close(fig)
    print("S5 done")


# ------------------------------------------------------------------ S6
def s6_allen():
    p = os.path.join(R, "AllenISH", "ish_correlation.csv")
    if not os.path.exists(p):
        print("S6 skipped")
        return
    d = pd.read_csv(p)
    d = d[d["spearman"].notna()]
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#4C78A8" if c else "#D1495B" for c in d.get("is_control", [False] * len(d))]
    ax.barh(d["gene"], d["spearman"], color=colors)
    ax.axvline(0, color="k", lw=1)
    ax.set_xlabel("Spearman ρ (local detection vs Allen ISH expression energy)")
    ax.set_title("Figure S6 | Allen ISH validation attempt failed for controls too\n"
                 "blue = positive-control genes with textbook spatial patterns", fontsize=9)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "figS6_allen.png"), dpi=180)
    plt.close(fig)
    print("S6 done")


if __name__ == "__main__":
    s1_qc(); s2_domains(); s3_cis(); s4_isoform(); s5_method(); s6_allen()
    print(f"\n输出目录: {FIG}")
