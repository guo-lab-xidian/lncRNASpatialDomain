# -*- coding: utf-8 -*-
"""
生成补充图 S1–S6（投稿规格，与 64_figures_bib.py 同一套排版参数）。
输出 results/figures/bib/figS*.pdf 与 600dpi PNG。
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

R = r"E:\研究工作\agent长非与空间域\lncRNA空间域\results"
OUT = os.path.join(R, "figures", "bib")
os.makedirs(OUT, exist_ok=True)
FULL, HALF = 180 / 25.4, 88 / 25.4

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "lines.linewidth": 0.8, "lines.markersize": 2.5,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.01,
})


def save(fig, name):
    fig.savefig(os.path.join(OUT, f"{name}.pdf"))
    fig.savefig(os.path.join(OUT, f"{name}.png"), dpi=600)
    plt.close(fig)
    print(f"  {name}.pdf/.png")


def _dom_map(ax, path, title):
    d = pd.read_csv(path, index_col=0)
    d = d[d["sample"].astype(str) == d["sample"].astype(str).iloc[0]]
    xc = "global_x" if "global_x" in d.columns else "pxl_row_in_fullres"
    yc = "global_y" if "global_y" in d.columns else "pxl_col_in_fullres"
    doms = d["L_domain"].astype("category").cat.categories
    palette = list(plt.cm.tab20(np.linspace(0, 1, 20))[::2]) + \
              list(plt.cm.tab20b(np.linspace(0, 1, 20))[::2]) + \
              list(plt.cm.tab20c(np.linspace(0, 1, 20))[::2])
    for j, dm in enumerate(doms):
        m = d["L_domain"] == dm
        ax.scatter(d.loc[m, xc], d.loc[m, yc], s=0.25,
                   color=palette[j % len(palette)], linewidths=0, rasterized=True)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=6.5, linespacing=1.15)


def s1():
    qc = r"E:\研究工作\agent长非与空间域\analysis\01_qc"
    files = [("cell_qc_WT.csv", "MERFISH WT"), ("cell_qc_ISO.csv", "MERFISH ISO")]
    fig, axes = plt.subplots(2, 2, figsize=(FULL * 0.7, FULL * 0.55))
    for i, (fn, name) in enumerate(files):
        p = os.path.join(qc, fn)
        if not os.path.exists(p):
            continue
        d = pd.read_csv(p, index_col=0, low_memory=False)
        for j, (col, xl) in enumerate([("total_counts", "total counts per cell"),
                                       ("n_detected", "detected genes per cell")]):
            ax = axes[i][j]
            v = pd.to_numeric(d[col], errors="coerce").dropna()
            ax.hist(v, bins=80, color="#4C78A8", linewidth=0)
            ax.set_yscale("log")
            ax.set_xlabel(xl)
            ax.set_ylabel("cells")
            ax.set_title(f"{name} (n={len(v):,}; median {v.median():.0f})", fontsize=6.5)
    fig.tight_layout(pad=0.3, w_pad=0.8, h_pad=0.8)
    save(fig, "figS1_qc")


def s2():
    # 按各数据集真实长宽比分配宽度，使三幅图等高，避免 Visium 面板出现大片留白
    #   MERFISH 宽/高 ≈ 0.29，Visium ≈ 0.97
    fig = plt.figure(figsize=(FULL, FULL * 0.72))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.291, 0.294, 0.967], wspace=0.08)
    for k, (path, title) in enumerate([
            (os.path.join(R, "Ldomain", "domain_labels.csv"), "MERFISH ISO\n(25 domains)"),
            (os.path.join(R, "Ldomain_WT", "domain_labels.csv"), "MERFISH WT\n(21 domains)"),
            (os.path.join(R, "Visium", "domain_labels.csv"), "Visium mouse\n(30 domains)")]):
        _dom_map(fig.add_subplot(gs[0, k]), path, title)
    save(fig, "figS2_domains")


def s3():
    fig, axes = plt.subplots(1, 2, figsize=(FULL * 0.78, FULL * 0.32))
    ax = axes[0]
    for tag, color, label in [("lncRNA", "#D1495B", "lncRNA"),
                              ("PC", "#4C78A8", "protein-coding (control)")]:
        p = os.path.join(R, "H2", f"cis_pairs_{tag}_250kb.csv")
        if not os.path.exists(p):
            continue
        d = pd.read_csv(p)["delta_raw"].dropna()
        ax.hist(d, bins=40, alpha=0.6, color=color, linewidth=0,
                label=f"{label} (median {d.median():+.4f})")
    ax.axvline(0, color="k", ls="--", lw=0.7)
    ax.set_xlabel("Δ correlation (cis neighbour − matched null)")
    ax.set_ylabel("gene pairs")
    ax.set_title("a | Pairwise cis coupling at ±250 kb", loc="left", fontsize=7)
    ax.legend(borderpad=0.2, labelspacing=0.25)

    ax = axes[1]
    p = os.path.join(R, "Method", "rank_compare.csv")
    if os.path.exists(p):
        d = pd.read_csv(p)
        sub = d[d["group"] == "lncRNA"]
        x = np.arange(len(sub))
        ax.bar(x, sub["spearman_cross_panel"], color="#59A14F", width=0.65)
        ax.set_xticks(x)
        ax.set_xticklabels(sub["method"], rotation=20, ha="right", fontsize=6)
        ax.set_ylabel("cross-panel Spearman ρ")
        ax.set_title("b | Estimators vs Pearson statistic", loc="left", fontsize=7)
        ax.set_ylim(0, 1)
    fig.tight_layout(pad=0.3, w_pad=1.5)
    save(fig, "figS3_negative")


def s4():
    p = os.path.join(R, "H3", "isoform_pairs.csv")
    if not os.path.exists(p):
        print("  S4 skipped (no file)")
        return
    d = pd.read_csv(p)
    fig, ax = plt.subplots(figsize=(HALF, HALF * 0.75))
    for cls, color in [("protein_coding", "#4C78A8"), ("lncRNA", "#D1495B")]:
        sub = d[d["gene_class"] == cls]
        if len(sub):
            ax.hist(sub["domain_cosine"].dropna(), bins=40, alpha=0.6, color=color,
                    linewidth=0,
                    label=f"{cls.replace('_', '-')} (n={len(sub)}, median "
                          f"{sub['domain_cosine'].median():.3f})")
    ax.set_xlabel("cosine similarity of isoform domain profiles")
    ax.set_ylabel("genes")
    ax.legend(borderpad=0.2, labelspacing=0.25)
    fig.tight_layout(pad=0.2)
    save(fig, "figS4_isoform")


def s5():
    p = os.path.join(R, "Method", "simulation_summary.csv")
    if not os.path.exists(p):
        print("  S5 skipped")
        return
    d = pd.read_csv(p)
    fig, axes = plt.subplots(1, 2, figsize=(FULL * 0.70, FULL * 0.30))
    for m, color in [("mcpearson", "#4C78A8"), ("chisq", "#E49444")]:
        if f"{m}_FPR_null" in d:
            axes[0].plot(d["pi"] * 100, d[f"{m}_FPR_null"], "o-", color=color, label=m)
        if f"{m}_power" in d:
            axes[1].plot(d["pi"] * 100, d[f"{m}_power"], "o-", color=color, label=m)
    axes[0].axhline(0.05, color="grey", ls="--", lw=0.7, label="nominal 5%")
    for ax, yl, t in [(axes[0], "empirical FPR (null genes)", "a | Calibration"),
                      (axes[1], "power (FDR = 5%)", "b | Power")]:
        ax.set_xscale("log")
        ax.set_xlabel("detection rate π (%)")
        ax.set_ylabel(yl)
        ax.set_title(t, loc="left", fontsize=7)
        ax.legend(borderpad=0.2, labelspacing=0.25)
    fig.tight_layout(pad=0.3, w_pad=1.5)
    save(fig, "figS5_simulation")


def s6():
    p = os.path.join(R, "AllenISH", "ish_correlation.csv")
    if not os.path.exists(p):
        print("  S6 skipped")
        return
    d = pd.read_csv(p)
    d = d[d["spearman"].notna()]
    fig, ax = plt.subplots(figsize=(HALF, HALF * 0.85))
    colors = ["#4C78A8" if c else "#D1495B" for c in d.get("is_control", [False] * len(d))]
    ax.barh(d["gene"], d["spearman"], color=colors, height=0.65)
    ax.axvline(0, color="k", lw=0.6)
    ax.set_xlabel("Spearman ρ (local detection vs Allen ISH)")
    ax.tick_params(labelsize=6)
    fig.tight_layout(pad=0.2)
    save(fig, "figS6_allen")


if __name__ == "__main__":
    for fn in [s1, s2, s3, s4, s5, s6]:
        fn()
    print(f"\n输出目录: {OUT}")
