# -*- coding: utf-8 -*-
"""绘制论文主图 Fig 1–5（英文标注，投稿用）。"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

R = r"E:\研究工作\agent长非与空间域\lncRNA空间域\results"
FIG = os.path.join(R, "figures")
os.makedirs(FIG, exist_ok=True)

LNC = {"lncRNA", "lincRNA", "antisense", "processed_transcript",
       "sense_intronic", "sense_overlapping", "macro_lncRNA"}
PALETTE = {"lncRNA": "#D1495B", "protein_coding": "#4C78A8", "other": "#B0B0B0"}


def load_dataset(name):
    if name == "MERFISH ISO":
        s = pd.read_csv(os.path.join(R, "H1H2", "gene_specificity.csv"), index_col=0)
        s = s[~s.index.duplicated(keep="first")].rename(
            columns={"panel_detect": "detect", "DSI_domain": "DSI"})
    elif name == "MERFISH WT":
        s = pd.read_csv(os.path.join(R, "H1H2_WT", "gene_specificity.csv"), index_col=0)
        s = s[~s.index.duplicated(keep="first")].rename(
            columns={"panel_detect": "detect", "DSI_domain": "DSI"})
    elif name == "Visium mouse":
        s = pd.read_csv(os.path.join(R, "Visium", "gene_specificity.csv"), index_col=0)
        s = s[~s.index.duplicated(keep="first")].rename(columns={"detect_frac": "detect"})
        pf = pd.read_csv(r"E:\研究工作\agent长非与空间域\analysis\03_lncrna\panel_features_class.csv",
                         index_col=0)
        pf = pf[~pf.index.duplicated(keep="first")]
        coords = pd.read_csv(os.path.join(R, "gene_coords_m25.tsv"), sep="\t")
        n2t = coords.drop_duplicates("gene_name").set_index("gene_name")["gene_type"]
        fb = n2t.reindex(s.index).map(lambda t: "lncRNA" if isinstance(t, str) and t in LNC
                                      else ("protein_coding" if t == "protein_coding" else "other"))
        s["gene_class"] = pf["gene_class"].reindex(s.index).fillna(fb)
    else:  # Visium human
        s = pd.read_csv(os.path.join(R, "VisiumHuman", "gene_specificity.csv"), index_col=0)
        s = s[~s.index.duplicated(keep="first")].rename(columns={"detect_frac": "detect"})
    return s[["gene_class", "detect", "DSI"]].dropna()


DATASETS = ["MERFISH ISO", "MERFISH WT", "Visium mouse", "Visium human"]


# ------------------------------------------------------------------ Fig 1
def fig1():
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for i, name in enumerate(DATASETS):
        s = load_dataset(name)
        ax = axes[0][i] if i < 3 else axes[1][0]
        for cls, color in [("protein_coding", PALETTE["protein_coding"]),
                           ("lncRNA", PALETTE["lncRNA"])]:
            v = s.loc[s.gene_class == cls, "detect"]
            ax.hist(np.log10(v.clip(1e-4)), bins=60, alpha=0.6, color=color,
                    label=f"{cls} (panel n={len(v):,})", density=True)
        ax.set_xlabel("log10 detection rate")
        ax.set_ylabel("density")
        ax.set_title(name, fontsize=10)
        ax.legend(fontsize=7)
    # 统计面板
    ax = axes[1][1]
    counts = {"lncRNA": [], "protein_coding": []}
    for name in DATASETS:
        s = load_dataset(name)
        for cls in counts:
            counts[cls].append(int(((s.gene_class == cls) & (s.detect >= 0.01)).sum()))
    x = np.arange(len(DATASETS))
    ax.bar(x - 0.2, counts["lncRNA"], width=0.4, color=PALETTE["lncRNA"], label="lncRNA")
    ax.bar(x + 0.2, counts["protein_coding"], width=0.4, color=PALETTE["protein_coding"],
           label="protein-coding")
    ax.set_xticks(x)
    ax.set_xticklabels([d.replace(" ", "\n") for d in DATASETS], fontsize=8)
    ax.set_ylabel("genes with detection ≥ 1%")
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.set_title("Analyzable genes per dataset", fontsize=10)

    # 空间域示例
    ax = axes[1][2]
    lab = pd.read_csv(os.path.join(R, "Ldomain", "domain_labels.csv"), index_col=0)
    sub = lab[lab["sample"].astype(str) == lab["sample"].astype(str).iloc[0]]
    doms = sub["L_domain"].astype("category").cat.categories
    cmap = plt.cm.tab20(np.linspace(0, 1, len(doms)))
    for j, d in enumerate(doms):
        m = sub["L_domain"] == d
        ax.scatter(sub.loc[m, "global_x"], sub.loc[m, "global_y"], s=0.8, color=cmap[j],
                   linewidths=0, rasterized=True)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("MERFISH spatial domains (L-domain)", fontsize=10)

    fig.suptitle("Fig 1 | Data overview: detection depth and analyzable lncRNAs", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig1_overview.png"), dpi=200)
    plt.close(fig)
    print("Fig1 完成")


# ------------------------------------------------------------------ Fig 2
def fig2():
    """检出率混杂：同一数据下，阈值改变会让不匹配比较的方向反转。"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

    # A: DSI 随检出率上升
    ax = axes[0]
    for name, color in [("MERFISH ISO", "#4C78A8"), ("Visium mouse", "#E49444")]:
        s = load_dataset(name)
        s = s[s.detect >= 0.01]
        bins = np.logspace(np.log10(0.01), 0, 15)
        idx = np.digitize(s["detect"], bins)
        med = [s.loc[idx == k, "DSI"].median() for k in range(1, len(bins))]
        ctr = [np.sqrt(bins[k - 1] * bins[k]) for k in range(1, len(bins))]
        ax.plot(ctr, med, "o-", color=color, label=name, ms=3)
    ax.set_xscale("log")
    ax.set_xlabel("detection rate")
    ax.set_ylabel("median DSI")
    ax.set_title("A | Domain specificity rises with detection", fontsize=9)
    ax.legend(fontsize=7)

    # B: 不匹配 vs 匹配，两种阈值
    ax = axes[1]
    labels, vals, colors = [], [], []
    for name in ["MERFISH ISO", "MERFISH WT"]:
        s = load_dataset(name)
        for thr in [0.01, 0.05]:
            lnc = s[(s.gene_class == "lncRNA") & (s.detect >= thr)]["DSI"]
            pc = s[(s.gene_class == "protein_coding") & (s.detect >= thr)]["DSI"]
            ratio = lnc.median() / pc.median()
            labels.append(f"{name.split()[-1]}\n≥{thr:.0%}")
            vals.append(ratio)
            colors.append("#D1495B" if ratio > 1 else "#4C78A8")
    x = np.arange(len(vals))
    ax.bar(x, vals, color=colors)
    ax.axhline(1, color="k", ls="--", lw=1)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("median DSI ratio\n(lncRNA / protein-coding)")
    ax.set_title("B | Unmatched comparison flips with threshold", fontsize=9)

    # C: 匹配后无差异
    ax = axes[2]
    rows = []
    for name in DATASETS:
        # 每个数据集独立种子，保证对照抽样与表格/敏感性分析逐位一致
        rng = np.random.default_rng(0)
        boot_rng = np.random.default_rng(1)
        s = load_dataset(name)
        lnc = s[(s.gene_class == "lncRNA") & (s.detect >= 0.01)]
        pc = s[(s.gene_class == "protein_coding") & (s.detect >= 0.01)]
        d = []
        for _, r in lnc.iterrows():
            sel = pc[(pc.detect >= r.detect * 0.8) & (pc.detect <= r.detect * 1.2)]
            if len(sel) < 5:
                sel = pc.iloc[np.argsort(np.abs(pc.detect - r.detect))[:5]]
            ctrl = sel.sample(min(5, len(sel)), random_state=int(rng.integers(1e9)))
            d.append(r.DSI - ctrl.DSI.median())
        d = pd.Series(d).dropna().to_numpy()
        # 中位数 + bootstrap 置信区间（分布右偏，均值会被少数极端值主导）
        boot = np.array([np.median(boot_rng.choice(d, len(d), replace=True))
                         for _ in range(1000)])
        rows.append((name, float(np.median(d)),
                     float(np.median(d) - np.percentile(boot, 2.5)),
                     float(np.percentile(boot, 97.5) - np.median(d))))
    names = [r[0] for r in rows]
    meds = [r[1] for r in rows]
    lo = [r[2] for r in rows]
    hi = [r[3] for r in rows]
    y = np.arange(len(names))
    ax.errorbar(meds, y, xerr=[lo, hi], fmt="o", color="#D1495B", capsize=3)
    ax.axvline(0, color="k", ls="--", lw=1)
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("ΔDSI, median ± 95% bootstrap CI", fontsize=8)
    ax.set_title("C | After matching, effect sizes are tiny", fontsize=9)
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax.invert_yaxis()

    fig.suptitle("Fig 2 | Detection-rate confounding drives class comparisons", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig2_confounding.png"), dpi=200)
    plt.close(fig)
    print("Fig2 完成")


# ------------------------------------------------------------------ Fig 3
def fig3():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    # A 与 B 统一用均值口径（B 的交互项来自 OLS），避免中位数/均值口径不一致
    bins_all = {}
    for name, color in [("MERFISH ISO", "#4C78A8"), ("MERFISH WT", "#59A14F"),
                        ("Visium mouse", "#E49444")]:
        s = load_dataset(name)
        s = s[s.gene_class.isin(["lncRNA", "protein_coding"]) & (s.detect >= 0.01)].copy()
        s["bin"] = pd.qcut(s["detect"], 10, labels=False, duplicates="drop")
        xs, ys = [], []
        for b, sub in s.groupby("bin"):
            a = sub.loc[sub.gene_class == "lncRNA", "DSI"]
            c = sub.loc[sub.gene_class == "protein_coding", "DSI"]
            if len(a) < 5:
                continue
            xs.append(int(b)); ys.append(a.mean() - c.mean())
        bins_all[name] = (xs, ys)
    ax = axes[0]
    for name, (xs, ys) in bins_all.items():
        color = {"MERFISH ISO": "#4C78A8", "MERFISH WT": "#59A14F",
                 "Visium mouse": "#E49444"}[name]
        ax.plot(xs, ys, "o-", color=color, label=name, ms=4)
    ax.axhline(0, color="k", ls="--", lw=1)
    ax.set_xlabel("detection-rate decile (0 = lowest)")
    ax.set_ylabel("mean ΔDSI (lncRNA − protein-coding)")
    ax.set_title("A | Class difference by detection stratum", fontsize=9)
    ax.legend(fontsize=8)

    ax = axes[1]
    it = pd.read_csv(os.path.join(R, "Robustness", "interaction_test.csv"))
    y = np.arange(len(it))
    ax.errorbar(it["interaction"], y,
                xerr=[it["interaction"] - it["interaction_ci_lo"],
                      it["interaction_ci_hi"] - it["interaction"]],
                fmt="o", color="#D1495B", capsize=4)
    ax.axvline(0, color="k", ls="--", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{d}\n(p={p:.1e})" for d, p in
                        zip(it["dataset"], it["interaction_p"])], fontsize=8)
    ax.set_xlabel("class × log10(detection) interaction coefficient")
    ax.set_title("B | Linear interaction is a poor summary", fontsize=9)
    ax.invert_yaxis()

    fig.suptitle("Fig 3 | The class difference is non-monotonic and dataset-specific", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig3_stratified.png"), dpi=200)
    plt.close(fig)
    print("Fig3 完成")


# ------------------------------------------------------------------ Fig 5
def fig5():
    platforms = ["MERFISH\n(Cohen 2026)", "Visium V1\n(mouse)", "Visium V1\n(human)",
                 "Visium HD\n(mouse)", "Xenium Prime 5K\n(mouse)"]
    genes = [22930, 32285, 36601, 19059, 5006]
    lncrna = [4740, 2312, 1361, 0, 4]
    pct = [100 * a / b for a, b in zip(lncrna, genes)]
    fig, ax = plt.subplots(figsize=(8, 4.4))
    x = np.arange(len(platforms))
    bars = ax.bar(x, pct, color=["#4C78A8", "#4C78A8", "#4C78A8", "#D1495B", "#D1495B"])
    for i, (b, n, t) in enumerate(zip(bars, lncrna, genes)):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.3,
                f"{n:,}/{t:,}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(platforms, fontsize=8)
    ax.set_ylabel("lncRNAs as % of panel")
    ax.set_title("Fig 5 | lncRNA coverage across spatial platforms\n"
                 "red = probe-enrichment panels", fontsize=10)
    ax.set_ylim(0, max(pct) * 1.25)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig5_platform.png"), dpi=200)
    plt.close(fig)
    print("Fig5 完成")


if __name__ == "__main__":
    fig1()
    fig2()
    fig3()
    fig5()
    print(f"输出目录: {FIG}")


