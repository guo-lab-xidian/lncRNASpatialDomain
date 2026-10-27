# -*- coding: utf-8 -*-
"""
生成投稿用图件（Briefings in Bioinformatics / OUP 规格）。

规格
----
  双栏宽 180 mm，单栏宽 88 mm
  输出矢量 PDF + 600 dpi PNG
  字号按**最终尺寸**设定：正文 7 pt，刻度 6.5 pt，图例 6 pt，最小不低于 6 pt
  线条 0.7 pt，坐标轴边框 0.6 pt

说明
----
OUP 作者指南页面当前被 Cloudflare 拦截，无法逐条核对；本脚本采用 OUP
通行规格。投稿前请对照当期 Instructions to Authors 复核宽度与格式。

输出
----
results/figures/bib/fig1.pdf … fig7.pdf 及对应 600dpi PNG
results/figures/bib/figS1…S6.pdf
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import spatial_io as sio

R = r"E:\研究工作\agent长非与空间域\lncRNA空间域\results"
OUT = os.path.join(R, "figures", "bib")
os.makedirs(OUT, exist_ok=True)

LNC = {"lncRNA", "lincRNA", "antisense", "processed_transcript",
       "sense_intronic", "sense_overlapping", "macro_lncRNA"}
PAL = {"lncRNA": "#D1495B", "protein_coding": "#4C78A8", "other": "#B0B0B0"}
DS_COLOR = {"MERFISH ISO": "#4C78A8", "MERFISH WT": "#59A14F", "Visium mouse": "#E49444"}
FULL, HALF = 180 / 25.4, 88 / 25.4          # inch

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7,
    "axes.titlesize": 7,
    "axes.labelsize": 7,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "lines.linewidth": 0.8,
    "lines.markersize": 2.5,
    "pdf.fonttype": 42,          # TrueType，避免 Illustrator 乱码
    "ps.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.01,
})

CATALOG = ["Cdr1", "Gm49698", "Gm27000", "Gm48623", "Miat", "Gm47221",
           "Gm35240", "Gm42428", "Gm49124", "Gm45895", "B230220B15Rik",
           "Gm47408", "Gm45163", "A830036E02Rik", "Gm49655"]


def save(fig, name):
    for ext, kw in [("pdf", {}), ("png", {"dpi": 600})]:
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), **kw)
    plt.close(fig)
    print(f"  {name}: {os.path.join(OUT, name)}.pdf/.png")


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
    else:
        s = pd.read_csv(os.path.join(R, "VisiumHuman", "gene_specificity.csv"), index_col=0)
        s = s[~s.index.duplicated(keep="first")].rename(columns={"detect_frac": "detect"})
    return s[["gene_class", "detect", "DSI"]].dropna()


DATASETS = ["MERFISH ISO", "MERFISH WT", "Visium mouse", "Visium human"]


# ------------------------------------------------------------------ Fig 1
def fig1():
    # 2 行 x 3 列：上排 3 个直方图，下排第 4 个 + 跨两列的柱状图。
    # 原先的域图面板已移出：切片长宽比约 1:3.4，在小版面里会被压成细条，
    # 且 Fig S2 已完整展示三个数据集的域划分。
    fig = plt.figure(figsize=(FULL, FULL * 0.50))
    gs = fig.add_gridspec(2, 3, hspace=0.55, wspace=0.35)
    slots = [gs[0, 0], gs[0, 1], gs[0, 2], gs[1, 0]]
    for i, name in enumerate(DATASETS):
        s = load_dataset(name)
        ax = fig.add_subplot(slots[i])
        for cls in ["protein_coding", "lncRNA"]:
            v = s.loc[s.gene_class == cls, "detect"]
            ax.hist(np.log10(v.clip(1e-4)), bins=50, alpha=0.6, color=PAL[cls],
                    density=True, linewidth=0,
                    label=f"{cls.replace('_', '-')} (n={len(v):,})")
        ax.set_xlabel("log$_{10}$ detection rate")
        ax.set_ylabel("density")
        ax.set_title(name, fontsize=7)
        ax.legend(loc="upper left", handlelength=1.0, borderpad=0.2,
                  labelspacing=0.25, handletextpad=0.4)

    ax = fig.add_subplot(gs[1, 1:])
    counts = {"lncRNA": [], "protein_coding": []}
    for name in DATASETS:
        s = load_dataset(name)
        for cls in counts:
            counts[cls].append(int(((s.gene_class == cls) & (s.detect >= 0.01)).sum()))
    x = np.arange(len(DATASETS))
    ax.bar(x - 0.2, counts["lncRNA"], width=0.4, color=PAL["lncRNA"], label="lncRNA")
    ax.bar(x + 0.2, counts["protein_coding"], width=0.4,
           color=PAL["protein_coding"], label="protein-coding")
    ax.set_xticks(x)
    ax.set_xticklabels([d.replace(" ", "\n") for d in DATASETS], fontsize=6)
    ax.set_ylabel("genes detected ≥ 1%")
    ax.set_yscale("log")
    ax.legend(handlelength=1.0, borderpad=0.2, labelspacing=0.25)
    ax.set_title("Analyzable genes per dataset", fontsize=7)

    save(fig, "fig1_overview")


# ------------------------------------------------------------------ Fig 2
def fig2():
    fig, axes = plt.subplots(1, 3, figsize=(FULL, FULL * 0.33))
    ax = axes[0]
    for name in ["MERFISH ISO", "Visium mouse"]:
        s = load_dataset(name)
        s = s[s.detect >= 0.01]
        bins = np.logspace(np.log10(0.01), 0, 15)
        idx = np.digitize(s["detect"], bins)
        med = [s.loc[idx == k, "DSI"].median() for k in range(1, len(bins))]
        ctr = [np.sqrt(bins[k - 1] * bins[k]) for k in range(1, len(bins))]
        ax.plot(ctr, med, "o-", color=DS_COLOR[name], ms=2.2, label=name)
    ax.set_xscale("log")
    ax.set_xlabel("detection rate")
    ax.set_ylabel("median DSI")
    ax.set_title("a | DSI rises with detection", loc="left", fontsize=7)
    ax.legend(borderpad=0.2, labelspacing=0.25)

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
    ax.bar(np.arange(len(vals)), vals, color=colors, width=0.65)
    ax.axhline(1, color="k", ls="--", lw=0.7)
    ax.set_xticks(np.arange(len(vals)))
    ax.set_xticklabels(labels, fontsize=6)
    ax.set_ylabel("median DSI ratio\n(lncRNA / protein-coding)")
    ax.set_title("b | Unmatched comparison flips", loc="left", fontsize=7)

    ax = axes[2]
    rows = []
    for name in DATASETS:
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
        boot = np.array([np.median(boot_rng.choice(d, len(d), replace=True)) for _ in range(1000)])
        rows.append((name, float(np.median(d)),
                     float(np.median(d) - np.percentile(boot, 2.5)),
                     float(np.percentile(boot, 97.5) - np.median(d))))
    y = np.arange(len(rows))
    ax.errorbar([r[1] for r in rows], y, xerr=[[r[2] for r in rows], [r[3] for r in rows]],
                fmt="o", color="#D1495B", capsize=2, ms=3)
    ax.axvline(0, color="k", ls="--", lw=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=6)
    ax.set_xlabel("ΔDSI, median ± 95% CI")
    ax.set_title("c | After matching, effects are tiny", loc="left", fontsize=7)
    ax.xaxis.set_major_locator(plt.MaxNLocator(4))
    ax.invert_yaxis()

    fig.tight_layout(pad=0.3, w_pad=1.2)
    save(fig, "fig2_confounding")


# ------------------------------------------------------------------ Fig 3
def fig3():
    fig, axes = plt.subplots(1, 2, figsize=(FULL * 0.78, FULL * 0.34))
    ax = axes[0]
    for name in ["MERFISH ISO", "MERFISH WT", "Visium mouse"]:
        s = load_dataset(name)
        s = s[s.gene_class.isin(["lncRNA", "protein_coding"]) & (s.detect >= 0.01)].copy()
        s["bin"] = pd.qcut(s["detect"], 10, labels=False, duplicates="drop")
        xs, ys = [], []
        for b, sub in s.groupby("bin"):
            a = sub.loc[sub.gene_class == "lncRNA", "DSI"]
            c = sub.loc[sub.gene_class == "protein_coding", "DSI"]
            if len(a) >= 5:
                xs.append(int(b)); ys.append(a.mean() - c.mean())
        ax.plot(xs, ys, "o-", color=DS_COLOR[name], ms=2.5, label=name)
    ax.axhline(0, color="k", ls="--", lw=0.7)
    ax.set_xlabel("detection-rate decile (0 = lowest)")
    ax.set_ylabel("mean ΔDSI (lncRNA − protein-coding)")
    ax.set_title("a | Class difference by detection stratum", loc="left", fontsize=7)
    ax.legend(borderpad=0.2, labelspacing=0.25)

    ax = axes[1]
    it = pd.read_csv(os.path.join(R, "Robustness", "interaction_test.csv"))
    y = np.arange(len(it))
    ax.errorbar(it["interaction"], y,
                xerr=[it["interaction"] - it["interaction_ci_lo"],
                      it["interaction_ci_hi"] - it["interaction"]],
                fmt="o", color="#D1495B", capsize=2.5, ms=3)
    ax.axvline(0, color="k", ls="--", lw=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{d}\n(p={p:.1e})" for d, p in
                        zip(it["dataset"], it["interaction_p"])], fontsize=6)
    ax.set_xlabel("class × log$_{10}$(detection) coefficient")
    ax.set_title("b | Linear interaction is a poor summary", loc="left", fontsize=7)
    ax.invert_yaxis()

    fig.tight_layout(pad=0.3, w_pad=1.5)
    save(fig, "fig3_stratified")


# ------------------------------------------------------------------ Fig 4
def fig4():
    gz = pd.read_csv(os.path.join(R, "Robustness", "gene_null_scores.csv"))
    gz = gz[~gz.duplicated(["dataset", "gene"])]
    na = pd.read_csv(os.path.join(R, "Robustness", "noise_adjusted_comparison.csv"))
    na["ds"] = na["dataset"].str.strip()
    NAMES = ["MERFISH ISO", "MERFISH WT", "Visium mouse"]

    fig, axes = plt.subplots(1, 3, figsize=(FULL, FULL * 0.33))

    ax = axes[0]
    iso = gz[(gz.dataset == "MERFISH ISO") & (gz.detect >= 0.004)].sort_values("detect")
    curve = iso.drop_duplicates("null_mean").sort_values("detect")
    ax.fill_between(curve["detect"], curve["null_mean"] - 1.96 * curve["null_sd"],
                    curve["null_mean"] + 1.96 * curve["null_sd"],
                    color="#BBBBBB", alpha=0.45, lw=0, label="null, 95% interval")
    ax.plot(curve["detect"], curve["null_mean"], "-", color="#333333", lw=1.0,
            label="null, mean")
    pc = iso[iso.gene_class == "protein_coding"]
    bins = np.logspace(np.log10(0.005), 0, 16)
    idx = np.digitize(pc["detect"], bins)
    ctr = [np.sqrt(bins[k - 1] * bins[k]) for k in range(1, len(bins))]
    med = [pc.loc[idx == k, "DSI"].median() for k in range(1, len(bins))]
    ax.plot(ctr, med, "o-", color=DS_COLOR["MERFISH ISO"], ms=2.2, lw=0.9,
            label="observed")
    ax.set_xscale("log")
    ax.set_xlabel("detection rate $p$")
    ax.set_ylabel("DSI")
    ax.set_title("a | A uniform gene scores above zero", loc="left", fontsize=7)
    ax.legend(loc="upper left", borderpad=0.2, labelspacing=0.25)

    ax = axes[1]
    edges = [0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.40, 1.01]
    for name in NAMES:
        d = gz[(gz.dataset == name) & (gz.gene_class == "protein_coding")]
        xs, ys = [], []
        for a, b in zip(edges[:-1], edges[1:]):
            sub = d[(d.detect >= a) & (d.detect < b)]
            if len(sub) >= 10:
                xs.append(np.sqrt(a * min(b, 0.6)))
                ys.append(sub["DSI"].median() / sub["null_mean"].median())
        ax.plot(xs, ys, "o-", color=DS_COLOR[name], ms=2.2, label=name)
    ax.axhline(1, color="k", ls="--", lw=0.7)
    ax.set_xscale("log")
    ax.set_xlabel("detection rate $p$")
    ax.set_ylabel("observed DSI / null DSI")
    ax.set_title("b | Low-detection genes sit on the floor", loc="left", fontsize=7)
    ax.legend(borderpad=0.2, labelspacing=0.25)

    ax = axes[2]
    wins = ["±20%", "±50%"]
    w = 0.36
    xpos = np.arange(len(NAMES))
    for i, win in enumerate(wins):
        vals, ps = [], []
        for nm in NAMES:
            row = na[(na.ds == nm) & (na.window == win)].iloc[0]
            vals.append(row["z_median"]); ps.append(row["z_p"])
        bars = ax.bar(xpos + (i - 0.5) * w, vals, width=w,
                      color=["#9ECAE1", "#2171B5"][i], label=win)
        for b, v, pv in zip(bars, vals, ps):
            star = "***" if pv < 1e-3 else ("**" if pv < 1e-2 else ("*" if pv < 0.05 else "n.s."))
            ax.text(b.get_x() + b.get_width() / 2, v + (0.03 if v >= 0 else -0.03), star,
                    ha="center", va="bottom" if v >= 0 else "top", fontsize=6)
    ax.axhline(0, color="k", ls="--", lw=0.7)
    ax.set_xticks(xpos)
    ax.set_xticklabels([n.replace(" ", "\n") for n in NAMES], fontsize=6)
    ax.set_ylabel("Δ$z$ (lncRNA − matched)")
    ax.set_title("c | Noise-corrected difference", loc="left", fontsize=7)
    ax.legend(borderpad=0.2, labelspacing=0.25, loc="upper left")
    ax.set_ylim(-1.1, 0.45)

    fig.tight_layout(pad=0.3, w_pad=1.2)
    save(fig, "fig4_noise_floor")


# ------------------------------------------------------------------ Fig 5
def fig5():
    platforms = ["MERFISH\n(Cohen 2026)", "Visium V1\n(mouse)", "Visium V1\n(human)",
                 "Visium HD\n(mouse)", "Xenium\nPrime 5K"]
    genes = [22930, 32285, 36601, 19059, 5006]
    lncrna = [4740, 2312, 1361, 0, 4]
    pct = [100 * a / b for a, b in zip(lncrna, genes)]
    fig, ax = plt.subplots(figsize=(HALF, HALF * 0.78))
    x = np.arange(len(platforms))
    bars = ax.bar(x, pct, color=["#4C78A8"] * 3 + ["#D1495B"] * 2, width=0.65)
    for b, n, t in zip(bars, lncrna, genes):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.5,
                f"{n:,}/{t:,}", ha="center", fontsize=5.5)
    ax.set_xticks(x)
    ax.set_xticklabels(platforms, fontsize=5.5)
    ax.set_ylabel("lncRNAs as % of panel")
    ax.set_ylim(0, max(pct) * 1.25)
    ax.set_title("lncRNA coverage across platforms\nred = probe-enrichment panels",
                 fontsize=7)
    fig.tight_layout(pad=0.2)
    save(fig, "fig5_platform")


# ------------------------------------------------------------------ Fig 6
def fig6():
    iso = pd.read_csv(os.path.join(R, "H1H2", "moransI.csv"), index_col=0)
    wt = pd.read_csv(os.path.join(R, "H1H2_WT", "moransI.csv"), index_col=0)
    iso = iso[~iso.index.duplicated(keep="first")]
    wt = wt[~wt.index.duplicated(keep="first")]
    boot = pd.read_csv(os.path.join(R, "CatalogValidation", "bootstrap_stability.csv"))
    j = iso[["moransI", "gene_class"]].join(
        wt[["moransI"]], lsuffix="_ISO", rsuffix="_WT", how="inner").dropna()

    fig, axes = plt.subplots(1, 2, figsize=(FULL * 0.78, FULL * 0.35))

    ax = axes[0]
    non = j[~j.index.isin(CATALOG)]
    ax.scatter(non["moransI_ISO"], non["moransI_WT"], s=2.5, color="#B0B0B0",
               alpha=0.55, linewidths=0, label="other genes", rasterized=True)
    cat = j[j.index.isin(CATALOG)]
    ax.scatter(cat["moransI_ISO"], cat["moransI_WT"], s=16, color="#D1495B",
               edgecolor="white", linewidths=0.4, zorder=3, label="catalogue (n=15)")
    # 只标注两个已知的神经元表达 lncRNA（内部阳性对照）；其余为 Gm*/Rik 未表征位点，
    # 全部标注会重叠成一团。
    for g in ["Miat", "Cdr1"]:
        if g in cat.index:
            r = cat.loc[g]
            ax.annotate(g, (r["moransI_ISO"], r["moransI_WT"]), fontsize=5.5,
                        xytext=(2.5, 1.5), textcoords="offset points", color="#7a1f2b")
    lim = max(j["moransI_ISO"].max(), j["moransI_WT"].max()) * 1.08
    ax.plot([0, lim], [0, lim], "--", color="grey", lw=0.6)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("Moran's $I$ (ISO panel)")
    ax.set_ylabel("Moran's $I$ (WT panel)")
    ax.set_title("a | Cross-panel reproducibility", loc="left", fontsize=7)
    ax.legend(loc="lower right", borderpad=0.2, labelspacing=0.25)

    ax = axes[1]
    # "other" 组须排除全部 16 个初筛候选（含被剔除的 AY512931），
    # 使 n = 99 − 16 = 83，与正文表述一致
    screened = set(CATALOG) | {"AY512931"}
    is_cat = boot["gene"].isin(CATALOG)
    in_screen = boot["gene"].isin(screened)
    cat_hit = boot.loc[is_cat, "bootstrap_hit_rate"]
    oth_hit = boot.loc[~in_screen, "bootstrap_hit_rate"]
    bins = np.linspace(0, 1, 21)
    ax.hist(oth_hit, bins=bins, color="#4C78A8", alpha=0.7, linewidth=0,
            label=f"other (n={len(oth_hit)}, median {oth_hit.median():.0%})")
    ax.hist(cat_hit, bins=bins, color="#D1495B", alpha=0.8, linewidth=0,
            label=f"catalogue (n={len(cat_hit)}, median {cat_hit.median():.0%})")
    ax.set_xlabel("bootstrap hit rate")
    ax.set_ylabel("lncRNAs")
    ax.set_title("b | Resampling stability", loc="left", fontsize=7)
    ax.legend(borderpad=0.2, labelspacing=0.25)

    fig.tight_layout(pad=0.3, w_pad=1.5)
    save(fig, "fig6_catalog")


# ------------------------------------------------------------------ Fig 7
def fig7():
    KEY = "cellxgene_ISO"
    LDIR = os.path.join(R, "Ldomain")
    lab = pd.read_csv(os.path.join(LDIR, "domain_labels.csv"), index_col=0)
    ranges = json.load(open(os.path.join(LDIR, "subset_rows.json")))
    var = sio.read_var(KEY)
    show = CATALOG[:6]
    pos = var.index.get_indexer(show)
    blocks = [sio.read_rows(KEY, r["start"], r["stop"])[:, pos].tocsr() for r in ranges]
    X = vstack(blocks, format="csr").toarray().astype(bool)

    sample_arr = lab["sample"].astype(str).to_numpy()
    s0 = sorted(set(sample_arr))[0]
    m = sample_arr == s0
    coords = lab[["global_x", "global_y"]].to_numpy()

    # 切片长宽比约 1:3.4（高:宽），因此改用 6 联竖幅并排，
    # 使每格都接近切片自身比例，避免大面积留白。
    fig, axes = plt.subplots(1, len(show), figsize=(FULL, FULL * 0.62))
    for k, g in enumerate(show):
        ax = axes[k]
        ax.scatter(coords[m, 0], coords[m, 1], s=0.03, c="lightgrey",
                   linewidths=0, rasterized=True)
        hit = X[:, k] & m
        ax.scatter(coords[hit, 0], coords[hit, 1], s=0.25, c="#D1495B",
                   linewidths=0, rasterized=True)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{g}\n{hit.sum():,} cells", fontsize=6, linespacing=1.1)
        for sp in ax.spines.values():
            sp.set_linewidth(0.4)
    fig.tight_layout(pad=0.15, w_pad=0.25)
    save(fig, "fig7_spatial")


if __name__ == "__main__":
    for fn in [fig1, fig2, fig3, fig4, fig5, fig6, fig7]:
        print(f"{fn.__name__} ...")
        fn()
    print(f"\n输出目录: {OUT}")
