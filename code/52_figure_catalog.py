# -*- coding: utf-8 -*-
"""绘制候选目录图 Fig 6-7。"""
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
FIG = os.path.join(R, "figures")
os.makedirs(FIG, exist_ok=True)

CATALOG = ["Cdr1", "Gm49698", "Gm27000", "Gm48623", "Miat", "Gm47221",
           "Gm35240", "Gm42428", "Gm49124", "Gm45895", "B230220B15Rik",
           "Gm47408", "Gm45163", "A830036E02Rik", "Gm49655"]


def fig6():
    iso = pd.read_csv(os.path.join(R, "H1H2", "moransI.csv"), index_col=0)
    wt = pd.read_csv(os.path.join(R, "H1H2_WT", "moransI.csv"), index_col=0)
    iso = iso[~iso.index.duplicated(keep="first")]
    wt = wt[~wt.index.duplicated(keep="first")]
    boot = pd.read_csv(os.path.join(R, "CatalogValidation", "bootstrap_stability.csv"))

    j = iso[["moransI", "gene_class"]].join(
        wt[["moransI"]], lsuffix="_ISO", rsuffix="_WT", how="inner").dropna()
    print(f"两面板共同基因: {len(j)}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    ax = axes[0]
    non = j[~j.index.isin(CATALOG)]
    ax.scatter(non["moransI_ISO"], non["moransI_WT"], s=12, color="#B0B0B0",
               alpha=0.55, linewidths=0, label="other genes")
    cat = j[j.index.isin(CATALOG)]
    sc = ax.scatter(cat["moransI_ISO"], cat["moransI_WT"], s=70, color="#D1495B",
                    edgecolor="white", linewidths=0.6, zorder=3, label="catalog (n=15)")
    for g, r in cat.iterrows():
        ax.annotate(g, (r["moransI_ISO"], r["moransI_WT"]), fontsize=6.5,
                    xytext=(3, 2), textcoords="offset points", color="#7a1f2b")
    lim = max(j["moransI_ISO"].max(), j["moransI_WT"].max()) * 1.08
    ax.plot([0, lim], [0, lim], "--", color="grey", lw=1)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("Moran's I (MERFISH ISO panel)")
    ax.set_ylabel("Moran's I (MERFISH WT panel)")
    ax.set_title("A | Cross-panel reproducibility of spatial structure", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")

    ax = axes[1]
    # 用最终 15 个（已剔除 AY512931，其 bootstrap 入选率仅 10%）
    is_cat = boot["gene"].isin(CATALOG)
    cat_hit = boot.loc[is_cat, "bootstrap_hit_rate"]
    oth_hit = boot.loc[~is_cat, "bootstrap_hit_rate"]
    bins = np.linspace(0, 1, 21)
    ax.hist(oth_hit, bins=bins, color="#4C78A8", alpha=0.7,
            label=f"other lncRNAs (n={len(oth_hit)}, median {oth_hit.median():.0%})")
    ax.hist(cat_hit, bins=bins, color="#D1495B", alpha=0.8,
            label=f"catalog (n={len(cat_hit)}, median {cat_hit.median():.0%})")
    ax.set_xlabel("bootstrap hit rate (fraction of 50 resamples in top 16)")
    ax.set_ylabel("lncRNAs")
    ax.set_title("B | Resampling stability", fontsize=10)
    ax.legend(fontsize=8)

    fig.suptitle("Fig 6 | A small, reproducible set of spatially structured lncRNAs", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig6_catalog.png"), dpi=200)
    plt.close(fig)
    print("Fig6 完成")


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

    fig, axes = plt.subplots(2, 3, figsize=(13, 8.4))
    for k, g in enumerate(show):
        ax = axes[k // 3][k % 3]
        ax.scatter(coords[m, 0], coords[m, 1], s=0.15, c="lightgrey", linewidths=0,
                   rasterized=True)
        hit = X[:, k] & m
        ax.scatter(coords[hit, 0], coords[hit, 1], s=1.2, c="#D1495B", linewidths=0,
                   rasterized=True)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{g}  (detected in {hit.sum():,} cells)", fontsize=10)
    fig.suptitle(f"Fig 7 | Spatial expression of top catalog lncRNAs (section {s0})",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig7_spatial.png"), dpi=200)
    plt.close(fig)
    print("Fig7 完成")


if __name__ == "__main__":
    fig6()
    fig7()
    print(f"输出目录: {FIG}")
