# -*- coding: utf-8 -*-
"""
课题三 / H1 + H2′：域特异性锐度 与 空间域/细胞类型解耦。

H1：在检出率匹配的条件下，lncRNA 的域特异性是否高于蛋白编码基因？
   指标 DSI(g) = max_k f_k(g) − median_k f_k(g)，f_k 为域 k 内检出率
   分别用 L-domain（空间域）与 L-cell（细胞类型）两套标签计算

H2′：有多少 lncRNA 是"域特异但细胞类型不特异"（即真正的空间域标记）？
   以匹配蛋白编码基因的 DSI 分布为参照，分四类：
   双特异 / 仅域特异 / 仅细胞类型特异 / 都不特异

产出（lncRNA空间域/results/H1H2/）：
  gene_specificity.csv   每个基因的 DSI_domain / DSI_cell / 两套标签下的峰值域
  h1_matched_test.csv     H1 的匹配对照检验结果
  h2_class_composition.csv H2′ 的四类构成（lncRNA vs 蛋白编码）
  fig_h1h2.png
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import h5py
from scipy.sparse import csr_matrix
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import spatial_io as sio

PANEL = (sys.argv[1] if len(sys.argv) > 1 else "ISO").upper()
SUFFIX = "" if PANEL == "ISO" else "_" + PANEL
KEY = "cellxgene_" + PANEL
RES = r"E:\研究工作\agent长非与空间域\lncRNA空间域\results"
LDIR = os.path.join(RES, "Ldomain" + SUFFIX)
CLS = r"E:\研究工作\agent长非与空间域\analysis\03_lncrna\panel_features_class.csv"
OUT = os.path.join(RES, "H1H2" + SUFFIX)
os.makedirs(OUT, exist_ok=True)

LNC_MIN_DETECT = 0.05
BLOCK = 15000
SEED = 0


def load_groups():
    lab = pd.read_csv(os.path.join(LDIR, "domain_labels.csv"), index_col=0)
    ranges = json.load(open(os.path.join(LDIR, "subset_rows.json")))
    n = sum(r["stop"] - r["start"] for r in ranges)
    assert len(lab) == n, f"标签数 {len(lab)} 与子集细胞数 {n} 不一致"
    dom_codes, dom_names = pd.factorize(lab["L_domain"].astype(str))
    cell_codes, cell_names = pd.factorize(lab["cell_label"].astype(str))
    print(f"  L-domain: {len(dom_names)} 个域 | L-cell: {len(cell_names)} 类细胞")
    return lab, ranges, dom_codes, dom_names, cell_codes, cell_names


def accumulate_counts(ranges, dom_codes, cell_codes, n_dom, n_cell):
    """一次流式扫描，累积每个域/细胞类型内每个特征的检出细胞数。"""
    path = sio._resolve(KEY)
    with h5py.File(path, "r") as f:
        n_var = f["var/_index"].shape[0]
        indptr = f["X/indptr"][:]
        dset, iset = f["X/data"], f["X/indices"]
        cnt_dom = np.zeros((n_dom, n_var), dtype=np.int32)
        cnt_cell = np.zeros((n_cell, n_var), dtype=np.int32)
        off = 0
        for r in ranges:
            for start in range(r["start"], r["stop"], BLOCK):
                end = min(start + BLOCK, r["stop"])
                lo, hi = int(indptr[start]), int(indptr[end])
                blk = csr_matrix(((dset[lo:hi] > 0).astype(np.float32), iset[lo:hi],
                                  indptr[start:end + 1] - lo),
                                 shape=(end - start, n_var))
                m = end - start
                gd = dom_codes[off:off + m]
                gc = cell_codes[off:off + m]
                Gd = csr_matrix((np.ones(m, dtype=np.float32), (np.arange(m), gd)),
                                shape=(m, n_dom))
                Gc = csr_matrix((np.ones(m, dtype=np.float32), (np.arange(m), gc)),
                                shape=(m, n_cell))
                cnt_dom += np.asarray((Gd.T @ blk).todense()).astype(np.int32)
                cnt_cell += np.asarray((Gc.T @ blk).todense()).astype(np.int32)
                off += m
                print(f"    rows {start}-{end}", flush=True)
        n_dom_cells = np.bincount(dom_codes, minlength=n_dom)
        n_cell_cells = np.bincount(cell_codes, minlength=n_cell)
    return cnt_dom, cnt_cell, n_dom_cells, n_cell_cells


def dsi(counts, group_sizes):
    f = counts / group_sizes[:, None]
    return f.max(axis=0) - np.median(f, axis=0), f


def main():
    pf = pd.read_csv(CLS, index_col=0)
    g = pf[(pf["panel"] == PANEL) & (pf["feature_type"] == "gene")].copy()
    var = sio.read_var(KEY)
    g = g.reindex(var.index)

    lab, ranges, dom_codes, dom_names, cell_codes, cell_names = load_groups()
    print("\n=== 累积各域内的检出细胞数 ===")
    cnt_dom, cnt_cell, nd, nc = accumulate_counts(ranges, dom_codes, cell_codes,
                                                  len(dom_names), len(cell_names))
    print(f"  完成：domain 计数 {cnt_dom.shape} | cell 计数 {cnt_cell.shape}")

    dsi_dom, f_dom = dsi(cnt_dom, nd)
    dsi_cell, f_cell = dsi(cnt_cell, nc)

    spec = pd.DataFrame({
        "feature": var.index,
        "detect_frac_subset": cnt_dom.sum(axis=0) / cnt_dom.sum(),
        "DSI_domain": dsi_dom,
        "DSI_cell": dsi_cell,
        "peak_domain": [dom_names[i] for i in f_dom.argmax(axis=0)],
        "peak_celltype": [cell_names[i] for i in f_cell.argmax(axis=0)],
    }).set_index("feature")
    spec["gene_class"] = g["gene_class"]
    spec["gene_name"] = g["gene_name"]
    spec["panel_detect"] = g["detect_frac"]
    spec.to_csv(os.path.join(OUT, "gene_specificity.csv"))

    # ---------------- H1：匹配对照 ----------------
    print("\n" + "=" * 90)
    print("=== H1：域特异性锐度（匹配对照）===")
    lnc = spec[(spec["gene_class"] == "lncRNA") & (spec["panel_detect"] >= LNC_MIN_DETECT)]
    pc = spec[(spec["gene_class"] == "protein_coding") & (spec["panel_detect"] >= LNC_MIN_DETECT)]
    print(f"  lncRNA {len(lnc)} | 蛋白编码候选 {len(pc):,}")

    rng = np.random.default_rng(SEED)
    rows = []
    for r in lnc.itertuples():
        det = r.panel_detect
        sel = pc[(pc["panel_detect"] >= det * 0.8) & (pc["panel_detect"] <= det * 1.2)]
        if len(sel) < 5:
            sel = pc.iloc[np.argsort(np.abs(pc["panel_detect"] - det))[:5]]
        ctrl = sel.sample(min(5, len(sel)), random_state=int(rng.integers(1e9)))
        rows.append({
            "gene": r.Index, "gene_name": r.gene_name, "detect": det,
            "DSI_domain_lnc": r.DSI_domain, "DSI_domain_ctrl": ctrl["DSI_domain"].median(),
            "DSI_cell_lnc": r.DSI_cell, "DSI_cell_ctrl": ctrl["DSI_cell"].median(),
        })
    h1 = pd.DataFrame(rows)
    h1["delta_domain"] = h1["DSI_domain_lnc"] - h1["DSI_domain_ctrl"]
    h1["delta_cell"] = h1["DSI_cell_lnc"] - h1["DSI_cell_ctrl"]
    h1.to_csv(os.path.join(OUT, "h1_matched_test.csv"), index=False)

    for col, label in [("delta_domain", "空间域 L-domain"), ("delta_cell", "细胞类型 L-cell")]:
        d = h1[col].dropna()
        w = stats.wilcoxon(d)
        print(f"\n  [{label}] delta 中位 {d.median():+.4f} | 均值 {d.mean():+.4f} | "
              f"为正 {int((d > 0).sum())}/{len(d)} | Wilcoxon p={w.pvalue:.2e}")
        print(f"     Cliff's delta ≈ {2*d.rank(pct=True).mean()-1:+.3f}（>0 表示 lncRNA 更特异）")

    # ---------------- H2′：四类构成 ----------------
    print("\n" + "=" * 90)
    print("=== H2′：空间域 vs 细胞类型解耦 ===")
    # 阈值：用匹配对照分布的 75 分位
    thr_dom = pc["DSI_domain"].quantile(0.75)
    thr_cell = pc["DSI_cell"].quantile(0.75)
    print(f"  判定阈值（蛋白编码基因的 75 分位）: DSI_domain={thr_dom:.4f}, DSI_cell={thr_cell:.4f}")

    def classify(df):
        dom_hi = df["DSI_domain"] >= thr_dom
        cell_hi = df["DSI_cell"] >= thr_cell
        out = np.where(dom_hi & cell_hi, "双特异",
                       np.where(dom_hi & ~cell_hi, "仅域特异",
                                np.where(~dom_hi & cell_hi, "仅细胞类型特异", "都不特异")))
        return pd.Series(out, index=df.index)

    spec["class"] = pd.NA
    spec.loc[lnc.index, "class"] = classify(lnc)
    spec.loc[pc.index, "class"] = classify(pc)
    spec.to_csv(os.path.join(OUT, "gene_specificity.csv"))

    comp = pd.DataFrame({
        "lncRNA": lnc.assign(c=classify(lnc))["c"].value_counts(normalize=True),
        "protein_coding": pc.assign(c=classify(pc))["c"].value_counts(normalize=True),
    }).fillna(0)
    comp["lncRNA_n"] = lnc.assign(c=classify(lnc))["c"].value_counts()
    comp["pc_n"] = pc.assign(c=classify(pc))["c"].value_counts()
    comp.to_csv(os.path.join(OUT, "h2_class_composition.csv"))
    print(comp.round(4).to_string())

    tab = pd.crosstab(pd.concat([lnc.assign(c=classify(lnc))["c"],
                                 pc.assign(c=classify(pc))["c"]]),
                      ["lncRNA"] * len(lnc) + ["protein_coding"] * len(pc))
    chi = stats.chi2_contingency(tab.values)
    print(f"\n  卡方检验: chi2={chi[0]:.1f}, p={chi[1]:.3e}")

    # ---------------- 图 ----------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    axes[0].scatter(pc["DSI_domain"], pc["DSI_cell"], s=4, alpha=0.25,
                    color="#4C78A8", label="protein-coding", rasterized=True)
    axes[0].scatter(lnc["DSI_domain"], lnc["DSI_cell"], s=18, alpha=0.9,
                    color="#D1495B", label="lncRNA")
    axes[0].axvline(thr_dom, color="grey", ls="--", lw=0.8)
    axes[0].axhline(thr_cell, color="grey", ls="--", lw=0.8)
    axes[0].set_xlabel("DSI (spatial domain, L-domain)")
    axes[0].set_ylabel("DSI (cell type, L-cell)")
    axes[0].set_title("Domain vs cell-type specificity")
    axes[0].legend(fontsize=8)

    order = ["双特异", "仅域特异", "仅细胞类型特异", "都不特异"]
    x = np.arange(len(order))
    axes[1].bar(x - 0.2, [comp.loc[o, "lncRNA"] if o in comp.index else 0 for o in order],
                width=0.4, label="lncRNA", color="#D1495B")
    axes[1].bar(x + 0.2, [comp.loc[o, "protein_coding"] if o in comp.index else 0 for o in order],
                width=0.4, label="protein-coding", color="#4C78A8")
    axes[1].set_xticks(x); axes[1].set_xticklabels(order, rotation=20, fontsize=8)
    axes[1].set_ylabel("fraction")
    axes[1].set_title("Class composition")
    axes[1].legend(fontsize=8)

    for col, color, lab in [("delta_domain", "#D1495B", "L-domain"),
                            ("delta_cell", "#4C78A8", "L-cell")]:
        axes[2].hist(h1[col].dropna(), bins=35, alpha=0.6, color=color, label=lab)
    axes[2].axvline(0, color="grey", ls="--", lw=1)
    axes[2].set_xlabel("DSI(lncRNA) − median DSI(matched protein-coding)")
    axes[2].set_ylabel("lncRNAs")
    axes[2].legend(fontsize=8)
    axes[2].set_title("H1: matched comparison")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_h1h2.png"), dpi=150)
    plt.close(fig)

    print(f"\n=== H1/H2′ 完成，输出: {OUT} ===")


if __name__ == "__main__":
    main()
