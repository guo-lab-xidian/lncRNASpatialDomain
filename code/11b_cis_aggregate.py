# -*- coding: utf-8 -*-
"""
课题三 / H2b：顺式邻域**聚合**耦合（解决单基因对相关性信噪比不足的问题）。

思路：
  单对 lncRNA–邻居 的二值相关被稀疏噪声淹没（实测效应量 0.003~0.015，噪声 sd 0.016）。
  改为把 lncRNA 的检出模式与**其整个顺式邻域的平均检出模式**做相关：
      agg_q(cell) = mean_{j ∈ N(q)} x_j(cell)
  聚合 20–30 个邻居可把信噪比提高一个量级。

  零分布：把邻域换成"大小相同、平均检出率相近（±30%）的随机蛋白编码基因集"，R=20 次。

同时跑蛋白编码阳性对照（检出率匹配的查询基因），比较两者的聚合耦合强度。

产出（lncRNA空间域/results/H2/）：
  cis_aggregate_<set>.csv      每个查询基因的 r_real / r_null 分布
  compare_aggregate.csv        lncRNA vs 蛋白编码对照
  fig_h2b_aggregate.png
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import h5py
from scipy.sparse import csr_matrix

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import spatial_io as sio

KEY = "cellxgene_ISO"
RES = r"E:\研究工作\agent长非与空间域\lncRNA空间域\results"
CLS = r"E:\研究工作\agent长非与空间域\analysis\03_lncrna\panel_features_class.csv"
COORD = os.path.join(RES, "gene_coords_m25.tsv")
OUT = os.path.join(RES, "H2")
os.makedirs(OUT, exist_ok=True)

LNC_MIN_DETECT = 0.05
NEIGH_MIN_DETECT = 0.01
WINDOW = int(sys.argv[1]) if len(sys.argv) > 1 else 250_000   # 可用命令行覆盖，如 1000000
MAX_NEIGH = 30
R_NULL = 20
NULL_TOL = 0.30
SEED = 0


def attach_coords(g, coords):
    c = coords.drop_duplicates("gene_id").set_index("gene_id")
    for col in ["chrom", "start", "end", "strand", "length"]:
        g[col] = g["gene_id"].map(c[col])
    return g


def neighbours_of(query_row, cand):
    sub = cand[cand["chrom"] == query_row.chrom]
    m = (sub["end"] >= query_row.start - WINDOW) & (sub["start"] <= query_row.end + WINDOW)
    return sub[m & (sub.index != query_row.Index)]


def build_design(qset, cand, rng):
    """构造每个查询基因的邻域基因集与 R 个随机对照基因集。"""
    near_lists, null_lists, info = [], [], []
    for r in qset.itertuples():
        nb = neighbours_of(r, cand)
        if len(nb) == 0:
            continue
        nb = nb.sort_values("detect_frac", ascending=False).head(MAX_NEIGH)
        names = list(nb.index)
        mean_det = nb["detect_frac"].mean()
        # 随机对照：大小相同、平均检出率相近
        pool = cand[(cand["detect_frac"] >= mean_det * (1 - NULL_TOL)) &
                    (cand["detect_frac"] <= mean_det * (1 + NULL_TOL)) &
                    (~cand.index.isin(names))]
        if len(pool) < len(names) * 3:
            pool = cand[~cand.index.isin(names)]
        if len(pool) < len(names):
            continue
        nulls = [list(pool.sample(len(names), random_state=int(rng.integers(1e9))).index)
                 for _ in range(R_NULL)]
        near_lists.append(names)
        null_lists.append(nulls)
        info.append({"query": r.Index, "gene_name": r.gene_name,
                     "detect_frac": r.detect_frac, "n_neighbours": len(names),
                     "neighbour_mean_detect": mean_det,
                     "random_set_mean_detect": float(
                         np.mean([cand.loc[n, "detect_frac"].mean() for n in nulls]))})
    return near_lists, null_lists, pd.DataFrame(info)


def build_matrix(lists, union_pos, n_cols, weights):
    """把基因列表变成稀疏矩阵（union × n_cols）。"""
    rows, cols, vals = [], [], []
    for j, gene_list in enumerate(lists):
        w = weights[j]
        for gname in gene_list:
            p = union_pos.get(gname)
            if p is None:
                continue
            rows.append(p)
            cols.append(j)
            vals.append(w)
    return csr_matrix((np.array(vals, dtype=np.float32), (rows, cols)),
                      shape=(len(union_pos), n_cols))


def run_set(name, qset, cand, rng):
    print(f"\n===== {name} =====")
    near_lists, null_lists, info = build_design(qset, cand, rng)
    print(f"  有邻域的查询基因: {len(info)} / {len(qset)}")
    if info.empty:
        return None
    print(f"  邻域基因数中位: {info['n_neighbours'].median():.0f} | "
          f"邻域平均检出率中位: {info['neighbour_mean_detect'].median():.3f}")

    # 需要的基因：查询基因 + 邻域基因 + 随机集基因
    need = set(info["query"])
    for lst in near_lists:
        need.update(lst)
    for nulls in null_lists:
        for s in nulls:
            need.update(s)
    need = sorted(need)
    union_pos = {g: i for i, g in enumerate(need)}
    nq = len(info)
    print(f"  需读取特征: {len(need)}（查询 {nq}）")

    # 聚合矩阵
    M_real = build_matrix(near_lists, union_pos, nq, [1.0 / len(l) for l in near_lists])
    flat_null = []
    weights_null = []
    for lst, nulls in zip(near_lists, null_lists):
        for s in nulls:
            flat_null.append(s)
            weights_null.append(1.0 / len(s))
    M_null = build_matrix(flat_null, union_pos, nq * R_NULL, weights_null)

    # 查询基因自身矩阵
    q_pos = [union_pos[q] for q in info["query"]]
    A_cols = len(need)

    var = sio.read_var(KEY)
    col_pos = var.index.get_indexer(need)
    assert (col_pos >= 0).all()

    # 累积量
    n_cells = 0
    Sa = np.zeros(nq); Saagg_r = np.zeros(nq)
    Sagg_r = np.zeros(nq); Sagg2_r = np.zeros(nq)
    Saagg_n = np.zeros(nq * R_NULL); Sagg_n = np.zeros(nq * R_NULL)
    Sagg2_n = np.zeros(nq * R_NULL)

    path = sio._resolve(KEY)
    with h5py.File(path, "r") as f:
        indptr = f["X/indptr"][:]
        n_obs = len(indptr) - 1
        dset, iset = f["X/data"], f["X/indices"]
        n_var = f["var/_index"].shape[0]
        sub_block = 3000
        for start in range(0, n_obs, sub_block):
            end = min(start + sub_block, n_obs)
            lo, hi = int(indptr[start]), int(indptr[end])
            blk = csr_matrix(((dset[lo:hi] > 0).astype(np.float32), iset[lo:hi],
                              indptr[start:end + 1] - lo), shape=(end - start, n_var))
            C = blk[:, col_pos].tocsr()            # chunk × need
            A = C[:, q_pos]                        # chunk × nq（二值）

            a = np.asarray(A.todense(), dtype=np.float64)
            agg_r = np.asarray((C @ M_real).todense(), dtype=np.float64)   # chunk × nq
            agg_n = np.asarray((C @ M_null).todense(), dtype=np.float64)   # chunk × nq*R

            n_cells += end - start
            Sa += a.sum(axis=0)
            Saagg_r += (a * agg_r).sum(axis=0)
            Sagg_r += agg_r.sum(axis=0)
            Sagg2_r += (agg_r ** 2).sum(axis=0)
            Saagg_n += (a.repeat(R_NULL, axis=1) * agg_n).sum(axis=0)
            Sagg_n += agg_n.sum(axis=0)
            Sagg2_n += (agg_n ** 2).sum(axis=0)
            if start % 30000 == 0:
                print(f"    cells {start}-{end}", flush=True)

    def pearson(sa, sagg, sagg2, saagg, n):
        cov = saagg - sa * sagg / n
        va = sa - sa ** 2 / n
        vb = sagg2 - sagg ** 2 / n
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where((va > 0) & (vb > 0), cov / np.sqrt(va * vb), np.nan)

    r_real = pearson(Sa, Sagg_r, Sagg2_r, Saagg_r, n_cells)
    r_null = pearson(np.repeat(Sa, R_NULL), Sagg_n, Sagg2_n, Saagg_n, n_cells).reshape(nq, R_NULL)

    out = info.copy()
    out["r_real"] = r_real
    out["r_null_mean"] = np.nanmean(r_null, axis=1)
    out["r_null_sd"] = np.nanstd(r_null, axis=1)
    out["delta"] = out["r_real"] - out["r_null_mean"]
    out["p_value"] = [(1 + int((rn >= rr).sum())) / (1 + len(rn))
                      for rr, rn in zip(r_real, r_null)]
    out["n_null"] = R_NULL
    out.to_csv(os.path.join(OUT, f"cis_aggregate_{name}.csv"), index=False)

    print(f"  [{name}] r_real 中位: {np.nanmedian(out['r_real']):+.4f}")
    print(f"  [{name}] r_null 中位: {np.nanmedian(out['r_null_mean']):+.4f}")
    print(f"  [{name}] delta 中位:  {np.nanmedian(out['delta']):+.4f} | "
          f"delta>0: {int((out['delta'] > 0).sum())}/{out['delta'].notna().sum()} | "
          f"p<0.05: {int((out['p_value'] < 0.05).sum())}")
    print("\n  delta 最大的 10 个:")
    print(out.nlargest(10, "delta")[["gene_name", "detect_frac", "n_neighbours",
                                     "neighbour_mean_detect", "r_real", "r_null_mean",
                                     "delta", "p_value"]].round(4).to_string(index=False))
    return out


def main():
    rng = np.random.default_rng(SEED)
    pf = pd.read_csv(CLS, index_col=0)
    g = pf[(pf["panel"] == "ISO") & (pf["feature_type"] == "gene")].copy()
    g = attach_coords(g, pd.read_csv(COORD, sep="\t"))
    lnc = g[(g["gene_class"] == "lncRNA") & (g["detect_frac"] >= LNC_MIN_DETECT) &
            g["chrom"].notna()].copy()
    cand = g[(g["gene_class"] == "protein_coding") & (g["detect_frac"] >= NEIGH_MIN_DETECT) &
             g["chrom"].notna()].copy()
    pc_pool = cand[cand["detect_frac"] >= LNC_MIN_DETECT]
    print(f"lncRNA 查询集 {len(lnc)} | 蛋白编码候选 {len(cand):,}")

    # 阳性对照：检出率匹配的蛋白编码基因
    picks = []
    for det in lnc["detect_frac"].to_numpy():
        sel = pc_pool[(pc_pool["detect_frac"] >= det * 0.8) & (pc_pool["detect_frac"] <= det * 1.2)]
        if len(sel) == 0:
            sel = pc_pool.iloc[np.argsort(np.abs(pc_pool["detect_frac"] - det))[:20]]
        picks.append(sel.sample(1, random_state=int(rng.integers(1e9))).index[0])
    ctrl = pc_pool.loc[list(dict.fromkeys(picks))]
    print(f"阳性对照查询集 {len(ctrl)}（检出率中位 {ctrl['detect_frac'].median():.3f} "
          f"vs lncRNA {lnc['detect_frac'].median():.3f}）")

    a = run_set("lncRNA", lnc, cand, rng)
    b = run_set("PC", ctrl, cand, rng)

    if a is not None and b is not None:
        from scipy import stats
        da = a["delta"].dropna()
        db = b["delta"].dropna()
        u = stats.mannwhitneyu(da, db, alternative="less")
        comp = pd.DataFrame([{
            "set": "lncRNA", "n": len(da), "delta_median": da.median(),
            "delta_mean": da.mean(), "frac_pos": (da > 0).mean(),
            "n_p05": int((a["p_value"] < 0.05).sum()),
        }, {
            "set": "protein_coding_control", "n": len(db), "delta_median": db.median(),
            "delta_mean": db.mean(), "frac_pos": (db > 0).mean(),
            "n_p05": int((b["p_value"] < 0.05).sum()),
        }])
        comp.to_csv(os.path.join(OUT, "compare_aggregate.csv"), index=False)
        print("\n" + "=" * 90)
        print("=== 邻域聚合耦合：lncRNA vs 蛋白编码对照 ===")
        print(comp.round(4).to_string(index=False))
        print(f"Mann-Whitney（单侧 lncRNA < PC）p = {u.pvalue:.3e}")

        fig, ax = plt.subplots(figsize=(6.4, 4.4))
        bins = np.linspace(min(da.min(), db.min()), max(da.max(), db.max()), 40)
        ax.hist(db, bins=bins, alpha=0.55, label="protein-coding (control)", color="#4C78A8")
        ax.hist(da, bins=bins, alpha=0.7, label="lncRNA", color="#D1495B")
        ax.axvline(0, color="grey", ls="--", lw=1)
        ax.set_xlabel("delta (neighborhood aggregate − matched random sets)")
        ax.set_ylabel("number of genes")
        ax.set_title("Cis-neighborhood aggregate coupling (±250 kb)")
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "fig_h2b_aggregate.png"), dpi=150)
        plt.close(fig)

    print(f"\n=== H2b 完成，输出: {OUT} ===")


if __name__ == "__main__":
    main()
