# -*- coding: utf-8 -*-
"""
课题三 / H2（v2）：lncRNA 的空间模式是否由其顺式邻域解释。

v1 的两个缺陷已修正：
  1. 零分布改为按**邻居基因**的检出率匹配（v1 错按 lncRNA 匹配，导致对照基因检出率偏高、相关偏高）
  2. 主指标改为 r_raw（总的空间模式一致性）；r_within（域内去混杂）作次要指标
  3. 零分布候选池 600 → 2500

做法：
  - 对每个 lncRNA，取同染色体 ±100 kb 内检出率最高的至多 5 个蛋白编码邻居
  - 用二值检出的 Phi 相关衡量 lncRNA 与邻居的空间模式一致性
  - 零分布 = 与该邻居检出率相近（±30%）的随机蛋白编码基因
  - delta = r_obs − mean(r_null)；delta>0.1 且 p<0.05 → cis 跟随型

产出（lncRNA空间域/results/H2/）：
  cis_pairs.csv / lncrna_class.csv / null_distribution.csv / fig_h2_null.png
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
WINDOWS = [100_000, 250_000]     # 主分析 + 敏感性
MAX_NEIGHBORS = 5
N_NULL_POOL = 2500
NULL_TOL = 0.30                   # 检出率匹配窗口（相对）
MIN_NULL = 50
DELTA_THR = 0.10
SEED = 0


def attach_coords(g, coords):
    c = coords.drop_duplicates("gene_id").set_index("gene_id")
    for col in ["chrom", "start", "end", "strand", "length", "gene_type"]:
        g[col] = g["gene_id"].map(c[col])
    print(f"  坐标匹配: {g['chrom'].notna().sum():,} / {len(g):,} "
          f"({100*g['chrom'].notna().mean():.1f}%)")
    return g


def build_sets(g):
    lnc = g[(g["gene_class"] == "lncRNA") & (g["detect_frac"] >= LNC_MIN_DETECT) &
            g["chrom"].notna()].copy()
    cand = g[(g["gene_class"] == "protein_coding") & (g["detect_frac"] >= NEIGH_MIN_DETECT) &
             g["chrom"].notna()].copy()
    print(f"  lncRNA 主分析集（≥{LNC_MIN_DETECT}）: {len(lnc)}")
    print(f"  蛋白编码候选（≥{NEIGH_MIN_DETECT}）: {len(cand):,}")
    # 零分布池：按检出率分层抽样
    cs = cand.sort_values("detect_frac")
    q = np.linspace(0, len(cs) - 1, N_NULL_POOL).astype(int)
    pool = cs.iloc[q].drop_duplicates(subset="gene_id")
    print(f"  零分布池（检出率分层抽样）: {len(pool)}")
    return lnc, cand, pool


def find_cis(lnc, cand, window, k=MAX_NEIGHBORS):
    by_chrom = {c: sub for c, sub in cand.groupby("chrom")}
    out = []
    for r in lnc.itertuples():
        sub = by_chrom.get(r.chrom)
        if sub is None:
            continue
        m = (sub["end"] >= r.start - window) & (sub["start"] <= r.end + window)
        hits = sub[m]
        if hits.empty:
            continue
        hits = hits[hits.index != r.Index]        # 排除自身（蛋白编码对照时会出现）
        if hits.empty:
            continue
        mid = (r.start + r.end) / 2
        hits = hits.assign(dist=((hits["start"] + hits["end"]) / 2 - mid).abs())
        for h in hits.sort_values("detect_frac", ascending=False).head(k).itertuples():
            out.append({"lncrna": r.Index, "lncrna_name": r.gene_name,
                        "neighbor": h.Index, "neighbor_name": h.gene_name,
                        "dist_bp": h.dist, "window": window})
    return pd.DataFrame(out)


def accumulate_cross(test_cols, cand_cols, var_index, group_codes, n_groups, block=15000):
    """累积 P = T^T C（二值跨积）、各组和向量。"""
    tp = var_index.get_indexer(test_cols)
    cp = var_index.get_indexer(cand_cols)
    assert (tp >= 0).all() and (cp >= 0).all()
    all_pos = np.concatenate([tp, cp])
    uniq, inv = np.unique(all_pos, return_inverse=True)
    t_idx = inv[:len(tp)]
    c_idx = inv[len(tp):]

    pt, pc = len(tp), len(cp)
    P = np.zeros((pt, pc), dtype=np.float64)
    SgT = np.zeros((n_groups, pt), dtype=np.float64)
    SgC = np.zeros((n_groups, pc), dtype=np.float64)
    ng = np.zeros(n_groups, dtype=np.int64)

    path = sio._resolve(KEY)
    with h5py.File(path, "r") as f:
        indptr = f["X/indptr"][:]
        n_obs = len(indptr) - 1
        dset, iset = f["X/data"], f["X/indices"]
        n_var = f["var/_index"].shape[0]
        for start in range(0, n_obs, block):
            end = min(start + block, n_obs)
            lo, hi = int(indptr[start]), int(indptr[end])
            data = (dset[lo:hi] > 0).astype(np.float32)
            blk = csr_matrix((data, iset[lo:hi], indptr[start:end + 1] - lo),
                             shape=(end - start, n_var))
            sub = blk[:, uniq]
            T = sub[:, t_idx].tocsr()
            C = sub[:, c_idx].tocsr()
            P += np.asarray((T.T @ C).todense())
            gc = group_codes[start:end]
            G = csr_matrix((np.ones(len(gc), dtype=np.float32), (np.arange(len(gc)), gc)),
                           shape=(len(gc), n_groups))
            SgT += np.asarray((G.T @ T).todense())
            SgC += np.asarray((G.T @ C).todense())
            ng += np.bincount(gc, minlength=n_groups)
            print(f"    cells {start}-{end}", flush=True)
    return P, SgT, SgC, ng


def corr_cross(P, SgT, SgC, ng):
    n = ng.sum()
    Sa = SgT.sum(axis=0)
    Sc = SgC.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        cov_raw = P - np.outer(Sa, Sc) / n
        var_a = Sa - Sa ** 2 / n
        var_c = Sc - Sc ** 2 / n
        den = np.sqrt(np.outer(var_a, var_c))
        r_raw = np.where(den > 0, cov_raw / den, np.nan)

        cov_w = P - SgT.T @ (SgC / ng[:, None])
        var_wa = Sa - np.asarray(np.sum(SgT ** 2 / ng[:, None], axis=0)).ravel()
        var_wc = Sc - np.asarray(np.sum(SgC ** 2 / ng[:, None], axis=0)).ravel()
        den_w = np.sqrt(np.outer(var_wa, var_wc))
        r_within = np.where(den_w > 0, cov_w / den_w, np.nan)
    return r_raw, r_within


def analyse(panel, lnc, cis, pool, var, group_codes, n_groups, tag):
    need_t = list(lnc.index)
    need_c = list(dict.fromkeys(list(cis["neighbor"]) + list(pool.index)))
    print(f"  [{tag}] 读取特征: lncRNA {len(need_t)} + 候选 {len(need_c)}")
    P, SgT, SgC, ng = accumulate_cross(need_t, need_c, var.index, group_codes, n_groups)
    r_raw, r_within = corr_cross(P, SgT, SgC, ng)
    tpos = {g: i for i, g in enumerate(need_t)}
    cpos = {g: i for i, g in enumerate(need_c)}
    pool_pos = [(g, cpos[g]) for g in pool.index]
    pool_det = pool["detect_frac"].to_numpy()

    pair_rows, cls_rows, null_rows = [], [], []
    for r in lnc.itertuples():
        ti = tpos[r.Index]
        sub = cis[cis["lncrna"] == r.Index] if len(cis) else pd.DataFrame()
        if sub.empty:
            cls_rows.append({"lncrna": r.Index, "gene_name": r.gene_name,
                             "detect_frac": r.detect_frac, "n_cis": 0,
                             "category": "无邻居在面板中"})
            continue
        best = None
        for h in sub.itertuples():
            ci = cpos[h.neighbor]
            det_n = panel.loc[h.neighbor, "detect_frac"]
            obs_raw = r_raw[ti, ci]
            obs_win = r_within[ti, ci]
            # 零分布：与邻居检出率匹配
            sel = np.where(np.abs(pool_det - det_n) <= NULL_TOL * det_n)[0]
            if len(sel) < MIN_NULL:
                sel = np.argsort(np.abs(pool_det - det_n))[:MIN_NULL]
            nv = np.array([r_raw[ti, pool_pos[s][1]] for s in sel])
            nv = nv[np.isfinite(nv)]
            if len(nv) == 0:
                continue
            p_up = (1 + int((nv >= obs_raw).sum())) / (1 + len(nv))
            p_dn = (1 + int((nv <= obs_raw).sum())) / (1 + len(nv))
            delta = obs_raw - nv.mean()
            pair_rows.append({
                "lncrna": r.Index, "lncrna_name": r.gene_name,
                "neighbor": h.neighbor, "neighbor_name": h.neighbor_name,
                "dist_bp": h.dist_bp,
                "lncrna_detect": r.detect_frac, "neighbor_detect": det_n,
                "r_raw_obs": obs_raw, "r_raw_null_mean": nv.mean(),
                "r_raw_null_sd": nv.std(), "delta_raw": delta,
                "p_raw": p_up, "p_raw_low": p_dn,
                "r_within_obs": obs_win,
                "n_null": len(nv),
            })
            if best is None or obs_raw > best["r_raw_obs"]:
                best = pair_rows[-1]
        if best is None:
            cls_rows.append({"lncrna": r.Index, "gene_name": r.gene_name,
                             "detect_frac": r.detect_frac, "n_cis": len(sub),
                             "category": "无法估计"})
            continue
        if best["delta_raw"] > DELTA_THR and best["p_raw"] < 0.05:
            cat = "cis跟随"
        elif best["delta_raw"] < -DELTA_THR and best["p_raw_low"] < 0.05:
            cat = "反耦合"
        else:
            cat = "独立"
        cls_rows.append({
            "lncrna": r.Index, "gene_name": r.gene_name, "detect_frac": r.detect_frac,
            "n_cis": len(sub), "best_neighbor": best["neighbor_name"],
            "best_neighbor_detect": best["neighbor_detect"], "dist_bp": best["dist_bp"],
            "r_raw_cis": best["r_raw_obs"], "r_raw_null_mean": best["r_raw_null_mean"],
            "delta_raw": best["delta_raw"], "p_value": best["p_raw"],
            "r_within_cis": best["r_within_obs"], "category": cat,
        })
        null_rows.append({"lncrna": r.Index, "gene_name": r.gene_name,
                          "null_mean": best["r_raw_null_mean"],
                          "null_sd": best["r_raw_null_sd"], "n_null": best["n_null"]})

    pairs = pd.DataFrame(pair_rows)
    cl = pd.DataFrame(cls_rows)
    # BH-FDR
    if len(cl) and "p_value" in cl:
        m = cl["p_value"].notna()
        p = cl.loc[m, "p_value"].to_numpy()
        order = np.argsort(p)
        q = np.empty_like(p)
        q[order] = np.minimum.accumulate((p[order] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
        cl.loc[m, "q_value"] = np.clip(q, 0, 1)
    return pairs, cl, pd.DataFrame(null_rows)


def run_one(name, qset, cand, pool, g, var, gc, n_groups, win):
    """对给定查询基因集跑一次 cis 耦合分析。"""
    tag = f"{name}_{win // 1000}kb"
    print(f"\n===== {tag} =====")
    cis = find_cis(qset, cand, win)
    n_cover = cis["lncrna"].nunique() if len(cis) else 0
    print(f"  cis 配对 {len(cis)} 对，覆盖 {n_cover} / {len(qset)} 个查询基因")
    if not len(cis):
        return None, None
    pairs, cl, nd = analyse(g, qset, cis, pool, var, gc, n_groups, tag)
    pairs.to_csv(os.path.join(OUT, f"cis_pairs_{tag}.csv"), index=False)
    cl.to_csv(os.path.join(OUT, f"class_{tag}.csv"), index=False)
    nd.to_csv(os.path.join(OUT, f"null_{tag}.csv"), index=False)

    print(f"  [{tag}] 分类:", cl["category"].value_counts().to_dict())
    s2 = cl[cl["n_cis"].fillna(0) > 0].dropna(subset=["delta_raw"]) if "delta_raw" in cl else cl.iloc[0:0]
    if len(s2):
        print(f"  [{tag}] 有邻居 {len(s2)} | delta 中位 {s2['delta_raw'].median():+.4f} "
              f"| delta>0 {int((s2['delta_raw'] > 0).sum())}/{len(s2)} "
              f"| FDR<0.05 {int((s2.get('q_value', pd.Series(dtype=float)) < 0.05).sum())}")
    return pairs, cl


def build_matched_control(lnc, cand, rng):
    """按检出率分布为每个 lncRNA 匹配一个蛋白编码基因，作为阳性对照查询集。"""
    pc = cand[cand["detect_frac"] >= LNC_MIN_DETECT].copy()
    picks = []
    for det in lnc["detect_frac"].to_numpy():
        sel = pc[(pc["detect_frac"] >= det * 0.8) & (pc["detect_frac"] <= det * 1.2)]
        if len(sel) == 0:
            sel = pc.iloc[np.argsort(np.abs(pc["detect_frac"] - det))[:20]]
        picks.append(sel.sample(1, random_state=int(rng.integers(1e9))).index[0])
    picks = list(dict.fromkeys(picks))
    ctrl = pc.loc[picks].copy()
    print(f"  阳性对照查询集（蛋白编码，检出率匹配）: {len(ctrl)}")
    print(f"    检出率中位: lncRNA {lnc['detect_frac'].median():.3f} vs 对照 {ctrl['detect_frac'].median():.3f}")
    return ctrl


def main():
    rng = np.random.default_rng(SEED)
    print("=== 准备 ===")
    pf = pd.read_csv(CLS, index_col=0)
    g = pf[(pf["panel"] == "ISO") & (pf["feature_type"] == "gene")].copy()
    coords = pd.read_csv(COORD, sep="\t")
    g = attach_coords(g, coords)
    lnc, cand, pool = build_sets(g)
    ctrl = build_matched_control(lnc, cand, rng)

    var = sio.read_var(KEY)
    obs = sio.read_obs(KEY, columns=["cell_label"])
    gc, gnames = pd.factorize(obs["cell_label"].astype(str))
    print(f"  分层口径: cell_label（{len(gnames)} 组）—— L-domain 建好后替换")

    for win in WINDOWS:
        run_one("lncRNA", lnc, cand, pool, g, var, gc, len(gnames), win)
        run_one("PC", ctrl, cand, pool, g, var, gc, len(gnames), win)

    # ---- 汇总对比 ----
    print("\n" + "=" * 90)
    print("=== lncRNA vs 蛋白编码阳性对照 ===")
    from scipy import stats
    comp = []
    for win in WINDOWS:
        tag = f"{win // 1000}kb"
        a = pd.read_csv(os.path.join(OUT, f"cis_pairs_lncRNA_{tag}.csv"))
        b = pd.read_csv(os.path.join(OUT, f"cis_pairs_PC_{tag}.csv"))
        u = stats.mannwhitneyu(a["delta_raw"].dropna(), b["delta_raw"].dropna(),
                               alternative="less")
        comp.append({
            "window": tag,
            "n_lncRNA_pairs": len(a), "n_PC_pairs": len(b),
            "lncRNA_delta_median": a["delta_raw"].median(),
            "PC_delta_median": b["delta_raw"].median(),
            "mwu_p_lncRNA_less": u.pvalue,
            "lncRNA_frac_pos": (a["delta_raw"] > 0).mean(),
            "PC_frac_pos": (b["delta_raw"] > 0).mean(),
        })
        print(f"\n[{tag}] lncRNA {len(a)} 对 vs 蛋白编码 {len(b)} 对")
        print(f"  delta 中位: lncRNA {a['delta_raw'].median():+.4f} | PC {b['delta_raw'].median():+.4f}")
        print(f"  delta>0 比例: lncRNA {(a['delta_raw']>0).mean():.1%} | PC {(b['delta_raw']>0).mean():.1%}")
        print(f"  Mann-Whitney（单侧 lncRNA < PC）p = {u.pvalue:.3e}")
    pd.DataFrame(comp).to_csv(os.path.join(OUT, "compare_lncRNA_vs_PC.csv"), index=False)

    # ---- 图 ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, win in zip(axes, WINDOWS):
        tag = f"{win // 1000}kb"
        a = pd.read_csv(os.path.join(OUT, f"cis_pairs_lncRNA_{tag}.csv"))["delta_raw"].dropna()
        b = pd.read_csv(os.path.join(OUT, f"cis_pairs_PC_{tag}.csv"))["delta_raw"].dropna()
        bins = np.linspace(min(a.min(), b.min()), max(a.max(), b.max()), 40)
        ax.hist(b, bins=bins, alpha=0.55, label="protein-coding (control)", color="#4C78A8")
        ax.hist(a, bins=bins, alpha=0.65, label="lncRNA", color="#D1495B")
        ax.axvline(0, color="grey", ls="--", lw=1)
        ax.set_xlabel("delta (cis neighbor − matched null)")
        ax.set_ylabel("pairs")
        ax.set_title(f"cis coupling at ±{win//1000} kb")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_h2_lncrna_vs_pc.png"), dpi=150)
    plt.close(fig)

    print(f"\n=== H2 完成，输出: {OUT} ===")


if __name__ == "__main__":
    main()
