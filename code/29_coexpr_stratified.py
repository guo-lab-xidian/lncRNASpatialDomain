# -*- coding: utf-8 -*-
"""
按切片分层的共表达分析（替代 25 脚本中合并切片的做法）。

原因：三张切片的整体检出深度不同（sample 7 系统性偏低），合并计算会让
"切片间差异"混入相关系数。改为每张切片内单独算 Phi 相关，再用 Fisher z 平均。

产出：results/annotation/coexpression_top200_stratified.csv
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import h5py

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import spatial_io as sio

KEY = "cellxgene_ISO"
RES = r"E:\研究工作\agent长非与空间域\lncRNA空间域\results"
CLS = r"E:\研究工作\agent长非与空间域\analysis\03_lncrna\panel_features_class.csv"
OUT = os.path.join(RES, "annotation")
os.makedirs(OUT, exist_ok=True)

TOP_COEXPR = 200
BLOCK = 15000
TARGETS = ["Miat", "Cdr1", "Gm27000", "Gm49698", "Gm48623", "Gm49655",
           "Gm35240", "Gm47221", "Gm42428", "Gm45163", "Gm45895", "Gm47408",
           "Gm49124", "AY512931", "B230220B15Rik", "A830036E02Rik"]


def main():
    pf = pd.read_csv(CLS, index_col=0)
    pf = pf[~pf.index.duplicated(keep="first")]
    spec = pd.read_csv(os.path.join(RES, "H1H2", "gene_specificity.csv"), index_col=0)
    spec = spec[~spec.index.duplicated(keep="first")]
    lab = pd.read_csv(os.path.join(RES, "Ldomain", "domain_labels.csv"), index_col=0)
    ranges = json.load(open(os.path.join(RES, "Ldomain", "subset_rows.json")))
    var = sio.read_var(KEY)

    pc_idx = pf[(pf["feature_type"] == "gene") & (pf["gene_class"] == "protein_coding")].index
    det = spec["panel_detect"].reindex(pc_idx)
    pc_genes = det[det >= 0.01].index.tolist()
    targets = [g for g in TARGETS if g in var.index]
    print(f"目标 {len(targets)} | 蛋白编码候选 {len(pc_genes):,}")

    tpos = var.index.get_indexer(targets)
    cpos = var.index.get_indexer(pc_genes)
    allpos = np.concatenate([tpos, cpos])
    uniq, inv = np.unique(allpos, return_inverse=True)
    ti, ci = inv[:len(tpos)], inv[len(tpos):]

    sample_arr = lab["sample"].astype(str).to_numpy()
    samples = sorted(set(sample_arr))

    # 每个切片单独累积
    acc = {s: {"P": np.zeros((len(targets), len(pc_genes))),
               "dt": np.zeros(len(targets)), "dc": np.zeros(len(pc_genes)), "n": 0}
           for s in samples}

    path = sio._resolve(KEY)
    with h5py.File(path, "r") as f:
        n_var = f["var/_index"].shape[0]
        indptr = f["X/indptr"][:]
        dset, iset = f["X/data"], f["X/indices"]
        off = 0
        for r in ranges:
            for start in range(r["start"], r["stop"], BLOCK):
                end = min(start + BLOCK, r["stop"])
                lo, hi = int(indptr[start]), int(indptr[end])
                blk = csr_matrix(((dset[lo:hi] > 0).astype(np.float32), iset[lo:hi],
                                  indptr[start:end + 1] - lo),
                                 shape=(end - start, n_var))[:, uniq]
                m = end - start
                ss = sample_arr[off:off + m]
                off += m
                for s in samples:
                    mask = ss == s
                    if not mask.any():
                        continue
                    T = blk[mask][:, ti].tocsr()
                    C = blk[mask][:, ci].tocsr()
                    acc[s]["P"] += np.asarray((T.T @ C).todense())
                    acc[s]["dt"] += np.asarray(T.sum(axis=0)).ravel()
                    acc[s]["dc"] += np.asarray(C.sum(axis=0)).ravel()
                    acc[s]["n"] += int(mask.sum())
        print("累积完成:", {s: acc[s]["n"] for s in samples})

    # 每切片相关 → Fisher z 平均
    zs = []
    for s in samples:
        a = acc[s]
        n = a["n"]
        pt = a["dt"] / n
        pc = a["dc"] / n
        cov = a["P"] - n * np.outer(pt, pc)
        vt = a["dt"] - n * pt ** 2
        vc = a["dc"] - n * pc ** 2
        with np.errstate(divide="ignore", invalid="ignore"):
            r = cov / np.sqrt(np.outer(vt, vc))
        r = np.clip(np.nan_to_num(r, nan=0.0), -0.999, 0.999)
        zs.append(np.arctanh(r))
        print(f"  sample {s}: {n:,} 细胞 | r 中位 {np.median(r):.4f}")
    z = np.mean(zs, axis=0)
    r_avg = np.tanh(z)

    rows = []
    for i, g in enumerate(targets):
        rr = pd.Series(r_avg[i], index=pc_genes).sort_values(ascending=False)
        top = rr.head(TOP_COEXPR)
        rows.append({"gene_name": g, "top1": top.index[0], "top1_r": top.iloc[0],
                     "top5": ";".join(top.index[:5]),
                     "top200": ";".join(top.index)})
    out = pd.DataFrame(rows).set_index("gene_name")
    out.to_csv(os.path.join(OUT, "coexpression_top200_stratified.csv"))
    print("\n=== 分层共表达 Top5（前 8 个 lncRNA）===")
    print(out[["top1", "top1_r", "top5"]].head(8).round(3).to_string())

    from collections import Counter
    cnt = Counter()
    for r in out.itertuples():
        for x in r.top5.split(";"):
            cnt[x] += 1
    print("\n=== 高频共表达基因（Top5 中出现次数）===")
    for x, n in cnt.most_common(12):
        print(f"  {x:14} {n}")
    print(f"\n=== 完成，输出: {OUT}/coexpression_top200_stratified.csv ===")


if __name__ == "__main__":
    main()
