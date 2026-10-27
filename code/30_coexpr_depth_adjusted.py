# -*- coding: utf-8 -*-
"""
控制"每细胞检出深度"后的共表达分析。

问题：在二值稀疏数据里，每细胞的总检出数差异巨大（50–3,400）。
     检出深的细胞几乎所有基因都更容易被检出，使任意两个中高表达基因出现虚假正相关。
     上一版 GO 富集全部落在"RNA Processing"，正是这个伪影的典型表现。

做法：把细胞按"每细胞总检出基因数"在当前切片内分成 20 个深度层，
     计算**层内相关**（扣除每层均值后合池），再做 Fisher z 跨切片平均。

产出：results/annotation/coexpression_depth_adjusted.csv
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
CLS = r"E:\研究方向"  # placeholder overwritten below
CLS = r"E:\研究工作\agent长非与空间域\analysis\03_lncrna\panel_features_class.csv"
OUT = os.path.join(RES, "annotation")
os.makedirs(OUT, exist_ok=True)

TOP_COEXPR = 200
BLOCK = 15000
N_DEPTH_BINS = 20
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
    n_t, n_c = len(tpos), len(cpos)

    # ---------- 第一遍：读每细胞检出深度 ----------
    depth = np.zeros(len(lab), dtype=np.int32)
    path = sio._resolve(KEY)
    with h5py.File(path, "r") as f:
        indptr = f["X/indptr"][:]
        off = 0
        for r in ranges:
            for start in range(r["start"], r["stop"], BLOCK):
                end = min(start + BLOCK, r["stop"])
                depth[off:off + end - start] = np.diff(indptr[start:end + 1])
                off += end - start
    print(f"检出深度: 中位 {np.median(depth):.0f} | 范围 {depth.min()}–{depth.max()}")

    # ---------- 深度分层（每切片内分位） ----------
    group = np.full(len(lab), -1, dtype=int)
    gid = 0
    for s in samples:
        m = sample_arr == s
        q = pd.qcut(depth[m], N_DEPTH_BINS, labels=False, duplicates="drop")
        group[m] = q + gid
        gid += int(q.max()) + 1
    G = gid
    print(f"深度层数: {G}（{len(samples)} 切片 × {N_DEPTH_BINS} 层）")

    # ---------- 第二遍：累积跨积与分层和 ----------
    P = np.zeros((n_t, n_c))
    SgT = np.zeros((G, n_t))
    SgC = np.zeros((G, n_c))
    ng = np.zeros(G)
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
                T = blk[:, ti].tocsr()
                C = blk[:, ci].tocsr()
                P += np.asarray((T.T @ C).todense())
                g = group[off:off + m]
                Gm = csr_matrix((np.ones(m, dtype=np.float32), (np.arange(m), g)),
                                shape=(m, G))
                SgT += np.asarray((Gm.T @ T).todense())
                SgC += np.asarray((Gm.T @ C).todense())
                ng += np.bincount(g, minlength=G)
                off += m
    print("累积完成")

    # ---------- 层内相关 ----------
    St = SgT.sum(axis=0)          # 各基因总检出
    Sc = SgC.sum(axis=0)
    cov = P - SgT.T @ (SgC / ng[:, None])
    var_t = St - np.asarray((SgT ** 2 / ng[:, None]).sum(axis=0)).ravel()
    var_c = Sc - np.asarray((SgC ** 2 / ng[:, None]).sum(axis=0)).ravel()
    with np.errstate(divide="ignore", invalid="ignore"):
        r = cov / np.sqrt(np.outer(var_t, var_c))
    r = np.nan_to_num(r, nan=0.0)
    print(f"层内相关: 中位 {np.median(r):.4f} | 范围 {r.min():.3f}–{r.max():.3f}")

    rows = []
    for i, g in enumerate(targets):
        rr = pd.Series(r[i], index=pc_genes).sort_values(ascending=False)
        top = rr.head(TOP_COEXPR)
        rows.append({"gene_name": g, "top1": top.index[0], "top1_r": top.iloc[0],
                     "top5": ";".join(top.index[:5]),
                     "top200": ";".join(top.index)})
    out = pd.DataFrame(rows).set_index("gene_name")
    out.to_csv(os.path.join(OUT, "coexpression_depth_adjusted.csv"))

    print("\n=== 深度校正后 Top5 ===")
    print(out[["top1", "top1_r", "top5"]].round(3).to_string())

    from collections import Counter
    cnt = Counter()
    for r_ in out.itertuples():
        for x in r_.top5.split(";"):
            cnt[x] += 1
    print("\n=== 高频共表达基因 ===")
    for x, n in cnt.most_common(12):
        print(f"  {x:14} {n}")
    print(f"\n=== 完成，输出: {OUT}/coexpression_depth_adjusted.csv ===")


if __name__ == "__main__":
    main()
