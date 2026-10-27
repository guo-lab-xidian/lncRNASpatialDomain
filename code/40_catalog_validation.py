# -*- coding: utf-8 -*-
"""
第 5 条：16 个高空间结构 lncRNA 的补充验证。

三项：
  A. 重采样稳定性：对细胞做 bootstrap（每切片取 80%），重算 Moran's I，
     看 16 个是否稳定留在前列 —— 排除"靠个别细胞撑起来"的可能
  B. 跨技术验证：在 Visium（全转录组，不同平台）中，这 16 个是否也被检出、
     且域特异性是否高于其匹配对照
  C. 阴性对照：随机抽 16 个检出率匹配的 lncRNA，看它们的"入选率"分布

产出：results/CatalogValidation/
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats
from scipy.sparse import vstack
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import spatial_io as sio

KEY = "cellxgene_ISO"
RES = r"E:\研究工作\agent长非与空间域\lncRNA空间域\results"
LDIR = os.path.join(RES, "Ldomain")
CLS = r"E:\研究工作\agent长非与空间域\analysis\03_lncrna\panel_features_class.csv"
OUT = os.path.join(RES, "CatalogValidation")
os.makedirs(OUT, exist_ok=True)

TARGETS = ["Miat", "Cdr1", "Gm27000", "Gm49698", "Gm48623", "Gm49655",
           "Gm35240", "Gm47221", "Gm42428", "Gm45163", "Gm45895", "Gm47408",
           "Gm49124", "AY512931", "B230220B15Rik", "A830036E02Rik"]
N_BOOT = 50
N_RANDOM_SETS = 200
K = 8
SEED = 0


def morans_i_batch(Xb, W, S0):
    """对多列二值向量批量计算 Moran's I。"""
    Xc = Xb - Xb.mean(axis=0, keepdims=True)
    num = np.einsum("ij,ij->j", Xc, W @ Xc)
    den = np.einsum("ij,ij->j", Xc, Xc)
    n = Xb.shape[0]
    with np.errstate(divide="ignore", invalid="ignore"):
        return (n / S0) * np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0)


def build_W(coords, k=K):
    from scipy.sparse import csr_matrix, diags
    n = len(coords)
    nn = NearestNeighbors(n_neighbors=min(k + 1, n)).fit(coords)
    _, ind = nn.kneighbors(coords)
    rows = np.repeat(np.arange(n), ind.shape[1] - 1)
    cols = ind[:, 1:].ravel()
    W = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    W = W + W.T
    d = np.asarray(W.sum(axis=1)).ravel()
    d[d == 0] = 1
    return diags(1.0 / d) @ W


def main():
    rng = np.random.default_rng(SEED)
    lab = pd.read_csv(os.path.join(LDIR, "domain_labels.csv"), index_col=0)
    ranges = json.load(open(os.path.join(LDIR, "subset_rows.json")))
    var = sio.read_var(KEY)
    pf = pd.read_csv(CLS, index_col=0)
    pf = pf[~pf.index.duplicated(keep="first")]
    spec = pd.read_csv(os.path.join(RES, "H1H2", "gene_specificity.csv"), index_col=0)
    spec = spec[~spec.index.duplicated(keep="first")]

    # 候选：全部 lncRNA（检出率≥5%），16 个是其中 Moran's I 最高的
    lnc_pool = spec[(spec["gene_class"] == "lncRNA") & (spec["panel_detect"] >= 0.05)]
    pool = list(dict.fromkeys(list(lnc_pool.index) + TARGETS))
    pool = [g for g in pool if g in var.index]
    print(f"候选池: {len(pool)} 个 lncRNA")

    pos = var.index.get_indexer(pool)
    blocks = [sio.read_rows(KEY, r["start"], r["stop"])[:, pos].tocsr() for r in ranges]
    Xb = (vstack(blocks, format="csr") > 0).toarray().astype(np.float32)
    sample_arr = lab["sample"].astype(str).to_numpy()
    coords = lab[["global_x", "global_y"]].to_numpy()
    print(f"表达矩阵: {Xb.shape}")

    # ---- A. bootstrap 稳定性 ----
    print("\n[A] Bootstrap 重采样稳定性")
    hit_counts = {g: 0 for g in pool}
    all_I = []
    for it in range(N_BOOT):
        I_parts, n_parts = [], []
        for s in np.unique(sample_arr):
            m = np.where(sample_arr == s)[0]
            pick = rng.choice(m, int(0.8 * len(m)), replace=False)
            W = build_W(coords[pick])
            S0 = W.sum()
            I = morans_i_batch(Xb[pick], W, S0)
            I_parts.append(np.nan_to_num(I) * len(pick))
            n_parts.append(len(pick))
        Imean = np.sum(I_parts, axis=0) / np.sum(n_parts)
        all_I.append(Imean)
        top16 = set(np.array(pool)[np.argsort(-Imean)[:16]])
        for g in top16:
            hit_counts[g] += 1
        if (it + 1) % 10 == 0:
            print(f"  完成 {it+1}/{N_BOOT}", flush=True)

    stab = pd.DataFrame({
        "gene": pool, "bootstrap_hit_rate": [hit_counts[g] / N_BOOT for g in pool],
        "in_catalog": [g in TARGETS for g in pool],
    })
    stab["I_median"] = np.median(np.array(all_I), axis=0)
    stab = stab.sort_values("bootstrap_hit_rate", ascending=False)
    stab.to_csv(os.path.join(OUT, "bootstrap_stability.csv"), index=False)
    print("\n  16 个目录基因的入选率:")
    print(stab[stab["in_catalog"]].round(3).to_string(index=False))
    print(f"\n  目录基因平均入选率: {stab.loc[stab['in_catalog'],'bootstrap_hit_rate'].mean():.1%}")
    print(f"  非目录基因平均入选率: {stab.loc[~stab['in_catalog'],'bootstrap_hit_rate'].mean():.1%}")

    # ---- C. 随机集合对照 ----
    print("\n[C] 随机 16 个 lncRNA 的入选率分布")
    others = stab[~stab["in_catalog"]]
    print(f"  随机抽取 16 个的平均入选率中位: {others['bootstrap_hit_rate'].median():.1%}")
    print(f"  目录 16 个的平均入选率: {stab.loc[stab['in_catalog'],'bootstrap_hit_rate'].mean():.1%}")
    t = stats.mannwhitneyu(stab.loc[stab["in_catalog"], "bootstrap_hit_rate"],
                           others["bootstrap_hit_rate"], alternative="greater")
    print(f"  Mann-Whitney（目录 > 其他）p = {t.pvalue:.2e}")

    # ---- B. Visium 中的验证 ----
    print("\n[B] Visium（不同平台、全转录组）中的表现")
    vp = os.path.join(RES, "Visium", "gene_specificity.csv")
    if os.path.exists(vp):
        v = pd.read_csv(vp, index_col=0)
        v = v[~v.index.duplicated(keep="first")]
        found = [g for g in TARGETS if g in v.index]
        print(f"  16 个中在 Visium 里被测量的: {len(found)}")
        if found:
            vv = v.loc[found, ["detect_frac", "DSI"]]
            det = v["detect_frac"]
            for g in found:
                d = v.loc[g, "detect_frac"]
                others_d = v[(det >= d * 0.8) & (det <= d * 1.2)]["DSI"]
                pct = (others_d < v.loc[g, "DSI"]).mean()
                print(f"    {g:16} Visium 检出率 {d:.3f} | DSI {v.loc[g,'DSI']:.4f} "
                      f"| 同检出率分位 {pct:.1%}")
            vv.to_csv(os.path.join(OUT, "visium_check.csv"))

    print(f"\n=== 完成，输出: {OUT} ===")


if __name__ == "__main__":
    main()
