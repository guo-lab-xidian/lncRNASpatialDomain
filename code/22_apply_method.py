# -*- coding: utf-8 -*-
"""
用 BB-DSI 的两种检验重跑真实数据，比较"是否做细胞类型条件化"的差异。

  MC-Pearson：零假设 = 检出率与域无关（不管细胞类型组成）
  MC-CT     ：零假设 = 检出率只由细胞类型决定（剔除组成效应）

核心问题：跨面板可重复的那批 lncRNA 中，有多少在剔除细胞类型组成后依然显著？
—— 这些才是真正的"空间域标记"，而非细胞类型标记。

用法：python 22_apply_method.py ISO|WT
产出：lncRNA空间域/results/Method/apply_<panel>.csv
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import spatial_io as sio
import bb_dsi as bb

PANEL = (sys.argv[1] if len(sys.argv) > 1 else "ISO").upper()
SUFFIX = "" if PANEL == "ISO" else "_" + PANEL
KEY = "cellxgene_" + PANEL
RES = r"E:\研究工作\agent长非与空间域\lncRNA空间域\results"
LDIR = os.path.join(RES, "Ldomain" + SUFFIX)
CLS = r"E:\研究工作\agent长非与空间域\analysis\03_lncrna\panel_features_class.csv"
OUT = os.path.join(RES, "Method")
os.makedirs(OUT, exist_ok=True)

LNC_MIN_DETECT = 0.05
N_CTRL = 5
N_SIM = 5000
SIM_CHUNK = 1000
LAPLACE = 20.0
SEED = 0


def bh_fdr(p):
    p = np.asarray(p, dtype=float)
    q = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    if ok.sum() == 0:
        return q
    pv = p[ok]
    order = np.argsort(pv)
    ranked = pv[order] * len(pv) / np.arange(1, len(pv) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    tmp = np.empty_like(pv)
    tmp[order] = np.clip(ranked, 0, 1)
    q[ok] = tmp
    return q


def mc_ct_pvalue(det, cell_codes, dom_codes, K, n_cell, n_ck, n_c, n_sim=N_SIM,
                 seed=0, laplace=LAPLACE, chunk=SIM_CHUNK):
    """MC-CT 的逐基因实现（零分布中重新估计细胞类型检出率）。"""
    n_cells = len(det)
    y_c = np.bincount(cell_codes, weights=det, minlength=n_cell)
    O = np.bincount(dom_codes, weights=det, minlength=K)
    p_glob = float(det.mean())
    rate = (y_c + laplace * p_glob) / (n_c + laplace)
    E = n_ck @ rate
    V = n_ck @ (rate * (1 - rate))
    sd = np.sqrt(np.maximum(V, 1e-12))
    T = float((((O - E) / sd) ** 2).sum())

    rng = np.random.default_rng(seed)
    cnt, done = 0, 0
    while done < n_sim:
        b = min(chunk, n_sim - done)
        Okc = rng.binomial(n_ck[None, :, :], rate[None, None, :],
                           size=(b, K, n_cell)).astype(np.float32)
        Os = Okc.sum(axis=2)
        yc = Okc.sum(axis=1)
        pgl = Os.sum(axis=1) / n_cells
        rt = (yc + laplace * pgl[:, None]) / (n_c + laplace)[None, :]
        Es = rt @ n_ck.T
        Vs = (rt * (1 - rt)) @ n_ck.T
        Ts = (((Os - Es) / np.sqrt(np.maximum(Vs, 1e-12))) ** 2).sum(axis=1)
        cnt += int((Ts >= T).sum())
        done += b
    return (1 + cnt) / (1 + n_sim), T


def main():
    rng = np.random.default_rng(SEED)
    lab = pd.read_csv(os.path.join(LDIR, "domain_labels.csv"), index_col=0)
    ranges = json.load(open(os.path.join(LDIR, "subset_rows.json")))
    dom_codes, dom_names = pd.factorize(lab["L_domain"].astype(str))
    cell_codes, cell_names = pd.factorize(lab["cell_label"].astype(str))
    K, C = len(dom_names), len(cell_names)
    n_k = np.bincount(dom_codes, minlength=K)
    n_c = np.bincount(cell_codes, minlength=C).astype(float)
    n_ck = np.zeros((K, C), dtype=np.int64)
    np.add.at(n_ck, (dom_codes, cell_codes), 1)
    print(f"{PANEL}: {len(lab):,} 细胞 | {K} 域 | {C} 细胞类型")

    spec = pd.read_csv(os.path.join(RES, "H1H2" + SUFFIX, "gene_specificity.csv"), index_col=0)
    lnc = spec[(spec["gene_class"] == "lncRNA") & (spec["panel_detect"] >= LNC_MIN_DETECT)]
    pc = spec[(spec["gene_class"] == "protein_coding") & (spec["panel_detect"] >= LNC_MIN_DETECT)]

    # lncRNA + 检出率匹配的蛋白编码对照
    rows = []
    for r in lnc.itertuples():
        sel = pc[(pc["panel_detect"] >= r.panel_detect * 0.8) &
                 (pc["panel_detect"] <= r.panel_detect * 1.2)]
        if len(sel) < N_CTRL:
            sel = pc.iloc[np.argsort(np.abs(pc["panel_detect"] - r.panel_detect))[:N_CTRL]]
        for g in sel.sample(min(N_CTRL, len(sel)),
                            random_state=int(rng.integers(1e9))).index:
            rows.append({"gene": g, "group": "protein_coding_control", "matched_to": r.Index})
        rows.append({"gene": r.Index, "group": "lncRNA", "matched_to": None})
    design = pd.DataFrame(rows).drop_duplicates("gene").reset_index(drop=True)
    genes = list(design["gene"])
    print(f"待检验基因: {len(genes)}（lncRNA {len(lnc)} + 匹配对照 {len(genes)-len(lnc)}）")

    # 读取子集表达（只取这些基因列）
    var = sio.read_var(KEY)
    pos = var.index.get_indexer(genes)
    assert (pos >= 0).all()
    blocks = []
    for r in ranges:
        blocks.append(sio.read_rows(KEY, r["start"], r["stop"])[:, pos].tocsr())
    from scipy.sparse import vstack
    X = vstack(blocks, format="csr")
    print(f"表达矩阵: {X.shape} | nnz={X.nnz:,}")
    Xb = (X > 0).astype(np.float32)

    res = []
    t0 = time.time()
    for gi, g in enumerate(genes):
        det = np.asarray(Xb[:, gi].todense()).ravel().astype(float)
        if det.sum() < 5:
            res.append({"gene": g, "n_pos": float(det.sum()), "p_pearson": np.nan,
                        "p_ct": np.nan, "T_ct": np.nan})
            continue
        y = np.bincount(dom_codes, weights=det, minlength=K)
        p_pear, _, _, _ = bb.pvalue_mc_pearson(y, n_k, n_sim=N_SIM, seed=SEED)
        p_ct, Tct = mc_ct_pvalue(det, cell_codes, dom_codes, K, C, n_ck, n_c, seed=SEED)
        res.append({"gene": g, "n_pos": float(det.sum()),
                    "p_pearson": p_pear, "p_ct": p_ct, "T_ct": Tct})
        if (gi + 1) % 50 == 0:
            print(f"  {gi+1}/{len(genes)} | {time.time()-t0:.0f}s", flush=True)

    out = pd.DataFrame(res).set_index("gene").join(design.set_index("gene"))
    out["gene_name"] = spec["gene_name"].reindex(out.index)
    out["panel_detect"] = spec["panel_detect"].reindex(out.index)
    out["DSI_domain"] = spec["DSI_domain"].reindex(out.index)
    out["q_pearson"] = bh_fdr(out["p_pearson"])
    out["q_ct"] = bh_fdr(out["p_ct"])
    out.to_csv(os.path.join(OUT, f"apply_{PANEL}.csv"))

    print("\n=== 结果 ===")
    for grp in ["lncRNA", "protein_coding_control"]:
        s = out[out["group"] == grp]
        n_p = int((s["q_pearson"] < 0.05).sum())
        n_c = int((s["q_ct"] < 0.05).sum())
        print(f"  {grp:24} n={len(s):>4} | MC-Pearson 显著 {n_p:>3} "
              f"| MC-CT 显著 {n_c:>3} | 条件化后保留率 {n_c/max(n_p,1):.1%}")

    print("\n=== lncRNA：仅不条件化显著 vs 两者都显著 ===")
    s = out[out["group"] == "lncRNA"]
    both = s[(s["q_pearson"] < 0.05) & (s["q_ct"] < 0.05)]
    only = s[(s["q_pearson"] < 0.05) & ~(s["q_ct"] < 0.05)]
    print(f"  两者都显著（真·空间域标记）: {len(both)}")
    print(both.nlargest(15, "DSI_domain")[["gene_name", "panel_detect", "DSI_domain",
                                           "q_pearson", "q_ct"]].round(4).to_string(index=False))
    print(f"\n  仅不条件化显著（疑似细胞类型连带）: {len(only)}")
    print(only.nlargest(10, "DSI_domain")[["gene_name", "panel_detect", "DSI_domain",
                                           "q_pearson", "q_ct"]].round(4).to_string(index=False))

    print(f"\n=== {PANEL} 完成，输出: {OUT}/apply_{PANEL}.csv ===")


if __name__ == "__main__":
    main()
