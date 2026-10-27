# -*- coding: utf-8 -*-
"""
BB-DSI 仿真评估：在真实空间结构下检验各方法的校准与功效。

仿真底图：直接使用真实 MERFISH 细胞的坐标与 L-domain 标签（25 个域、45,000 细胞取样），
保证空间自相关结构、域大小不平衡等特性与真实数据一致。

基因类别：
  null          —— 全细胞均匀检出（真阴性）
  domain        —— 在 1 个域内检出率提升 r 倍（真阳性）
  celltype      —— 检出由细胞类型决定，与域无直接关系（混杂类，单独报告）

评价：
  1. null 基因在名义 5% 下的经验假阳性率（校准）
  2. domain 基因在 FDR=5% 下的功效与 FDR
  3. AUROC
  4. celltype 基因被误判的比例（混杂敏感性）
"""
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bb_dsi as bb

RES = r"E:\研究工作\agent长非与空间域\lncRNA空间域\results"
LDIR = os.path.join(RES, "Ldomain")
OUT = os.path.join(RES, "Method")
os.makedirs(OUT, exist_ok=True)

N_CELLS = 20000
N_NULL = 200
N_DOMAIN = 100
N_CELLTYPE = 100
PIS = [0.005, 0.01, 0.02, 0.05]
R_EFFECT = 3.0
N_SIM_NULL = 5000   # 置换次数决定 p 值分辨率；500 时最小 p=0.002，BH 后无法达 FDR 5%
KAPPA = 100.0       # 收缩强度（伪计数）；越大越保守
SEED = 0


def bh_fdr(p):
    p = np.asarray(p, dtype=float)
    q = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    pv = p[ok]
    if len(pv) == 0:
        return q
    order = np.argsort(pv)
    ranked = pv[order] * len(pv) / np.arange(1, len(pv) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    tmp = np.empty_like(pv)
    tmp[order] = np.clip(ranked, 0, 1)
    q[ok] = tmp
    return q


def auroc(score, label):
    label = np.asarray(label, dtype=bool)
    if label.sum() == 0 or (~label).sum() == 0:
        return np.nan
    u = stats.mannwhitneyu(score[label], score[~label], alternative="two-sided").statistic
    return float(u / (label.sum() * (~label).sum()))


def main():
    rng = np.random.default_rng(SEED)
    lab = pd.read_csv(os.path.join(LDIR, "domain_labels.csv"), index_col=0)
    if len(lab) > N_CELLS:
        sel = rng.choice(len(lab), N_CELLS, replace=False)
        sel.sort()
        lab = lab.iloc[sel]
    dom = lab["L_domain"].astype(str).to_numpy()
    cell = lab["cell_label"].astype(str).to_numpy()
    dom_codes, dom_names = pd.factorize(dom)
    cell_codes, cell_names = pd.factorize(cell)
    K = len(dom_names)
    n_k = np.bincount(dom_codes, minlength=K)
    print(f"仿真底图: {len(lab):,} 细胞 | {K} 个域 | 细胞类型 {len(cell_names)} 类")
    print(f"域大小: min={n_k.min()}, 中位={int(np.median(n_k))}, max={n_k.max()}")

    # 空间平滑算子（按切片分别构造，因为三张切片坐标系不同）
    print("构造空间平滑算子 ...", flush=True)
    W = np.zeros((len(lab), len(lab)), dtype=np.float32) if False else None
    from scipy.sparse import block_diag
    blocks = []
    sample_arr = lab["sample"].astype(str).to_numpy()
    for s in sorted(set(sample_arr)):
        m = sample_arr == s
        idx = np.where(m)[0]
        coord = lab.loc[m, ["global_x", "global_y"]].to_numpy()
        Ws = bb.build_smoother(coord, k=8)
        blocks.append((idx, Ws))
    # 组装成全局块对角矩阵（按原顺序）
    from scipy.sparse import coo_matrix
    rows, cols, vals = [], [], []
    for idx, Ws in blocks:
        Wc = Ws.tocoo()
        rows.append(idx[Wc.row])
        cols.append(idx[Wc.col])
        vals.append(Wc.data)
    W = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                   shape=(len(lab), len(lab))).tocsr()
    print(f"  平滑算子: {W.shape}, nnz={W.nnz:,}")

    rows = []
    for pi in PIS:
        t0 = time.time()
        print(f"\n===== π = {pi:.3%} =====")

        genes = []          # (类别, 检出向量)
        # null
        for _ in range(N_NULL):
            genes.append(("null", rng.random(len(dom)) < pi))
        # domain-specific
        for _ in range(N_DOMAIN):
            k = rng.integers(K)
            p_vec = np.where(dom_codes == k, min(pi * R_EFFECT, 0.95), pi)
            genes.append(("domain", rng.random(len(dom)) < p_vec))
        # cell-type driven（与域无直接关系，但因域与细胞类型相关而"看似"域特异）
        for _ in range(N_CELLTYPE):
            c = rng.integers(len(cell_names))
            p_vec = np.where(cell_codes == c, min(pi * R_EFFECT, 0.95), pi)
            genes.append(("celltype", rng.random(len(dom)) < p_vec))

        labels = [g[0] for g in genes]
        res = {"mcct": [], "mcpearson": [], "smooth": [], "dsimax": [], "chisq": [], "maxbin": [],
               "dsi_raw": [], "dsi_shrunk": []}
        for kind, det in genes:
            y = np.bincount(dom_codes, weights=det.astype(float), minlength=K)
            # 以细胞类型为条件的检验（剥离细胞类型混杂）
            res["mcct"].append(bb.pvalue_mc_ct_conditioned(det.astype(float), cell_codes, dom_codes, K, len(cell_names), n_sim=2000, seed=SEED)["pvalue"])
            # 空间平滑检验（利用邻域证据，稀疏端提升信噪比）
            res["smooth"].append(
                bb.smoothed_domain_test(det.astype(float), W, dom_codes, K)["pvalue"])
            # 主方法：Pearson 统计量 + 蒙特卡洛精确零分布（按域大小加权，稀疏端仍校准）
            res["mcpearson"].append(
                bb.pvalue_mc_pearson(y, n_k, n_sim=N_SIM_NULL, seed=SEED)[0])
            # 消融：收缩 max−median 统计量
            res["dsimax"].append(
                bb.pvalue_shrunk_fixed(y, n_k, kappa=KAPPA, n_sim=N_SIM_NULL, seed=SEED)[0])
            # 基线
            res["chisq"].append(bb.pvalue_chisq(y, n_k))
            res["maxbin"].append(bb.pvalue_max_binomial(y, n_k))
            r = bb.bb_dsi(y, n_k)
            res["dsi_raw"].append(r["dsi_raw"])
            res["dsi_shrunk"].append(r["dsi_shrunk"])

        df = pd.DataFrame(res)
        df["kind"] = labels
        is_null = df["kind"] == "null"
        is_dom = df["kind"] == "domain"
        is_ct = df["kind"] == "celltype"

        for m in ["mcct", "smooth", "mcpearson", "dsimax", "chisq", "maxbin"]:
            p = df[m].to_numpy()
            q = bh_fdr(p)
            df[m + "_q"] = q

        summ = {"pi": pi}
        for m in ["mcct", "smooth", "mcpearson", "dsimax", "chisq", "maxbin"]:
            p = df[m].to_numpy()
            # 校准：null 基因在名义 5% 下的经验假阳性率
            fpr = np.nanmean(p[is_null] < 0.05) if is_null.any() else np.nan
            # 功效：FDR=5% 时 domain 基因的检出率
            q = df[m + "_q"].to_numpy()
            calls = q < 0.05
            tp = (calls & is_dom.to_numpy()).sum()
            fp = (calls & is_null.to_numpy()).sum()
            power = tp / max(is_dom.sum(), 1)
            fdr = fp / max(tp + fp, 1)
            ct_fp = np.nanmean(q[is_ct.to_numpy()] < 0.05) if is_ct.any() else np.nan
            au = auroc(-p, is_dom.to_numpy())
            summ[f"{m}_FPR_null"] = fpr
            summ[f"{m}_power"] = power
            summ[f"{m}_FDR"] = fdr
            summ[f"{m}_celltype_callrate"] = ct_fp
            summ[f"{m}_AUROC"] = au
        summ["bb_dsi_shrunk_median_null"] = np.nanmedian(df.loc[is_null, "dsi_shrunk"])
        summ["bb_dsi_raw_median_null"] = np.nanmedian(df.loc[is_null, "dsi_raw"])
        rows.append(summ)
        print(f"  null FPR@5%: 平滑 {summ['smooth_FPR_null']:.3f} | MC-Pearson {summ['mcpearson_FPR_null']:.3f} | "
              f"卡方 {summ['chisq_FPR_null']:.3f} | 最大二项 {summ['maxbin_FPR_null']:.3f}")
        print(f"  功效:       平滑 {summ['smooth_power']:.3f} | MC-Pearson {summ['mcpearson_power']:.3f} | "
              f"收缩DSImax {summ['dsimax_power']:.3f} | "
              f"卡方 {summ['chisq_power']:.3f} | 最大二项 {summ['maxbin_power']:.3f}")
        print(f"  AUROC:      平滑 {summ['smooth_AUROC']:.3f} | MC-Pearson {summ['mcpearson_AUROC']:.3f} | "
              f"收缩DSImax {summ['dsimax_AUROC']:.3f} | "
              f"卡方 {summ['chisq_AUROC']:.3f} | 最大二项 {summ['maxbin_AUROC']:.3f}")
        print(f"  用时 {time.time()-t0:.0f}s", flush=True)

    s = pd.DataFrame(rows)
    s.to_csv(os.path.join(OUT, "simulation_summary.csv"), index=False)
    print("\n=== 汇总 ===")
    cols = ["pi", "mcct_FPR_null", "smooth_FPR_null", "mcpearson_FPR_null", "dsimax_FPR_null", "chisq_FPR_null", "maxbin_FPR_null",
            "mcct_power", "smooth_power", "mcpearson_power", "dsimax_power", "chisq_power", "maxbin_power",
            "mcct_AUROC", "smooth_AUROC", "mcpearson_AUROC", "dsimax_AUROC", "chisq_AUROC", "maxbin_AUROC"]
    print(s[cols].round(4).to_string(index=False))

    # 图
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for m, c in [("smooth", "#D1495B"), ("mcpearson", "#9B59B6"), ("dsimax", "#F28E2B"), ("chisq", "#4C78A8"), ("maxbin", "#59A14F")]:
        axes[0].plot(s["pi"] * 100, s[f"{m}_FPR_null"], "o-", color=c, label=m)
        axes[1].plot(s["pi"] * 100, s[f"{m}_power"], "o-", color=c, label=m)
        axes[2].plot(s["pi"] * 100, s[f"{m}_AUROC"], "o-", color=c, label=m)
    axes[0].axhline(0.05, color="grey", ls="--", lw=1, label="nominal 5%")
    axes[0].set_xlabel("overall detection rate π (%)")
    axes[0].set_ylabel("empirical FPR (null genes)")
    axes[0].set_title("Calibration")
    axes[1].set_xlabel("detection rate π (%)")
    axes[1].set_ylabel("power (FDR=5%)")
    axes[1].set_title("Power for domain-specific genes")
    axes[2].set_xlabel("detection rate π (%)")
    axes[2].set_ylabel("AUROC")
    axes[2].set_title("Discrimination")
    for ax in axes:
        ax.legend(fontsize=8)
        ax.set_xscale("log")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_simulation.png"), dpi=150)
    plt.close(fig)
    print(f"\n=== 仿真完成，输出: {OUT} ===")


if __name__ == "__main__":
    main()






