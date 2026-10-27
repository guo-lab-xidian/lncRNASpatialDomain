# -*- coding: utf-8 -*-
"""
第 2 条（负结果）的稳健性补充实验。

四项检验：
  A. 分层匹配：按检出率十分位分层，在每一层内比较 lncRNA 与蛋白编码的域特异性
     —— 若每层都无差异，"整体无差异"就不是匹配方式造成的
  B. 等效性检验（TOST）：不看"是否显著不同"，而看"差异是否落在等效边界内"
     —— 大样本下"不显著"可能只是功效不足，等效性检验正面支持"可视为无差异"
  C. 置换检验：打乱类别标签，看观测差异在零分布中的位置
  D. 指标敏感性：换用 Moran's I、DSI 在 L-cell（细胞类型）口径下的结果

产出：results/Robustness/
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

R = r"E:\研究工作\agent长非与空间域\lncRNA空间域\results"
OUT = os.path.join(R, "Robustness")
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(0)
N_PERM = 5000
N_BOOT = 2000
EQUIV_FRAC = 0.2          # 等效边界：0.2 × 对照 DSI 的标准差（Cohen 小效应）


def load_spec(path, dsi_col, det_col="panel_detect"):
    s = pd.read_csv(path, index_col=0)
    s = s[~s.index.duplicated(keep="first")]
    return s, dsi_col, det_col


def stratified(s, dsi_col, det_col, min_d=0.01, n_bins=10):
    """按检出率十分位分层，层内比较。"""
    lnc = s[(s["gene_class"] == "lncRNA") & (s[det_col] >= min_d)]
    pc = s[(s["gene_class"] == "protein_coding") & (s[det_col] >= min_d)]
    allg = pd.concat([lnc.assign(cls="lncRNA"), pc.assign(cls="PC")])
    allg = allg[allg[det_col] >= max(min_d, allg[det_col].quantile(0.01))]
    allg["bin"] = pd.qcut(allg[det_col], n_bins, labels=False, duplicates="drop")
    rows = []
    for b, sub in allg.groupby("bin"):
        a = sub.loc[sub["cls"] == "lncRNA", dsi_col].dropna()
        c = sub.loc[sub["cls"] == "PC", dsi_col].dropna()
        if len(a) < 5 or len(c) < 20:
            continue
        u = stats.mannwhitneyu(a, c, alternative="two-sided")
        rows.append({"bin": int(b), "n_lnc": len(a), "n_pc": len(c),
                     "detect_range": f"{sub[det_col].min():.3f}-{sub[det_col].max():.3f}",
                     "lnc_median": a.median(), "pc_median": c.median(),
                     "delta": a.median() - c.median(), "p": u.pvalue})
    return pd.DataFrame(rows), lnc, pc


def main():
    all_rows = []
    details = {}

    for name, path, dsi_col, det_col, extra in [
        ("MERFISH ISO", R + r"\H1H2\gene_specificity.csv", "DSI_domain", "panel_detect", None),
        ("MERFISH WT", R + r"\H1H2_WT\gene_specificity.csv", "DSI_domain", "panel_detect", None),
        ("Visium V1", R + r"\Visium\gene_specificity.csv", "DSI", "detect_frac", "visium"),
    ]:
        s, dsi_col, det_col = load_spec(path, dsi_col, det_col)
        if extra == "visium":
            pf = pd.read_csv(r"E:\研究工作\agent长非与空间域\analysis\03_lncrna\panel_features_class.csv",
                             index_col=0)
            pf = pf[~pf.index.duplicated(keep="first")]
            coords = pd.read_csv(R + r"\gene_coords_m25.tsv", sep="\t")
            n2t = coords.drop_duplicates("gene_name").set_index("gene_name")["gene_type"]
            LNC = {"lncRNA", "lincRNA", "antisense", "processed_transcript",
                   "sense_intronic", "sense_overlapping", "macro_lncRNA"}
            fb = n2t.reindex(s.index).map(lambda t: "lncRNA" if isinstance(t, str) and t in LNC
                                          else ("protein_coding" if t == "protein_coding" else "other"))
            s["gene_class"] = pf["gene_class"].reindex(s.index).fillna(fb)

        print(f"\n{'='*80}\n=== {name} ===")
        print("[A] 分层匹配（检出率十分位）")
        st, lnc, pc = stratified(s, dsi_col, det_col)
        st["dataset"] = name
        all_rows.append(st)
        print(st.round(5).to_string(index=False))
        n_sig = int((st["p"] < 0.05).sum())
        print(f"  10 层中差异显著(p<0.05)的层数: {n_sig}/{len(st)}；"
              f"delta 中位 {st['delta'].median():+.5f}，为正的层数 {int((st['delta']>0).sum())}/{len(st)}")

        # [B] 等效性检验
        print("\n[B] 等效性检验（TOST）")
        a = lnc[dsi_col].dropna().to_numpy()
        c = pc[dsi_col].dropna().to_numpy()
        sd_ctrl = c.std()
        bound = EQUIV_FRAC * sd_ctrl
        diff = a.mean() - c.mean()
        se = np.sqrt(a.var(ddof=1) / len(a) + c.var(ddof=1) / len(c))
        ci_lo, ci_hi = diff - 1.96 * se, diff + 1.96 * se
        p_low = stats.norm.sf((diff - (-bound)) / se)     # H0: diff <= -bound
        p_high = stats.norm.cdf((diff - bound) / se)      # H0: diff >= bound
        p_tost = max(p_low, p_high)
        print(f"  均值差 {diff:+.5f} | 95%CI [{ci_lo:+.5f}, {ci_hi:+.5f}]")
        print(f"  等效边界 ±{bound:.5f}（0.2×对照SD={sd_ctrl:.5f}）")
        print(f"  TOST p = {p_tost:.4g} → {'差异可视为等效（无实质差别）' if p_tost < 0.05 else '无法判定等效'}")

        # [C] 置换检验
        print("\n[C] 置换检验（打乱类别标签，比较中位数差）")
        obs = np.median(a) - np.median(c)
        pool = np.concatenate([a, c])
        na = len(a)
        null = np.empty(N_PERM)
        for i in range(N_PERM):
            idx = rng.permutation(len(pool))
            null[i] = np.median(pool[idx[:na]]) - np.median(pool[idx[na:]])
        p_perm = (1 + (np.abs(null) >= abs(obs)).sum()) / (1 + N_PERM)
        print(f"  观测中位数差 {obs:+.5f} | 置换 p = {p_perm:.4g}")
        # bootstrap CI
        boots = np.empty(N_BOOT)
        for i in range(N_BOOT):
            boots[i] = (np.median(rng.choice(a, len(a), replace=True)) -
                        np.median(rng.choice(c, len(c), replace=True)))
        print(f"  bootstrap 95%CI [{np.percentile(boots,2.5):+.5f}, {np.percentile(boots,97.5):+.5f}]")

        details[name] = {"diff_mean": diff, "ci": (ci_lo, ci_hi), "bound": bound,
                         "p_tost": p_tost, "perm_p": p_perm,
                         "boot_ci": (float(np.percentile(boots, 2.5)),
                                     float(np.percentile(boots, 97.5)))}

    pd.concat(all_rows).to_csv(os.path.join(OUT, "stratified_comparison.csv"), index=False)
    pd.DataFrame(details).T.to_csv(os.path.join(OUT, "equivalence_tests.csv"))
    print(f"\n=== 完成，输出: {OUT} ===")


if __name__ == "__main__":
    main()
