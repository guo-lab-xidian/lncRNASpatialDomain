# -*- coding: utf-8 -*-
"""
检验"类别 × 检出率"的交互效应：
  低表达时 lncRNA 的域特异性低于 mRNA，高表达时高于 mRNA。

模型：DSI ~ gene_class * log10(检出率) + 样本（数据集）
在每个数据集内单独拟合，并做似然比检验比较有/无交互项。

产出：results/Robustness/interaction_test.csv
"""
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

R = r"E:\研究工作\agent长非与空间域\lncRNA空间域\results"
OUT = os.path.join(R, "Robustness")
os.makedirs(OUT, exist_ok=True)


def load(name):
    if name == "MERFISH ISO":
        s = pd.read_csv(R + r"\H1H2\gene_specificity.csv", index_col=0)
        s = s[~s.index.duplicated(keep="first")]
        return s.rename(columns={"DSI_domain": "DSI", "panel_detect": "detect"})[
            ["gene_class", "detect", "DSI", "DSI_cell"]], "DSI"
    if name == "MERFISH WT":
        s = pd.read_csv(R + r"\H1H2_WT\gene_specificity.csv", index_col=0)
        s = s[~s.index.duplicated(keep="first")]
        return s.rename(columns={"DSI_domain": "DSI", "panel_detect": "detect"})[
            ["gene_class", "detect", "DSI", "DSI_cell"]], "DSI"
    s = pd.read_csv(R + r"\Visium\gene_specificity.csv", index_col=0)
    s = s[~s.index.duplicated(keep="first")]
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
    s = s.rename(columns={"detect_frac": "detect"})
    return s[["gene_class", "detect", "DSI", "peak_domain"]], "DSI"


rows = []
for name in ["MERFISH ISO", "MERFISH WT", "Visium V1"]:
    s, _ = load(name)
    d = s[s["gene_class"].isin(["lncRNA", "protein_coding"])].dropna(subset=["detect", "DSI"])
    d = d[d["detect"] >= 0.01].copy()
    d["is_lnc"] = (d["gene_class"] == "lncRNA").astype(float)
    d["logdet"] = np.log10(d["detect"])
    d["logdet_c"] = d["logdet"] - d["logdet"].mean()
    d["inter"] = d["is_lnc"] * d["logdet_c"]

    y = d["DSI"].to_numpy()
    X0 = sm.add_constant(d[["is_lnc", "logdet_c"]].to_numpy())
    X1 = sm.add_constant(d[["is_lnc", "logdet_c", "inter"]].to_numpy())
    m0 = sm.OLS(y, X0).fit()
    m1 = sm.OLS(y, X1).fit()
    lr = 2 * (m1.llf - m0.llf)
    p_inter = stats.chi2.sf(lr, 1)

    b = m1.params
    ci = m1.conf_int()
    print(f"\n{'='*70}\n=== {name} ===  (n={len(d):,}, lncRNA {int(d.is_lnc.sum()):,})")
    print(f"  截距 {b[0]:+.5f}")
    print(f"  类别主效应(lncRNA) {b[1]:+.5f}  95%CI [{ci[1][0]:+.5f}, {ci[1][1]:+.5f}]  p={m1.pvalues[1]:.3g}")
    print(f"  检出率斜率        {b[2]:+.5f}  95%CI [{ci[2][0]:+.5f}, {ci[2][1]:+.5f}]  p={m1.pvalues[2]:.3g}")
    print(f"  ★ 交互项          {b[3]:+.5f}  95%CI [{ci[3][0]:+.5f}, {ci[3][1]:+.5f}]  p={m1.pvalues[3]:.3g}")
    print(f"  似然比检验 p = {p_inter:.3g}")
    print(f"  解释：检出率每升高 10 倍，lncRNA 相对 mRNA 的 DSI 变化 {b[3]:+.5f}")

    rows.append({"dataset": name, "n_genes": len(d), "n_lncRNA": int(d.is_lnc.sum()),
                 "main_effect_lnc": b[1], "main_p": m1.pvalues[1],
                 "logdet_slope": b[2], "logdet_p": m1.pvalues[2],
                 "interaction": b[3], "interaction_ci_lo": ci[3][0],
                 "interaction_ci_hi": ci[3][1], "interaction_p": m1.pvalues[3],
                 "lrt_p": p_inter, "r2": m1.rsquared})

pd.DataFrame(rows).to_csv(os.path.join(OUT, "interaction_test.csv"), index=False)
print(f"\n=== 完成，输出: {OUT}/interaction_test.csv ===")
