# -*- coding: utf-8 -*-
"""
阶段 4：lncRNA 在 ISO 面板中的表达与空间特征（为"空间域中 lncRNA 特异表达"做可行性探针）。

依赖：
  analysis/02_annotation/panel_features_final.csv   (阶段 3 最终注释)
  analysis/04_explore/embedding_ISO.csv             (阶段 5 的细胞子集 + Leiden 标签)

产出（analysis/03_lncrna/）：
  lncrna_detect_by_celltype.csv   lncRNA × 细胞类型检出率
  lncrna_specificity.csv          跨 Leiden 簇的特异性评分（选出候选）
  lncrna_moransI.csv              候选 lncRNA 的莫兰指数（空间自相关）
  fig_lncrna_by_celltype.png
  fig_lncrna_spatial.png
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import spatial_io as sio

ANN = r"E:\研究工作\agent长非与空间域\analysis\02_annotation"
EXP = r"E:\研究工作\agent长非与空间域\analysis\04_explore"
OUT = r"E:\研究工作\agent长非与空间域\analysis\03_lncrna"
os.makedirs(OUT, exist_ok=True)

KEY = "cellxgene_ISO"
PANEL = "ISO"
MIN_DETECT = 0.05      # 候选 lncRNA 的最低总检出率
N_CAND = 100           # 参与后续分析的候选数


def load_subset(emb, n_per_sample=8000):
    """
    复现 04_explore.py 的子集抽取规则（每样本取中段 n_per_sample 个连续细胞），
    行顺序与 embedding_{panel}.csv 一致。
    （embedding 里的 obs_names 经过 make_unique，不能反查原始行号，故重算。）
    """
    from scipy.sparse import vstack
    obs = sio.read_obs(KEY)
    samples = list(obs["sample"].cat.categories) if hasattr(obs["sample"], "cat") \
        else sorted(obs["sample"].unique())
    blocks, obs_parts = [], []
    for s in samples:
        idx = np.where(obs["sample"].astype(str).to_numpy() == str(s))[0]
        lo, n = idx.min(), len(idx)
        take = min(n_per_sample, n)
        start = lo + (n - take) // 2
        blocks.append(sio.read_rows(KEY, start, start + take))
        obs_parts.append(obs.iloc[start:start + take])
    X = vstack(blocks, format="csr").tocsr()
    o = pd.concat(obs_parts)
    if len(o) != len(emb):
        raise ValueError(f"子集细胞数 {len(o)} 与 embedding {len(emb)} 不一致")
    o.index = emb.index
    print(f"  复现子集: {o.shape[0]:,} 细胞 × {X.shape[1]:,} 特征", flush=True)
    return X, o


def morans_i(X_dense, coords, k=8):
    """简易 kNN 空间权重下的 Moran's I（对每个特征向量化计算）。"""
    from sklearn.neighbors import NearestNeighbors
    n = len(coords)
    nn = NearestNeighbors(n_neighbors=min(k + 1, n)).fit(coords)
    _, ind = nn.kneighbors(coords)
    ind = ind[:, 1:]
    row = np.repeat(np.arange(n), ind.shape[1])
    col = ind.ravel()
    W = np.zeros((n, n), dtype=np.float32)
    W[row, col] = 1.0
    W = (W + W.T) / 2
    deg = W.sum(axis=1)
    Xc = X_dense - X_dense.mean(axis=0, keepdims=True)
    num = np.einsum("ij,ij->j", Xc, W @ Xc)
    den = np.einsum("ij,ij->j", Xc, Xc)
    S0 = W.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        I = (n / S0) * np.divide(num, den, out=np.zeros_like(num), where=den > 0)
    return I


def main():
    # 优先用 06 生成的宽口径分类表（lncRNA + lincRNA 等亚类已合并）
    cls = os.path.join(OUT, "panel_features_class.csv")
    final = os.path.join(ANN, "panel_features_final.csv")
    if os.path.exists(cls):
        src, col = cls, "gene_class"
    elif os.path.exists(final):
        src, col = final, "gene_biotype"
    else:
        src, col = os.path.join(ANN, "panel_features.csv"), "gene_biotype"
    print("使用注释表:", os.path.basename(src), f"(过滤列={col})", flush=True)
    pf = pd.read_csv(src, index_col=0)
    iso = pf[pf["panel"] == PANEL]
    lnc = iso[(iso["feature_type"] == "gene") & (iso[col] == "lncRNA")]
    print(f"ISO 面板 lncRNA 基因: {len(lnc):,}")

    lnc = lnc.sort_values("detect_frac", ascending=False)
    cand = lnc[lnc["detect_frac"] >= MIN_DETECT].head(N_CAND)
    print(f"检出率 >= {MIN_DETECT} 的 lncRNA: {int((lnc['detect_frac'] >= MIN_DETECT).sum()):,}，"
          f"取前 {len(cand)} 个作候选")
    if len(cand) == 0:
        print("没有候选 lncRNA，跳过后续分析")
        return
    cand.to_csv(os.path.join(OUT, "lncrna_candidates.csv"))
    print("\n候选 lncRNA（前 15）:")
    print(cand[["gene_name", "detect_frac", "n_cells_detected", "mean_positive"]].head(15).to_string())

    emb = pd.read_csv(os.path.join(EXP, "embedding_ISO.csv"), index_col=0)
    print(f"\n读取子集表达 ...", flush=True)
    X, obs_sub = load_subset(emb)
    obs_sub = obs_sub.loc[emb.index]
    # 校验对齐：重算子集的 cell_label 应与 embedding 完全一致
    same = (obs_sub["cell_label"].astype(str).to_numpy() == emb["cell_label"].astype(str).to_numpy()).all()
    print(f"  子集与 embedding 对齐校验: {'通过' if same else '不一致！'}", flush=True)
    var = sio.read_var(KEY)

    feat_names = cand.index.to_numpy()
    col_idx = var.index.get_indexer(feat_names)
    Xl = X[:, col_idx]                      # 子集 × 候选 lncRNA
    detected = np.asarray((Xl > 0).toarray())

    # ---- 细胞类型 x lncRNA 检出率 ----
    cl = obs_sub["cell_label"].astype(str).to_numpy()
    rows = []
    for ct in pd.unique(cl):
        m = cl == ct
        if m.sum() < 50:
            continue
        rows.append(pd.Series(detected[m].mean(axis=0), index=feat_names, name=ct))
    by_ct = pd.DataFrame(rows).T
    by_ct.to_csv(os.path.join(OUT, "lncrna_detect_by_celltype.csv"))
    print(f"\n细胞类型 × lncRNA 检出率矩阵: {by_ct.shape}")

    # ---- 特异性评分：跨 Leiden 簇的变异系数 / max-min ----
    lei = emb["leiden"].astype(str).to_numpy()
    rows = []
    for clu in pd.unique(lei):
        m = lei == clu
        rows.append(pd.Series(detected[m].mean(axis=0), index=feat_names, name=clu))
    by_clu = pd.DataFrame(rows).T
    by_clu.to_csv(os.path.join(OUT, "lncrna_detect_by_leiden.csv"))

    spec = pd.DataFrame({
        "gene_name": cand["gene_name"].to_numpy(),
        "overall_detect": cand["detect_frac"].to_numpy(),
        "max_cluster_frac": by_clu.max(axis=1),
        "argmax_cluster": by_clu.idxmax(axis=1),
        "min_cluster_frac": by_clu.min(axis=1),
        "frac_range": by_clu.max(axis=1) - by_clu.min(axis=1),
        "cv": by_clu.std(axis=1) / by_clu.mean(axis=1).replace(0, np.nan),
    }, index=feat_names)
    spec = spec.sort_values("frac_range", ascending=False)
    spec.to_csv(os.path.join(OUT, "lncrna_specificity.csv"))
    print("\n跨空间/细胞簇差异最大的 12 个 lncRNA:")
    print(spec.head(12).to_string())

    # ---- Moran's I（在每个 sample 内单独算）----
    samples = obs_sub["sample"].astype(str).to_numpy()
    mi_rows = {}
    for s in pd.unique(samples):
        m = samples == s
        if m.sum() < 200:
            continue
        coords = obs_sub.loc[m, ["global_x", "global_y"]].to_numpy()
        Xd = Xl[m].toarray().astype(np.float32)
        keep = Xd.sum(axis=0) > 0
        I = np.full(len(feat_names), np.nan)
        I[keep] = morans_i(Xd[:, keep], coords)
        mi_rows[s] = I
        print(f"  Moran's I 计算完成 sample {s} (n={m.sum():,})", flush=True)
    mi = pd.DataFrame(mi_rows, index=feat_names)
    mi["mean_I"] = mi.mean(axis=1)
    mi["gene_name"] = cand["gene_name"]
    mi = mi.sort_values("mean_I", ascending=False)
    mi.to_csv(os.path.join(OUT, "lncrna_moransI.csv"))
    print("\n空间自相关最强的 12 个 lncRNA（Mean Moran's I）:")
    print(mi.head(12).to_string())

    # ---- 图：热图 + 空间散点 ----
    top_ct = by_ct.mean(axis=1).sort_values(ascending=False).head(18).index
    show = by_ct.loc[top_ct]
    fig, ax = plt.subplots(figsize=(max(8, 0.32 * show.shape[1]), 0.36 * show.shape[0] + 2))
    im = ax.imshow(show.to_numpy(), aspect="auto", cmap="magma", vmin=0)
    ax.set_yticks(range(len(show)))
    ax.set_yticklabels([cand.loc[i, "gene_name"] for i in show.index], fontsize=7)
    ax.set_xticks(range(show.shape[1]))
    ax.set_xticklabels(show.columns, rotation=90, fontsize=7)
    ax.set_title("ISO panel: lncRNA detection fraction by cell type")
    fig.colorbar(im, ax=ax, label="detection fraction")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_lncrna_by_celltype.png"), dpi=150)
    plt.close(fig)

    # 空间图：取特异性最高的 6 个，画在一个 sample 上
    s0 = pd.Series(samples).value_counts().index[0]
    m0 = samples == s0
    top6 = spec.head(6).index
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.6))
    for k, f in enumerate(top6):
        ax = axes[k // 3][k % 3]
        j = list(feat_names).index(f)
        vals = Xl[m0][:, j].toarray().ravel()
        keep = vals > 0
        ax.scatter(obs_sub.loc[m0, "global_x"], obs_sub.loc[m0, "global_y"], s=0.2, c="lightgrey",
                   linewidths=0, rasterized=True)
        sc_ = ax.scatter(obs_sub.loc[m0, "global_x"][keep], obs_sub.loc[m0, "global_y"][keep],
                         s=1.0, c=vals[keep], cmap="viridis", linewidths=0, rasterized=True)
        gname = cand.loc[f, "gene_name"]
        ax.set_title(f"{gname} (range={spec.loc[f,'frac_range']:.2f})", fontsize=10)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(sc_, ax=ax, fraction=0.04, pad=0.01)
    fig.suptitle(f"Spatial expression of top cluster-specific lncRNAs (sample {s0})")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_lncrna_spatial.png"), dpi=150)
    plt.close(fig)

    print(f"\n=== 阶段 4 完成，输出目录: {OUT} ===")


if __name__ == "__main__":
    main()
