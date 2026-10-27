# -*- coding: utf-8 -*-
"""
加载 10x Visium（V1 小鼠脑两切片），整理成统一缓存供后续分析。

Visium 是**全转录组**（32,285 个特征，无需预设探针），
正好用来检验"MERFISH 面板选择偏差"假说。

产出（data/visium/cache/）：
  visium_counts.npz   稀疏计数矩阵（spots × genes）
  visium_meta.parquet 每个 spot 的样本、组织内标记、像素坐标
  visium_genes.txt    基因名
"""
import os
import numpy as np
import pandas as pd
import h5py
from scipy.sparse import csr_matrix

DIR = r"E:\研究工作\agent长非与空间域\data\visium"
OUT = os.path.join(DIR, "cache")
os.makedirs(OUT, exist_ok=True)

SAMPLES = [("V1_Ant", "V1_Ant.h5", "spatial_ant"),
           ("V1_Post", "V1_Post.h5", "spatial_post")]


def decode(arr):
    return [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in arr]


def load_h5(path):
    with h5py.File(path, "r") as f:
        m = f["matrix"]
        data = m["data"][:]
        indices = m["indices"][:]
        indptr = m["indptr"][:]
        shape = tuple(m["shape"][:])
        feats = f["matrix/features"]
        names = decode(feats["name"][:])
        ftypes = decode(feats["feature_type"][:])
        barcodes = decode(m["barcodes"][:])
    # 10x h5 里 indptr 长度 = n_barcodes+1，是 CSC（列=barcode）
    from scipy.sparse import csc_matrix
    X = csc_matrix((data, indices, indptr), shape=shape)   # genes × spots
    return X.T.tocsr(), names, ftypes, barcodes


def load_positions(spatial_dir):
    p = os.path.join(spatial_dir, "tissue_positions_list.csv")
    if not os.path.exists(p):
        p = os.path.join(spatial_dir, "tissue_positions.csv")
    pos = pd.read_csv(p)
    if pos.columns[0].startswith("Unnamed") or "barcode" not in pos.columns:
        pos = pd.read_csv(p, header=None,
                          names=["barcode", "in_tissue", "array_row", "array_col",
                                 "pxl_row_in_fullres", "pxl_col_in_fullres"])
    else:
        pos = pos.rename(columns={"pxl_col_in_fullres": "pxl_col_in_fullres",
                                  "pxl_row_in_fullres": "pxl_row_in_fullres"})
    return pos.set_index("barcode")


def main():
    mats, metas, gene_names = [], [], None
    for name, h5, sp in SAMPLES:
        X, names, ftypes, barcodes = load_h5(os.path.join(DIR, h5))
        pos = load_positions(os.path.join(DIR, sp))
        keep = np.array([t == "Gene Expression" for t in ftypes])
        if gene_names is None:
            gene_names = [n for n, k in zip(names, keep) if k]
            print(f"基因特征: {keep.sum():,} / {len(names):,}")
        else:
            assert gene_names == [n for n, k in zip(names, keep) if k], "两切片基因不一致"
        Xg = X[:, keep]
        # 只保留组织内 spot
        in_tissue = pos.reindex(barcodes)["in_tissue"].fillna(0).astype(int).to_numpy()
        print(f"{name}: {Xg.shape[0]:,} spots（组织内 {in_tissue.sum():,}）")
        idx = np.where(in_tissue == 1)[0]
        Xg = Xg[idx]
        meta = pos.reindex([barcodes[i] for i in idx]).copy()
        meta["sample"] = name
        meta["barcode"] = meta.index
        mats.append(Xg)
        metas.append(meta)

    from scipy.sparse import vstack
    Xall = vstack(mats, format="csr")
    meta = pd.concat(metas)
    print(f"\n合并: {Xall.shape[0]:,} spots × {Xall.shape[1]:,} genes | nnz={Xall.nnz:,}")
    print(f"每 spot 检出基因数中位: {np.median(np.diff(Xall.indptr)):.0f}")

    import scipy.sparse as sp
    sp.save_npz(os.path.join(OUT, "visium_counts.npz"), Xall)
    meta.to_parquet(os.path.join(OUT, "visium_meta.parquet"))
    pd.Series(gene_names).to_csv(os.path.join(OUT, "visium_genes.txt"),
                                 index=False, header=False)
    print(f"\n=== 缓存完成: {OUT} ===")


if __name__ == "__main__":
    main()
