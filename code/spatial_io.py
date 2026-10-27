# -*- coding: utf-8 -*-
"""
内存友好的 h5ad 访问层。

本机仅 8 GB 内存（可用约 3.4 GB），而 cellxgene_WT/X 有 1.1 亿非零元、
cellxgene_ISO/X 有 1.96 亿非零元，直接 sc.read_h5ad 会 OOM。
因此这里一律按 CSR 行块流式读取，峰值内存控制在数百 MB。

约定：h5ad 中的稀疏矩阵存成 anndata 的 h5py Group（data/indices/indptr）。
"""

import h5py
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

DATA_DIR = r"E:\研究工作\agent长非与空间域\data"

# 逻辑名 -> 文件名
FILES = {
    "cellxgene_WT": "cellxgene_WT_labels.h5ad",
    "cellxgene_ISO": "cellxgene_ISO_labels.h5ad",
    "binned_WT": "binned_WT_ccf.h5ad",
    "binned_ISO": "binned_ISO_ccf.h5ad",
}


def _decode(arr):
    """bytes -> str"""
    return [x.decode("utf-8") if isinstance(x, (bytes, np.bytes_)) else x for x in arr]


def _sparse_group(f, layer):
    """layer='X' 或 'layers/xxx'，返回 h5py Group。"""
    g = f[layer]
    if not isinstance(g, h5py.Group):
        raise ValueError(f"{layer} 是稠密数据集的，本工具只处理 CSR 稀疏存储")
    enc = g.attrs.get("encoding-type", b"")
    enc = enc.decode() if isinstance(enc, bytes) else enc
    if enc not in ("csr_matrix", "csc_matrix"):
        raise ValueError(f"{layer} 的 encoding-type={enc}，不是 CSR")
    return g


def shape_of(path_or_key, layer="X"):
    """返回 (n_obs, n_var)，不读数据。"""
    path = _resolve(path_or_key)
    with h5py.File(path, "r") as f:
        g = _sparse_group(f, layer)
        nnz = g["data"].shape[0]
        n_obs = g["indptr"].shape[0] - 1
        # n_var 从 var 组取（CSC 时 indptr 行数不同）
        n_var = f["var/_index"].shape[0]
        return n_obs, n_var, nnz


def _resolve(path_or_key):
    if path_or_key in FILES:
        return DATA_DIR + "\\" + FILES[path_or_key]
    return path_or_key


def read_obs(path_or_key, columns=None):
    """读取 obs 为 DataFrame（含分类列解码）。"""
    path = _resolve(path_or_key)
    with h5py.File(path, "r") as f:
        og = f["obs"]
        index = _decode(og["_index"][:])
        data = {"_index": index}
        for k in og.keys():
            if k == "_index":
                continue
            if columns is not None and k not in columns:
                continue
            item = og[k]
            if isinstance(item, h5py.Group):  # categorical
                cats = _decode(item["categories"][:])
                codes = item["codes"][:]
                vals = np.array(cats, dtype=object)[np.where(codes >= 0, codes, 0)]
                vals[codes < 0] = np.nan
                data[k] = pd.Categorical(vals)
            else:
                data[k] = item[:]
    df = pd.DataFrame(data).set_index("_index")
    return df


def read_var(path_or_key):
    """读取 var 为 DataFrame。"""
    path = _resolve(path_or_key)
    with h5py.File(path, "r") as f:
        vg = f["var"]
        index = _decode(vg["_index"][:])
        data = {"_index": index}
        for k in vg.keys():
            if k == "_index":
                continue
            item = vg[k]
            if isinstance(item, h5py.Group):
                cats = _decode(item["categories"][:])
                codes = item["codes"][:]
                vals = np.array(cats, dtype=object)[np.where(codes >= 0, codes, 0)]
                vals[codes < 0] = np.nan
                data[k] = pd.Categorical(vals)
            else:
                data[k] = item[:]
    return pd.DataFrame(data).set_index("_index")


def read_obsm(path_or_key, key="spatial"):
    path = _resolve(path_or_key)
    with h5py.File(path, "r") as f:
        if "obsm" not in f or key not in f["obsm"]:
            raise KeyError(f"{path_or_key} 没有 obsm/{key}")
        return f["obsm"][key][:]


def iter_csr_blocks(path_or_key, layer="X", block_rows=20000, as_float32=True):
    """
    按行块流式产出 (start, end, csr_block)。

    每次只把 data/indices 的一个切片读进内存。
    """
    path = _resolve(path_or_key)
    with h5py.File(path, "r") as f:
        g = _sparse_group(f, layer)
        indptr = g["indptr"][:]
        n_obs = indptr.shape[0] - 1
        dset = g["data"]
        iset = g["indices"]
        n_var = f["var/_index"].shape[0]
        for start in range(0, n_obs, block_rows):
            end = min(start + block_rows, n_obs)
            lo, hi = int(indptr[start]), int(indptr[end])
            if hi == lo:
                yield start, end, csr_matrix((end - start, n_var), dtype=np.float32)
                continue
            data = dset[lo:hi]
            if as_float32 and data.dtype != np.float32:
                data = data.astype(np.float32)
            indices = iset[lo:hi]
            ind = indptr[start:end + 1] - lo
            yield start, end, csr_matrix((data, indices, ind), shape=(end - start, n_var))


def read_rows(path_or_key, start, end, layer="X", as_float32=True):
    """读取指定行区间 [start, end) 的 CSR 子矩阵。"""
    path = _resolve(path_or_key)
    with h5py.File(path, "r") as f:
        g = _sparse_group(f, layer)
        indptr = g["indptr"][start:end + 1]
        lo, hi = int(indptr[0]), int(indptr[-1])
        data = g["data"][lo:hi]
        if as_float32 and data.dtype != np.float32:
            data = data.astype(np.float32)
        indices = g["indices"][lo:hi]
        n_var = f["var/_index"].shape[0]
        ind = indptr - lo
        return csr_matrix((data, indices, ind), shape=(end - start, n_var))


def cell_metrics(path_or_key, layer="X", block_rows=20000, verbose=False):
    """
    流式计算每细胞 QC 指标。

    返回 DataFrame: total_counts / n_detected / max_count / top_gene_frac
    - n_detected: 非零（>0）特征数
    - top_gene_frac: 该细胞最丰富特征占总量比例（衡量单一基因主导程度）
    """
    rows = []
    for start, end, blk in iter_csr_blocks(path_or_key, layer=layer, block_rows=block_rows):
        n = blk.shape[0]
        total = np.asarray(blk.sum(axis=1)).ravel().astype(np.float64)
        if blk.nnz:
            maxv = np.asarray(blk.max(axis=1).toarray()).ravel()
        else:
            maxv = np.zeros(n, dtype=np.float32)
        # CSR 每行非零个数，减去其中恰好为 0 的存储项
        row_idx = np.repeat(np.arange(n), np.diff(blk.indptr))
        zero_per_row = np.bincount(row_idx[blk.data == 0], minlength=n)
        n_det = np.diff(blk.indptr) - zero_per_row
        top_frac = np.divide(maxv, total, out=np.zeros(n), where=total > 0)
        rows.append(pd.DataFrame({
            "total_counts": total,
            "n_detected": n_det,
            "max_count": maxv,
            "top_gene_frac": top_frac,
        }, index=np.arange(start, end)))
        if verbose:
            print(f"  cells {start}-{end}", flush=True)
    out = pd.concat(rows)
    out.index.name = "cell_idx"
    return out


def feature_metrics(path_or_key, layer="X", block_rows=20000, verbose=False):
    """
    流式计算每特征指标。

    返回 DataFrame: n_cells_detected (检出该特征的细胞数) / total_counts / mean_positive
    """
    path = _resolve(path_or_key)
    with h5py.File(path, "r") as f:
        n_var = f["var/_index"].shape[0]
    names = read_var(path_or_key).index.to_numpy()

    n_cells = np.zeros(n_var, dtype=np.int64)
    total = np.zeros(n_var, dtype=np.float64)
    for start, end, blk in iter_csr_blocks(path, layer=layer, block_rows=block_rows):
        data = blk.data.astype(np.float64)
        cols = blk.indices
        n_cells += np.bincount(cols, minlength=n_var)
        total += np.bincount(cols, weights=data, minlength=n_var)
        if verbose:
            print(f"  cells {start}-{end}", flush=True)

    with h5py.File(path, "r") as f:
        n_obs = f["obs/_index"].shape[0]
    mean_pos = np.divide(total, n_cells, out=np.zeros(n_var), where=n_cells > 0)
    out = pd.DataFrame({
        "n_cells_detected": n_cells,
        "detect_frac": n_cells / n_obs,
        "total_counts": total,
        "mean_positive": mean_pos,
    }, index=names)
    out.index.name = "feature"
    return out
