# -*- coding: utf-8 -*-
"""IO 层冒烟测试：取两个行块，核对 sum/max/检出数，并记录内存峰值。"""
import itertools
import os
import sys
import tracemalloc

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import spatial_io as sio

KEY = "cellxgene_WT"
tracemalloc.start()
for start, end, blk in itertools.islice(sio.iter_csr_blocks(KEY, "X", block_rows=5000), 2):
    cur, peak = tracemalloc.get_traced_memory()
    print(f"block {start}-{end}: shape={blk.shape} nnz={blk.nnz} dtype={blk.dtype} "
          f"当前内存={cur/1e6:.0f}MB 峰值={peak/1e6:.0f}MB")
    total = np.asarray(blk.sum(axis=1)).ravel()
    mx = np.asarray(blk.max(axis=1).toarray()).ravel()
    nd = np.diff(blk.indptr)
    print(f"  total: min={total.min():.0f} med={np.median(total):.0f} max={total.max():.0f}")
    print(f"  max  : min={mx.min():.0f} med={np.median(mx):.0f} max={mx.max():.0f}")
    print(f"  n_det: min={nd.min()} med={np.median(nd):.0f} max={nd.max()}")
    print(f"  数据是否整数: {np.allclose(blk.data[:1000], np.round(blk.data[:1000]))}")

# 核对一个整行：直接取第 0 行原始数据比对
import h5py
with h5py.File(sio._resolve(KEY), "r") as f:
    ip = f["X/indptr"][:3]
    d = f["X/data"][ip[0]:ip[1]]
    i = f["X/indices"][ip[0]:ip[1]]
print(f"\n第0行直接读取: nnz={len(d)} sum={d.sum():.0f} max={d.max():.0f}")
print("特征名前5(该行):", sio.read_var(KEY).iloc[i[:5]].index.tolist())
