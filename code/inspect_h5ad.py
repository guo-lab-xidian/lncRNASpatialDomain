# -*- coding: utf-8 -*-
"""只读方式检查 h5ad / parquet 文件结构，不加载矩阵数据。"""
import sys
import h5py

DATA = r"E:\研究工作\agent长非与空间域\data"
FILES = [
    "cellxgene_WT_labels.h5ad",
    "cellxgene_ISO_labels.h5ad",
    "binned_WT_ccf.h5ad",
    "binned_ISO_ccf.h5ad",
]


def describe_group(g, path="", depth=0, maxdepth=2):
    out = []
    for k in g.keys():
        item = g[k]
        p = f"{path}/{k}"
        if isinstance(item, h5py.Group):
            enc = item.attrs.get("encoding-type", "")
            val = item.attrs.get("encoding-version", "")
            out.append("  " * depth + f"[G] {p}  encoding-type={enc.decode() if isinstance(enc, bytes) else enc}")
            if depth < maxdepth:
                out.extend(describe_group(item, p, depth + 1, maxdepth))
        else:
            shape = item.shape
            dtype = item.dtype
            out.append("  " * depth + f"    {p}  shape={shape} dtype={dtype}")
    return out


for fname in FILES:
    path = DATA + "\\" + fname
    print("=" * 90)
    print("FILE:", fname)
    with h5py.File(path, "r") as f:
        print("  top-level keys:", list(f.keys()))
        # X
        if "X" in f:
            x = f["X"]
            if isinstance(x, h5py.Group):
                enc = x.attrs.get("encoding-type", b"")
                enc = enc.decode() if isinstance(enc, bytes) else enc
                print(f"  X: GROUP encoding-type={enc}")
                for k in x.keys():
                    print(f"      X/{k}: shape={x[k].shape} dtype={x[k].dtype}")
            else:
                print(f"  X: DATASET shape={x.shape} dtype={x.dtype}")
        print("  root attrs:", dict(f.attrs))
        for key in ["obs", "var", "layers", "obsm", "uns", "obsp"]:
            if key in f:
                print(f"  --- {key} ---")
                print("\n".join(describe_group(f[key], "/" + key, 1, maxdepth=2)))
    print()
