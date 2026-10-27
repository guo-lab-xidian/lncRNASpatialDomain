# -*- coding: utf-8 -*-
"""只读查看 var 名称格式、codebook 取值、raw 组结构。"""
import h5py
import numpy as np

DATA = r"E:\研究工作\agent长非与空间域\data"


def decode(arr):
    return [x.decode() if isinstance(x, bytes) else str(x) for x in arr]


def show_cat(f, path):
    g = f[path]
    cats = decode(g["categories"][:])
    codes = g["codes"][:]
    uniq, cnt = np.unique(codes, return_counts=True)
    print(f"  {path} categories={cats}")
    print(f"  {path} value_counts={dict(zip([cats[i] if i >= 0 else 'NA' for i in uniq], cnt.tolist()))}")


for fname in ["cellxgene_WT_labels.h5ad", "cellxgene_ISO_labels.h5ad"]:
    path = DATA + "\\" + fname
    print("=" * 90)
    print("FILE:", fname)
    with h5py.File(path, "r") as f:
        names = decode(f["var/_index"][:])
        print(f"  n_features={len(names)}")
        print("  前 15:", names[:15])
        print("  后 15:", names[-15:])
        show_cat(f, "var/codebook")
        show_cat(f, "obs/sample")
        show_cat(f, "obs/cell_label")
        # 是否含特殊后缀
        suffixes = {}
        for n in names:
            if "-" in n:
                s = n.rsplit("-", 1)[-1]
                suffixes[s] = suffixes.get(s, 0) + 1
        top = sorted(suffixes.items(), key=lambda x: -x[1])[:15]
        print("  连字符后缀分布 top15:", top)
        # raw 组
        if "raw" in f:
            raw = f["raw"]
            print("  raw keys:", list(raw.keys()))
            for k in raw.keys():
                if k == "X":
                    x = raw[k]
                    if isinstance(x, h5py.Group):
                        enc = x.attrs.get("encoding-type", b"")
                        enc = enc.decode() if isinstance(enc, bytes) else enc
                        print(f"    raw/X GROUP enc={enc}")
                        for kk in x.keys():
                            print(f"       raw/X/{kk} shape={x[kk].shape}")
                    else:
                        print(f"    raw/X DATASET shape={x.shape} dtype={x.dtype}")
                elif isinstance(raw[k], h5py.Group):
                    print(f"    raw/{k} -> {list(raw[k].keys())}")
