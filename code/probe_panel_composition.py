# -*- coding: utf-8 -*-
"""快速统计面板构成（不加载表达矩阵）：基因符号 vs 转录本 ID、Rik 类等。"""
import h5py

DATA = r"E:\研究工作\agent长非与空间域\data"


def decode(arr):
    return [x.decode() if isinstance(x, bytes) else str(x) for x in arr]


def panel_stats(names, label):
    ens = [n for n in names if n.startswith("ENSMUS")]
    rik = [n for n in names if n.endswith("Rik")]
    gm = [n for n in names if n.startswith("Gm") and n[2:].isdigit()]
    print(f"  [{label}] n={len(names)}  ENSMUS*={len(ens)}  *Rik={len(rik)}  Gm<num>={len(gm)}")


with h5py.File(DATA + r"\cellxgene_WT_labels.h5ad", "r") as f:
    wt_names = decode(f["var/_index"][:])
with h5py.File(DATA + r"\cellxgene_ISO_labels.h5ad", "r") as f:
    iso_names = decode(f["var/_index"][:])
    iso_cb = decode(f["var/codebook/categories"][:])
    iso_codes = f["var/codebook/codes"][:]

print("=== 面板构成 ===")
panel_stats(wt_names, "WT 22930")
panel_stats(iso_names, "ISO 32684")

print("\n=== ISO 按 codebook 拆分 ===")
for i, c in enumerate(iso_cb):
    sub = [n for n, code in zip(iso_names, iso_codes) if code == i]
    panel_stats(sub, f"{c} n={len(sub)}")

print("\n=== ISO 中基因符号特征是否与 WT 面板一致 ===")
iso_gene = [n for n in iso_names if not n.startswith("ENSMUS")]
print("  ISO 非 ENSMUS 特征数:", len(iso_gene), " WT 特征数:", len(wt_names))
print("  完全相同:", set(iso_gene) == set(wt_names))
print("  仅 WT 有:", len(set(wt_names) - set(iso_gene)), " 仅 ISO 有:", len(set(iso_gene) - set(wt_names)))

print("\n=== ISO 异构体特征示例 ===")
iso_only = [n for n in iso_names if n.startswith("ENSMUS")]
print("  总数:", len(iso_only))
print("  前 10:", iso_only[:10])
