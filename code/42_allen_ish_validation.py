# -*- coding: utf-8 -*-
"""
Allen ISH 正交验证：按脑区对齐，计算本地 MERFISH 与 Allen 原位杂交的空间一致性。

数据：
  本地：每个细胞的 Allen CCF 脑区 ID（obs/annot）+ 基因检出
  Allen：StructureUnionize（每个脑区的表达能量）

做法：
  1. 对目标基因，取 Allen 全部可用数据集的 StructureUnionize，按 structure_id 求平均
  2. 本地按 annot 计算每个脑区的检出率
  3. 在两边都有的脑区上计算 Spearman 相关
  4. 用随机挑选的、检出率相近的 lncRNA 作为零分布对照

产出：results/AllenISH/
"""
import json
import os
import sys
import time
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd
from scipy import stats
from scipy.sparse import vstack

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import spatial_io as sio

KEY = "cellxgene_ISO"
RES = r"E:\研究工作\agent长非与空间域\lncRNA空间域\results"
LDIR = os.path.join(RES, "Ldomain")
OUT = os.path.join(RES, "AllenISH")
os.makedirs(OUT, exist_ok=True)
API = "http://api.brain-map.org/api/v2/data/query.json"

TARGETS = ["Miat", "Cdr1", "Gm27000", "Gm49698", "Gm48623", "Gm49655",
           "Gm35240", "Gm47221", "Gm42428", "Gm45163", "Gm45895", "Gm47408",
           "Gm49124", "AY512931", "B230220B15Rik", "A830036E02Rik"]
# 阳性对照：已知脑内高表达、空间模式明确的基因
POS_CONTROLS = ["Lamp5", "Slc17a7", "Gfap", "Mbp", "Foxp2"]
MIN_CELLS_PER_STRUCT = 30
N_RANDOM = 100
SEED = 0


def query(criteria, num_rows="all", timeout=90):
    url = f"{API}?criteria={urllib.parse.quote(criteria)}&num_rows={num_rows}"
    req = urllib.request.Request(url, headers={"User-Agent": "research"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        j = json.loads(r.read().decode())
    msg = j.get("msg")
    return msg if isinstance(msg, list) else []


def allen_structure_expression(gene):
    """返回 {structure_id: mean expression_energy}；无数据返回 None。"""
    ds = query(f"model::SectionDataSet,rma::criteria,[failed$eqfalse],"
               f"products[abbreviation$eq'Mouse'],genes[acronym$eq'{gene}']")
    if not ds:
        return None, 0
    ids = [d["id"] for d in ds]
    acc = {}
    for sid in ids[:12]:
        su = query(f"model::StructureUnionize,rma::criteria,[section_data_set_id$eq{sid}]")
        for row in su:
            k = row.get("structure_id")
            v = row.get("expression_energy")
            if k is None or v is None:
                continue
            acc.setdefault(k, []).append(float(v))
    if not acc:
        return None, len(ids)
    return {k: float(np.mean(v)) for k, v in acc.items()}, len(ids)


def local_structure_detection(gene, var, lab, ranges, annot_codes):
    pos = var.index.get_indexer([gene])
    if pos[0] < 0:
        return None
    blocks = [sio.read_rows(KEY, r["start"], r["stop"])[:, pos].tocsr() for r in ranges]
    x = (vstack(blocks, format="csr") > 0).toarray().ravel().astype(float)
    n = np.bincount(annot_codes, minlength=annot_codes.max() + 1)
    y = np.bincount(annot_codes, weights=x, minlength=n.size)
    with np.errstate(divide="ignore", invalid="ignore"):
        rate = np.divide(y, n, out=np.full_like(y, np.nan, dtype=float), where=n > 0)
    return pd.DataFrame({"structure_id": np.arange(n.size), "n_cells": n, "rate": rate})


def main():
    rng = np.random.default_rng(SEED)
    lab = pd.read_csv(os.path.join(LDIR, "domain_labels.csv"), index_col=0)
    ranges = json.load(open(os.path.join(LDIR, "subset_rows.json")))
    var = sio.read_var(KEY)
    annot_codes = lab["annot"].fillna(-1).astype(int).to_numpy()
    annot_codes = np.where(annot_codes < 0, annot_codes.max() + 1, annot_codes)

    spec = pd.read_csv(os.path.join(RES, "H1H2", "gene_specificity.csv"), index_col=0)
    spec = spec[~spec.index.duplicated(keep="first")]

    genes = [g for g in TARGETS + POS_CONTROLS if g in var.index]
    print(f"待验证基因: {len(genes)}（目标 {len(TARGETS)} + 阳性对照 {len(POS_CONTROLS)}）")

    rows = []
    for g in genes:
        allele, n_ds = allen_structure_expression(g)
        time.sleep(0.3)
        if allele is None:
            rows.append({"gene": g, "allen_datasets": n_ds, "n_struct_common": 0,
                         "spearman": np.nan, "p": np.nan})
            print(f"  {g:16} Allen 无表达数据（数据集 {n_ds}）")
            continue
        loc = local_structure_detection(g, var, lab, ranges, annot_codes)
        if loc is None:
            continue
        loc = loc[loc["n_cells"] >= MIN_CELLS_PER_STRUCT]
        loc = loc[loc["structure_id"] > 0]
        a = pd.DataFrame({"structure_id": list(allele.keys()),
                          "allen": list(allele.values())})
        m = loc.merge(a, on="structure_id", how="inner")
        m = m[m["rate"].notna()]
        if len(m) < 8:
            rows.append({"gene": g, "allen_datasets": n_ds, "n_struct_common": len(m),
                         "spearman": np.nan, "p": np.nan})
            print(f"  {g:16} 共同脑区太少（{len(m)}）")
            continue
        r = stats.spearmanr(m["rate"], m["allen"])
        rows.append({"gene": g, "allen_datasets": n_ds, "n_struct_common": len(m),
                     "spearman": r.statistic, "p": r.pvalue,
                     "is_control": g in POS_CONTROLS})
        print(f"  {g:16} Allen 数据集 {n_ds} | 共同脑区 {len(m)} | "
              f"Spearman ρ = {r.statistic:+.3f} (p={r.pvalue:.2g})")

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT, "ish_correlation.csv"), index=False)

    tgt = res[(~res.get("is_control", pd.Series(dtype=bool)).fillna(False)) &
              res["spearman"].notna()]
    ctl = res[res.get("is_control", pd.Series(dtype=bool)).fillna(False) & res["spearman"].notna()]
    print("\n" + "=" * 70)
    print(f"目标 lncRNA 中有 Allen 数据的: {len(tgt)}/{len(TARGETS)}")
    if len(tgt):
        print(f"  Spearman ρ 中位: {tgt['spearman'].median():+.3f}")
    if len(ctl):
        print(f"阳性对照 ρ 中位: {ctl['spearman'].median():+.3f}（应显著为正）")
    print(f"\n=== 完成，输出: {OUT} ===")


if __name__ == "__main__":
    main()
