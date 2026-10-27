# -*- coding: utf-8 -*-
"""检查 Allen 结构与本地 annot 字段的可对齐性，并侦察 16 个 lncRNA 的 ISH 数据可得性。"""
import json
import os
import urllib.request
import urllib.parse

import numpy as np
import pandas as pd

RES = r"E:\研究工作\agent长非与空间域\lncRNA空间域\results"
OUT = os.path.join(RES, "AllenISH")
os.makedirs(OUT, exist_ok=True)

TARGETS = ["Miat", "Cdr1", "Gm27000", "Gm49698", "Gm48623", "Gm49655",
           "Gm35240", "Gm47221", "Gm42428", "Gm45163", "Gm45895", "Gm47408",
           "Gm49124", "AY512931", "B230220B15Rik", "A830036E02Rik"]
API = "http://api.brain-map.org/api/v2/data/query.json"


def query(criteria, num_rows="all", timeout=60):
    url = f"{API}?criteria={urllib.parse.quote(criteria)}&num_rows={num_rows}"
    req = urllib.request.Request(url, headers={"User-Agent": "research"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def main():
    print("=== 1. 下载 Allen 结构表 ===")
    j = query("model::Structure,rma::criteria,[ontology_id$eq1]")
    st = pd.DataFrame(j["msg"])
    st = st[["id", "acronym", "name", "depth", "graph_order"]]
    st.to_csv(os.path.join(OUT, "allen_structures.csv"), index=False)
    print(f"  结构数: {len(st)} | id 范围 {st['id'].min()}–{st['id'].max()}")

    print("\n=== 2. 本地 annot 字段分布 ===")
    lab = pd.read_csv(os.path.join(RES, "Ldomain", "domain_labels.csv"), index_col=0)
    print(f"  annot 非空: {lab['annot'].notna().sum():,} / {len(lab):,}")
    vc = lab["annot"].value_counts()
    print(f"  不同取值: {vc.size} | 取值范围 {lab['annot'].min():.0f}–{lab['annot'].max():.0f}")
    print(f"  出现最多的 10 个取值及其细胞数:\n{vc.head(10).to_string()}")
    # 与结构表交集
    ids = set(int(x) for x in lab["annot"].dropna().unique())
    overlap = ids & set(st["id"].astype(int))
    print(f"  与 Allen 结构 id 的交集: {len(overlap)} / {len(ids)}")

    print("\n=== 3. 16 个 lncRNA 的 ISH 数据可得性 ===")
    rows = []
    for g in TARGETS:
        try:
            jj = query(f"model::Gene,rma::criteria,[acronym$eq'{g}']")
            genes = jj.get("msg", [])
            if not genes:
                rows.append({"gene": g, "found": False, "n_datasets": 0})
                print(f"  {g:16} 未收录")
                continue
            gid = genes[0]["id"]
            jd = query(f"model::SectionDataSet,rma::criteria,[gene_id$eq{gid}]")
            msg = jd.get("msg")
            if not isinstance(msg, list):
                msg = []
            n_ds = len(msg)
            n_plane = n_sag = 0
            if n_ds:
                ds = pd.DataFrame(msg)
                if "plane_of_section_id" in ds.columns:
                    n_cor = int((ds["plane_of_section_id"] == 2).sum())
                    n_sag = int((ds["plane_of_section_id"] == 1).sum())
                    n_plane = n_cor
            rows.append({"gene": g, "found": True, "gene_id": gid,
                         "n_datasets": n_ds, "n_coronal": n_plane, "n_sagittal": n_sag})
            print(f"  {g:16} 收录 | 数据集 {n_ds}（冠状 {n_plane}，矢状 {n_sag}）")
        except Exception as e:
            rows.append({"gene": g, "found": None, "error": str(e)})
            print(f"  {g:16} 查询失败: {e}")

    pd.DataFrame(rows).to_csv(os.path.join(OUT, "ish_availability.csv"), index=False)
    print(f"\n=== 完成，输出: {OUT} ===")


if __name__ == "__main__":
    main()
