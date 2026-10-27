# -*- coding: utf-8 -*-
"""
16 个跨面板可重复的高空间结构 lncRNA 的功能注释。

分析内容：
  1. 空间/细胞类型标记：在 L-domain 与 L-cell 上的峰值与特异性
  2. 基因组邻域：±100 kb / ±1 Mb 内的蛋白编码基因
  3. 共表达模块：与全部蛋白编码基因的检出相关性，取 Top 200 做 GO 富集
  4. 文献记录：PubMed 中该基因名的命中数（判断是否为已研究基因）

产出：lncRNA空间域/results/annotation/
"""
import json
import os
import sys
import time
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd
from scipy.sparse import vstack, csr_matrix
import h5py

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import spatial_io as sio

KEY = "cellxgene_ISO"
RES = r"E:\研究工作\agent长非与空间域\lncRNA空间域\results"
LDIR = os.path.join(RES, "Ldomain")
CLS = r"E:\研究工作\agent长非与空间域\analysis\03_lncrna\panel_features_class.csv"
COORD = os.path.join(RES, "gene_coords_m25.tsv")
OUT = os.path.join(RES, "annotation")
os.makedirs(OUT, exist_ok=True)

TOP_COEXPR = 200
BLOCK = 15000

# 跨面板可重复的 16 个 lncRNA（两套面板 Moran's I 均排名前 20）
TARGETS = ["Miat", "Cdr1", "Gm27000", "Gm49698", "Gm48623", "Gm49655",
           "Gm35240", "Gm47221", "Gm42428", "Gm45163", "Gm45895", "Gm47408",
           "Gm49124", "AY512931", "B230220B15Rik", "A830036E02Rik"]


def enrichr(gene_list, library="GO_Biological_Process_2023", timeout=60):
    """调用 Enrichr REST API（小鼠符号需转成人类大写形式）。"""
    genes = [g.upper() if not g.startswith("Gm") else g.upper() for g in gene_list]
    add_url = "https://maayanlab.cloud/Enrichr/addList"
    data = urllib.parse.urlencode({"list": "\n".join(genes),
                                   "description": "lncRNA coexpr"}).encode()
    req = urllib.request.Request(add_url, data=data)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        j = json.loads(r.read().decode())
    uid = j["userListId"]
    q = urllib.parse.urlencode({"userListId": uid, "backgroundType": library})
    with urllib.request.urlopen(
            f"https://maayanlab.cloud/Enrichr/enrich?{q}", timeout=timeout) as r:
        res = json.loads(r.read().decode())
    rows = []
    for term, vals in res.get(library, [])[:15]:
        # vals: [rank, term, p, z, combined, genes, adj_p, ...]
        rows.append({"term": term, "p": vals[2], "adj_p": vals[6],
                     "n_genes": len(vals[5]), "genes": ";".join(vals[5][:10])})
    return pd.DataFrame(rows)


def main():
    pf = pd.read_csv(CLS, index_col=0)
    pf = pf[~pf.index.duplicated(keep="first")]
    spec = pd.read_csv(os.path.join(RES, "H1H2", "gene_specificity.csv"), index_col=0)
    spec = spec[~spec.index.duplicated(keep="first")]
    moran = pd.read_csv(os.path.join(RES, "H1H2", "moransI.csv"), index_col=0)
    moran = moran[~moran.index.duplicated(keep="first")]
    coords = pd.read_csv(COORD, sep="\t").drop_duplicates("gene_id").set_index("gene_id")

    targets = [g for g in TARGETS if g in spec.index]
    print(f"目标 lncRNA: {len(targets)}")

    # ---------- 1. 标记特征 ----------
    rows = []
    for g in targets:
        rows.append({
            "gene_name": g,
            "gene_id": pf.loc[g, "gene_id"],
            "panel_detect": spec.loc[g, "panel_detect"],
            "DSI_domain": spec.loc[g, "DSI_domain"],
            "DSI_cell": spec.loc[g, "DSI_cell"],
            "peak_domain": spec.loc[g, "peak_domain"],
            "peak_celltype": spec.loc[g, "peak_celltype"],
            "moransI": moran.loc[g, "moransI"] if g in moran.index else np.nan,
            "gene_biotype": pf.loc[g, "gene_biotype"],
        })
    tab = pd.DataFrame(rows).set_index("gene_name")

    # ---------- 2. 基因组邻域 ----------
    for win, lab in [(100_000, "nb100kb"), (1_000_000, "nb1mb")]:
        pc = pf[(pf["feature_type"] == "gene") & (pf["gene_class"] == "protein_coding")].copy()
        pc["chrom"] = pc["gene_id"].map(coords["chrom"])
        pc["start"] = pc["gene_id"].map(coords["start"])
        pc["end"] = pc["gene_id"].map(coords["end"])
        pc = pc.dropna(subset=["chrom"])
        by = {c: s for c, s in pc.groupby("chrom")}
        col = []
        for g in targets:
            gid = pf.loc[g, "gene_id"]
            c = coords.loc[gid] if gid in coords.index else None
            if c is None:
                col.append("")
                continue
            s = by.get(c["chrom"])
            if s is None:
                col.append("")
                continue
            m = (s["end"] >= c["start"] - win) & (s["start"] <= c["end"] + win)
            col.append(";".join(sorted(s.loc[m].index)[:10]))
        tab[lab] = col

    # ---------- 3. 共表达 ----------
    lab_df = pd.read_csv(os.path.join(LDIR, "domain_labels.csv"), index_col=0)
    ranges = json.load(open(os.path.join(LDIR, "subset_rows.json")))
    var = sio.read_var(KEY)

    pc_idx = pf[(pf["feature_type"] == "gene") &
                (pf["gene_class"] == "protein_coding")].index
    _det = spec["panel_detect"].reindex(pc_idx)
    pc_genes = _det[_det >= 0.01].index.tolist()
    print(f"蛋白编码候选 {len(pc_genes):,}")

    tpos = var.index.get_indexer(targets)
    cpos = var.index.get_indexer(pc_genes)
    assert (tpos >= 0).all() and (cpos >= 0).all()
    allpos = np.concatenate([tpos, cpos])
    uniq, inv = np.unique(allpos, return_inverse=True)
    ti = inv[:len(tpos)]
    ci = inv[len(tpos):]

    P = np.zeros((len(targets), len(pc_genes)), dtype=np.float64)
    det_t = np.zeros(len(targets))
    det_c = np.zeros(len(pc_genes))
    path = sio._resolve(KEY)
    n_obs = 0
    with h5py.File(path, "r") as f:
        indptr = f["X/indptr"][:]
        dset, iset = f["X/data"], f["X/indices"]
        n_var = f["var/_index"].shape[0]
        for r in ranges:
            for start in range(r["start"], r["stop"], BLOCK):
                end = min(start + BLOCK, r["stop"])
                lo, hi = int(indptr[start]), int(indptr[end])
                blk = csr_matrix(((dset[lo:hi] > 0).astype(np.float32), iset[lo:hi],
                                  indptr[start:end + 1] - lo),
                                 shape=(end - start, n_var))[:, uniq]
                T = blk[:, ti].tocsr()
                C = blk[:, ci].tocsr()
                P += np.asarray((T.T @ C).todense())
                det_t += np.asarray(T.sum(axis=0)).ravel()
                det_c += np.asarray(C.sum(axis=0)).ravel()
                n_obs += end - start
    print(f"共表达累积完成：{n_obs:,} 细胞")

    # 由 P 计算 Phi 相关：cov = Σ x_t x_c − n·p_t·p_c
    pt = det_t / n_obs
    pc_ = det_c / n_obs
    cov = P - n_obs * np.outer(pt, pc_)
    var_t = det_t - n_obs * pt ** 2
    var_c = det_c - n_obs * pc_ ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        r = cov / np.sqrt(np.outer(var_t, var_c))

    coex_rows = []
    for i, g in enumerate(targets):
        rr = pd.Series(r[i], index=pc_genes).dropna().sort_values(ascending=False)
        top = rr.head(TOP_COEXPR)
        coex_rows.append({"gene_name": g,
                          "top1": top.index[0] if len(top) else "",
                          "top1_r": float(top.iloc[0]) if len(top) else np.nan,
                          "top5": ";".join(top.index[:5]),
                          "top200": ";".join(top.index)})
    coex = pd.DataFrame(coex_rows).set_index("gene_name")
    coex.to_csv(os.path.join(OUT, "coexpression_top200.csv"))

    tab["top_coexpressed"] = coex["top5"]
    tab["top1_r"] = coex["top1_r"]

    # ---------- 4. GO 富集 ----------
    go_rows = []
    for g in targets:
        genes = coex.loc[g, "top200"].split(";") if coex.loc[g, "top200"] else []
        if len(genes) < 10:
            continue
        try:
            e = enrichr(genes)
            time.sleep(0.6)
            if len(e):
                top = e.iloc[0]
                go_rows.append({"gene_name": g, "top_term": top["term"],
                                "adj_p": top["adj_p"],
                                "top5_terms": " | ".join(e["term"].head(5))})
            else:
                go_rows.append({"gene_name": g, "top_term": "", "adj_p": np.nan,
                                "top5_terms": ""})
        except Exception as ex:
            print(f"  [{g}] Enrichr 失败: {ex}")
            go_rows.append({"gene_name": g, "top_term": "", "adj_p": np.nan,
                            "top5_terms": ""})
        print(f"  [{g}] GO 完成", flush=True)
    if go_rows:
        go = pd.DataFrame(go_rows).set_index("gene_name")
        go.to_csv(os.path.join(OUT, "go_enrichment.csv"))
        tab = tab.join(go, how="left")

    tab.to_csv(os.path.join(OUT, "lncrna_annotation.csv"))
    print("\n=== 注释汇总 ===")
    show = ["panel_detect", "DSI_domain", "DSI_cell", "peak_celltype", "moransI",
            "top_coexpressed"]
    print(tab[show].round(3).to_string())
    print(f"\n=== 完成，输出: {OUT} ===")


if __name__ == "__main__":
    main()
