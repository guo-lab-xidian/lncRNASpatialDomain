# -*- coding: utf-8 -*-
"""检索：检出率/表达量是否已知会影响空间或细胞类型特异性的估计。"""
import json
import os
import re
import subprocess
import sys

SCRIPT = r"C:\Users\Administrator\AppData\Roaming\CherryStudio\Data\Agents\.claude\skills\unified-research-finder\scripts\pubmed_search.py"
env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")

QUERIES = [
    "spatial variability gene expression level bias statistics",
    "Moran's I dependent on expression level single cell",
    "mean-variance relationship single cell RNA-seq dropout detection",
    "lower expression of lncRNAs compared to protein coding genes",
    "spatial pattern detection power depends on gene expression level",
]

for q in QUERIES:
    p = subprocess.run([sys.executable, SCRIPT, "--query", q, "--retmax", "6"],
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace", env=env)
    print("=" * 100)
    print("QUERY:", q)
    try:
        j = json.loads(p.stdout)
    except Exception:
        print("  parse error")
        continue
    print("  hits:", j.get("count"))
    for a in j.get("articles", []):
        print(f"  - [{a.get('pubdate','')[:7]}] {a.get('title','')[:125]}")
        print(f"      {a.get('journal','')} | PMID {a.get('pmid')}")
        ab = a.get("abstract") or ""
        for s in re.split(r"(?<=[.!?])\s+", ab):
            if re.search(r"(expression level|detection|dropout|variance|power)", s, re.I) and \
               re.search(r"(bias|depend|confound|correlat|affect|increas|decreas)", s, re.I):
                print("      » " + s[:280])
