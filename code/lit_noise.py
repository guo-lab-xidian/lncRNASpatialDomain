# -*- coding: utf-8 -*-
"""再检索：单细胞数据中"噪声/检出与表达量相关"的经典文献依据。"""
import json
import os
import re
import subprocess
import sys

SCRIPT = r"C:\Users\Administrator\AppData\Roaming\CherryStudio\Data\Agents\.claude\skills\unified-research-finder\scripts\pubmed_search.py"
env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")

QUERIES = [
    "expression-dependent noise single-cell RNA sequencing measurement error",
    "dropout events single cell RNA-seq zero inflation bias",
    "technical noise depends on gene expression level single cell",
    "lowly expressed genes unreliable single-cell variability estimates",
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
        print("  parse error"); continue
    print("  hits:", j.get("count"))
    for a in j.get("articles", []):
        print(f"  - [{a.get('pubdate','')[:7]}] {a.get('title','')[:125]}")
        print(f"      {a.get('journal','')} | PMID {a.get('pmid')}")
        ab = a.get("abstract") or ""
        for s in re.split(r"(?<=[.!?])\s+", ab):
            if re.search(r"(expression level|abundance|lowly|rare)", s, re.I) and \
               re.search(r"(noise|variance|unreliable|bias|dropout)", s, re.I):
                print("      » " + s[:280])
