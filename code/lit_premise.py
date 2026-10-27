# -*- coding: utf-8 -*-
"""检索：是否有文献明确主张"lncRNA 比蛋白编码基因更具空间/细胞类型特异性"。"""
import json
import os
import re
import subprocess
import sys

SCRIPT = r"C:\Users\Administrator\AppData\Roaming\CherryStudio\Data\Agents\.claude\skills\unified-research-finder\scripts\pubmed_search.py"
env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")

QUERIES = [
    "lncRNA more cell-type specific than protein-coding genes",
    "long non-coding RNA tissue specificity compared mRNA",
    "lncRNA spatial expression marker region specific brain",
    "noncoding RNA cell type specificity single-cell comparison",
]

for q in QUERIES:
    p = subprocess.run([sys.executable, SCRIPT, "--query", q, "--retmax", "8"],
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace", env=env)
    print("=" * 100)
    print("QUERY:", q)
    try:
        j = json.loads(p.stdout)
    except Exception:
        print("  parse error:", (p.stdout or "")[:200])
        continue
    print("  命中:", j.get("count"))
    for a in j.get("articles", []):
        print(f"  - [{a.get('pubdate','')[:7]}] {a.get('title','')[:130]}")
        print(f"      {a.get('journal','')} | PMID {a.get('pmid')}")
        ab = a.get("abstract") or ""
        # 抽取与"特异性比较"相关的句子
        for s in re.split(r"(?<=[.!?])\s+", ab):
            if re.search(r"(more|higher|greater|less).{0,40}(specific|specificity)", s, re.I) \
               and re.search(r"lncRNA|non-?coding", s, re.I):
                print("      » " + s[:300])
