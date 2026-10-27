# -*- coding: utf-8 -*-
"""批量 PubMed 检索并精简输出（标题/期刊/年份/DOI），便于快速做新颖性摸底。"""
import json
import os
import subprocess
import sys

SCRIPT = r"C:\Users\Administrator\AppData\Roaming\CherryStudio\Data\Agents\.claude\skills\unified-research-finder\scripts\pubmed_search.py"
PY = sys.executable

QUERIES = [
    "lncRNA AND (MERFISH OR seqFISH OR \"spatial transcriptomic\")",
    "\"spatial transcriptomics\" AND \"long non-coding RNA\" AND brain",
    "\"isoform\" AND \"spatial transcriptomics\"",
    "lncRNA AND \"spatial domain\"",
    "\"spatial atlas\" AND lncRNA",
    "noncoding RNA AND single-cell spatial transcriptomics AND tissue",
    "lncRNA AND (brain OR cortex) AND spatial",
    "\"spatially resolved transcriptomics\" AND lncRNA",
]

env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")


def run(q, retmax=8):
    p = subprocess.run([PY, SCRIPT, "--query", q, "--retmax", str(retmax)],
                       capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)
    try:
        j = json.loads(p.stdout)
    except Exception:
        return q, None, (p.stdout or "")[:200] + " | ERR:" + (p.stderr or "")[:200]
    return q, j, None


for q in QUERIES:
    qq, j, err = run(q)
    print("=" * 100)
    print("QUERY:", qq)
    if j is None:
        print("  失败:", err)
        continue
    print(f"  命中 {j.get('count')} 条")
    for a in j.get("articles", []):
        title = a.get("title", "").strip().replace("\n", " ")
        print(f"  - [{a.get('pubdate','')[:7]}] {title}")
        print(f"      {a.get('journal','')} | PMID {a.get('pmid')} | {a.get('doi','')}")
