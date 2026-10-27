# -*- coding: utf-8 -*-
"""立意复核（二）：用更宽的检索式。"""
import json
import os
import re
import subprocess
import sys

SCRIPT = r"C:\Users\Administrator\AppData\Roaming\CherryStudio\Data\Agents\.claude\skills\unified-research-finder\scripts\pubmed_search.py"
env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")

QUERIES = [
    'lncRNA "spatially variable"',
    'lncRNA spatial transcriptomics marker tissue region',
    '"spatially variable genes" comparison gene classes',
    'expression level confound spatial autocorrelation gene',
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
        print("  parse error:", (p.stdout or "")[:150]); continue
    print("  hits:", j.get("count"))
    for a in j.get("articles", []):
        print(f"  - [{a.get('pubdate','')[:7]}] {a.get('title','')[:120]}")
        print(f"      {a.get('journal','')} | PMID {a.get('pmid')}")
        ab = a.get("abstract") or ""
        for s in re.split(r"(?<=[.!?])\s+", ab):
            if re.search(r"(more|higher|greater).{0,50}(specific|variable|structured)", s, re.I) \
               and re.search(r"lncRNA|non-?coding", s, re.I):
                print("      » " + s[:280])
