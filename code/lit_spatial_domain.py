# -*- coding: utf-8 -*-
"""检索"空间域"概念的定义与用法，并抽取摘要中与定义相关的句子。"""
import json
import os
import re
import subprocess
import sys

SCRIPT = r"C:\Users\Administrator\AppData\Roaming\CherryStudio\Data\Agents\.claude\skills\unified-research-finder\scripts\pubmed_search.py"
PY = sys.executable
env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")

QUERIES = [
    ('BayesSpace', '"BayesSpace" spatial transcriptomics'),
    ('SpaGCN', '"SpaGCN" spatial domain'),
    ('STAGATE', '"STAGATE" spatial domains'),
    ('GraphST', '"GraphST" spatial'),
    ('BANKSY', '"BANKSY" spatial domains'),
    ('CellCharter', '"CellCharter" spatial niche'),
    ('review-methods', 'spatial transcriptomics computational methods review spatial domain identification'),
    ('definition', '"spatial domains" definition tissue transcriptomics clustering'),
]

KEY = re.compile(r'[^.]*\b(spatial domain|spatial niche|niche|domain|region|architecture)\b[^.]*\.', re.I)


def run(q, retmax=6):
    p = subprocess.run([PY, SCRIPT, "--query", q, "--retmax", str(retmax)],
                       capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)
    try:
        return json.loads(p.stdout)
    except Exception:
        return {"error": (p.stdout or "")[:300] + " | " + (p.stderr or "")[:300]}


for tag, q in QUERIES:
    print("=" * 110)
    print(f"[{tag}] {q}")
    j = run(q)
    if "error" in j:
        print("  失败:", j["error"]); continue
    print(f"  命中 {j.get('count')} 条")
    for a in j.get("articles", []):
        print(f"\n  ● [{a.get('pubdate','')[:7]}] {a.get('title','')}")
        print(f"    {a.get('journal','')} | PMID {a.get('pmid')} | {a.get('doi','')}")
        ab = a.get("abstract") or ""
        sents = [s.strip() for s in KEY.findall(ab)]
        # 取包含关键定义的句子
        picks = [s for s in re.split(r'(?<=[.!?])\s+', ab)
                 if re.search(r'\b(spatial domain|niche|region|domain)\b', s, re.I)][:4]
        for s in picks:
            print("      » " + s[:400])
