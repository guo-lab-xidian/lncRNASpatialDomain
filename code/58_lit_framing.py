# -*- coding: utf-8 -*-
"""立意复核的文献检索：
1) 是否有人明确主张 lncRNA 在**空间**层面（而非细胞类型层面）更特异？
2) 空间统计方法学是否已讨论"表达量/检出率影响空间特异性检验"？
3) 平台 lncRNA 覆盖是否已被讨论？
"""
import json
import os
import subprocess
import sys

SCRIPT = r"C:\Users\Administrator\AppData\Roaming\CherryStudio\Data\Agents\.claude\skills\unified-research-finder\scripts\pubmed_search.py"
env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")

QUERIES = [
    ("是否有人主张 lncRNA 空间层面更特异", "lncRNA spatially variable genes spatial transcriptomics marker class"),
    ("空间统计的表达量依赖", "SPARK spatially expressed genes statistical power expression level"),
    ("方法学综述是否讨论混杂", "spatially variable gene detection benchmark false positive expression"),
    ("平台探针覆盖", "probe panel gene coverage bias targeted spatial transcriptomics noncoding"),
]

for tag, q in QUERIES:
    p = subprocess.run([sys.executable, SCRIPT, "--query", q, "--retmax", "6"],
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace", env=env)
    print("=" * 100)
    print(f"[{tag}]  {q}")
    try:
        j = json.loads(p.stdout)
    except Exception:
        print("  parse error"); continue
    print("  hits:", j.get("count"))
    for a in j.get("articles", []):
        print(f"  - [{a.get('pubdate','')[:7]}] {a.get('title','')[:120]}")
        print(f"      {a.get('journal','')} | PMID {a.get('pmid')}")

