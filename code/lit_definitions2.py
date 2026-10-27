# -*- coding: utf-8 -*-
"""从已保存的全文中更宽松地抽取 domain / niche 的定义与用法，并补检基准评测论文。"""
import glob
import json
import os
import re
import subprocess
import sys

LIT = r"E:\研究工作\agent长非与空间域\lncRNA空间域\results\literature"
SCRIPT = r"C:\Users\Administrator\AppData\Roaming\CherryStudio\Data\Agents\.claude\skills\unified-research-finder\scripts\pubmed_search.py"
env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")

# 1) 从已保存全文里找含域/生态位定义的句子
FILES = {
    "38413725_PMC10937399.xml": "BANKSY (Nat Genet 2024)",
    "35365632_PMC8976049.xml": "STAGATE (Nat Commun 2022)",
    "36859400_PMC9977836.xml": "GraphST (Nat Commun 2023)",
    "42555503_PMC13440128.xml": "BIB review 2026",
    "42893560_PMC13677061.xml": "BIB review AI 2026",
}

PAT = re.compile(
    r"([^.]{30,400}?(?:spatial domain|tissue domain|spatial niche|cell niche|domain segmentation|"
    r"spatial region|spatial clustering)[^.]{0,400}\.)", re.I)

for fn, label in FILES.items():
    p = os.path.join(LIT, fn)
    if not os.path.exists(p):
        print(f"[缺失] {fn}")
        continue
    txt = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", open(p, encoding="utf-8").read()))
    sents = PAT.findall(txt)
    # 优先留含定义意味的
    pri = [s for s in sents if re.search(r"\b(define|defined|refer|correspond|regard|represent|"
                                         r"denote|comprise|consist|are clusters|is a)\b", s, re.I)]
    print("=" * 100)
    print(f"[{label}] 命中 {len(sents)} 句，其中定义性 {len(pri)} 句")
    for s in pri[:10]:
        print("  » " + s.strip()[:420])

# 2) 补检基准评测论文
print("\n" + "=" * 100)
q = 'benchmarking spatial domain detection spatial transcriptomics'
p = subprocess.run([sys.executable, SCRIPT, "--query", q, "--retmax", "8"],
                   capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)
try:
    j = json.loads(p.stdout)
    print(f"[基准评测检索] {q} — 命中 {j.get('count')}")
    for a in j.get("articles", []):
        print(f"  - [{a.get('pubdate','')[:7]}] {a.get('title','')}")
        print(f"    {a.get('journal','')} | PMID {a.get('pmid')} | {a.get('doi','')}")
except Exception as e:
    print("检索失败:", e, p.stdout[:200])
