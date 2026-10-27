# -*- coding: utf-8 -*-
"""把分节草稿组装成单一稿件，并做基本的一致性检查。"""
import os
import re

MD = r"E:\研究工作\agent长非与空间域\lncRNA空间域\manuscript"
OUT = os.path.join(MD, "manuscript_full.md")

TITLE = """# What can long non-coding RNAs tell us about spatial domains? A multi-platform reassessment in mouse and human brain

**Running title:** Reassessing lncRNA spatial specificity

*Draft assembled automatically from section files. Figures in `results/figures/`.*
"""

ORDER = ["00_Abstract.md", "02_Introduction.md", "03_Results.md",
         "04_Discussion.md", "01_Methods.md", "08_Backmatter.md"]

parts = [TITLE]
for fn in ORDER:
    p = os.path.join(MD, fn)
    if not os.path.exists(p):
        print(f"缺失: {fn}")
        continue
    txt = open(p, encoding="utf-8").read().strip()
    # Abstract 保留标题；其余节标题原样
    parts.append(txt)
    print(f"并入 {fn} ({len(txt.split())} words)")

body = "\n\n---\n\n".join(parts)
open(OUT, "w", encoding="utf-8").write(body)

# ---- 一致性检查 ----
txt = body
issues = []

nums = {"475": "MERFISH ISO lncRNA 数", "604": "MERFISH WT lncRNA 数",
        "613": "Visium human lncRNA 数",
        "1,090": "Visium mouse lncRNA 数（千分位）",
        "0.879": "跨面板 Moran's I", "93.8": "bootstrap 入选率",
        "0.057": "噪声地板（高检出）", "0.010": "噪声地板（低检出）",
        "−0.0047": "Visium mouse 匹配后 ΔDSI", "−0.0012": "MERFISH WT 匹配后 ΔDSI"}
for k, v in nums.items():
    if k not in txt:
        issues.append(f"未见数值 {k}（{v}）")

for pat, desc in [(r"TODO|TBD|XXX", "占位符")]:
    m = re.findall(pat, txt)
    if m:
        issues.append(f"{desc}: 出现 {len(m)} 次")

# 图号连续性：正文出现的 Fig N 应覆盖 1,2,3,4,5
figs = {int(x) for x in re.findall(r"Fig(?:ure)?\.?\s*(\d)", txt)}
missing = [n for n in range(1, 6) if n not in figs]
if missing:
    issues.append(f"正文未引用图号: {missing}")

# 首次引用顺序应单调递增（期刊要求按序引用）
first_seen, order = set(), []
for m in re.finditer(r"Fig(?:ure)?\.?\s*(\d)", txt):
    n = int(m.group(1))
    if n not in first_seen:
        first_seen.add(n)
        order.append(n)
if order != sorted(order):
    issues.append(f"图号首次引用顺序非递增: {order}")

# 文献引用完整性：正文引用的编号应覆盖 1..N，且不超过参考文献条目数
ref_path = os.path.join(MD, "05_References.md")
if os.path.exists(ref_path):
    n_ref = len(re.findall(r"^\d+\.\s", open(ref_path, encoding="utf-8").read(), flags=re.M))
    cited = set()
    for grp in re.findall(r"\[([\d,\s]+)\]", txt):
        cited.update(int(x) for x in re.findall(r"\d+", grp))
    uncited = [i for i in range(1, n_ref + 1) if i not in cited]
    overflow = sorted(i for i in cited if i > n_ref)
    if uncited:
        issues.append(f"参考文献未被正文引用: {uncited}")
    if overflow:
        issues.append(f"正文引用了不存在的文献编号: {overflow}")

    # Vancouver：应按首次引用顺序编号
    seq = []
    for grp in re.findall(r"\[([\d,\s]+)\]", txt):
        for x in re.findall(r"\d+", grp):
            if int(x) not in seq:
                seq.append(int(x))
    if seq != sorted(seq):
        issues.append(f"文献编号未按首次引用顺序: {seq}")
    print(f"\n文献: 表内 {n_ref} 条，正文引用 {len(cited)} 条，首引顺序 {seq}")

wc = len(txt.split())
print(f"\n总词数: {wc:,}")
print(f"输出: {OUT}")
if issues:
    print("\n[需人工确认]")
    for i in issues:
        print("  -", i)
else:
    print("\n一致性检查通过")
