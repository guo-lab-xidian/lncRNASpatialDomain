# -*- coding: utf-8 -*-
"""把分节草稿组装成单一稿件，并做基本的一致性检查。"""
import os
import re

MD = r"E:\研究工作\agent长非与空间域\lncRNA空间域\manuscript"
OUT = os.path.join(MD, "manuscript_full.md")

TITLE = """# What can long non-coding RNAs tell us about spatial domains? A multi-platform reassessment in mouse and human brain

**Running title:** Reassessing lncRNA spatial specificity

**Xingli Guo**^1,\\*

^1 School of Computer Science and Technology, Xidian University, 266 Xinglong Section of Xifeng Road, Xi'an, Shaanxi 710126, China

\\* Corresponding author. Email: xlguo@mail.xidian.edu.cn

*Draft assembled from section files. Figures in results/figures/.*
"""

ORDER = ["00_Abstract.md", "02_Introduction.md", "03_Results.md",
         "04_Discussion.md", "01_Methods.md", "06_Figure_captions.md", "07_Supplementary.md",
         "05_References.md"]

BACKMATTER = """
---

## Data availability

The isoform-resolved MERFISH data are available from Zenodo (10.5281/zenodo.20682863). The mouse and human Visium datasets are public 10x Genomics resources; the Visium HD mouse binned outputs were obtained from the same provider. Derived tables (gene-level DSI, Moran's *I*, matched comparisons, bootstrap stability) are provided with this submission as supplementary tables.

## Code availability

All analysis code is available at [repository URL to be inserted]. The MERFISH matrices were processed with a custom row-block streaming layer because the full matrices exceed available memory; this layer and all figure-generating scripts are included.

## Author contributions

X.G. conceived the study, performed all data analysis, generated the figures, and wrote the manuscript.

## Competing interests

The authors declare no competing interests.

## Acknowledgements

[To be completed.]
"""

parts = [TITLE]
for fn in ORDER:
    p = os.path.join(MD, fn)
    if not os.path.exists(p):
        print("missing:", fn)
        continue
    # 注意：PowerShell 的 Set-Content -Encoding UTF8 会写入 BOM，需剔除
    txt = open(p, encoding="utf-8-sig").read().replace("\ufeff", "").strip()
    parts.append(txt)
    print("merged %-26s %5d words" % (fn, len(txt.split())))

parts.append(BACKMATTER.strip())
body = "\n\n---\n\n".join(parts)
open(OUT, "w", encoding="utf-8").write(body)

issues = []
for k in ["475", "604", "1,090", "613", "0.879", "93.8", "20.7%", "0.08%"]:
    if k not in body:
        issues.append("missing number: " + k)
if re.search(r"Fig\.\s*4", body):
    issues.append("Fig. 4 is referenced but was merged into Fig. 2c")
if re.search(r"TODO|TBD|XXX", body):
    issues.append("placeholder text found")
if re.search(r"^#+\s*\ufeff", body, flags=re.M):
    issues.append("BOM character before a heading")

print("\ntotal words: %d" % len(body.split()))
print("sections: %d" % sum(1 for l in body.split("\n") if l.startswith("# ")))
print("output:", OUT)
if issues:
    print("\n[check]")
    for i in issues:
        print("  -", i)
else:
    print("\nconsistency check passed")

