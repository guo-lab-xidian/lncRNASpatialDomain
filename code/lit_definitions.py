# -*- coding: utf-8 -*-
"""
抓取空间转录组领域奠基性方法论文的全文，提取其中对"spatial domain / niche / region"
的定义性语句，用于界定概念。

数据源：Europe PMC REST（全文 XML，开放获取）
"""
import json
import os
import re
import sys
import urllib.request
import urllib.parse

OUT = r"E:\研究工作\agent长非与空间域\lncRNA空间域\results\literature"
os.makedirs(OUT, exist_ok=True)

PMIDS = {
    "34711970": "SpaGCN (Nat Methods 2021)",
    "35365632": "STAGATE (Nat Commun 2022)",
    "36859400": "GraphST (Nat Commun 2023)",
    "38413725": "BANKSY (Nat Genet 2024)",
    "34083791": "BayesSpace (Nat Biotechnol 2021)",
    "38066188": "CellCharter (Nat Genet 2024)",
    "42555503": "Review: Computational analysis in ST (BIB 2026)",
    "42893560": "Review: AI in ST (BIB 2026)",
    "42462710": "Cohen 2026 Cell (source data paper)",
}

EPMC = "https://www.ebi.ac.uk/europepmc/webservices/rest"


def get(url, timeout=60):
    req = urllib.request.Request(url, headers={"User-Agent": "research-script"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", errors="replace")


def find_pmcid(pmid):
    q = urllib.parse.quote(f"EXT_ID:{pmid} AND SRC:MED")
    j = json.loads(get(f"{EPMC}/search?query={q}&format=json&resultType=core"))
    res = j.get("resultList", {}).get("result", [])
    if not res:
        return None, None
    r = res[0]
    return r.get("pmcid"), r.get("isOpenAccess")


def fulltext(pmcid):
    return get(f"{EPMC}/{pmcid}/fullTextXML", timeout=120)


DEF_PATTERNS = [
    r"[^.]{0,200}\bspatial domains?\b[^.]{0,300}\.",
    r"[^.]{0,200}\b(?:are|is) defined as\b[^.]{0,300}\.",
    r"[^.]{0,200}\brefers? to\b[^.]{0,300}\.",
    r"[^.]{0,200}\b(?:spatial niche|niches)\b[^.]{0,250}\.",
    r"[^.]{0,200}\b(?:spatial regions?|tissue regions?)\b[^.]{0,250}\.",
]
DEF_PATTERNS = [re.compile(p, re.I) for p in DEF_PATTERNS]


def strip_tags(xml):
    txt = re.sub(r"<[^>]+>", " ", xml)
    txt = re.sub(r"\s+", " ", txt)
    return txt


def main():
    summary = {}
    for pmid, label in PMIDS.items():
        print("=" * 100)
        print(f"[{pmid}] {label}")
        try:
            pmcid, oa = find_pmcid(pmid)
        except Exception as e:
            print("  检索 PMC 失败:", e)
            continue
        print("  PMCID:", pmcid, "| OA:", oa)
        if not pmcid:
            continue
        try:
            xml = fulltext(pmcid)
        except Exception as e:
            print("  抓全文失败:", e)
            continue
        open(os.path.join(OUT, f"{pmid}_{pmcid}.xml"), "w", encoding="utf-8").write(xml)
        txt = strip_tags(xml)
        print("  全文长度:", len(txt))

        hits = []
        for pat in DEF_PATTERNS:
            hits.extend(pat.findall(txt))
        # 去重、截断、优先包含 domain/niche 的句子
        seen, uniq = set(), []
        for s in hits:
            s = s.strip(" .") + "."
            key = s[:80].lower()
            if key in seen or len(s) < 40:
                continue
            seen.add(key)
            uniq.append(s)
        # 只保留含定义性词汇的
        uniq = [s for s in uniq if re.search(r"\b(domain|niche|region)\b", s, re.I)]
        print(f"  候选定义句: {len(uniq)}")
        for s in uniq[:12]:
            print("    » " + s[:420])
        summary[pmid] = {"label": label, "pmcid": pmcid, "sentences": uniq[:40]}

    with open(os.path.join(OUT, "definition_sentences.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("\n已写出:", os.path.join(OUT, "definition_sentences.json"))


if __name__ == "__main__":
    main()
