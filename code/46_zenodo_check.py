# -*- coding: utf-8 -*-
"""检查 Zenodo 记录是否含脑区注释对照表（是否能修复 Allen 映射）。"""
import json
import urllib.request

for rec in ["20682863"]:
    try:
        u = f"https://zenodo.org/api/records/{rec}"
        req = urllib.request.Request(u, headers={"User-Agent": "research"})
        j = json.loads(urllib.request.urlopen(req, timeout=60).read().decode())
        print("标题:", j.get("metadata", {}).get("title", ""))
        print("文件数:", len(j.get("files", [])))
        for f in j.get("files", []):
            print(f"  {f.get('key')}  {round(f.get('size', 0)/1e6, 1)} MB")
        print("\n描述片段:")
        desc = j.get("metadata", {}).get("description", "")
        print(desc[:1500])
    except Exception as e:
        print("失败:", type(e).__name__, e)
