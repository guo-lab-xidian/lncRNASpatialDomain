# -*- coding: utf-8 -*-
"""用真实数据演示匹配的具体过程：以 Miat 为例。"""
import numpy as np
import pandas as pd

R = r"E:\研究工作\agent长非与空间域\lncRNA空间域\results"
s = pd.read_csv(R + r"\H1H2\gene_specificity.csv", index_col=0)
s = s[~s.index.duplicated(keep="first")]
s = s.rename(columns={"panel_detect": "detect", "DSI_domain": "DSI"})

target = "Miat"
d = s.loc[target, "detect"]
print(f"目标 lncRNA: {target}")
print(f"  检出率 = {d:.4f}（{d:.1%} 的细胞检出）")
print(f"  DSI    = {s.loc[target, 'DSI']:.4f}")
print(f"  匹配窗口 = [{d*0.8:.4f}, {d*1.2:.4f}]  (即 {d*0.8:.1%} – {d*1.2:.1%})")

pc = s[s["gene_class"] == "protein_coding"]
pool = pc[(pc.detect >= d * 0.8) & (pc.detect <= d * 1.2)]
print(f"\n候选池（检出率落在窗口内的蛋白编码基因）: {len(pool)} 个")

rng = np.random.default_rng(0)
ctrl = pool.sample(5, random_state=int(rng.integers(1e9)))
print(f"\n随机抽取 5 个对照：")
print(ctrl[["detect", "DSI"]].round(4).to_string())
print(f"\n对照 DSI 中位数 = {ctrl['DSI'].median():.4f}")
print(f"统计量 Δ = DSI(Miat) − 对照中位数 = {s.loc[target,'DSI']:.4f} − "
      f"{ctrl['DSI'].median():.4f} = {s.loc[target,'DSI'] - ctrl['DSI'].median():+.4f}")

print("\n" + "=" * 70)
print("如果不做匹配，直接和全部蛋白编码基因比：")
print(f"  全部蛋白编码基因 DSI 中位数 = {pc['DSI'].median():.4f}")
print(f"  它们的中位检出率 = {pc['detect'].median():.4f}（Miat 是它的 {d/pc['detect'].median():.1f} 倍）")
print(f"  直接相减 = {s.loc[target,'DSI'] - pc['DSI'].median():+.4f}")
print("  → 但这个差值混入了'检出率不同'的影响，不能归因于基因类别")

