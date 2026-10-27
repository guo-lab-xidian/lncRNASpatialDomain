# -*- coding: utf-8 -*-
"""
把所有脚本中的原始绝对路径前缀替换成你自己的项目根目录。

用法
----
    python retarget_paths.py /path/to/your/root
    python retarget_paths.py "D:\\spatial_lncRNA"      # Windows

行为
----
  * 只替换前缀 `E:\\研究工作\\agent长非与空间域`，其后的相对结构保持不变
  * 逐文件写成 `<name>.py.bak` 备份，可随时还原
  * 打印每个被改动的行数，便于核对

替换示例
--------
  r"E:\\研究工作\\agent长非与空间域\\lncRNA空间域\\results"
      ->  r"D:\\spatial_lncRNA\\lncRNA空间域\\results"

注意
----
Windows 路径里的反斜杠在 Python 原始字符串中无需转义，因此替换时
保留反斜杠原样即可。替换前建议先 `git diff` 复查。
"""
import os
import shutil
import sys

OLD = "E:\\研究工作\\agent长非与空间域"
HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.join(HERE, "code")


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    new_root = sys.argv[1].rstrip("\\/")
    if not os.path.isdir(CODE):
        sys.exit(f"找不到 code 目录: {CODE}")

    changed_total = 0
    for fn in sorted(os.listdir(CODE)):
        if not fn.endswith(".py"):
            continue
        p = os.path.join(CODE, fn)
        with open(p, encoding="utf-8") as f:
            txt = f.read()
        n = txt.count(OLD)
        if n == 0:
            continue
        shutil.copy2(p, p + ".bak")
        with open(p, "w", encoding="utf-8", newline="\n") as f:
            f.write(txt.replace(OLD, new_root))
        changed_total += n
        print(f"  {fn}: {n} 处")

    print(f"\n共替换 {changed_total} 处，根目录 → {new_root}")
    print("备份后缀: .py.bak")


if __name__ == "__main__":
    main()
