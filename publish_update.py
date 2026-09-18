# -*- coding: utf-8 -*-
"""更新 GitHub 发布链接内容: 一键重新生成发布版 app.py + 导出最新 power_data.db。

用法 (在你自己电脑上, 不是云端):
  1. 确保本机 MySQL 已启动 (账号密码见 export_to_sqlite.py)
  2. 双击本文件, 或在 publish_app 目录打开命令行跑:  python publish_update.py
  3. 脚本跑完后, 去 GitHub 仓库网页, 把生成的两个文件拖进去覆盖旧的即可。
     Streamlit 云检测到文件变化会自动重新部署, 你的链接地址不变、内容变新。

说明:
  - app.py 由 make_publish.py 从【最新 dashboard.py】重新生成(含你新加的面板)
  - power_data.db 由 export_to_sqlite.py 从【本机 MySQL 当前数据】重新导出
  - 如果你用 Git 客户端(而非网页拖文件), 脚本跑完后可自行 git add/commit/push
"""
import subprocess, os, sys

base = os.path.dirname(os.path.abspath(__file__))

print("=" * 52)
print("  更新发布链接内容（生成最新 app.py + 最新 db）")
print("=" * 52)

# 1) 导出最新数据 (需本机 MySQL 运行)
print("\n[1/2] 从 MySQL 导出最新数据 → power_data.db ...")
try:
    subprocess.run([sys.executable, os.path.join(base, "export_to_sqlite.py")], check=True)
except Exception as e:
    print("  ⚠ 导出失败:", e)
    print("  请确认本机 MySQL 已启动、账号密码正确。将沿用旧 db 快照继续。")
    try:
        input("  按回车仍要继续重新生成 app.py ...")
    except Exception:
        pass

# 2) 从最新 dashboard.py 生成发布版 app.py
print("\n[2/2] 从最新 dashboard.py 生成发布版 app.py ...")
try:
    subprocess.run([sys.executable, os.path.join(base, "make_publish.py")], check=True)
except Exception as e:
    print("  ❌ 生成 app.py 失败:", e)
    print("  常见原因: 你改动了 dashboard.py 顶部的 run_query 函数, 导致 make_publish 匹配失败。")
    print("  解决: 找我, 我更新一下同步脚本即可。")
    sys.exit(1)

print("\n" + "=" * 52)
print("本地文件已更新完成:")
print(f"  app.py        : {os.path.join(base, 'app.py')}")
print(f"  power_data.db : {os.path.join(base, 'power_data.db')}")
print("-" * 52)
print("下一步 (二选一):")
print("  A. 网页法(最简单): 打开 GitHub 仓库 → Add file → Upload → 把上面")
print("     两个文件拖进去覆盖旧文件 → Commit。Streamlit 自动重新部署。")
print("  B. Git 法: 在 publish_app 目录 git add . && git commit -m 更新 && git push")
print("=" * 52)
