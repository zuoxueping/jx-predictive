# -*- coding: utf-8 -*-
"""一键把 app.py + power_data.db 推送到 GitHub 发布仓库.
比网页拖文件更可靠, 失败会明确报错, 不会假装成功.

用法:
  1. 先跑一次 publish_update.py (确保本地 app.py 和 db 都是最新的)
  2. 双击本脚本, 或在 publish_app 目录跑:  python git_push.py
  3. 第一次推送会要求认证, 按下面"认证失败解决"操作

⚠ 本脚本只推送 app.py 和 power_data.db 两个发布文件, 不会动其它东西.
"""
import os, subprocess, sys
from datetime import datetime

base = os.path.dirname(os.path.abspath(__file__))
os.chdir(base)

REPO = "https://github.com/zuoxueping/jx-predictive.git"
BRANCH = "main"
FILES = ["app.py", "power_data.db"]  # 发布仓库只放这两个文件 + README/requirements


def run(cmd, check=False):
    """跑一条 git 命令, 返回 (returncode, stdout, stderr)."""
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    return r.returncode, r.stdout.strip(), r.stderr.strip()


print("=" * 56)
print("  一键推送 app.py + power_data.db 到 GitHub")
print("=" * 56)

# ---------- 1. 确认本地文件存在且比上次新 ----------
print("\n[1/5] 检查本地文件...")
for f in FILES:
    if not os.path.exists(f):
        print(f"  ❌ 找不到 {f}, 请先跑 publish_update.py 重新生成")
        sys.exit(1)
mtime_db = datetime.fromtimestamp(os.path.getmtime("power_data.db"))
print(f"  ✓ app.py        {os.path.getsize('app.py'):>8} 字节")
print(f"  ✓ power_data.db {os.path.getsize('power_data.db'):>8} 字节  (修改于 {mtime_db:%Y-%m-%d %H:%M})")

# ---------- 2. 初始化 git / 远程 ----------
print("\n[2/5] 初始化 git 仓库 ...")
if not os.path.exists(".git"):
    run(["git", "init", "-b", BRANCH])
    run(["git", "remote", "add", "origin", REPO])
    print("  ✓ 首次初始化完成")
else:
    rc, out, _ = run(["git", "remote", "get-url", "origin"])
    if rc != 0:
        run(["git", "remote", "add", "origin", REPO])
    elif REPO not in out:
        run(["git", "remote", "set-url", "origin", REPO])
    print("  ✓ git 仓库已就绪")

# ---------- 3. 拉取 GitHub 已有文件(README, requirements) ----------
print("\n[3/5] 同步 GitHub 现有文件 (README / requirements.txt) ...")
rc, _, err = run(["git", "pull", "origin", BRANCH,
                  "--allow-unrelated-histories", "--no-edit", "--no-rebase"])
if rc != 0 and "Couldn't find remote ref" not in err and "There is no tracking" not in err:
    print(f"  (忽略: {err[:120]})")
else:
    print("  ✓ 已同步")

# ---------- 4. 配置 git 用户(commit 需要) ----------
print("\n[4/5] 配置 git 用户 ...")
run(["git", "config", "user.email", "zuxueping@users.noreply.github.com"])
run(["git", "config", "user.name", "zuxueping"])
print("  ✓ 已配置")

# ---------- 5. add + commit + push ----------
print("\n[5/5] 暂存 → 提交 → 推送 ...")
run(["git", "add"] + FILES)
ts = datetime.now().strftime("%Y-%m-%d %H:%M")
rc, out, err = run(["git", "commit", "-m", f"更新发布版 {ts}"])
if "nothing to commit" in (out + err):
    print("  ⚠ 本地文件与上次提交内容完全一样, 跳过 commit")
    print("     如果你刚跑过 publish_update.py, 那就是真的没变化;")
    print("     如果你改了 dashboard.py, 请先跑 publish_update.py 再跑本脚本")
else:
    print(f"  ✓ 已提交: 更新发布版 {ts}")

rc, out, err = run(["git", "push", "-u", "origin", BRANCH])
if rc == 0:
    print()
    print("=" * 56)
    print("✅ 推送成功!")
    print("   Streamlit Cloud 将在 1-2 分钟内自动重新部署")
    print("   你的链接地址不变, 内容已更新为最新版")
    print("   部署进度可看: https://share.streamlit.io/ 你的应用")
    print("=" * 56)
    sys.exit(0)
else:
    print()
    print("❌ 推送失败:")
    print(err)
    if any(k in err for k in ("Authentication", "denied", "403", "support for password")):
        print()
        print("=" * 56)
        print("【认证失败解决办法】(只需做一次, 之后不用再输)")
        print()
        print("  1. 浏览器打开  https://github.com/settings/tokens")
        print("  2. 点 'Generate new token' → 'Generate new token (classic)'")
        print("     - Note: 随便填, 如 jx-predictive-push")
        print("     - Expiration: 选 'No expiration' 或 '90 days'")
        print("     - 勾选 'repo' 那一行 (Full control of private repositories)")
        print("     - 滚到底点 'Generate token' → 复制 token (以 ghp_ 开头)")
        print()
        print("  3. 重新双击本脚本")
        print("     - 弹窗用户名: 填  zuxueping")
        print("     - 弹窗密码:   粘贴刚才复制的 token (不是 GitHub 登录密码!)")
        print()
        print("  4. 成功后 Windows 凭据管理器会记住, 以后推送不会再问")
        print("=" * 56)
    else:
        print()
        print("如果是其它错误, 把上面 ❌ 那段截图发给我看.")
    sys.exit(1)
