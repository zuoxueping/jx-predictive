# -*- coding: utf-8 -*-
"""发布启动脚本: 读平台注入的 PORT 环境变量, 启动 streamlit 并绑定 0.0.0.0."""
import os, sys, subprocess

port = os.environ.get("PORT", "8501")
print(f"[run.py] 使用 PORT={port}, 启动 streamlit ...")

subprocess.run([
    sys.executable, "-m", "streamlit", "run", "app.py",
    "--server.port", str(port),
    "--server.address", "0.0.0.0",
    "--server.headless", "true",
    "--browser.gatherUsageStats", "false",
])
