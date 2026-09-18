@echo off
cd /d D:\ym-predictive\publish_app
D:\ym-predictive\pythonProject\.venv\Scripts\python.exe -m streamlit run app.py --server.port 8501 --browser.gatherUsageStats false
pause
