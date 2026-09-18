@echo off
set PY=D:\ym-predictive\pythonProject\.venv\Scripts\python.exe
"%PY%" D:\ym-predictive\pythonProject\import_spot_price.py
"%PY%" D:\ym-predictive\pythonProject\crawl_weather.py
"%PY%" D:\ym-predictive\publish_app\publish_update.py
"%PY%" D:\ym-predictive\publish_app\git_push.py
pause
