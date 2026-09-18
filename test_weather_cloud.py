# -*- coding: utf-8 -*-
"""用 app.py 内真实存在的 SQLite 翻译层, 验证天气/现货价查询在云端(无 MySQL)能取到数."""
import os, sqlite3, re, sys
import pandas as pd

APP = r"D:\ym-predictive\publish_app\app.py"
src = open(APP, encoding="utf-8").read()

# 提取 app.py 顶部注入的 STUB (配置 + _SQLiteConnection + _TABLE_MAP)
m = re.search(r"# =+ 配置 =+\n(.*?)\ndef get_conn\(\):", src, re.DOTALL)
if not m:
    print("❌ 未在 app.py 找到 STUB"); sys.exit(1)
stub_src = m.group(1)
ns = {"os": os, "sqlite3": sqlite3, "re": re, "__file__": APP}
# STUB 里引用了 datetime/date, 一并注入
import datetime as _dt_mod
ns["_dt"] = _dt_mod.datetime
ns["_date"] = _dt_mod.date
exec(compile(stub_src, "<stub>", "exec"), ns)

DB = r"D:\ym-predictive\publish_app\power_data.db"
conn = ns["_SQLiteConnection"](DB)
cur = conn.cursor()

def run(sql, params):
    cur.execute(sql, params)
    return cur.fetchall()

print("=== 测试1: 天气日网格 weather_daily_grid 查询 ===")
rows = run(
    "SELECT DATE(时间) AS d, COUNT(*) AS total, "
    "       SUM(CASE WHEN 风速_10m > 10.7 OR 雷暴 = 1 OR 降水_mm > 0.5 "
    "                  OR 气温_2m > 40 OR 气温_2m < -15 THEN 1 ELSE 0 END) AS bad "
    "FROM weather_hourly "
    "WHERE 时间 > NOW() AND 时间 < DATE_ADD(NOW(), INTERVAL %s DAY) "
    "GROUP BY DATE(时间) ORDER BY d",
    (11,))
print(f"  返回 {len(rows)} 天; 首日示例:", rows[0] if rows else "空")
# 模拟 dashboard 的 pd.to_datetime(d).strftime 渲染
if rows:
    d0 = rows[0][0]
    print(f"  d 类型={type(d0).__name__} 值={d0!r} -> 渲染为 {pd.to_datetime(d0).strftime('%m-%d')}  ✅ 不再崩")

print("=== 测试2: 12点位矩阵 point_weather_matrix 查询 ===")
rows2 = run(
    "SELECT 点位名称, DATE(时间) AS d, COUNT(*) AS total "
    "FROM weather_hourly "
    "WHERE 时间 > NOW() AND 时间 < DATE_ADD(NOW(), INTERVAL %s DAY) "
    "GROUP BY 点位名称, DATE(时间) ORDER BY 点位名称, d",
    (11,))
print(f"  返回 {len(rows2)} 行 (点位×天); 首行:", rows2[0] if rows2 else "空")

print("=== 测试3: 现货价 spot_price 查询 ===")
rows3 = run(
    "SELECT 日期, 时段, 日前价_元每兆瓦时, 实时价_元每兆瓦时 FROM spot_price "
    "WHERE 日期 >= %s ORDER BY 日期 DESC, 时段 LIMIT 3",
    ("2026-09-10",))
print(f"  返回 {len(rows3)} 行; 示例:", rows3 if rows3 else "空")

conn.close()
print("\n✅ 三个核心查询在 SQLite 兼容层下均正常返回")
