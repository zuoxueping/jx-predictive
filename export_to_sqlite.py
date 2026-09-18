# -*- coding: utf-8 -*-
"""把 MySQL 核心表导出到 SQLite 文件(power_data.db), 供发布版看板离线读取."""
import pymysql, pandas as pd, sqlite3, os

MYSQL = dict(host="localhost", port=3306, user="root",
             password="123456", database="power_maintenance", charset="utf8mb4")
# 导出为英文表名, 避免 SQLite 中文标识符兼容问题
TABLES = {
    "检修记录": "maint",
    "月度平衡": "balance",
    "断面限额": "section",
    "月度披露报告": "disclosure",
    "月度交易计划": "trade_plan",
    "联络线分时": "tieline",  # 2026-09-08 新增: 外送/受入的小时级实测
    "weather_hourly": "weather",  # 2026-09-11 新增: 12 点位小时级天气(检修作业窗口)
    "daily_spot_price": "spot_price",  # 2026-09-11 新增: 现货日前/实时价(1-8月, 24时段)
}

# 日度态势空表: 在云发布版中保持结构一致, 数据待接入
EMPTY_TABLES = {
    "daily_renewable": """
        `日期` DATE, `时段` TINYINT,
        `风电出力_MW` REAL, `光伏出力_MW` REAL, `新能源总出力_MW` REAL,
        PRIMARY KEY (`日期`, `时段`)
    """,
    "daily_load": """
        `日期` DATE, `时段` TINYINT,
        `统调负荷_MW` REAL, `全社会负荷_MW` REAL,
        PRIMARY KEY (`日期`, `时段`)
    """,
    "daily_tie_line": """
        `日期` DATE, `时段` TINYINT, `通道名` VARCHAR(50),
        `外送_MW` REAL, `受入_MW` REAL, `净送出_MW` REAL,
        PRIMARY KEY (`日期`, `时段`, `通道名`)
    """,
}

out_db = os.path.join(os.path.dirname(os.path.abspath(__file__)), "power_data.db")
if os.path.exists(out_db):
    os.remove(out_db)

src = pymysql.connect(**MYSQL)
sq = sqlite3.connect(out_db)
total = 0
for cn, en in TABLES.items():
    df = pd.read_sql(f"SELECT * FROM `{cn}`", src)
    df.to_sql(en, sq, if_exists="replace", index=False)
    total += len(df)
    print(f"  {cn} → {en}: {len(df)} 行, {len(df.columns)} 列")

# 创建空表, 保持发布版 SQL 结构一致
for en, ddl in EMPTY_TABLES.items():
    sq.execute(f"DROP TABLE IF EXISTS `{en}`")
    sq.execute(f"CREATE TABLE `{en}` ({ddl})")
    print(f"  (空表) {en}: 已创建")

src.close(); sq.close()
print(f"完成 → {out_db} (共 {total} 行)")
