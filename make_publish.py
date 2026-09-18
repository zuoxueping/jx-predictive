# -*- coding: utf-8 -*-
"""从 dashboard.py 生成发布版 app.py: 注入 SQLite 兼容层, 使 GitHub/Streamlit Cloud 无需本地 MySQL."""
import os

SRC = r"D:\ym-predictive\pythonProject\dashboard.py"
DST = r"D:\ym-predictive\publish_app\app.py"

STUB = '''# ==================== 配置 ====================
# 发布版(GitHub/Streamlit Cloud)无本地 MySQL, 改用同目录 SQLite 数据库.
import sqlite3
from datetime import datetime as _dt, date as _date, timedelta as _td, timezone as _tz
import re

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "power_data.db")

# 表名映射: dashboard.py 里的 MySQL 表名 -> SQLite 里的英文表名
_TABLE_MAP = {
    "检修记录": "maint",
    "月度平衡": "balance",
    "断面限额": "section",
    "月度披露报告": "disclosure",
    "月度交易计划": "trade_plan",
    "联络线分时": "tieline",
    "weather_hourly": "weather",
    "daily_spot_price": "spot_price",
    "daily_renewable": "daily_renewable",
    "daily_load": "daily_load",
    "daily_tie_line": "daily_tie_line",
}


class _SQLiteCursor:
    """兼容 MySQL cursor 的 SQLite cursor 包装, 负责 SQL 方言转换."""
    def __init__(self, cur):
        self._cur = cur

    @staticmethod
    def _translate(sql, params=None):
        params = list(params) if params else []
        # 1) 表名映射
        for cn, en in _TABLE_MAP.items():
            sql = re.sub(r"(?<![A-Za-z0-9_])" + re.escape(cn) + r"(?![A-Za-z0-9_])", en, sql)
        # 2) 去掉 MySQL 反引号(SQLite 中文列名可直接用)
        sql = sql.replace("`", "")
        # 2.5) 防御: 归一化源文件里偶发的双百分号 %% -> % (否则 strftime 拿到 %% 返回字面量)
        sql = sql.replace("%%", "%")
        # 3) MySQL 日期函数 -> SQLite
        # 3.0) GREATEST(a, b) -> SQLite 标量 MAX(a, b)
        #      必须在 NOW() 替换之前做: 此时参数是 %s / NOW(), 均无括号, 正则才能正确捕获两个参数.
        sql = re.sub(r"GREATEST\(([^)]+),\s*([^)]+)\)", lambda m: f"MAX({m.group(1)}, {m.group(2)})", sql)
        def _date_add_repl(m):
            nonlocal params
            placeholder = m.group(1)
            if placeholder == "%s" and params:
                val = params.pop(0)
            else:
                val = placeholder.strip()
            return f"datetime(NOW(), '+{val} day')"
        sql = re.sub(r"DATE_ADD\\(\\s*NOW\\(\\)\\s*,\\s*INTERVAL\\s+(%s|\\d+)\\s+DAY\\s*\\)", _date_add_repl, sql)
        # NOW()/CURDATE() 不在这里替换, 由 _SQLiteConnection.create_function 实现, 统一按北京时间(UTC+8)返回.
        sql = re.sub(r"YEAR\\(([^)]+)\\)", lambda m: f"CAST(strftime('%Y', {m.group(1)}) AS INTEGER)", sql)
        sql = re.sub(r"MONTH\\(([^)]+)\\)", lambda m: f"CAST(strftime('%m', {m.group(1)}) AS INTEGER)", sql)
        sql = re.sub(r"DAY\\(([^)]+)\\)", lambda m: f"CAST(strftime('%d', {m.group(1)}) AS INTEGER)", sql)
        sql = re.sub(r"DATE_FORMAT\\(([^,]+),\\s*'([^']+)'\\s*\\)", lambda m: f"strftime('{m.group(2)}', {m.group(1)})", sql)
        # 4) 参数占位符 %s -> ?
        sql = sql.replace("%s", "?")
        return sql, params

    def execute(self, sql, params=None):
        sql, params = self._translate(sql, params)
        if params:
            return self._cur.execute(sql, params)
        return self._cur.execute(sql)

    def fetchone(self):
        return self._cur.fetchone()

    def fetchall(self):
        return self._cur.fetchall()

    def __getattr__(self, name):
        return getattr(self._cur, name)


class _SQLiteConnection(sqlite3.Connection):
    """SQLite 连接子类: pandas.read_sql 能识别为 sqlite3 连接, 且 cursor() 做 SQL 翻译."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Streamlit Cloud 服务器为 UTC; 数据库里的 weather 时间均为北京时间, 所以 NOW()/CURDATE() 统一按 UTC+8 返回.
        self.create_function("NOW", 0, lambda: (_dt.now(_tz.utc) + _td(hours=8)).strftime("%Y-%m-%d %H:%M:%S"))
        self.create_function("CURDATE", 0, lambda: (_dt.now(_tz.utc) + _td(hours=8)).date().strftime("%Y-%m-%d"))

    def cursor(self):
        return _SQLiteCursor(super().cursor())
'''

s = open(SRC, encoding="utf-8").read()

# 1. 删除 MySQL 驱动 import
assert "import pymysql\n" in s
s = s.replace("import pymysql\n", "", 1)

# 2. 在顶部加 os (用于定位 db 文件)
s = s.replace("# -*- coding: utf-8 -*-\n", "# -*- coding: utf-8 -*-\nimport os\n", 1)

# 3. 替换 get_conn(): MySQL -> SQLite
OLD_GET_CONN = '''def get_conn():
    """统一数据库连接入口。本地返回 MySQL 连接；发布版由 make_publish.py 替换为 SQLite。"""
    return pymysql.connect(**DB_CONFIG)'''

NEW_GET_CONN = '''def get_conn():
    return sqlite3.connect(DB_PATH, factory=_SQLiteConnection)'''

assert OLD_GET_CONN in s, "get_conn() 未匹配, 请检查 dashboard.py 是否改动过"
s = s.replace(OLD_GET_CONN, NEW_GET_CONN, 1)

# 4. 中文表名 -> 英文表名 (仅简单 SELECT * FROM 模式, 复杂 SQL 由 SQLite stub 运行时翻译)
for cn, en in [("检修记录", "maint"), ("月度平衡", "balance"), ("断面限额", "section"),
               ("月度披露报告", "disclosure"), ("月度交易计划", "trade_plan"),
               ("联络线分时", "tieline")]:
    s = s.replace(f"SELECT * FROM {cn}", f"SELECT * FROM {en}")

# 5. 注入 SQLite 兼容 stub, 覆盖 DB_CONFIG 块
import re
m = re.search(r"# =+ 配置 =+\nDB_CONFIG = \{.*?\}\n", s, re.DOTALL)
if not m:
    raise RuntimeError("未找到 DB_CONFIG 配置块")
s = s[:m.start()] + STUB + s[m.end():]

open(DST, "w", encoding="utf-8").write(s)
print("✅ app.py 已生成:", DST, "字符数", len(s))
print("   含 import pymysql:", "import pymysql" in s)
print("   含 DB_PATH:", "DB_PATH" in s)
print("   含 get_conn:", "def get_conn()" in s)
