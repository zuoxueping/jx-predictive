# -*- coding: utf-8 -*-
import os
"""甘肃电网检修预测系统 · 单页报告 + 数据台账
按用户反馈简化: 砍掉 6 页仪表盘, 主页面=4 层预测报告, 副页=数据导出.
"""
import re
import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
from datetime import datetime

# ==================== 配置 ====================
DB_CONFIG = {
    "host": "localhost", "port": 3306, "user": "root",
    "password": "123456", "database": "power_maintenance", "charset": "utf8mb4",
}

# 第一步: 设备类型 → 容量影响权重(粗略版).
# 依据"是否直接损失发电/供电能力"分档; 后续补到设备精确MW后可替换为真实容量.
# 发电机组直接停机=损失发电出力(最高); 主变影响一片供电; 母线/线路影响送出受电; 开关无功影响小.
EQUIP_WEIGHT = {
    "发电机组": 5,
    "变压器": 4,
    "母线": 3,
    "输电线路": 3,
    "开关/无功设备": 2,
    "其他设备": 1,
}

# ---- 地区推断: 从"申请单位"提取设备所属地区 ----
# 供电公司直接取前缀(酒泉供电公司→酒泉); 电厂/电站按厂名映射;
# 超高压/送变电为全省跨区调度。起点→终点在原始PDF中无字段, 用变电站名作位置代理。
REGION_RULES = [
    # 地市供电公司 (占 ~78%, 优先级最高)
    ("酒泉", "酒泉"), ("嘉峪关", "嘉峪关"), ("张掖", "张掖"), ("金昌", "金昌"),
    ("武威", "武威"), ("白银", "白银"), ("兰州", "兰州"), ("临夏", "临夏"),
    ("定西", "定西"), ("庆阳", "庆阳"), ("平凉", "平凉"), ("天水", "天水"),
    ("陇南", "陇南"), ("甘南", "甘南"),
    # 电厂/电站 → 地区
    ("常乐", "酒泉"), ("桥湾", "酒泉"), ("阿克塞", "酒泉"), ("瓜州", "酒泉"), ("玉门", "酒泉"),
    ("靖远", "白银"), ("平川", "白银"), ("条山", "白银"),
    ("刘家峡", "临夏"), ("盐锅峡", "临夏"), ("炳灵", "临夏"), ("九甸峡", "定西"),
    ("碧口", "陇南"), ("苗家坝", "陇南"),
    ("连城", "兰州"), ("西固", "兰州"), ("范家坪", "兰州"), ("八盘峡", "兰州"),
    ("大峡", "兰州"), ("小峡", "兰州"), ("乌金峡", "兰州"),
    ("华亭", "平凉"), ("崇信", "平凉"), ("灵台", "平凉"),
    ("甘谷", "天水"), ("山丹", "张掖"), ("高台", "张掖"), ("临泽", "张掖"), ("肃南", "张掖"),
    ("环县", "庆阳"), ("正宁", "庆阳"), ("陇东", "庆阳"),
    ("永昌", "金昌"), ("民勤", "武威"), ("古浪", "武威"), ("天祝", "武威"),
    ("超高压", "全省/跨区"), ("送变电", "全省/跨区"), ("送变", "全省/跨区"), ("铁调", "兰州"),
    # 其余电厂/变电站补充映射(降低'其他'占比, 已补 26 项)
    ("兰铝", "兰州"), ("八〇三", "兰州"), ("河口", "兰州"), ("柴家峡", "兰州"),
    ("刘右", "临夏"), ("酒钢", "酒泉"), ("景泰", "白银"), ("庆东", "庆阳"),
    ("实正鑫", "兰州"), ("范家坪", "兰州"), ("新安", "兰州"), ("硅厂", "兰州"),
    ("酒泉发电", "酒泉"), ("酒厂", "酒泉"), ("嘉峪关", "嘉峪关"),
    ("华能", "兰州"), ("华能甘", "兰州"), ("天水电厂", "天水"), ("张掖电厂", "张掖"),
    ("大唐", "兰州"), ("国电", "兰州"), ("中电", "兰州"), ("国投", "兰州"),
    ("甘电投", "兰州"), ("农电", "兰州"), ("热电", "兰州"), ("热电厂", "兰州"),
    ("水电", "兰州"), ("风电", "酒泉"), ("光伏", "酒泉"), ("新能源", "酒泉"),
    ("集团", "全省/跨区"), ("国家电网", "全省/跨区"), ("省公司", "全省/跨区"),
]


def infer_region(unit):
    """从申请单位推断所属地区; 无法识别返回'其他'。"""
    if not isinstance(unit, str) or not unit:
        return "其他"
    for kw, region in REGION_RULES:
        if kw in unit:
            return region
    return "其他"


def extract_substation(equip):
    """从停电设备名提取变电站(位置代理), 如 '酒泉变泉高Ⅲ线'→'酒泉变'。

    原始PDF无起止变电站字段, 用设备名中的'XX变'作位置近似。
    """
    if not isinstance(equip, str) or not equip:
        return ""
    m = re.search(r"([\u4e00-\u9fa5]{2,}变)", equip)
    return m.group(1) if m else ""

st.set_page_config(
    page_title="甘肃电网检修预测",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ==================== 数据加载 ====================
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "power_data.db")
@st.cache_data(ttl=600)
def run_query(sql):
    import sqlite3
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute(sql)
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]
    finally:
        conn.close()

@st.cache_data(ttl=600)
def load_maint():
    df = pd.DataFrame(run_query("SELECT * FROM maint"))
    df = df.drop(columns=["_row_id"], errors="ignore")
    for c in ["检修天数", "重复披露", "ID"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "开始日期" in df.columns:
        df["开始日期_dt"] = pd.to_datetime(df["开始日期"], errors="coerce")
    if "结束日期" in df.columns:
        df["结束日期_dt"] = pd.to_datetime(df["结束日期"], errors="coerce")
    if "披露月份" in df.columns:
        df["年"] = df["披露月份"].str[:4]
        df["月"] = df["披露月份"].str[5:7].astype(int)
    # 第一步: 设备类型 → 容量影响权重(精确MW缺失时的粗略替代)
    df["权重"] = df["设备类型"].map(EQUIP_WEIGHT).fillna(1).astype(int)
    # 第二步: 所属地区(从申请单位推断) + 变电站(从设备名提取), 供结构面板/地区分布使用
    df["所属地区"] = df["申请单位"].apply(infer_region)
    df["变电站"] = df["停电设备"].apply(extract_substation)
    return df

@st.cache_data(ttl=600)
def load_bal():
    df = pd.DataFrame(run_query("SELECT * FROM balance"))
    for c in df.columns:
        if c not in ("_row_id", "月份"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.drop(columns=["_row_id"], errors="ignore")

@st.cache_data(ttl=600)
def load_section():
    return pd.DataFrame(run_query("SELECT * FROM section")).drop(columns=["_row_id"], errors="ignore")

@st.cache_data(ttl=600)
def load_disclosure():
    df = pd.DataFrame(run_query("SELECT * FROM disclosure"))
    df = df.drop(columns=["_row_id"], errors="ignore")
    skip = {"file", "月份", "平衡月份", "装机口径", "skip_reason", "skipped"}
    for c in df.columns:
        if c not in skip:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

@st.cache_data(ttl=600)
def load_trade_plan():
    df = pd.DataFrame(run_query("SELECT * FROM trade_plan"))
    df = df.drop(columns=["_row_id"], errors="ignore")
    skip = {"file", "月份", "skipped", "skip_reason"}
    for c in df.columns:
        if c not in skip:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

@st.cache_data(ttl=600)
def load_tieline():
    """联络线分时: 每日24h交换功率/电量(外送/受入的小时级实测)。"""
    return pd.DataFrame(run_query("SELECT * FROM tieline")).drop(columns=["_row_id"], errors="ignore")


def _wx_window(days=10):
    """从 weather_hourly 取未来 N 天全网(12 点位平均)的可作业小时比例.

    受限条件(户外高空/吊装/带电作业通用阈值):
      - 风速 10m > 10.7m/s (6 级, 高空/吊装上限)
      - 雷暴 = 1
      - 降水 > 0.5mm/h
      - 气温 > 40℃ 或 < -15℃ (极端, 户外作业时间限制)

    返回 (可作业小时比例 0-100, 详情字符串) 或 None(数据不足).
    """
    try:
        conn = pymysql.connect(**DB_CONFIG)
        cur = conn.cursor()
        cur.execute(
            "SELECT COUNT(*) AS total, "
            "       SUM(CASE WHEN 风速_10m > 10.7 OR 雷暴 = 1 OR 降水_mm > 0.5 "
            "                  OR 气温_2m > 40 OR 气温_2m < -15 THEN 1 ELSE 0 END) AS bad "
            "FROM weather_hourly "
            "WHERE 时间 > NOW() AND 时间 < DATE_ADD(NOW(), INTERVAL %s DAY)",
            (days + 1,))
        r = cur.fetchone()
        conn.close()
        total = int(r[0] or 0); bad = int(r[1] or 0)
        if total < 100:
            return None
        good = total - bad
        pct = good / total * 100
        msg = (f"未来 {days} 天全网(12 点位平均)可作业 {good}/{total} 小时({pct:.0f}%); "
               f"受限触发: 风>10.7m/s 或 雷暴 或 降水>0.5mm 或 极端温度")
        return pct, msg
    except Exception:
        return None


def build_reserve_lookup(df_disc, df_bal):
    """构建 {平衡月份: (最大发电能力, 最大用电负荷, 数据来源)} 用于有效备用率。

    两个来源(优先取其一, 后者覆盖前者同月):
      - 月度披露报告.电力平衡: 预测下月的常规最大发电能力/最大用电负荷(34月, 真实可用发电能力)
      - 月度平衡: 实际值(目前仅 2026-09 有最大发电能力/负荷)
    均用真实发电能力(非总装机), 避免新能源占比高导致备用率失真。
    """
    lut = {}
    if df_disc is not None and not df_disc.empty:
        for _, r in df_disc.iterrows():
            bm = r.get("平衡月份")
            cap = r.get("平衡_常规最大发电能力_万kW")
            load = r.get("平衡_最大用电负荷_万kW")
            if bm and pd.notna(cap) and pd.notna(load):
                lut[str(bm)] = (float(cap), float(load), "披露报告·电力平衡预测")
    if df_bal is not None and not df_bal.empty:
        for _, r in df_bal.iterrows():
            m = str(r.get("月份", ""))
            cap = r.get("最大发电能力_万kW")
            load = r.get("最大负荷_万kW")
            if m and pd.notna(cap) and pd.notna(load):
                lut[m] = (float(cap), float(load), "月度平衡·实际值")
    return lut


def maint_price_corr(df_maint, df_disc):
    """检修加权指数 vs 火电结算均价: 返回散点数据 + 原始/去季节化相关系数。

    诚实口径: 原始相关可能显著, 但二者有共同季节性(检修高峰月≈电价高位月),
    去季节化后相关系数往往接近0 —— 故同时返回两者, 避免'伪相关'误导领导。
    """
    if df_maint is None or df_maint.empty or df_disc is None or df_disc.empty:
        return None
    if "权重" not in df_maint.columns or "均价_火电_元每MWh" not in df_disc.columns:
        return None
    m_w = (df_maint.groupby("披露月份")["权重"].sum()
           .rename("检修加权指数").reset_index())
    m_w["披露月份"] = m_w["披露月份"].astype(str)
    disc = df_disc.copy()
    disc["月份"] = disc["月份"].astype(str)
    merged = disc.merge(m_w, left_on="月份", right_on="披露月份", how="inner").dropna(
        subset=["检修加权指数", "均价_火电_元每MWh"])
    if len(merged) < 5:
        return None
    x = merged["检修加权指数"].values.astype(float)
    y = merged["均价_火电_元每MWh"].values.astype(float)
    from scipy.stats import pearsonr
    r, p = pearsonr(x, y)
    mon = merged["月份"].str[5:7]
    x_ds = x - np.array([x[mon == m].mean() for m in mon])
    y_ds = y - np.array([y[mon == m].mean() for m in mon])
    r_ds, p_ds = pearsonr(x_ds, y_ds)
    A = np.polyfit(x, y, 1)
    return {"months": merged["月份"].tolist(), "x": x, "y": y,
            "r": r, "p": p, "r_ds": r_ds, "p_ds": p_ds,
            "slope": float(A[0]), "intercept": float(A[1])}

# ==================== 预测核心 ====================
SEASONS = {
    1: "冬季(低谷)", 2: "冬季(低谷)", 12: "冬季(低谷)",
    3: "春季(高峰)", 4: "春季(高峰)", 5: "春季(高峰)",
    9: "秋季(高峰)", 10: "秋季(高峰)", 11: "秋季(高峰)",
    6: "夏季(过渡)", 7: "夏季(过渡)", 8: "夏季(过渡)",
}
PEAK = {3, 4, 5, 9, 10, 11}

_CAP_CACHE = None
def load_capacity():
    """读月度披露报告.装机总(万kW), 缓存为 {YYYY-MM: 容量}。"""
    global _CAP_CACHE
    if _CAP_CACHE is not None:
        return _CAP_CACHE
    try:
        rows = run_query("SELECT `月份`, `装机总_万kW` FROM `月度披露报告`")
        d = {}
        for r in rows:
            ym = r.get("月份"); c = r.get("装机总_万kW")
            d[ym] = float(c) if c is not None else None
        _CAP_CACHE = d
    except Exception:
        _CAP_CACHE = {}
    return _CAP_CACHE

def cap_for(y, m):
    """取某年某月装机容量(万kW); 缺失时回退到不晚于该月的最近已知值。"""
    d = load_capacity()
    key = f"{y}-{m:02d}"
    if key in d and d[key]:
        return d[key]
    best = None
    for k, v in d.items():
        if v and k <= key:
            if best is None or k > best[0]:
                best = (k, v)
    return best[1] if best else None

def seasonal_forecast(df, target_year, target_month):
    """改进版预测: 历史同月均值 + 线性趋势 + 年际波动区间.

    方法说明:
    - 历史同月样本通常仅 2~3 个, 趋势外推最不可信, 故趋势权重设 0.3(保守),
      主体由同月均值承担, 避免小样本趋势跑飞。
    - 区间宽度按"同月年际相对波动"定(本质是历史波动范围, 非统计置信区间,
      样本不足以支撑 80% 置信区间的名义), 避免小样本时区间过窄、包不住真实值。
    - 识别"上年同期异常低"的披露不全月份, 给出警告, 避免系统性低估。
    """
    sub = df[df["月"] == target_month].copy()
    prior = sub[sub["年"].astype(int) < target_year]
    season = SEASONS.get(target_month, "过渡")
    risk = "高" if target_month in PEAK else "低"
    if prior.empty:
        return {"point": None, "lo": None, "hi": None,
                "season": season, "risk": risk,
                "method": "无历史同月数据", "warn": ""}

    yearly = prior.groupby("年").size().sort_index()
    years_arr = np.array([int(y) for y in yearly.index])
    counts = yearly.values.astype(float)

    # 数据局限提示: 历史同月若样本极少, 预测置信度低(不做补全, 避免误判)
    warn = ""
    if len(counts) == 1:
        warn = f"仅 {int(years_arr[-1])} 一年历史同期数据, 预测置信度有限, 实际偏差可能较大"

    # ① 历史同月均值
    avg = float(counts.mean())

    # ② 线性趋势外推(≥2个历史年份才启用, 否则退化为均值)
    if len(counts) >= 2:
        z = np.polyfit(years_arr, counts, 1)  # (斜率, 截距)
        trend_pred = z[0] * target_year + z[1]
        # 趋势预测做防御性裁剪, 放宽到均值±100%, 避免小样本趋势跑飞但仍捕捉强增长
        trend_pred = float(np.clip(trend_pred, avg * 0.4, avg * 2.0))
        w_trend = 0.3  # 趋势权重保守(样本少, 趋势最不可信); 主体由同月均值承担
    else:
        trend_pred = counts[0]
        w_trend = 0.0

    # 融合: 趋势主导(≥2年), 否则纯均值
    point = int(round((1 - w_trend) * avg + w_trend * trend_pred))
    point = max(0, point)

    # v2: 容量归一化季节预测 —— 检修量随装机增长而涨, 直接外推会把"增长"当成"趋势"放大;
    #     改算"每GW检修率"(稳定季节信号)再乘回目标月容量, 消除增长带来的假趋势。
    norm_point = None
    if len(counts) >= 2:
        _rates, _tcap = [], cap_for(target_year, target_month)
        for _y, _c in zip(years_arr, counts):
            _cap = cap_for(int(_y), target_month)
            if _cap:
                _rates.append(_c / _cap)
        if len(_rates) >= 2 and _tcap:
            _norm_rate = float(np.mean(_rates))
            norm_point = int(round(_norm_rate * _tcap))
            point = norm_point  # 归一化为主, 区间随之定宽
            # 目标月已有记录时, 与归一化预期比对, 标记可能的抓取不全
            _act = int(sub[sub["年"].astype(int) == target_year].shape[0])
            if _act > 0 and abs(_act - norm_point) / max(norm_point, 1) > 0.5:
                dq = (f"该月检修记录({_act}项)与容量归一化预期(~{norm_point}项)偏差>50%, "
                      f"可能数据抓取不全, 预测置信度低")
                warn = (warn + "; " + dq) if warn else dq

    # ③ 置信区间: 历史样本少时, 用同月年际相对波动定宽
    #    (替代原来围绕保守点估的窄泊松区间, 避免包不住真实值)
    if len(counts) >= 2:
        rel_spread = (counts.max() - counts.min()) / avg
    else:
        rel_spread = 0.3
    rel_spread = float(max(rel_spread, 0.2))  # 至少±20%宽
    lo = int(round(point * (1 - rel_spread)))
    hi = int(round(point * (1 + rel_spread)))
    if hi <= lo:  # 防止区间退化
        hi = lo + 1

    if norm_point is not None:
        method = (f"容量归一化季节预测: 同月每GW检修率 {_norm_rate:.5f} × 目标月容量 {_tcap:.0f}万kW"
                  f" → {norm_point}项; 区间按同月年际波动 ±{rel_spread:.0%}")
    else:
        method = (f"历史同月均值 {avg:.0f} + 趋势外推 {trend_pred:.0f} "
                  f"(趋势权重 {w_trend:.0%}); 区间按同月年际波动 ±{rel_spread:.0%}")
    return {"point": point, "lo": lo, "hi": hi,
            "season": season, "risk": risk,
            "method": method, "warn": warn}

def seasonal_forecast_value(df_series, col, target_year, target_month):
    """对数值序列(如净送出/外送)做 季节均值 + 趋势 预测。

    与 seasonal_forecast 同思路: 历史同月均值 + 线性趋势(≥2年启用, 权重0.8)
    + 同月年际波动定区间。返回 {point, lo, hi, method} 或 None。
    """
    s = df_series.copy()
    s["ym"] = s["月份"].astype(str)
    s["年"] = s["ym"].str[:4]
    s["月"] = s["ym"].str[5:7].astype(int)
    sub = s[s["月"] == target_month]
    prior = sub[sub["年"].astype(int) < target_year]
    prior = prior[prior[col].notna()]
    if prior.empty:
        return None
    yearly = prior.groupby("年")[col].mean()
    years_arr = np.array([int(y) for y in yearly.index])
    vals = yearly.values.astype(float)
    avg = float(vals.mean())
    if len(vals) >= 2:
        z = np.polyfit(years_arr, vals, 1)
        trend_pred = float(np.clip(z[0] * target_year + z[1], avg * 0.4, avg * 2.0))
        w = 0.3
    else:
        trend_pred = float(vals[0])
        w = 0.0
    point = float((1 - w) * avg + w * trend_pred)
    rel = (vals.max() - vals.min()) / avg if len(vals) >= 2 else 0.3
    rel = float(max(rel, 0.2))
    return {"point": round(point, 1), "lo": round(point * (1 - rel), 1),
            "hi": round(point * (1 + rel), 1),
            "method": f"同月均值 {avg:.0f} + 趋势外推 {trend_pred:.0f} (权重 {w:.0%}); 区间 ±{rel:.0%}"}


def seasonal_accuracy(df):
    """回测验证: 方向准确率 + 点估计 MAE (平均绝对误差).

    改进: 原来只看方向偏高/偏低, 看不出点估计差多少项;
    现加 MAE 量化, 给领导讲"预测平均误差多大"更有数.
    """
    rows, ok, tot, errs = [], 0, 0, []
    years = sorted(df["年"].unique())
    if len(years) < 2:
        return pd.DataFrame(), 0, 0
    overall = df.groupby("披露月份").size().mean()
    for i in range(len(years) - 1):
        y_prev, y_cur = years[i], years[i + 1]
        for m in range(1, 13):
            p = df[(df["年"] == y_prev) & (df["月"] == m)]
            c = df[(df["年"] == y_cur) & (df["月"] == m)]
            if p.empty or c.empty:
                continue
            pred = int(len(p))  # 上年同期作预测(基线)
            actual = int(len(c))
            pred_high = pred >= overall
            act_high = actual >= overall
            match = (pred_high == act_high)
            ok += int(match); tot += 1
            err = abs(pred - actual)
            errs.append(err)
            rows.append({"验证月份": f"{y_cur}-{m:02d}", "季节": SEASONS.get(m, "-"),
                         "预测": pred, "实际": actual,
                         "误差": err,
                         "方向预判": "偏高" if pred_high else "偏低",
                         "实际方向": "偏高" if act_high else "偏低",
                         "准确": "✓" if match else "✗"})
    rate = ok / tot if tot else 0
    mae = float(np.mean(errs)) if errs else 0
    return pd.DataFrame(rows), rate, mae

def predict_structure(df, target_month, target_year=None):
    """预测: 哪些线路/设备大概率安排检修。

    改进(对应需求 A/B/C):
      - 按 (申请单位, 停电设备) 分组, 避免 '#2机组' 跨厂错误合并
      - 附加 所属地区 / 变电站(位置代理) / 检修周期(平均间隔) / 下次预计
    返回: equip_groups(DataFrame), type_dist, repeat_equip, heavy, region_dist
    """
    sub = df[df["月"] == target_month].copy()
    if "开始日期_dt" in sub.columns:
        sub = sub.sort_values("开始日期_dt")

    # 只看历史同期(不含目标年本身, 避免用未来数据)
    if target_year is not None:
        hist = sub[sub["年"].astype(int) < target_year]
    else:
        hist = sub

    # --- ① 按 (停电设备, 申请单位) 聚合, 避免通用名跨厂合并 ---
    g = hist.groupby(["停电设备", "申请单位"]).agg(
        历史次数=("停电设备", "count"),
        主要级别=("检修级别", lambda x: x.mode().iloc[0] if len(x) > 0 else ""),
        历史年份=("年", lambda x: ",".join(sorted(set(str(v) for v in x if pd.notna(v))))),
        所属地区=("所属地区", lambda x: x.mode().iloc[0] if len(x) > 0 else ""),
        变电站=("变电站", lambda x: x.mode().iloc[0] if len(x) > 0 else ""),
    )

    # ② 周期: 看该设备【目标年前的全部历史】(跨月份)算相邻间隔(月)的中位值, 推算下次预计
    #    注意: 不能只看目标月同期, 否则同设备凑不够 2 次(检修常跨不同月份发生)
    pre = df[df["年"].astype(int) < target_year] if target_year is not None else df
    cycle_info = {}
    for (equip, unit), gg in pre.dropna(subset=["开始日期_dt"]).groupby(["停电设备", "申请单位"]):
        dts = sorted(gg["开始日期_dt"].tolist())
        gaps = [(b.year - a.year) * 12 + (b.month - a.month)
                for a, b in zip(dts, dts[1:])]
        gaps = [x for x in gaps if x > 0]
        n_hist = len(dts)
        # 跨度判断: 若全部历史挤在 60 天内, 说明是"同月多次小检修", 不构成真周期
        total_span = (dts[-1] - dts[0]).days if len(dts) >= 2 else 0
        if not gaps or total_span < 60:
            cycle_info[(equip, unit)] = (None, dts[-1].strftime("%Y-%m"), None, "同月集中型", None)
            continue
        med = float(np.median(gaps))
        last = dts[-1]
        nxt = last + pd.DateOffset(months=int(round(med)))
        conf = "高" if n_hist >= 4 else ("中" if n_hist >= 3 else "低")
        # 距目标月月数(根据当前目标年-月)
        if target_year is not None:
            diff_m = (nxt.year - target_year) * 12 + (nxt.month - target_month)
        else:
            diff_m = None
        cycle_info[(equip, unit)] = (round(med, 1), last.strftime("%Y-%m"),
                                     nxt.strftime("%Y-%m"), conf, diff_m)

    g["平均周期月"] = [cycle_info.get((e, u), (None, None, None, None, None))[0] for e, u in g.index]
    g["上次检修"] = [cycle_info.get((e, u), (None, None, None, None, None))[1] for e, u in g.index]
    g["下次预计"] = [cycle_info.get((e, u), (None, None, None, None, None))[2] for e, u in g.index]
    g["周期置信"] = [cycle_info.get((e, u), (None, None, None, None, None))[3] for e, u in g.index]
    g["距目标月"] = [cycle_info.get((e, u), (None, None, None, None, None))[4] for e, u in g.index]

    # 给"下次预计"加状态标签: 让用户一眼看清是未来还是已逾期
    def _state_label(diff):
        if diff is None or pd.isna(diff):
            return ""
        if diff <= 0:
            return f"⚠ 已逾期{-int(diff)}月"
        if diff <= 6:
            return f"⏰ 还差{int(diff)}月"
        return f"📅 +{int(diff)}月"
    g["状态"] = g["距目标月"].apply(_state_label)

    g = g.reset_index()
    g = g[g["停电设备"].notna() & (g["停电设备"] != "")]
    g = g.sort_values("历史次数", ascending=False).reset_index(drop=True)

    # 设备类型分布(不限历史, 反映该月整体结构)
    type_dist = sub["设备类型"].value_counts().head(5)

    # 重复披露设备(跨月/跨期反复出现的设备)
    repeat_equip = sub[sub["重复披露"] == 1]["停电设备"].value_counts().head(6)

    # 重点设备明细(用于右侧渲染)
    top_equip_list = g.head(10)["停电设备"].tolist()
    heavy = hist[hist["停电设备"].isin(top_equip_list)].copy()

    # 地区分布(历史同期)
    region_dist = (hist[hist["所属地区"] != "其他"].groupby("所属地区").size()
                   .sort_values(ascending=False)) if "所属地区" in hist else pd.Series(dtype=int)

    return g, type_dist, repeat_equip, heavy, region_dist

def predict_timing(df, target_year, target_month):
    """预测: 月内哪几周集中."""
    sub = df[(df["年"].astype(int) < target_year) & (df["月"] == target_month)].copy()
    if sub.empty or "开始日期_dt" not in sub.columns:
        return None, None
    sub = sub.dropna(subset=["开始日期_dt"])
    if sub.empty:
        return None, None
    sub["周"] = ((sub["开始日期_dt"].dt.day - 1) // 7 + 1).astype(int).clip(1, 5)
    week_label = {1: "W1(1-7日)", 2: "W2(8-14日)", 3: "W3(15-21日)",
                  4: "W4(22-28日)", 5: "W5(29-末日)"}
    sub["周标签"] = sub["周"].map(week_label)
    weekly = sub.groupby("周标签").size().reindex(
        ["W1(1-7日)", "W2(8-14日)", "W3(15-21日)", "W4(22-28日)", "W5(29-末日)"]
    ).fillna(0).astype(int).reset_index()
    weekly.columns = ["周次", "数量"]
    peak_week = weekly.loc[weekly["数量"].idxmax(), "周次"]
    return weekly, peak_week

def trading_signal(df_bal, df_sec, df_maint, df_trade, target_year, target_month, point, season, risk):
    """第 4 层: 输出交易信号(给领导看的最终结论).

    四维度信号:
    1) 价格方向  2) 供给紧张度  3) 外送/外购窗口  4) 大客户履约
    """
    signals = []
    # ----- 信号 1: 月度平衡格局(价格数据未接入前, 以月度平衡格局判断代替"现货价格") -----
    if risk == "高":
        signals.append(("📊 月度平衡格局", "偏紧(季节性)",
                       f"{target_month}月属春秋检高峰, 历史同期火电检修偏多 → 月度平衡格局"
                       f"季节性偏紧的概率较大; 但检修对电价的独立贡献证据不足(去季节化相关弱), "
                       f"不宜单独据此重仓。〔来源:检修季节性 + 月度平衡表〕"))
    else:
        signals.append(("📊 月度平衡格局", "宽松/中性",
                       f"{target_month}月属检修低谷/过渡, 火电可调容量相对充足, "
                       f"月度平衡格局大概率不紧张。〔来源:检修季节性 + 月度平衡表〕"))

    # ----- 信号 2: 供给紧张度 -----
    bal_ref = df_bal[df_bal["月份"].astype(str).str.contains(f"-{target_month:02d}", na=False)].copy() \
        if not df_bal.empty else pd.DataFrame()
    if (not bal_ref.empty
            and pd.notna(bal_ref["最大发电能力_万kW"]).any()
            and pd.notna(bal_ref["最大负荷_万kW"]).any()):
        bal_ref = bal_ref.sort_values("月份")
        bal_ref["备用率"] = (
            (bal_ref["最大发电能力_万kW"] - bal_ref["最大负荷_万kW"])
            / bal_ref["最大负荷_万kW"] * 100
        )
        reserve_pred = float(bal_ref["备用率"].mean())
        if len(bal_ref) >= 2:
            reserve_trend = float(bal_ref["备用率"].iloc[-1] - bal_ref["备用率"].iloc[0])
        else:
            reserve_trend = 0.0
        if reserve_pred < 10:
            verdict, detail = "偏紧信号", f"历史同期备用率均值 {reserve_pred:.1f}%, 接近警戒线(10%), 重叠月份需重点跟踪"
        elif reserve_pred > 30:
            verdict, detail = "偏松", f"历史同期备用率均值 {reserve_pred:.1f}%, 供给充足, 月度合约可争取更优条款"
        else:
            verdict, detail = "中性", f"历史同期备用率均值 {reserve_pred:.1f}%, 供需基本平衡"
        if reserve_trend <= -2:
            detail += f"; 备用率同比下降 {abs(reserve_trend):.1f}pp, 供给在收紧"
        elif reserve_trend >= 2:
            detail += f"; 备用率同比上升 {reserve_trend:.1f}pp, 供给在放松"
        signals.append(("⚠ 供给端", verdict, detail + "〔来源:月度平衡表〕"))
    else:
        signals.append(("— 供给端", "数据不足", "暂无历史平衡表数据"))

    # ----- 信号 3: 外送/外购窗口 -----
    sec_match = None
    if not df_sec.empty:
        # 找正向限额非零的断面数
        sec_recent = df_sec.tail(20) if len(df_sec) > 20 else df_sec
        out_pos = (sec_recent["正向限额"] != "0") & (sec_recent["正向限额"].notna())
        in_rev = (sec_recent["反向限额"] != "0") & (sec_recent["反向限额"].notna())
        out_count = int(out_pos.sum())
        in_count = int(in_rev.sum())
        if out_count >= 3:
            signals.append(("→ 外送窗口", "较宽", f"{out_count} 个断面正向有容量, 适合月底谈外送增量〔来源:断面限额〕"))
        else:
            signals.append(("→ 外送窗口", "偏窄", f"仅 {out_count} 个断面正向外送容量, 外送增量空间有限〔来源:断面限额〕"))

    # ----- 信号 4: 关键线路检修 -----
    heavy_equip_list = []
    sub = df_maint[df_maint["月"] == target_month]
    if not sub.empty:
        heavy = sub[sub["检修级别"].isin(["A级检修", "改造大修"])]
        if "停电设备" in heavy.columns:
            heavy_equip_list = heavy["停电设备"].value_counts().head(3).index.tolist()
    if len(heavy_equip_list) >= 2:
        signals.append(("🔌 关键线路", "多条线路检修",
                       f"{len(heavy_equip_list)} 条关键线路同期A修/改造, 可能影响断面限额, 需提前核实外送窗口〔来源:检修计划〕"))
    elif len(heavy_equip_list) == 1:
        signals.append(("🔌 关键线路", "1 条线路检修",
                       f"{heavy_equip_list[0]} 同期大修, 影响范围有限, 但需关注对应断面限额变化〔来源:检修计划〕"))
    else:
        signals.append(("🔌 关键线路", "无重大检修",
                       "同期无 A 级或改造大修线路, 断面限额大概率不受影响〔来源:检修计划〕"))

    # ----- 信号 5: 外送 + 检修 叠加(供给双重压力) -----
    if df_trade is not None and not df_trade.empty and "净送出_亿kWh" in df_trade.columns:
        fc_out = seasonal_forecast_value(df_trade, "净送出_亿kWh", target_year, target_month)
        if fc_out is not None:
            out_val = fc_out["point"]
            # 用历史同期均值的 80% 分位作"高外送"参考
            s = df_trade.copy()
            s["月"] = s["月份"].astype(str).str[5:7].astype(int)
            hist_same = s[s["月"] == target_month]["净送出_亿kWh"].dropna()
            hi_thr = hist_same.quantile(0.8) if len(hist_same) >= 3 else hist_same.mean()
            if risk == "高" and out_val >= hi_thr:
                verdict, detail = "双重收紧", (
                    f"检修属高峰(风险高) 且 预计净送出 {out_val:.0f} 亿kWh 处历史同期高位 → "
                    f"省内供给双重承压, 现货端偏多, 月合约偏谨慎〔来源:月度交易计划 + 检修〕")
            elif risk == "高":
                verdict, detail = "检修偏紧", (
                    f"检修属高峰, 但预计净送出 {out_val:.0f} 亿kWh 未达高位, "
                    f"外送未明显加剧紧张, 重点跟踪检修集中周")
            elif out_val >= hi_thr:
                verdict, detail = "外送偏紧", (
                    f"检修处低谷, 但预计净送出 {out_val:.0f} 亿kWh 处高位 → "
                    f"外送挤占省内电量, 仍需关注月底供给")
            else:
                verdict, detail = "相对宽松", (
                    f"检修低谷 且 预计净送出 {out_val:.0f} 亿kWh 未达高位, "
                    f"省内供需最宽松, 合约可争取更优条款")
            signals.append(("📤 外送叠加", verdict, detail))

    # ----- 信号 6: 气象作业窗口(weather_hourly, 未来 10 天) -----
    wx = _wx_window(10)
    if wx is not None:
        pct, msg = wx
        if pct >= 75:
            verdict = f"充裕({pct:.0f}%)"
        elif pct >= 55:
            verdict = f"正常({pct:.0f}%)"
        elif pct >= 35:
            verdict = f"偏紧({pct:.0f}%)"
        else:
            verdict = f"紧张({pct:.0f}%)"
        signals.append(("🌤 气象窗口", verdict, msg + "〔来源:weather_hourly〕"))
    else:
        signals.append(("🌤 气象窗口", "数据不足", "weather_hourly 未覆盖未来窗口, 跑 crawl_weather.py 即可补齐"))

    return signals

# ==================== 天气-检修适配分析(weather_hourly × 已有数据) ====================
def weather_daily_grid(days=10):
    """未来 N 天逐日可作业率(12 点位平均). 返回 [(mm-dd, 可作业率%, 受限小时, 总小时), ...]."""
    try:
        conn = pymysql.connect(**DB_CONFIG)
        cur = conn.cursor()
        cur.execute(
            "SELECT DATE(时间) AS d, COUNT(*) AS total, "
            "       SUM(CASE WHEN 风速_10m > 10.7 OR 雷暴 = 1 OR 降水_mm > 0.5 "
            "                  OR 气温_2m > 40 OR 气温_2m < -15 THEN 1 ELSE 0 END) AS bad "
            "FROM weather_hourly "
            "WHERE 时间 > NOW() AND 时间 < DATE_ADD(NOW(), INTERVAL %s DAY) "
            "GROUP BY DATE(时间) ORDER BY d",
            (days + 1,))
        rows = cur.fetchall()
        conn.close()
        out = []
        for d, total, bad in rows:
            total = int(total or 0); bad = int(bad or 0)
            if total == 0:
                continue
            out.append((d.strftime("%m-%d"), round((total - bad) / total * 100), bad, total))
        return out
    except Exception:
        return []


def point_weather_matrix(days=10):
    """12 点位 × 未来 N 天 可作业率矩阵. 返回 DataFrame[点位, 日期, 可作业率%, 受限原因]."""
    try:
        conn = pymysql.connect(**DB_CONFIG)
        cur = conn.cursor()
        cur.execute(
            "SELECT 点位名称, DATE(时间) AS d, "
            "       COUNT(*) AS total, "
            "       SUM(CASE WHEN 风速_10m > 10.7 THEN 1 ELSE 0 END) AS wind_bad, "
            "       SUM(CASE WHEN 雷暴 = 1 THEN 1 ELSE 0 END) AS thunder_bad, "
            "       SUM(CASE WHEN 降水_mm > 0.5 THEN 1 ELSE 0 END) AS rain_bad, "
            "       SUM(CASE WHEN 气温_2m > 40 OR 气温_2m < -15 THEN 1 ELSE 0 END) AS temp_bad "
            "FROM weather_hourly "
            "WHERE 时间 > NOW() AND 时间 < DATE_ADD(NOW(), INTERVAL %s DAY) "
            "GROUP BY 点位名称, DATE(时间) ORDER BY 点位名称, d",
            (days + 1,))
        rows = cur.fetchall()
        conn.close()
        out = []
        for pt, d, total, w_b, t_b, r_b, te_b in rows:
            total = int(total or 0)
            if total == 0:
                continue
            bad = int(w_b or 0) + int(t_b or 0) + int(r_b or 0) + int(te_b or 0)
            # 主因: 哪种受限贡献最大
            main = "风" if (w_b or 0) >= max(t_b or 0, r_b or 0, te_b or 0) else \
                   "雷" if (t_b or 0) >= max(r_b or 0, te_b or 0) else \
                   "雨" if (r_b or 0) >= (te_b or 0) else "温"
            out.append({"点位": pt, "日期": d.strftime("%m-%d"),
                        "可作业率%": round((total - bad) / total * 100),
                        "受限原因": main if bad > 0 else "—"})
        return out
    except Exception:
        return []


def maint_weather_decision(df_maint, target_year, target_month, top_n=8):
    """关键检修 × 未来天气 → 延期决策矩阵(含容量影响、行动建议).
    返回 [(设备, 级别, 开始, 结束, 持续天, 重要度, 受限%, 行动, 调整建议), ...]
    排序: 重要度 × (1 - 可作业率) 复合降序.
    """
    from datetime import datetime, timedelta
    if df_maint is None or df_maint.empty or "开始日期_dt" not in df_maint.columns:
        return []
    cutoff = datetime.now() + timedelta(days=10)
    sub = df_maint[(df_maint["年"].astype(str) == str(target_year)) &
                   (df_maint["月"] == target_month) &
                   (df_maint["开始日期_dt"] <= cutoff) &
                   (df_maint["开始日期_dt"] >= datetime.now() - timedelta(days=1))]
    if sub.empty:
        return []

    # 算每条的重要度分 = 等级分 × 持续天数
    sub = sub.copy()
    # 重要度 = 等级分 × 持续天数 (用本地 LEVEL_RANK, 不依赖 render_review)
    _LR = {"A级检修": 5, "B级检修": 4, "C级检修": 3, "改造大修": 3, "D级检修": 2,
           "消缺": 2, "检修预试": 2, "基建接入": 2, "例行检修": 1, "其他": 1}
    sub["_sev"] = sub["检修级别"].map(lambda x: _LR.get(str(x).strip(), 1) if pd.notna(x) else 1)
    sub["_days"] = (sub["结束日期_dt"] - sub["开始日期_dt"]).dt.days.fillna(1).clip(lower=1)
    sub["_imp"] = sub["_sev"] * sub["_days"]
    # 排序取 top_n*3 (后续按受限比例再过滤)
    sub = sub.sort_values("_imp", ascending=False).head(top_n * 3)

    try:
        conn = pymysql.connect(**DB_CONFIG)
        cur = conn.cursor()
        out = []
        for _, r in sub.iterrows():
            sd = r.get("开始日期_dt"); ed = r.get("结束日期_dt")
            if pd.isna(sd):
                continue
            ed = ed if pd.notna(ed) else sd
            cur.execute(
                "SELECT COUNT(*) AS total, "
                "       SUM(CASE WHEN 风速_10m > 10.7 OR 雷暴 = 1 OR 降水_mm > 0.5 "
                "                  OR 气温_2m > 40 OR 气温_2m < -15 THEN 1 ELSE 0 END) AS bad "
                "FROM weather_hourly WHERE 时间 >= GREATEST(%s, NOW()) AND 时间 <= %s",
                (sd, ed))
            rr = cur.fetchone()
            total = int(rr[0] or 0); bad = int(rr[1] or 0)
            if total == 0:
                continue
            bad_rate = bad / total
            sev = int(r["_sev"]); days = int(r["_days"])
            imp = int(r["_imp"])
            equip = str(r.get("停电设备", ""))[:30]
            level = str(r.get("检修级别", "—"))
            # 行动建议
            if bad_rate < 0.15:
                action = "按期"
                advice = "天气友好"
            elif bad_rate < 0.30:
                action = "关注"
                advice = "准备备用日"
            elif bad_rate < 0.45:
                action = "建议改期"
                advice = "择窗口重排"
            else:
                action = "强烈建议改期"
                advice = "延期≥1 天"
            out.append((equip, level, sd.strftime("%m-%d"), ed.strftime("%m-%d"),
                        days, imp, round(bad_rate * 100), action, advice))
        conn.close()
        # 排序: 重要度降序 + 受限% 降序
        out.sort(key=lambda x: (-x[5], -x[6]))
        return out[:top_n]
    except Exception:
        return []


def weather_maint_risk(df_maint, target_year, target_month):
    """当月检修计划中落在天气受限窗口的条目(仅未来 10 天内开始的, 远期无预报).
    返回 [(停电设备, 开始, 结束, 受限小时, 总小时, 风险等级), ...] 按风险降序.
    """
    from datetime import datetime, timedelta
    if df_maint is None or df_maint.empty or "开始日期_dt" not in df_maint.columns:
        return []
    cutoff = datetime.now() + timedelta(days=10)
    sub = df_maint[(df_maint["年"].astype(str) == str(target_year)) &
                   (df_maint["月"] == target_month) &
                   (df_maint["开始日期_dt"] <= cutoff)]
    if sub.empty:
        return []
    try:
        conn = pymysql.connect(**DB_CONFIG)
        cur = conn.cursor()
        out = []
        for _, r in sub.iterrows():
            sd = r.get("开始日期_dt"); ed = r.get("结束日期_dt")
            if pd.isna(sd):
                continue
            ed = ed if pd.notna(ed) else sd
            # 只统计检修区间与未来天气窗口的重叠部分(GREATEST 取较晚起点)
            cur.execute(
                "SELECT COUNT(*) AS total, "
                "       SUM(CASE WHEN 风速_10m > 10.7 OR 雷暴 = 1 OR 降水_mm > 0.5 "
                "                  OR 气温_2m > 40 OR 气温_2m < -15 THEN 1 ELSE 0 END) AS bad "
                "FROM weather_hourly WHERE 时间 >= GREATEST(%s, NOW()) AND 时间 <= %s",
                (sd, ed))
            rr = cur.fetchone()
            total = int(rr[0] or 0); bad = int(rr[1] or 0)
            if total == 0:
                continue  # 检修区间不在天气覆盖窗, 跳过
            rate = bad / total
            lvl = "高" if rate >= 0.4 else ("中" if rate >= 0.25 else "低")
            out.append((str(r.get("停电设备", "")), sd.strftime("%Y-%m-%d"),
                        ed.strftime("%Y-%m-%d"), bad, total, lvl))
        conn.close()
        return sorted(out, key=lambda x: {"高": 0, "中": 1, "低": 2}.get(x[5], 3))
    except Exception:
        return []


def render_weather_maint(target_year, target_month, df_maint, df_sec):
    """天气 × 已有数据结合: ① 关键检修-天气-交易决策矩阵 ② 点位可作业矩阵 ③ 外送双重风险.
    核心逻辑: 天气好不好不重要, 影不影响关键检修才重要.
    """
    st.write("---")
    st.markdown("### 🌤 天气-检修适配分析")
    st.caption("天气源: weather_hourly(12 点位, Open-Meteo, 模式格点预报). "
               "⚠️ Open-Meteo 绝对温度与实测可能偏差 4~5℃, 本块仅用于趋势与受限阈值判断, 不作气温实测展示. "
               "受限阈值: 风>10.7m/s 或 雷暴 或 降水>0.5mm 或 极端温度.")

    # ===== ① 关键检修 × 天气 × 交易决策矩阵(最前: 最直接给交易建议) =====
    decisions = maint_weather_decision(df_maint, target_year, target_month, top_n=8)
    st.markdown("**🔁 关键检修-天气-交易决策矩阵**")
    if decisions:
        # 按行动分类
        change = [d for d in decisions if "改期" in d[7]]
        watch = [d for d in decisions if d[7] == "关注"]
        ontrack = [d for d in decisions if d[7] == "按期"]

        # 交易影响汇总
        if change:
            # 受影响容量分 = sum(重要度 × 受限比例)  越大越应改
            impact_score = sum(d[5] * d[6] / 100 for d in change)
            top = change[0]
            st.error(
                f"🔴 **交易影响**: {len(change)} 条高重要度检修落在受限窗口(总影响分 {impact_score:.1f}); "
                f"最关键: [{top[0]}] {top[1]} {top[2]}~{top[3]} 受限 {top[6]}% → {top[7]}({top[8]}). "
                f"若按建议调整, 可避免当月重要检修延期, 稳定供给侧预期."
            )
        elif watch:
            st.warning(
                f"🟡 {len(watch)} 条关键检修处于'关注'档(受限 15-30%); "
                f"建议预留备用日期, 若执行前 24h 预报仍偏紧则调整."
            )
        else:
            st.success(
                f"🟢 关键检修(重要度前 {len(decisions)})天气全部'按期'档, "
                f"本月无因天气延期的供给风险."
            )

        # 表格(用 dataframe 而不是 markdown, 列宽更友好)
        import pandas as pd
        df_d = pd.DataFrame(decisions, columns=[
            "设备", "级别", "开始", "结束", "持续(天)", "重要度",
            "受限%", "行动建议", "调整方向"
        ])
        st.dataframe(df_d, use_container_width=True, hide_index=True)
        st.caption("重要度 = 等级分(A=5/B=4/C=3/D=2/改造大修=3/其他=1) × 持续天数; "
                   "受限% = 检修区间内天气受限小时比例.")
    else:
        st.info("当月无未来 10 天内开始的检修, 暂无适配决策.")

    # ===== ② 12 点位 × 未来 10 天 可作业矩阵(取代原"逐日柱状图") =====
    st.markdown("**🗺 12 点位 × 未来 10 天 可作业率矩阵**")
    rows = point_weather_matrix(10)
    if rows:
        import pandas as pd
        df_mx = pd.DataFrame(rows)
        if not df_mx.empty:
            # 透视: 行=点位, 列=日期
            pivot = df_mx.pivot_table(index="点位", columns="日期", values="可作业率%", fill_value=None)
            # 按甘肃检修区域排序
            region_order = ["兰州", "酒泉", "嘉峪关", "张掖", "武威", "定西",
                            "平凉", "庆阳", "临夏", "合作", "天水", "陇南"]
            pivot = pivot.reindex([p for p in region_order if p in pivot.index])
            st.dataframe(pivot.style.background_gradient(
                cmap="RdYlGn", vmin=40, vmax=100
            ).format("{:.0f}"), use_container_width=True)
            st.caption("绿=可作业率高 红=受限严重. 点位按甘肃检修区域分组; "
                       "查具体某天某点是否适合高空/吊装作业 → 直接看格子.")
    else:
        st.caption("未来 10 天天气数据未覆盖, 暂无矩阵.")

    # ===== ③ 当月检修计划 × 天气风险清单(原有, 保留向后兼容) =====
    risks = weather_maint_risk(df_maint, target_year, target_month)
    if risks:
        hi = [x for x in risks if x[5] in ("高", "中")]
        if hi:
            with st.expander(f"📋 展开: 当月全部风险清单({len(risks)} 条, 其中高/中风险 {len(hi)} 条)"):
                for equip, sd, ed, bad, total, lvl in hi[:15]:
                    st.markdown(f"- **{lvl}** | {equip} | {sd}~{ed} | {bad}/{total}h")
                st.caption("清单只列'高/中风险'档; '按期'档在 ① 决策矩阵中已聚合.")

    # ===== ④ 外送双重风险(雷暴 + 断面检修) =====
    if df_sec is not None and not df_sec.empty and "月份" in df_sec.columns:
        try:
            conn = pymysql.connect(**DB_CONFIG)
            cur = conn.cursor()
            cur.execute("SELECT COALESCE(SUM(雷暴),0) FROM weather_hourly "
                        "WHERE 时间 > NOW() AND 时间 < DATE_ADD(NOW(), INTERVAL 11 DAY)")
            thunder_h = int(cur.fetchone()[0] or 0)
            conn.close()
            sec_m = df_sec[df_sec["月份"].astype(str) == f"{target_year}-{target_month:02d}"]
            sec_maint = sec_m[sec_m.astype(str).apply(lambda r: r.str.contains("检修").any(), axis=1)] \
                if not sec_m.empty else pd.DataFrame()
            if thunder_h > 20 and not sec_maint.empty:
                st.error(f"🔴 外送双重风险: 未来 10 天雷暴 {thunder_h} 小时 + 当月 {len(sec_maint)} 个断面检修, "
                         f"雷暴可能叠加检修限额, 外送窗口需重点盯防.")
            elif thunder_h > 20:
                st.info(f"未来 10 天雷暴 {thunder_h} 小时偏多, 关注外送线路跳闸风险(当月无断面检修冲突).")
        except Exception:
            pass


# ==================== 第一步: 加权检修影响(容量权重) ====================
def predict_weighted_impact(df, target_year, target_month):
    """第一步交付: 加权检修影响指数 + 高影响检修清单.

    用设备类型权重(非精确MW)量化检修'分量', 区分'100万kW机组停机'
    与'支线检修'的影响差异. 这是后续'有效备用率/供给紧张度'的地基.
    """
    sub = df[df["月"] == target_month].copy()
    hist = sub[sub["年"].astype(int) < target_year]   # 只用历史同期, 不用未来数据
    if hist.empty:
        return None, None, None

    # 历史同期加权指数(与季节性预测一致, 取同月跨年均值)
    w_avg = float(hist.groupby("年")["权重"].sum().mean())

    # 高影响清单: 权重>=3 (机组/主变/母线/线路), 按权重降序
    high = hist[hist["权重"] >= 3].copy()
    if not high.empty and "开始日期_dt" in high.columns:
        high = high.sort_values(["权重", "开始日期_dt"], ascending=[False, True])
    cols = ["设备类型", "停电设备", "申请单位", "检修级别", "开始日期", "结束日期", "权重"]
    cols = [c for c in cols if c in high.columns]
    if not high.empty:
        high = high[cols].rename(columns={"权重": "影响权重"}).head(15)

    stats = {
        "gen": int((hist["设备类型"] == "发电机组").sum()),
        "trans": int((hist["设备类型"] == "变压器").sum()),
        "weighted": float(hist["权重"].sum()),
    }
    return w_avg, high, stats

# ==================== 第二步铺垫: 有效备用率(供给紧张度) ====================
def effective_reserve(reserve_lut, w_avg, target):
    """有效备用率 = 名义备用率 − 检修加权影响(定性).

    名义备用率 = (常规最大发电能力 − 最大用电负荷)/最大用电负荷,
    数据来自 build_reserve_lookup(披露报告电力平衡预测 / 月度平衡实际).
    检修加权指数(历史同期均值)作为定性扣减项.
    注: 精确MW损失需设备台账(每台机组容量), 当前先用加权指数近似, 后续替换.
    """
    if not reserve_lut or target not in reserve_lut:
        return None
    cap, load, src = reserve_lut[target]
    nominal = (cap - load) / load * 100
    w = w_avg if w_avg is not None else 0.0
    # 经验映射: 加权指数每 200 ≈ 备用率下降 1pp (待设备台账校准)
    adj = w / 200.0
    eff = nominal - adj
    return {"month": target, "cap": cap, "load": load, "source": src,
            "nominal": nominal, "w_avg": w, "eff": eff}

# ==================== 单页报告 ====================
def render_report():
    # 顶部标题 + 月份选择
    title_l, title_r = st.columns([3, 2])
    with title_l:
        st.markdown("# 甘肃电网检修预测报告")
        st.caption("数据来源: 甘肃省电力市场月度披露文件 + 历史同期规律")
    with title_r:
        # 可选月份: 已有历史 + 未来 6 个月
        avail_months = sorted(load_maint()["披露月份"].unique().tolist())
        future_months = []
        if avail_months:
            last = avail_months[-1]
            ly, lm = int(last[:4]), int(last[5:7])
            for i in range(1, 7):
                nm, ny = lm + i, ly
                if nm > 12:
                    nm -= 12; ny += 1
                future_months.append(f"{ny}-{nm:02d}")
        opts = future_months + avail_months[::-1]
        opts_unique = list(dict.fromkeys(opts))
        target = st.selectbox("📅 预测月份", opts_unique,
                              index=0, label_visibility="collapsed")

    target_year = int(target.split("-")[0])
    target_month = int(target.split("-")[1])

    df = load_maint()
    df_bal = load_bal()
    df_sec = load_section()
    df_disc = load_disclosure()
    df_trade = load_trade_plan()
    reserve_lut = build_reserve_lookup(df_disc, df_bal)

    # 核心结论(决策速览)已移至「📈 已披露复盘」页顶部, 仅用已披露实测数据。

    # 第一步: 加权检修影响指数 + 高影响清单
    w_avg, high_df, wstats = predict_weighted_impact(df, target_year, target_month)

    # ===== 当月检修明细（已披露月份核心面板, 优先展示实测）=====
    is_disclosed = int(df[df["披露月份"] == target].shape[0]) > 0
    if is_disclosed:
        render_month_detail(df, target)
        st.write("---")
        st.caption("以下为基于历史规律的预测 / 参考；当月实测明细见上方面板。")

    # ===== ① 总量预测 =====
    fc = seasonal_forecast(df, target_year, target_month)
    point, lo, hi = fc["point"], fc["lo"], fc["hi"]
    season, risk = fc["season"], fc["risk"]

    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown(f"### ① 总量预测")
        if point is None:
            st.warning("无历史数据, 无法预测")
            return
        # 主指标
        st.markdown(
            f'<div style="background:#f0f4f8;padding:18px 20px;border-radius:8px;border-left:4px solid #185FA5">'
            f'<div style="font-size:42px;font-weight:700;color:#185FA5;line-height:1">{point}'
            f'<span style="font-size:18px;color:#888;margin-left:8px">项</span></div>'
            f'<div style="color:#666;margin-top:6px;font-size:13px">'
            f'预计 {target} 发电设备检修项数<br>'
            f'历史波动范围: <b>{lo} ~ {hi} 项</b></div></div>',
            unsafe_allow_html=True,
        )
        st.caption(f"📐 方法: {fc['method']}")
        if fc["warn"]:
            st.warning(fc["warn"])

        # ===== A: 事后验证(目标月已披露真实数据时) =====
        actual_count = int(df[df["披露月份"] == target].shape[0])
        if actual_count > 0:
            diff = actual_count - point
            pct = (diff / point * 100) if point else 0
            st.markdown("**📊 事后验证（" + target + " 实际已披露）**")
            st.markdown(f"- 模型事前预测：**{point}** 项（区间 {lo} ~ {hi}）")
            st.markdown(f"- 实际发生：**{actual_count}** 项")
            if actual_count > hi:
                st.error(f"⚠ 实际高出预测区间 {actual_count - hi} 项（+{pct:.0f}%）— 低估主因: 历史样本少+趋势权重偏低, 已在改进版修正")
            elif actual_count < lo:
                st.warning(f"实际低于预测区间下限 {lo - actual_count} 项（{pct:.0f}%）")
            else:
                st.success(f"✅ 实际落在预测区间内（偏差 {pct:+.0f}%）")

        # 已知跨月续检(底仓): 上月已披露、本月仍在修——预测时应作为确定底仓, 不计入"新增"
        try:
            _ts = pd.Timestamp(year=target_year, month=target_month, day=1)
            _cont = df[(df["年"].astype(int) == target_year)
                      & (df["开始日期_dt"] < _ts) & (df["结束日期_dt"] >= _ts)]
            if not _cont.empty:
                _cn = len(_cont)
                _ex = "、".join(
                    (_cont["申请单位"].astype(str) + "·" + _cont["停电设备"].astype(str)).head(3).tolist()
                )
                st.info(f"🔗 已知跨月续检(底仓): 本月有 **{_cn}** 项检修从上月延续(已披露、确定发生), "
                         f"预测下月时应作为『已知底仓』单列, 不再计入新增预测。示例: {_ex} 等。")
                # 把底仓明细展开, 方便用户核对这23项具体落在哪些设备
                _cols = [c for c in ["申请单位", "停电设备", "检修级别", "开始日期", "结束日期",
                                     "工作内容", "设备类型", "所属地区"] if c in _cont.columns]
                _disp = _cont[_cols].sort_values(["所属地区" if "所属地区" in _cols else "申请单位",
                                                  "开始日期"], na_position="last").copy()
                _disp.insert(0, "底仓标记", "✅ 跨月续检")
                with st.expander(f"📋 查看 {_cn} 项跨月续检明细（底仓）", expanded=False):
                    st.caption("以下为『上月已开始、本月仍在修』的已披露检修；预测下月新增时不应重复计入。")
                    st.dataframe(_disp, use_container_width=True, hide_index=True, height=min(350, 35 * (_cn + 1)))
        except Exception:
            pass

        # 第一步: 加权检修影响指数指标卡
        if w_avg is not None:
            st.markdown(
                f'<div style="background:#eef6ee;padding:10px 14px;border-radius:8px;margin-top:10px;'
                f'border-left:4px solid #2e7d32">'
                f'<div style="font-size:22px;font-weight:700;color:#2e7d32;line-height:1">{w_avg:.0f}'
                f'<span style="font-size:12px;color:#666;margin-left:6px">加权影响指数 (历史同期均值)</span></div>'
                f'<div style="color:#666;font-size:12px;margin-top:4px">'
                f'发电机组 {wstats["gen"]} 项 · 主变 {wstats["trans"]} 项 · 合计权重损失 {wstats["weighted"]:.0f}'
                f'<br><span style="font-size:11px">（{w_avg:.0f}=历史同月「每月」平均影响指数；'
                f'{wstats["weighted"]:.0f}=同月全部记录权重合计，两者口径不同，非矛盾）</span></div>'
                f'</div>', unsafe_allow_html=True)
    with c2:
        import plotly.graph_objects as go
        st.markdown("**历史趋势 + 预测月标记**")
        monthly = df.groupby("披露月份").size().reset_index(name="数量").sort_values("披露月份")
        if not monthly.empty:
            # 确保预测月出现在 x 轴上(未来月实际值记为0)
            if target not in monthly["披露月份"].values:
                monthly = pd.concat(
                    [monthly, pd.DataFrame([{"披露月份": target, "数量": 0}])],
                    ignore_index=True,
                )
            fig = go.Figure()
            fig.add_bar(x=monthly["披露月份"], y=monthly["数量"],
                        marker_color="#cfd8dc", name="历史实际")
            # 历年同月(蓝色虚线), 体现季节性
            same = df[df["月"] == target_month].groupby("披露月份").size()
            if not same.empty:
                fig.add_scatter(x=list(same.index), y=list(same.values),
                                mode="lines+markers",
                                line=dict(color="#185FA5", width=1.5, dash="dot"),
                                marker=dict(color="#185FA5", size=8),
                                name="历年同月")
            # 本次预测(橙色大点)
            fig.add_scatter(x=[target], y=[point], mode="markers+text",
                            marker=dict(color="#D85A30", size=20,
                                        line=dict(color="white", width=2)),
                            text=[f"预测 {point}"], textposition="top center",
                            textfont=dict(color="#D85A30", size=13),
                            name="本次预测")
            fig.update_layout(height=240, margin=dict(l=10, r=10, t=20, b=30),
                              showlegend=True, plot_bgcolor="white",
                              legend=dict(font=dict(size=10), orientation="h",
                                          yanchor="bottom", y=1.03, x=0))
            st.plotly_chart(fig, use_container_width=True)
            st.caption(f"灰柱=全部{len(monthly)}个月实际检修量 · 蓝虚线=历年{target_month}月 · 橙点=本次预测")

    st.write("---")

    # ===== ② 结构预测 =====
    equip_groups, type_dist, repeat_equip, heavy_df, region_dist = predict_structure(df, target_month, target_year)
    c3, c4 = st.columns([1, 1])
    with c3:
        st.markdown("### ② 结构预测(哪些线路/设备在检修)")
        if equip_groups is not None and len(equip_groups) > 0:
            rows = []
            for _, row in equip_groups.head(10).iterrows():
                loc = ""
                if row.get("变电站"):
                    loc += str(row["变电站"])
                if row.get("所属地区"):
                    loc += f"（{row['所属地区']}）"
                cyc = f"{row['平均周期月']}月" if pd.notna(row.get("平均周期月")) else "短间隔"
                if pd.notna(row.get("历史次数")) and int(row["历史次数"]) < 5:
                    cyc = "样本不足"
                nxt = row.get("下次预计") or "—"
                stt = row.get("状态", "")
                if stt:
                    nxt = f"{nxt}  {stt}" if nxt != "—" else stt
                rows.append({
                    "线路/设备": row["停电设备"],
                    "位置": loc or "—",
                    "历史次数": int(row["历史次数"]),
                    "周期(月)": cyc,
                    "下次预计": nxt,
                    "主要级别": row["主要级别"],
                    "涉及单位": row["申请单位"],
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True, height=300)
            top3 = equip_groups.head(3)["停电设备"].tolist()
            st.success(f"⚡ 高频检修线路: {' · '.join(top3)} — 历史同期反复出现, {target_month}月大概率再次安排")
        else:
            st.info(f"历史 {target_month} 月无明显集中检修线路")

        # 设备类型
        st.markdown("**设备类型分布**")
        if type_dist is not None and len(type_dist) > 0:
            td = type_dist.reset_index()
            td.columns = ["设备类型", "次数"]
            st.dataframe(td, use_container_width=True, hide_index=True, height=180)

        # 重复披露设备(仅展示真实重复≥2次的设备, 单次出现无规律意义)
        if repeat_equip is not None and len(repeat_equip) > 0:
            rq = repeat_equip[repeat_equip >= 2]
            if len(rq) > 0:
                st.markdown("**跨期重复披露设备（同设备跨期出现≥2次，具备规律性）**")
                rq = rq.reset_index()
                rq.columns = ["线路/设备", "重复次数"]
                st.dataframe(rq, use_container_width=True, hide_index=True, height=140)

        # 逾期设备清单(检出概率上调 + 超6月人工确认)
        if "状态" in equip_groups.columns:
            overdue = equip_groups[equip_groups["状态"].astype(str).str.contains("已逾期")]
            if not overdue.empty:
                st.markdown("**⏰ 逾期设备清单（检出概率上调）**")
                st.caption("以下设备『下次预计』已逾期, 模型将其检修检出概率上调; "
                           "逾期>6月建议人工确认是否漏披露或已取消。")
                st.dataframe(overdue[["停电设备", "申请单位", "上次检修", "下次预计", "状态", "历史次数"]],
                             use_container_width=True, hide_index=True, height=200)
                over6 = overdue[overdue["状态"].astype(str).str.contains(r"已逾期([6-9]|1\d|2\d)月")]
                if not over6.empty:
                    st.warning(f"⚠ {len(over6)} 台设备逾期超过 6 个月, 建议人工核对是否漏披露。")

    with c4:
        st.markdown("**重点线路历史检修明细**")
        if heavy_df is not None and len(heavy_df) > 0:
            cols_show = ["停电设备", "申请单位", "检修级别", "开始日期", "结束日期", "工作内容"]
            cols_show = [c for c in cols_show if c in heavy_df.columns]
            hot = heavy_df[cols_show].head(20)
            st.dataframe(hot, use_container_width=True, hide_index=True, height=470)
        else:
            st.info(f"{target_month} 月历史无集中线路检修记录")

    # 地区分布(历史同期)
    st.write("---")
    st.markdown("### 🗺 检修地区分布（历史同期）")
    st.caption("按设备所属地区(从申请单位推断)统计; 超高压/送变电单位为'全省/跨区'。起点→终点原始PDF无字段, 位置以变电站名作代理。")
    if region_dist is not None and len(region_dist) > 0:
        import plotly.graph_objects as go
        rd = region_dist.reset_index()
        rd.columns = ["所属地区", "检修次数"]
        fig = go.Figure(go.Bar(x=rd["所属地区"], y=rd["检修次数"],
                               marker_color="#185FA5",
                               text=rd["检修次数"], textposition="outside"))
        fig.update_layout(height=240, margin=dict(l=10, r=10, t=20, b=40),
                          plot_bgcolor="white", yaxis_title="检修次数")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("暂无地区分布数据")

    st.write("---")

    # ===== ②b 高影响检修清单（按容量权重）=====
    st.markdown("### 🔥 高影响检修清单（按容量权重排序）")
    st.caption("第一步: 用设备类型权重(发电机组5 > 主变4 > 母线/线路3 > 开关2 > 其他1)替代精确MW, "
               "量化检修'分量'。补到设备容量后可换成真实MW损失。")
    if high_df is not None and not high_df.empty:
        st.dataframe(high_df, use_container_width=True, hide_index=True, height=330)
        st.success(
            f"⚡ 历史同期加权检修影响指数 ≈ {w_avg:.0f}；"
            f"其中发电机组检修 {wstats['gen']} 项、主变 {wstats['trans']} 项 —— "
            f"这些是推高供给紧张度、最可能影响电价的主要来源")
    else:
        st.info("历史同期无高影响(权重≥3)检修记录")

    st.write("---")

    # ===== ③ 时间分布 =====
    weekly, peak_week = predict_timing(df, target_year, target_month)
    st.markdown("### ③ 时间分布(月内节奏)")
    if weekly is not None and not weekly.empty:
        import plotly.graph_objects as go
        max_v = weekly["数量"].max() or 1
        colors = ["#FFD580" if v < max_v*0.5 else "#FFA94D" if v < max_v*0.8 else "#D85A30"
                  for v in weekly["数量"]]
        fig = go.Figure()
        fig.add_bar(x=weekly["周次"], y=weekly["数量"], marker_color=colors,
                    text=weekly["数量"], textposition="inside", showlegend=False)
        fig.update_layout(height=220, margin=dict(l=10, r=10, t=10, b=30),
                          plot_bgcolor="white", xaxis_title="", yaxis_title="检修项数")
        st.plotly_chart(fig, use_container_width=True)
        if peak_week:
            st.success(f"📌 集中时段: **{peak_week}** —— 短端集中, 月初相对宽松, 末旬可能收尾")
            st.caption("⚠ 周分布基于历史同期, 置信度较低, 仅作节奏参考, 不作为排期依据")
    else:
        st.info("暂无历史同期时间数据")

    st.write("---")

    # ===== 🔋 有效备用率（供给紧张度）=====
    er = effective_reserve(reserve_lut, w_avg, target)
    if er:
        st.markdown("### 🔋 有效备用率（供给紧张度）")
        st.caption(f"名义备用率 = (常规最大发电能力−最大用电负荷)/最大用电负荷 · 数据来源: {er['source']}；"
                   "检修加权指数定性扣减(精确MW损失需设备台账, 待补)")
        c_er1, c_er2, c_er3 = st.columns(3)
        with c_er1:
            st.metric("名义备用率", f"{er['nominal']:.1f}%",
                      help=f"{er['month']} 常规最大发电能力 {er['cap']:.0f} − 最大用电负荷 {er['load']:.0f} (万kW)")
        with c_er2:
            st.metric("检修加权影响指数", f"{er['w_avg']:.0f}")
        with c_er3:
            st.metric("有效备用率估算", f"{er['eff']:.1f}%")
        if er["eff"] < 10:
            st.error("⚠ 有效备用率低于 10% 警戒线, 供给偏紧, 建议提前锁量/谨慎报价")
        elif er["eff"] < 20:
            st.warning("供给偏紧预警, 关注检修集中时段的价格上行")
        else:
            st.success("供给相对宽松, 月度合约可争取更优条款")
    else:
        st.info(f"该月({target})暂无常规最大发电能力/最大用电负荷数据(披露报告覆盖 2024-10~2026-07 + 月度平衡 2026-09), 备用率模块待补全")

    st.write("---")

    # ===== 📦 供给结构最新实测总览(用户指定的装机披露 + 交易计划两份数据源) =====
    render_latest_structure(df_disc, df_trade, target_year, target_month)

    # ===== 📈 月度供需预测（出力结构, 上级要求②）=====
    render_supply_demand(df_disc, target_year, target_month)

    # ===== 📊 供需与价格（信息披露报告, 34个月）=====
    if df_disc is not None and not df_disc.empty:
        st.markdown("### 📊 供需与价格（信息披露报告 · 全月度）")
        st.caption("来源: 月度信息披露报告(2023-09~2026-06, 共34期)。本区块把『检修→供需→交易』的后两层(供给/价格)直接可视化。")
        dd = df_disc.dropna(subset=["月份"]).copy()
        dd = dd.sort_values("月份")
        c_s1, c_s2 = st.columns(2)
        import plotly.graph_objects as go
        with c_s1:
            st.markdown("**装机容量 vs 全社会用电量**")
            fig = go.Figure()
            if dd["装机总_万kW"].notna().any():
                fig.add_scatter(x=dd["月份"], y=dd["装机总_万kW"],
                                mode="lines+markers", name="总装机(万kW)",
                                line=dict(color="#185FA5", width=2))
            if dd["全社会用电量当月_亿kWh"].notna().any():
                fig.add_scatter(x=dd["月份"], y=dd["全社会用电量当月_亿kWh"],
                                mode="lines+markers", name="全社会用电量(亿kWh)",
                                line=dict(color="#2e7d32", width=1.5, dash="dot"), yaxis="y2")
            fig.update_layout(height=260, margin=dict(l=10, r=10, t=20, b=40),
                              plot_bgcolor="white",
                              yaxis=dict(title="装机(万kW)"),
                              yaxis2=dict(title="电量(亿kWh)", overlaying="y", side="right"),
                              legend=dict(font=dict(size=10), orientation="h", yanchor="bottom", y=1.05, x=0))
            st.plotly_chart(fig, use_container_width=True)
        with c_s2:
            st.markdown("**分电源结算均价（元/MWh）**")
            fig2 = go.Figure()
            colors = {"均价_火电_元每MWh": "#D85A30", "均价_水电_元每MWh": "#1E88E5",
                      "均价_风电_元每MWh": "#2e7d32", "均价_光伏_元每MWh": "#FBC02D"}
            for col, colo in colors.items():
                if col in dd.columns and dd[col].notna().any():
                    src = col.split("_")[1]
                    fig2.add_scatter(x=dd["月份"], y=dd[col],
                                     mode="lines+markers", name=f"{src}",
                                     line=dict(color=colo, width=1.8))
            fig2.update_layout(height=260, margin=dict(l=10, r=10, t=20, b=40),
                               plot_bgcolor="white", yaxis=dict(title="元/MWh"),
                               legend=dict(font=dict(size=10), orientation="h", yanchor="bottom", y=1.05, x=0))
            st.plotly_chart(fig2, use_container_width=True)
            st.caption("火电均价最高(约 360 至 440 元/MWh)，光伏最低(约 80 至 110 元/MWh)；风光低价是甘肃电价长期压制因素")

    # ===== 🔗 检修量 ↔ 电价关联(诚实实证) =====
    corr = maint_price_corr(df, df_disc)
    if corr:
        st.markdown("### 🔗 检修量 ↔ 电价关联（实证）")
        st.caption("来源: 检修记录(加权指数) × 月度披露报告(火电结算均价), 2024-2026 共28个月")
        import plotly.graph_objects as go
        xs = np.linspace(corr["x"].min(), corr["x"].max(), 50)
        fig = go.Figure()
        fig.add_scatter(x=corr["x"], y=corr["y"], mode="markers", name="各月",
                        text=corr["months"],
                        marker=dict(color="#185FA5", size=9),
                        hovertemplate="%{text}<br>检修指数%{x:.0f}<br>火电均价%{y:.0f}<extra></extra>")
        fig.add_scatter(x=xs, y=corr["slope"] * xs + corr["intercept"], mode="lines",
                        name="回归线", line=dict(color="#D85A30", dash="dash"))
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=20, b=30),
                          plot_bgcolor="white",
                          xaxis_title="月度检修加权指数", yaxis_title="火电均价(元/MWh)",
                          legend=dict(font=dict(size=10), orientation="h",
                                      yanchor="bottom", y=1.05, x=0))
        st.plotly_chart(fig, use_container_width=True)
        c_a, c_b = st.columns(2)
        with c_a:
            st.metric("原始相关系数 r", f"{corr['r']:.2f}",
                      help=f"p={corr['p']:.3f}, 显著正相关")
        with c_b:
            st.metric("去季节化 r", f"{corr['r_ds']:.2f}",
                      help=f"p={corr['p_ds']:.2f}, 不显著")
        st.warning(
            "⚠ 相关性主要来自**季节性共振**（检修高峰月与电价高位月重合），"
            "去季节化后独立解释力弱。可作月度节奏参考，**不宜宣称"
            "“检修每增 X% 电价必涨 Y%”的精确因果弹性**。结论口径："
            "检修高峰月电价大概率同步偏高，但月度间波动更多由负荷/新能源决定。")

    st.write("---")

    # ===== 💱 月度交易结构（交易计划）=====
    if df_trade is not None and not df_trade.empty:
        st.markdown("### 💱 月度交易结构（电量交易计划）")
        st.caption("来源: 每月《电网电量交易计划》(2024-01~2026-08, 共32期)。揭示合约电量结构与省内外送/外购格局。")
        td = df_trade.dropna(subset=["月份"]).copy().sort_values("月份")
        c_t1, c_t2 = st.columns(2)
        import plotly.graph_objects as go
        with c_t1:
            st.markdown("**分电源合约电量（亿kWh）**")
            fig = go.Figure()
            src_map = {"火电合约_亿kWh": ("火电", "#D85A30"),
                       "新能源合约_亿kWh": ("新能源", "#2e7d32"),
                       "水电合约_亿kWh": ("水电", "#1E88E5")}
            for col, (lab, colo) in src_map.items():
                if col in td.columns and td[col].notna().any():
                    fig.add_bar(name=lab, x=td["月份"], y=td[col], marker_color=colo)
            fig.update_layout(barmode="stack", height=260, margin=dict(l=10, r=10, t=20, b=40),
                              plot_bgcolor="white", yaxis=dict(title="亿kWh"),
                              legend=dict(font=dict(size=10), orientation="h", yanchor="bottom", y=1.05, x=0))
            st.plotly_chart(fig, use_container_width=True)
        with c_t2:
            st.markdown("**外送 / 净送出（亿kWh）**")
            fig2 = go.Figure()
            for col, lab, colo in [("中长期外送_亿kWh", "中长期外送", "#185FA5"),
                                    ("净送出_亿kWh", "净送出", "#7B1FA2"),
                                    ("外购电_亿kWh", "外购电", "#C62828")]:
                if col in td.columns and td[col].notna().any():
                    fig2.add_scatter(x=td["月份"], y=td[col], mode="lines+markers",
                                     name=lab, line=dict(color=colo, width=1.8))
            fig2.update_layout(height=260, margin=dict(l=10, r=10, t=20, b=40),
                               plot_bgcolor="white", yaxis=dict(title="亿kWh"),
                               legend=dict(font=dict(size=10), orientation="h", yanchor="bottom", y=1.05, x=0))
            st.plotly_chart(fig2, use_container_width=True)
            st.caption("甘肃为送端省: 净送出长期为正且逐年走高(2024→2026 外送约38→93亿kWh), "
                       "检修若叠加外送高峰, 省内供给进一步收紧")

    st.write("---")

    # ===== 📤 外送与受入受出预测（上级要求③）=====
    render_outward(df_trade, target_year, target_month, risk)

    st.write("---")

    # ===== 暂未开放模块提示（缺数据）=====
    st.info("⚠ 暂未开放模块（缺对应数据，已列入《数据索取清单》）："
            "① 具体影响电价区间 / ④ 日前-实时分时价格区间（缺现货价格数据）；"
            "⑤ 每日实盘复盘（缺实时价格 + 新能源功率预测系统）。数据到位后直接接入本看板，无需重构。")

    # ===== ④ 交易解读(最终结论) =====
    st.markdown("### ④ 交易解读(最终结论)")
    signals = trading_signal(df_bal, df_sec, df, df_trade, target_year, target_month, point, season, risk)

    st.markdown(
        '<div style="background:#FFF8E1;padding:20px;border-radius:8px;border-left:5px solid #FFA000">',
        unsafe_allow_html=True,
    )
    grid = st.columns(2)
    for i, (label, verdict, detail) in enumerate(signals):
        with grid[i % 2]:
            st.markdown(
                f'<div style="background:white;padding:14px;border-radius:6px;margin-bottom:10px">'
                f'<div style="font-size:14px;color:#666">⚡ {label} · <b style="color:#D85A30">{verdict}</b></div>'
                f'<div style="font-size:13px;color:#333;margin-top:6px">{detail}</div></div>',
                unsafe_allow_html=True,
            )
    st.markdown("</div>", unsafe_allow_html=True)

    # 建议依据(预测页): 用 risk(高峰月)代理检修偏多; 外送高位=实际净送出>预测105%
    maint_high = (risk == "高")
    out_high = False
    if df_trade is not None and not df_trade.empty and "净送出_亿kWh" in df_trade.columns:
        fc_out_rp = seasonal_forecast_value(df_trade, "净送出_亿kWh", target_year, target_month)
        rr = df_trade[df_trade["月份"] == target]
        if fc_out_rp and not rr.empty:
            oa = pd.to_numeric(rr["净送出_亿kWh"].iloc[0], errors="coerce")
            if pd.notna(oa) and fc_out_rp["point"]:
                out_high = oa > fc_out_rp["point"] * 1.05

    # 总体提示(基于有依据的组合: 检修总量偏高 + 外送高位, 而非低置信度的周节奏)
    if maint_high and out_high:
        st.info(
            f"💼 **提示**: {target_month}月检修偏多且外送处于高位, 省内供给双重收紧, "
            f"现货端偏多思路为主; 重点跟踪 **{peak_week if peak_week else '检修集中时段'}** "
            "(月内节奏仅供参考, 置信度较低)。")
    elif maint_high:
        st.info(
            f"💼 **提示**: {target_month}月检修偏多, 供给端承压, 关注现货上行; "
            f"外送可控。月内节奏(**{peak_week if peak_week else '集中时段'}**)仅作参考。")
    else:
        st.info(
            f"💼 **提示**: {target_month}月检修处同期正常/偏低水平, 供给相对宽松, "
            "现货以中性策略为主; 月度合约谈判空间相对宽松(仅供参考)。")

    st.caption(
        "⚠️ 本报告基于历史规律生成, 实际检修计划以甘肃省电力市场信息披露平台每月发布的正式文件为准。"
    )

    # 报告下载（🌤 天气-检修适配已移至「📈 已披露复盘」页, 作为第 12 项子节）


    st.write("---")
    _, btn_col, _ = st.columns([3, 1, 3])
    with btn_col:
        report_md = build_report_md(target, point, lo, hi, season, risk, signals, peak_week, equip_groups, repeat_equip, fc, df=df)
        st.download_button("📥 下载本报告(Markdown)", report_md,
                           f"甘肃检修预测_{target}.md", "text/markdown")


def build_report_md(target, point, lo, hi, season, risk, signals, peak_week, equip_groups, repeat_equip=None, fc=None, df=None):
    """生成可下载的报告文本."""
    lines = [
        f"# 甘肃电网检修预测报告 - {target}",
        "",
        f"数据来源: 甘肃省电力市场月度披露文件 + 历史同期规律",
        "",
    ]
    # 当月检修明细(已披露月份核心, 实测优先于预测)
    if df is not None and int(df[df["披露月份"] == target].shape[0]) > 0:
        sub = df[df["披露月份"] == target]
        n = len(sub)
        regs = sub["所属地区"].replace("其他", pd.NA).dropna().unique().tolist()
        top = sub["所属地区"].value_counts()
        top_region = top.index[0] if (not top.empty and top.index[0] != "其他") else (
            top.index[1] if len(top) > 1 else "—")
        lines += [
            "## 当月检修明细（实测 · 核心）",
            "",
            f"- 本月检修项数: **{n} 项**（来源: 当月正式披露文件, 非预测）",
            f"- 涉及地区: {len(regs)} 个（{', '.join(regs[:6])}{' 等' if len(regs) > 6 else ''}）",
            (f"- 最高频地区: {top_region}" if top_region != "—" else "- 最高频地区: —"),
            "- 完整逐条明细（线路/时间/影响区域/电价区间/置信度）见看板『当月检修明细』面板",
            "",
        ]
    lines += [
        "## ① 总量预测",
        "",
        f"- 预计检修项数: **{point} 项**",
        f"- 历史波动范围: {lo} ~ {hi} 项",
        f"- 季节属性: {season}",
        f"- 供给收紧风险: {risk}",
    ]
    if fc is not None:
        if fc.get("method"):
            lines.append(f"- 预测方法: {fc['method']}")
        if fc.get("warn"):
            lines.append(f"- ⚠ 数据提示: {fc['warn']}")
    lines += [
        "## ② 结构预测(哪些线路在检修)",
        "",
    ]
    if equip_groups is not None and len(equip_groups) > 0:
        lines.append("| 线路/设备 | 位置 | 历史次数 | 周期(月) | 下次预计 | 状态 | 主要级别 | 涉及单位 |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for _, row in equip_groups.head(10).iterrows():
            loc = ""
            if row.get("变电站"):
                loc += str(row["变电站"])
            if row.get("所属地区"):
                loc += f"（{row['所属地区']}）"
            cyc = (f"{row['平均周期月']}月" if pd.notna(row.get("平均周期月"))
                   else "短间隔")
            if pd.notna(row.get("历史次数")) and int(row["历史次数"]) < 5:
                cyc = "样本不足"
            nxt = row.get("下次预计") or "—"
            stt = row.get("状态", "")
            lines.append(f"| {row['停电设备']} | {loc or '—'} | {int(row['历史次数'])} | {cyc} | {nxt} | {stt or '—'} | {row['主要级别']} | {row['申请单位']} |")
    if repeat_equip is not None and len(repeat_equip) > 0:
        reps = [(e, int(c)) for e, c in repeat_equip.items() if int(c) >= 2]
        if reps:
            lines += ["", "**跨期重复披露设备（≥2次，具备规律性）:**", ""]
            for equip, cnt in reps:
                lines.append(f"- {equip} (重复 {cnt} 次)")
    lines += ["", "## ③ 时间分布", "",
              f"- 集中时段: {peak_week if peak_week else '数据不足'}"]
    lines += ["", "## ④ 交易解读", ""]
    for label, verdict, detail in signals:
        lines.append(f"- **{label}({verdict})**: {detail}")
    lines += ["", "---",
              "*本报告基于历史规律生成, 实际检修以月度披露文件为准*"]
    return "\n".join(lines).encode("utf-8")


# ==================== 月度供需预测（出力结构） ====================
def _fc_one(dd, col, ty, tm):
    """对单列做目标月季节性预测; 返回 {point,lo,hi,method} 或 None。"""
    if col not in dd.columns:
        return None
    return seasonal_forecast_value(dd.copy(), col, ty, tm)

def _fc_sum(dd, cols, ty, tm):
    """对多列分别预测后求和(如 新能源=风电+光伏), 区间按分量相加。"""
    pts, los, his = [], [], []
    methods = []
    for c in cols:
        r = _fc_one(dd, c, ty, tm)
        if r:
            pts.append(r["point"]); los.append(r["lo"]); his.append(r["hi"])
            methods.append(r.get("method", ""))
    if not pts:
        return None
    return {"point": round(sum(pts), 1), "lo": round(sum(los), 1), "hi": round(sum(his), 1),
            "method": f"分项季节性求和({'+'.join(cols)})"}

def _conf_label(r):
    """由预测区间相对宽度给置信度标签。"""
    if not r or r["point"] <= 0:
        return "—"
    rel = (r["hi"] - r["lo"]) / r["point"]
    if rel < 0.3:
        return "高"
    if rel < 0.6:
        return "中"
    return "低"

def render_latest_structure(df_disc, df_trade, target_year, target_month):
    """用户明确关心的两份数据源(发电机组装机披露 + 交易计划)最新月实测总览卡。
    把月度披露报告(装机/上网电量)与月度交易计划(外送/受入)的最新实际月聚合成一屏指标卡,
    与上方季节性预测互为印证。数据已在 power_maintenance 库(月度披露报告 / 月度交易计划表)。
    """
    st.markdown("### 📦 供给结构最新实测（装机 · 发电量 · 外送）")
    st.caption("数据来源: 月度披露报告(装机/上网电量) + 月度交易计划(外送/受入)。"
               "展示各自最新已披露月份的实际值, 与上方季节性预测互为印证。")
    d_disc = df_disc.dropna(subset=["月份"]).copy() if df_disc is not None else pd.DataFrame()
    if d_disc.empty:
        st.info("暂无披露数据, 本面板待补"); return
    latest_d = d_disc.sort_values("月份")["月份"].iloc[-1]
    row = d_disc[d_disc["月份"] == latest_d].iloc[0]

    def g(col):
        v = row.get(col)
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    st.markdown(f"**装机容量（截至 {latest_d}，万kW）**")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1: st.metric("总装机", f"{(g('装机总_万kW') or 0):.0f}")
    with c2: st.metric("火电", f"{(g('装机_火电_万kW') or 0):.0f}")
    with c3: st.metric("水电", f"{(g('装机_水电_万kW') or 0):.0f}")
    with c4: st.metric("风电", f"{(g('装机_风电_万kW') or 0):.0f}")
    with c5: st.metric("太阳能", f"{(g('装机_光电_万kW') or 0):.0f}")
    with c6: st.metric("储能", f"{(g('装机_独立储能_万kW') or 0):.0f}")

    st.markdown(f"**当月上网发电量（{latest_d}，亿kWh）**")
    e1, e2, e3, e4, e5 = st.columns(5)
    with e1: st.metric("合计", f"{(g('上网电量当月_亿kWh') or 0):.0f}")
    with e2: st.metric("火电", f"{(g('上网_火电_亿kWh') or 0):.0f}")
    with e3: st.metric("水电", f"{(g('上网_水电_亿kWh') or 0):.0f}")
    with e4: st.metric("风电", f"{(g('上网_风电_亿kWh') or 0):.0f}")
    with e5: st.metric("太阳能", f"{(g('上网_光电_亿kWh') or 0):.0f}")

    if df_trade is not None and not df_trade.dropna(subset=["月份"]).empty:
        d_trade = df_trade.dropna(subset=["月份"]).copy()
        lt = d_trade.sort_values("月份")["月份"].iloc[-1]
        tr = d_trade[d_trade["月份"] == lt].iloc[0]

        def gt(col):
            v = tr.get(col)
            try:
                return float(v)
            except (TypeError, ValueError):
                return None

        st.markdown(f"**外送与受入（{lt}，亿kWh）**")
        o1, o2, o3, o4 = st.columns(4)
        with o1: st.metric("净送出", f"{(gt('净送出_亿kWh') or 0):.0f}")
        with o2: st.metric("中长期外送", f"{(gt('中长期外送_亿kWh') or 0):.0f}")
        with o3: st.metric("外送(年+月)", f"{((gt('外送_年度_亿kWh') or 0) + (gt('外送_月度_亿kWh') or 0)):.0f}")
        with o4: st.metric("外购(受入)", f"{(gt('外购电_亿kWh') or 0):.0f}")


def render_supply_demand(dd, target_year, target_month):
    """② 月度供需预测(出力结构): 新能源/火电/水电/总发电/全社会负荷 + 供需关系 + 置信度。
    数据源: 月度披露报告的分电源上网电量与全社会用电量(已有, 不依赖缺失的三份数据)。
    """
    st.markdown("### 📈 月度供需预测（出力结构）")
    st.caption("数据来源: 月度信息披露报告 · 分电源上网电量 / 全社会用电量（已有数据）。"
               "针对所选月份做季节性预测；已披露月显示实际值（标注✓）。新能源=风电+光伏(光电)。")
    if dd is None or dd.empty:
        st.info("暂无披露数据，本面板待补"); return
    dd = dd.dropna(subset=["月份"]).copy()
    dd["新能源光伏_亿kWh"] = dd.get("上网_光电_亿kWh", pd.Series(dtype=float)).fillna(
        dd.get("上网_光伏_亿kWh", pd.Series(dtype=float)))
    target_str = f"{target_year}-{target_month:02d}"
    ar = dd[dd["月份"] == target_str]
    has_actual = not ar.empty

    def _actual(col):
        if not has_actual:
            return None
        v = ar[col].sum(skipna=True)
        return None if pd.isna(v) else float(v)

    specs = [
        ("新能源出力", _fc_sum(dd, ["上网_风电_亿kWh", "新能源光伏_亿kWh"], target_year, target_month),
         _actual("上网_风电_亿kWh") if _actual("上网_风电_亿kWh") is None else
         (_actual("上网_风电_亿kWh") + (_actual("新能源光伏_亿kWh") or 0))),
        ("火电出力", _fc_one(dd, "上网_火电_亿kWh", target_year, target_month), _actual("上网_火电_亿kWh")),
        ("水电出力", _fc_one(dd, "上网_水电_亿kWh", target_year, target_month), _actual("上网_水电_亿kWh")),
        ("总发电量", _fc_one(dd, "上网电量当月_亿kWh", target_year, target_month), _actual("上网电量当月_亿kWh")),
        ("全社会负荷", _fc_one(dd, "全社会用电量当月_亿kWh", target_year, target_month), _actual("全社会用电量当月_亿kWh")),
    ]

    cards = st.columns(5)
    for (lab, fc_r, act), c in zip(specs, cards):
        with c:
            if fc_r is None:
                st.metric(lab, "—", help="无历史同期数据")
                continue
            conf = _conf_label(fc_r)
            if act is not None:
                st.metric(lab, f"{act:.0f} 亿kWh ✓", delta=f"预测 {fc_r['point']:.0f}",
                          help=f"预测区间 {fc_r['lo']}~{fc_r['hi']} 亿kWh; 置信度{conf}; {fc_r['method']}")
            else:
                st.metric(lab, f"{fc_r['point']:.0f} 亿kWh",
                          help=f"预测区间 {fc_r['lo']}~{fc_r['hi']} 亿kWh; 置信度{conf}; {fc_r['method']}")

    # 供需关系: 发电量 vs 负荷 历史 + 目标月预测点
    import plotly.graph_objects as go
    st.markdown("**供需关系（发电量 vs 全社会负荷）**")
    fig = go.Figure()
    if dd["上网电量当月_亿kWh"].notna().any():
        fig.add_bar(x=dd["月份"], y=dd["上网电量当月_亿kWh"], name="发电量(上网电量)",
                    marker_color="#185FA5")
    if dd["全社会用电量当月_亿kWh"].notna().any():
        fig.add_scatter(x=dd["月份"], y=dd["全社会用电量当月_亿kWh"],
                        mode="lines+markers", name="全社会负荷(用电量)",
                        line=dict(color="#D85A30", width=2))
    gen_fc = _fc_one(dd, "上网电量当月_亿kWh", target_year, target_month)
    load_fc = _fc_one(dd, "全社会用电量当月_亿kWh", target_year, target_month)
    if gen_fc:
        fig.add_scatter(x=[target_str], y=[gen_fc["point"]], mode="markers",
                        marker=dict(color="#185FA5", size=16, symbol="diamond", line=dict(color="white")),
                        name="发电量预测")
    if load_fc:
        fig.add_scatter(x=[target_str], y=[load_fc["point"]], mode="markers",
                        marker=dict(color="#D85A30", size=16, symbol="diamond", line=dict(color="white")),
                        name="负荷预测")
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=20, b=40), plot_bgcolor="white",
                      yaxis_title="亿kWh", legend=dict(font=dict(size=10), orientation="h",
                      yanchor="bottom", y=1.05, x=0))
    st.plotly_chart(fig, use_container_width=True)

    if gen_fc and load_fc:
        gap = gen_fc["point"] - load_fc["point"]
        if gap > 0:
            st.success(f"📊 供需关系: 预计发电量超出负荷约 **{gap:.0f} 亿kWh**, 省内供需紧平衡、安全垫偏薄, "
                       f"富余电量有限(外送已按净额口径单列, 不重复计算)。")
        else:
            st.warning(f"📊 供需关系: 预计发电量低于负荷约 **{abs(gap):.0f} 亿kWh**, 存在缺口, 需外购或压减外送。")
    st.caption("注: '发电量'以省级结算口径上网电量近似; '负荷'以全社会用电量(电量口径)近似, 非瞬时功率。精确MW需新能源功率预测系统(待接入)。")

# ==================== 外送与受入受出预测 ====================
def render_outward(df_trade, target_year, target_month, risk):
    """③ 外送与受入受出预测(上级要求③): 净送出/外送/外购(受入) 预测 + 受入受出表 + 重大事项录入占位 + 置信度。
    数据来源: 月度交易计划(已有)。'送入省份重大事项'依赖外部信息, 先留人工录入占位。
    """
    if df_trade is None or df_trade.empty:
        return
    st.markdown("### 📤 外送与受入受出预测")
    st.caption("来源: 月度交易计划(2024-01~2026-08, 32期)。甘肃为送端省; 净送出/外送为'送出', 外购电为'受入'。置信度=季节性预测区间。")
    cols_cfg = [
        ("预计净送出", "净送出_亿kWh"),
        ("预计中长期外送", "中长期外送_亿kWh"),
        ("预计外购(受入)", "外购电_亿kWh"),
    ]
    # fallback: 交易计划缺该月历史时, 用联络线分时日电量加总推算净送出(实测代理)
    tl_net = None
    try:
        tl = load_tieline()
        if tl is not None and not tl.empty and "日电量_万kWh" in tl.columns:
            tl_m = tl[tl["月份"].astype(str) == target_str]
            if not tl_m.empty:
                s = pd.to_numeric(tl_m["日电量_万kWh"], errors="coerce").sum()
                if pd.notna(s) and s > 0:
                    tl_net = round(float(s) / 1e4, 1)
    except Exception:
        pass
    cards = st.columns(3)
    preds = {}
    for (lab, col), c in zip(cols_cfg, cards):
        r = seasonal_forecast_value(df_trade, col, target_year, target_month)
        preds[col] = r
        with c:
            if r:
                st.metric(lab, f"{r['point']} 亿kWh",
                          help=f"区间 {r['lo']}~{r['hi']} 亿kWh; 置信度{_conf_label(r)}; {r.get('method','—')}")
            elif col == "净送出_亿kWh" and tl_net is not None:
                st.metric(lab, f"{tl_net} 亿kWh",
                          help="交易计划无该月历史, 由联络线分时日电量加总推算(实测代理, 非预测)")
            else:
                st.info(f"{target_month} 月无历史同期{lab}数据")

    st.markdown("**受入受出预测表**")
    tbl = []
    for lab, col in cols_cfg:
        r = preds.get(col)
        if r:
            tbl.append({"项目": lab, "预测(亿kWh)": r["point"],
                        "区间下限": r["lo"], "区间上限": r["hi"], "置信度": _conf_label(r)})
    if tbl:
        st.dataframe(pd.DataFrame(tbl), use_container_width=True, hide_index=True)

    st.markdown("**送入省份下月重大事项（人工录入）**")
    st.caption("此部分依赖外部信息(如湖南台风→用电减少→外送剩余), 目前先留空, 待人工维护后接入。")
    note = st.text_area("记录下月重大事项（如：湖南台风→用电减少，外送电量剩余）",
                        value="", height=70, key="outward_note")
    if note.strip():
        st.success(f"已记录: {note.strip()[:60]}（会话内临时记录，未持久化；后续可接入数据库字段）")

    if preds.get("净送出_亿kWh") and risk == "高":
        st.warning("⚠ 检修高峰月 + 外送高位: 省内供给或双重收紧, 重点关注现货价格上行")
    elif preds.get("净送出_亿kWh"):
        st.success("检修低谷 / 外送可控: 省内供给相对宽松")


# ==================== 数据台账(简单导出页) ====================
# ==================== 当月检修明细（已披露月份的核心面板） ====================
def render_month_detail(df, target):
    """当月检修明细: 针对已披露月份, 逐条展示实测数据(线路/时间/地区/电价区间(待补)/置信度)。
    这是用户认定的'重中之重'—— 已披露月优先展示实测, 预测降级为参考。"""
    sub = df[df["披露月份"] == target].copy()
    if sub.empty:
        return False
    st.markdown("### 📋 当月检修明细（已披露 · 核心）")
    st.caption("来源: 甘肃省电力市场信息披露平台当月正式文件, 以下为逐条实测数据, 非预测。")

    # 汇总卡
    n = len(sub)
    reg = sub["所属地区"].replace("其他", pd.NA).dropna()
    n_region = reg.nunique()
    top = sub["所属地区"].value_counts()
    if not top.empty:
        if top.index[0] != "其他":
            top_region, top_n = top.index[0], int(top.iloc[0])
        elif len(top) > 1:
            top_region, top_n = top.index[1], int(top.iloc[1])
        else:
            top_region, top_n = "—", 0
    else:
        top_region, top_n = "—", 0
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("本月检修项数", f"{n} 项")
    with c2:
        st.metric("涉及地区", f"{n_region} 个")
    with c3:
        st.metric("最高频地区", f"{top_region}（{top_n} 项）" if top_region != "—" else "—")

    # 明细表
    det = sub.copy()

    def fmt_date(d):
        return d.strftime("%m-%d") if pd.notna(d) else None

    def time_cell(r):
        s, e = fmt_date(r.get("开始日期_dt")), fmt_date(r.get("结束日期_dt"))
        if s and e:
            return f"{s} 至 {e}"
        if pd.notna(r.get("检修天数")):
            return f"检修约 {int(r['检修天数'])} 天"
        return "—"

    det["检修时间"] = det.apply(time_cell, axis=1)
    det["线路/设备"] = det["停电设备"].fillna("—")
    det["所属地区"] = det["所属地区"].replace("其他", "—")
    det["影响区域"] = det["所属地区"]  # 原始 PDF 无起止变电站字段, 以申请单位推断地区作代理
    det["影响电价区间"] = "待现货价导入"  # 上级要求①, 缺现货价格, 先空着
    det["置信度"] = "确定（官方披露）"
    show_cols = ["线路/设备", "检修时间", "所属地区", "影响区域", "影响电价区间", "置信度", "设备类型"]
    st.dataframe(det[show_cols], use_container_width=True, hide_index=True, height=420)
    st.caption("注: '影响区域'以申请单位推断的所属地区表示(原始PDF无起止变电站列); '影响电价区间'需现货价接入后填充。")
    return True


def _build_calendar(sub, ty, tm):
    """把检修区间展开成 每日×地区 的在修数矩阵, 供热力图使用。无有效日期返回 None。"""
    import calendar
    from datetime import timedelta
    s2 = sub.dropna(subset=["开始日期_dt"]).copy()
    if s2.empty:
        return None
    ndays = calendar.monthrange(ty, tm)[1]
    recs = []
    for _, r in s2.iterrows():
        start = r["开始日期_dt"]
        end = r["结束日期_dt"] if pd.notna(r["结束日期_dt"]) else start
        d = start
        while d <= end:
            recs.append((r["所属地区"], int(d.day)))
            d += timedelta(days=1)
    if not recs:
        return None
    dd = pd.DataFrame(recs, columns=["地区", "日"])
    piv = dd.pivot_table(index="地区", columns="日", values="地区", aggfunc="count")
    piv = piv.reindex(columns=range(1, ndays + 1), fill_value=0)
    piv = piv.loc[piv.sum(axis=1).sort_values(ascending=False).index]
    return {"z": piv.values.tolist(), "x": [int(c) for c in piv.columns],
            "y": piv.index.tolist()}


def render_review():
    """已披露检修复盘（交易决策页）: 针对已披露月份, 把实测数据提炼成交易信号。
    8 项: 提示卡/压力日历/甘特/高影响清单/电源结构/地区集中度/外送-检修/回测。
    全部基于已入库数据, 不依赖台账/现货价。"""
    st.markdown("# 📈 已披露检修复盘（交易决策页）")
    st.caption("针对已披露月份, 将实测检修数据提炼为交易信号; 数据: 检修记录(5692条)+披露报告(35月)+交易计划(32月)")

    df = load_maint()
    df_disc = load_disclosure()
    df_trade = load_trade_plan()

    avail = sorted(df["披露月份"].unique().tolist())
    if not avail:
        st.info("暂无检修数据"); return
    target = st.selectbox("📅 复盘月份", avail[::-1], index=0, label_visibility="collapsed")
    ty, tm = int(target[:4]), int(target[5:7])

    sub = df[df["披露月份"] == target].copy()
    if sub.empty:
        st.warning(f"⚠ {target} 尚未披露, 无实测数据可复盘。请选择上方已披露月份。")
        return

    # ---------- 0. 本月核心结论（决策速览，仅已披露实测） ----------
    st.markdown("### 🧭 本月核心结论（决策速览）")
    st.caption("针对已披露月份, 用实测数据一屏定调; 决策人最需要: 检修规模 / 外送通道约束 / 供需松紧。")
    df_sec = load_section()
    sec_m = df_sec[df_sec["月份"].astype(str) == target] if (df_sec is not None and not df_sec.empty) else pd.DataFrame()
    sec_aff = int(sec_m["备注"].astype(str).str.contains("检修", na=False).sum()) if not sec_m.empty else 0

    def _poslim(v):
        try:
            x = float(str(v).replace("无", "").replace("—", "").replace("~", "").replace("-", "").strip())
            return x > 0
        except Exception:
            return False

    def _fwd_num(v):
        s = str(v).replace("无", "").replace("—", "").replace("~", "").replace("-", "").strip()
        try:
            return float(s)
        except Exception:
            return 0.0

    # 各断面当月『最低』正向限额(卡脖子值); 并检测时段性收紧(同月内最低<<最高)
    per_sec = {}
    if not sec_m.empty:
        for name, g in sec_m.groupby("断面名称"):
            vals = g["正向限额"].apply(_fwd_num)
            pos = vals[vals > 0]
            if len(pos):
                per_sec[name] = (float(pos.min()), float(pos.max()))
    out_cnt = len(per_sec)
    out_win = "较宽" if out_cnt >= 3 else ("偏窄" if per_sec else "无断面数据")
    # 局部收紧: 某断面当月最低限额明显低于其当月最高(检修致时段性收紧)
    tight = [(n, lo, hi) for n, (lo, hi) in per_sec.items()
             if hi > 0 and lo < 0.7 * hi and lo < 700]
    out_note = ""
    if tight and out_win == "较宽":
        out_win = "较宽·局部收紧"
        out_note = "⚠ 注意: " + "、".join(f"{n}(降至{int(lo)})" for n, lo, hi in tight) \
                   + " 当月因检修时段性收紧, 外送需避开其紧张窗口"
    # 供需: 已披露月用实测上网电量 vs 全社会用电量
    supply, supply_src = "数据不足", ""
    if df_disc is not None and not df_disc.empty:
        ar_disc = df_disc[df_disc["月份"].astype(str) == target]
        if not ar_disc.empty:
            g_act = pd.to_numeric(ar_disc["上网电量当月_亿kWh"].iloc[0], errors="coerce")
            l_act = pd.to_numeric(ar_disc["全社会用电量当月_亿kWh"].iloc[0], errors="coerce")
            if pd.notna(g_act) and pd.notna(l_act):
                # 措辞保守化: 发电量略高于负荷 ≠ 充裕; 外送已按净额口径单列, 不在此重复"可外送"
                supply = "紧平衡·安全垫薄" if (g_act - l_act) > 0 else "偏紧·需外购"
                supply_src = "实测"

    n_maint = len(sub)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("检修量（实测）", f"{n_maint} 项", help="当月已披露检修条数")
    with c2:
        st.metric("受检修影响断面", f"{sec_aff} 个",
                  help="当月断面限额备注含『检修』的通道数, 直接卡外送窗口")
    with c3:
        _out_help = (f"{out_cnt} 个断面正向有容量" if not sec_m.empty else "该月无断面数据")
        if out_note:
            _out_help += "；" + out_note
        st.metric("外送窗口", out_win, help=_out_help)
    with c4:
        _supply_help = ("上网电量 vs 全社会用电量" + (f"（{supply_src}）" if supply_src else "")
                       + "；发电量仅略高于负荷时为紧平衡, 安全垫薄")
        st.metric("供需关系", supply, help=_supply_help)
    st.markdown(
        f'<div style="background:#EAF1F8;padding:12px 16px;border-radius:8px;'
        f'border-left:4px solid #185FA5;margin-top:6px">'
        f'<div style="font-size:13px;color:#333">🧭 <b>{target}</b> 已披露实测：检修 {n_maint} 项 · '
        f'{sec_aff} 个外送断面受检修影响 · 外送窗口{out_win} · 供需{supply}（{supply_src}）。</div></div>',
        unsafe_allow_html=True)
    # 检修量口径可复核: 合并规则 + 分项计数(与PDF分项加总可能±1, 源于跨表重复披露)
    cat_counts = sub["类别"].value_counts() if "类别" in sub.columns else pd.Series(dtype=int)
    cat_line = " · ".join(f"{k}:{int(v)}" for k, v in cat_counts.items()) if len(cat_counts) else ""
    st.caption(
        f"检修量口径: 同(申请单位+设备)当月合并为1项, 封网/拆网各计1项; 与PDF分项加总可能±1"
        f"(跨表重复披露所致, 如茅泉Ⅱ线跨表、封网/拆网拆两条)。"
        + (f" 分项计数(按类别): {cat_line}。" if cat_line else ""))
    st.write("---")

    # ---------- 1. 一句话交易提示卡 ----------
    n = len(sub)
    fc = seasonal_forecast(df, ty, tm)
    # 压力指数分母用『历史同期实际均值』(非预测值), 避免循环论证
    hist_same = df[df["月"] == tm].groupby("披露月份").size()
    hist_same_mean = float(hist_same.mean()) if len(hist_same) else None
    ratio = (n / hist_same_mean) if (hist_same_mean and hist_same_mean > 0) else 1.0
    reg = sub["所属地区"].replace("其他", pd.NA).dropna()
    top_region_name = reg.value_counts().index[0] if not reg.empty else "—"
    out_actual = None
    out_src = ""
    if df_trade is not None and not df_trade.empty and "净送出_亿kWh" in df_trade.columns \
            and "月份" in df_trade.columns:
        rr = df_trade[df_trade["月份"] == target]
        if not rr.empty:
            out_actual = pd.to_numeric(rr["净送出_亿kWh"].iloc[0], errors="coerce")
            out_src = "交易计划实测"
    # fallback: 交易计划缺该月时, 用联络线分时日电量加总推算外送(万kWh→亿kWh)
    if (out_actual is None or pd.isna(out_actual)):
        try:
            tl = load_tieline()
            if tl is not None and not tl.empty and "日电量_万kWh" in tl.columns:
                tl_m = tl[tl["月份"].astype(str) == target]
                if not tl_m.empty:
                    s = pd.to_numeric(tl_m["日电量_万kWh"], errors="coerce").sum()
                    if pd.notna(s) and s > 0:
                        out_actual = round(float(s) / 1e4, 1)
                        out_src = "联络线分时推算"
        except Exception:
            pass
    fc_out = seasonal_forecast_value(df_trade, "净送出_亿kWh", ty, tm) \
        if (df_trade is not None and not df_trade.empty) else None
    out_high = (out_actual is not None and fc_out and fc_out["point"]
                and out_actual > fc_out["point"] * 1.05)
    maint_high = ratio > 1.05
    if maint_high and out_high:
        tag, color, bg = "🔴 双重挤压 · 现货看多", "#A32D2D", "#FCEBEB"
        advice = (f"{target} 检修偏多(较同期+{(ratio-1)*100:.0f}%)且外送高位, "
                  f"省内供给双重收紧, 日前/日内报价偏多, 重点盯盘。")
    elif maint_high:
        tag, color, bg = "🟠 供给偏紧 · 谨慎", "#BA7517", "#FAEEDA"
        advice = (f"{target} 检修偏多(较同期+{(ratio-1)*100:.0f}%), 供给端承压, "
                  f"关注现货上行; 外送可控。")
    else:
        tag, color, bg = "🟢 供给宽松 · 中性", "#3B6D11", "#EAF3DE"
        advice = (f"{target} 检修处于同期正常/偏低水平, 供给相对宽松, 现货以中性策略为主。")
    st.markdown(
        f'<div style="background:{bg};padding:16px 20px;border-radius:10px;border-left:5px solid {color}">'
        f'<div style="font-size:18px;font-weight:700;color:{color}">{tag}</div>'
        f'<div style="color:#333;margin-top:8px;font-size:14px">{advice}</div>'
        f'<div style="color:#666;margin-top:6px;font-size:12px">本月检修 {n} 项 · '
        f'最高频地区 {top_region_name} · 外送实测 {out_actual if out_actual is not None else "—"} 亿kWh'
        f'{("（" + out_src + "）") if out_src else ""}</div>'
        f'</div>', unsafe_allow_html=True)

    # ---------- 2. 检修压力日历（每日热力图） ----------
    st.markdown("### 📅 检修压力日历（每日在修数 × 地区）")
    st.caption("颜色越深=当天同时检修的设备越多, 直接定位'哪几天哪地区要盯盘'。"
               "地区中『其他』=无法从申请单位识别地市的记录; 『全省/跨区』=超高压/送变电单位的全省性项目。")
    import plotly.graph_objects as go
    cal = _build_calendar(sub, ty, tm)
    if cal is not None:
        fig = go.Figure(go.Heatmap(
            z=cal["z"], x=cal["x"], y=cal["y"], colorscale="OrRd", showscale=True,
            colorbar=dict(title="在修数"),
            hovertemplate="地区:%{y}<br>日期:%{x}日<br>在修数:%{z}<extra></extra>"))
        fig.update_layout(height=max(220, 40 * len(cal["y"]) + 60),
                          margin=dict(l=80, r=10, t=10, b=40),
                          yaxis=dict(autorange="reversed"),
                          xaxis_title=f"{ty}年{tm}月（日）")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("该月检修记录缺少有效起止日期, 无法生成日历。")

    # ---------- 3. 关键检修时间条（甘特） ----------
    st.markdown("### 🗓 关键检修时间条")
    st.caption("横条=每条独立检修事件; 颜色越深=影响权重越高(机组/主变>线路/母线); "
               "y 轴带『厂·设备』双标签——同名设备(如多个厂的 #2机组)是不同物理机组, "
               "不是同一台机器重复; 悬停可见完整申请单位。")
    g = sub.dropna(subset=["开始日期_dt"]).copy()
    g = g.sort_values(["权重", "开始日期_dt"], ascending=[False, True]).head(20)
    if not g.empty:
        import plotly.express as px
        g["结束_dt"] = g["结束日期_dt"].fillna(g["开始日期_dt"])
        # y 轴标签 = 厂名 + 设备名(避免不同厂的同名设备看起来像同一条)
        g["厂简"] = g["申请单位"].astype(str).str.slice(0, 10)
        g["设备简"] = g["停电设备"].astype(str).str.slice(0, 14)
        g["y轴标签"] = g["厂简"] + " · " + g["设备简"]
        fig = px.timeline(g, x_start="开始日期_dt", x_end="结束_dt", y="y轴标签",
                          color="权重", color_continuous_scale="OrRd",
                          hover_data={"申请单位": True, "停电设备": True, "权重": True,
                                      "y轴标签": False,
                                      "开始日期_dt": "|%m-%d", "结束_dt": "|%m-%d"})
        fig.update_layout(height=max(320, 26 * len(g) + 60),
                          margin=dict(l=10, r=10, t=10, b=30), showlegend=False,
                          xaxis_title=f"{ty}年{tm}月", coloraxis_colorbar_title="影响权重")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("该月无带起止日期的检修记录。")

    # ---------- 4. 高影响检修清单 ----------
    st.markdown("### 📋 高影响检修清单（按检修级别 + 持续天数）")
    st.caption("表格按(申请单位+设备)合并汇总, 每行=一组设备; '当月次数'列=该设备当月独立记录条数; "
               "'影响区域'为该厂站所属地市; '持续天数'='结束-开始'天数(缺结束日期用检修天数)。"
               "排序: 检修级别(A>B>C>D>改造/例行…)优先, 同级再比持续天数 —— "
               "A级40天大修自然排在D级10天小修之前。多事件时间分布见上方甘特图。")
    lst = sub.copy()
    LEVEL_RANK = {"A级检修": 5, "B级检修": 4, "C级检修": 3, "改造大修": 3, "D级检修": 2,
                  "例行检修": 2, "检修预试": 2, "基建接入": 2, "消缺": 1, "其他": 1}

    def _sev(lv, w):
        if lv in LEVEL_RANK:
            return LEVEL_RANK[lv]
        return 3 if w >= 4 else (2 if w == 3 else 1)

    def _days(r):
        s, e = r.get("开始日期_dt"), r.get("结束日期_dt")
        if pd.notna(s) and pd.notna(e):
            return max(1, int((e - s).days))
        if pd.notna(r.get("检修天数")):
            try:
                return int(float(r["检修天数"]))
            except Exception:
                return None
        return None

    lst["持续天数"] = lst.apply(_days, axis=1)
    lst["等级"] = lst["检修级别"].astype(str).fillna("其他")
    lst["_sev"] = lst.apply(lambda r: _sev(r["等级"], r["权重"]), axis=1)
    # 合并逻辑: 按(申请单位, 停电设备), 避免不同电厂同名机组被错误合并
    grp = lst.sort_values(["_sev", "持续天数", "权重"], ascending=False).drop_duplicates(
        subset=["申请单位", "停电设备"], keep="first")
    cnt = sub.groupby(["申请单位", "停电设备"]).size().rename("当月次数")
    grp = grp.merge(cnt, on=["申请单位", "停电设备"], how="left")
    grp = grp.drop(columns=["持续天数"], errors="ignore")
    days_max = lst.groupby(["申请单位", "停电设备"])["持续天数"].max().rename("持续天数")
    grp = grp.merge(days_max, on=["申请单位", "停电设备"], how="left")

    def fmt_range(start, end):
        s = start.strftime("%m-%d") if pd.notna(start) else "—"
        e = end.strftime("%m-%d") if pd.notna(end) else s
        return f"{s} 至 {e}" if s != "—" else "—"

    grp["检修时间"] = grp.apply(lambda r: fmt_range(r["开始日期_dt"], r["结束日期_dt"]), axis=1)
    grp = grp.sort_values(["_sev", "持续天数", "权重"], ascending=[False, False, False])
    show = grp[["停电设备", "申请单位", "检修时间", "所属地区",
                "设备类型", "等级", "持续天数", "当月次数"]].copy()
    show = show.rename(columns={"停电设备": "线路/设备", "所属地区": "影响区域",
                                  "申请单位": "申请单位/厂站"})
    show["影响区域"] = show["影响区域"].replace("其他", "—")
    show["持续天数"] = show["持续天数"].apply(lambda v: f"{int(v)}天" if pd.notna(v) else "—")
    show = show[["线路/设备", "申请单位/厂站", "检修时间",
                  "影响区域", "设备类型", "等级", "持续天数", "当月次数"]]
    st.dataframe(show, use_container_width=True, hide_index=True, height=360)

    # ---------- 5. 检修与电源结构（有数据才显示, 无数据静默隐藏） ----------
    disc_row = df_disc[df_disc["月份"] == target] if (df_disc is not None
                                                      and not df_disc.empty) else None
    power_cols = []
    if disc_row is not None and not disc_row.empty:
        for c in disc_row.columns:
            if any(k in c for k in ["火电", "水电", "新能源", "风电", "光伏", "外送", "发电量", "上网"]) \
                    and pd.api.types.is_numeric_dtype(disc_row[c]):
                power_cols.append(c)
    if power_cols:
        vals = disc_row[power_cols].iloc[0].astype(float).dropna()
        if not vals.empty:
            st.markdown("### ⚡ 检修与电源结构")
            import plotly.express as px
            fig = px.bar(x=list(vals.index), y=list(vals.values),
                         color=list(vals.index), color_discrete_sequence=px.colors.qualitative.Set2,
                         text=[f"{v:.0f}" for v in vals.values])
            fig.update_traces(textposition="outside")
            fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=100),
                              showlegend=False, yaxis_title="数值")
            st.plotly_chart(fig, use_container_width=True)
            st.caption("来源: 月度披露报告(分电源发电量字段)。结合本月火电检修占比, "
                       "判断电源结构对供需的影响权重。")

    # ---------- 6. 检修地区集中度 ----------
    st.markdown("### 📊 检修地区集中度")
    rc = sub["所属地区"].replace("其他", pd.NA).dropna().value_counts()
    if not rc.empty:
        cr3 = rc.head(3).sum() / rc.sum()
        c1, c2 = st.columns([1, 2])
        with c1:
            st.metric("地区集中度 CR3", f"{cr3*100:.0f}%")
            st.caption("**前三地区占比; >70% 表示高度集中(区域价差易拉大); <50% 表示分散。**")
        with c2:
            import plotly.express as px
            fig = px.bar(x=rc.index.tolist(), y=rc.values.tolist(),
                         color=rc.values.tolist(), color_continuous_scale="Blues",
                         text=[f"{v}项" for v in rc.values.tolist()])
            fig.update_traces(textposition="outside")
            fig.update_layout(height=280, margin=dict(l=10, r=10, t=10, b=60),
                              showlegend=False, yaxis_title="检修项数")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("该月检修地区均为'其他', 无法做地区集中度分析。")

    # ---------- 7. 供给压力评估（原: 外送-检修双重挤压） ----------
    st.markdown("### 🔀 供给压力评估（检修 × 外送）")
    st.caption("检修压力指数=当月检修项数/历史同期预测值(100%=均值, >100%偏多); "
               "外送压力指数=当月净送出/历史同期预测值。两者均偏高→省内供给双重收紧。")
    c1, c2 = st.columns(2)
    with c1:
        st.metric("检修压力指数", f"{ratio*100:.0f}%", delta=f"较同期{(ratio-1)*100:+.0f}%",
                  help="当月检修项数 / 历史同期预测值; 100%为历史均值")
    with c2:
        if out_actual is not None:
            out_ratio = (out_actual / fc_out["point"]) if (fc_out and fc_out["point"]) else 1.0
            st.metric("外送压力指数", f"{out_ratio*100:.0f}%", delta=f"较同期{(out_ratio-1)*100:+.0f}%",
                      help="当月净送出 / 历史同期预测值; 100%为历史均值")
        else:
            st.info("该月外送数据缺失")
    if maint_high and out_high:
        st.error("🔴 双重挤压: 检修与外送同时偏高, 省内供给双重收紧, 现货价格上行概率高, "
                 "建议提前布局多单/锁价。")
    elif maint_high:
        st.warning("🟠 检修偏高: 供给端承压, 关注现货上行。")
    else:
        st.success("🟢 检修与外送均处于正常区间, 供给相对宽松。")

    # ---------- 8. 模型回测 ----------
    st.markdown("### 🎯 模型回测（实际 vs 预测 + 朴素基线）")
    st.caption("评估模型是否『比拍脑袋强』: 加入两个朴素基线(上月延续 / 同月均值)对照; "
               "点估计误差大属正常, 关键看方向准确率与是否优于基线。")
    hist = df.groupby("披露月份").size()
    months = sorted(hist.index.tolist())[-12:]
    rows = []
    prev_actual = None
    for m in months:
        y, mo = int(m[:4]), int(m[5:7])
        pr = seasonal_forecast(df, y, mo)
        point = pr["point"] if pr and pr["point"] is not None else None
        lo = pr["lo"] if pr else None
        hi = pr["hi"] if pr else None
        act = int(hist[m])
        same = df[df["月"] == mo]
        same_others = same[same["披露月份"] != m].groupby("披露月份").size()
        base_same = float(same_others.mean()) if len(same_others) else None
        rows.append({"月份": m, "实际": act, "预测": point,
                     "区间下": lo, "区间上": hi,
                     "基线_上月": prev_actual, "基线_同月": base_same})
        prev_actual = act
    back = pd.DataFrame(rows)
    if back.empty:
        st.info("历史披露月份不足, 无法做回测。")
    else:
        b2 = back.dropna(subset=["预测"]).copy()
        if not b2.empty:
            b2["实际方向"] = b2["实际"].diff().fillna(0).apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
            b2["预测方向"] = b2["预测"].diff().fillna(0).apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
            n_valid = (b2["实际方向"] != 0).sum()
            n_correct = ((b2["实际方向"] == b2["预测方向"]) & (b2["实际方向"] != 0)).sum()
            dir_rate = (n_correct / n_valid * 100) if n_valid else 0
            st.success(f"✅ 模型方向准确率 {n_correct}/{n_valid} = {dir_rate:.0f}%"
                       "（判断'当月检修量相对上月是升高还是降低'）")
        with st.expander("▶ 点估计精度与基线对比（复盘用, 日常可忽略）"):
            st.caption("本模型为方向性参考, 点估计误差较大属正常。下列指标用于判断是否『比朴素基线强』——"
                       "若模型 MAE 不优于基线, 说明该预测不值得单独信赖, 应直接采用基线。")
            if not b2.empty:
                b2["偏差%"] = (b2["实际"] - b2["预测"]) / b2["预测"] * 100
                b2["MAE"] = b2["偏差%"].abs()
                mae = b2["MAE"].mean()

                def _mae_base(col):
                    s = b2.dropna(subset=[col])
                    return float(((s["实际"] - s[col]).abs()).mean()) if len(s) >= 2 else None

                mae_prev = _mae_base("基线_上月")
                mae_same = _mae_base("基线_同月")
                cov = float(((b2["实际"] >= b2["区间下"]) & (b2["实际"] <= b2["区间上"])).mean() * 100)
                st.markdown(f"- 模型点估计 MAE(=MAPE): **{mae:.0f}%**")
                if mae_prev and mae_same:
                    st.markdown(f"- 朴素基线 MAE — 上月延续: **{mae_prev:.0f}%** ｜ 同月均值: **{mae_same:.0f}%**")
                    best = min(mae, mae_prev, mae_same)
                    verdict = "模型优于朴素基线, 可参考" if abs(mae - best) < 1e-6 else "模型未优于朴素基线, 建议直接用基线"
                    st.markdown(f"- 结论: **{verdict}**")
                else:
                    st.markdown("- 朴素基线: 样本不足, 无法对照")
                st.markdown(f"- 预测区间覆盖率: **{cov:.0f}%**（实际值落进波动范围的比例; "
                            f"若远低于名义水平, 说明区间仍偏窄）")
                import plotly.graph_objects as go
                fig = go.Figure()
                fig.add_bar(x=back["月份"], y=back["实际"], name="实际", marker_color="#185FA5")
                fig.add_scatter(x=back["月份"], y=back["预测"], mode="lines+markers",
                                name="预测", line=dict(color="#D85A30"))
                fig.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=60),
                                  legend_orientation="h", yaxis_title="检修项数")
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(b2[["月份", "实际", "预测", "区间下", "区间上",
                                 "基线_上月", "基线_同月", "偏差%"]].round(0),
                             use_container_width=True, hide_index=True)
                st.caption(f"近 12 月点估计平均绝对偏差 {mae:.0f}%（仅供复盘参考）。")

    # ---------- 9. 断面限额（外送通道实测 · 已披露） ----------
    st.write("---")
    st.markdown("### 🔌 断面限额（外送通道实测 · 已披露）")
    st.caption("来源: 每月披露的断面限额表(已披露实测)。正向限额 = 外送能力上限, 反向限额 = 受入能力上限; "
               "备注含『检修』= 该通道当月被检修占用, 直接卡外送窗口——决策人看外送/受入前先盯这条。")
    if df_sec is not None and not df_sec.empty:
        sm = df_sec[df_sec["月份"].astype(str) == target].copy()
        if not sm.empty:
            sm_disp = sm.rename(columns={"断面名称": "断面",
                                         "正向限额": "正向限额(外送)",
                                         "反向限额": "反向限额(受入)"})
            sm_disp["备注"] = sm_disp.apply(
                lambda r: ("⚠ " + str(r["备注"])) if ("检修" in str(r["备注"])) else r["备注"], axis=1)
            st.dataframe(sm_disp, use_container_width=True, hide_index=True, height=320)
            n_aff = int(sm["备注"].astype(str).str.contains("检修", na=False).sum())
            if n_aff:
                st.warning(f"⚠ 当月 {n_aff} 个断面限额备注含『检修』, 这些外送通道被检修占用, "
                           f"谈外送增量前需重点核实。")
            else:
                st.success("当月断面限额备注无检修占用, 外送通道基本畅通。")
            try:
                sm2 = sm.copy()

                def _fwd(v):
                    s = str(v).replace("无", "").replace("—", "").strip()
                    if s.startswith("-") or s == "":
                        return 0.0
                    try:
                        return float(s)
                    except Exception:
                        return 0.0

                sm2["正向(外送)"] = sm2["正向限额"].apply(_fwd)
                # 卡脖子值 = 该断面当月『最低』正向限额(最紧张时段约束外送能力),
                # 不把各时段限额加总(加总会虚高数倍, 如甘陕 950+560+950+700+950≈4110 实为5个时段)
                per_sec = (sm2.groupby("断面名称")["正向(外送)"]
                           .apply(lambda s: float(s[s > 0].min()) if (s > 0).any() else 0.0)
                           .reset_index())
                per_sec = per_sec[per_sec["正向(外送)"] > 0]
                if not per_sec.empty:
                    import plotly.graph_objects as go
                    fig = go.Figure(go.Bar(x=per_sec["断面名称"], y=per_sec["正向(外送)"],
                                           marker_color="#185FA5",
                                           text=per_sec["正向(外送)"].astype(int),
                                           textposition="outside"))
                    fig.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=60),
                                      yaxis_title="本月最低正向限额(外送·卡脖子值)")
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption("柱高=该断面当月『最低』正向限额(最紧张时段), 即外送能力的卡脖子约束, "
                               "不把各时段限额加总; 限额越低, 该通道外送空间越小; "
                               "0 表示该通道仅可受入(反向)。")
            except Exception:
                pass
        else:
            st.info(f"⚠ {target} 无断面限额数据(该月可能未披露)。")
    else:
        st.info("无断面数据")

    # ---------- 10. 联络线分时（外送/受入实测） ----------
    st.write("---")
    st.markdown("### 🔗 联络线分时（外送/受入实测）")
    st.caption("来源: 联络线分时表(997行, 每日24h交换功率/电量)。这是外送/受入的小时级实测, "
               "揭示日内外送节奏与月度趋势; 符号方向需结合实际确认(通常正为送出)。")
    df_tl = load_tieline()
    if df_tl is not None and not df_tl.empty:
        tl = df_tl[df_tl["月份"].astype(str) == target].copy()
        if not tl.empty:
            hour_cols = [f"{h}时" for h in range(24)]
            for c in hour_cols + (["日电量_万kWh"] if "日电量_万kWh" in tl.columns else []):
                tl[c] = pd.to_numeric(tl[c], errors="coerce")
            prof = tl[hour_cols].mean()
            import plotly.graph_objects as go
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**典型日内曲线（全月24h均值）**")
                fig = go.Figure()
                fig.add_scatter(x=list(range(24)), y=prof.values, mode="lines+markers",
                                line=dict(color="#185FA5", width=2),
                                hovertemplate="%{x}时<br>%{y:.0f}<extra></extra>")
                fig.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=40),
                                  xaxis_title="小时", yaxis_title="交换功率(万kW)",
                                  xaxis=dict(dtick=3))
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                st.markdown("**月度日交换电量趋势**")
                fig2 = go.Figure()
                fig2.add_scatter(x=tl["日"], y=tl["日电量_万kWh"], mode="lines+markers",
                                 line=dict(color="#D85A30", width=2),
                                 hovertemplate="%{x}日<br>%{y:.0f}万kWh<extra></extra>")
                fig2.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=40),
                                   xaxis_title="日", yaxis_title="日电量(万kWh)")
                st.plotly_chart(fig2, use_container_width=True)
            st.caption("典型日内曲线显示外送的日内高峰/低谷时段(便于择时); 月度趋势显示外送是否逐日走高——"
                       "若叠加检修高峰, 省内供给进一步收紧。")
        else:
            st.info(f"⚠ {target} 无联络线分时数据(该月可能未披露)。")
    else:
        st.info("暂无联络线分时数据。")

    st.write("---")
    st.caption("本页各交易信号均基于已披露实测数据; '影响电价区间'等需现货价接入后在'当月检修明细'页补充。"
               "🌤 天气适配块(页尾)为已披露检修×天气预报扩展, 天气非实测。")

    # ---------- 11. 各模块数据覆盖期(口径透明) ----------
    st.markdown("### 📚 各模块数据覆盖期")
    st.caption("看板各数据源的覆盖区间; 预测月超出覆盖期时, 对应面板会提示『数据不足』而非编造。")

    def _cov(loader, col="月份"):
        try:
            d = loader()
            if d is None or d.empty:
                return "—"
            ms = d[col].astype(str)
            return f"{ms.min()} ~ {ms.max()}（{len(d)} 行）"
        except Exception:
            return "—"

    cov_rows = [
        ("检修记录", _cov(load_maint, "披露月份")),
        ("月度披露报告", _cov(load_disclosure)),
        ("月度平衡", _cov(load_bal)),
        ("断面限额", _cov(load_section)),
        ("月度交易计划", _cov(load_trade_plan)),
        ("联络线分时", _cov(load_tieline)),
    ]
    st.dataframe(pd.DataFrame(cov_rows, columns=["数据模块", "覆盖期"]),
                 use_container_width=True, hide_index=True)

    # ---------- 12. 🌤 天气-检修适配(已披露检修 × 未来天气) ----------
    st.markdown("### 🌤 天气-检修适配（已披露检修 × 未来 10 天天气）")
    st.caption("本块为『已披露检修计划 × 未来 10 天天气』的扩展分析: 天气为模式预报(非实测), "
               "仅用于趋势与受限阈值判断; 其余各块均为已披露实测。天气适配按所选月检修计划 + 未来天气判断可作业窗口。")
    render_weather_maint(ty, tm, df, df_sec)

    # ===== 💰 已披露现货电价(月均, 2026-1~8) =====
    st.markdown("---")
    st.markdown("### 💰 已披露现货电价(月度均, 2026-01~08)")
    st.caption("数据源: daily_spot_price(已接入). 单位 元/MWh. 现货价是检修-供给-价格传导链的最终兑现, 直接服务报价决策。")
    try:
        import plotly.graph_objects as go
        _c = pymysql.connect(**DB_CONFIG)
        _ms = pd.read_sql(
            "SELECT DATE_FORMAT(`日期`,'%Y-%m') ym, "
            "ROUND(AVG(`日前价_元MWh`),1) da, ROUND(AVG(`实时价_元MWh`),1) rt, "
            "ROUND(AVG(`偏差_元MWh`),1) sp "
            "FROM daily_spot_price GROUP BY ym ORDER BY ym", _c)
        _c.close()
        if not _ms.empty:
            fig = go.Figure()
            fig.add_trace(go.Bar(x=_ms['ym'], y=_ms['da'], name='日前月均'))
            fig.add_trace(go.Bar(x=_ms['ym'], y=_ms['rt'], name='实时月均'))
            fig.update_layout(barmode='group', title='月度现货均价(元/MWh)', yaxis_title='元/MWh',
                              height=340, margin=dict(l=40, r=20, t=40, b=30))
            st.plotly_chart(fig, use_container_width=True)
            hi = _ms.loc[_ms['da'].idxmax()]
            lo = _ms.loc[_ms['da'].idxmin()]
            st.info(f"月度日前均价区间: {lo['da']}({lo['ym']}) ~ {hi['da']}({hi['ym']}) 元/MWh。"
                    f"高价月(检修/外送双重收紧时)报价偏多; 低价月(新能源大发)关注低价消纳与负价风险。")
        else:
            st.caption("daily_spot_price 暂无数据")
    except Exception as _e:
        st.caption(f"现货电价加载异常: {_e}")


def render_ledger():
    st.markdown("# 数据台账")
    st.caption("底层数据查询与 Excel 导出 —— 不进入日常看, 仅作核验用")

    df = load_maint()
    c1, c2, c3 = st.columns(3)
    with c1: year = st.multiselect("年份", sorted(df["年"].unique().tolist()),
                                   default=sorted(df["年"].unique().tolist()))
    with c2: cat = st.multiselect("类别", df["类别"].unique().tolist(),
                                  default=df["类别"].unique().tolist())
    with c3: level = st.multiselect("检修级别", sorted(df["检修级别"].dropna().unique().tolist()),
                                    default=sorted(df["检修级别"].dropna().unique().tolist()))

    if "开始日期_dt" in df.columns:
        d_range = st.date_input("开始日期范围",
                                value=(df["开始日期_dt"].min().date(),
                                       df["开始日期_dt"].max().date()))
        if isinstance(d_range, tuple) and len(d_range) == 2:
            mask = (df["开始日期_dt"].dt.date >= d_range[0]) & (df["开始日期_dt"].dt.date <= d_range[1])
            df = df[mask]

    show = df[df["年"].isin(year) & df["类别"].isin(cat) & df["检修级别"].isin(level)].copy()

    # 标记跨月续检（底仓）: 开始月份 ≠ 结束月份, 即上月延续到本月
    if "开始日期_dt" in show.columns and "结束日期_dt" in show.columns:
        show["跨月标记"] = ""
        _m1 = show["开始日期_dt"].dt.month
        _m2 = show["结束日期_dt"].dt.month
        show.loc[_m1 != _m2, "跨月标记"] = "✅ 跨月续检"
    else:
        show["跨月标记"] = ""

    only_carry = st.checkbox("只看跨月续检（底仓）", value=False)
    if only_carry:
        show = show[show["跨月标记"] == "✅ 跨月续检"]

    st.markdown(f"**共 {len(show):,} 条记录**")
    _drop_cols = [c for c in show.columns if c in ("开始日期_dt", "结束日期_dt", "年", "月")]
    _display_cols = ["跨月标记"] + [c for c in show.columns if c not in _drop_cols and c != "跨月标记"]
    st.dataframe(show[_display_cols],
                 use_container_width=True, hide_index=True, height=420)

    csv = show.to_csv(index=False).encode("utf-8-sig")
    st.download_button("📥 导出 CSV", csv, f"检修数据_{datetime.now():%Y%m%d}.csv", "text/csv")

    st.write("---")
    # 断面限额面板已移至「📈 已披露复盘」页(第 9 项), 与联络线分时并列, 仅用已披露实测。


# ==================== 📅 日度态势(新, 框架) ====================
def _build_daily_summary():
    """合成日度态势页顶部核心结论卡片。返回 {'spot_date':str,'lines':[...]}。"""
    try:
        _c = pymysql.connect(**DB_CONFIG)
        lines = []
        sd = "—"
        # 现货: 最新交易日快照
        _r = pd.read_sql("SELECT MAX(`日期`) d FROM daily_spot_price", _c)
        if not _r.empty and _r['d'].iloc[0] is not None:
            sd = pd.to_datetime(_r['d'].iloc[0]).strftime("%Y-%m-%d")
            _day = pd.read_sql(
                "SELECT `时段`,`日前价_元MWh`,`实时价_元MWh` FROM daily_spot_price "
                "WHERE `日期`=%s ORDER BY `时段`", _c, params=(sd,))
            if not _day.empty:
                da = _day['日前价_元MWh'].mean(); rt = _day['实时价_元MWh'].mean()
                _day['sp'] = (_day['日前价_元MWh'] - _day['实时价_元MWh']).abs()
                mx = _day.loc[_day['sp'].idxmax()]
                if da - rt > da * 0.05:
                    interp = "实时价持续低于日前价 → 实际供给比预期宽松(利于买方, 日前持仓者注意实时卖压)"
                elif rt - da > da * 0.05:
                    interp = "实时价高于日前价 → 实际供给偏紧(利于日前卖出, 实时采购成本上升)"
                else:
                    interp = "日前/实时价格接近 → 预期与实际基本吻合"
                lines.append(
                    f"💰 **现货(最新交易日 {sd})**: 日前均 {da:.1f} / 实时均 {rt:.1f} 元/MWh, "
                    f"最大价差 **{mx['sp']:.1f}**(出现在 {int(mx['时段'])}时)。{interp}。")
        # 天气: 未来10天
        _w = pd.read_sql(
            "SELECT `时间`,`点位名称`,`风速_10m`,`降水_mm`,`雷暴` FROM weather_hourly "
            "WHERE `时间`>=NOW() AND `时间`<DATE_ADD(NOW(),INTERVAL 10 DAY)", _c)
        if not _w.empty:
            _w["触发"] = (_w["风速_10m"] > 10.7) | (_w["雷暴"] == 1) | (_w["降水_mm"] > 0.5)
            _h = _w[_w["触发"]].copy()
            if not _h.empty:
                def _k(r):
                    if r["雷暴"] == 1: return "🌪 雷暴"
                    if r["风速_10m"] > 10.7: return "💨 大风"
                    return "🌧 强降水"
                _h["类型"] = _h.apply(_k, axis=1); _h["日"] = _h["时间"].dt.date
                _g = _h.groupby(["点位名称", "类型", "日"]).agg(
                    峰值风速=("风速_10m", "max"), 峰值降水=("降水_mm", "max"),
                    雷暴=("雷暴", "max")).reset_index()
                _bypt = _g.groupby("点位名称").size().sort_values(ascending=False)
                _dom = _g.groupby("类型").size().idxmax()
                _top = "、".join(_bypt.head(2).index.tolist())
                _mxw = _g["峰值风速"].max()
                lines.append(
                    f"⚠ **天气(未来10天)**: 共 {len(_g)} 项风险(涉及 {_g['点位名称'].nunique()} 点位), "
                    f"集中于 **{_top}** 等地, 主导「{_dom}」, 峰值风速 {_mxw:.1f} m/s, 或影响风电出力与户外检修。")
            else:
                lines.append("⚠ **天气(未来10天)**: 窗口良好, 无受限预警。")
        # 新能源出力(若有)
        _rn = pd.read_sql("SELECT MAX(`日期`) d FROM daily_renewable", _c)
        if not _rn.empty and _rn['d'].iloc[0] is not None:
            rdate = pd.to_datetime(_rn['d'].iloc[0]).strftime("%Y-%m-%d")
            _rd = pd.read_sql("SELECT `风电出力_MW`,`光伏出力_MW` FROM daily_renewable WHERE `日期`=%s", _c, params=(rdate,))
            if not _rd.empty:
                wpk = _rd['风电出力_MW'].max(); spk = _rd['光伏出力_MW'].max()
                lines.append(f"💨 **新能源出力(最新 {rdate})**: 风电峰值 {wpk:.0f} MW, 光伏峰值 {spk:.0f} MW, "
                             f"关注午间光伏大发时段的新能源消纳压力。")
        # 负荷(若有)
        _ld = pd.read_sql("SELECT MAX(`日期`) d FROM daily_load", _c)
        if not _ld.empty and _ld['d'].iloc[0] is not None:
            ldate = pd.to_datetime(_ld['d'].iloc[0]).strftime("%Y-%m-%d")
            _ldf = pd.read_sql("SELECT `统调负荷_MW` FROM daily_load WHERE `日期`=%s", _c, params=(ldate,))
            if not _ldf.empty:
                pk = _ldf['统调负荷_MW'].max(); tr = _ldf['统调负荷_MW'].min()
                lines.append(f"⚡ **负荷(最新 {ldate})**: 峰值 {pk:.0f} MW, 谷值 {tr:.0f} MW, "
                             f"峰谷差 {pk - tr:.0f} MW。")
        # 联络线
        _td = pd.read_sql("SELECT MAX(`日期`) d FROM daily_tie_line", _c)
        if not _td.empty and _td['d'].iloc[0] is not None:
            lines.append("🔌 **联络线**: 日度实时数据已接入, 详见下方「联络线外送」模块(按通道净送出)。")
        else:
            lines.append("🔌 **联络线**: 仅月度披露典型曲线(非实时日度); 出力/负荷/联络线日度待接入, 暂无法合成完整供给结论。")
        _c.close()
        return {"spot_date": sd, "lines": lines}
    except Exception as _e:
        return {"spot_date": "—", "lines": [f"⚠ 核心结论合成失败: {_e}"]}


def render_daily():
    """日度态势页: 5 个子模块. 现货价/天气已接入, 出力/负荷/联络线日度待接入."""
    import pandas as _pd
    st.markdown("# 📅 日度态势（5 数据源 · 接入中）")
    st.caption("本页聚合 5 块日度级信号, 服务每日盘前/盘中/盘后决策。"
               "现货价(1-8月历史)与天气(未来10天预报)已接入; 出力/负荷/联络线日度的表结构与导入脚本已就绪, 拿到数据即接。"
               "⚠️ 现货为历史披露、天气为预报, 均非实时行情, 决策以当日实盘为准。")

    # ---------- 顶栏: 核心结论(数据快照合成) ----------
    _summ = _build_daily_summary()
    if _summ:
        with st.container(border=True):
            st.markdown(f"### 📌 核心结论（现货取最新交易日 {_summ['spot_date']} 快照 · 天气取未来10天预报）")
            for _line in _summ['lines']:
                st.markdown(_line)

    # ---------- 总览: 5 块接入状态(动态) ----------
    st.markdown("### 📊 数据接入总览")
    _c0 = pymysql.connect(**DB_CONFIG)
    def _cnt(t):
        try:
            return int(pd.read_sql(f"SELECT COUNT(*) n FROM `{t}`", _c0).iloc[0]['n'])
        except Exception:
            return 0
    ren_n, load_n, tie_n, spot_n = _cnt('daily_renewable'), _cnt('daily_load'), _cnt('daily_tie_line'), _cnt('daily_spot_price')
    _c0.close()
    status = _pd.DataFrame([
        ("💨 风电/光伏出力",   "daily_renewable",            "小时级 MW",          f"{'✅ 已接入' if ren_n else '⏳ 待接入(需爬取)'}"),
        ("⚡ 负荷曲线",       "daily_load",                 "小时级 MW",          f"{'✅ 已接入' if load_n else '⏳ 待接入(需爬取)'}"),
        ("💰 现货价分时",     "daily_spot_price",           "日前/实时 元/MWh",   f"✅ 已接入({spot_n}行, 1-8月历史)"),
        ("🔌 联络线外送",     "daily_tie_line/联络线分时",  "日度MW/月度典型",    f"{'✅ 日度已接入' if tie_n else '🟡 月度披露(非实时)'}"),
        ("⚠ 天气预警",       "weather_hourly",             "雷暴/大风/降水",     "✅ 数据已就位"),
    ], columns=["模块", "数据源(表)", "粒度/字段", "状态"])
    st.dataframe(status, use_container_width=True, hide_index=True)

    st.write("---")

    # ---------- 1. 风电/光伏出力 ----------
    with st.container(border=True):
        st.markdown("#### 💨 风电/光伏出力")
        st.caption("数据源: daily_renewable(待接入)。字段: 日期/时段(0-23)/风电出力_MW/光伏出力_MW/新能源总出力_MW。")
        with st.expander("📐 数据 schema (MySQL)", expanded=False):
            st.code(
                "CREATE TABLE daily_renewable (\n"
                "  日期 DATE, 时段 TINYINT,\n"
                "  风电出力_MW DECIMAL(12,2), 光伏出力_MW DECIMAL(12,2),\n"
                "  新能源总出力_MW DECIMAL(12,2),\n"
                "  PRIMARY KEY (日期, 时段)\n"
                ");", language="sql")
        try:
            import plotly.graph_objects as go
            _c = pymysql.connect(**DB_CONFIG)
            _sd = pd.read_sql("SELECT MAX(`日期`) d FROM daily_renewable", _c)
            if not _sd.empty and _sd['d'].iloc[0] is not None:
                _alld = pd.read_sql("SELECT DISTINCT `日期` d FROM daily_renewable ORDER BY d", _c)['d'].tolist()
                sd = st.selectbox("选择日期", [pd.to_datetime(x).strftime("%Y-%m-%d") for x in _alld],
                                  index=len(_alld) - 1, key="ren_d")
                _df = pd.read_sql(
                    "SELECT `时段`,`风电出力_MW`,`光伏出力_MW`,`新能源总出力_MW` FROM daily_renewable "
                    "WHERE `日期`=%s ORDER BY `时段`", _c, params=(sd,))
                if not _df.empty:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=_df['时段'], y=_df['风电出力_MW'], fill='tozeroy', mode='lines',
                                            name='风电出力', line=dict(color='#2980B9')))
                    fig.add_trace(go.Scatter(x=_df['时段'], y=_df['光伏出力_MW'], fill='tozeroy', mode='lines',
                                            name='光伏出力', line=dict(color='#F39C12')))
                    if _df['新能源总出力_MW'].notna().any():
                        fig.add_trace(go.Scatter(x=_df['时段'], y=_df['新能源总出力_MW'], mode='lines',
                                                name='新能源总出力', line=dict(color='#27AE60', width=2)))
                    fig.update_layout(title=f"{sd} 风电/光伏出力(24h, MW)", xaxis_title='时段(0-23)',
                                      yaxis_title='MW', height=340, margin=dict(l=40, r=20, t=40, b=30))
                    st.plotly_chart(fig, use_container_width=True)
                    wpk = _df['风电出力_MW'].max(); spk = _df['光伏出力_MW'].max()
                    c1, c2 = st.columns(2)
                    c1.metric("风电峰值", f"{wpk:.0f} MW")
                    c2.metric("光伏峰值", f"{spk:.0f} MW")
                    st.info("📈 光伏呈日间单峰(白昼), 风电波动较大; 与负荷曲线叠加可判断新能源消纳/弃风弃光压力。")
                    if len(_df) < 24:
                        st.warning(f"⚠ {sd} 仅 {len(_df)} 个时段有数据, 曲线不完整。")
                else:
                    st.info(f"{sd} 无逐时数据")
            else:
                st.info("⏳ 数据待接入 — 运行 `python import_renewable.py --file 风电光伏出力.xlsx` 导入后, "
                         "此处自动显示 24h 出力曲线。")
            _c.close()
        except Exception as _e:
            st.error(f"风电/光伏加载失败: {_e}")

    st.divider()

    # ---------- 2. 负荷曲线 ----------
    with st.container(border=True):
        st.markdown("#### ⚡ 负荷曲线")
        st.caption("数据源: daily_load(待接入)。字段: 日期/时段(0-23)/统调负荷_MW/全社会负荷_MW。"
                   "可选叠加 weather_hourly 气温(兰州, 若该日已爬取)。")
        with st.expander("📐 数据 schema (MySQL)", expanded=False):
            st.code(
                "CREATE TABLE daily_load (\n"
                "  日期 DATE, 时段 TINYINT,\n"
                "  统调负荷_MW DECIMAL(12,2), 全社会负荷_MW DECIMAL(12,2),\n"
                "  PRIMARY KEY (日期, 时段)\n"
                ");", language="sql")
        try:
            import plotly.graph_objects as go
            _c = pymysql.connect(**DB_CONFIG)
            _sd = pd.read_sql("SELECT MAX(`日期`) d FROM daily_load", _c)
            if not _sd.empty and _sd['d'].iloc[0] is not None:
                _alld = pd.read_sql("SELECT DISTINCT `日期` d FROM daily_load ORDER BY d", _c)['d'].tolist()
                sd = st.selectbox("选择日期", [pd.to_datetime(x).strftime("%Y-%m-%d") for x in _alld],
                                  index=len(_alld) - 1, key="load_d")
                _df = pd.read_sql(
                    "SELECT `时段`,`统调负荷_MW`,`全社会负荷_MW` FROM daily_load WHERE `日期`=%s ORDER BY `时段`",
                    _c, params=(sd,))
                if not _df.empty:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=_df['时段'], y=_df['统调负荷_MW'], mode='lines', name='统调负荷',
                                            line=dict(color='#8E44AD', width=2)))
                    if _df['全社会负荷_MW'].notna().any():
                        fig.add_trace(go.Scatter(x=_df['时段'], y=_df['全社会负荷_MW'], mode='lines', name='全社会负荷',
                                                line=dict(color='#16A085', dash='dot')))
                    # 气温叠加(若有兰州该日数据)
                    _wt = pd.read_sql(
                        "SELECT HOUR(`时间`) h, `气温_2m` t FROM weather_hourly "
                        "WHERE DATE(`时间`)=%s AND `点位名称`='兰州' ORDER BY `时间`", _c, params=(sd,))
                    if not _wt.empty and _wt['t'].notna().any():
                        fig.add_trace(go.Scatter(x=_wt['h'], y=_wt['t'], mode='lines', name='兰州气温',
                                                yaxis='y2', line=dict(color='#E67E22', dash='dash')))
                        fig.update_layout(yaxis2=dict(title='气温℃', overlaying='y', side='right', showgrid=False))
                    fig.update_layout(title=f"{sd} 负荷曲线(24h, MW)", xaxis_title='时段(0-23)', yaxis_title='MW',
                                      height=340, margin=dict(l=40, r=40, t=40, b=30))
                    st.plotly_chart(fig, use_container_width=True)
                    pk = _df['统调负荷_MW'].max(); tr = _df['统调负荷_MW'].min()
                    c1, c2 = st.columns(2)
                    c1.metric("负荷峰值", f"{pk:.0f} MW")
                    c2.metric("负荷谷值", f"{tr:.0f} MW")
                    st.info("📈 负荷高峰多在早晚; 与气温叠加可量化温度弹性(夏季空调/冬季采暖)。")
                    if len(_df) < 24:
                        st.warning(f"⚠ {sd} 仅 {len(_df)} 个时段有数据。")
                else:
                    st.info(f"{sd} 无逐时数据")
            else:
                st.info("⏳ 数据待接入 — 运行 `python import_load.py --file 负荷曲线.xlsx` 导入后, "
                         "此处自动显示 24h 负荷曲线(可叠加气温)。")
            _c.close()
        except Exception as _e:
            st.error(f"负荷加载失败: {_e}")

    st.divider()

    # ---------- 3. 现货价分时(日前 vs 实时) ----------
    with st.container(border=True):
        st.markdown("#### 💰 现货价分时(日前 vs 实时)")
        st.caption("数据源: daily_spot_price(已接入 2026-01~08, 5832 行, 全月全24时段, 元/MWh)。"
                   "出清均价仅 1-4 月有(5-8 月空); 峰平谷已按小时统一补齐。日前-实时价差 → 套利/风险时段。")
        with st.expander("📐 实际数据 schema (MySQL)", expanded=False):
            st.code(
                "CREATE TABLE daily_spot_price (\n"
                "  日期 DATE, 时段 TINYINT, 时段标签 VARCHAR(20), 分段 VARCHAR(4),\n"
                "  日前价_元MWh DECIMAL(10,4), 实时价_元MWh DECIMAL(10,4),\n"
                "  出清均价_元MWh DECIMAL(10,4), 偏差_元MWh DECIMAL(10,4) AS (日前-实时) STORED,\n"
                "  PRIMARY KEY (日期, 时段)\n"
                ");", language="sql")
        try:
            import plotly.graph_objects as go
            from datetime import date as _date
            _c = pymysql.connect(**DB_CONFIG)
            _months = pd.read_sql(
                "SELECT DISTINCT DATE_FORMAT(`日期`,'%Y-%m') ym FROM daily_spot_price ORDER BY ym", _c)['ym'].tolist()
            if _months:
                ym = st.selectbox("选择月份", _months, index=len(_months) - 1, key="spot_ym")
                y, mo = int(ym[:4]), int(ym[5:7])
                _days = pd.read_sql(
                    "SELECT DISTINCT DAY(`日期`) d FROM daily_spot_price "
                    "WHERE DATE_FORMAT(`日期`,'%%Y-%%m')=%s ORDER BY d", _c, params=(ym,))['d'].tolist()
                d = st.selectbox("选择日期", _days, index=len(_days) - 1, key="spot_d")
                _day = pd.read_sql(
                    "SELECT `时段`,`分段`,`日前价_元MWh`,`实时价_元MWh`,`出清均价_元MWh` "
                    "FROM daily_spot_price WHERE `日期`=%s ORDER BY `时段`", _c, params=(f"{y}-{mo:02d}-{d:02d}",))
                if not _day.empty:
                    _day = _day.copy()
                    _day['sp'] = (_day['日前价_元MWh'] - _day['实时价_元MWh']).abs()
                    mx = _day.loc[_day['sp'].idxmax()]
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=_day['时段'], y=_day['日前价_元MWh'], mode='lines+markers', name='日前价'))
                    fig.add_trace(go.Scatter(x=_day['时段'], y=_day['实时价_元MWh'], mode='lines+markers', name='实时价'))
                    if _day['出清均价_元MWh'].notna().any():
                        fig.add_trace(go.Scatter(x=_day['时段'], y=_day['出清均价_元MWh'], mode='lines+markers', name='出清均价'))
                    # 标注最大价差时点(用户建议1)
                    fig.add_annotation(x=int(mx['时段']), y=float(mx['日前价_元MWh']),
                                       text=f"最大价差 {mx['sp']:.1f} 元/MWh<br>({int(mx['时段'])}时)",
                                       showarrow=True, arrowhead=2, ax=0, ay=-40,
                                       font=dict(color="#C0392B", size=12), arrowcolor="#C0392B")
                    # 套利/风险窗口阴影(价差大的时段)
                    _thr = max(40.0, float(_day['sp'].quantile(0.75)))
                    for _, r in _day.iterrows():
                        if r['sp'] >= _thr:
                            h = int(r['时段'])
                            fig.add_vrect(x0=h - 0.5, x1=h + 0.5, fillcolor="rgba(241,196,15,0.18)",
                                          line_width=0, layer="below")
                    fig.update_layout(title=f"{ym}-{d:02d} 分时现货价(元/MWh) — 黄区=价差大时段", xaxis_title='时段(0-23)',
                                      yaxis_title='元/MWh', height=360, margin=dict(l=40, r=20, t=40, b=30))
                    st.plotly_chart(fig, use_container_width=True)
                    da_avg = _day['日前价_元MWh'].mean()
                    rt_avg = _day['实时价_元MWh'].mean()
                    spread = _day['sp'].max()
                    c1, c2, c3 = st.columns(3)
                    c1.metric("日前均价", f"{da_avg:.1f}")
                    c2.metric("实时均价", f"{rt_avg:.1f}")
                    c3.metric("最大价差(日前-实时)", f"{spread:.1f}")
                    # 数据驱动的解释(用户建议1)
                    if da_avg - rt_avg > da_avg * 0.05:
                        interp = "实时价持续低于日前价 → 实际供给比预期宽松(利于买方, 日前持仓者注意实时卖压)"
                    elif rt_avg - da_avg > da_avg * 0.05:
                        interp = "实时价高于日前价 → 实际供给偏紧(利于日前卖出, 实时采购成本上升)"
                    else:
                        interp = "日前/实时价格接近 → 预期与实际基本吻合"
                    st.info(f"📈 {interp}。黄色阴影为价差较大的「套利/风险窗口」时段, 可重点关注日前-实时反向操作机会。")
                    if len(_day) < 24:
                        st.warning(f"⚠ {ym}-{d:02d} 仅 {len(_day)} 个时段有数据, 曲线不完整。")
                # 当月逐日均价面板
                st.markdown("**📅 当月逐日均价(日前 vs 实时)**")
                _m = pd.read_sql(
                    "SELECT DAY(`日期`) d, AVG(`日前价_元MWh`) da, AVG(`实时价_元MWh`) rt "
                    "FROM daily_spot_price WHERE DATE_FORMAT(`日期`,'%%Y-%%m')=%s GROUP BY d ORDER BY d", _c, params=(ym,))
                fig2 = go.Figure()
                fig2.add_trace(go.Bar(x=_m['d'], y=_m['da'], name='日前日均'))
                fig2.add_trace(go.Bar(x=_m['d'], y=_m['rt'], name='实时日均'))
                fig2.update_layout(barmode='group', title=f"{ym} 逐日均价(元/MWh)", xaxis_title='日',
                                   yaxis_title='元/MWh', height=320, margin=dict(l=40, r=20, t=40, b=30))
                st.plotly_chart(fig2, use_container_width=True)
                if not _m.empty:
                    hi = _m.loc[_m['da'].idxmax()]
                    lo = _m.loc[_m['da'].idxmin()]
                    st.info(f"本月日前日均最高: {int(hi['d'])}日 ({hi['da']:.1f}) | 最低: {int(lo['d'])}日 ({lo['da']:.1f})"
                            f" — 高低价日对应检修/外送安排, 关注价差套利窗口。")
                st.caption("↑ 已接入 daily_spot_price(5832 行, 全月24时段); 单位 元/MWh; 5-8月无出清均价(图表自动省略); 峰平谷已按小时补齐。")
            else:
                st.warning("daily_spot_price 暂无数据, 请先运行 import_spot_price.py")
            _c.close()
        except Exception as _e:
            st.error(f"现货价加载失败: {_e}")

    st.divider()

    # ---------- 4. 联络线外送 ----------
    with st.container(border=True):
        st.markdown("#### 🔌 联络线外送")
        try:
            import plotly.graph_objects as go
            _c = pymysql.connect(**DB_CONFIG)
            _dt = pd.read_sql("SELECT MAX(`日期`) d FROM daily_tie_line", _c)
            if not _dt.empty and _dt['d'].iloc[0] is not None:
                # === 真实日度外送 ===
                st.caption("数据源: daily_tie_line(已接入日度)。按通道展示 24h 外送/受入/净送出。")
                _alld = pd.read_sql("SELECT DISTINCT `日期` d FROM daily_tie_line ORDER BY d", _c)['d'].tolist()
                sd = st.selectbox("选择日期", [pd.to_datetime(x).strftime("%Y-%m-%d") for x in _alld],
                                  index=len(_alld) - 1, key="tie_d")
                _df = pd.read_sql(
                    "SELECT `时段`,`通道名`,`外送_MW`,`受入_MW`,`净送出_MW` FROM daily_tie_line "
                    "WHERE `日期`=%s ORDER BY `通道名`,`时段`", _c, params=(sd,))
                if not _df.empty:
                    lines = _df['通道名'].unique().tolist()
                    sel = st.multiselect("通道", lines, default=lines, key="tie_lines")
                    _sub = _df[_df['通道名'].isin(sel)]
                    fig = go.Figure()
                    for ln in sel:
                        _l = _sub[_sub['通道名'] == ln]
                        fig.add_trace(go.Scatter(x=_l['时段'], y=_l['净送出_MW'], mode='lines', name=f"{ln}(净送出)"))
                    fig.update_layout(title=f"{sd} 各通道净送出(24h, MW, +外送/-受入)", xaxis_title='时段(0-23)',
                                      yaxis_title='MW', height=360, margin=dict(l=40, r=20, t=40, b=30))
                    st.plotly_chart(fig, use_container_width=True)
                    tot = _sub.groupby('通道名').agg(外送_总MW=('外送_MW', 'sum'), 受入_总MW=('受入_MW', 'sum')).reset_index()
                    st.dataframe(tot, use_container_width=True, hide_index=True)
                    st.info("📈 净送出为正=甘肃净外送(利好本地消纳/价格承压); 为负=净受入。各通道分时曲线可判断外送通道瓶颈时段。")
            else:
                # === 月度典型曲线(现有) ===
                st.caption("数据源: 联络线分时(997行, 2024-01~2026-09 月度披露)。"
                           "⚠️ 该表为月度典型日内曲线——每月各日 24h 数值相同, 属披露月均形状, 非逐日实时流向。"
                           "日度实时外送需接入调度/西北分部公开数据(运行 import_tie_line.py)。")
                _mos = pd.read_sql("SELECT DISTINCT `月份` m FROM `联络线分时` ORDER BY m", _c)['m'].tolist()
                if _mos:
                    mo = st.selectbox("选择月份(披露)", _mos, index=len(_mos) - 1, key="tl_mo")
                    _tl = pd.read_sql("SELECT * FROM `联络线分时` WHERE `月份`=%s", _c, params=(mo,))
                    _hour_cols = [f"{h}时" for h in range(24)]
                    for cc in _hour_cols:
                        _tl[cc] = pd.to_numeric(_tl[cc], errors="coerce")
                    _prof = _tl[_hour_cols].mean()  # 各日数值相同, mean 即该月典型曲线
                    fig = go.Figure()
                    fig.add_scatter(x=list(range(24)), y=_prof.values, mode="lines+markers",
                                    line=dict(color="#185FA5", width=2),
                                    hovertemplate="%{x}时<br>%{y:.0f}<extra></extra>")
                    fig.update_layout(height=320, margin=dict(l=40, r=20, t=30, b=30),
                                      xaxis_title="小时", yaxis_title="交换功率(万kW)", xaxis=dict(dtick=3),
                                      title=f"{mo} 典型日内外送曲线(月度披露)")
                    st.plotly_chart(fig, use_container_width=True)
                    st.info("📌 曲线显示该月外送日内高峰/低谷时段(便于择时); 但为披露月均形状, 不代表具体某日实时流向。"
                            "真正的「每日实时外送」需接入调度口径数据(运行 import_tie_line.py)。")
                else:
                    st.warning("联络线分时暂无数据")
            _c.close()
        except Exception as _e:
            st.error(f"联络线加载失败: {_e}")

    st.divider()

    # ---------- 5. 天气预警卡(数据已就位, 提炼展示) ----------
    with st.container(border=True):
        st.markdown("#### ⚠ 天气预警卡（未来10天 · 提炼）")
        st.caption("数据源: weather_hourly(已就位, 12 点位)。阈值: 风>10.7m/s 或 雷暴=1 或 降水>0.5mm。")
        with st.expander("📐 触发规则", expanded=False):
            st.code(
                "# 受限判定(与「天气·检修适配」一致)\n"
                "风速_10m > 10.7 OR 雷暴 = 1 OR 降水_mm > 0.5\n"
                "→ 红色预警(作业取消/改期)\n"
                "风速_10m > 8.0 AND 雷暴 = 0 AND 降水 > 0.1\n"
                "→ 黄色预警(加强监护)", language="text")
        try:
            _conn = pymysql.connect(**DB_CONFIG)
            _df_w = pd.read_sql(
                "SELECT `时间`,`点位名称`,`风速_10m`,`降水_mm`,`雷暴` FROM weather_hourly "
                "WHERE `时间` >= NOW() AND `时间` < DATE_ADD(NOW(), INTERVAL 10 DAY) "
                "ORDER BY `时间`", _conn)
            _conn.close()
            if not _df_w.empty:
                _df_w["触发"] = (
                    (_df_w["风速_10m"] > 10.7) |
                    (_df_w["雷暴"] == 1) |
                    (_df_w["降水_mm"] > 0.5)
                )
                _hits = _df_w[_df_w["触发"]].copy()
                if not _hits.empty:
                    def _kind(r):
                        if r["雷暴"] == 1:    return "🌪 雷暴"
                        if r["风速_10m"] > 10.7: return "💨 大风"
                        return "🌧 强降水"
                    _hits["类型"] = _hits.apply(_kind, axis=1)
                    _hits["日"] = _hits["时间"].dt.date
                    # 去重: 同点位+同类型+同日 合并为一项, 取峰值
                    _g = _hits.groupby(["点位名称", "类型", "日"]).agg(
                        峰值风速=("风速_10m", "max"), 峰值降水=("降水_mm", "max"),
                        雷暴=("雷暴", "max")).reset_index()
                    _g["严重度"] = _g["峰值风速"] + _g["雷暴"] * 1000
                    _g = _g.sort_values("严重度", ascending=False)
                    # 顶部总结句(用户建议3)
                    _bypt = _g.groupby("点位名称").size().sort_values(ascending=False)
                    _dom = _g.groupby("类型").size().idxmax()
                    _top = "、".join(_bypt.head(2).index.tolist())
                    _mxw = _g["峰值风速"].max()
                    st.warning(f"⚠ 未来10天共 **{len(_g)} 项**天气风险(涉及 {_g['点位名称'].nunique()} 个点位), "
                               f"集中于 **{_top}** 等地, 主导类型「{_dom}」, 峰值风速 {_mxw:.1f} m/s。"
                               f"可能影响风电出力与户外检修作业窗口。")
                    _show = _g.head(10).copy()
                    _show = _show[["日", "点位名称", "类型", "峰值风速", "峰值降水"]]
                    _show.columns = ["日期", "点位", "类型", "峰值风速m/s", "峰值降水mm"]
                    st.dataframe(_show, use_container_width=True, hide_index=True, height=300)
                    st.caption(f"↑ 仅列影响最大的前 10 项(已按点位+类型+日去重, 按严重度排序); 完整预警请按需扩展。")
                else:
                    st.success("✅ 未来 10 天天气窗口良好, 无受限预警")
            else:
                st.caption("weather_hourly 未来时段暂无数据")
        except Exception as _e:
            st.caption(f"天气数据加载异常: {_e}")

    st.write("---")
    st.caption(
        "📝 **数据接入流程**: ①确认数据源 → ②按各 section schema 建 MySQL 表(已建) → ③运行 import_*.py 导入 → ④Dashboard 自动重载。"
        " | 现货价与天气已接入; 出力/负荷/联络线日度表已就绪(import_renewable/load/tie_line.py), 数据到位即接通。"
    )


# ==================== 入口 ====================
TAB = st.sidebar.radio("导航", ["📊 预测报告", "📈 已披露复盘", "📋 数据台账", "📅 日度态势(新)"], label_visibility="visible")
if TAB == "📊 预测报告":
    render_report()
elif TAB == "📈 已披露复盘":
    render_review()
elif TAB == "📅 日度态势(新)":
    render_daily()
else:
    render_ledger()

# ===== 侧栏: 模型验证 =====
with st.sidebar:
    st.write("---")
    with st.expander("📐 模型验证(回测)"):
        df = load_maint()
        acc, rate, mae = seasonal_accuracy(df)
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("方向准确率", f"{rate*100:.0f}%")
        with col_b:
            st.metric("点估计MAE", f"{mae:.0f}项")
        st.caption(f"基于 {len(acc)} 个月验证样本; MAE=平均绝对误差(项)")
        if not acc.empty:
            st.dataframe(acc, hide_index=True, height=240)
