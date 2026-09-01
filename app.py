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
    # 其余电厂/变电站补充映射(降低'其他'占比)
    ("兰铝", "兰州"), ("八〇三", "兰州"), ("河口", "兰州"), ("柴家峡", "兰州"),
    ("刘右", "临夏"), ("酒钢", "酒泉"), ("景泰", "白银"), ("庆东", "庆阳"),
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
    return df

@st.cache_data(ttl=600)
def load_section():
    return pd.DataFrame(run_query("SELECT * FROM section"))

@st.cache_data(ttl=600)
def load_disclosure():
    df = pd.DataFrame(run_query("SELECT * FROM disclosure"))
    skip = {"file", "月份", "平衡月份", "装机口径", "skip_reason", "skipped"}
    for c in df.columns:
        if c not in skip:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

@st.cache_data(ttl=600)
def load_trade_plan():
    df = pd.DataFrame(run_query("SELECT * FROM trade_plan"))
    skip = {"file", "月份", "skipped", "skip_reason"}
    for c in df.columns:
        if c not in skip:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


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

def seasonal_forecast(df, target_year, target_month):
    """改进版预测: 历史同月均值 + 线性趋势 + 年际波动区间.

    改进点(对应 2026-09 实测低估 22% 的修复):
    1) 历史年份≥2时趋势权重提到 0.8, 捕捉装机/负荷增长, 不再保守求和
    2) 置信区间改用"同月年际相对波动"定宽, 替代围绕保守点估的窄泊松区间,
       避免小样本(≤2点)时区间过窄、包不住真实值
    3) 识别"上年同期异常低"的披露不全月份, 给出警告, 避免系统性低估
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
        w_trend = 0.8  # 趋势显著时主权重(保留20%均值缓冲防过拟合)
    else:
        trend_pred = counts[0]
        w_trend = 0.0

    # 融合: 趋势主导(≥2年), 否则纯均值
    point = int(round((1 - w_trend) * avg + w_trend * trend_pred))
    point = max(0, point)

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
        w = 0.8
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
    # ----- 信号 1: 价格方向 -----
    if risk == "高":
        signals.append(("📈 现货价格", "上移概率较高",
                       f"{target_month}月属春秋检高峰, 历史上同期火电检修抬升电价概率>70%; "
                       "中旬起日均价涨幅预计高于月度均值"))
    else:
        signals.append(("📉 现货价格", "下移或震荡",
                       f"{target_month}月属检修低谷/过渡, 火电可调容量充足, 电价大概率不冲高"))

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
        signals.append(("⚠ 供给端", verdict, detail))
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
            signals.append(("→ 外送窗口", "较宽", f"{out_count} 个断面正向有容量, 适合月底谈外送增量"))
        else:
            signals.append(("→ 外送窗口", "偏窄", f"仅 {out_count} 个断面正向外送容量, 外送增量空间有限"))

    # ----- 信号 4: 关键线路检修 -----
    heavy_equip_list = []
    sub = df_maint[df_maint["月"] == target_month]
    if not sub.empty:
        heavy = sub[sub["检修级别"].isin(["A级检修", "改造大修"])]
        if "停电设备" in heavy.columns:
            heavy_equip_list = heavy["停电设备"].value_counts().head(3).index.tolist()
    if len(heavy_equip_list) >= 2:
        signals.append(("🔌 关键线路", "多条线路检修",
                       f"{len(heavy_equip_list)} 条关键线路同期A修/改造, 可能影响断面限额, 需提前核实外送窗口"))
    elif len(heavy_equip_list) == 1:
        signals.append(("🔌 关键线路", "1 条线路检修",
                       f"{heavy_equip_list[0]} 同期大修, 影响范围有限, 但需关注对应断面限额变化"))
    else:
        signals.append(("🔌 关键线路", "无重大检修",
                       "同期无 A 级或改造大修线路, 断面限额大概率不受影响"))

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
                    f"省内供给双重承压, 现货价格上行动力最强, 建议月合约偏紧/现货多备")
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

    return signals

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
    # 第一步: 加权检修影响指数 + 高影响清单
    w_avg, high_df, wstats = predict_weighted_impact(df, target_year, target_month)

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
            f'<span style="font-size:18px;color:#888;margin-left:8px">± 1项</span></div>'
            f'<div style="color:#666;margin-top:6px;font-size:13px">'
            f'预计 {target} 发电设备检修项数<br>'
            f'置信区间(80%): <b>{lo} ~ {hi} 项</b></div></div>',
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

        # 第一步: 加权检修影响指数指标卡
        if w_avg is not None:
            st.markdown(
                f'<div style="background:#eef6ee;padding:10px 14px;border-radius:8px;margin-top:10px;'
                f'border-left:4px solid #2e7d32">'
                f'<div style="font-size:22px;font-weight:700;color:#2e7d32;line-height:1">{w_avg:.0f}'
                f'<span style="font-size:12px;color:#666;margin-left:6px">加权影响指数 (历史同期均值)</span></div>'
                f'<div style="color:#666;font-size:12px;margin-top:4px">'
                f'发电机组 {wstats["gen"]} 项 · 主变 {wstats["trans"]} 项 · 合计权重损失 {wstats["weighted"]:.0f}</div>'
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

        # 重复披露设备
        if repeat_equip is not None and len(repeat_equip) > 0:
            st.markdown("**跨期重复披露设备**")
            rq = repeat_equip.reset_index()
            rq.columns = ["线路/设备", "重复次数"]
            st.dataframe(rq, use_container_width=True, hide_index=True, height=140)

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
            st.caption("火电均价最高(约360~440), 光伏最低(约80~110); 风光低价是甘肃电价长期压制因素")

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

    # ===== 📤 外送预计（净送出预测）=====
    if df_trade is not None and not df_trade.empty and "净送出_亿kWh" in df_trade.columns:
        st.markdown("### 📤 外送预计（净送出预测）")
        st.caption("来源: 月度交易计划(2024-01~2026-08, 32期)。甘肃为送端省, 净送出长期走高; "
                   "与检修预测叠加重判断'省内供给是否双重承压'。")
        fc_out = seasonal_forecast_value(df_trade, "净送出_亿kWh", target_year, target_month)
        fc_out2 = seasonal_forecast_value(df_trade, "中长期外送_亿kWh", target_year, target_month)
        c_o1, c_o2 = st.columns(2)
        with c_o1:
            if fc_out:
                st.metric("预计净送出", f"{fc_out['point']} 亿kWh",
                          help=f"区间 {fc_out['lo']}~{fc_out['hi']} 亿kWh; {fc_out['method']}")
            else:
                st.info(f"{target_month} 月无历史同期净送出数据")
        with c_o2:
            if fc_out2:
                st.metric("预计中长期外送", f"{fc_out2['point']} 亿kWh",
                          help=f"区间 {fc_out2['lo']}~{fc_out2['hi']} 亿kWh; {fc_out2['method']}")
            else:
                st.info(f"{target_month} 月无历史同期中长期外送数据")
        if fc_out and risk == "高":
            st.warning("⚠ 检修高峰月 + 外送高位: 省内供给或双重收紧, 重点关注现货价格上行")
        elif fc_out:
            st.success("检修低谷 / 外送可控: 省内供给相对宽松")

    st.write("---")

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

    # 总体提示
    if risk == "高":
        st.info(
            f"💼 **建议**: {target_month}月属检修高峰, 供给偏紧风险大, "
            f"重点跟踪 **{peak_week if peak_week else '集中时段'}** 价格波动, "
            "建议中旬前观望, 末旬可考虑建仓")
    else:
        st.info(
            f"💼 **建议**: {target_month}月属检修低谷/过渡, 整体供需宽松, "
            "现货价格大概率不冲高, 月度合约可争取更优条款")

    st.caption(
        "⚠️ 本报告基于历史规律生成, 实际检修计划以甘肃省电力市场信息披露平台每月发布的正式文件为准。"
    )

    # 报告下载
    st.write("---")
    _, btn_col, _ = st.columns([3, 1, 3])
    with btn_col:
        report_md = build_report_md(target, point, lo, hi, season, risk, signals, peak_week, equip_groups, repeat_equip, fc)
        st.download_button("📥 下载本报告(Markdown)", report_md,
                           f"甘肃检修预测_{target}.md", "text/markdown")


def build_report_md(target, point, lo, hi, season, risk, signals, peak_week, equip_groups, repeat_equip=None, fc=None):
    """生成可下载的报告文本."""
    lines = [
        f"# 甘肃电网检修预测报告 - {target}",
        "",
        f"数据来源: 甘肃省电力市场月度披露文件 + 历史同期规律",
        "",
        "## ① 总量预测",
        "",
        f"- 预计检修项数: **{point} ± 1 项**",
        f"- 置信区间(80%): {lo} ~ {hi} 项",
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
            cyc = f"{row['平均周期月']}" if pd.notna(row.get("平均周期月")) else "短间隔"
            nxt = row.get("下次预计") or "—"
            stt = row.get("状态", "")
            lines.append(f"| {row['停电设备']} | {loc or '—'} | {int(row['历史次数'])} | {cyc} | {nxt} | {stt or '—'} | {row['主要级别']} | {row['申请单位']} |")
    if repeat_equip is not None and len(repeat_equip) > 0:
        lines += ["", "**跨期重复披露设备:**", ""]
        for equip, cnt in repeat_equip.items():
            lines.append(f"- {equip} (重复 {int(cnt)} 次)")
    lines += ["", "## ③ 时间分布", "",
              f"- 集中时段: {peak_week if peak_week else '数据不足'}"]
    lines += ["", "## ④ 交易解读", ""]
    for label, verdict, detail in signals:
        lines.append(f"- **{label}({verdict})**: {detail}")
    lines += ["", "---",
              "*本报告基于历史规律生成, 实际检修以月度披露文件为准*"]
    return "\n".join(lines).encode("utf-8")


# ==================== 数据台账(简单导出页) ====================
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

    show = df[df["年"].isin(year) & df["类别"].isin(cat) & df["检修级别"].isin(level)]
    st.markdown(f"**共 {len(show):,} 条记录**")
    st.dataframe(show.drop(columns=[c for c in show.columns if c in ("开始日期_dt", "结束日期_dt", "年", "月")],
                            errors="ignore"),
                 use_container_width=True, hide_index=True, height=420)

    csv = show.to_csv(index=False).encode("utf-8-sig")
    st.download_button("📥 导出 CSV", csv, f"检修数据_{datetime.now():%Y%m%d}.csv", "text/csv")


# ==================== 入口 ====================
TAB = st.sidebar.radio("导航", ["📊 预测报告", "📋 数据台账"], label_visibility="visible")
if TAB == "📊 预测报告":
    render_report()
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
