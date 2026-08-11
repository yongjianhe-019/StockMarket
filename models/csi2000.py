"""
CSI 2000 打分模型 v2 — 流动性 + 动量驱动

不依赖 PE/PB（微盘盈利不稳定，估值数据少），
改用宏观流动性 + 趋势动量 + 相对强弱。

参考模型：
- dao-quant M2/PPI/波动率 三因子 (累计+307%)
- 平安信用-通胀时钟 小盘象限
- etf-rotation-strategy 动量打分
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SCORE_WAIT = 50
SCORE_WATCH = 65
SCORE_BUY = 80

TRADING_DAYS = 252


def compute_csi2000_score(daily: pd.DataFrame,
                          valuation: Optional[pd.DataFrame],
                          csi300_daily: pd.DataFrame,
                          macro: pd.DataFrame = None) -> dict:
    """
    CSI 2000 流动性+动量 打分。

    Parameters
    ----------
    daily : CSI 2000 指数日线
    valuation : 估值数据（仅 PB 参考）
    csi300_daily : CSI 300 日线（计算相对强弱）
    macro : 宏观数据（M2/社融/利差等），可选

    Returns
    -------
    {total_score, season, action, details}
    """
    latest_date = daily["date"].max()
    price_latest = float(daily["close"].iloc[-1])
    n = len(daily)

    # ═══════════════════════════════════════
    # 1. 宏观流动性 (35分)
    # ═══════════════════════════════════════
    s_macro, d_macro = 0.0, {}
    if macro is not None and not macro.empty:
        s_macro, d_macro = _score_macro_liquidity(macro)

    # ═══════════════════════════════════════
    # 2. 趋势动量 (30分)
    # ═══════════════════════════════════════
    s_trend, d_trend = _score_trend_momentum(daily)

    # ═══════════════════════════════════════
    # 3. 相对强弱 vs CSI300 (20分)
    # ═══════════════════════════════════════
    s_rs, d_rs = _score_relative_strength(daily, csi300_daily)

    # ═══════════════════════════════════════
    # 4. 市场结构 (15分)
    # ═══════════════════════════════════════
    s_struct, d_struct = _score_market_structure(daily)

    # ═══════════════════════════════════════
    # 5. 宏观底部确认 (加分项, +5)
    # ═══════════════════════════════════════
    s_macro_btm, d_macro_btm = 0.0, {"状态": "无宏观数据"}
    if macro is not None and not macro.empty:
        m = macro[macro["date"] <= latest_date]
        if not m.empty:
            confirmations = []
            if "pmi" in m.columns:
                pmi_data = m["pmi"].dropna()
                if len(pmi_data) >= 3:
                    pmi_now = pmi_data.iloc[-1]
                    pmi_1m = pmi_data.iloc[-2]
                    pmi_2m = pmi_data.iloc[-3]
                    if pmi_now >= pmi_1m and pmi_1m >= pmi_2m:
                        confirmations.append(f"PMI企稳({pmi_now:.1f})")
            if "m2_yoy" in m.columns:
                m2_data = m["m2_yoy"].dropna()
                if len(m2_data) >= 3:
                    m2_now = m2_data.iloc[-1]
                    m2_3m = m2_data.iloc[-min(4, len(m2_data))]
                    if m2_now > m2_3m:
                        confirmations.append(f"M2回升({m2_now:.1f}%)")
            if "social_finance" in m.columns:
                sf = m["social_finance"].dropna()
                if len(sf) >= 6:
                    sf_recent = sf.iloc[-3:].mean()
                    sf_prior = sf.iloc[-6:-3].mean()
                    if sf_recent >= sf_prior * 0.95:
                        confirmations.append("社融稳定")

            if len(confirmations) >= 2:
                s_macro_btm = 5.0
                d_macro_btm = {"确认信号": "; ".join(confirmations), "等级": "宏观底部确认", "得分": 5.0}
            elif len(confirmations) == 1:
                s_macro_btm = 2.0
                d_macro_btm = {"确认信号": confirmations[0], "等级": "宏观部分企稳", "得分": 2.0}
            else:
                d_macro_btm = {"状态": "宏观仍在恶化或数据不足"}

    # ═══════════════════════════════════════
    # 汇总
    # ═══════════════════════════════════════
    total = s_macro + s_trend + s_rs + s_struct + s_macro_btm

    # 趋势质量过滤：R²<0.3 时趋势不可信，对动量分数打折
    r2_val = d_trend.get("趋势质量R²", 0)
    if isinstance(r2_val, str):
        try: r2_val = float(r2_val)
        except: r2_val = 0
    if r2_val < 0.3 and s_trend > 5:
        penalty = min(s_trend * 0.5, 10)  # 趋势不可信，最多扣10分
        total -= penalty
        d_trend["趋势质量惩罚"] = f"R²={r2_val:.2f}<0.3，动量打折 -{penalty:.0f}分"

    if total >= SCORE_BUY:
        season, action = "深冬", "补仓"
    elif total >= SCORE_WATCH:
        season, action = "冬天", "首次建仓"
    elif total >= SCORE_WAIT:
        season, action = "秋末", "关注"
    else:
        season, action = "夏/秋", "等待"

    return {
        "total_score": round(total, 1),
        "season": season,
        "action": action,
        "details": {
            "宏观流动性(35)": d_macro,
            "趋势动量(30)": d_trend,
            "相对强弱(20)": d_rs,
            "市场结构(15)": d_struct,
            "宏观底部确认(+5)": d_macro_btm,
        },
        "price": price_latest,
        "signal_date": latest_date,
    }


# ═══════════════════════════════════════════
# 维度 1: 宏观流动性 (35分)
# ═══════════════════════════════════════════

def _score_macro_liquidity(macro: pd.DataFrame) -> tuple[float, dict]:
    """M2 + 社融 + 利差 = 流动性环境。"""
    scores = {}

    # M2增速：>10%强烈利好小盘, >8%利好, <6%利空
    if "m2_yoy" in macro.columns:
        m2 = macro["m2_yoy"].dropna().iloc[-1]
        m2_chg = macro["m2_yoy"].dropna().diff(3).iloc[-1] if len(macro["m2_yoy"].dropna()) > 3 else 0
        if m2 > 12:     s = 15
        elif m2 > 10:   s = 12
        elif m2 > 8:    s = 8
        elif m2 > 6:    s = 4
        else:           s = 0
        if m2_chg > 0.5: s = min(s + 2, 15)
        scores["M2"] = f"{m2:.1f}% → {s}分"
    else:
        s = 0
        scores["M2"] = "无数据"

    # 信用环境：社融扩张
    if "social_finance" in macro.columns:
        sf = macro["social_finance"].dropna()
        if len(sf) >= 12:
            recent = sf.iloc[-6:].mean()
            prior = sf.iloc[-12:-6].mean()
            sf_ratio = recent / prior if prior > 0 else 1
            if sf_ratio > 1.2:      s_sf = 10
            elif sf_ratio > 1.05:   s_sf = 7
            elif sf_ratio > 0.95:   s_sf = 4
            else:                   s_sf = 0
        else:
            s_sf = 4
    else:
        s_sf = 0
    scores["信用"] = f"社融近6/前6月={sf_ratio:.2f} → {s_sf}分" if 'sf_ratio' in dir() else "无数据"

    # 中美利差：倒挂程度
    if "spread_10y" in macro.columns:
        sp = macro["spread_10y"].dropna().iloc[-1]
        if sp > -1.5:       s_sp = 10
        elif sp > -2.5:     s_sp = 6
        elif sp > -3.5:     s_sp = 3
        else:               s_sp = 0
    else:
        s_sp = 0
    scores["利差"] = f"{sp:.1f}% → {s_sp}分" if 'sp' in dir() else "无数据"

    total = s + (s_sf if 's_sf' in dir() else 4) + (s_sp if 's_sp' in dir() else 5)
    scores["总分"] = total
    return total, scores


# ═══════════════════════════════════════════
# 维度 2: 趋势动量 (30分)
# ═══════════════════════════════════════════

def _score_trend_momentum(daily: pd.DataFrame) -> tuple[float, dict]:
    """价格趋势 + 动量强度。"""
    close = daily["close"]
    n = len(close)
    if n < 60:
        return 0, {"状态": "数据不足"}

    current = float(close.iloc[-1])

    # 均线位置
    ma20 = float(close.tail(20).mean())
    ma60 = float(close.tail(60).mean())
    above_ma20 = current > ma20
    above_ma60 = current > ma60

    # 近期收益
    ret_1m = float(close.iloc[-1] / close.iloc[-min(n, 21)] - 1)
    ret_3m = float(close.iloc[-1] / close.iloc[-min(n, 63)] - 1)
    ret_6m = float(close.iloc[-1] / close.iloc[-min(n, 126)] - 1)

    # 动量打分
    s = 0
    if above_ma20 and above_ma60:      s += 15  # 多头排列
    elif above_ma20:                   s += 10
    elif above_ma60:                   s += 5

    # 动量强度（正收益加分，负收益不加）
    if ret_3m > 0.15:   s += 10   # 强势上涨
    elif ret_3m > 0.05: s += 7
    elif ret_3m > 0:    s += 3

    # 动量质量（趋势是否稳定，用 R² 近似）
    if n >= 60:
        y = np.log(close.tail(60).values)
        x = np.arange(60)
        slope = np.polyfit(x, y, 1)[0]
        y_pred = slope * x + np.mean(y) - slope * np.mean(x)
        r2 = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)
        if r2 > 0.7:  s += 5
        elif r2 > 0.4: s += 2
    else:
        r2 = 0

    detail = {
        "均线": f"{'多头' if above_ma20 and above_ma60 else '偏多' if above_ma20 else '偏空'}",
        "3月收益": f"{ret_3m:.1%}",
        "趋势质量R²": f"{r2:.2f}" if 'r2' in dir() else "N/A",
        "得分": s,
    }
    return s, detail


# ═══════════════════════════════════════════
# 维度 3: 相对强弱 vs CSI300 (20分)
# ═══════════════════════════════════════════

def _score_relative_strength(daily: pd.DataFrame, csi300: pd.DataFrame) -> tuple[float, dict]:
    """CSI2000/CSI300 比值趋势：上升=小盘强。"""
    merged = daily[["date", "close"]].merge(
        csi300[["date", "close"]], on="date", suffixes=("_2000", "_300"), how="inner"
    )
    if len(merged) < 20:
        return 0, {"状态": "数据不足"}

    merged["ratio"] = merged["close_2000"] / merged["close_300"]
    n = len(merged)

    current_ratio = float(merged["ratio"].iloc[-1])

    # 比值趋势
    ma20 = float(merged["ratio"].tail(20).mean())
    ma60 = float(merged["ratio"].tail(60).mean()) if n >= 60 else ma20

    ratio_trend = "up" if current_ratio > ma20 else "down"

    # 历史分位：比值越低=小盘越被抛弃=抄底机会
    pct = (merged["ratio"] < current_ratio).sum() / n

    s = 0
    if pct < 0.20:        s = 20  # 极度弱势，抄底
    elif pct < 0.35:      s = 15
    elif pct < 0.50:      s = 10
    elif ratio_trend == "up": s = 10  # 趋势向上，跟随
    elif ratio_trend == "down": s = 3

    return s, {
        "比值分位": f"{pct:.0%}",
        "趋势": "小盘走强" if ratio_trend == "up" else "小盘走弱",
        "得分": s,
    }


# ═══════════════════════════════════════════
# 维度 4: 市场结构 (15分)
# ═══════════════════════════════════════════

def _score_market_structure(daily: pd.DataFrame) -> tuple[float, dict]:
    """波动率 + 成交额 结构分析。"""
    n = len(daily)
    if n < 60:
        return 0, {"状态": "数据不足"}

    close = daily["close"]

    # 波动率收敛 = 蓄势
    returns = close.pct_change().dropna()
    vol_20d = float(returns.tail(20).std() * np.sqrt(TRADING_DAYS))
    vol_60d = float(returns.tail(60).std() * np.sqrt(TRADING_DAYS)) if len(returns) >= 60 else vol_20d
    vol_ratio = vol_20d / vol_60d if vol_60d > 0 else 1

    s = 0
    if vol_ratio < 0.6:     s += 8   # 波动率大幅收敛，蓄势待发
    elif vol_ratio < 0.8:   s += 5
    elif vol_ratio > 1.5:   s += 2   # 高波动中，有交易机会

    # 成交额趋势
    if "volume" in daily.columns:
        vol = daily["volume"]
        vol_ma20 = float(vol.tail(20).mean())
        vol_ma60 = float(vol.tail(60).mean()) if n >= 60 else vol_ma20
        vol_ratio_v = vol_ma20 / vol_ma60 if vol_ma60 > 0 else 1

        if vol_ratio_v > 1.3:       s += 7   # 放量
        elif vol_ratio_v > 1.1:     s += 4
        elif vol_ratio_v > 0.7:     s += 2
        else:                       s += 0   # 极度缩量
    else:
        vol_ratio_v = 1

    return s, {
        "波动率": f"20日={vol_20d:.1%}  vs 60日={vol_60d:.1%} (比={vol_ratio:.2f})",
        "成交量": f"20日均/60日均={vol_ratio_v:.2f}",
        "得分": s,
    }


# ═══════════════════════════════════════════
# 入口
# ═══════════════════════════════════════════

def analyze_csi2000(data: dict, macro_df: pd.DataFrame = None) -> dict:
    """从 fetch_all_data() + 宏观数据 结果中分析 CSI 2000。"""
    return compute_csi2000_score(
        daily=data["csi2000_daily"],
        valuation=data.get("csi2000_valuation"),
        csi300_daily=data["csi300_daily"],
        macro=macro_df,
    )
