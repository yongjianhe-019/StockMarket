"""
卖出信号 v3 — 多维度泡沫检测器

泡沫的定义：估值贵 + (价格加速/量价背离/宏观恶化) ≥2 类 → 泡沫确认
借鉴：
- 广发证券：PE见顶先于价格见顶，真正的顶部=盈利增速拐点+情绪极端
- "买在无人问津，卖在人声鼎沸"：市场反应度>70%注意减仓
- 量价经典：价格加速赶顶+成交量背离=聪明钱离场

三级泡沫 → 分批卖出：
- 轻度泡沫 (3~4信号): 卖出25%
- 中度泡沫 (5~6信号): 卖出50%
- 严重泡沫 (7+信号):  卖出75%
"""

from __future__ import annotations
import numpy as np
import pandas as pd


def is_bubble(macro_df: pd.DataFrame, csi300_val=None, csi2000_val=None,
              idx: int = -1, daily_300: pd.DataFrame = None) -> dict:
    """
    多维度泡沫检测。

    Parameters
    ----------
    macro_df : 宏观数据
    csi300_val : CSI300 估值数据
    daily_300 : CSI300 日线数据（用于价格加速和量价背离检测）
    idx : 当前日期在 macro_df 中的索引

    Returns
    -------
    {
        is_bubble: bool,
        level: str,          # 轻度泡沫 / 中度泡沫 / 严重泡沫
        sell_pct: float,     # 建议卖出比例 (0.25/0.50/0.75)
        reasons: [str],
        signals: dict,       # 各类信号详情
        pe_pct: float,
    }
    """
    hist = macro_df.iloc[:max(idx + 1, 1)] if idx >= 0 else macro_df
    row = hist.iloc[-1]

    # ═══════════════════════════════════════
    # 前提：PE 必须贵 (>60% 分位)
    # ═══════════════════════════════════════
    pe_pct = _pe_percentile(csi300_val, row['date'])
    if pe_pct is None or pe_pct < 0.60:
        return {
            "is_bubble": False, "level": "PE合理", "sell_pct": 0.0,
            "reasons": [], "signals": {}, "pe_pct": pe_pct,
        }

    # ═══════════════════════════════════════
    # A类: 价格加速赶顶（动量泡沫）
    # ═══════════════════════════════════════
    signals_a = _check_price_acceleration(daily_300, row['date']) if daily_300 is not None else []

    # ═══════════════════════════════════════
    # B类: 量价背离（资金离场）
    # ═══════════════════════════════════════
    signals_b = _check_volume_divergence(daily_300, row['date']) if daily_300 is not None else []

    # ═══════════════════════════════════════
    # C类: 宏观恶化（基本面拐点 — 保留原有）
    # ═══════════════════════════════════════
    signals_c = _check_macro_deterioration(hist)

    # ═══════════════════════════════════════
    # 汇总判断：≥2 类触发 → 泡沫
    # ═══════════════════════════════════════
    categories_triggered = 0
    if signals_a: categories_triggered += 1
    if signals_b: categories_triggered += 1
    if signals_c: categories_triggered += 1

    all_signals = signals_a + signals_b + signals_c
    total_count = len(all_signals)

    if categories_triggered >= 2:
        if total_count >= 7:
            level, sell_pct = "严重泡沫", 0.75
        elif total_count >= 5:
            level, sell_pct = "中度泡沫", 0.50
        else:
            level, sell_pct = "轻度泡沫", 0.25

        return {
            "is_bubble": True,
            "level": level,
            "sell_pct": sell_pct,
            "reasons": all_signals,
            "signals": {
                "A_价格加速": signals_a,
                "B_量价背离": signals_b,
                "C_宏观恶化": signals_c,
                "触发类别数": categories_triggered,
                "总信号数": total_count,
            },
            "pe_pct": pe_pct,
        }
    else:
        return {
            "is_bubble": False,
            "level": f"PE偏高但仅{categories_triggered}/3类触发",
            "sell_pct": 0.0,
            "reasons": all_signals,
            "signals": {
                "A_价格加速": signals_a,
                "B_量价背离": signals_b,
                "C_宏观恶化": signals_c,
            },
            "pe_pct": pe_pct,
        }


# ═══════════════════════════════════════════
# A类: 价格加速赶顶
# ═══════════════════════════════════════════

def _check_price_acceleration(daily: pd.DataFrame, date) -> list:
    """检测价格是否在加速赶顶。"""
    if daily is None or daily.empty:
        return []

    d = daily[daily['date'] <= date]
    if len(d) < 126:
        return []

    signals = []
    close = d['close']
    current = float(close.iloc[-1])

    # 1) 6个月涨幅 > 30% = 强势上涨
    ret_6m = current / float(close.iloc[-min(len(d), 126)]) - 1
    if ret_6m > 0.30:
        signals.append(f"6月涨幅{ret_6m:.1%} (加速赶顶)")

    # 2) 1个月涨幅 > 10% + 6个月 > 20% = 末段冲刺
    ret_1m = current / float(close.iloc[-min(len(d), 21)]) - 1
    if ret_1m > 0.10 and ret_6m > 0.20:
        if f"6月涨幅{ret_6m:.1%} (加速赶顶)" not in signals:
            signals.append(f"1月涨幅{ret_1m:.1%}+6月{ret_6m:.1%} (末段冲刺)")

    # 3) 价格偏离 MA60 > 30% = 极度超买
    if len(d) >= 60:
        ma60 = float(close.tail(60).mean())
        deviation = current / ma60 - 1
        if deviation > 0.30:
            signals.append(f"偏离MA60={deviation:.1%} (极度超买)")

    # 4) 3个月RSI式检测：连续上涨后开始钝化
    if len(d) >= 63:
        ret_3m = current / float(close.iloc[-63]) - 1
        ret_1w = current / float(close.iloc[-min(len(d), 5)]) - 1
        if ret_3m > 0.20 and ret_1w < 0.02:
            signals.append(f"3月涨{ret_3m:.1%}但近周滞涨{ret_1w:.2%} (动能衰竭)")

    return signals


# ═══════════════════════════════════════════
# B类: 量价背离
# ═══════════════════════════════════════════

def _check_volume_divergence(daily: pd.DataFrame, date) -> list:
    """检测量价背离——聪明钱离场信号。"""
    if daily is None or daily.empty or 'volume' not in daily.columns:
        return []

    d = daily[daily['date'] <= date]
    if len(d) < 60:
        return []

    signals = []
    close = d['close']
    volume = d['volume']
    current = float(close.iloc[-1])

    # 1) 价格近1年高位，但20日均量 < 60日均量 × 0.7
    high_1y = float(close.tail(252).max()) if len(d) >= 252 else float(close.max())
    if current > high_1y * 0.90:  # 接近1年高点
        vol_ma20 = float(volume.tail(20).mean())
        vol_ma60 = float(volume.tail(60).mean()) if len(d) >= 60 else vol_ma20
        if vol_ma20 < vol_ma60 * 0.7:
            signals.append(f"近高点量缩 (20日均量/60日={vol_ma20/vol_ma60:.1%})")

    # 2) 连续3月价格上涨但月均量递减
    if len(d) >= 63:
        close_monthly = [float(close.iloc[-21 * i - 1]) for i in range(3, 0, -1)] + [current]
        vol_monthly = [float(volume.iloc[-21 * i: -21 * (i - 1)].mean()) if i > 1
                        else float(volume.tail(21).mean()) for i in range(3, 0, -1)]
        # 价格逐月上升
        if close_monthly[-1] > close_monthly[0]:
            # 成交量逐月递减
            if len(vol_monthly) >= 2 and all(vol_monthly[i] < vol_monthly[i-1] * 0.95
                                               for i in range(1, len(vol_monthly))):
                signals.append("价升量缩(连续3月)")

    # 3) 换手率极端高但近5日回落（天量见天价）
    if len(d) >= 252:
        vol_pct = _series_pct(volume, min(len(volume), 252))
        if vol_pct > 0.90:  # 成交量在近1年最高10%
            vol_5d_avg = float(volume.tail(5).mean())
            vol_20d_max = float(volume.tail(20).max())
            if vol_5d_avg < vol_20d_max * 0.6:
                signals.append(f"天量后急缩 (量分位{vol_pct:.0%})")

    return signals


# ═══════════════════════════════════════════
# C类: 宏观恶化（保留原有逻辑）
# ═══════════════════════════════════════════

def _check_macro_deterioration(hist: pd.DataFrame) -> list:
    """检测宏观经济恶化信号。"""
    warnings = []

    # 1. 中美利差急剧扩大（>0.5%/半年）
    sp = hist['spread_10y'].dropna()
    if len(sp) >= 126:
        sp_widen = sp.iloc[-1] - sp.iloc[-126]
        if sp_widen < -0.5:
            warnings.append(f"利差扩大{sp_widen:+.2f}%/6月")

    # 2. 人民币贬值 >3%/3月
    usd = hist['usd_cny'].dropna()
    if len(usd) >= 63:
        chg = (usd.iloc[-1] - usd.iloc[-63]) / usd.iloc[-63]
        if chg > 0.03:
            warnings.append(f"人民币贬{chg:.1%}/3月")

    # 3. 黄金异常飙升 >15%/3月
    gold = hist['gold_price'].dropna()
    if len(gold) >= 63:
        chg = (gold.iloc[-1] - gold.iloc[-63]) / gold.iloc[-63]
        if chg > 0.15:
            warnings.append(f"黄金飙{chg:.1%}/3月")

    # 4. M2 急剧收紧
    if 'm2_yoy' in hist.columns:
        m2 = hist['m2_yoy'].dropna()
        if len(m2) >= 6:
            if m2.iloc[-1] - m2.iloc[-6] < -1.5:
                warnings.append(f"M2收紧({m2.iloc[-6]:.1f}%→{m2.iloc[-1]:.1f}%)")

    # 5. 美联储加息
    if 'fed_rate' in hist.columns:
        fed = hist['fed_rate'].dropna()
        if len(fed) >= 2 and fed.iloc[-1] > fed.iloc[-2]:
            warnings.append(f"美联储加息至{fed.iloc[-1]:.2f}%")

    # 6. 社融急剧收缩
    sf = hist['social_finance'].dropna()
    if len(sf) >= 12:
        if sf.iloc[-6:].mean() < sf.iloc[-12:-6].mean() * 0.8:
            warnings.append("社融大幅收缩")

    return warnings


# ═══════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════

def _pe_percentile(val_df, date):
    """计算当前 PE 在历史上的分位数。"""
    if val_df is None or val_df.empty:
        return None
    v = val_df[val_df['date'] <= date]
    if v.empty or 'pe' not in v.columns:
        return None
    pe = v['pe'].dropna()
    if len(pe) < 252:
        return None
    pe_now = float(pe.iloc[-1])
    return float((pe < pe_now).sum() / len(pe))


def _series_pct(series: pd.Series, window: int) -> float:
    """计算序列最新值在窗口内的分位数。"""
    s = series.dropna()
    if len(s) < window:
        window = len(s)
    recent = s.tail(window)
    latest = recent.iloc[-1]
    return float((recent < latest).sum() / len(recent))


def is_ice_point(csi300_score: float, pe_pct: float = None) -> bool:
    """冰点判断：分数≥50。"""
    return csi300_score >= 50
