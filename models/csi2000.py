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
        s_macro, d_macro = _score_macro_liquidity(macro, latest_date)

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
            if "pmi_manufacturing" in m.columns:
                pmi_data = m["pmi_manufacturing"].dropna()
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
    # 6. 两融出清确认 (加分项, +13)
    #    V型急跌底盲区修复(2026-08-28): 杠杆快速出清+站回MA20+量能回升 → 确认底
    #    预研 margin_flush_study.py: 历史0飞刀、2023阴跌不触发
    # ═══════════════════════════════════════
    s_flush, d_flush = _score_margin_flush(daily, macro, latest_date)

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

    # 确认类加分在 R² 惩罚之后：动量质量差只惩罚动量分，不影响出清确认
    total += s_flush

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
            "两融出清确认(+13)": d_flush,
        },
        "price": price_latest,
        "signal_date": latest_date,
    }


# ═══════════════════════════════════════════
# 维度 1: 宏观流动性 (35分)
# ═══════════════════════════════════════════

def _score_rate_percentile(series: pd.Series, max_score: float,
                           lookback: int = 1260, min_periods: int = 756) -> tuple[float, float | None]:
    """利率水平 5 年滚动分位打分：低分位(利率低=货币宽松)加分。

    绝对阈值会时代漂移（cn_2y 从 2015 年 3.4% 降到 2026 年 1.25%），
    必须用相对自身历史的分位。返回 (分数, 当前分位)。
    """
    s = pd.Series(series).dropna()
    if len(s) < min_periods:
        return 0.0, None
    cur = float(s.iloc[-1])
    pct = float((s.tail(lookback) < cur).mean())
    if pct < 0.20:      return max_score, pct
    elif pct < 0.40:    return round(max_score * 0.7), pct
    elif pct < 0.60:    return round(max_score * 0.4), pct
    elif pct < 0.80:    return round(max_score * 0.2), pct
    else:               return 0.0, pct


def _score_macro_liquidity(macro: pd.DataFrame,
                           latest_date=None) -> tuple[float, dict]:
    """流动性环境 = M2 + 社融 + 中国利率 + 全球流动性（35分）。

    2026-08-30 重构（数据接入，研究依据 /tmp/study_unused_fields.py）:
    - 原结构 M2(15)+社融(10)+中美利差(10)：24 列里利率水平/全球利率从未用上
    - 新结构 M2(10)+社融(8)+中国利率 cn_2y(10)+全球流动性 us_2y(7)
    - 滚动分位 IC 验证: cn_2y=-0.46、us_2y=-0.40 全样本显著（利率低→小盘未来涨）
    - 利率维度必须用 5 年滚动分位（绝对阈值随时代漂移失效）
    - cn_2y/us_2y 缺失时回退旧逻辑（兼容旧数据/测试 fixture）
    - 不接入: cpi/gold/retail/house_price/jpy_cny(IC<0.10 无效)、usd_cny(0.18
      边缘且与利差相关)、m1_yoy(与 M2 相关)、fed_rate(停更2025-07，由 us_2y 代表)
    """
    scores = {}
    total = 0.0
    m = macro
    if latest_date is not None:
        latest_date = pd.Timestamp(latest_date)
        m = macro[macro["date"] <= latest_date]
        if m.empty:
            return 0.0, {"状态": "信号日无宏观数据"}

    # ── M2 增速 (10分)：绝对阈值，货币总量 ──
    if "m2_yoy" in m.columns:
        m2 = m["m2_yoy"].dropna()
        if len(m2) >= 1:
            m2_now = float(m2.iloc[-1])
            m2_chg = float(m2.diff(3).iloc[-1]) if len(m2) > 3 else 0
            if m2_now > 12:     s_m2 = 10
            elif m2_now > 10:   s_m2 = 8
            elif m2_now > 8:    s_m2 = 6
            elif m2_now > 6:    s_m2 = 3
            else:               s_m2 = 1
            if m2_chg > 0.5:
                s_m2 = min(s_m2 + 1, 10)
            scores["M2"] = f"{m2_now:.1f}% → {s_m2}分"
            total += s_m2
        else:
            scores["M2"] = "无数据"
    else:
        scores["M2"] = "无数据"

    # ── 社融 (8分)：信用扩张 ──
    if "social_finance" in m.columns:
        sf = m["social_finance"].dropna()
        if len(sf) >= 12:
            recent = float(sf.iloc[-6:].mean())
            prior = float(sf.iloc[-12:-6].mean())
            sf_ratio = recent / prior if prior > 0 else 1
            if sf_ratio > 1.2:      s_sf = 8
            elif sf_ratio > 1.05:   s_sf = 6
            elif sf_ratio > 0.95:   s_sf = 3
            else:                   s_sf = 0
            scores["信用"] = f"社融近6/前6月={sf_ratio:.2f} → {s_sf}分"
            total += s_sf
        else:
            scores["信用"] = "数据不足"
            total += 3  # 中性兜底
    else:
        scores["信用"] = "无数据"

    # ── 中国利率环境 (10分)：cn_2y 5年滚动分位（低利率=宽松）──
    if "cn_2y" in m.columns:
        s_cn, p_cn = _score_rate_percentile(m["cn_2y"], 10)
        if p_cn is not None:
            scores["中国利率"] = f"cn_2y分位={p_cn:.0%} → {s_cn}分"
            total += s_cn
        else:
            scores["中国利率"] = "数据不足"
    elif "spread_10y" in m.columns:
        # 回退：原中美利差逻辑（兼容缺 cn_2y 的旧数据）
        sp = float(m["spread_10y"].dropna().iloc[-1])
        if sp > -1.5:       s_sp = 10
        elif sp > -2.5:     s_sp = 6
        elif sp > -3.5:     s_sp = 3
        else:               s_sp = 0
        scores["利差"] = f"{sp:.1f}% → {s_sp}分"
        total += s_sp
    else:
        scores["中国利率"] = "无数据"

    # ── 全球流动性 (7分)：us_2y 5年滚动分位（低=全球宽松）──
    if "us_2y" in m.columns:
        s_us, p_us = _score_rate_percentile(m["us_2y"], 7)
        if p_us is not None:
            scores["全球流动性"] = f"us_2y分位={p_us:.0%} → {s_us}分"
            total += s_us
        else:
            scores["全球流动性"] = "数据不足"
    elif "fed_rate" in m.columns:
        # 回退：fed 降息周期
        fr = m["fed_rate"].dropna()
        if len(fr) >= 1:
            fr_now = float(fr.iloc[-1])
            fr_1y = float(fr.iloc[-max(2, min(len(fr), 13))]) if len(fr) > 2 else fr_now
            if fr_now < fr_1y:      s_fr = 7
            elif fr_now < 2.5:      s_fr = 5
            elif fr_now < 4:        s_fr = 3
            else:                   s_fr = 1
            scores["全球流动性"] = f"fed {fr_now:.1f}% → {s_fr}分"
            total += s_fr
        else:
            scores["全球流动性"] = "无数据"
    else:
        scores["全球流动性"] = "无数据"

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

# ═══════════════════════════════════════════
# 维度 6: 两融出清确认 (加分项, +13)
# ═══════════════════════════════════════════

def _score_margin_flush(daily: pd.DataFrame, macro: pd.DataFrame,
                        latest_date) -> tuple[float, dict]:
    """
    两融出清确认 (+13, 2026-08-28 新增): 杠杆快速出清后的 V 型急跌底确认。

    触发条件（全部满足）:
      1. 两融余额距6月高点回撤 >10%（126交易日窗口；两融行数≥60；
         最新两融日期距信号日 ≤7 天——新鲜度校验，防缓存冻结）
      2. 收盘价站回 MA20（日线 ≥80 行）
      3. 5日均量 > 20日均量（量能回升；volume 列缺失时跳过）
    冷却: 最近 60 自然日内已出现过三条件全满足 → 不加分（防连续触发叠加）

    注: 拼接补丁段 volume 为 ETF 量纲与指数段不同，拼接后 20 日内量能判断
    降级（比率类判断不敏感；实际加分触发点都在拼接窗口外）。
    """
    if daily is None or daily.empty or len(daily) < 80:
        return 0.0, {"状态": "数据不足(日线<80行)"}
    if macro is None or macro.empty or 'margin_balance' not in macro.columns:
        return 0.0, {"状态": "数据不足(无两融数据)"}

    latest_date = pd.Timestamp(latest_date)
    d = daily[daily['date'] <= latest_date].sort_values('date').reset_index(drop=True)
    if len(d) < 80:
        return 0.0, {"状态": "数据不足(日线<80行)"}

    mb = macro[['date', 'margin_balance']].dropna().sort_values('date')
    mb = mb[mb['date'] <= latest_date]
    # 统一日期 dtype（parquet 可能读成 datetime64[us]，merge_asof 要求一致）
    d['date'] = pd.to_datetime(d['date']).astype('datetime64[ns]')
    mb['date'] = pd.to_datetime(mb['date']).astype('datetime64[ns]')
    if len(mb) < 60:
        return 0.0, {"状态": "数据不足(两融<60行)"}
    lag_days = (latest_date - mb['date'].iloc[-1]).days
    if lag_days > 7:
        return 0.0, {"状态": f"两融数据滞后{lag_days}天>7天，跳过"}

    # 按日线交易日对齐两融（asof 后向取最近值）
    m_align = pd.merge_asof(d[['date']], mb, on='date', direction='backward')
    mb_6m_high = m_align['margin_balance'].rolling(126, min_periods=60).max()
    dd_from_high = m_align['margin_balance'] / mb_6m_high - 1
    fresh = (d['date'] - m_align['date']).dt.days <= 7

    close = d['close']
    above_ma20 = (close > close.rolling(20).mean()).fillna(False)
    triggered = ((dd_from_high < -0.10) & above_ma20 & fresh).fillna(False)
    if 'volume' in d.columns:
        vol_recovering = (d['volume'].rolling(5).mean()
                          > d['volume'].rolling(20).mean()).fillna(False)
        triggered = triggered & vol_recovering

    today_trigger = bool(triggered.iloc[-1])
    # 冷却: 60自然日内（不含今日）已有触发 → 不加分
    cutoff = latest_date - pd.Timedelta(days=60)
    prior = triggered[(d['date'] >= cutoff) & (d['date'] < latest_date)]
    cooldown = bool(prior.any())

    now = float(close.iloc[-1])
    ma20 = float(close.tail(20).mean())
    detail_base = {
        "两融距6月高": f"{float(dd_from_high.iloc[-1]):.1%}",
        "站回MA20": f"{'✅' if bool(above_ma20.iloc[-1]) else '❌'} {now:.0f} vs MA20 {ma20:.0f}",
        "最新两融日": f"{m_align['date'].iloc[-1].date()}",
    }

    if today_trigger and not cooldown:
        return 13.0, dict(detail_base, **{
            "触发": "✅ 两融出清确认(杠杆快速出清+V型反弹确认)"})
    if cooldown:
        return 0.0, dict(detail_base, **{
            "触发": "冷却期内(60自然日内已触发过)"})

    missing = []
    if not bool((dd_from_high < -0.10).fillna(False).iloc[-1]):
        missing.append(f"两融未出清({float(dd_from_high.iloc[-1]):.1%}>-10%)")
    if not bool(above_ma20.iloc[-1]):
        missing.append("未站回MA20")
    if 'volume' in d.columns and not bool(vol_recovering.iloc[-1]):
        missing.append("量能未回升")
    return 0.0, dict(detail_base, **{
        "触发": "未触发", "缺条件": "; ".join(missing)})


# ═══════════════════════════════════════════
# v7: 三通道买入信号（动态阈值）
# ═══════════════════════════════════════════

def _rolling_percentile(series: pd.Series, window: int) -> pd.Series:
    """滚动分位（0~1，当前值在窗口内的严格排名），窗口不足返回 NaN。"""
    return series.rolling(window=window).apply(
        lambda x: (x < x[-1]).sum() / len(x), raw=True
    )


def _monthly_score_history(daily: pd.DataFrame, csi300_daily: pd.DataFrame,
                           macro: pd.DataFrame, latest_date, n_months: int = 60) -> list:
    """生成截至 latest_date 的月度分数历史（每月最后交易日打分）。

    用于分数分位视图（通道A）：score 是绝对分，其 5 年滚动分位随市场
    水位变化——熊市里 40 分可能是历史低分位，牛市里 40 分可能是高分位。
    调用方（回测）可传 score_history 复用已算分数，避免重复打分。
    """
    daily = daily[daily['date'] <= latest_date].sort_values('date')
    hist = []
    month_ends = daily.groupby(daily['date'].dt.to_period('M'))['date'].max().tail(n_months)
    for d in month_ends:
        d = pd.Timestamp(d)
        res = compute_csi2000_score(
            daily=daily[daily['date'] <= d],
            valuation=None,
            csi300_daily=csi300_daily[csi300_daily['date'] <= d],
            macro=macro[macro['date'] <= d] if macro is not None else None,
        )
        hist.append((d, float(res['total_score'])))
    return hist


def csi2000_buy_channel(daily: pd.DataFrame,
                        csi300_daily: pd.DataFrame,
                        macro: pd.DataFrame = None,
                        latest_date=None,
                        score_history: list = None,
                        current_score: float = None) -> dict:
    """v7 三通道买入信号（动态阈值，2026-08-30）。

    背景：固定阈值 50 分在低波年份（如 2026）永远凑不到冰点，2026 三次
    下跌全踏空。业界共识（z-score/滚动分位/右侧确认/动态仓位）：
    "阈值"不能是常数，必须是市场水位（波动率/分位）的函数。

    通道（优先级 A > B；全部要求"站回MA10"企稳确认，防阴跌飞刀）:
      A 冰点:    (absolute分≥50) 或 (分数5年分位≤15% 且 r20分位≤15% 超跌确认)
                 + 回撤250日>10%                     → 基础仓位 40%
                 （分位视图必须叠加超跌：2017-02/12 微盘熊分数低但
                   r20分位 20%+，是阴跌飞刀；2026-08 大底 r20分位 2%）
      B 超跌企稳: r20 5年分位≤10%（超跌深度自适应，已验证 5/59 vs 固定 1/59）
                 + 回撤250日>15%                     → 基础仓位 25%

    动态仓位: 仓位 = 基础 × 按 r20 分位的乘数（只加不减），封顶 50%。
      r20分位 ≤5% → ×1.4 | ≤10% → ×1.2 | 其余 → ×1.0

    所有阈值都是"过去5年自身历史的分位"——每年随市场水位自动重算，
    不需要为 2026/2027/2028 逐年调参（结构固定、数值自适应）。

    v7.1/v7.2 修正（2026-08-30 回测诊断，收益下降归因）:
      ① 初版无企稳过滤：通道A 分数分位视图在阴跌中继触发（2018-06
         -21.5% 飞刀）→ 全通道统一站回MA10 右侧确认
      ② 初版动态乘数 >20% 分位 ×0.4：把非超跌月仓位压到 16%（2019-03）
         低于 v6.1 基线 25% → 收益被摊薄 → 乘数只加不减
      ③ 回测信号序列从 full 拼接改为 index 生产序列（与实盘同源）
      ④ 初版通道C(40-49分)在 2016/2017 震荡阴跌市反复触发（-17.9%/
         -22.8% 飞刀）且抢占90天冷却挤掉优质A买入 → 删除
      ⑤ 分位视图须叠加 r20≤15% 超跌确认（挡 2017-02/12 微盘熊飞刀，
         2026-08 大底 r20 分位仅 2% 不受影响）

    score_history: [(month_end_date, score), ...] 由调用方维护（回测每轮
    append）；为 None 时内部生成 60 个月分数历史。current_score 复用调用
    方已算好的分数（回测场景省一次全量打分）。
    """
    out = {'channel': None, 'position_pct': 0.0, 'score': None,
           'score_pct': None, 'r20_pct': None, 'dd250': None, 'detail': {}}
    if daily is None or len(daily) < 250:
        out['detail'] = {'状态': '数据不足(日线<250行)'}
        return out

    if latest_date is None:
        latest_date = daily['date'].max()
    latest_date = pd.Timestamp(latest_date)
    d = daily[daily['date'] <= latest_date].sort_values('date').reset_index(drop=True)
    if len(d) < 250:
        out['detail'] = {'状态': '数据不足(日线<250行)'}
        return out

    close = d['close']
    cur = float(close.iloc[-1])

    # ── 当前分数（复用调用方已算值，避免重复全量打分）──
    if current_score is not None:
        score = float(current_score)
    else:
        res = compute_csi2000_score(
            daily=d,
            valuation=None,
            csi300_daily=csi300_daily[csi300_daily['date'] <= latest_date],
            macro=macro[macro['date'] <= latest_date] if macro is not None else None,
        )
        score = float(res['total_score'])

    # ── 分数 5 年分位（月度；取最近 60 个月，5年滚动窗口）──
    if score_history is None:
        score_history = _monthly_score_history(daily, csi300_daily, macro, latest_date)
    hist_scores = [s for _, s in score_history][-60:]  # 截断防超窗（回测传全历史）
    score_pct = None
    if len(hist_scores) >= 24:
        score_pct = float((pd.Series(hist_scores) < score).mean())

    # ── r20 5年分位（只需当日值：O(w) 直接排名，避免全列滚动 apply）──
    r20 = close / close.shift(20) - 1
    w = min(1260, len(d) - 21)
    r20_pct = None
    if w >= 60:
        r20_pct = float((r20.tail(w) < r20.iloc[-1]).mean())

    # ── 回撤 250 日 ──
    dd250 = float(close.iloc[-1] / close.rolling(250, min_periods=120).max().iloc[-1] - 1)

    # ── 企稳确认（通道B 右侧，防飞刀）──
    ma10 = float(close.tail(10).mean())
    above_ma10 = cur > ma10

    # ── 通道判定（A > B；全部要求企稳确认站回MA10，防阴跌飞刀）──
    # v7.1 修正（2026-08-30 回测诊断）：
    #   ① 初版无企稳过滤 → 2018-06 分位视图在阴跌中继触发（-21.5%）
    #   ② 初版通道C(40-49分)在 2016/2017 震荡阴跌市反复触发并抢占
    #      90天冷却，挤掉 2016-06/09 优质A买入 → 删除通道C
    #   ③ 分位视图须叠加 r20≤15% 超跌确认 → 挡 2017-02/12 微盘熊飞刀
    ch, base = None, 0.0
    if above_ma10:
        pct_view = score_pct is not None and score_pct <= 0.15 \
            and r20_pct is not None and r20_pct <= 0.15
        if score >= 50 or pct_view:
            if dd250 < -0.10:
                ch, base = 'A', 0.40
        if ch is None and r20_pct is not None and r20_pct <= 0.10 and dd250 < -0.15:
            ch, base = 'B', 0.25

    # ── 动态仓位（只加不减：超跌越深买越多，不超跌维持基础仓位）──
    # v7.1 修正：初版 >20% 分位 ×0.4 把 2019-03 等非超跌月摊薄到 16%，
    # 低于 v6.1 的 25% 基线 → 收益被摊薄。改为只奖励超跌，不惩罚。
    mult = 1.0
    if r20_pct is not None:
        if r20_pct <= 0.05:      mult = 1.4
        elif r20_pct <= 0.10:    mult = 1.2
    pct = min(0.50, base * mult) if ch else 0.0

    out.update({
        'channel': ch,
        'position_pct': round(pct, 4),
        'score': score,
        'score_pct': score_pct,
        'r20_pct': r20_pct,
        'dd250': dd250,
        'detail': {
            '通道': ch or '无',
            '分数': f"{score:.1f}",
            '分数分位': f"{score_pct:.0%}" if score_pct is not None else 'N/A',
            'r20分位': f"{r20_pct:.0%}" if r20_pct is not None else 'N/A',
            '回撤250日': f"{dd250:.1%}",
            '站回MA10': '✅' if above_ma10 else '❌',
            '基础仓位': f"{base:.0%}" if ch else '—',
            '动态乘数': f"×{mult:.1f}" if ch else '—',
            '买入仓位': f"{pct:.0%}" if ch else '—',
        },
    })
    return out


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
