"""
CSI 300 季节打分模型

估值驱动：PE/PB 分位 + 股债性价比为主，量价衰竭为辅。

每周运行一次，输出 0-100 的综合分数。
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# 打分阈值
SCORE_WAIT = 50       # < 50: 等待
SCORE_WATCH = 65      # 50-65: 关注
SCORE_BUY = 80        # 65-80: 建仓; > 80: 补仓

# 回测窗口
LOOKBACK_YEARS = 5
TRADING_DAYS_PER_YEAR = 252


def _rolling_percentile(series: pd.Series, window: int) -> pd.Series:
    """滚动分位数（0~1），窗口不足时返回 NaN。"""
    return series.rolling(window=window).apply(
        lambda x: (x < x.iloc[-1]).sum() / len(x), raw=True
    )


# ---------------------------------------------------------------------------
# 维度 1: 估值温度 (30 分)
# ---------------------------------------------------------------------------


def score_valuation(pe_pct: float, pb_pct: float) -> tuple[float, dict]:
    """
    PE/PB 滚动5年分位均值。

    pe_pct, pb_pct: 0~1，越小表示越低估
    """
    avg_pct = (pe_pct + pb_pct) / 2

    if avg_pct < 0.10:
        s = 30.0
        level = "极度低估"
    elif avg_pct < 0.20:
        s = 24.0
        level = "显著低估"
    elif avg_pct < 0.30:
        s = 18.0
        level = "低估"
    elif avg_pct < 0.50:
        s = 10.0
        level = "中性偏低"
    else:
        s = 0.0
        level = "不便宜"

    detail = {"PE分位": f"{pe_pct:.1%}", "PB分位": f"{pb_pct:.1%}",
              "均值分位": f"{avg_pct:.1%}", "等级": level, "得分": s}
    return s, detail


# ---------------------------------------------------------------------------
# 维度 2: 股债性价比 (25 分)
# ---------------------------------------------------------------------------


def score_erp(pe: float, bond_yield: float, erp_pct: float) -> tuple[float, dict]:
    """
    股权风险溢价：ERP = 1/PE - 10年期国债收益率
    erp_pct: ERP 在近5年的分位数，越高越好
    """
    erp = (1 / pe * 100) - bond_yield if pe > 0 else 0

    if erp_pct > 0.90:
        s = 25.0
        level = "股票极具性价比"
    elif erp_pct > 0.80:
        s = 20.0
        level = "股票很有性价比"
    elif erp_pct > 0.70:
        s = 15.0
        level = "股票较有性价比"
    elif erp_pct > 0.50:
        s = 8.0
        level = "中性"
    else:
        s = 0.0
        level = "股票偏贵"

    detail = {"ERP": f"{erp:.2f}%", "ERP分位": f"{erp_pct:.1%}",
              "等级": level, "得分": s}
    return s, detail


# ---------------------------------------------------------------------------
# 维度 3: 回撤深度 (20 分)
# ---------------------------------------------------------------------------


def score_drawdown(drawdown: float) -> tuple[float, dict]:
    """回撤越深，分数越高（越安全）。"""
    if drawdown > 0.30:
        s, level = 20.0, "深度回撤"
    elif drawdown > 0.20:
        s, level = 15.0, "显著回撤"
    elif drawdown > 0.10:
        s, level = 10.0, "中等回撤"
    elif drawdown > 0.05:
        s, level = 5.0, "轻度回撤"
    else:
        s, level = 0.0, "接近高点"

    return s, {"回撤幅度": f"{drawdown:.1%}", "等级": level, "得分": s}


# ---------------------------------------------------------------------------
# 维度 4: 量价衰竭 (15 分)
# ---------------------------------------------------------------------------


def score_volume_exhaustion(vol_ratio: float, turnover_pct: float) -> tuple[float, dict]:
    """
    量价衰竭信号。

    vol_ratio: 近10日均成交额 / 近120日均成交额
    turnover_pct: 换手率在近2年的分位数
    """
    if vol_ratio < 0.60 and turnover_pct < 0.20:
        s, level = 15.0, "地量见地价"
    elif vol_ratio < 0.75 and turnover_pct < 0.35:
        s, level = 10.0, "量缩显著"
    elif vol_ratio < 0.90:
        s, level = 5.0, "量能偏弱"
    else:
        s, level = 0.0, "量能正常"

    return s, {"量比": f"{vol_ratio:.1%}", "换手率分位": f"{turnover_pct:.1%}",
               "等级": level, "得分": s}


# ---------------------------------------------------------------------------
# 维度 5: 波动收敛 (10 分)
# ---------------------------------------------------------------------------


def score_volatility_regression(vol_20d: float, vol_60d_high: float) -> tuple[float, dict]:
    """波动率从高点回落。"""
    ratio = vol_20d / vol_60d_high if vol_60d_high > 0 else 1.0

    if ratio < 0.50:
        s, level = 10.0, "恐慌消退"
    elif ratio < 0.70:
        s, level = 6.0, "波动回落"
    elif ratio < 0.85:
        s, level = 3.0, "波动略降"
    else:
        s, level = 0.0, "高波动中"

    return s, {"波动率": f"{vol_20d:.2%}", "回落比": f"{ratio:.1%}",
               "等级": level, "得分": s}


# ---------------------------------------------------------------------------
# 综合打分
# ---------------------------------------------------------------------------


def compute_csi300_score(daily: pd.DataFrame,
                         valuation: Optional[pd.DataFrame],
                         bond_yield_df: pd.DataFrame,
                         macro: pd.DataFrame = None) -> dict:
    """
    计算 CSI 300 季节分数。

    Parameters
    ----------
    daily : DataFrame
        指数日线，列: date, close, high, low, volume, amount
    valuation : DataFrame or None
        PE/PB 估值，列: date, pe, pb。为 None 时估值维度计 0 分。
    bond_yield_df : DataFrame
        10年国债收益率，列: date, yield_10y
    macro : DataFrame or None
        宏观数据，列: date, pmi, m2_yoy 等。用于极端信号加分。

    Returns
    -------
    dict
        {
            "total_score": float,          # 总分 0-100
            "season": str,                 # 季节标签
            "action": str,                 # 建议操作
            "details": dict,               # 各维度详情
            "signal_date": datetime,
        }
    """
    latest_date = daily["date"].max()
    price_latest = float(daily["close"].iloc[-1])

    # ---- 维度 1: 估值温度 ----
    score_v, detail_v = 0.0, {"状态": "无估值数据"}
    pe_latest = None
    if valuation is not None and not valuation.empty:
        val = valuation.copy()
        pe_latest = float(val["pe"].dropna().iloc[-1]) if "pe" in val.columns else None
        pb_latest = float(val["pb"].dropna().iloc[-1]) if "pb" in val.columns else None

        if pe_latest is not None and pb_latest is not None and not (np.isnan(pe_latest) or np.isnan(pb_latest)):
            window = LOOKBACK_YEARS * TRADING_DAYS_PER_YEAR
            pe_pct = _compute_series_pct(val, "pe", window)
            pb_pct = _compute_series_pct(val, "pb", window)
            score_v, detail_v = score_valuation(pe_pct, pb_pct)

    # ---- 维度 2: 股债性价比 ----
    score_e, detail_e = 0.0, {"状态": "无法计算"}
    if pe_latest is not None and pe_latest > 0 and bond_yield_df is not None and not bond_yield_df.empty:
        bond_latest = float(bond_yield_df["yield_10y"].iloc[-1])
        if not np.isnan(bond_latest):
            erp = (1 / pe_latest * 100) - bond_latest

            # 计算 ERP 历史序列和分位
            merged = daily[["date", "close"]].merge(
                valuation[["date", "pe"]], on="date", how="inner"
            )
            merged = merged.merge(bond_yield_df[["date", "yield_10y"]], on="date", how="left")
            merged["yield_10y"] = merged["yield_10y"].ffill()
            merged["erp"] = (1 / merged["pe"] * 100) - merged["yield_10y"]
            merged = merged.dropna(subset=["erp"])

            if len(merged) > 60:
                window = LOOKBACK_YEARS * TRADING_DAYS_PER_YEAR
                erp_pct = _compute_series_pct(merged, "erp", window)
                score_e, detail_e = score_erp(pe_latest, bond_latest, erp_pct)

    # ---- 维度 3: 回撤深度 ----
    high_1y = float(daily["high"].tail(TRADING_DAYS_PER_YEAR).max())
    drawdown = 1 - price_latest / high_1y if high_1y > 0 else 0
    score_d, detail_d = score_drawdown(drawdown)

    # ---- 维度 4: 量价衰竭 ----
    score_q, detail_q = 0.0, {"状态": "数据不足"}
    if "volume" in daily.columns and len(daily) >= 120:
        vol_ma10 = float(daily["volume"].tail(10).mean())
        vol_ma120 = float(daily["volume"].tail(120).mean())
        vol_ratio = vol_ma10 / vol_ma120 if vol_ma120 > 0 else 1.0

        # 换手率用成交额/自由流通市值近似，这里用成交额相对于均值的分位
        turnover_pct = _compute_series_pct(daily, "volume", min(len(daily), 2 * TRADING_DAYS_PER_YEAR))
        score_q, detail_q = score_volume_exhaustion(vol_ratio, turnover_pct)

    # ---- 维度 5: 波动收敛 ----
    score_w, detail_w = 0.0, {"状态": "数据不足"}
    if len(daily) >= 60:
        returns = daily["close"].pct_change().dropna()
        vol_20d = float(returns.tail(20).std() * np.sqrt(TRADING_DAYS_PER_YEAR))
        vol_60d_high = float(returns.tail(60).rolling(20).std().max() * np.sqrt(TRADING_DAYS_PER_YEAR))
        score_w, detail_w = score_volatility_regression(vol_20d, vol_60d_high)

    # ---- 维度 6: ERP 极端信号 (加分项, +15) ----
    # 借鉴广发证券：ERP 达到均值+2σ 是历史大底信号
    score_erp_extreme, detail_erp_extreme = 0.0, {"状态": "未触发"}
    if pe_latest is not None and pe_latest > 0 and bond_yield_df is not None and not bond_yield_df.empty:
        bond_latest = float(bond_yield_df["yield_10y"].iloc[-1])
        if not np.isnan(bond_latest):
            erp_now = (1 / pe_latest * 100) - bond_latest
            merged = daily[["date", "close"]].merge(
                valuation[["date", "pe"]], on="date", how="inner"
            )
            merged = merged.merge(bond_yield_df[["date", "yield_10y"]], on="date", how="left")
            merged["yield_10y"] = merged["yield_10y"].ffill()
            merged["erp"] = (1 / merged["pe"] * 100) - merged["yield_10y"]
            merged = merged.dropna(subset=["erp"])
            if len(merged) > 252:
                erp_5y = merged["erp"].tail(LOOKBACK_YEARS * TRADING_DAYS_PER_YEAR)
                erp_mean = erp_5y.mean()
                erp_std = erp_5y.std()
                erp_2sigma = erp_mean + 2 * erp_std
                if erp_now > erp_2sigma:
                    score_erp_extreme = 15.0
                    detail_erp_extreme = {
                        "ERP": f"{erp_now:.2f}%",
                        "均值+2σ": f"{erp_2sigma:.2f}%",
                        "等级": "ERP极端—历史大底信号",
                        "得分": 15.0,
                    }

    # ---- 维度 7: 宏观底部确认 (加分项, +5) ----
    # 借鉴华泰证券：基本面数据逆向使用，不再恶化=底部确认
    score_macro_bottom, detail_macro_bottom = 0.0, {"状态": "无宏观数据"}
    if macro is not None and not macro.empty:
        m = macro[macro["date"] <= latest_date]
        if not m.empty:
            confirmations = []
            # PMI 不再恶化（连续2月不创新低）
            if "pmi" in m.columns:
                pmi_data = m["pmi"].dropna()
                if len(pmi_data) >= 3:
                    pmi_now = pmi_data.iloc[-1]
                    pmi_1m = pmi_data.iloc[-2]
                    pmi_2m = pmi_data.iloc[-3]
                    if pmi_now >= pmi_1m and pmi_1m >= pmi_2m:
                        confirmations.append(f"PMI企稳({pmi_now:.1f})")
            # M2 开始回升
            if "m2_yoy" in m.columns:
                m2_data = m["m2_yoy"].dropna()
                if len(m2_data) >= 3:
                    m2_now = m2_data.iloc[-1]
                    m2_3m = m2_data.iloc[-min(4, len(m2_data))]
                    if m2_now > m2_3m:
                        confirmations.append(f"M2回升({m2_now:.1f}%)")
            # 社融不再收缩
            if "social_finance" in m.columns:
                sf = m["social_finance"].dropna()
                if len(sf) >= 6:
                    sf_recent = sf.iloc[-3:].mean()
                    sf_prior = sf.iloc[-6:-3].mean()
                    if sf_recent >= sf_prior * 0.95:
                        confirmations.append("社融稳定")

            if len(confirmations) >= 2:
                score_macro_bottom = 5.0
                detail_macro_bottom = {
                    "确认信号": "; ".join(confirmations),
                    "等级": "宏观底部确认",
                    "得分": 5.0,
                }
            elif len(confirmations) == 1:
                score_macro_bottom = 2.0
                detail_macro_bottom = {
                    "确认信号": confirmations[0],
                    "等级": "宏观部分企稳",
                    "得分": 2.0,
                }
            else:
                detail_macro_bottom = {"状态": "宏观仍在恶化或数据不足"}

    # ---- 汇总 ----
    total = score_v + score_e + score_d + score_q + score_w + score_erp_extreme + score_macro_bottom

    if total >= SCORE_BUY:
        season, action = "深冬", "补仓（极度低估）"
    elif total >= SCORE_WATCH:
        season, action = "冬天", "首次建仓"
    elif total >= SCORE_WAIT:
        season, action = "秋末", "重点关注，准备资金"
    else:
        season, action = "夏/秋", "等待"

    return {
        "total_score": round(total, 1),
        "season": season,
        "action": action,
        "details": {
            "估值温度(30)": detail_v,
            "股债性价比(25)": detail_e,
            "回撤深度(20)": detail_d,
            "量价衰竭(15)": detail_q,
            "波动收敛(10)": detail_w,
            "ERP极端信号(+15)": detail_erp_extreme,
            "宏观底部确认(+5)": detail_macro_bottom,
        },
        "price": price_latest,
        "signal_date": latest_date,
    }


def _compute_series_pct(df: pd.DataFrame, col: str, window: int) -> float:
    """计算某列在给定窗口内的最新分位数 (0~1)。"""
    series = df[col].dropna()
    if len(series) < window:
        window = len(series)
    recent = series.tail(window)
    latest = recent.iloc[-1]
    return float((recent < latest).sum() / len(recent))


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------


def analyze_csi300(data: dict) -> dict:
    """从 fetch_all_data() 结果中分析 CSI 300。"""
    return compute_csi300_score(
        daily=data["csi300_daily"],
        valuation=data.get("csi300_valuation"),
        bond_yield_df=data["bond_yield_10y"],
    )
