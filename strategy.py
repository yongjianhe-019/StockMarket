"""
择时信号 — 冰点买入 + 泡沫卖出

冰点 = 足够便宜 + 有性价比（A股指标 + 宏观指标共振）
泡沫 = PE贵 + 宏观恶化（必须先贵，才看宏观）

信号频率：月频
输出：BUY / SELL / HOLD
"""

from __future__ import annotations
from datetime import datetime
import pandas as pd
import numpy as np

from models.csi300 import compute_csi300_score, _window_pct
from models.csi2000 import compute_csi2000_score, csi2000_buy_channel
from macro.sell_signal import is_bubble


# ═══════════════════════════════════
# 冰点检测
# ═══════════════════════════════════

def detect_ice_point(data: dict, macro_df: pd.DataFrame, date,
                     score_history_2000: list = None) -> dict:
    """
    检测冰点买入信号。

    CSI300: 估值驱动模型，分数≥50 → 冰点
    CSI2000: 流动性+动量模型 + v7 三通道（动态阈值）
      - 通道A 冰点: 绝对分≥50 或 5年分数分位≤15%，回撤>10%
      - 通道B 超跌企稳: r20 5年分位≤10% + 站回MA10，回撤>15%
      - 通道C 次冰点: 分数40~49 + 回撤>10%

    score_history_2000: [(date, score), ...] 月度分数历史（回测每轮
    append 复用，避免通道A 分数分位重复打分；单点调用可省略）。

    Returns {csi300: bool, csi2000: bool, details: ...,
             channel_2000: 'A'|'B'|'C'|None, position_pct_2000: float}
    """
    idx300 = data['csi300_daily']
    idx2000 = data['csi2000_daily']
    val300 = data.get('csi300_valuation')
    bond = data['bond_yield_10y']

    # CSI300 打分（传入宏观数据用于极端信号加分）
    s300 = compute_csi300_score(
        daily=idx300[idx300['date'] <= date],
        valuation=val300[val300['date'] <= date] if val300 is not None and not val300.empty else None,
        bond_yield_df=bond[bond['date'] <= date] if bond is not None and not bond.empty else bond,
        macro=macro_df[macro_df['date'] <= date] if macro_df is not None else None,
    )

    # CSI2000 打分
    s2000 = compute_csi2000_score(
        daily=idx2000[idx2000['date'] <= date],
        valuation=data.get('csi2000_valuation'),
        csi300_daily=idx300[idx300['date'] <= date],
        macro=macro_df[macro_df['date'] <= date] if macro_df is not None else None,
    )

    ice_300 = s300['total_score'] >= 50

    # CSI2000 v7 三通道（动态阈值）：通道非空 = 冰点
    ch2000 = csi2000_buy_channel(
        daily=idx2000[idx2000['date'] <= date],
        csi300_daily=idx300[idx300['date'] <= date],
        macro=macro_df[macro_df['date'] <= date] if macro_df is not None else None,
        latest_date=date,
        current_score=s2000['total_score'],
        score_history=score_history_2000,
    )
    ice_2000 = ch2000['channel'] is not None

    # PE 分位
    pe_pct_300 = _get_pe_pct(val300, date) if val300 is not None else None

    return {
        'csi300': ice_300,
        'csi2000': ice_2000,
        'score_300': s300['total_score'],
        'score_2000': s2000['total_score'],
        'pe_pct_300': pe_pct_300,
        'details_300': s300.get('details', {}),
        'details_2000': s2000.get('details', {}),
        # v7 三通道扩展字段
        'channel_2000': ch2000['channel'],
        'position_pct_2000': ch2000['position_pct'],
        'channel_detail_2000': ch2000.get('detail', {}),
    }


def _get_pe_pct(val_df, date):
    """当前 PE 在**近5年**历史中的分位（v8.2 修复）。

    原实现按"全历史"取分位（legulegu 月频≈258行/21.5年），与模型内部
    `compute_csi300_score` 及卖出端 `_pe_percentile` 的 5 年口径不一致，
    把"偏贵"显示成"合理"（2026-09 实盘：全历史 50.8% vs 正确5年 70.5%，
    仪表盘显示🟡合理、卖出理由却写"PE分位70%"，自相矛盾）。现统一走
    `_window_pct`，口径与 v8/v8.1 对齐；观测不足返回 None（降级，不回退全历史）。
    """
    return _window_pct(val_df, "pe", date, years=5)


# ═══════════════════════════════════
# 泡沫检测
# ═══════════════════════════════════

def detect_bubble(data: dict, macro_df: pd.DataFrame, date) -> dict:
    """
    检测泡沫卖出信号（v5 分标的）。

    前提：PE分位 > 60%（全市场背景，CSI300 估值）
    三分类确认：A.价格加速  B.量价背离  C.宏观恶化  D.流动性狂热
    ≥2 类触发 → 泡沫确认
    趋势兜底/回补：按持仓标的各自日线判断
      - 沪深300仓位: csi300_daily
      - 中证2000仓位: etf_159531 日线

    Returns {is_bubble, level, sell_pct, reasons, ..., leg_300, leg_2000}
    """
    return is_bubble(macro_df, data.get('csi300_valuation'),
                     idx=macro_df['date'].searchsorted(date),
                     daily_300=data.get('csi300_daily'),
                     daily_2000=data.get('etf_159531'))


# ═══════════════════════════════════
# 综合决策
# ═══════════════════════════════════

def generate_signal(data: dict, macro_df: pd.DataFrame,
                    date: datetime = None) -> dict:
    """
    月度信号生成。返回当前应该做什么。

    Returns
    -------
    {
        'date': datetime,
        'ice_300': bool,       # CSI300 冰点
        'ice_2000': bool,      # CSI2000 冰点
        'bubble': bool,        # 泡沫警告
        'bubble_reasons': [],  # 泡沫原因
        'action_300': 'BUY' | 'HOLD' | 'SELL',
        'action_2000': 'BUY' | 'HOLD' | 'SELL',
        'position_advice': str,
    }
    """
    if date is None:
        date = macro_df['date'].iloc[-1]

    ice = detect_ice_point(data, macro_df, date)
    bubble = detect_bubble(data, macro_df, date)

    # v5 分标的: 有 leg 时用各自腿，否则回退到旧的市场级信号
    leg_300 = bubble.get('leg_300', bubble)
    leg_2000 = bubble.get('leg_2000', bubble)

    def _action(leg, ice_flag):
        if leg.get('is_bubble'):
            return 'SELL'
        if leg.get('recovery'):
            return 'RESTORE'  # 回补: 趋势破坏减仓后趋势修复
        return 'BUY' if ice_flag else 'HOLD'

    action_300 = _action(leg_300, ice['csi300'])
    action_2000 = _action(leg_2000, ice['csi2000'])

    buying = [e for e, ice in [('沪深300', ice['csi300']), ('中证2000', ice['csi2000'])] if ice]
    advice_parts = []
    for name, leg, act in [('沪深300', leg_300, action_300), ('中证2000', leg_2000, action_2000)]:
        if act == 'SELL':
            sell_pct = leg.get('sell_pct', 0.5)
            if leg.get('signal_type') == 'trend_breakdown':
                # 趋势破坏是独立风控信号，非泡沫极端信号
                advice_parts.append(f"📉 {name}: {leg['level']} → 风控减仓{sell_pct:.0%}（非泡沫信号）")
            else:
                advice_parts.append(f"⚠️ {name}: {leg['level']} → 卖出{sell_pct:.0%}")
        elif act == 'RESTORE':
            # v6 回补阶梯化: 档位越高累计回补比例越高（档1=1/3 → 档2=2/3 → 档3=3/3）
            pct = leg.get('recovery_pct', 1 / 3)
            advice_parts.append(f"↩️ {name}: {leg['level']} → 回补（累计回补{pct:.0%}；"
                                f"冰点出现时按分数档位优先）")
    if advice_parts:
        advice = '; '.join(advice_parts)
    elif buying:
        advice = f"🧊 冰点: {'+'.join(buying)} → 分批买入"
    else:
        advice = "— 持有等待"

    return {
        'date': date,
        'ice_300': ice['csi300'],
        'ice_2000': ice['csi2000'],
        'score_300': ice['score_300'],
        'score_2000': ice['score_2000'],
        'details_300': ice.get('details_300', {}),
        'details_2000': ice.get('details_2000', {}),
        'pe_pct_300': ice['pe_pct_300'],
        'bubble': bubble['is_bubble'],
        'bubble_signal_type': bubble.get('signal_type'),
        'bubble_level': bubble.get('level', ''),
        'bubble_sell_pct': bubble.get('sell_pct', 0.5),
        'bubble_reasons': bubble.get('reasons', []),
        'bubble_signals': bubble.get('signals', {}),
        'leg_300': leg_300,
        'leg_2000': leg_2000,
        'action_300': action_300,
        'action_2000': action_2000,
        'position_advice': advice,
    }
