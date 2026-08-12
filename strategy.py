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

from models.csi300 import compute_csi300_score
from models.csi2000 import compute_csi2000_score
from macro.sell_signal import is_bubble


# ═══════════════════════════════════
# 冰点检测
# ═══════════════════════════════════

def detect_ice_point(data: dict, macro_df: pd.DataFrame, date) -> dict:
    """
    检测冰点买入信号。

    CSI300: 估值驱动模型，分数≥50 → 冰点
    CSI2000: 流动性+动量模型，分数≥50 → 冰点

    Returns {csi300: bool, csi2000: bool, details: ...}
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
    ice_2000 = s2000['total_score'] >= 50

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
    }


def _get_pe_pct(val_df, date):
    """当前 PE 在历史中的分位数。"""
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


# ═══════════════════════════════════
# 泡沫检测
# ═══════════════════════════════════

def detect_bubble(data: dict, macro_df: pd.DataFrame, date) -> dict:
    """
    检测泡沫卖出信号。

    前提：PE分位 > 60%（必须先贵）
    三分类确认：A.价格加速  B.量价背离  C.宏观恶化
    ≥2 类触发 → 泡沫确认

    Returns {is_bubble: bool, level: str, sell_pct: float, reasons: [...]}
    """
    return is_bubble(macro_df, data.get('csi300_valuation'),
                     idx=macro_df['date'].searchsorted(date),
                     daily_300=data.get('csi300_daily'))


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

    action_300 = 'SELL' if bubble['is_bubble'] else ('BUY' if ice['csi300'] else 'HOLD')
    action_2000 = 'SELL' if bubble['is_bubble'] else ('BUY' if ice['csi2000'] else 'HOLD')

    buying = [e for e, ice in [('沪深300', ice['csi300']), ('中证2000', ice['csi2000'])] if ice]
    if bubble['is_bubble']:
        sell_pct = bubble.get('sell_pct', 0.5)
        if bubble.get('signal_type') == 'trend_breakdown':
            # 趋势破坏是独立风控信号，非泡沫极端信号
            advice = f"📉 {bubble['level']}: {'; '.join(bubble['reasons'][:2])} → 风控减仓{sell_pct:.0%}（非泡沫信号）"
        else:
            advice = f"⚠️ {bubble['level']}: {'; '.join(bubble['reasons'][:3])} → 卖出{sell_pct:.0%}"
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
        'pe_pct_300': ice['pe_pct_300'],
        'bubble': bubble['is_bubble'],
        'bubble_signal_type': bubble.get('signal_type'),
        'bubble_level': bubble.get('level', ''),
        'bubble_sell_pct': bubble.get('sell_pct', 0.5),
        'bubble_reasons': bubble.get('reasons', []),
        'bubble_signals': bubble.get('signals', {}),
        'action_300': action_300,
        'action_2000': action_2000,
        'position_advice': advice,
    }
