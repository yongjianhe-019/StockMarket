"""
值得博弈的冰点买点 —— 机会发现模块（生产）

本质仍是"冰点买入"，但把二元的"分数≥50"换成**性价比（R:R）+ 条件胜率 + 期望值**
的量化判定，解决"刻舟求剑"（固定阈值不随大盘水位变）的问题。

对每个标的，在**下跌（冰点）状态**下计算：
    风险 risk   = 现价 - 近20日低（跌破即证伪 → 止损位）
    回报 reward = 近120日高 - 现价（前高 → 目标）
    赔率 R:R    = reward / risk（性价比）
    条件胜率 p  = 历史 (回撤档 × regime) 未来60日胜率（含 Wilson 下界与样本数）
    期望 EV     = p×reward - (1-p)×risk
    值得博弈    = EV>0 且 R:R≥RR_MIN 且 现价>MA20（企稳，不接飞刀）
    建议仓位    = 分数凯利 (p×R:R-(1-p))/R:R，封顶 CAP；Wilson下界≤0.5 再打对折

止盈不在本模块范围（用户自行处理）。
"""
from __future__ import annotations

from math import sqrt
from typing import Optional

import numpy as np
import pandas as pd

H = 60            # 条件胜率的未来视野（交易日）
RR_MIN = 1.5      # 性价比门槛
KELLY = 0.5       # 分数凯利系数
CAP = 0.50        # 单标仓位上限
LOOKBACK_DD = 252
LOOKBACK_LOW = 20
LOOKBACK_HIGH = 120
MA_TREND = 200

TARGETS = [("沪深300", "csi300_daily"), ("中证2000", "csi2000_daily")]


def wilson_lb(w: int, n: int, z: float = 1.96) -> float:
    """Wilson 95% 置信下界（小样本胜率的诚实下限）。"""
    if n == 0:
        return 0.0
    p = w / n
    return max(0.0, (p + z * z / (2 * n)
                     - z * sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / (1 + z * z / n))


def prep(daily: pd.DataFrame) -> pd.DataFrame:
    """构造时序特征（全部用当日及以前，无未来函数）。"""
    d = daily.sort_values("date").reset_index(drop=True).copy()
    d["date"] = pd.to_datetime(d["date"])
    c = d["close"]
    d["ma20"] = c.rolling(20).mean()
    d[f"ma{MA_TREND}"] = c.rolling(MA_TREND).mean()
    d["dd"] = 1 - c / c.rolling(LOOKBACK_DD).max()
    d["low20"] = d["low"].rolling(LOOKBACK_LOW).min()
    d["high120"] = d["high"].rolling(LOOKBACK_HIGH).max()
    d["regime"] = np.where(c > d[f"ma{MA_TREND}"], "risk_on", "risk_off")
    d["dd_bucket"] = pd.cut(d["dd"], [-1, .05, .10, .15, .20, .30, 1],
                            labels=["<5%", "5-10%", "10-15%", "15-20%", "20-30%", ">30%"])
    # 标签（仅用于统计条件胜率；预测时不可见）
    f = np.full(len(d), np.nan)
    for i in range(len(d) - 1):
        j = min(i + H, len(d) - 1)
        f[i] = c.iloc[j] / c.iloc[i] - 1
    d["fwd"] = f
    return d


def win_table(d: pd.DataFrame) -> dict:
    """条件胜率表：{(回撤档, regime): (胜率, Wilson下界, n)}。

    v10 重叠修正：前瞻窗口 H 日前瞻收益在日频上高度重叠（相邻样本共享
    H-1 天），独立样本数 ≈ n/H。Wilson 下界若按原始 n 计算会**过度自信**
    （n=532 实为 n_eff≈9），进而把凯利仓位放得过大。故下界改按有效样本
    n_eff=round(n/H)、w_eff=round(w/H) 计算；点估计胜率 p 仍用全样本
    （每个交易日的条件胜率是无偏的），raw n 一并保留供展示。
    """
    t = {}
    for (dd, rg), g in d.dropna(subset=["fwd", "dd_bucket"]).groupby(
            ["dd_bucket", "regime"], observed=True):
        n = len(g)
        w = int((g["fwd"] > 0).sum())
        n_eff = max(1, int(round(n / H)))
        w_eff = int(round(w / H))
        t[(str(dd), rg)] = (w / n if n else 0.0, wilson_lb(w_eff, n_eff), n)
    return t


def evaluate(row: pd.Series, wt: dict) -> dict:
    """对单个时点给出机会判定。"""
    risk = float(row["close"] - row["low20"])
    reward = float(row["high120"] - row["close"])
    p, wlb, n = wt.get((str(row["dd_bucket"]), row["regime"]), (0.0, 0.0, 0))
    rr = reward / risk if risk > 0 else float("nan")
    ev = p * reward - (1 - p) * risk
    eligible = bool(np.isfinite(rr) and ev > 0 and rr >= RR_MIN
                    and np.isfinite(row["ma20"]) and row["close"] > row["ma20"])
    pos = 0.0
    if np.isfinite(rr) and rr > 0:
        pos = float(np.clip(KELLY * (p * rr - (1 - p)) / rr, 0.0, CAP))
        if wlb <= 0.50:
            pos *= 0.5
    return {
        "date": row["date"], "close": float(row["close"]), "dd": float(row["dd"]),
        "regime": row["regime"], "risk": risk, "reward": reward, "rr": rr,
        "win": p, "wilson": wlb, "n": n, "ev": ev,
        "eligible": eligible, "position_pct": round(pos, 4),
        "stop": float(row["low20"]), "target": float(row["high120"]),
    }


def scan(a_data: dict) -> list[dict]:
    """扫描所有标的的当前机会（值得博弈的买点）。"""
    out = []
    for name, key in TARGETS:
        if key not in a_data or a_data[key] is None:
            continue
        d = prep(a_data[key])
        wt = win_table(d)
        e = evaluate(d.iloc[-1], wt)
        e["name"] = name
        e["code"] = key
        out.append(e)
    return out


def scan_history(a_data: dict, name: str, key: str, tail: int = 90) -> list[dict]:
    """最近 tail 个交易日的触发记录（供复盘）。"""
    d = prep(a_data[key])
    wt = win_table(d)
    hits = []
    for _, row in d.tail(tail).iterrows():
        e = evaluate(row, wt)
        if e["eligible"]:
            hits.append(e)
    return hits


def missing_reasons(e: dict, ma20: float) -> list[str]:
    """当前不满足"值得博弈"卡在哪。"""
    miss = []
    if not (e["ev"] > 0):
        miss.append("EV≤0")
    if not (np.isfinite(e["rr"]) and e["rr"] >= RR_MIN):
        miss.append(f"R:R<{RR_MIN}")
    if not (np.isfinite(ma20) and e["close"] > ma20):
        miss.append(f"未站回MA20({ma20:.3f})")
    return miss
