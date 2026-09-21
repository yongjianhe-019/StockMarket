"""
逐次下跌的机会与"值得博弈点"分析

对每一次下跌，量化回答：有没有出现值得博弈的买点？判据（全部时序、无未来）：
  - 风险 risk   = 入场价 - 近20日低（跌破即证伪）
  - 回报 reward = 近120日高 - 入场价（前高为目标）
  - 赔率 R:R    = reward / risk
  - 条件胜率 p  = 历史 (回撤档 × regime) 未来60日胜率（Wilson 下界一并给）
  - 期望值 EV   = p×reward - (1-p)×risk     （>0 才值得博弈）

"值得博弈" = EV>0 且 R:R>=1.5 且 价格站回MA20（企稳，不接飞刀）。

运行: .venv/bin/python backtest/opportunity_rr.py [start] [end]
"""
from __future__ import annotations

import sys
import warnings
from math import sqrt
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.fetcher import fetch_all_data, patch_csi2000_index  # noqa: E402
from macro.fetcher import fetch_all_macro  # noqa: E402

H = 60
DD_EPISODE = 0.08      # 回撤>=8% 视为一次下跌
RR_MIN = 1.5
START = pd.Timestamp(sys.argv[1] if len(sys.argv) > 1 else "2024-01-01")
END = pd.Timestamp(sys.argv[2] if len(sys.argv) > 2 else "2026-09-21")


def wilson_lb(w, n, z=1.96):
    if n == 0:
        return 0.0
    p = w / n
    return max(0.0, (p + z*z/(2*n) - z*sqrt(p*(1-p)/n + z*z/(4*n*n))) / (1 + z*z/n))


def prep(daily: pd.DataFrame) -> pd.DataFrame:
    d = daily.sort_values("date").reset_index(drop=True).copy()
    c = d["close"]
    d["ma20"] = c.rolling(20).mean()
    d["ma200"] = c.rolling(200).mean()
    d["dd"] = 1 - c / c.rolling(252).max()
    d["low20"] = d["low"].rolling(20).min()
    d["high120"] = d["high"].rolling(120).max()
    d["regime"] = np.where(c > d["ma200"], "risk_on", "risk_off")
    d["dd_bucket"] = pd.cut(d["dd"], [-1, .05, .10, .15, .20, .30, 1],
                            labels=["<5%", "5-10%", "10-15%", "15-20%", "20-30%", ">30%"])
    f = np.full(len(d), np.nan)
    for i in range(len(d) - 1):
        j = min(i + H, len(d) - 1)
        f[i] = c.iloc[j] / c.iloc[i] - 1
    d["fwd"] = f
    return d


def win_table(d):
    t = {}
    for (dd, rg), g in d.dropna(subset=["fwd", "dd_bucket"]).groupby(["dd_bucket", "regime"], observed=True):
        n = len(g); w = int((g["fwd"] > 0).sum())
        t[(str(dd), rg)] = (w / n if n else 0, wilson_lb(w, n), n)
    return t


def analyze(d, wt, name):
    d = d[(d["date"] >= START) & (d["date"] <= END)].reset_index(drop=True)
    # 连续下跌段（dd>=8%）
    ep_id = (d["dd"] >= DD_EPISODE).astype(int)
    grp = (ep_id.diff() != 0).cumsum()
    print(f"\n{'='*104}")
    print(f"  {name}  逐次下跌的机会分析  {START.date()} ~ {END.date()}")
    print(f"{'='*104}")
    print(f"  {'下跌段':<22}{'谷底':<12}{'最大回撤':>8}{'谷底R:R':>9}{'谷底EV':>9}"
          f"{'胜率':>7}{'值得博弈日':<12}{'买点价':>9}{'至今':>9}{'期间最低':>9}")
    for _, g in d[ep_id == 1].groupby(grp[ep_id == 1]):
        s = g.iloc[0]
        trough = g.loc[g["close"].idxmin()]
        # 谷底当天的 R:R / EV
        def rr_ev(row):
            risk = row["close"] - row["low20"]
            reward = row["high120"] - row["close"]
            if risk <= 0 or not np.isfinite(reward):
                return np.nan, np.nan, np.nan, np.nan
            p, wlb, n = wt.get((str(row["dd_bucket"]), row["regime"]), (0, 0, 0))
            return reward/risk, p*reward-(1-p)*risk, p, wlb
        rr_t, ev_t, p_t, wlb_t = rr_ev(trough)
        # 段内第一个"值得博弈"日
        bet = None
        for _, row in g.iterrows():
            rr, ev, p, wlb = rr_ev(row)
            if np.isfinite(ev) and ev > 0 and rr >= RR_MIN and row["close"] > row["ma20"]:
                bet = row
                break
        end_px = float(d["close"].iloc[-1])
        if bet is not None:
            sub = d[d["date"] >= bet["date"]]
            lo = float(sub["low"].min())
            bet_str = f"{str(bet['date'])[5:10]}"
            bet_line = (f"{bet_str:<12}{bet['close']:>9.1f}{end_px:>9.1f}"
                        f"{lo:>9.1f}")
        else:
            bet_line = f"{'—':<12}{'':>9}{'':>9}{'':>9}"
        print(f"  {str(s['date'])[:10]}~{str(g['date'].iloc[-1])[:10]}  {str(trough['date'])[:10]:<12}"
              f"{trough['dd']:>8.1%}{rr_t:>9.2f}{ev_t:>9.1f}{p_t:>7.0%}{bet_line}")

    # 汇总：值得博弈日的实际胜率（分年）
    bets = []
    for _, row in d.iterrows():
        risk = row["close"] - row["low20"]
        reward = row["high120"] - row["close"]
        if risk <= 0 or not np.isfinite(reward) or row["close"] <= row["ma20"]:
            continue
        p, wlb, n = wt.get((str(row["dd_bucket"]), row["regime"]), (0, 0, 0))
        rr = reward/risk; ev = p*reward-(1-p)*risk
        if np.isfinite(ev) and ev > 0 and rr >= RR_MIN and np.isfinite(row["fwd"]):
            bets.append((row["date"].year, row["fwd"] > 0, row["fwd"]))
    if bets:
        bdf = pd.DataFrame(bets, columns=["year", "win", "fwd"])
        print(f"\n  汇总[值得博弈日] 分年：")
        for y, g in bdf.groupby("year"):
            w = int(g["win"].sum())
            print(f"    {y}: n={len(g):<3} 胜率={g['win'].mean():>4.0%} Wilson下界={wilson_lb(w,len(g)):>4.0%} "
                  f"均值60日收益={g['fwd'].mean():>+6.1%}")
        w = int(bdf["win"].sum())
        print(f"    合计: n={len(bdf)} 胜率={bdf['win'].mean():.0%} Wilson下界={wilson_lb(w,len(bdf)):.0%}")
    else:
        print(f"\n  汇总[值得博弈日]: 无")


def main():
    a = fetch_all_data(force=False)
    m = fetch_all_macro(force=False)
    patch_csi2000_index(a)
    for name, key in [("CSI300", "csi300_daily"), ("CSI2000", "csi2000_daily")]:
        d = prep(a[key])
        wt = win_table(d)
        analyze(d, wt, name)


if __name__ == "__main__":
    main()
