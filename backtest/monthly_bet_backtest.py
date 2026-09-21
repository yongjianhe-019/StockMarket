"""
近3年按月回测：值得博弈买点 + 10%止盈半仓

规则（用户指定）：
  - 入场：机会发现（值得博弈：EV>0 且 R:R≥1.5 且 站回MA20），按分数凯利仓位
  - 止盈：浮盈 +10% 卖出**一半**，余下一半继续持有
  - 卖完后继续寻找新机会买入
  - 条件胜率表 walk-forward（只用测试年之前数据，embargo=H）
输出：按月划分的收益 + 胜率（命中+10%的比例）+ 汇总。

运行: .venv/bin/python backtest/monthly_bet_backtest.py [start] [end]
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.fetcher import fetch_all_data, patch_csi2000_index  # noqa: E402
from macro.fetcher import fetch_all_macro  # noqa: E402
from models.opportunity_entry import H, evaluate, prep, win_table  # noqa: E402

START = pd.Timestamp(sys.argv[1] if len(sys.argv) > 1 else "2023-09-21")
END = pd.Timestamp(sys.argv[2] if len(sys.argv) > 2 else "2026-09-21")
TP = 0.10          # 止盈线
COOLDOWN = 10      # 信号冷却（交易日）
CAP_TOTAL = 0.80   # 总仓位上限


def sim(name: str, d: pd.DataFrame, a: dict) -> pd.DataFrame:
    """逐日推进：机会买入 + 10%止盈半仓。返回日度净值表。"""
    cash, lots, last = 100000.0, [], -10**9
    nav_rows, trades = [], []
    year_tables = {}
    for y in sorted(d["date"].dt.year.unique()):
        first = int(d[d["date"].dt.year == y].index[0])
        train = d.iloc[:max(0, first - H)]
        year_tables[y] = win_table(train) if len(train) >= 500 else None

    for idx, row in d.iterrows():
        if row["date"] < START or row["date"] > END:
            continue
        px = float(row["close"])
        # 止盈半仓
        for lot in lots:
            if not lot["half"] and px >= lot["entry"] * (1 + TP):
                half = lot["shares"] / 2
                cash += half * px
                lot["shares"] -= half
                lot["half"] = True
                lot["tp_date"] = row["date"]
                lot["tp_ret"] = px / lot["entry"] - 1
        # 机会买入
        wt = year_tables.get(row["date"].year)
        if wt is not None and idx - last >= COOLDOWN:
            e = evaluate(row, wt)
            total = cash + sum(l["shares"] * px for l in lots)
            held = total - cash
            if e["eligible"] and e["position_pct"] > 0 and cash > 1000:
                amt = min(cash, total * e["position_pct"], max(0.0, total * CAP_TOTAL - held))
                if amt > 1000:
                    lots.append({"entry": px, "shares": amt / px, "half": False,
                                 "entry_date": row["date"], "pct": e["position_pct"]})
                    cash -= amt
                    last = idx
                    trades.append({"entry_date": row["date"], "entry": px,
                                   "pct": e["position_pct"], "rr": e["rr"], "win_p": e["win"]})
        nav_rows.append({"date": row["date"], "nav": cash + sum(l["shares"] * px for l in lots)})

    nav = pd.DataFrame(nav_rows).set_index("date")["nav"]
    # 交易胜率：命中+10% 为一胜；否则按平仓/期末浮动
    tdf = pd.DataFrame(trades)
    if not tdf.empty:
        tp_hits = {l["entry_date"] for l in lots if l.get("half")}
        tdf["hit_tp"] = tdf["entry_date"].isin(tp_hits)
    return nav, tdf, lots


def report(name: str, nav: pd.Series, tdf: pd.DataFrame, lots: list) -> None:
    print(f"\n{'='*80}")
    print(f"  {name}  (10%止盈半仓)")
    print(f"{'='*80}")
    ret = nav.iloc[-1] / nav.iloc[0] - 1
    dd = float((nav / nav.cummax() - 1).min())
    n = len(tdf)
    hits = int(tdf["hit_tp"].sum()) if n else 0
    print(f"  总收益 {ret:+.1%}   最大回撤 {dd:.1%}   信号 {n} 笔   命中+10% {hits} 笔 ({hits/n:.0%})" if n
          else f"  总收益 {ret:+.1%}   最大回撤 {dd:.1%}   无信号")
    # 按月
    m = nav.resample("ME").last().pct_change().dropna()
    m.iloc[0] = nav[nav.index <= m.index[0]].iloc[-1] / 100000 - 1
    print(f"\n  {'月份':<10}{'收益':>9}")
    for dt, r in m.items():
        print(f"  {str(dt)[:7]:<10}{r:>+9.2%}")
    if n:
        print(f"\n  交易明细:")
        for _, t in tdf.iterrows():
            print(f"    {str(t['entry_date'])[:10]} 入{t['entry']:.1f} 仓位{t['pct']:.0%} "
                  f"R:R{t['rr']:.1f} 胜率{t['win_p']:.0%}  {'✅命中+10%' if t['hit_tp'] else '持有中'}")


def main() -> None:
    a = fetch_all_data(force=False)
    _ = fetch_all_macro(force=False)
    patch_csi2000_index(a)
    navs = {}
    for name, key in [("沪深300", "csi300_daily"), ("中证2000", "csi2000_daily")]:
        d = prep(a[key])
        nav, tdf, lots = sim(name, d, a)
        navs[name] = nav
        report(name, nav, tdf, lots)
    # 组合：等权合并日收益
    if len(navs) == 2:
        r = pd.concat([navs["沪深300"].pct_change(), navs["中证2000"].pct_change()], axis=1).fillna(0).mean(axis=1)
        comb = 100000 * (1 + r).cumprod()
        report("组合(300+2000等权)", comb, pd.DataFrame(), [])


if __name__ == "__main__":
    main()
