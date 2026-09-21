"""
"值得博弈买点"回测（walk-forward，防未来函数）

每次提交必跑：验证机会发现逻辑的效果。
- 条件胜率表**只用测试年之前**的数据构建（并在训练/测试间留 H 交易日 embargo），
  杜绝用未来信息定阈值。
- 入场规则：值得博弈（EV>0 且 R:R≥1.5 且 站回MA20）→ 按分数凯利仓位买入，持有。
- 对照：同期 Buy&Hold；逐笔 60 日胜率（Wilson 下界）。

运行: .venv/bin/python backtest/opportunity_backtest.py [start_year] [end]
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
from models.opportunity_entry import (  # noqa: E402
    H, evaluate, prep, wilson_lb, win_table,
)

START_YEAR = int(sys.argv[1]) if len(sys.argv) > 1 else 2018
COOLDOWN = 20      # 同一标的信号冷却（交易日），避免连日重复计数


def run(name: str, key: str, a: dict) -> None:
    d = prep(a[key])
    years = sorted(y for y in d["date"].dt.year.unique() if y >= START_YEAR)

    trades = []
    for y in years:
        test = d[d["date"].dt.year == y]
        if test.empty:
            continue
        first_idx = int(test.index[0])
        train = d.iloc[:max(0, first_idx - H)]        # embargo = H
        if len(train) < 500:
            continue
        wt = win_table(train)
        last = -10**9
        for idx, row in test.iterrows():
            e = evaluate(row, wt)
            if e["eligible"] and idx - last >= COOLDOWN:
                last = idx
                e["fwd"] = row["fwd"]
                trades.append(e)

    if not trades:
        print(f"  {name}: 无信号")
        return

    tdf = pd.DataFrame(trades)
    tdf["date"] = pd.to_datetime(tdf["date"])

    print(f"\n{'='*92}")
    print(f"  {name} — 值得博弈买点 walk-forward 回测（{START_YEAR}~{int(d['date'].dt.year.max())}）")
    print(f"{'='*92}")
    print(f"  {'年份':<7}{'信号数':>7}{'60日胜率':>10}{'Wilson下界':>11}{'均值60日收益':>13}{'平均R:R':>9}")
    for y, g in tdf.groupby(tdf["date"].dt.year):
        valid = g.dropna(subset=["fwd"])
        n = len(valid)
        if n == 0:
            continue
        w = int((valid["fwd"] > 0).sum())
        print(f"  {y:<7}{len(g):>7}{w/n:>10.0%}{wilson_lb(w,n):>11.0%}"
              f"{valid['fwd'].mean():>+13.1%}{g['rr'].mean():>9.2f}")
    valid = tdf.dropna(subset=["fwd"])
    n = len(valid); w = int((valid["fwd"] > 0).sum())
    print(f"  {'合计':<7}{len(tdf):>7}{w/n:>10.0%}{wilson_lb(w,n):>11.0%}{valid['fwd'].mean():>+13.1%}{tdf['rr'].mean():>9.2f}")

    # 组合：信号日按凯利仓位买入，持有
    cash, units, last_d = 100000.0, 0.0, None
    nav = []
    sig = tdf.set_index(tdf["date"])["position_pct"]
    for _, row in d.iterrows():
        if row["date"] in sig.index:
            p = float(sig.loc[row["date"]])
            if isinstance(p, pd.Series):
                p = float(p.iloc[0])
            amt = cash * p
            units += amt / row["close"]
            cash -= amt
        nav.append(cash + units * row["close"])
    nav = pd.Series(nav)
    ret = nav.iloc[-1] / nav.iloc[0] - 1
    bh = d["close"].iloc[-1] / d["close"].iloc[0] - 1
    dd = float((nav / nav.cummax() - 1).min())
    print(f"  组合(凯利仓位,不卖出): {ret:+.1%}  vs B&H {bh:+.1%}  最大回撤 {dd:.1%}")


def main() -> None:
    a = fetch_all_data(force=False)
    _ = fetch_all_macro(force=False)
    patch_csi2000_index(a)
    for name, key in [("沪深300", "csi300_daily"), ("中证2000", "csi2000_daily")]:
        run(name, key, a)


if __name__ == "__main__":
    main()
