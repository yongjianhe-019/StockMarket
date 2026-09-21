"""
入场逻辑对比：老冰点(分数≥50) vs 新机会发现(值得博弈)

同一时期、同一止盈规则(10%止盈半仓)、同仓位(25%)，只换入场信号，隔离"买点优化"的效果。
运行: .venv/bin/python backtest/entry_compare.py
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

START = pd.Timestamp("2023-10-01")
END = pd.Timestamp("2026-09-21")
TP, COOLDOWN, POS = 0.10, 10, 0.25


def sim(d: pd.DataFrame, sig: pd.Series, name: str) -> dict:
    cash, lots, last = 100000.0, [], -10**9
    nav, n_sig, n_tp = [], 0, 0
    for idx, row in d.iterrows():
        if row["date"] < START or row["date"] > END:
            continue
        px = float(row["close"])
        for lot in lots:
            if not lot["half"] and px >= lot["entry"] * (1 + TP):
                cash += lot["shares"] / 2 * px
                lot["shares"] /= 2
                lot["half"] = True
                n_tp += 1
        if bool(sig.loc[idx]) and idx - last >= COOLDOWN and cash > 1000:
            total = cash + sum(l["shares"] * px for l in lots)
            amt = min(cash, total * POS)
            lots.append({"entry": px, "shares": amt / px, "half": False})
            cash -= amt
            last = idx
            n_sig += 1
        nav.append(cash + sum(l["shares"] * px for l in lots))
    nav = pd.Series(nav, index=d.loc[(d["date"] >= START) & (d["date"] <= END), "date"].values)
    ret = nav.iloc[-1] / nav.iloc[0] - 1
    dd = float((nav / nav.cummax() - 1).min())
    return {"name": name, "ret": ret, "dd": dd, "n": n_sig, "tp": n_tp,
            "tp_rate": n_tp / n_sig if n_sig else 0}


def main() -> None:
    a = fetch_all_data(force=False)
    _ = fetch_all_macro(force=False)
    patch_csi2000_index(a)
    d = prep(a["csi300_daily"])
    sc = pd.read_csv(ROOT / "data/csi300_score_history.csv")
    sc["date"] = pd.to_datetime(sc["date"])
    d = d.merge(sc, on="date", how="left")
    ice_sig = (d["score"] >= 50).fillna(False)

    # 新机会发现（walk-forward 表）
    year_tables = {}
    for y in sorted(d["date"].dt.year.unique()):
        first = int(d[d["date"].dt.year == y].index[0])
        train = d.iloc[:max(0, first - H)]
        year_tables[y] = win_table(train) if len(train) >= 500 else None
    opp = []
    for idx, row in d.iterrows():
        wt = year_tables.get(row["date"].year)
        opp.append(bool(wt and evaluate(row, wt)["eligible"]))
    opp_sig = pd.Series(opp, index=d.index)

    print("=" * 74)
    print(f"  沪深300 入场逻辑对比（{START.date()} ~ {END.date()}，10%止盈半仓，仓位25%）")
    print("=" * 74)
    print(f"  {'入场逻辑':<26}{'收益':>9}{'最大回撤':>10}{'信号':>6}{'命中+10%':>10}")
    for sig, name in [(ice_sig, "老冰点(分数≥50)"), (opp_sig, "新机会发现(值得博弈)")]:
        r = sim(d, sig, name)
        print(f"  {r['name']:<26}{r['ret']:>+9.1%}{r['dd']:>+10.1%}{r['n']:>6}{r['tp_rate']:>10.0%}")


if __name__ == "__main__":
    main()
