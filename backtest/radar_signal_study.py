"""
机会雷达信号实证研究（2026-09-10）

目的：用雷达自身的 A+B 信号在历史上的真实分布，为"仓位上限"和"退出规则"
提供实证依据，而不是拍脑袋定 5% / 10%。

方法：
  1. 在雷达 ETF 池上回放 A+B 共振信号（与 models/opportunity_radar.py 同口径）
  2. 对每个信号计算：
     - 前向收益（5/10/20/60 交易日）
     - 最大不利偏移 MAE（持有期内最深浮亏）→ 决定认错线
     - 两种退出规则的收益分布：
       ① 时间退出：固定持有 N 日
       ② 因子退出：因子窗口关闭（r20 ≤ 5%）或 ETF 跌破 MA10 即出
  3. 用 MAE 分位数反推仓位上限（风险预算法）

输出：控制台报告 + data/radar_signal_study.csv（逐笔明细）
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

# 行业池（含因子代理）
POOL = [
    {"code": "159825", "name": "农业", "factors": ["M0", "SR0", "C0"]},
    {"code": "159587", "name": "粮食", "factors": ["M0", "SR0", "C0"]},
    {"code": "512400", "name": "有色", "factors": ["CU0"]},
    {"code": "515220", "name": "煤炭", "factors": ["JM0"]},
    {"code": "512880", "name": "证券", "factors": ["margin"]},
    {"code": "512480", "name": "半导体", "factors": None},
    {"code": "512010", "name": "医药", "factors": None},
    {"code": "512660", "name": "军工", "factors": None},
    {"code": "159928", "name": "消费", "factors": None},
    {"code": "159611", "name": "电力", "factors": None},
    {"code": "512690", "name": "白酒", "factors": None},
    {"code": "515030", "name": "新能源车", "factors": None},
]

COOLDOWN_DAYS = 60      # 与 radar 一致
FWD_WINDOWS = [5, 10, 20, 60]


def _load(fname: str) -> pd.DataFrame | None:
    p = DATA / fname
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    dc = "date" if "date" in df.columns else df.columns[0]
    df[dc] = pd.to_datetime(df[dc])
    return df.sort_values(dc).reset_index(drop=True)


def _factor_window(df: pd.DataFrame, chg_th: float = 0.05,
                   lookback: int = 20, brk: int = 60) -> pd.Series:
    """因子窗口：近20日涨幅 > 5% 或 突破60日新高。逐日布尔序列。"""
    c = df["close"]
    chg = c / c.shift(lookback) - 1
    breakout = c > c.shift(1).rolling(brk).max()
    return (chg > chg_th) | breakout


def _confirm(df: pd.DataFrame) -> pd.Series:
    """B 行情确认（与 radar._confirm_signal 同口径），逐日布尔序列。"""
    c = df["close"]
    v = df["volume"] if "volume" in df.columns else None
    ma10 = c.rolling(10).mean()
    above = c > ma10
    if v is not None:
        vol_ratio = v.rolling(5).mean() / v.rolling(20).mean()
    else:
        vol_ratio = pd.Series(1.5, index=df.index)
    near = (c >= ma10 * 0.99) & (vol_ratio >= 2.0)
    lo250 = c.rolling(250, min_periods=120).min()
    rise = c / lo250 - 1
    return (above | near) & (vol_ratio > 1.1) & (rise >= 0.10) & (rise <= 0.35)


def build_signals() -> pd.DataFrame:
    rows = []
    for item in POOL:
        if not item["factors"]:
            continue  # 无因子代理的行业不报机会（radar 同口径）
        df = _load(f"etf_{item['code']}_daily.parquet")
        if df is None or len(df) < 300:
            continue

        # A 条件：任一因子开窗
        ok_a = pd.Series(False, index=df.index)
        fac_top = None  # 主因子（用于因子退出判定）
        best_len = -1
        for f in item["factors"]:
            if f == "margin":
                fd = _load("margin_balance.parquet")
                if fd is None:
                    continue
                fd = fd.rename(columns={"margin_balance": "close"})
            else:
                fd = _load(f"futures_{f}_daily.parquet")
            if fd is None or len(fd) < 100:
                continue
            fd = fd[["date", "close"]].dropna()
            fw = _factor_window(fd)
            m = df[["date"]].merge(
                pd.DataFrame({"date": fd["date"], "ok_a": fw.values}), on="date", how="left"
            )
            ok_a = ok_a | m["ok_a"].fillna(False).values
            if len(fd) > best_len:
                best_len, fac_top = len(fd), f

        ok_b = _confirm(df).values
        sig = ok_a & ok_b

        # 主因子序列（退出用）
        if fac_top and fac_top != "margin":
            fd = _load(f"futures_{fac_top}_daily.parquet")[["date", "close"]].dropna()
            fac_ok = _factor_window(fd)
            fm = df[["date"]].merge(
                pd.DataFrame({"date": fd["date"], "fac_ok": fac_ok.values}), on="date", how="left"
            )
            fac_ok_al = fm["fac_ok"].fillna(False).values
        else:
            fac_ok_al = ok_a.values

        c = df["close"].values
        ma10 = df["close"].rolling(10).mean().values
        dates = df["date"].values

        last = None
        for i in range(len(df) - 1):
            if not sig[i]:
                continue
            if last is not None and (dates[i] - last) / np.timedelta64(1, "D") < COOLDOWN_DAYS:
                continue
            last = dates[i]

            entry = c[i]
            rec = {"code": item["code"], "name": item["name"], "date": pd.Timestamp(dates[i]),
                   "entry": entry, "factor": fac_top}
            # 前向收益
            for w in FWD_WINDOWS:
                rec[f"ret_{w}d"] = (c[i + w] / entry - 1) if i + w < len(c) else np.nan

            # MAE：持有到 因子窗口关闭 或 跌破MA10 或 最长60日
            end = min(i + 60, len(c) - 1)
            exit_idx, exit_reason = end, "max60"
            for j in range(i + 1, min(i + 61, len(c))):
                if not fac_ok_al[j]:
                    exit_idx, exit_reason = j, "因子关闭"
                    break
                if c[j] < ma10[j]:
                    exit_idx, exit_reason = j, "破MA10"
                    break
            seg = c[i + 1:exit_idx + 1]
            rec["mae"] = (seg.min() / entry - 1) if len(seg) else 0.0
            rec["mfe"] = (seg.max() / entry - 1) if len(seg) else 0.0
            rec["hold_days"] = exit_idx - i
            rec["ret_factor_exit"] = c[exit_idx] / entry - 1
            rec["exit_reason"] = exit_reason
            rows.append(rec)

    return pd.DataFrame(rows)


def compare_exits() -> None:
    """对比多种退出规则在同一批信号上的表现（同一入场集合，只变退出）。

    规则：
      T20      固定持有 20 个交易日
      T60      固定持有 60 个交易日
      F5       因子窗口关闭（r20 ≤ 5% 且非突破）即出
      F5+M     因子关闭 或 跌破 MA20 即出
      ST       止损 5% / 止盈 10%（先到先出）
      ST6      止损 4% / 止盈 12%
      MA20     跌破 MA20 即出
    """
    rules = ["T20", "T60", "F5", "F5+M20", "ST_5_10", "ST_4_12", "MA20"]
    all_rows = []

    for item in POOL:
        if not item["factors"]:
            continue
        df = _load(f"etf_{item['code']}_daily.parquet")
        if df is None or len(df) < 300:
            continue

        ok_a = pd.Series(False, index=df.index)
        fac_series = {}
        for f in item["factors"]:
            if f == "margin":
                fd = _load("margin_balance.parquet")
                if fd is None:
                    continue
                fd = fd.rename(columns={"margin_balance": "close"})
            else:
                fd = _load(f"futures_{f}_daily.parquet")
            if fd is None or len(fd) < 100:
                continue
            fd = fd[["date", "close"]].dropna()
            fw = _factor_window(fd)
            fac_series[f] = (fd["date"].values, fw.values)
            m = df[["date"]].merge(
                pd.DataFrame({"date": fd["date"], "ok_a": fw.values}), on="date", how="left")
            ok_a = ok_a | m["ok_a"].fillna(False).values

        ok_b = _confirm(df).values
        sig = ok_a & ok_b
        c = df["close"].values
        dates = df["date"].values
        ma20 = df["close"].rolling(20).mean().values

        last = None
        for i in range(len(df) - 1):
            if not sig[i]:
                continue
            if last is not None and (dates[i] - last) / np.timedelta64(1, "D") < COOLDOWN_DAYS:
                continue
            last = dates[i]
            entry = c[i]

            # F5 因子关闭日
            f5_idx, f5m_idx = None, None
            for j in range(i + 1, min(i + 61, len(c))):
                closed = True
                for f, (fdt, fok) in fac_series.items():
                    pos = np.searchsorted(fdt, dates[j], side="right") - 1
                    if pos >= 0 and fok[pos]:
                        closed = False
                        break
                if closed and f5_idx is None:
                    f5_idx = j
                if closed and c[j] < ma20[j] and f5m_idx is None:
                    f5m_idx = j
                    break

            def _ret(idx):
                if idx is None or idx >= len(c):
                    idx = min(i + 60, len(c) - 1)
                return c[idx] / entry - 1

            # 止损止盈
            def _st(sl, tp):
                for j in range(i + 1, min(i + 61, len(c))):
                    r = c[j] / entry - 1
                    if r <= -sl:
                        return -sl
                    if r >= tp:
                        return tp
                return _ret(min(i + 60, len(c) - 1))

            ma20_idx = next((j for j in range(i + 1, min(i + 61, len(c)))
                             if c[j] < ma20[j]), None)

            all_rows.append({
                "name": item["name"], "date": pd.Timestamp(dates[i]), "entry": entry,
                "T20": _ret(min(i + 20, len(c) - 1)),
                "T60": _ret(min(i + 60, len(c) - 1)),
                "F5": _ret(f5_idx),
                "F5+M20": _ret(f5m_idx if f5m_idx is not None else f5_idx),
                "ST_5_10": _st(0.05, 0.10),
                "ST_4_12": _st(0.04, 0.12),
                "MA20": _ret(ma20_idx),
            })

    df = pd.DataFrame(all_rows)
    if df.empty:
        print("无信号")
        return
    print("\n" + "=" * 78)
    print(f"  退出规则对比（{len(df)} 个信号，入场集合完全相同，只变退出方式）")
    print("=" * 78)
    print(f"\n  {'规则':>9s} {'均值':>8s} {'中位':>8s} {'胜率':>7s} {'盈亏比':>7s} "
          f"{'最差':>8s} {'最好':>8s} {'期望×胜率':>10s}")
    for r in rules:
        col = df[r].dropna()
        w, l = col[col > 0], col[col <= 0]
        pl = abs(w.mean() / l.mean()) if len(w) and len(l) else np.nan
        exp = col.mean()
        print(f"  {r:>9s} {col.mean():>7.2%} {col.median():>7.2%} {(col > 0).mean():>6.1%} "
              f"{pl:>7.2f} {col.min():>7.1%} {col.max():>7.1%} {exp:>9.2%}")

    df.to_csv(ROOT / "data" / "radar_exit_compare.csv", index=False, encoding="utf-8-sig")
    print(f"\n  明细: {ROOT / 'data' / 'radar_exit_compare.csv'}")


def report(df: pd.DataFrame) -> None:
    if df.empty:
        print("无信号")
        return
    n = len(df)
    print("=" * 78)
    print(f"  机会雷达信号实证 —— 全池 {n} 个 A+B 信号（{df['date'].min().date()} ~ {df['date'].max().date()}）")
    print("=" * 78)
    print(f"\n  分标的信号数：")
    for nm, g in df.groupby("name"):
        print(f"    {nm:8s} {len(g):3d} 个")

    print(f"\n  【前向收益分布】")
    print(f"  {'窗口':>6s} {'均值':>8s} {'中位':>8s} {'胜率':>7s} {'最好':>8s} {'最差':>8s}")
    for w in FWD_WINDOWS:
        col = df[f"ret_{w}d"].dropna()
        if len(col) < 5:
            continue
        print(f"  {w:>4d}日 {col.mean():>7.2%} {col.median():>7.2%} "
              f"{(col > 0).mean():>6.1%} {col.max():>7.1%} {col.min():>7.1%}")

    print(f"\n  【最大不利偏移 MAE】（持有期内最深浮亏）")
    mae = df["mae"].dropna()
    for q in [0.50, 0.75, 0.90, 0.95, 1.00]:
        print(f"    {q:>5.0%} 分位: {mae.quantile(q):>7.2%}")

    print(f"\n  【最大有利偏移 MFE】")
    mfe = df["mfe"].dropna()
    for q in [0.50, 0.75, 0.90, 0.95, 1.00]:
        print(f"    {q:>5.0%} 分位: {mfe.quantile(q):>7.2%}")

    print(f"\n  【因子退出规则表现】（窗口关闭 or 破MA10 即出）")
    r = df["ret_factor_exit"].dropna()
    print(f"    均值 {r.mean():>7.2%}  中位 {r.median():>7.2%}  胜率 {(r > 0).mean():>6.1%}  "
          f"最差 {r.min():>7.1%}  最好 {r.max():>7.1%}")
    print(f"    平均持有 {df['hold_days'].mean():.1f} 个交易日（中位 {df['hold_days'].median():.0f}）")
    print(f"    退出原因：{dict(df['exit_reason'].value_counts())}")

    print(f"\n  【盈亏比 / 期望值】")
    win, loss = r[r > 0], r[r <= 0]
    if len(win) and len(loss):
        print(f"    平均盈利 {win.mean():>7.2%}（{len(win)}笔）  "
              f"平均亏损 {loss.mean():>7.2%}（{len(loss)}笔）  盈亏比 {abs(win.mean()/loss.mean()):.2f}")
        print(f"    期望值/笔 {r.mean():>7.2%}")

    csv = ROOT / "data" / "radar_signal_study.csv"
    df.to_csv(csv, index=False, encoding="utf-8-sig")
    print(f"\n  逐笔明细已存: {csv}")


if __name__ == "__main__":
    report(build_signals())
    compare_exits()
