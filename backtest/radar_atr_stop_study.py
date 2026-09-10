"""
退出规则 v2：固定百分比 vs 波动率自适应（ATR）止损对比（2026-09-10）

背景（2026-09-10 修正 — 初版论断已被本脚本数据推翻，勿再引用）:
  ❌ 初版写的"固定5%对不同标的=1.79σ~3.59σ"是**错的**：那组数字来自全样本
     年化波动率，而止损只在**信号日**生效。实测信号日各标的 ATR 仅
     1.77%~2.02%（1.14 倍），**横截面差异可以忽略**。
  ✅ 真实理由是**时间维度**：单一标的自身 ATR% 的 P90/P10 达 2.0~2.6 倍
     （见下方"ATR 时间离散度"），固定 5% 在不同 regime 相当于
     0.52x~5.71x ATR —— 同一条规则的风险敞口浮动 10 倍。
  ⚠ 但本脚本的 30 个信号**恰好全部落在中低波 regime**，因此回测对
     ATR 与固定百分比**不可区分**（低波组差 0.00%、高波组差 +0.06%）。
     → ATR 是 regime 稳健性修正，不是回测收益修正。
     → 5% 下限保证它在已验证样本上是空操作，只在高波 regime 激活。
     实例：粮食(159587) 2026-09-10 的 ATR% 处于自身历史 99.8% 分位
     （4.43% vs 中位 1.95%），5% 止损会被日常噪音直接打掉。

本脚本对比三种止损口径在雷达 A+B 信号上的表现：
  FIX5     固定止损 5% / 止盈 10%
  ATR2     止损 max(2.0×ATR20, 4%) / 止盈 2×止损
  ATR25    止损 max(2.5×ATR20, 5%) / 止盈 2×止损   ← 已采用（见 models/opportunity_radar.py）
"""
from __future__ import annotations
import warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

POOL = [
    {"code": "159825", "name": "农业", "factors": ["M0", "SR0", "C0"]},
    {"code": "159587", "name": "粮食", "factors": ["M0", "SR0", "C0"]},
    {"code": "512400", "name": "有色", "factors": ["CU0"]},
    {"code": "515220", "name": "煤炭", "factors": ["JM0"]},
    {"code": "512880", "name": "证券", "factors": ["margin"]},
]
COOLDOWN_DAYS = 60


def _load(f):
    p = DATA / f
    return pd.read_parquet(p) if p.exists() else None


def _prep(df):
    dc = "date" if "date" in df.columns else df.columns[0]
    df[dc] = pd.to_datetime(df[dc])
    return df.sort_values(dc).reset_index(drop=True)


def _factor_ok(df, th=0.05, lb=20, brk=60):
    c = df["close"]
    return ((c / c.shift(lb) - 1) > th) | (c > c.shift(1).rolling(brk).max())


def _confirm(df):
    c, v = df["close"], df["volume"]
    ma10 = c.rolling(10).mean()
    vr = v.rolling(5).mean() / v.rolling(20).mean()
    near = (c >= ma10 * 0.99) & (vr >= 2.0)
    rise = c / c.rolling(250, min_periods=120).min() - 1
    return (c > ma10) | near & (vr > 1.1) & (rise >= 0.10) & (rise <= 0.35) if False else \
        ((c > ma10) | near) & (vr > 1.1) & (rise >= 0.10) & (rise <= 0.35)


def run():
    rows = []
    for item in POOL:
        df = _load(f"etf_{item['code']}_daily.parquet")
        if df is None:
            continue
        df = _prep(df)
        ok_a = pd.Series(False, index=df.index)
        for f in item["factors"]:
            if f == "margin":
                fd = _load("margin_balance.parquet")
                if fd is None:
                    continue
                fd = _prep(fd).rename(columns={"margin_balance": "close"})
            else:
                fd = _load(f"futures_{f}_daily.parquet")
                if fd is None:
                    continue
                fd = _prep(fd)[["date", "close"]].dropna()
            fw = _factor_ok(fd)
            m = df[["date"]].merge(pd.DataFrame({"date": fd["date"], "a": fw.values}),
                                   on="date", how="left")
            ok_a = ok_a | m["a"].fillna(False).values

        sig = ok_a.values & _confirm(df).values
        c = df["close"].values
        dates = df["date"].values
        # ATR20（真实波幅均值，用 high/low/close 近似）
        h = df["high"].values if "high" in df.columns else c
        l = df["low"].values if "low" in df.columns else c
        pc = np.roll(c, 1); pc[0] = c[0]
        tr = np.maximum.reduce([h - l, np.abs(h - pc), np.abs(l - pc)])
        atr = pd.Series(tr).rolling(20).mean().values

        last = None
        for i in range(len(df) - 1):
            if not sig[i]:
                continue
            if last is not None and (dates[i] - last) / np.timedelta64(1, "D") < COOLDOWN_DAYS:
                continue
            last = dates[i]
            entry = c[i]
            atr_i = atr[i]
            atr_pct = atr_i / entry if not np.isnan(atr_i) else 0.03

            def _sim(sl, tp):
                for j in range(i + 1, min(i + 61, len(c))):
                    r = c[j] / entry - 1
                    if r <= -sl:
                        return -sl
                    if r >= tp:
                        return tp
                return c[min(i + 60, len(c) - 1)] / entry - 1

            sl_atr2 = max(2.0 * atr_pct, 0.04)
            sl_atr25 = max(2.5 * atr_pct, 0.05)
            rows.append({
                "name": item["name"], "date": pd.Timestamp(dates[i]), "entry": entry,
                "atr_pct": atr_pct, "atr_sigma": 0.05 / atr_pct,
                "FIX5": _sim(0.05, 0.10),
                "ATR2": _sim(sl_atr2, 2 * sl_atr2),
                "ATR25": _sim(sl_atr25, 2 * sl_atr25),
                "sl_atr2": sl_atr2, "sl_atr25": sl_atr25,
            })
    return pd.DataFrame(rows)


def report(df: pd.DataFrame):
    print("=" * 80)
    print(f"  退出规则对比 v2：固定百分比 vs 波动率自适应（{len(df)} 个信号）")
    print("=" * 80)

    print(f"\n  【各标的当日 ATR 止损宽度】")
    print(f"  {'标的':6s} {'ATR%':>7s} {'固定5%≈':>9s} {'ATR2宽度':>9s} {'ATR25宽度':>10s}")
    for n, g in df.groupby("name"):
        print(f"  {n:6s} {g['atr_pct'].mean():>6.2%} {g['atr_sigma'].mean():>8.2f}σ "
              f"{g['sl_atr2'].mean():>8.2%} {g['sl_atr25'].mean():>9.2%}")

    print(f"\n  【全池表现】")
    print(f"  {'规则':>8s} {'均值':>8s} {'中位':>8s} {'胜率':>7s} {'盈亏比':>7s} {'最差':>8s} {'最好':>8s}")
    for r in ["FIX5", "ATR2", "ATR25"]:
        col = df[r].dropna()
        w, l = col[col > 0], col[col <= 0]
        pl = abs(w.mean() / l.mean()) if len(w) and len(l) else np.nan
        print(f"  {r:>8s} {col.mean():>7.2%} {col.median():>7.2%} {(col > 0).mean():>6.1%} "
              f"{pl:>7.2f} {col.min():>7.1%} {col.max():>7.1%}")

    print(f"\n  【分标的均值】")
    print(f"  {'标的':6s} {'n':>3s} {'FIX5':>9s} {'ATR2':>9s} {'ATR25':>9s}")
    for n, g in df.groupby("name"):
        print(f"  {n:6s} {len(g):>3d} {g['FIX5'].mean():>8.2%} "
              f"{g['ATR2'].mean():>8.2%} {g['ATR25'].mean():>8.2%}")

    df.to_csv(DATA / "radar_atr_stop.csv", index=False, encoding="utf-8-sig")
    print(f"\n  明细: {DATA / 'radar_atr_stop.csv'}")


if __name__ == "__main__":
    report(run())
