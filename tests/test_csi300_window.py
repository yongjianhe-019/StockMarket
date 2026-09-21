"""
CSI300 分位窗口回归测试 — 锁死"5年"口径

背景（2026-09-20 实盘复盘）:
- 估值源 legulegu 返回的是**月频**数据（约258行/21年），但代码原先按
  "5年×252交易日=1260行"取窗，`_compute_series_pct` 在行数不足时**静默
  回退到全部行**，把"5年分位"算成了"21.5年分位"。
- 后果：CSI300 估值温度虚增 +10、股债性价比虚增 +15，分数被抬高 25 分，
  在回撤仅 11%（历史低胜率区）假触发冰点 BUY。
- 修复：改为按**日期**取近5年窗口（`_window_pct`），数据不足则降级，绝不
  静默回退全历史。

本测试用构造数据确保：5年分位 ≠ 21年分位时，模型取的是前者。
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.csi300 import (  # noqa: E402
    MIN_WINDOW_OBS, compute_csi300_score, _window_pct,
)


def _monthly_valuation() -> pd.DataFrame:
    """15年月频 PE/PB：前10年 10~20，近5年 5~8，当前值取近5年高位。

    如此构造使 5年分位（高，~95%）与 15年分位（低，~30%）显著不同——
    旧代码（回退全历史）会得出后者。
    """
    dates = pd.date_range("2011-01-01", periods=180, freq="MS")
    early = np.linspace(10, 20, 120)          # 前10年
    recent = np.linspace(5, 8, 60)            # 近5年，末值=8
    pe = np.concatenate([early, recent])
    return pd.DataFrame({"date": dates, "pe": pe, "pb": pe / 10})


def _daily(end: pd.Timestamp) -> pd.DataFrame:
    dates = pd.bdate_range("2011-01-03", end)
    n = len(dates)
    close = np.linspace(3000, 4500, n)
    return pd.DataFrame({
        "date": dates, "open": close, "close": close,
        "high": close * 1.01, "low": close * 0.99,
        "volume": np.full(n, 1e8),
    })


class TestWindowPct(unittest.TestCase):
    def test_5y_window_not_full_history(self):
        val = _monthly_valuation()
        end = val["date"].iloc[-1]
        p5 = _window_pct(val, "pe", end, years=5)
        pfull = float((val["pe"] < val["pe"].iloc[-1]).mean())
        # 5年窗口：末值 8 在 5..8 中偏高
        self.assertGreater(p5, 0.85)
        # 全历史口径显著更低——若回归到全历史，此断言会失败
        self.assertLess(pfull, 0.45)
        self.assertNotAlmostEqual(p5, pfull, places=2)

    def test_insufficient_window_returns_none(self):
        """窗口观测不足必须返回 None（降级），不得回退全历史。"""
        dates = pd.date_range("2026-01-01", periods=6, freq="MS")  # 仅6个月
        df = pd.DataFrame({"date": dates, "pe": np.arange(6.0)})
        self.assertLess(6, MIN_WINDOW_OBS)
        self.assertIsNone(_window_pct(df, "pe", dates[-1], years=5))


class TestScoreUsesFiveYearWindow(unittest.TestCase):
    def test_valuation_dim_uses_5y_not_21y(self):
        val = _monthly_valuation()
        end = val["date"].iloc[-1]
        daily = _daily(end)
        bond = pd.DataFrame({
            "date": val["date"],
            "yield_10y": np.full(len(val), 2.0),
        })

        r = compute_csi300_score(daily=daily, valuation=val,
                                 bond_yield_df=bond, macro=None)
        pe5 = _window_pct(val, "pe", end, years=5)
        pb5 = _window_pct(val, "pb", end, years=5)
        avg5 = (pe5 + pb5) / 2

        shown = r["details"]["估值温度(30)"]["均值分位"]
        self.assertAlmostEqual(float(shown.rstrip("%")) / 100, avg5, places=2)
        # 5年口径下当前不便宜 → 0 分（旧口径会给 10 分）
        self.assertEqual(r["details"]["估值温度(30)"]["得分"], 0.0)
        self.assertEqual(r["details"]["估值温度(30)"]["等级"], "不便宜")


if __name__ == "__main__":
    unittest.main()
