"""
卖出端 PE 分位窗口回归测试 — 锁死"5年"口径

背景（2026-09-21）: `macro/sell_signal._pe_percentile` 与买入端同病——
CSI300 估值源为月频（约258行/21.5年），旧实现 `len<252` 判定后用**全量**分位，
把"偏贵"算成"合理"，导致卖出门槛（PE>60%）该开不开（2026-09：21.5年口径
52.7% vs 正确5年口径 75.8%）。修复后必须按**日期**取近5年窗口。
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from macro.sell_signal import PE_MIN_OBS, _pe_percentile  # noqa: E402


def _val_15y() -> pd.DataFrame:
    """前10年 PE 10~20，近5年 PE 5~8，当前 8 → 5年分位高、15年分位低。"""
    dates = pd.date_range("2011-01-01", periods=180, freq="MS")
    pe = np.concatenate([np.linspace(10, 20, 120), np.linspace(5, 8, 60)])
    return pd.DataFrame({"date": dates, "pe": pe})


class TestPePercentileWindow(unittest.TestCase):
    def test_uses_5y_not_full_history(self):
        v = _val_15y()
        end = v["date"].iloc[-1]
        p5 = _pe_percentile(v, end)
        pfull = float((v["pe"] < v["pe"].iloc[-1]).mean())
        self.assertGreater(p5, 0.85)       # 近5年偏上
        self.assertLess(pfull, 0.45)       # 全历史口径偏低
        self.assertNotAlmostEqual(p5, pfull, places=2)

    def test_insufficient_window_returns_none(self):
        dates = pd.date_range("2026-01-01", periods=6, freq="MS")
        v = pd.DataFrame({"date": dates, "pe": np.arange(6.0)})
        self.assertLess(6, PE_MIN_OBS)
        self.assertIsNone(_pe_percentile(v, dates[-1]))

    def test_asof_excludes_future(self):
        """asof 之前的分位不得受未来 PE 影响。"""
        v = _val_15y()
        early = _pe_percentile(v, v["date"].iloc[130])   # 近5年内
        later = _pe_percentile(v, v["date"].iloc[-1])
        self.assertIsNotNone(early)
        self.assertIsNotNone(later)
        self.assertNotAlmostEqual(early, later, places=3)


if __name__ == "__main__":
    unittest.main()
