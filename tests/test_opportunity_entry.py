"""
机会发现模块测试（models/opportunity_entry.py）

覆盖：R:R / EV / 企稳门槛 / 凯利仓位 / 条件胜率表 / scan 输出。
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.opportunity_entry import (  # noqa: E402
    CAP, KELLY, evaluate, prep, scan, win_table,
)


def _row(close, low20, high120, ma20, dd_bucket="10-15%", regime="risk_off"):
    return pd.Series({
        "date": pd.Timestamp("2026-09-21"), "close": close, "low20": low20,
        "high120": high120, "ma20": ma20, "dd_bucket": dd_bucket, "regime": regime,
        "dd": 0.12,
    })


WT = {("10-15%", "risk_off"): (0.50, 0.50, 100)}


class TestEvaluate(unittest.TestCase):
    def test_eligible_when_ev_positive_rr_ok_and_above_ma20(self):
        e = evaluate(_row(100, 90, 130, 95), WT)
        self.assertAlmostEqual(e["risk"], 10)
        self.assertAlmostEqual(e["reward"], 30)
        self.assertAlmostEqual(e["rr"], 3.0)
        self.assertAlmostEqual(e["ev"], 0.5 * 30 - 0.5 * 10)  # +10
        self.assertTrue(e["eligible"])
        self.assertEqual(e["stop"], 90)
        self.assertEqual(e["target"], 130)

    def test_not_eligible_below_ma20(self):
        """未站回 MA20（未企稳）→ 不博弈（防接飞刀）。"""
        e = evaluate(_row(100, 90, 130, 105), WT)
        self.assertFalse(e["eligible"])

    def test_not_eligible_when_ev_nonpositive(self):
        """EV≤0（胜率太低）→ 不博弈。"""
        wt = {("10-15%", "risk_off"): (0.10, 0.05, 100)}
        e = evaluate(_row(100, 90, 130, 95), wt)
        self.assertLessEqual(e["ev"], 0)
        self.assertFalse(e["eligible"])

    def test_kelly_position_capped_and_halved_when_wilson_weak(self):
        e = evaluate(_row(100, 90, 130, 95), WT)
        raw = KELLY * (0.5 * 3.0 - 0.5) / 3.0
        self.assertAlmostEqual(e["position_pct"], round(raw * 0.5, 4))  # wilson=0.5 → 打对折
        self.assertLessEqual(e["position_pct"], CAP)


class TestWinTable(unittest.TestCase):
    def test_cell_stats(self):
        d = pd.DataFrame({
            "dd_bucket": ["10-15%"] * 4,
            "regime": ["risk_off"] * 4,
            "fwd": [0.1, -0.05, 0.2, 0.3],
        })
        t = win_table(d)
        p, wlb, n = t[("10-15%", "risk_off")]
        self.assertEqual(n, 4)
        self.assertAlmostEqual(p, 0.75)
        self.assertLessEqual(wlb, p)

    def test_overlap_uses_effective_sample(self):
        """v10: H=60 重叠窗口下 Wilson 下界按有效样本 n/H 计，不再过度自信。"""
        from models.opportunity_entry import H, wilson_lb
        n = H * 10
        d = pd.DataFrame({
            "dd_bucket": ["10-15%"] * n, "regime": ["risk_off"] * n,
            "fwd": [0.1] * n,                      # 全胜
        })
        p, wlb, raw_n = win_table(d)[("10-15%", "risk_off")]
        self.assertEqual(p, 1.0)
        self.assertEqual(raw_n, n)                 # 展示仍用原始 n
        self.assertAlmostEqual(wlb, wilson_lb(10, 10), places=6)  # n_eff = n/H = 10
        self.assertLess(wlb, wilson_lb(n, n))      # 较原始口径显著更保守


class TestPrepNoLookahead(unittest.TestCase):
    def test_features_use_past_only(self):
        n = 400
        d = pd.DataFrame({
            "date": pd.bdate_range("2024-01-01", periods=n),
            "open": np.linspace(100, 200, n),
            "close": np.linspace(100, 200, n),
            "high": np.linspace(101, 201, n),
            "low": np.linspace(99, 199, n),
            "volume": np.full(n, 1e6),
        })
        p = prep(d)
        # 截断到第 300 天，第 300 天的特征应与全长一致（不受未来影响）
        p_cut = prep(d.iloc[:300].copy())
        for col in ["ma20", "dd", "low20", "high120", "regime"]:
            a, b = p[col].iloc[299], p_cut[col].iloc[299]
            if isinstance(a, float) or isinstance(a, np.floating):
                self.assertAlmostEqual(float(a), float(b), places=6)
            else:
                self.assertEqual(a, b)


class TestScan(unittest.TestCase):
    def test_scan_returns_targets(self):
        n = 400
        def mk(v):
            return pd.DataFrame({
                "date": pd.bdate_range("2024-01-01", periods=n),
                "open": v, "close": v, "high": v * 1.01, "low": v * 0.99,
                "volume": np.full(n, 1e6),
            })
        a = {"csi300_daily": mk(np.linspace(100, 200, n)),
             "csi2000_daily": mk(np.linspace(100, 150, n))}
        out = scan(a)
        self.assertEqual(len(out), 2)
        for e in out:
            self.assertIn("eligible", e)
            self.assertIn("position_pct", e)
            self.assertIn("stop", e)
            self.assertIn("target", e)


if __name__ == "__main__":
    unittest.main()
