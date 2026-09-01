"""
机会雷达 v1 测试（2026-09-01 独立模型，不影响 300/2000 核心）

定位:
- 机会 = 因子确认(A) + 行情确认(B) 共振才报；无因子行业降级观察
  （回放实证：B-only 假信号率过高——军工 7/3 信号后 -18.4%、医药 8/10 -7.2%）
- 期货因子是行业"预期温度计"：豆粕 8 月 +10% 领先 159587 行情启动
- 冷却：同标的 60 自然日不重复报

验收基准（真实数据回放）:
- 粮食案例: 2026-08-18 首次报出（A+B），比用户 9/1 发现早 10 个交易日
- 2026-09-01: 农业(159825) + 粮食(159587) 报机会（豆粕+7.7% 开窗）
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.opportunity_radar import (  # noqa: E402
    ETF_POOL, _confirm_signal, _factor_window, radar_scan,
)
from data.fetcher import fetch_etf_daily  # noqa: E402


def _load_etf(code: str) -> pd.DataFrame:
    df = fetch_etf_daily(code, name="t", force=False)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


class TestFactorWindow(unittest.TestCase):
    """条件A 金融因子窗口：近20日涨幅>5% 或 突破60日新高。"""

    def test_margin_since_sep1_window_open(self):
        """2026-09-01 豆粕近20日+7.7% → 窗口开启（粮食案例的因子侧）。"""
        from models.opportunity_radar import _load_futures
        m = _load_futures("M0")
        m = m[m["date"] <= pd.Timestamp("2026-09-01")]
        ok, det = _factor_window(m)
        self.assertTrue(ok)
        self.assertIn("窗口", det)

    def test_flat_factor_window_closed(self):
        """近20日涨幅不足且无突破 → 窗口关闭。"""
        # 构造平盘序列
        dates = pd.date_range("2026-01-01", periods=200, freq="B")
        df = pd.DataFrame({"date": dates, "close": 100.0})
        ok, _ = _factor_window(df)
        self.assertFalse(ok)

    def test_breakout_opens_window(self):
        """收盘突破60日新高 → 窗口开启（即使涨幅不足5%）。"""
        dates = pd.date_range("2026-01-01", periods=200, freq="B")
        vals = [100.0] * 190 + [105.0] * 10
        df = pd.DataFrame({"date": dates, "close": vals})
        ok, det = _factor_window(df)
        self.assertTrue(ok)


class TestConfirmSignal(unittest.TestCase):
    """条件B 行情确认：企稳 + 量能 + 启动区（距250日低点10~35%）。"""

    @classmethod
    def setUpClass(cls):
        cls.agri = _load_etf("159825")

    def test_2026_09_01_agriculture_confirmed(self):
        """2026-09-01 农业: 站回MA10 + 量能1.12 + 距低点+18% → 确认。"""
        d = self.agri[self.agri["date"] <= pd.Timestamp("2026-09-01")]
        ok, det = _confirm_signal(d)
        self.assertTrue(ok)
        self.assertEqual(det["企稳"], "✅站回")

    def test_2026_07_01_agriculture_not_confirmed(self):
        """2026-07-01 农业（下跌中）→ 未确认。"""
        d = self.agri[self.agri["date"] <= pd.Timestamp("2026-07-01")]
        ok, _ = _confirm_signal(d)
        self.assertFalse(ok)


class TestRadarScan(unittest.TestCase):
    """全池扫描行为。"""

    @classmethod
    def setUpClass(cls):
        cls.r = radar_scan()

    def test_agriculture_food_in_opportunities(self):
        """2026-09-01 农业/粮食报机会（豆粕窗口+行情确认共振）。"""
        names = {o["name"] for o in self.r["opportunities"]}
        self.assertIn("农业", names)
        self.assertIn("粮食", names)

    def test_no_factor_industries_not_in_opportunities(self):
        """无因子行业（半导体/医药/军工…）绝不报机会——B-only 假信号降级。"""
        names = {o["name"] for o in self.r["opportunities"]}
        for n in ["半导体", "医药", "军工", "消费", "电力", "白酒", "新能源车"]:
            self.assertNotIn(n, names)

    def test_opportunity_has_ref_buy_zone_and_pct(self):
        """机会输出带参考买区与建议仓位。"""
        o = self.r["opportunities"][0]
        self.assertIn("~", o["ref_buy_zone"])
        self.assertLessEqual(o["pct"], 0.05)

    def test_watching_lists_missing_dims(self):
        """观察列表带缺失维度说明。"""
        watching_names = {w["name"] for w in self.r["watching"]}
        self.assertTrue(watching_names)  # 观察列表非空
        for w in self.r["watching"]:
            self.assertTrue(w["missing"])


if __name__ == "__main__":
    unittest.main()
