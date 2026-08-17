"""
卖出信号 v5 — 分标的趋势兜底 + 回补规则 测试

背景（2026-08-17 实盘样本）:
- 模型曾用沪深300趋势兜底信号一刀切卖出中证2000，8/17开盘减仓50%当日中证2000 +2.4%
- v5 修复: 趋势兜底按持仓标的各自判断（300 用 csi300_daily，2000 用 etf_159531 日线）
- 新增回补规则: 减仓后标的回站MA60且MA60拐头向上 → 回补信号
- 保留全市场估值门槛: CSI300 PE 分位 >60%（PE<60% 永不激活）
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from macro.sell_signal import is_bubble, _check_trend_recovery  # noqa: E402
from strategy import generate_signal  # noqa: E402


# ═══════════════════════════════════════════════════════════
# 工具
# ═══════════════════════════════════════════════════════════

def _macro_from_local_files() -> pd.DataFrame:
    """重建 fetch_all_macro 的合并结果（与生产一致，merge_asof 日频对齐）。"""
    fx = pd.read_parquet(ROOT / "data/fx_daily.parquet")
    spread = pd.read_parquet(ROOT / "data/bond_spread.parquet")
    gold = pd.read_parquet(ROOT / "data/gold_daily.parquet")
    china = pd.read_parquet(ROOT / "data/china_macro_monthly.parquet")
    fed = pd.read_parquet(ROOT / "data/fed_rate.parquet")
    margin = pd.read_parquet(ROOT / "data/margin_balance.parquet")

    df = fx[["date"]].copy()
    df["date"] = pd.to_datetime(df["date"]).astype("datetime64[ns]")

    def add(right, cols):
        r = right[["date"] + cols].dropna(subset=cols).sort_values("date").copy()
        r["date"] = pd.to_datetime(r["date"]).astype("datetime64[ns]")
        m = pd.merge_asof(df, r, on="date", direction="backward")
        for c in cols:
            df[c] = m[c].values

    add(fx[["date", "usd_cny", "jpy_cny"]], ["usd_cny", "jpy_cny"])
    add(spread[["date", "spread_10y"]], ["spread_10y"])
    add(gold[["date", "gold_price"]], ["gold_price"])
    add(china[["date", "social_finance", "m2_yoy"]], ["social_finance", "m2_yoy"])
    add(fed[["date", "fed_rate"]], ["fed_rate"])
    add(margin[["date", "margin_balance"]], ["margin_balance"])
    return df


def _data_dict() -> dict:
    """构建 generate_signal 需要的数据字典（全部来自本地缓存）。"""
    d = {
        "csi300_daily": pd.read_parquet(ROOT / "data/csi300_daily.parquet"),
        "csi2000_daily": pd.read_parquet(ROOT / "data/csi2000_daily.parquet"),
        "etf_159531": pd.read_parquet(ROOT / "data/etf_159531_daily.parquet"),
        "etf_159330": pd.read_parquet(ROOT / "data/etf_159330_daily.parquet"),
        "csi300_valuation": pd.read_parquet(ROOT / "data/valuation_000300.parquet"),
        "csi2000_valuation": pd.read_parquet(ROOT / "data/valuation_932000.parquet"),
    }
    bs = pd.read_parquet(ROOT / "data/bond_spread.parquet")[["date", "cn_10y"]]
    d["bond_yield_10y"] = bs.rename(columns={"cn_10y": "yield_10y"}).dropna()
    return d


def _idx_for(macro_df: pd.DataFrame, date) -> int:
    return int(macro_df["date"].searchsorted(pd.Timestamp(date), side="right")) - 1


def _synth_daily(n1: int = 200, n2: int = 120,
                 start: float = 100.0, bottom: float = 50.0,
                 end: float = 130.0) -> pd.DataFrame:
    """合成日线: n1 天单边下跌(形成趋势破坏) + n2 天单边上涨(修复)。"""
    dates = pd.bdate_range("2025-01-01", periods=n1 + n2)
    close = np.concatenate([np.linspace(start, bottom, n1), np.linspace(bottom, end, n2)])
    return pd.DataFrame({
        "date": dates, "open": close, "close": close,
        "high": close * 1.001, "low": close * 0.999, "volume": 1e6,
    })


def _synth_macro(daily: pd.DataFrame) -> pd.DataFrame:
    """仅日期列 + 空宏观因子（C类全部跳过）。"""
    return pd.DataFrame({
        "date": daily["date"].values,
        "spread_10y": np.nan, "usd_cny": np.nan, "gold_price": np.nan,
        "social_finance": np.nan, "m2_yoy": np.nan, "fed_rate": np.nan,
        "margin_balance": np.nan,
    })


def _synth_valuation_high_pe(daily: pd.DataFrame) -> pd.DataFrame:
    """300行估值序列，最新PE远高于历史 → 分位100% > 60%门槛。"""
    dates = pd.bdate_range("2024-01-01", periods=300)
    pe = np.full(300, 10.0)
    pe[-1] = 15.0
    return pd.DataFrame({"date": dates, "pe": pe, "pb": np.full(300, 1.2)})


# ═══════════════════════════════════════════════════════════
# 1. 核心回归: 2026-08-14 实盘场景
# ═══════════════════════════════════════════════════════════

class TestPerAssetTrendBreakdown(unittest.TestCase):
    """8/14: 300腿触发减仓、2000腿不触发（8/12已回站自身MA60）。"""

    @classmethod
    def setUpClass(cls):
        cls.macro = _macro_from_local_files()
        cls.val300 = pd.read_parquet(ROOT / "data/valuation_000300.parquet")
        cls.csi300 = pd.read_parquet(ROOT / "data/csi300_daily.parquet")
        cls.etf531 = pd.read_parquet(ROOT / "data/etf_159531_daily.parquet")

    def _res(self):
        return is_bubble(self.macro, self.val300, idx=_idx_for(self.macro, "2026-08-14"),
                         daily_300=self.csi300, daily_2000=self.etf531)

    def test_leg300_sells(self):
        res = self._res()
        self.assertTrue(res["leg_300"]["is_bubble"])
        self.assertEqual(res["leg_300"]["signal_type"], "trend_breakdown")
        self.assertEqual(res["leg_300"]["sell_pct"], 0.5)

    def test_leg2000_does_not_sell(self):
        res = self._res()
        self.assertFalse(res["leg_2000"]["is_bubble"])
        self.assertIsNone(res["leg_2000"]["signal_type"])

    def test_leg2000_recovery_not_yet(self):
        """8/14: ETF已回站MA60上方(+0.24%)但MA60仍拐头向下 → 回补未触发。"""
        res = self._res()
        self.assertFalse(res["leg_2000"]["recovery"])

    def test_generate_signal_per_asset(self):
        """8/14 用户可见输出: 300=SELL, 2000=HOLD（不再被一刀切）。"""
        sig = generate_signal(_data_dict(), self.macro, date=pd.Timestamp("2026-08-14"))
        self.assertEqual(sig["action_300"], "SELL")
        self.assertEqual(sig["action_2000"], "HOLD")


# ═══════════════════════════════════════════════════════════
# 2. 回补规则
# ═══════════════════════════════════════════════════════════

class TestTrendRecovery(unittest.TestCase):

    def test_recovery_after_v_repair(self):
        """V型: 先趋势破坏后修复 → 回补信号触发。"""
        daily = _synth_daily()
        macro = _synth_macro(daily)
        val = _synth_valuation_high_pe(daily)
        res = is_bubble(macro, val, idx=len(macro) - 1,
                        daily_300=daily, daily_2000=daily)
        leg = res["leg_2000"]
        self.assertTrue(leg["recovery"])
        self.assertEqual(leg["signal_type"], "trend_recovery")
        self.assertFalse(leg["is_bubble"])

    def test_no_recovery_while_below_ma60(self):
        """单边下跌末端: 仍在MA60下方 → 回补不触发。"""
        daily = _synth_daily(n1=200, n2=0)
        self.assertFalse(_check_trend_recovery(daily, daily["date"].iloc[-1]))

    def test_no_recovery_without_recent_breakdown(self):
        """长期健康上行(无近期趋势破坏) → 不是'回补'，只是普通趋势。"""
        dates = pd.bdate_range("2024-01-01", periods=320)
        close = np.linspace(100, 200, 320)
        daily = pd.DataFrame({"date": dates, "open": close, "close": close,
                              "high": close, "low": close, "volume": 1e6})
        self.assertFalse(_check_trend_recovery(daily, daily["date"].iloc[-1]))


# ═══════════════════════════════════════════════════════════
# 3. 估值门槛（全市场背景条件保留）
# ═══════════════════════════════════════════════════════════

class TestPeGate(unittest.TestCase):

    def test_no_sell_when_pe_below_60pct(self):
        """PE分位<60%: 即使两个标的自趋势都破坏，也不触发减仓。"""
        daily = _synth_daily(n1=200, n2=0)  # 末端单边下跌
        macro = _synth_macro(daily)
        dates = pd.bdate_range("2024-01-01", periods=300)
        pe = np.full(300, 10.0)
        pe[-1] = 5.0  # 最新PE低于全部历史 → 分位0%
        val = pd.DataFrame({"date": dates, "pe": pe, "pb": np.full(300, 1.2)})
        res = is_bubble(macro, val, idx=len(macro) - 1,
                        daily_300=daily, daily_2000=daily)
        for key in ("leg_300", "leg_2000"):
            self.assertFalse(res[key]["is_bubble"])
            self.assertEqual(res[key]["sell_pct"], 0.0)


# ═══════════════════════════════════════════════════════════
# 4. 向后兼容（回测依赖）
# ═══════════════════════════════════════════════════════════

class TestBackwardCompat(unittest.TestCase):

    def test_without_daily_2000_top_level_unchanged(self):
        """不传 daily_2000 时输出与旧版一致（含'无leg键'）。"""
        macro = _macro_from_local_files()
        val300 = pd.read_parquet(ROOT / "data/valuation_000300.parquet")
        csi300 = pd.read_parquet(ROOT / "data/csi300_daily.parquet")
        etf531 = pd.read_parquet(ROOT / "data/etf_159531_daily.parquet")
        idx = _idx_for(macro, "2026-08-14")

        old = is_bubble(macro, val300, idx=idx, daily_300=csi300)
        new = is_bubble(macro, val300, idx=idx, daily_300=csi300, daily_2000=etf531)

        self.assertNotIn("leg_300", old)
        for k in ("is_bubble", "signal_type", "level", "sell_pct", "reasons", "pe_pct"):
            self.assertEqual(new[k], old[k], f"top-level字段 {k} 与旧版不一致")


# ═══════════════════════════════════════════════════════════
# 5. generate_signal 动作映射
# ═══════════════════════════════════════════════════════════

class TestGenerateSignalMapping(unittest.TestCase):

    def test_restore_action_mapping(self):
        """leg_2000 recovery=True → action_2000='RESTORE'。"""
        fake_bubble = {
            "is_bubble": False, "signal_type": None, "level": "", "sell_pct": 0.0,
            "reasons": [], "signals": {}, "pe_pct": 0.65, "recovery": False,
            "leg_300": {
                "is_bubble": False, "signal_type": None, "level": "PE偏高但仅0/4类触发",
                "sell_pct": 0.0, "reasons": [], "signals": {}, "pe_pct": 0.65,
                "recovery": False,
            },
            "leg_2000": {
                "is_bubble": False, "signal_type": "trend_recovery",
                "level": "趋势修复·回补", "sell_pct": 0.0,
                "reasons": ["回站MA60且MA60拐头向上(趋势修复)"],
                "signals": {}, "pe_pct": 0.65, "recovery": True,
            },
        }
        macro = _macro_from_local_files()
        data = _data_dict()
        with mock.patch("strategy.detect_bubble", return_value=fake_bubble):
            sig = generate_signal(data, macro, date=pd.Timestamp("2026-08-14"))
        self.assertEqual(sig["action_2000"], "RESTORE")
        self.assertEqual(sig["action_300"], "HOLD")
        self.assertIn("回补", sig["position_advice"])


if __name__ == "__main__":
    unittest.main()
