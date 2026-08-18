"""
数据治理测试 — 宁缺毋滥原则（用户铁律）

背景（2026-08-17 实盘复盘）:
- 当晚 bond_zh_us_rate 主源失败，fetch_bond_yield_10y 降级到 bond_china_yield
  （2021年起冻结的旧中债源，3.12%），静默污染 ERP 维度
- 而宏观路径 bond_spread.parquet 里躺着新鲜的真实值（1.6964%）
- 原则: 严重滞后/错误数据一律抛弃，绝不静默使用

修复目标:
1. 国债收益率: 主源(≤7天) → 缓存(≤7天) → bond_spread的cn_10y(≤7天) → None+醒目告警
   停更源 bond_china_yield 永久移除
2. 宏观各序列缓存兜底: 超龄缓存宁缺毋滥，直接报错（各序列按频率设限）
3. bond=None 时模型不崩溃，ERP维度跳过
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data import fetcher as data_fetcher  # noqa: E402
from macro import fetcher as macro_fetcher  # noqa: E402
from strategy import generate_signal  # noqa: E402


def _fresh_bond_df(days_ago: int = 0, value: float = 1.6964) -> pd.DataFrame:
    return pd.DataFrame({
        "date": [pd.Timestamp(datetime.now() - timedelta(days=days_ago)).normalize()],
        "yield_10y": [value],
    })


def _write_parquet(path: Path, df: pd.DataFrame) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


# ═══════════════════════════════════════════════════════════
# 1. 国债收益率: 停更数据绝不使用
# ═══════════════════════════════════════════════════════════

class TestBondYieldNoStaleData(unittest.TestCase):

    def setUp(self):
        # 生产缓存文件如存在，先挪走，测试结束恢复
        self.cache_path = data_fetcher.DATA_DIR / "bond_yield_10y.parquet"
        self._backup = None
        if self.cache_path.exists():
            self._backup = self.cache_path.read_bytes()
            self.cache_path.unlink()

    def tearDown(self):
        if self.cache_path.exists():
            self.cache_path.unlink()
        if self._backup is not None:
            self.cache_path.write_bytes(self._backup)

    def test_all_sources_fail_returns_none(self):
        """主源失败 + 无缓存 + spread不可用 → None，且绝不调用停更源。"""
        with mock.patch("akshare.bond_zh_us_rate", side_effect=RuntimeError("down")), \
             mock.patch("akshare.bond_china_yield", create=True) as frozen_source, \
             mock.patch.object(data_fetcher, "_bond_yield_from_spread", return_value=None):
            result = data_fetcher.fetch_bond_yield_10y()
        self.assertIsNone(result)
        frozen_source.assert_not_called()  # 停更源已被永久移除

    def test_fresh_cache_used_when_primary_fails(self):
        """主源失败 → 使用≤7天的缓存。"""
        _write_parquet(self.cache_path, _fresh_bond_df(days_ago=1, value=1.70))
        with mock.patch("akshare.bond_zh_us_rate", side_effect=RuntimeError("down")), \
             mock.patch.object(data_fetcher, "_bond_yield_from_spread", return_value=None):
            result = data_fetcher.fetch_bond_yield_10y()
        self.assertIsNotNone(result)
        self.assertAlmostEqual(float(result["yield_10y"].iloc[-1]), 1.70)

    def test_stale_cache_discarded(self):
        """缓存超过7天 → 抛弃，返回None（而不是用旧值）。"""
        _write_parquet(self.cache_path, _fresh_bond_df(days_ago=10, value=3.12))
        with mock.patch("akshare.bond_zh_us_rate", side_effect=RuntimeError("down")), \
             mock.patch.object(data_fetcher, "_bond_yield_from_spread", return_value=None):
            result = data_fetcher.fetch_bond_yield_10y()
        self.assertIsNone(result)

    def test_fresh_spread_fallback_used(self):
        """主源失败、缓存陈旧 → bond_spread 新鲜cn_10y可用 → 使用它。"""
        _write_parquet(self.cache_path, _fresh_bond_df(days_ago=10, value=3.12))
        fresh_spread = _fresh_bond_df(days_ago=2, value=1.6964)
        with mock.patch("akshare.bond_zh_us_rate", side_effect=RuntimeError("down")), \
             mock.patch.object(data_fetcher, "_bond_yield_from_spread", return_value=fresh_spread):
            result = data_fetcher.fetch_bond_yield_10y()
        self.assertIsNotNone(result)
        self.assertAlmostEqual(float(result["yield_10y"].iloc[-1]), 1.6964)

    def test_primary_success_saves_cache(self):
        """主源成功 → 返回数据并写入缓存。"""
        fresh = _fresh_bond_df(days_ago=0, value=1.70)
        with mock.patch("akshare.bond_zh_us_rate", return_value=pd.DataFrame({
                "日期": fresh["date"], "中国国债收益率10年": fresh["yield_10y"]})):
            result = data_fetcher.fetch_bond_yield_10y()
        self.assertIsNotNone(result)
        self.assertTrue(self.cache_path.exists())


class TestBondSpreadHelper(unittest.TestCase):

    def test_fresh_spread_parquet_extracted(self):
        """bond_spread.parquet 新鲜 → 提取最后一行 cn_10y。"""
        with tempfile.TemporaryDirectory() as tmp:
            p = _write_parquet(Path(tmp) / "bond_spread.parquet", pd.DataFrame({
                "date": [pd.Timestamp(datetime.now() - timedelta(days=2)).normalize()],
                "cn_10y": [1.6964], "us_10y": [4.2],
            }))
            result = data_fetcher._bond_yield_from_spread(p)
        self.assertIsNotNone(result)
        self.assertEqual(list(result.columns), ["date", "yield_10y"])
        self.assertAlmostEqual(float(result["yield_10y"].iloc[-1]), 1.6964)

    def test_stale_spread_discarded(self):
        """bond_spread 超过7天 → None。"""
        with tempfile.TemporaryDirectory() as tmp:
            p = _write_parquet(Path(tmp) / "bond_spread.parquet", pd.DataFrame({
                "date": [pd.Timestamp(datetime.now() - timedelta(days=30)).normalize()],
                "cn_10y": [1.5], "us_10y": [4.0],
            }))
            result = data_fetcher._bond_yield_from_spread(p)
        self.assertIsNone(result)


# ═══════════════════════════════════════════════════════════
# 2. 宏观缓存兜底: 超龄缓存宁缺毋滥
# ═══════════════════════════════════════════════════════════

class TestCacheAgeGate(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "fx_daily.parquet"

    def _write_with_mtime(self, days_ago: int):
        df = pd.DataFrame({"date": [pd.Timestamp.now().normalize()], "v": [1.0]})
        _write_parquet(self.path, df)
        t = datetime.now() - timedelta(days=days_ago)
        os.utime(self.path, (t.timestamp(), t.timestamp()))

    def test_fresh_cache_loads(self):
        self._write_with_mtime(0)
        df = macro_fetcher._load_any(str(self.path), max_age_days=10)
        self.assertFalse(df.empty)

    def test_stale_cache_raises(self):
        """缓存超过年龄上限 → 宁可报错也不用陈旧数据。"""
        self._write_with_mtime(20)
        with self.assertRaises(RuntimeError):
            macro_fetcher._load_any(str(self.path), max_age_days=10)

    def test_default_gate_is_20_days(self):
        """默认门禁=用户规则: ≤20天可用，>20天直接抛弃。"""
        self._write_with_mtime(20)
        df = macro_fetcher._load_any(str(self.path))  # 恰好20天 → 可用
        self.assertFalse(df.empty)
        self._write_with_mtime(21)
        with self.assertRaises(RuntimeError):
            macro_fetcher._load_any(str(self.path))  # 21天 → 抛弃

    def test_borderline_cache_warns_confidence(self):
        """7~20天: 可用但必须醒目警告置信度降低。"""
        self._write_with_mtime(10)
        with self.assertLogs("macro.fetcher", level="WARNING") as cm:
            macro_fetcher._load_any(str(self.path))
        self.assertTrue(any("置信度降低" in line for line in cm.output))


# ═══════════════════════════════════════════════════════════
# 3. bond=None 时模型安全降级（ERP跳过，不崩溃）
# ═══════════════════════════════════════════════════════════

class TestModelWithoutBond(unittest.TestCase):

    def test_generate_signal_without_bond_no_crash(self):
        """国债收益率缺失 → 信号正常生成，ERP维度跳过。"""
        from tests.test_sell_signal import _macro_from_local_files, _data_dict
        d = _data_dict()
        d["bond_yield_10y"] = None  # 宁缺毋滥的终态
        macro = _macro_from_local_files()

        sig = generate_signal(d, macro, date=pd.Timestamp("2026-08-14"))
        self.assertIn("score_300", sig)
        self.assertGreaterEqual(sig["score_300"], 0)
        self.assertIn(sig["action_300"], ("SELL", "HOLD", "BUY", "RESTORE"))


# ═══════════════════════════════════════════════════════════
# 4. 缓存日历日过期 + _safe_fetch 兜底（2026-08-18 暴露的两个bug）
# ═══════════════════════════════════════════════════════════

class TestCalendarDayCache(unittest.TestCase):
    """1天缓存必须按日历日判断：昨天23点写的缓存，今晚收盘后必须过期。"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def _write(self, days_ago: int):
        """days_ago=1 → mtime设为'昨晚深夜23:30'（复现真实bug: 距今~23h被误判新鲜）。"""
        p = Path(self.tmp.name) / "c.parquet"
        df = pd.DataFrame({"date": [pd.Timestamp.now().normalize()], "v": [1.0]})
        _write_parquet(p, df)
        if days_ago == 1:
            t = (datetime.now() - timedelta(days=1)).replace(hour=23, minute=30)
        else:
            t = datetime.now() - timedelta(days=days_ago)
        os.utime(p, (t.timestamp(), t.timestamp()))
        return str(p)

    def test_data_fetcher_yesterday_mtime_is_stale(self):
        """昨晚深夜写入的缓存 → 今晚必须视为过期（回归: 8/17数据冒充8/18）。"""
        p = self._write(1)
        self.assertIsNone(data_fetcher._load(p))

    def test_data_fetcher_today_mtime_fresh(self):
        self._write(0)
        self.assertIsNotNone(data_fetcher._load(self._write(0)))

    def test_macro_fetcher_yesterday_mtime_is_stale(self):
        p = self._write(1)
        self.assertIsNone(macro_fetcher._load(p))

    def test_macro_fetcher_today_mtime_fresh(self):
        p = self._write(0)
        self.assertIsNotNone(macro_fetcher._load(p))


class TestSafeFetchFallback(unittest.TestCase):
    """_safe_fetch: 抓取失败必须真的兜底到缓存（中文名分发bug回归）。"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def _write(self, days_ago: int = 0):
        p = Path(self.tmp.name) / "bond_spread.parquet"
        df = pd.DataFrame({"date": [pd.Timestamp.now().normalize()], "cn_10y": [1.7]})
        _write_parquet(p, df)
        t = datetime.now() - timedelta(days=days_ago)
        os.utime(p, (t.timestamp(), t.timestamp()))
        return str(p)

    def test_falls_back_to_cache_on_failure(self):
        def boom(force=False):
            raise RuntimeError("源挂了")
        p = self._write(0)
        df = macro_fetcher._safe_fetch(boom, "利差", False, cache_file=p)
        self.assertIsNotNone(df)
        self.assertAlmostEqual(float(df["cn_10y"].iloc[-1]), 1.7)

    def test_stale_cache_raises(self):
        """兜底缓存超过20天 → 宁缺毋滥，报错。"""
        def boom(force=False):
            raise RuntimeError("源挂了")
        p = self._write(25)
        with self.assertRaisesRegex(RuntimeError, "宁缺毋滥"):
            macro_fetcher._safe_fetch(boom, "利差", False, cache_file=p)


if __name__ == "__main__":
    unittest.main()
