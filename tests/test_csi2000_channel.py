"""
v7 三通道买入信号测试 — 动态阈值 + 企稳过滤 + 动态仓位

背景（2026-08-30）:
- 固定阈值 50 分在低波年份（2026）凑不到冰点 → v7 动态化
- 通道A 冰点:   绝对分≥50 或 (分数5年分位≤15% 且 r20分位≤15%) + 回撤250日>10%
- 通道B 超跌企稳: r20 5年分位≤10% + 回撤250日>15%
- 全通道要求站回MA10 右侧确认（防阴跌飞刀）
- 动态仓位: 基础40%(A)/25%(B) × 按r20分位乘数(≤5%→×1.4, ≤10%→×1.2)，封顶50%
- 所有阈值均为"过去5年自身历史分位"，结构固定数值自适应

修正验证点（v7.1/v7.2 回测诊断归因）:
  ① 企稳过滤: 2018-06 阴跌中继 B 条件满足但未站回MA10 → 挡
  ② 乘数只加不减: 非超跌月保持基础仓位（2016-06-30 = 40% 而非 16%）
  ⑤ r20≤15% 超跌约束: 2017-12 分数分位 12% 低但 r20 27% 不超跌 → 挡微盘熊飞刀
  右侧确认: 2020-02-05 分数 55 但 V 底左侧未企稳 → 挡（避免左侧买入）

数据全部用生产序列（data/*.parquet 本地文件，与回测/实盘同源）。
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.csi2000 import csi2000_buy_channel  # noqa: E402


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
    add(spread[["date", "cn_2y", "us_2y", "spread_10y"]], ["cn_2y", "us_2y", "spread_10y"])
    add(gold[["date", "gold_price"]], ["gold_price"])
    add(china[["date", "social_finance", "m2_yoy"]], ["social_finance", "m2_yoy"])
    add(fed[["date", "fed_rate"]], ["fed_rate"])
    add(margin[["date", "margin_balance"]], ["margin_balance"])
    return df


class TestChannel(unittest.TestCase):
    """通道判定基准（生产序列实测钉住；类级缓存避免重复打分）。"""

    @classmethod
    def setUpClass(cls):
        cls.idx2000 = pd.read_parquet(ROOT / "data/csi2000_daily.parquet")
        cls.idx2000["date"] = pd.to_datetime(cls.idx2000["date"])
        cls.idx300 = pd.read_parquet(ROOT / "data/csi300_daily.parquet")
        cls.idx300["date"] = pd.to_datetime(cls.idx300["date"])
        cls.macro = _macro_from_local_files()
        cls._cache = {}

    @classmethod
    def channel_on(cls, date_str: str) -> dict:
        if date_str not in cls._cache:
            dt = pd.Timestamp(date_str)
            cls._cache[date_str] = csi2000_buy_channel(
                daily=cls.idx2000[cls.idx2000["date"] <= dt],
                csi300_daily=cls.idx300[cls.idx300["date"] <= dt],
                macro=cls.macro[cls.macro["date"] <= dt],
                latest_date=dt,
            )
        return cls._cache[date_str]

    # ── 2026 大底捕捉（用户核心诉求）──
    def test_2026_08_04_channel_A_50(self):
        """2026-08 V型底: 分数分位13% + r20分位5% + 回撤-22% + 站回MA10
        → 通道A分位视图 + ×1.4 → 40%×1.4=56% 封顶 50%。"""
        r = self.channel_on("2026-08-04")
        self.assertEqual(r["channel"], "A")
        self.assertEqual(r["position_pct"], 0.50)

    def test_2026_08_07_channel_A_40(self):
        """2026-08-07 两融出清加分后绝对分 59≥50 → 通道A（绝对分路径）×1.0。"""
        r = self.channel_on("2026-08-07")
        self.assertEqual(r["channel"], "A")
        self.assertEqual(r["position_pct"], 0.40)

    # ── 动态仓位 ──
    def test_2024_02_23_channel_A_50(self):
        """2024-02 小微盘崩盘底: 绝对分61 + r20分位4%(≤5%→×1.4) → 封顶50%。"""
        r = self.channel_on("2024-02-23")
        self.assertEqual(r["channel"], "A")
        self.assertEqual(r["position_pct"], 0.50)

    def test_2016_06_30_channel_A_base_40(self):
        """2016-06 非超跌月（r20分位57%）→ 乘数×1.0 保持基础40%
        （修复②: 乘数只加不减，不被摊薄到16%）。"""
        r = self.channel_on("2016-06-30")
        self.assertEqual(r["channel"], "A")
        self.assertEqual(r["position_pct"], 0.40)

    # ── 飞刀/中继被挡 ──
    def test_2017_02_15_blocked(self):
        """2017-02 微盘熊: 分数41 + 分位28% + r20 27% → 不触发。"""
        self.assertIsNone(self.channel_on("2017-02-15")["channel"])

    def test_2017_12_15_blocked(self):
        """2017-12 微盘熊: 分数分位12% 虽低但 r20 27% 不超跌 → 分位视图被
        r20≤15% 约束挡（修复⑤）。"""
        r = self.channel_on("2017-12-15")
        self.assertIsNone(r["channel"])

    def test_2018_06_15_blocked(self):
        """2018-06 阴跌中继: r20 10% + 回撤-26.8% 满足B，但未站回MA10
        → 企稳过滤挡（修复①）。"""
        r = self.channel_on("2018-06-15")
        self.assertIsNone(r["channel"])

    def test_2020_02_05_blocked(self):
        """2020-02-05 新冠暴跌左侧: 绝对分55 + 回撤-17.6% 但未站回MA10
        → 右侧确认挡（避免V底左侧买入）。"""
        r = self.channel_on("2020-02-05")
        self.assertIsNone(r["channel"])

    def test_2024_04_15_blocked(self):
        """2024-04 阴跌: 28分 + r20 11% → 不触发。"""
        self.assertIsNone(self.channel_on("2024-04-15")["channel"])

    def test_2022_04_27_blocked(self):
        """2022-04-27 大底当天: r20分位0% + 回撤-29.3% 满足B但当日未站回
        MA10 → 右侧确认挡（企稳后由回测捕捉）。"""
        r = self.channel_on("2022-04-27")
        self.assertIsNone(r["channel"])


if __name__ == "__main__":
    unittest.main()
