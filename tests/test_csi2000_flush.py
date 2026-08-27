"""
两融出清确认加分 (+13) 测试 — v6 新增维度

背景（2026-08-28）:
- 模型盲区: V型急跌底在 3~5 天完成时冰点分数凑不到 50（2024-02/2026-07 错过）
- 加分: 两融距6月高回撤>10%（杠杆快速出清）+ 站回MA20 + 5日均量>20日均量
  → 确认 V 型底，+13 分（38+13=51，留 3 分余量）
- 冷却: 最近 60 自然日内已触发过 → 不加分（同一出清事件不重复加分）
- 数据全部用生产序列（csi2000_daily.parquet index 全量到 2026-08-27）

基准（生产序列实测）:
  2024-02-23 = 57（44+13 新增买入 ✅）
  2026-08-07 = 51（38+13 新增买入 ✅，V 底捕捉——比预研 full 序列的 8/11 早2日）
  2026-08-04 = 28（未触发：未站回MA20/量能未回升）
  2022-07-01 = 47（冷却挡：6/1 已触发）
  2026-08-25 = 35（冷却：8/7 已触发 → 当前不亮，避免追高）
  2016-07-01 = 45（冷却挡：6/1 已触发）

已知代价（2026-08-28 验收，逐日扫描生产逻辑）:
  全历史实际加分日仅 6 个；其中 2015-07-21（49.5→62.5）是唯一加分新增
  飞刀（120日后 -14.9%，2015 杠杆崩盘第一轮出清反弹）。决策接受:
  ① 回测实证零影响（触发日全部落在月扫的 60 天冷却窗口内）② 无结构
  性区分条件（两融水位/dd深度均无法与目标底区分）③ 用户反过拟合原则。
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.csi2000 import compute_csi2000_score, _score_margin_flush  # noqa: E402


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


def _score_on(date_str: str) -> dict:
    """生产序列（index 全量）+ 宏观，计算指定日期 CSI2000 总分。"""
    idx2000 = pd.read_parquet(ROOT / "data/csi2000_daily.parquet")
    idx2000["date"] = pd.to_datetime(idx2000["date"])
    idx300 = pd.read_parquet(ROOT / "data/csi300_daily.parquet")
    idx300["date"] = pd.to_datetime(idx300["date"])
    dt = pd.Timestamp(date_str)
    macro = _macro_from_local_files()
    return compute_csi2000_score(
        daily=idx2000[idx2000["date"] <= dt],
        valuation=None,
        csi300_daily=idx300[idx300["date"] <= dt],
        macro=macro[macro["date"] <= dt],
    )


def _flush_detail(res: dict) -> dict:
    return res["details"].get("两融出清确认(+13)", {})


class TestFlushBaselines(unittest.TestCase):
    """4 条核心基准 + 冷却挡（生产序列实测钉住）。"""

    def test_2024_02_23_triggers(self):
        """2024-02 小微盘崩盘底: 44+13=57 → 冰点新增买入。"""
        res = _score_on("2024-02-23")
        self.assertGreaterEqual(res["total_score"], 50)
        self.assertIn("✅", _flush_detail(res).get("触发", ""))

    def test_2026_08_07_triggers(self):
        """2026-08 V 型底: 38+13=51 → 冰点捕捉（生产序列比预研早2日）。"""
        res = _score_on("2026-08-07")
        self.assertGreaterEqual(res["total_score"], 50)
        self.assertIn("✅", _flush_detail(res).get("触发", ""))

    def test_2026_08_04_not_triggered(self):
        """8/4 确认未完成（未站回MA20/量能未回升）→ 不加分不亮灯。"""
        res = _score_on("2026-08-04")
        self.assertLess(res["total_score"], 50)
        self.assertIn("未触发", _flush_detail(res).get("触发", ""))

    def test_2022_07_01_cooldown_blocks(self):
        """2022-07-01 月度冰点被冷却挡: 6/1 已触发 → 47 分不亮（无新增买入）。"""
        res = _score_on("2022-07-01")
        self.assertLess(res["total_score"], 50)
        self.assertIn("冷却", _flush_detail(res).get("触发", ""))

    def test_2016_07_01_cooldown_blocks(self):
        """2016-07-01 同样被冷却挡（距上次触发30天<60）→ 45 分不亮。"""
        res = _score_on("2016-07-01")
        self.assertLess(res["total_score"], 50)
        self.assertIn("冷却", _flush_detail(res).get("触发", ""))

    def test_current_2026_08_25_no_bonus(self):
        """当前 8/25: 冷却期内 → 35 分不亮（8/7 已加分，避免追高）。"""
        res = _score_on("2026-08-25")
        self.assertLess(res["total_score"], 50)
        self.assertIn("冷却", _flush_detail(res).get("触发", ""))


class TestFlushGuards(unittest.TestCase):
    """数据防御: 数据不足 / volume缺失 / margin滞后。"""

    def test_daily_too_short(self):
        """日线<80行 → 0分"数据不足"，不崩溃。"""
        daily = pd.DataFrame({
            "date": pd.bdate_range("2026-01-01", periods=50),
            "open": 1.0, "close": 1.0, "high": 1.0, "low": 1.0, "volume": 1e6,
        })
        macro = _macro_from_local_files()
        s, detail = _score_margin_flush(daily, macro, pd.Timestamp("2026-08-25"))
        self.assertEqual(s, 0.0)
        self.assertIn("数据不足", detail["状态"])

    def test_volume_missing_skips_cond3(self):
        """volume 列缺失 → 条件3对触发与冷却一并跳过，系统不崩溃不加分。

        实测: 无 volume 时 2024-02-23/2026-08-07 均不触发（7/31 等更早的
        "价格+两融"宽松条件触发 → 冷却）。保守且自洽，防御有效。
        """
        idx2000 = pd.read_parquet(ROOT / "data/csi2000_daily.parquet")
        idx2000["date"] = pd.to_datetime(idx2000["date"])
        no_vol = idx2000.drop(columns=["volume"])
        macro = _macro_from_local_files()
        dt = pd.Timestamp("2026-08-07")
        s, detail = _score_margin_flush(no_vol[no_vol["date"] <= dt], macro, dt)
        self.assertEqual(s, 0.0)
        self.assertIn(detail["触发"], ("冷却期内(60自然日内已触发过)", "未触发"))

    def test_stale_margin_skipped(self):
        """两融数据滞后>7天（缓存冻结场景）→ 跳过不加分。"""
        idx2000 = pd.read_parquet(ROOT / "data/csi2000_daily.parquet")
        idx2000["date"] = pd.to_datetime(idx2000["date"])
        macro = _macro_from_local_files().copy()
        # 冻结: 8/1 之后的两融全部置空 → 对 8/11 信号滞后10天
        macro.loc[macro["date"] > "2026-08-01", "margin_balance"] = np.nan
        dt = pd.Timestamp("2026-08-11")
        s, detail = _score_margin_flush(idx2000[idx2000["date"] <= dt], macro, dt)
        self.assertEqual(s, 0.0)
        self.assertIn("滞后", detail["状态"])

    def test_no_margin_data(self):
        """宏观无两融列 → 0分"数据不足"。"""
        idx2000 = pd.read_parquet(ROOT / "data/csi2000_daily.parquet")
        idx2000["date"] = pd.to_datetime(idx2000["date"])
        macro = _macro_from_local_files().drop(columns=["margin_balance"])
        s, detail = _score_margin_flush(idx2000, macro, pd.Timestamp("2026-08-07"))
        self.assertEqual(s, 0.0)
        self.assertIn("无两融", detail["状态"])


if __name__ == "__main__":
    unittest.main()
