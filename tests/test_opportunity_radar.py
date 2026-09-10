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

import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.opportunity_radar import (  # noqa: E402
    ATR_MULT, ETF_POOL, MAX_HOLD_BARS, SATELLITE_SINGLE_PCT,
    SATELLITE_TOTAL_PCT, STOP_FLOOR, TP_MULT,
    _confirm_signal, _factor_window, _risk_lines, check_exits, close_position,
    radar_scan, satellite_room,
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


class _IsolatedRadarTest(unittest.TestCase):
    """radar_scan 会读写状态文件——测试必须用临时 state_path，
    否则会污染生产 data/radar_state.json（并让测试之间互相干扰冷却状态）。"""

    @classmethod
    def setUpClass(cls):
        cls._td = tempfile.TemporaryDirectory()
        cls._state = Path(cls._td.name) / "state.json"

    @classmethod
    def tearDownClass(cls):
        cls._td.cleanup()

    @classmethod
    def _scan(cls, **kw):
        return radar_scan(state_path=cls._state, **kw)


class TestRadarScan(_IsolatedRadarTest):
    """全池扫描行为。

    日期固定为 2026-09-01（验收基准日）。此前不传 date → 断言的是**当天**行情，
    随着市场变化会自然失效（9/10 农业已转为"行情未确认"）——那是测试脆，
    不是模型回归。固定日期后这里才是真正的回归测试。
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.r = cls._scan(date="2026-09-01")

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

    def test_opportunity_has_ref_buy_zone(self):
        """机会输出带参考买区。"""
        o = self.r["opportunities"][0]
        self.assertIn("~", o["ref_buy_zone"])

    def test_watching_lists_missing_dims(self):
        """观察列表带缺失维度说明。"""
        watching_names = {w["name"] for w in self.r["watching"]}
        self.assertTrue(watching_names)  # 观察列表非空
        for w in self.r["watching"]:
            self.assertTrue(w["missing"])


class TestSecuritiesDowngrade(_IsolatedRadarTest):
    """证券(512880)从因子池降级为观察——实证 n=5 胜率 0%，机理不成立。

    两融余额是月度慢变量（证金公司口径），用"20日涨幅>5%或破60日新高"的
    期货动量模板去套它，逻辑上不成立：期货是日频预期定价，两融是月度存量。
    """

    def test_securities_factors_is_none(self):
        sec = next(i for i in ETF_POOL if i["code"] == "512880")
        self.assertIsNone(sec["factors"])

    def test_securities_never_in_opportunities(self):
        self.assertNotIn("证券", {o["name"] for o in self._scan()["opportunities"]})

    def test_securities_appears_in_watching(self):
        self.assertIn("证券", {w["name"] for w in self._scan()["watching"]})


class TestPositionSizing(_IsolatedRadarTest):
    """单标仓位 = 1/4 Kelly 区间中点（依据见模块 docstring）。日期固定保证可复现。"""

    def test_single_pct(self):
        for o in self._scan(date="2026-09-01")["opportunities"]:
            self.assertAlmostEqual(o["pct"], SATELLITE_SINGLE_PCT, places=6)

    def test_opportunity_carries_risk_lines(self):
        """机会必须带止损/止盈价，否则无法执行。"""
        o = self._scan(date="2026-09-01")["opportunities"][0]
        for k in ("stop", "tp", "stop_pct", "tp_pct"):
            self.assertIn(k, o)
        self.assertLess(o["stop"], o["price"])
        self.assertGreater(o["tp"], o["price"])


class TestRiskLines(unittest.TestCase):
    """止损 = max(2.5×ATR20, 5%)；止盈 = 2×止损（波动率自适应）。"""

    def test_low_vol_floors_at_5pct(self):
        """低波(ATR 1.8%)：2.5×1.8%=4.5% < 5% → 取 5% 下限（=已验证口径）。"""
        rl = _risk_lines(1.000, 0.018)
        self.assertAlmostEqual(rl["stop_pct"], STOP_FLOOR, places=6)
        self.assertAlmostEqual(rl["stop"], 0.950, places=3)

    def test_high_vol_widens_stop(self):
        """高波(ATR 4%)：2.5×4%=10% → 止损放宽到 10%（同理止盈 20%）。"""
        rl = _risk_lines(1.000, 0.040)
        self.assertAlmostEqual(rl["stop_pct"], ATR_MULT * 0.040, places=6)
        self.assertAlmostEqual(rl["stop"], 0.900, places=3)

    def test_take_profit_is_2x_stop(self):
        rl = _risk_lines(1.000, 0.040)
        self.assertAlmostEqual(rl["tp_pct"], ATR_MULT * 0.040 * TP_MULT, places=6)
        self.assertAlmostEqual(rl["tp"], 1.200, places=3)

    def test_floor_never_tightens_below_tested(self):
        """下限保证：任何波动率下止损都不比已验证的 5% 更紧。"""
        for atr in [0.005, 0.01, 0.018, 0.02, 0.04, 0.09]:
            self.assertGreaterEqual(_risk_lines(1.0, atr)["stop_pct"], STOP_FLOOR)


class TestExits(unittest.TestCase):
    """退出规则：止损 / 止盈 / 最长持有 60 交易日（与回测同口径，收盘价触发）。"""

    def _pos(self, **kw):
        p = {"name": "农业", "code": "159825", "entry": 1.000,
             "entry_date": "2026-09-01", "pct": SATELLITE_SINGLE_PCT,
             "stop": 0.950, "tp": 1.100, "bars_held": 0}
        p.update(kw)
        return p

    def test_stop_loss(self):
        still, exits = check_exits([self._pos()], {"159825": {"close": 0.949}})
        self.assertEqual(len(exits), 1)
        self.assertEqual(exits[0]["reason"], "止损")
        self.assertEqual(still, [])

    def test_take_profit(self):
        still, exits = check_exits([self._pos()], {"159825": {"close": 1.101}})
        self.assertEqual(exits[0]["reason"], "止盈")

    def test_stop_at_exact_line_triggers(self):
        """触及即出（≤/≥），不留模糊地带。"""
        _, exits = check_exits([self._pos()], {"159825": {"close": 0.950}})
        self.assertEqual(exits[0]["reason"], "止损")

    def test_time_exit_at_60_bars(self):
        still, exits = check_exits([self._pos(bars_held=59)], {"159825": {"close": 1.000}})
        self.assertEqual(exits[0]["reason"], "到期")

    def test_holds_within_band(self):
        still, exits = check_exits([self._pos()], {"159825": {"close": 1.000}})
        self.assertEqual(exits, [])
        self.assertEqual(len(still), 1)
        self.assertEqual(still[0]["bars_held"], 1)

    def test_missing_quote_keeps_position(self):
        """数据缺失时不得误判为退出。"""
        still, exits = check_exits([self._pos()], {})
        self.assertEqual(exits, [])
        self.assertEqual(len(still), 1)

    def test_priority_stop_over_time(self):
        """止损优先于到期（同一天两条都满足时按止损记）。"""
        _, exits = check_exits([self._pos(bars_held=59)], {"159825": {"close": 0.900}})
        self.assertEqual(exits[0]["reason"], "止损")


class TestSatelliteCap(unittest.TestCase):
    """卫星资金划分：单标 ≤7.5%，总计 ≤15%。"""

    def test_room_empty(self):
        self.assertAlmostEqual(satellite_room([]), SATELLITE_TOTAL_PCT, places=6)

    def test_room_shrinks(self):
        self.assertAlmostEqual(satellite_room([{"pct": 0.075}]), 0.075, places=6)

    def test_room_exhausted(self):
        self.assertAlmostEqual(
            satellite_room([{"pct": 0.075}, {"pct": 0.075}]), 0.0, places=6)

    def test_room_never_negative(self):
        self.assertAlmostEqual(
            satellite_room([{"pct": 0.075}] * 5), 0.0, places=6)


class TestCooldownPersistence(unittest.TestCase):
    """冷却必须跨扫描持久化——否则 last_signal 每次重置，60日冷却形同虚设。"""

    def test_cooldown_survives_across_scans(self):
        """冷却必须落盘——否则 last_signal 每次重置，60日冷却形同虚设。

        跑法：首扫建仓 → 平仓 → 原样再扫。此时无持仓可展示，若冷却没持久化，
        第二次会把同一标的当新信号再报一次（这正是 v1 的 bug）。
        """
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "state.json"
            first = radar_scan(state_path=p)
            self.assertTrue(first["opportunities"], "首扫应有信号（前置条件）")
            for c in [x["code"] for x in first["positions"]]:
                close_position(c, state_path=p)

            second = radar_scan(state_path=p)
            self.assertEqual(
                [o["name"] for o in second["opportunities"]], [],
                "60自然日冷却内不得重复报同一标的")

    def test_state_file_written(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "state.json"
            radar_scan(state_path=p)
            self.assertTrue(p.exists())
            self.assertIn("last_signal", json.loads(p.read_text()))

    def test_returns_open_positions(self):
        """扫描必须回传当前持仓——否则用户看不到止损位在哪，跟踪形同虚设。"""
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "state.json"
            first = radar_scan(state_path=p)
            self.assertTrue(first["opportunities"], "首扫应有信号（前置条件）")
            self.assertTrue(first["positions"], "开出信号仓后应能看到持仓")
            pos = first["positions"][0]
            self.assertIn("stop", pos)
            self.assertIn("tp", pos)

    def test_same_day_rescan_not_suppressed_and_not_duplicated(self):
        """同一天重复扫描：仍应显示机会，但不得重复建仓。

        （60日冷却是防"连续多日刷屏"，不是防"同一天再跑一次"。）
        """
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "state.json"
            first = radar_scan(state_path=p)
            second = radar_scan(state_path=p)
            self.assertEqual([o["name"] for o in second["opportunities"]],
                             [o["name"] for o in first["opportunities"]],
                             "同一天重扫应仍显示同一机会")
            self.assertEqual(len(second["positions"]), len(first["positions"]),
                             "同一天重扫不得重复建仓")

    def test_next_day_still_in_cooldown(self):
        """次日（仍在60自然日内）→ 冷却生效，不再把已报过的标的当**新信号**报。"""
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "state.json"
            first = radar_scan(state_path=p)
            self.assertTrue(first["opportunities"], "首扫应有信号（前置条件）")
            names = [o["name"] for o in first["opportunities"]]
            for c in [x["code"] for x in first["positions"]]:
                close_position(c, state_path=p)   # 先平仓，隔离"持仓展示"与"冷却"两件事
            st = json.loads(p.read_text())
            st["last_signal"] = {k: "2026-09-01" for k in st["last_signal"]}  # 模拟次日
            p.write_text(json.dumps(st), encoding="utf-8")
            later = radar_scan(state_path=p)
            for n in names:
                self.assertNotIn(n, [o["name"] for o in later["opportunities"]])

    def test_close_position_reconciles_without_buying(self):
        """用户没实际买入 / 已自行卖出 → 手动平掉雷达持仓，避免幽灵仓位。"""
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "state.json"
            first = radar_scan(state_path=p)
            code = first["positions"][0]["code"]
            closed = close_position(code, state_path=p)
            self.assertEqual(closed["code"], code)
            after = radar_scan(state_path=p)
            self.assertEqual([x["code"] for x in after["positions"]],
                             [x["code"] for x in first["positions"] if x["code"] != code])

    def test_close_unknown_position_returns_none(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "state.json"
            self.assertIsNone(close_position("000000", state_path=p))

    def test_replay_does_not_touch_state(self):
        """历史回放(date=...)不得读写状态文件。

        否则会用历史价格去判定**当前持仓**的止损止盈，并把 bars_held 算乱。
        """
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "state.json"
            r = radar_scan(date="2026-09-01", state_path=p)
            self.assertTrue(r["opportunities"], "回放本身应正常出信号")
            self.assertFalse(p.exists(), "回放不得写入状态文件")

    def test_replay_ignores_existing_positions(self):
        """回放不得用历史价判定当前持仓。"""
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "state.json"
            p.write_text(json.dumps({
                "last_signal": {},
                "positions": [{"name": "农业", "code": "159825", "entry": 0.5,
                               "entry_date": "2026-09-01", "pct": 0.075,
                               "stop": 0.4, "tp": 0.6, "bars_held": 0}],
            }), encoding="utf-8")
            before = p.read_text()
            r = radar_scan(date="2026-09-01", state_path=p)
            self.assertEqual(r["exits"], [], "回放不得对当前持仓产生退出判定")
            self.assertEqual(p.read_text(), before, "回放不得改写状态文件")

    def test_exited_position_removed_from_state(self):
        """触发退出的持仓不再留在状态文件里（同日新开的信号仓不受影响）。"""
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "state.json"
            p.write_text(json.dumps({
                "last_signal": {},
                "positions": [{"name": "农业", "code": "159825", "entry": 999.0,
                               "entry_date": "2026-09-01", "pct": 0.075,
                               "stop": 1.0, "tp": 1.1, "bars_held": 0}],
            }), encoding="utf-8")
            r = radar_scan(state_path=p)
            self.assertIn("159825", [x["code"] for x in r["exits"]])
            state = json.loads(p.read_text())
            self.assertNotIn("159825", [x["code"] for x in state["positions"]])


if __name__ == "__main__":
    unittest.main()
