"""
回测引擎 v4 — 简化版

投资哲学：ETF 降噪 → 左侧找好买点 → 买入后长期持有 → 只在极端情况卖出
- 买入：分数越高，仓位越重
- 持有：默认不动
- 卖出：仅两种情况 — 翻倍止盈 / 熊市确认
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from models.csi300 import compute_csi300_score
from models.csi2000 import compute_csi2000_score

logger = logging.getLogger(__name__)

TOTAL_CAPITAL = 100_000
COMMISSION_RATE = 0.00015
MIN_COMMISSION = 5.0

ETF_CONFIG = {
    "159330": {"name": "沪深300ETF", "max_capital": TOTAL_CAPITAL * 0.60},
    "159531": {"name": "中证2000ETF", "max_capital": TOTAL_CAPITAL * 0.40},
}


@dataclass
class BacktestParams:
    """可调参数（精简版）。"""
    core_pct: float = 0.50           # 底仓比例
    buy_score_enter: float = 50      # 超过此分数开始加仓
    profit_double_sell: bool = True  # 翻倍止盈
    bear_market_exit: bool = True    # 熊市确认后减仓

    # 固定参数（不扫描）
    buy_score_max: float = 80        # 满分加仓对应的分数
    bear_ma: int = 200               # 熊市判断均线


@dataclass
class Trade:
    date: datetime
    etf: str
    action: str
    shares: int
    price: float
    amount: float
    commission: float
    score: float
    reason: str


class Position:
    def __init__(self, etf: str):
        self.etf = etf
        self.shares = 0
        self.avg_cost = 0.0
        self.total_cost = 0.0
        self.max_profit_pct = 0.0     # 历史最高浮盈
        self.bear_weeks = 0            # 连续熊市周数

    def buy(self, shares, price, commission):
        cost = shares * price
        self.total_cost += cost + commission
        self.shares += shares
        self.avg_cost = self.total_cost / self.shares if self.shares > 0 else 0

    def sell(self, shares, price, commission):
        actual = min(shares, self.shares)
        if actual == 0:
            return 0
        proceeds = actual * price - commission
        cost_relieved = actual * self.avg_cost
        self.shares -= actual
        self.total_cost -= cost_relieved
        if self.shares == 0:
            self.avg_cost = 0
            self.total_cost = 0
            self.max_profit_pct = 0
            self.bear_weeks = 0
        return proceeds

    def profit_pct(self, price):
        if self.shares == 0 or self.avg_cost == 0:
            return 0
        return (price - self.avg_cost) / self.avg_cost


@dataclass
class WeekSnapshot:
    date: datetime
    cash: float
    pos_159330_shares: int
    pos_159330_price: float
    pos_159531_shares: int
    pos_159531_price: float

    @property
    def total_value(self) -> float:
        return (self.cash
                + self.pos_159330_shares * self.pos_159330_price
                + self.pos_159531_shares * self.pos_159531_price)


class BacktestEngine:
    def __init__(self, params: BacktestParams = None, capital: float = TOTAL_CAPITAL,
                 macro_decisions: pd.DataFrame = None):
        self.p = params or BacktestParams()
        self.macro = macro_decisions
        self.capital = capital
        self.cash = capital
        self.positions = {
            "159330": Position("159330"),
            "159531": Position("159531"),
        }
        self.trades: list[Trade] = []
        self.snapshots: list[WeekSnapshot] = []
        self._etf300_ref = None
        self._etf531_ref = None

    def _macro_cap(self, date) -> float:
        """查宏观仓位上限，无数据时默认 0.3。"""
        if self.macro is None or self.macro.empty:
            return 0.3
        row = self.macro[self.macro["date"] <= date]
        if row.empty:
            return 0.3
        return float(row["max_position"].iloc[-1])

    def run(self, data: dict) -> dict:
        etf300 = data["etf_159330"].copy()
        etf531 = data["etf_159531"].copy()
        self._etf300_ref = etf300
        self._etf531_ref = etf531
        dates = self._get_fridays(etf300, etf531)

        logger.info("回测: %d周 core=%.0f%% enter=%.0f double=%s bear=%s",
                     len(dates), self.p.core_pct * 100, self.p.buy_score_enter,
                     self.p.profit_double_sell, self.p.bear_market_exit)

        for friday in dates:
            try:
                self._step(friday, data, etf300, etf531)
            except Exception as e:
                logger.debug("skip %s: %s", friday.date(), str(e)[:80])

        metrics = self._compute_metrics(dates, etf300, etf531)
        return {"trades": self.trades, "snapshots": self.snapshots, "metrics": metrics}

    @staticmethod
    def _get_fridays(etf300, etf531):
        start = max(etf300["date"].min(), etf531["date"].min())
        end = min(etf300["date"].max(), etf531["date"].max())
        all_fridays = pd.date_range(start, end, freq="W-FRI")
        dates_300 = set(etf300["date"].dt.date)
        dates_531 = set(etf531["date"].dt.date)
        valid = []
        for f in all_fridays:
            for offset in range(6):
                d = (f - pd.Timedelta(days=offset)).date()
                if d in dates_300 and d in dates_531:
                    valid.append(pd.Timestamp(d))
                    break
        return valid

    def _step(self, date, data, etf300, etf531):
        score_300 = self._score_csi300(date, data)
        score_2000 = self._score_csi2000(date, data)
        price_300 = self._get_price(etf300, date)
        price_531 = self._get_price(etf531, date)
        if price_300 is None or price_531 is None:
            return

        pos_300 = self.positions["159330"]
        pos_531 = self.positions["159531"]

        # 更新浮盈记录
        for pos, price in [(pos_300, price_300), (pos_531, price_531)]:
            pct = pos.profit_pct(price)
            if pct > pos.max_profit_pct:
                pos.max_profit_pct = pct

        # 更新熊市计数
        self._update_bear_signal("159330", etf300, date, score_300)
        self._update_bear_signal("159531", etf531, date, score_2000)

        # 1. 卖出（仅极端情况）
        self._check_sell("159330", price_300, date, score_300, etf300)
        self._check_sell("159531", price_531, date, score_2000, etf531)

        # 2. 买入
        self._check_buy("159330", price_300, date, score_300)
        self._check_buy("159531", price_531, date, score_2000)

        self.snapshots.append(WeekSnapshot(
            date=date, cash=self.cash,
            pos_159330_shares=pos_300.shares, pos_159330_price=price_300,
            pos_159531_shares=pos_531.shares, pos_159531_price=price_531,
        ))

    # ---- 打分 ----

    def _score_csi300(self, date, data):
        d = data["csi300_daily"]
        v = data.get("csi300_valuation")
        b = data["bond_yield_10y"]
        try:
            r = compute_csi300_score(
                daily=d[d["date"] <= date],
                valuation=v[v["date"] <= date] if v is not None and not v.empty else None,
                bond_yield_df=b[b["date"] <= date] if b is not None and not b.empty else b,
            )
            return r["total_score"]
        except Exception:
            return 0

    def _score_csi2000(self, date, data):
        d = data["csi2000_daily"]
        v = data.get("csi2000_valuation")
        c = data["csi300_daily"]
        m = data.get("macro_df")  # 宏观数据
        try:
            r = compute_csi2000_score(
                daily=d[d["date"] <= date],
                valuation=v[v["date"] <= date] if v is not None and not v.empty else None,
                csi300_daily=c[c["date"] <= date],
                macro=m[m["date"] <= date] if m is not None and not m.empty else None,
            )
            return r["total_score"]
        except Exception:
            return 0

    @staticmethod
    def _get_price(etf_df, date):
        row = etf_df[etf_df["date"] == date]
        return float(row["close"].iloc[0]) if not row.empty else None

    # ---- 熊市判断 ----

    def _update_bear_signal(self, code, etf_df, date, score):
        """连续熊市信号计数：价格在 MA 之下 且 分数持续很低。"""
        pos = self.positions[code]
        hist = etf_df[etf_df["date"] <= date].tail(self.p.bear_ma)
        if len(hist) < self.p.bear_ma:
            return

        ma = float(hist["close"].mean())
        current = float(hist["close"].iloc[-1])
        is_bear = (current < ma) and (score < 20)

        if is_bear:
            pos.bear_weeks += 1
        else:
            pos.bear_weeks = 0

    # ---- 目标仓位 ----

    def _target_position_pct(self, score: float) -> float:
        """分数 → 目标仓位。底仓 + 分数驱动的额外仓位。"""
        p = self.p
        if score <= p.buy_score_enter:
            return p.core_pct

        # 超过 enter 后，线性加仓到 100%
        extra = (score - p.buy_score_enter) / (p.buy_score_max - p.buy_score_enter) * (1.0 - p.core_pct)
        return min(p.core_pct + extra, 1.0)

    def _effective_target(self, code: str, score: float, date) -> float:
        """最终目标 = min(A股目标, 宏观上限) × 风格 × 波动率闸门。"""
        a_target = self._target_position_pct(score)
        m_cap = self._macro_cap(date)
        style_w = self._style_weight(code, date)
        vol_gate = self._volatility_gate(code, date)
        return min(a_target, m_cap) * style_w * vol_gate

    def _volatility_gate(self, code: str, date) -> float:
        """
        波动率闸门（参考 etf-rotation-strategy）：
        基于 510300 波动率分位自动降仓。
        低波→100%，中低→70%，中高→40%，高波→10%。
        """
        etf = self._etf300_ref if code == "159330" else self._etf531_ref
        if etf is None:
            return 1.0
        hist = etf[etf["date"] <= date].tail(252)
        if len(hist) < 60:
            return 1.0
        returns = hist["close"].pct_change().dropna()
        recent_vol = returns.tail(20).std() * np.sqrt(252)
        hist_vol = returns.rolling(20).std() * np.sqrt(252)
        pct = (hist_vol < recent_vol).sum() / len(hist_vol) if len(hist_vol) > 0 else 0.5

        if pct < 0.25:   return 1.0    # 低波 → 全仓
        elif pct < 0.50: return 0.70   # 中低波
        elif pct < 0.75: return 0.40   # 中高波
        else:            return 0.10   # 高波 → 极度降仓

    def _style_weight(self, code: str, date) -> float:
        """风格轮动权重：根据宏观象限调整大小盘配比。"""
        if self.macro is None or self.macro.empty:
            return 1.0
        row = self.macro[self.macro["date"] <= date]
        if row.empty:
            return 1.0
        r = row.iloc[-1]
        w300 = float(r.get("csi300_weight", 0.6))
        w2000 = float(r.get("csi2000_weight", 0.4))
        # 归一化到单个 ETF 最大仓位
        if code == "159330":
            return w300 / 0.6  # 基准 60% → 归一化
        else:
            return w2000 / 0.4

    # ---- 卖出（仅两种极端情况） ----

    def _check_sell(self, code, price, date, score, etf_df):
        pos = self.positions[code]
        if pos.shares == 0:
            return

        cfg = ETF_CONFIG[code]

        # 宏观强平：macro_cap = 0 → 全卖
        macro_cap = self._macro_cap(date)
        if macro_cap <= 0.01 and pos.shares > 0:
            self._do_sell(code, pos, price, date, score, pos.shares,
                          f"宏观risk-off清仓")
            return

        core_shares = int(cfg["max_capital"] * self.p.core_pct / price / 100) * 100

        # 情况1: 翻倍止盈 — 卖出可变仓位的一半
        if self.p.profit_double_sell and pos.profit_pct(price) >= 1.0:
            variable = max(0, pos.shares - core_shares)
            sell_shares = max(0, variable // 2)
            sell_shares = (sell_shares // 100) * 100
            if sell_shares > 0:
                self._do_sell(code, pos, price, date, score, sell_shares,
                              f"翻倍止盈(浮盈{pos.profit_pct(price):.0%})")
                return

        # 情况2: 熊市确认 — 连续4周熊市信号 → 减仓到最小
        if self.p.bear_market_exit and pos.bear_weeks >= 4:
            sell_shares = max(0, pos.shares - core_shares)
            sell_shares = (sell_shares // 100) * 100
            if sell_shares > 0:
                self._do_sell(code, pos, price, date, score, sell_shares,
                              f"熊市减仓(连续{pos.bear_weeks}周)")
                pos.bear_weeks = 0  # 重置，避免重复触发

    def _do_sell(self, code, pos, price, date, score, shares, reason):
        commission = max(shares * price * COMMISSION_RATE, MIN_COMMISSION)
        proceeds = pos.sell(shares, price, commission)
        self.cash += proceeds
        self.trades.append(Trade(date=date, etf=code, action="sell", shares=shares,
                                 price=price, amount=proceeds, commission=commission,
                                 score=score, reason=reason))
        logger.debug("SELL %s %d股@%.3f [%s]", code, shares, price, reason)

    # ---- 买入 ----

    def _check_buy(self, code, price, date, score):
        pos = self.positions[code]
        cfg = ETF_CONFIG[code]
        target_pct = self._effective_target(code, score, date)
        target_value = target_pct * cfg["max_capital"]
        current_value = pos.shares * price
        gap = target_value - current_value

        if gap <= 0:
            return

        affordable = min(gap, self.cash)
        if affordable < price:
            return

        shares = int(affordable / price / 100) * 100
        if shares <= 0:
            return

        cost = shares * price
        commission = max(cost * COMMISSION_RATE, MIN_COMMISSION)
        total = cost + commission

        if total > self.cash:
            shares = int((self.cash - MIN_COMMISSION) / (price * (1 + COMMISSION_RATE)) / 100) * 100
            if shares <= 0:
                return
            cost = shares * price
            commission = max(cost * COMMISSION_RATE, MIN_COMMISSION)
            total = cost + commission

        pos.buy(shares, price, commission)
        self.cash -= total
        self.trades.append(Trade(date=date, etf=code, action="buy", shares=shares,
                                 price=price, amount=cost, commission=commission,
                                 score=score, reason=f"目标{target_pct:.0%}"))
        logger.debug("BUY %s %d股@%.3f [%.0f→%.0f%%]", code, shares, price, score, target_pct * 100)

    # ---- 绩效 ----

    def _compute_metrics(self, dates, etf300, etf531):
        if not self.snapshots:
            return {}

        df = pd.DataFrame([{"date": s.date, "total_value": s.total_value} for s in self.snapshots])

        ip_300 = float(etf300[etf300["date"] == df["date"].iloc[0]]["close"].iloc[0])
        ip_531 = float(etf531[etf531["date"] == df["date"].iloc[0]]["close"].iloc[0])
        bm_s300 = int(TOTAL_CAPITAL * 0.60 / ip_300 / 100) * 100
        bm_s531 = int(TOTAL_CAPITAL * 0.40 / ip_531 / 100) * 100
        bm_cash = TOTAL_CAPITAL - bm_s300 * ip_300 - bm_s531 * ip_531

        df["benchmark"] = [bm_s300 * self._get_price(etf300, d) + bm_s531 * self._get_price(etf531, d) + bm_cash
                           for d in df["date"]]

        total_return = (df["total_value"].iloc[-1] - TOTAL_CAPITAL) / TOTAL_CAPITAL
        benchmark_return = (df["benchmark"].iloc[-1] - TOTAL_CAPITAL) / TOTAL_CAPITAL
        alpha = total_return - benchmark_return

        peak_v = df["total_value"].expanding().max()
        max_dd = float(((df["total_value"] - peak_v) / peak_v).min())

        total_days = (df["date"].iloc[-1] - df["date"].iloc[0]).days
        annual_return = (1 + total_return) ** (365 / total_days) - 1 if total_days > 0 else 0

        df["year"] = df["date"].dt.year
        yearly = []
        for year, grp in df.groupby("year"):
            sv, ev = grp["total_value"].iloc[0], grp["total_value"].iloc[-1]
            ret = (ev - sv) / sv if sv > 0 else 0
            bm_sv, bm_ev = grp["benchmark"].iloc[0], grp["benchmark"].iloc[-1]
            bm_ret = (bm_ev - bm_sv) / bm_sv if bm_sv > 0 else 0
            peak = grp["total_value"].expanding().max()
            y_dd = float(((grp["total_value"] - peak) / peak).min())
            yt = [t for t in self.trades if t.date.year == year]
            yearly.append({"year": year, "start_value": round(sv, 0), "end_value": round(ev, 0),
                           "return": f"{ret:.2%}", "benchmark": f"{bm_ret:.2%}",
                           "max_drawdown": f"{y_dd:.2%}",
                           "trades": len(yt), "buys": sum(1 for t in yt if t.action == "buy"),
                           "sells": sum(1 for t in yt if t.action == "sell")})

        return {
            "yearly": yearly,
            "total_return": total_return,
            "benchmark_return": benchmark_return,
            "alpha": alpha,
            "annual_return": annual_return,
            "max_drawdown": max_dd,
            "total_trades": len(self.trades),
            "buy_trades": len([t for t in self.trades if t.action == "buy"]),
            "sell_trades": len([t for t in self.trades if t.action == "sell"]),
            "final_cash": round(self.cash, 0),
            "df": df,
        }
