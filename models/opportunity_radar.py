"""
机会雷达 v1 — 挖掘核心(沪深300/中证2000)之外的行业机会（独立模型）。

定位（2026-09-01 用户确认）:
- 不碰 300/2000 的任何决定——只输出"额外机会"，轻仓参考（单标≤5%，总卫星≤15%）
- 区别于纯冰点扫描：不买"便宜但永远不涨"的宽基——需要环境/行情驱动的机会确认
- 核心洞察：期货是行业的"预期温度计"（夜盘+杠杆+预期定价，领先股票现货），
  商品/汇率/两融等金融因子作为"环境窗口"的领先指标

三条件共振（按行业可用性可选，无因子代理的行业走两条件模式）:
  A. 金融因子窗口  代理因子近20日涨幅>5% 或 突破60日新高（趋势性变化）
  B. 行情确认      站回MA10（或收盘≥MA10×0.99 且 5日均量>2×20日均量，防8/21型差0.3%错过）
                   + 量能5/20>1.1 + 距250日低点 10%~35%（启动初期未加速）
  C. 估值约束      行业PE分位<60%（防追高；数据链路可用后启用，缺数据时跳过并标注）

输出:
  opportunities: [{'name','code','date','factors','price','ref_buy_zone','pct',
                   'stop','tp','stop_pct','tp_pct','detail'}]
  watching:      [{'name','code','missing':[...]}]  # 接近触发但条件不全
  exits:         [{'name','code','reason','exit_price','ret',...}]  # 触发的退出
  factor_status: {行业: {'factor','chg_20d','window'}}

资金划分（核心 / 卫星）:
  核心 沪深300 + 中证2000 —— 由各自模型独立决定仓位，本模型不干预
  卫星 本雷达 —— 单标 ≤7.5%，总计 ≤15%（SATELLITE_SINGLE_PCT / SATELLITE_TOTAL_PCT）
  硬约束 核心 + 卫星 ≤ 100%；卫星额度被占满后新信号降级为观察，不报机会
  仓位 7.5% 的推导（实测 n=25，剔除证券）:
       胜率 68.0%（90%CI 下界 52.7%）、盈亏比 1.63、盈亏平衡胜率 38.0%
       → 全 Kelly 23.6%~48.4%，取 1/4 Kelly 区间中点 ≈ 7.5%
       → 风险预算: 2 标×7.5%×5% 止损 = 组合最大损失 0.75%

退出规则（与 backtest/radar_atr_stop_study.py 同口径，收盘价触发）:
  止损 = max(2.5×ATR20, 5%)   止盈 = 2×止损   最长持有 60 交易日
  为何用 ATR 而非固定 5%（2026-09-10 实测）:
    横截面差异很小（信号日各标的 ATR 仅 1.77%~2.02%，1.14 倍）——**不是**理由；
    真实理由是**时间维度**：单一标的自身 ATR 的 P90/P10 达 2.0~2.6 倍，
    固定 5% 在不同 regime 相当于 0.52x~5.71x ATR，即同一规则的风险敞口浮动 10 倍。
    30 个历史信号恰好全部落在中低波 regime，回测对两者不可区分
    （低波组差 0.00%、高波组差 +0.06%），故 ATR 是**regime 稳健性**修正，
    不是回测收益修正；5% 下限保证它在已验证样本上是空操作，只在高波 regime 激活。
"""

from __future__ import annotations

import json
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

warnings.filterwarnings("ignore")

# ── 资金划分与退出参数（依据见模块 docstring）──────────────
SATELLITE_SINGLE_PCT = 0.075   # 单标仓位（1/4 Kelly 区间中点）
SATELLITE_TOTAL_PCT = 0.15     # 卫星总仓位上限
ATR_MULT = 2.5                 # 止损 = ATR_MULT × ATR20
STOP_FLOOR = 0.05              # 止损下限（= 已验证口径，只放宽不收紧）
TP_MULT = 2.0                  # 止盈 = TP_MULT × 止损
MAX_HOLD_BARS = 60             # 最长持有交易日
COOLDOWN_DAYS = 60             # 同标的冷却自然日

STATE_FILE = DATA_DIR / "radar_state.json"

# ── 行业池：code, name, 因子组 ─────────────────────────────
# 因子组为 None → 两条件模式（B+C），有代理则三条件（A+B+C）
ETF_POOL = [
    {"code": "159825", "name": "农业",   "factors": ["M0", "SR0", "C0"]},  # 豆粕/白糖/玉米
    {"code": "159587", "name": "粮食",   "factors": ["M0", "SR0", "C0"]},
    {"code": "512400", "name": "有色",   "factors": ["CU0"]},              # 沪铜
    {"code": "515220", "name": "煤炭",   "factors": ["JM0"]},              # 焦煤
    # 证券 2026-09-10 降级：实证 n=5 胜率 0%（5个信号全部止损）。
    # 机理：两融余额是月度存量慢变量，套用期货"20日涨幅>5%"的日频动量模板不成立。
    {"code": "512880", "name": "证券",   "factors": None},
    {"code": "512480", "name": "半导体", "factors": None},                 # 无期货代理→两条件
    {"code": "512010", "name": "医药",   "factors": None},
    {"code": "512660", "name": "军工",   "factors": None},
    {"code": "159928", "name": "消费",   "factors": None},
    {"code": "159611", "name": "电力",   "factors": None},
    {"code": "512690", "name": "白酒",   "factors": None},
    {"code": "515030", "name": "新能源车", "factors": None},
]

# 期货品种名（akshare futures_main_sina symbol）
FUTURE_NAMES = {"M0": "豆粕", "SR0": "白糖", "C0": "玉米", "CU0": "沪铜", "JM0": "焦煤"}


# ═══════════════════════════════════
# 因子数据
# ═══════════════════════════════════

def _futures_cache_path(symbol: str) -> Path:
    return DATA_DIR / f"futures_{symbol}_daily.parquet"


def _load_futures(symbol: str) -> pd.DataFrame | None:
    """期货主力连续日线（新浪 akshare；1天缓存按日历日过期，与管道一致）。"""
    cache = _futures_cache_path(symbol)
    if cache.exists():
        mtime = datetime.fromtimestamp(cache.stat().st_mtime)
        if mtime.date() == datetime.now().date():  # 当日缓存直接复用
            return pd.read_parquet(cache)
    try:
        import akshare as ak
        df = ak.futures_main_sina(symbol=symbol)
        df.columns = ["date", "open", "high", "low", "close", "vol", "amt", "hold"]
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        df.to_parquet(cache)
        return df
    except Exception as e:
        print(f"⚠ 期货{symbol}({FUTURE_NAMES.get(symbol,'')})拉取失败: {str(e)[:60]}")
        return pd.read_parquet(cache) if cache.exists() else None


def _load_margin() -> pd.DataFrame | None:
    """两融余额（管道已有数据，证券行业因子）→ 统一为 close 列接口。"""
    p = DATA_DIR / "margin_balance.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df.rename(columns={"margin_balance": "close"})


# ═══════════════════════════════════
# 三条件判定
# ═══════════════════════════════════

def _factor_window(factor_df: pd.DataFrame, lookback: int = 20,
                   chg_th: float = 0.05) -> tuple[bool, dict]:
    """条件A 金融因子窗口：近 lookback 日涨幅>5% 或 突破60日新高。"""
    if factor_df is None or len(factor_df) < 80:
        return False, {"可用": False}
    c = factor_df["close"]
    px = float(c.iloc[-1])
    chg = (px / float(c.iloc[-1 - lookback]) - 1) if len(c) > lookback else 0.0
    hi60 = float(c.tail(60).max())
    breakout = px > hi60  # 严格大于：平盘序列整天等于最高点不算突破
    detail = {
        "近20日涨幅": f"{chg:+.1%}",
        "突破60日新高": "✅" if breakout else "❌",
        "窗口": "开启" if (chg > chg_th or breakout) else "关闭",
    }
    return (chg > chg_th or breakout), detail


def _confirm_signal(df: pd.DataFrame) -> tuple[bool, dict]:
    """条件B 行情确认：企稳（站回MA10或贴近+爆量）+ 量能回暖 + 启动初期未加速。"""
    if df is None or len(df) < 70:
        return False, {"可用": False}
    c = df["close"]
    px = float(c.iloc[-1])
    ma10 = float(c.tail(10).mean())
    above = px > ma10
    vol5 = float(df["volume"].tail(5).mean())
    vol20 = float(df["volume"].tail(20).mean())
    vol_ratio = vol5 / vol20 if vol20 > 0 else 0.0
    near = px >= ma10 * 0.99 and vol_ratio >= 2.0  # 8/21型：贴近MA10+爆量视为企稳
    lo250 = float(c.tail(250).min())
    rise = (px / lo250 - 1) if lo250 > 0 else 0.0
    detail = {
        "收盘": round(px, 3),
        "MA10": round(ma10, 3),
        "企稳": "✅站回" if above else ("✅贴近+爆量" if near else "❌"),
        "量能5/20": f"{vol_ratio:.2f}",
        "距250日低点": f"{rise:+.1%}",
        "启动区": "✅" if 0.10 <= rise <= 0.35 else "❌",
    }
    ok = (above or near) and vol_ratio > 1.1 and 0.10 <= rise <= 0.35
    return ok, detail


# ═══════════════════════════════════
# 风险线 / 持仓状态 / 退出
# ═══════════════════════════════════

def _atr_pct(df: pd.DataFrame, n: int = 20) -> float | None:
    """ATR(n) / 收盘价 —— 波动率的"当下水位"，用于自适应止损宽度。"""
    if df is None or len(df) < n + 1:
        return None
    c = df["close"].values.astype(float)
    h = df["high"].values.astype(float) if "high" in df.columns else c
    l = df["low"].values.astype(float) if "low" in df.columns else c
    pc = np.roll(c, 1)
    pc[0] = c[0]
    tr = np.maximum.reduce([h - l, np.abs(h - pc), np.abs(l - pc)])
    atr = float(pd.Series(tr).rolling(n).mean().iloc[-1])
    return atr / float(c[-1]) if c[-1] > 0 and not np.isnan(atr) else None


def _risk_lines(px: float, atr_pct: float | None) -> dict:
    """止损/止盈价。止损 = max(ATR_MULT×ATR, STOP_FLOOR)，止盈 = TP_MULT×止损。"""
    sp = max(ATR_MULT * atr_pct, STOP_FLOOR) if atr_pct else STOP_FLOOR
    tp_pct = sp * TP_MULT
    return {
        "stop_pct": round(sp, 4), "tp_pct": round(tp_pct, 4),
        "stop": round(px * (1 - sp), 3), "tp": round(px * (1 + tp_pct), 3),
    }


def satellite_room(positions: list[dict]) -> float:
    """卫星剩余可用额度（总上限 15% − 已占用）。"""
    used = sum(float(p.get("pct", 0.0)) for p in positions)
    return max(0.0, SATELLITE_TOTAL_PCT - used)


def _load_state(path=None) -> dict:
    """加载冷却记录 + 持仓。文件损坏时返回空状态（不阻断当日扫描）。"""
    p = Path(path) if path else STATE_FILE
    if not p.exists():
        return {"last_signal": {}, "positions": []}
    try:
        s = json.loads(p.read_text(encoding="utf-8"))
        s.setdefault("last_signal", {})
        s.setdefault("positions", [])
        return s
    except Exception as e:
        print(f"⚠ 雷达状态文件损坏({str(e)[:40]})，按空状态继续")
        return {"last_signal": {}, "positions": []}


def _save_state(state: dict, path=None) -> None:
    p = Path(path) if path else STATE_FILE
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def close_position(code: str, reason: str = "手动", state_path=None) -> dict | None:
    """手动平掉某标的的雷达持仓并返回它（无则 None）。

    用于对账：信号报了但用户没实际买入，或用户已自行卖出。
    不解除冷却——该信号已报过，避免平仓后立刻被重新建仓。
    """
    state = _load_state(state_path)
    hit = next((p for p in state["positions"] if p.get("code") == code), None)
    if hit is None:
        return None
    state["positions"] = [p for p in state["positions"] if p.get("code") != code]
    _save_state(state, state_path)
    return {**hit, "reason": reason}


def check_exits(positions: list[dict], quotes: dict) -> tuple[list[dict], list[dict]]:
    """逐日核对持仓：返回 (继续持有, 退出触发)。

    收盘价触发（与回测同口径，不用盘中最高/最低）。优先级：止损 > 止盈 > 到期。
    同一交易日重复扫描不重复计入持有天数（按 last_bar 去重）。
    """
    still, exits = [], []
    for pos in positions:
        p = dict(pos)
        q = quotes.get(p.get("code"))
        if not q or "close" not in q:
            still.append(p)          # 数据缺失不得误判为退出
            continue
        px = float(q["close"])
        bar = q.get("date")
        if bar is None or bar != p.get("last_bar"):
            p["bars_held"] = p.get("bars_held", 0) + 1
            if bar is not None:
                p["last_bar"] = bar

        reason = None
        if px <= p["stop"]:
            reason = "止损"
        elif px >= p["tp"]:
            reason = "止盈"
        elif p["bars_held"] >= MAX_HOLD_BARS:
            reason = "到期"

        if reason:
            exits.append({**p, "reason": reason, "exit_price": px,
                          "ret": round(px / p["entry"] - 1, 4)})
        else:
            still.append(p)
    return still, exits


# ═══════════════════════════════════
# 主扫描
# ═══════════════════════════════════

def radar_scan(date=None, state_path=None) -> dict:
    """全池扫描。返回 {opportunities, watching, exits, factor_status}。

    持仓与冷却状态落盘到 state_path（默认 data/radar_state.json），
    因此重复调用不会重复报同一信号（60 自然日冷却）。

    传入 date= 进入**历史回放模式**：不读也不写状态文件——否则会用历史价格
    去判定当前持仓的止损止盈，并把持有天数算乱。
    """
    from data.fetcher import fetch_etf_daily

    replay = date is not None
    factor_cache: dict = {}
    margin_df = _load_margin()
    opportunities, watching = [], []
    factor_status = {}
    state = {"last_signal": {}, "positions": []} if replay else _load_state(state_path)
    last_signal: dict = state["last_signal"]
    positions: list[dict] = state["positions"]
    room = satellite_room(positions)
    quotes: dict = {}
    today = pd.Timestamp(date) if replay else pd.Timestamp(datetime.now().date())

    for item in ETF_POOL:
        code, name, factors = item["code"], item["name"], item["factors"]
        try:
            df = fetch_etf_daily(code, name, force=False)
            df["date"] = pd.to_datetime(df["date"])
            if date is not None:
                df = df[df["date"] <= pd.Timestamp(date)]
            df = df.sort_values("date").reset_index(drop=True)
        except Exception as e:
            watching.append({"name": name, "code": code, "missing": [f"数据:{str(e)[:40]}"]})
            continue

        missing = []
        ok_a, det_a = False, {"窗口": "无因子"}
        if factors:
            det_a = {"窗口": "关闭"}
            fs_detail = {}
            for f in factors:
                if f == "margin":
                    fd = margin_df
                    fname = "两融余额"
                else:
                    if f not in factor_cache:
                        factor_cache[f] = _load_futures(f)
                    fd = factor_cache[f]
                    fname = FUTURE_NAMES.get(f, f)
                if fd is None:
                    missing.append(f"因子{fname}缺数据")
                    continue
                ok_f, det_f = _factor_window(fd)
                fs_detail[fname] = {"chg_20d": det_f.get("近20日涨幅", "?"),
                                    "window": "✅" if ok_f else "—"}
                if ok_f:
                    ok_a = True
                    det_a = det_f
            factor_status[name] = fs_detail or {"—": {"chg_20d": "缺数据", "window": "—"}}
        else:
            factor_status[name] = {"无因子代理": {"chg_20d": "—", "window": "两条件观察"}}

        ok_b, det_b = _confirm_signal(df)
        px = float(df["close"].iloc[-1])
        bar_date = str(df["date"].iloc[-1].date())
        quotes[code] = {"close": px, "date": bar_date}
        detail = {}
        if det_a: detail["因子窗口"] = det_a
        detail["行情确认"] = det_b

        # 冷却：同一标的 60 自然日内已报过机会 → 不重复报（防连续多日刷屏）
        last_dt = last_signal.get(name)
        last_dt = pd.Timestamp(last_dt) if last_dt else None
        in_cooldown = last_dt is not None and (today - last_dt).days < COOLDOWN_DAYS

        # 已持仓 → 始终展示（那是活仓，不是新信号；同一天重扫也不会消失）；
        # 新信号 → 受 60 日冷却 与 卫星额度 双重约束
        # 机会只报"因子确认 + 行情确认"（A+B）；无因子行业（B-only）实证假信号
        # 率过高（军工-18%/医药-7%），降级为观察，不报机会
        held = next((p for p in positions if p.get("code") == code), None)
        no_room = room < SATELLITE_SINGLE_PCT
        if factors and ok_a and ok_b and (held is not None or (not in_cooldown and not no_room)):
            if held is not None:
                rl = {k: held[k] for k in ("stop", "tp", "stop_pct", "tp_pct")}
            else:
                rl = _risk_lines(px, _atr_pct(df))
                last_signal[name] = bar_date
                room -= SATELLITE_SINGLE_PCT
                positions.append({
                    "name": name, "code": code, "entry": px, "entry_date": bar_date,
                    "pct": SATELLITE_SINGLE_PCT, "bars_held": 0, "last_bar": bar_date, **rl,
                })
            opportunities.append({
                "name": name, "code": code, "date": bar_date,
                "factors": factors, "price": px, "held": held is not None,
                "ref_buy_zone": f"{round(px * 0.97, 3)}~{round(px * 1.01, 3)}",
                "pct": SATELLITE_SINGLE_PCT, **rl, "detail": detail,
            })
        else:
            need = []
            if not factors:
                need.append("无因子代理·仅技术观察")
            else:
                if not ok_a: need.append("因子窗口未开")
                if not ok_b: need.append("行情未确认")
                if in_cooldown: need.append("冷却期内")
                if no_room: need.append(f"卫星额度不足(剩{room:.1%})")
            need.extend(missing)
            watching.append({"name": name, "code": code, "missing": need or ["观察"]})

    if replay:
        return {"opportunities": opportunities, "watching": watching, "positions": [],
                "exits": [], "factor_status": factor_status}

    positions, exits = check_exits(positions, quotes)
    state.update({"last_signal": last_signal, "positions": positions})
    _save_state(state, state_path)
    return {"opportunities": opportunities, "watching": watching, "positions": positions,
            "exits": exits, "factor_status": factor_status}
