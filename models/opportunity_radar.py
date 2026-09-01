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
  opportunities: [{'name','code','date','factors','price','ref_buy_zone','pct','detail'}]
  watching:      [{'name','code','missing':[...]}]  # 接近触发但条件不全
  factor_status: {行业: {'factor','chg_20d','window'}}
"""

from __future__ import annotations

import json
import warnings
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

warnings.filterwarnings("ignore")

# ── 行业池：code, name, 因子组 ─────────────────────────────
# 因子组为 None → 两条件模式（B+C），有代理则三条件（A+B+C）
ETF_POOL = [
    {"code": "159825", "name": "农业",   "factors": ["M0", "SR0", "C0"]},  # 豆粕/白糖/玉米
    {"code": "159587", "name": "粮食",   "factors": ["M0", "SR0", "C0"]},
    {"code": "512400", "name": "有色",   "factors": ["CU0"]},              # 沪铜
    {"code": "515220", "name": "煤炭",   "factors": ["JM0"]},              # 焦煤
    {"code": "512880", "name": "证券",   "factors": ["margin"]},           # 两融（管道已有）
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
# 主扫描
# ═══════════════════════════════════

def radar_scan(date=None) -> dict:
    """全池扫描。返回 {opportunities, watching, factor_status}。"""
    from data.fetcher import fetch_etf_daily

    factor_cache: dict = {}
    margin_df = _load_margin()
    opportunities, watching = [], []
    factor_status = {}
    last_signal: dict = {}  # 冷却：同一标的历史信号日（回放时传入或内部维护）
    today = pd.Timestamp(date) if date is not None else pd.Timestamp(datetime.now().date())

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
        detail = {}
        if det_a: detail["因子窗口"] = det_a
        detail["行情确认"] = det_b

        # 冷却：同一标的 60 自然日内已报过机会 → 不重复报（防连续多日刷屏）
        last_dt = last_signal.get(name)
        in_cooldown = last_dt is not None and (today - last_dt).days < 60

        # 机会只报"因子确认 + 行情确认"（A+B）；无因子行业（B-only）实证假信号
        # 率过高（军工-18%/医药-7%），降级为观察，不报机会
        if factors and ok_a and ok_b and not in_cooldown:
            last_signal[name] = today
            opportunities.append({
                "name": name, "code": code, "date": str(df["date"].iloc[-1].date()),
                "factors": factors, "price": px,
                "ref_buy_zone": f"{round(px * 0.97, 3)}~{round(px * 1.01, 3)}",
                "pct": 0.05, "detail": detail,
            })
        else:
            need = []
            if not factors:
                need.append("无因子代理·仅技术观察")
            else:
                if not ok_a: need.append("因子窗口未开")
                if not ok_b: need.append("行情未确认")
                if in_cooldown: need.append("冷却期内")
            need.extend(missing)
            watching.append({"name": name, "code": code, "missing": need or ["观察"]})
    return {"opportunities": opportunities, "watching": watching, "factor_status": factor_status}
