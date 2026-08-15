"""
A股数据获取 — 标的: CSI 300(159330) / CSI 2000(159531)

多渠道冗余: 每个数据至少两个独立源，主源失败自动降级
- 指数日线: Sina 主源 + BaoStock 校验
- ETF 日线: Sina 主源 → 东财备源 → 缓存兜底 + BaoStock 校验
- 估值: 乐股(PE/PB) / csindex(PE)
东财网络不稳定（本机代理劫持），仅作备源，不影响主流程
"""

from __future__ import annotations

import json, logging, time
from datetime import datetime
from pathlib import Path
from typing import Optional
import pandas as pd
import requests

_orig_init = requests.Session.__init__
def _patch(self, *a, **kw):
    _orig_init(self, *a, **kw)
    self.trust_env = False
    self.verify = False
    self.headers.update({"User-Agent": "Mozilla/5.0"})
requests.Session.__init__ = _patch

logger = logging.getLogger(__name__)
DATA_DIR = Path(__file__).resolve().parent
CACHE_AGE = 1


def _cp(f): return DATA_DIR / f


def _load(f):
    p = _cp(f)
    if not p.exists(): return None
    if (datetime.now() - datetime.fromtimestamp(p.stat().st_mtime)).days >= CACHE_AGE:
        return None
    return pd.read_parquet(p)


def _save(df, f):
    _cp(f).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(_cp(f), index=False)
    logger.info("缓存: %s (%d行)", f, len(df))


def _v(df, n):
    if df is None or df.empty: raise ValueError(f"{n}: 空")
    return df.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)


def _xv(df_main, df_check, name, threshold=0.005):
    m = df_main[["date", "close"]].merge(df_check[["date", "close"]], on="date",
                                          suffixes=("_m", "_c"), how="inner")
    if m.empty: return df_main
    m["d"] = (m["close_m"] - m["close_c"]).abs() / m["close_c"]
    bad = m[m["d"] > threshold]
    if len(bad):
        logger.warning("%s: %d/%d日差异>%.1f%%", name, len(bad), len(m), threshold * 100)
    else:
        logger.info("%s: ✓ (%d日)", name, len(m))
    return df_main


# ============================================================
# CSI 300 — Sina 主源 + BaoStock 校验（稳定，2002至今）
# ============================================================

def _bs300():
    import baostock as bs; bs.login()
    rs = bs.query_history_k_data_plus("sh.000300",
        "date,open,close,high,low,volume,amount",
        start_date="2002-01-01", end_date=datetime.now().strftime("%Y-%m-%d"),
        frequency="d", adjustflag="3")
    rows = []
    while (rs.error_code == '0') & rs.next(): rows.append(rs.get_row_data())
    bs.logout()
    df = pd.DataFrame(rows, columns=rs.fields)
    df["date"] = pd.to_datetime(df["date"])
    for c in ["open","close","high","low","volume","amount"]:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
    return _v(df, "BS300")


def fetch_csi300_daily(force=False):
    cache = "csi300_daily.parquet"
    if not force:
        c = _load(cache)
        if c is not None: return c

    import akshare as ak
    df = ak.stock_zh_index_daily(symbol="sh000300")
    df = df.rename(columns={"open":"open","high":"high","low":"low","close":"close","volume":"volume"})
    df["date"] = pd.to_datetime(df["date"])
    df = df[["date","open","close","high","low","volume"]]
    df = _v(df, "CSI300-Sina")

    try: df = _xv(df, _bs300(), "CSI300×BaoStock")
    except Exception as e: logger.warning("BaoStock: %s", e)

    logger.info("CSI300: %d行 %s~%s", len(df),
                df["date"].min().strftime("%Y-%m-%d"), df["date"].max().strftime("%Y-%m-%d"))
    _save(df, cache)
    return df


# ============================================================
# CSI 2000 — csindex 主源 + 东财补充（东财失败时用缓存）
# ============================================================

def _em_csi2000():
    """东财 CSI2000 K线（可选，可能失败）。"""
    url = ("https://push2his.eastmoney.com/api/qt/stock/kline/get"
           "?secid=2.932000&klt=101&fqt=1&beg=20100101"
           f"&end={datetime.now().strftime('%Y%m%d')}"
           "&fields1=f1,f2,f3,f4,f5,f6"
           "&fields2=f51,f52,f53,f54,f55,f56,f57")
    r = requests.get(url, timeout=30); r.raise_for_status()
    klines = json.loads(r.text).get("data", {}).get("klines")
    if not klines: raise ValueError("空")
    rows = []
    for line in klines:
        p = line.split(",")
        if len(p) < 7: continue
        rows.append({"date": pd.Timestamp(p[0]), "open": float(p[1]), "close": float(p[2]),
                     "high": float(p[3]), "low": float(p[4]),
                     "volume": float(p[5]), "amount": float(p[6])})
    return _v(pd.DataFrame(rows), "CSI2000-EM")


def fetch_csi2000_daily(force=False):
    cache = "csi2000_daily.parquet"
    if not force:
        c = _load(cache)
        if c is not None: return c

    import akshare as ak

    # 1) csindex 历史（稳定）
    df_hist = pd.DataFrame()
    try:
        df = ak.stock_zh_index_hist_csindex(symbol="932000")
        df = df.rename(columns={"日期":"date","开盘":"open","收盘":"close",
                                "最高":"high","最低":"low","成交量":"volume","成交额":"amount"})
        df["date"] = pd.to_datetime(df["date"])
        cols = ["date","open","close","high","low","volume"]
        if "amount" in df.columns: cols.append("amount")
        df_hist = _v(df[cols], "CSI2000-csindex")
        logger.info("CSI2000-csindex: %d行 ~%s", len(df_hist),
                    df_hist["date"].max().strftime("%Y-%m-%d"))
    except Exception as e:
        logger.warning("CSI2000-csindex: %s", e)

    # 2) 东财补充最新（可选，失败不阻塞）
    try:
        df_em = _em_csi2000()
        logger.info("CSI2000-东财: %d行 ~%s", len(df_em),
                    df_em["date"].max().strftime("%Y-%m-%d"))
        if not df_hist.empty:
            df = pd.concat([df_hist, df_em], ignore_index=True)
            df = df.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
        else:
            df = df_em
    except Exception as e:
        logger.warning("CSI2000-东财不可用(%s)", str(e)[:80])
        if df_hist.empty:
            raise RuntimeError("CSI2000: 所有数据源均失败")
        df = df_hist

    logger.info("CSI2000: %d行 %s~%s", len(df),
                df["date"].min().strftime("%Y-%m-%d"), df["date"].max().strftime("%Y-%m-%d"))
    _save(df, cache)
    return df


# ============================================================
# ETF — 东财 + BaoStock 校验（东财失败时走缓存）
# ============================================================

def _bs_etf(code):
    import baostock as bs; bs.login()
    rs = bs.query_history_k_data_plus(f"sz.{code}",
        "date,open,close,high,low,volume,amount,turn",
        start_date="2024-01-01", end_date=datetime.now().strftime("%Y-%m-%d"),
        frequency="d", adjustflag="2")
    rows = []
    while (rs.error_code == '0') & rs.next(): rows.append(rs.get_row_data())
    bs.logout()
    df = pd.DataFrame(rows, columns=rs.fields)
    df["date"] = pd.to_datetime(df["date"])
    for c in ["open","close","high","low","volume","amount","turn"]:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
    return _v(df.rename(columns={"turn":"turnover"}), f"BS-{code}")


def _sina_etf(code):
    """新浪 ETF 日K线（稳定主源，东财不可用时的保障）。"""
    url = ("https://money.finance.sina.com.cn/quotes_service/api/json_v2.php/"
           "CN_MarketData.getKLineData")
    r = requests.get(url, params={
        "symbol": f"sz{code}", "scale": 240, "ma": "no", "datalen": 1023,
    }, timeout=20); r.raise_for_status()
    data = json.loads(r.text)
    if not data:
        raise ValueError("空")
    rows = []
    for d in data:
        rows.append({"date": pd.Timestamp(d["day"]), "open": float(d["open"]),
                     "close": float(d["close"]), "high": float(d["high"]),
                     "low": float(d["low"]), "volume": float(d["volume"])})
    return _v(pd.DataFrame(rows), f"ETF{code}-Sina")


def fetch_etf_daily(code, name, force=False):
    cache = f"etf_{code}_daily.parquet"
    if not force:
        c = _load(cache)
        if c is not None: return c

    df = None
    last_err = None

    # 1) 新浪主源（稳定）
    try:
        df = _sina_etf(code)
    except Exception as e:
        last_err = e
        logger.warning("ETF%s 新浪: %s", name, str(e)[:80])

    # 2) 东财补充（可选，失败不阻塞）
    if df is None:
        try:
            secid_map = {"159330": "0.159330", "159531": "0.159531"}
            url = ("https://push2his.eastmoney.com/api/qt/stock/kline/get"
                   f"?secid={secid_map[code]}&klt=101&fqt=1&beg=20230101"
                   f"&end={datetime.now().strftime('%Y%m%d')}"
                   "&fields1=f1,f2,f3,f4,f5,f6"
                   "&fields2=f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61")
            r = requests.get(url, timeout=30); r.raise_for_status()
            klines = json.loads(r.text).get("data", {}).get("klines")
            if not klines: raise RuntimeError(f"ETF{code} 东财空")
            rows = []
            for line in klines:
                p = line.split(",")
                if len(p) < 7: continue
                rows.append({"date": pd.Timestamp(p[0]), "open": float(p[1]),
                             "close": float(p[2]), "high": float(p[3]),
                             "low": float(p[4]), "volume": float(p[5]),
                             "amount": float(p[6])})
            df = _v(pd.DataFrame(rows), f"ETF{code}")
        except Exception as e:
            last_err = e
            logger.warning("ETF%s 东财: %s", name, str(e)[:80])

    # 3) 双源都失败 → 回退缓存
    if df is None:
        cached = _load(cache) if not force else None
        if cached is not None:
            logger.warning("ETF%s: 全部数据源失败(%s)，使用缓存", name, str(last_err)[:60])
            return cached
        raise RuntimeError(f"ETF{code}: 所有数据源均失败: {last_err}")

    # 4) BaoStock 校验（可选）
    try: df = _xv(df, _bs_etf(code), f"ETF{name}")
    except Exception as e: logger.warning("ETF%s BaoStock: %s", name, e)

    logger.info("ETF%s: %d行 %s~%s", name, len(df),
                df["date"].min().strftime("%Y-%m-%d"), df["date"].max().strftime("%Y-%m-%d"))
    _save(df, cache)
    return df


# ============================================================
# PE/PB 估值 & 国债
# ============================================================

# 指数代码 → 估值数据源配置
# type "legulegu": stock_index_pe_lg / stock_index_pb_lg（乐股，历史长）
# type "csindex":  stock_zh_index_value_csindex（中证指数公司，仅最近20日 + PE only）
_VALUATION_CONFIG = {
    "000300": {"type": "legulegu", "index_name": "沪深300"},
    "932000": {"type": "csindex", "index_code": "932000"},
}


def fetch_valuation(symbol, name):
    """
    获取指数 PE/PB 历史数据。

    - CSI 300: 乐股 stock_index_pe_lg/pb_lg，全历史 PE+PB
    - CSI 2000: 中证指数公司 stock_zh_index_value_csindex，仅 PE，渐进累积缓存
    """
    cfg = _VALUATION_CONFIG.get(symbol)
    if cfg is None:
        logger.warning("%s估值: 未找到对应的估值配置", name)
        return None

    cache = f"valuation_{symbol}.parquet"
    cached = _load(cache)

    import akshare as ak

    if cfg["type"] == "legulegu":
        df = _fetch_valuation_legulegu(ak, cfg["index_name"], name)
    elif cfg["type"] == "csindex":
        df = _fetch_valuation_csindex(ak, cfg["index_code"], name, cached)
    else:
        return None

    if df is None or df.empty:
        return cached

    # 渐进累积：新数据合并到缓存
    if cached is not None and not cached.empty:
        df = pd.concat([cached, df], ignore_index=True)
        df = df.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)

    df = df[(df["pe"] > 0) | (df["pb"] > 0)]
    logger.info("%s估值: %d行 PE%.2f~%.2f PB%.2f~%.2f (缓存+新增)",
                name, len(df), df["pe"].min(), df["pe"].max(),
                df["pb"].min() if "pb" in df.columns else float("nan"),
                df["pb"].max() if "pb" in df.columns else float("nan"))
    _save(df, cache)
    return _v(df, f"{name}-估值")


def _fetch_valuation_legulegu(ak, index_name, name):
    """乐股数据源：stock_index_pe_lg + stock_index_pb_lg。"""
    try:
        df_pe = ak.stock_index_pe_lg(symbol=index_name)
        df_pb = ak.stock_index_pb_lg(symbol=index_name)
    except Exception as e:
        logger.warning("%s估值(乐股): %s", name, str(e)[:80])
        return None

    # PE: 列名 ['日期','指数','等权静态市盈率','静态市盈率','静态市盈率中位数','等权滚动市盈率','滚动市盈率','滚动市盈率中位数']
    pe_date_col = _find_col(df_pe, ["日期"])
    pe_val_col = _find_col(df_pe, ["滚动市盈率", "静态市盈率"])
    if pe_date_col is None or pe_val_col is None:
        logger.warning("%s估值: 无法解析 PE 列名 %s", name, list(df_pe.columns))
        return None

    pe_df = pd.DataFrame({
        "date": pd.to_datetime(df_pe[pe_date_col], errors="coerce"),
        "pe": pd.to_numeric(df_pe[pe_val_col], errors="coerce"),
    }).dropna(subset=["date", "pe"])

    # PB: 列名 ['日期','指数','市净率','等权市净率','市净率中位数']
    pb_date_col = _find_col(df_pb, ["日期"])
    pb_val_col = _find_col(df_pb, ["市净率"])
    if pb_date_col is None or pb_val_col is None:
        logger.warning("%s估值: 无法解析 PB 列名 %s", name, list(df_pb.columns))
        return None

    pb_df = pd.DataFrame({
        "date": pd.to_datetime(df_pb[pb_date_col], errors="coerce"),
        "pb": pd.to_numeric(df_pb[pb_val_col], errors="coerce"),
    }).dropna(subset=["date", "pb"])

    df = pe_df.merge(pb_df, on="date", how="inner").sort_values("date").reset_index(drop=True)
    return df


def _fetch_valuation_csindex(ak, index_code, name, cached):
    """中证指数公司数据源：stock_zh_index_value_csindex（仅 PE，最近 20 日）。"""
    try:
        raw = ak.stock_zh_index_value_csindex(symbol=index_code)
    except Exception as e:
        logger.warning("%s估值(csindex): %s", name, str(e)[:80])
        return None

    # 列名: ['日期','指数代码','指数中文全称','指数中文简称',...,'市盈率1','市盈率2','股息率1','股息率2']
    date_col = _find_col(raw, ["日期"])
    pe_col = _find_col(raw, ["市盈率2", "市盈率1"])  # 市盈率2=TTM, 市盈率1=静态
    if date_col is None or pe_col is None:
        logger.warning("%s估值: 无法解析 csindex 列名 %s", name, list(raw.columns))
        return None

    df = pd.DataFrame({
        "date": pd.to_datetime(raw[date_col], errors="coerce"),
        "pe": pd.to_numeric(raw[pe_col], errors="coerce"),
    }).dropna(subset=["date", "pe"])
    df["pb"] = float("nan")  # csindex 无 PB，模型会降级

    return df


def _find_col(df, candidates):
    """从 DataFrame 列名中找到第一个匹配的候选列名。"""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def fetch_bond_yield_10y():
    """
    10年国债收益率，多渠道降级:
    1. bond_zh_us_rate (datacenter-web, 1990至今) — 当前可用
    2. bond_china_yield (中债, 2020-2021 已停更) — 仅作回退
    """
    import akshare as ak
    last_err = None

    # 1) 主源: 中美利差接口的 cn_10y（东财 datacenter，验证可用至昨日）
    try:
        raw = ak.bond_zh_us_rate()
        df = pd.DataFrame({
            "date": pd.to_datetime(raw["日期"]),
            "yield_10y": pd.to_numeric(raw["中国国债收益率10年"], errors="coerce"),
        }).dropna(subset=["yield_10y"])
        df = df.sort_values("date").reset_index(drop=True)
        # 校验新鲜度: 数据必须更新到近7天内，否则视为源失效
        if (datetime.now() - df["date"].iloc[-1]).days <= 7:
            logger.info("国债: %d行 %s~%s 最新=%.2f%%",
                        len(df), df["date"].min().strftime("%Y-%m-%d"),
                        df["date"].max().strftime("%Y-%m-%d"), df["yield_10y"].iloc[-1])
            return _v(df, "国债")
        last_err = f"数据停更于{df['date'].iloc[-1].date()}"
    except Exception as e:
        last_err = str(e)[:80]

    # 2) 回退: 中债信息网
    try:
        df = ak.bond_china_yield()
        col = next((c for c in df.columns if "10" in str(c)), None)
        if col is None: raise ValueError("无10年列")
        df = df.rename(columns={"日期":"date", col:"yield_10y"})
        df["date"] = pd.to_datetime(df["date"])
        df["yield_10y"] = pd.to_numeric(df["yield_10y"], errors="coerce")
        df = df[["date","yield_10y"]].dropna()
        logger.info("国债(中债): %d行 %.2f%%", len(df), df["yield_10y"].iloc[-1])
        return _v(df, "国债")
    except Exception as e:
        last_err = str(e)[:80]

    raise RuntimeError(f"国债收益率: 所有数据源失败 ({last_err})")


# ============================================================
# 一键拉取
# ============================================================

def fetch_all_data(force=False):
    logger.info("==== 数据拉取 ====")
    r = {"fetched_at": datetime.now()}

    r["csi300_daily"] = fetch_csi300_daily(force)
    r["csi2000_daily"] = fetch_csi2000_daily(force)

    for code, name in [("159330", "沪深300ETF"), ("159531", "中证2000ETF")]:
        try:
            r[f"etf_{code}"] = fetch_etf_daily(code, name, force)
        except Exception as e:
            logger.warning("ETF%s: %s（使用缓存）", name, str(e)[:80])
            r[f"etf_{code}"] = _load(f"etf_{code}_daily.parquet")

    r["csi300_valuation"] = fetch_valuation("000300", "CSI300")
    r["csi2000_valuation"] = fetch_valuation("932000", "CSI2000")
    r["bond_yield_10y"] = fetch_bond_yield_10y()

    logger.info("==== 完成 ====")
    return r
