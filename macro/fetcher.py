"""
宏观数据拉取 — 八大因子

数据源:
- 汇率/美元/日元: currency_boc_safe()
- 中美利差: bond_zh_us_rate()
- 黄金: spot_golden_benchmark_sge()
- CPI/社融/房地产: macro_china_*()
- 辅助: 美联储利率 macro_bank_usa_interest_rate()
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CACHE_AGE = 1  # 天


def _load(fname):
    p = DATA_DIR / fname
    if not p.exists():
        return None
    if (datetime.now() - datetime.fromtimestamp(p.stat().st_mtime)).days >= CACHE_AGE:
        return None
    return pd.read_parquet(p)


def _save(df, fname):
    (DATA_DIR / fname).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(DATA_DIR / fname, index=False)
    logger.info("缓存: %s (%d行)", fname, len(df))


# ═══════════════════════════════════════════════════════════════
# 汇率三兄弟: 美元、人民币、日元
# ═══════════════════════════════════════════════════════════════

def fetch_fx_daily(force=False):
    """USD/CNY + JPY/CNY 日频。"""
    cache = "fx_daily.parquet"
    if not force:
        c = _load(cache)
        if c is not None:
            return c

    import akshare as ak
    raw = ak.currency_boc_safe()
    df = pd.DataFrame({
        "date": pd.to_datetime(raw["日期"]),
        "usd_cny": pd.to_numeric(raw["美元"], errors="coerce") / 100.0,
        "jpy_cny": pd.to_numeric(raw["日元"], errors="coerce") / 100.0,
    }).dropna()
    df = df.sort_values("date").reset_index(drop=True)
    logger.info("汇率: %d行 %s~%s USD=%.4f JPY=%.4f",
                len(df), df["date"].min().date(), df["date"].max().date(),
                df["usd_cny"].iloc[-1], df["jpy_cny"].iloc[-1])
    _save(df, cache)
    return df


# ═══════════════════════════════════════════════════════════════
# 中美利差
# ═══════════════════════════════════════════════════════════════

def fetch_bond_spread(force=False):
    """中美 10 年期利差 = 中国10Y - 美国10Y。"""
    cache = "bond_spread.parquet"
    if not force:
        c = _load(cache)
        if c is not None:
            return c

    import akshare as ak
    raw = ak.bond_zh_us_rate()
    df = pd.DataFrame({
        "date": pd.to_datetime(raw["日期"]),
        "cn_10y": pd.to_numeric(raw["中国国债收益率10年"], errors="coerce"),
        "us_10y": pd.to_numeric(raw["美国国债收益率10年"], errors="coerce"),
        "cn_2y": pd.to_numeric(raw["中国国债收益率2年"], errors="coerce"),
        "us_2y": pd.to_numeric(raw["美国国债收益率2年"], errors="coerce"),
    }).dropna(subset=["cn_10y", "us_10y"])
    df["spread_10y"] = df["cn_10y"] - df["us_10y"]
    df["spread_2y"] = df["cn_2y"] - df["us_2y"]
    df = df.sort_values("date").reset_index(drop=True)
    logger.info("中美利差: %d行 10Y=%.2f%% (CN%.2f-US%.2f)",
                len(df), df["spread_10y"].iloc[-1], df["cn_10y"].iloc[-1], df["us_10y"].iloc[-1])
    _save(df, cache)
    return df


# ═══════════════════════════════════════════════════════════════
# 黄金
# ═══════════════════════════════════════════════════════════════

def fetch_gold(force=False):
    """上海金交所黄金现货（晚盘价）。"""
    cache = "gold_daily.parquet"
    if not force:
        c = _load(cache)
        if c is not None:
            return c

    import akshare as ak
    raw = ak.spot_golden_benchmark_sge()
    df = pd.DataFrame({
        "date": pd.to_datetime(raw["交易时间"]),
        "gold_evening": pd.to_numeric(raw["晚盘价"], errors="coerce"),
        "gold_morning": pd.to_numeric(raw["早盘价"], errors="coerce"),
    }).dropna(subset=["gold_evening"])
    # 用晚盘价作为主力价格
    df["gold_price"] = df["gold_evening"].fillna(df["gold_morning"])
    df = df.sort_values("date").reset_index(drop=True)
    logger.info("黄金: %d行 %s~%s 最新=%.2f",
                len(df), df["date"].min().date(), df["date"].max().date(),
                df["gold_price"].iloc[-1])
    _save(df, cache)
    return df


# ═══════════════════════════════════════════════════════════════
# 中国宏观月度数据
# ═══════════════════════════════════════════════════════════════

def fetch_china_macro(force=False):
    """CPI + 社融 + 房地产 + PMI，月频统一。"""
    cache = "china_macro_monthly.parquet"
    if not force:
        c = _load(cache)
        if c is not None:
            return c

    import akshare as ak

    # --- CPI ---
    cpi = ak.macro_china_cpi_monthly()
    cpi_df = pd.DataFrame({
        "date": pd.to_datetime(cpi["日期"]),
        "cpi_yoy": pd.to_numeric(cpi["今值"], errors="coerce"),
    })

    # --- 社融 ---
    sf = ak.macro_china_new_financial_credit()
    # 日期格式可能是 "2026年06月份" 或 "202606"
    sf_date_str = sf["月份"].astype(str)
    sf_date_str = sf_date_str.str.replace("年", "").str.replace("月份", "").str.replace("月", "")
    sf_df = pd.DataFrame({
        "date": pd.to_datetime(sf_date_str + "01", format="%Y%m%d"),
        "social_finance": pd.to_numeric(sf["当月"], errors="coerce"),
    })

    # --- 房地产 ---
    re = ak.macro_china_real_estate()
    val_col = "最新值" if "最新值" in re.columns else re.columns[1]
    re_df = pd.DataFrame({
        "date": pd.to_datetime(re["日期"]),
        "real_estate": pd.to_numeric(re[val_col], errors="coerce"),
    })

    # --- 70城房价（补充房地产，更新更及时） ---
    try:
        hp = ak.macro_china_new_house_price()
        # 取全国平均（所有城市同比中位数近似）
        hp_monthly = hp.groupby("日期")["新建商品住宅价格指数-同比"].mean().reset_index()
        hp_monthly.columns = ["date", "house_price_index"]
        hp_monthly["date"] = pd.to_datetime(hp_monthly["date"])
    except Exception:
        hp_monthly = pd.DataFrame({"date": [], "house_price_index": []})

    # --- PMI ---
    pmi = ak.macro_china_pmi()
    if "月份" in pmi.columns:
        pmi_date_str = pmi["月份"].astype(str).str.replace("年","").str.replace("月份","").str.replace("月","")
        pmi_df = pd.DataFrame({
            "date": pd.to_datetime(pmi_date_str + "01", format="%Y%m%d"),
            "pmi_manufacturing": pd.to_numeric(pmi["制造业-指数"], errors="coerce"),
        })
    else:
        pmi_df = pd.DataFrame({"date": [], "pmi_manufacturing": []})

    # --- M1/M2 货币供应 ---
    try:
        ms = ak.macro_china_money_supply()
        # 日期格式:"2008年01月份"
        ms_date = ms["月份"].astype(str).str.replace("年","").str.replace("月份","").str.replace("月","")
        ms_df = pd.DataFrame({
            "date": pd.to_datetime(ms_date + "01", format="%Y%m%d"),
            "m1_yoy": pd.to_numeric(ms["货币(M1)-同比增长"], errors="coerce"),
            "m2_yoy": pd.to_numeric(ms["货币和准货币(M2)-同比增长"], errors="coerce"),
            "m1_m2_scissors": pd.to_numeric(ms["货币(M1)-同比增长"], errors="coerce")
                            - pd.to_numeric(ms["货币和准货币(M2)-同比增长"], errors="coerce"),
        })
    except Exception:
        ms_df = pd.DataFrame({"date": [], "m1_yoy": [], "m2_yoy": [], "m1_m2_scissors": []})

    # --- PPI ---
    try:
        ppi = ak.macro_china_ppi_yearly()
        ppi_df = pd.DataFrame({
            "date": pd.to_datetime(ppi["日期"]),
            "ppi_yoy": pd.to_numeric(ppi["今值"], errors="coerce"),
        })
    except Exception:
        ppi_df = pd.DataFrame({"date": [], "ppi_yoy": []})

    # --- 社会消费品零售 ---
    try:
        retail = ak.macro_china_consumer_goods_retail()
        retail_date = retail["月份"].astype(str).str.replace("年","").str.replace("月份","").str.replace("月","")
        retail_df = pd.DataFrame({
            "date": pd.to_datetime(retail_date + "01", format="%Y%m%d"),
            "retail_yoy": pd.to_numeric(retail["同比增长"], errors="coerce"),
        })
    except Exception:
        retail_df = pd.DataFrame({"date": [], "retail_yoy": []})

    # --- 合并（月频 → 日频前向填充在外层处理） ---
    merged = cpi_df.merge(sf_df, on="date", how="outer") \
                   .merge(re_df, on="date", how="outer")
    for extra in [pmi_df, hp_monthly, ms_df, ppi_df, retail_df]:
        if not extra.empty:
            merged = merged.merge(extra, on="date", how="outer")

    merged = merged.sort_values("date").reset_index(drop=True)
    hp_last = merged["house_price_index"].dropna().iloc[-1] if "house_price_index" in merged.columns and not merged["house_price_index"].dropna().empty else 0
    logger.info("中国宏观: %d行 %s~%s CPI=%.1f%% 社融=%.0f亿 PMI=%.1f 房价=%.1f",
                len(merged), merged["date"].min().date(), merged["date"].max().date(),
                merged["cpi_yoy"].dropna().iloc[-1] if not merged["cpi_yoy"].dropna().empty else float('nan'),
                merged["social_finance"].dropna().iloc[-1] if not merged["social_finance"].dropna().empty else float('nan'),
                merged["pmi_manufacturing"].dropna().iloc[-1] if not merged["pmi_manufacturing"].dropna().empty else float('nan'),
                hp_last)
    _save(merged, cache)
    return merged


# ═══════════════════════════════════════════════════════════════
# 美联储利率（辅助）
# ═══════════════════════════════════════════════════════════════

def fetch_fed_rate(force=False):
    """美联储联邦基金利率决议。"""
    cache = "fed_rate.parquet"
    if not force:
        c = _load(cache)
        if c is not None:
            return c

    import akshare as ak
    raw = ak.macro_bank_usa_interest_rate()
    df = pd.DataFrame({
        "date": pd.to_datetime(raw["日期"]),
        "fed_rate": pd.to_numeric(raw["今值"], errors="coerce"),
    }).dropna(subset=["fed_rate"])
    df = df.sort_values("date").reset_index(drop=True)
    logger.info("美联储利率: %d次决议 最新=%.2f%%", len(df), df["fed_rate"].iloc[-1])
    _save(df, cache)
    return df


# ═══════════════════════════════════════════════════════════════
# 一键拉取
# ═══════════════════════════════════════════════════════════════

def _safe_fetch(fetcher, name, force, **kwargs):
    """安全拉取：失败时尝试缓存。"""
    try:
        return fetcher(force=force, **kwargs)
    except Exception as e:
        logger.warning("%s 拉取失败: %s，尝试缓存...", name, str(e)[:80])
        # 强制读缓存（忽略过期）
        if "fx" in name.lower():
            return _load_any("fx_daily.parquet")
        elif "spread" in name.lower():
            return _load_any("bond_spread.parquet")
        elif "gold" in name.lower():
            return _load_any("gold_daily.parquet")
        elif "china" in name.lower():
            return _load_any("china_macro_monthly.parquet")
        elif "fed" in name.lower():
            return _load_any("fed_rate.parquet")
        raise


def _load_any(fname):
    """无论缓存是否过期都加载。"""
    p = DATA_DIR / fname
    if p.exists():
        logger.info("使用缓存: %s", fname)
        return pd.read_parquet(p)
    raise RuntimeError(f"无缓存且拉取失败: {fname}")


def fetch_all_macro(force=False):
    """拉取全部宏观数据，统一日频对齐。"""
    logger.info("==== 宏观数据拉取 ====")

    fx = _safe_fetch(fetch_fx_daily, "汇率", force)
    spread = _safe_fetch(fetch_bond_spread, "利差", force)
    gold = _safe_fetch(fetch_gold, "黄金", force)
    china = _safe_fetch(fetch_china_macro, "中国宏观", force)
    fed = _safe_fetch(fetch_fed_rate, "美联储", force)

    # 基准日期序列（统一 ns 精度）
    df = fx[["date"]].copy().sort_values("date")
    df["date"] = pd.to_datetime(df["date"]).astype("datetime64[ns]")

    def _add_cols(left, right, cols):
        """用 merge_asof 将 cols 追加到 left。统一日期为 datetime64[ns]。"""
        r = right[["date"] + cols].dropna(subset=cols).sort_values("date")
        l = left.copy()
        # 统一日期精度到 ns
        l["date"] = pd.to_datetime(l["date"]).astype("datetime64[ns]")
        r = r.copy()
        r["date"] = pd.to_datetime(r["date"]).astype("datetime64[ns]")
        merged = pd.merge_asof(l, r, on="date", direction="backward")
        for c in cols:
            left[c] = merged[c].values
        return left

    # 逐因子追加到 df
    df = _add_cols(df, fx[["date", "usd_cny", "jpy_cny"]], ["usd_cny", "jpy_cny"])
    df = _add_cols(df, spread[["date", "cn_10y", "us_10y", "cn_2y", "us_2y", "spread_10y", "spread_2y"]],
                   ["cn_10y", "us_10y", "cn_2y", "us_2y", "spread_10y", "spread_2y"])
    df = _add_cols(df, gold[["date", "gold_price"]], ["gold_price"])

    # 月频数据
    for col in ["cpi_yoy", "social_finance", "real_estate", "house_price_index",
                "pmi_manufacturing", "m1_yoy", "m2_yoy", "m1_m2_scissors",
                "ppi_yoy", "retail_yoy"]:
        if col in china.columns:
            sub = china[["date", col]].dropna(subset=[col]).sort_values("date")
            if not sub.empty:
                df = _add_cols(df, sub, [col])

    df = _add_cols(df, fed[["date", "fed_rate"]].dropna(subset=["fed_rate"]).sort_values("date"),
                   ["fed_rate"])

    # 衍生因子（仅当原始列存在时计算）
    if "cn_10y" in df.columns and "cn_2y" in df.columns:
        df["cn_term_spread"] = df["cn_10y"] - df["cn_2y"]
    if "us_10y" in df.columns and "us_2y" in df.columns:
        df["us_term_spread"] = df["us_10y"] - df["us_2y"]

    # 最终清理
    df = df.sort_values("date").reset_index(drop=True)

    logger.info("宏观合并: %d行 %s~%s (%d列)",
                len(df), df["date"].min().date(), df["date"].max().date(),
                len(df.columns))
    # 检查各列覆盖
    for c in df.columns:
        if c == "date":
            continue
        valid = df[c].notna().sum()
        if valid < len(df) * 0.5:
            logger.warning("  ⚠ %s: %d/%d 有效 (%.0f%%)", c, valid, len(df), valid/len(df)*100)

    logger.info("==== 宏观拉取完成 ====")
    return df
