"""
回测工具 — 历史周期验证

单独运行: python backtest.py
"""

import pandas as pd
import numpy as np
from datetime import datetime

from data.fetcher import fetch_all_data
from macro.fetcher import fetch_all_macro
from strategy import detect_ice_point, detect_bubble


def run_backtest(start_year=2020, end_year=2026):
    """运行全周期回测。"""
    a = fetch_all_data(force=False)
    macro = fetch_all_macro(force=False)

    # ETF 价格
    idx300 = a['csi300_daily'].copy()
    etf300 = a['etf_159330'].copy()
    m = idx300.merge(etf300[['date','close']].rename(columns={'close':'e'}), on='date', how='inner')
    r300 = (m['e']/m['close']).mean()

    idx2000 = a['csi2000_daily'].copy()
    etf531 = a['etf_159531'].copy()
    nd = etf531[~etf531['date'].isin(idx2000['date'])]
    if not nd.empty:
        ol = idx2000.merge(etf531[['date','close']].rename(columns={'close':'e'}), on='date', how='inner')
        r531 = (ol['e']/ol['close']).mean() if not ol.empty else 500
        nr = pd.DataFrame({'date':nd['date'],'open':nd['open']*r531,'close':nd['close']*r531,
                           'high':nd['high']*r531,'low':nd['low']*r531,'volume':nd['volume']})
        idx2000 = pd.concat([idx2000,nr]).sort_values('date').reset_index(drop=True)

    # 月频扫描
    months = pd.date_range(f'{start_year}-01-01', f'{end_year}-08-01', freq='MS')
    cash, pos300, pos2000 = 100000, 0, 0
    trades, values = [], []

    for dt in months:
        p300 = float(idx300[idx300['date']<=dt]['close'].iloc[-1]) * r300
        p2000_row = etf531[etf531['date']<=dt].tail(1)
        if not p2000_row.empty:
            p2000 = float(p2000_row['close'].iloc[0])
        else:
            idx2_row = idx2000[idx2000['date']<=dt].tail(1)
            p2000 = float(idx2_row['close'].iloc[0]) * r531 if not idx2_row.empty else 0

        ice = detect_ice_point(a, macro, dt)
        bubble = detect_bubble(a, macro, dt)

        # 卖出
        if bubble['is_bubble']:
            for code, pos, price in [('300',pos300,p300),('2000',pos2000,p2000)]:
                if pos > 0:
                    s = pos // 2
                    if s > 0:
                        cash += s * price * 0.9998 - 5
                        if code == '300': pos300 -= s
                        else: pos2000 -= s
                        trades.append({'date':dt,'action':'SELL','code':code,'shares':s,'price':price})

        # 买入（冰点时分批，每次最多用25%现金）
        for code, is_ice, price, pos_var in [
            ('300', ice['csi300'], p300, 'pos300'),
            ('2000', ice['csi2000'], p2000, 'pos2000'),
        ]:
            if is_ice and cash > 5000:
                amount = cash * 0.25
                shares = int(amount / price / 100) * 100
                if shares > 0:
                    cost = shares * price * 1.0002 + 5
                    if cost <= cash:
                        cash -= cost
                        if code == '300': pos300 += shares
                        else: pos2000 += shares
                        trades.append({'date':dt,'action':'BUY','code':code,'shares':shares,'price':price})

        values.append({'date':dt,'total':cash+pos300*p300+pos2000*p2000})

    vals = pd.DataFrame(values)
    tdf = pd.DataFrame(trades) if trades else pd.DataFrame()

    # 逐年统计
    print(f"\n{'='*70}")
    print(f"  回测 {start_year}-{end_year}: 冰点买入 + 泡沫卖出")
    print(f"{'='*70}")
    print(f"  {'年份':<6} {'策略':>8} {'CSI300':>8} {'CSI2000':>8} {'60/40':>8} {'α':>8} {'冰点':>4} {'泡沫':>4}")
    print(f"  {'─'*65}")

    vals['year'] = vals['date'].dt.year
    for year, grp in vals.groupby('year'):
        sv, ev = grp['total'].iloc[0], grp['total'].iloc[-1]
        strat = ev/sv - 1
        # 基准
        f3 = idx300[idx300['date']>=grp['date'].iloc[0]].head(1)
        l3 = idx300[(idx300['date']>=grp['date'].iloc[0])&(idx300['date']<=grp['date'].iloc[-1])].tail(1)
        f2 = idx2000[idx2000['date']>=grp['date'].iloc[0]].head(1)
        l2 = idx2000[(idx2000['date']>=grp['date'].iloc[0])&(idx2000['date']<=grp['date'].iloc[-1])].tail(1)
        r3 = (float(l3['close'].iloc[0])*r300/float(f3['close'].iloc[0])/r300-1) if not f3.empty else 0
        r2 = (float(l2['close'].iloc[0])*r531/float(f2['close'].iloc[0])/r531-1) if not f2.empty else 0
        bm = 0.6*r3 + 0.4*r2
        alpha = strat - bm
        b_count = len(tdf[(tdf['date'].dt.year==year)&(tdf['action']=='BUY')]) if not tdf.empty else 0
        s_count = len(tdf[(tdf['date'].dt.year==year)&(tdf['action']=='SELL')]) if not tdf.empty else 0
        marker = '✅' if alpha > 0 else '  '
        print(f'  {year:<6} {strat:>+7.2%} {r3:>+7.1%}% {r2:>+7.1%}% {bm:>+7.2%} {alpha:>+7.2%} {b_count:>4} {s_count:>4}  {marker}')

    total_ret = vals['total'].iloc[-1]/vals['total'].iloc[0] - 1
    print(f'\n  全周期: {total_ret:+.2%}  |  {len(tdf)}笔交易')
    return vals, tdf


if __name__ == '__main__':
    run_backtest(2020, 2026)
