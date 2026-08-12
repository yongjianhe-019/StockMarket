"""
CSI300 专项回测: 2013-2018（验证 2015 泡沫卖出 + 2016 冰点买入）

关键问题:
1. 2015年6月股灾前，泡沫检测能否触发？什么级别？卖多少？
2. 2016年初冰点，买入端能否抓住？
3. 2018年熊市冰点表现？

使用项目实际模型: models/csi300.py (compute_csi300_score) + macro/sell_signal.py (is_bubble)
"""
import json
import sys
sys.path.insert(0, '/Users/hyj/PycharmProjects/StockMarket')

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from models.csi300 import compute_csi300_score
from macro.sell_signal import is_bubble

DATA = Path('/Users/hyj/PycharmProjects/StockMarket/data')

# --- 1. 加载数据 ---
print("=" * 60)
print("  加载数据 (2013-2018 专项回测)")
print("=" * 60)

idx300 = pd.read_parquet(DATA / 'csi300_daily.parquet')
val300 = pd.read_parquet(DATA / 'valuation_000300.parquet')

# 国债收益率: 从 bond_spread 的 cn_10y 重建
bond_spread = pd.read_parquet(DATA / 'bond_spread.parquet')
bond = pd.DataFrame({
    'date': bond_spread['date'],
    'yield_10y': bond_spread['cn_10y'],
}).dropna(subset=['yield_10y']).sort_values('date').reset_index(drop=True)

# 宏观数据: 用缓存合并（上一步刚刷新过）
from macro.fetcher import fetch_all_macro
macro = fetch_all_macro(force=False)

print(f"CSI300: {idx300['date'].min().date()} ~ {idx300['date'].max().date()}")
print(f"估值:   {val300['date'].min().date()} ~ {val300['date'].max().date()}")
print(f"国债:   {bond['date'].min().date()} ~ {bond['date'].max().date()}")
print(f"宏观:   {macro['date'].min().date()} ~ {macro['date'].max().date()}")

# --- 2. 逐月回测 2013-01 ~ 2018-12 ---
start_date = pd.Timestamp('2013-01-01')
end_date = pd.Timestamp('2018-12-01')
months = pd.date_range(start_date, end_date, freq='MS')

cash = 100000.0
units = 0.0  # 指数单位（1单位 = 1点）
trades = []
records = []
last_buy = None
cooldown_days = 90

print(f"\n{'='*60}")
print(f"  逐月扫描: {start_date.date()} ~ {end_date.date()}")
print(f"{'='*60}")

# 记录关键月份详细分数
key_months_detail = []

for dt in months:
    d300 = idx300[idx300['date'] <= dt]
    if d300.empty or len(d300) < 120:
        continue
    price = float(d300['close'].iloc[-1])

    v300 = val300[val300['date'] <= dt]
    b = bond[bond['date'] <= dt]
    m = macro[macro['date'] <= dt]

    # --- 冰点打分 ---
    try:
        score = compute_csi300_score(
            daily=d300,
            valuation=v300 if not v300.empty else None,
            bond_yield_df=b if not b.empty else None,
            macro=m if not m.empty else None,
        )
    except Exception as e:
        print(f"  [{dt.date()}] 打分失败: {e}")
        continue

    # --- 泡沫检测 ---
    try:
        idx_pos = macro['date'].searchsorted(dt)
        bubble = is_bubble(
            macro_df=macro.iloc[:max(idx_pos, 1)],
            csi300_val=val300,
            daily_300=idx300,
        )
    except Exception as e:
        bubble = {'is_bubble': False, 'pe_pct': None}
        print(f"  [{dt.date()}] 泡沫检测失败: {e}")

    total_score = score['total_score']
    pe_pct = bubble.get('pe_pct')

    # --- 关键月份记录详情 ---
    if dt in [pd.Timestamp('2015-05-01'), pd.Timestamp('2015-06-01'), pd.Timestamp('2015-07-01'),
              pd.Timestamp('2016-01-01'), pd.Timestamp('2016-02-01'), pd.Timestamp('2016-03-01'),
              pd.Timestamp('2018-10-01'), pd.Timestamp('2018-11-01'), pd.Timestamp('2018-12-01')]:
        key_months_detail.append({
            'date': str(dt.date()),
            'score': total_score,
            'pe_pct': pe_pct,
            'is_bubble': bubble.get('is_bubble', False),
            'bubble_level': bubble.get('level', ''),
            'signals': bubble.get('signals', {}),
            'reasons': bubble.get('reasons', []),
            'details': score['details'],
        })

    # --- 泡沫卖出 ---
    if bubble.get('is_bubble') and units > 0:
        sell_pct = bubble.get('sell_pct', 0.5)
        sell_units = units * sell_pct
        proceeds = sell_units * price * 0.9998
        cash += proceeds
        units -= sell_units
        trades.append({
            'date': dt, 'action': 'SELL',
            'units': round(sell_units, 2), 'price': round(price, 2),
            'value': round(proceeds, 0),
            'level': bubble.get('level', ''),
            'reason': '; '.join(bubble.get('reasons', [])[:2])
        })
        print(f"  ⚠️ [{dt.date()}] 泡沫卖出! {bubble.get('level')} 卖{sell_pct:.0%} "
              f"(PE分位{pe_pct:.0%}, {len(bubble.get('reasons',[]))}信号)")

    # --- 冰点买入 ---
    if total_score >= 50 and cash > 5000:
        if last_buy is None or (dt - last_buy).days >= cooldown_days:
            amount = cash * 0.25
            buy_units = amount / price
            cost = amount * 1.0002
            if cost <= cash and buy_units > 0:
                cash -= cost
                units += buy_units
                last_buy = dt
                trades.append({
                    'date': dt, 'action': 'BUY',
                    'units': round(buy_units, 2), 'price': round(price, 2),
                    'value': round(cost, 0), 'score': total_score
                })
                print(f"  🧊 [{dt.date()}] 冰点买入! 评分={total_score:.0f} (PE分位{pe_pct:.0%})")

    # 记录月末
    records.append({
        'date': dt, 'total': cash + units * price,
        'cash': cash, 'pos': units * price, 'price': price,
        'score': total_score, 'pe_pct': pe_pct,
    })

vals = pd.DataFrame(records)
tdf = pd.DataFrame(trades) if trades else pd.DataFrame()

# --- 3. 统计 ---
final_total = float(vals['total'].iloc[-1])
total_ret = final_total / 100000 - 1
years = (vals['date'].iloc[-1] - vals['date'].iloc[0]).days / 365.25
annual_ret = ((final_total / 100000) ** (1/years) - 1) if years > 0 else 0
peak = vals['total'].expanding().max()
dd = (vals['total'] - peak) / peak
max_dd = float(dd.min())

# 基准
bm_start_row = idx300[idx300['date'] >= start_date]
bm_start = float(bm_start_row.iloc[0]['close'])
bm_end_row = idx300[idx300['date'] <= vals['date'].iloc[-1]]
bm_end = float(bm_end_row['close'].iloc[-1])
bm_ret = bm_end / bm_start - 1

print(f"\n{'='*70}")
print(f"  🧊 CSI300 专项回测: 2013-01 ~ 2018-12 (5.9年)")
print(f"{'='*70}")
print(f"  策略累计收益: {total_ret:+.2%}")
print(f"  策略年化收益: {annual_ret:+.2%}")
print(f"  策略最大回撤: {max_dd:.2%}")
print(f"  基准 CSI300 B&H: {bm_ret:+.2%}")
print(f"  α: {total_ret - bm_ret:+.2%}")
print(f"  交易: {len(tdf)}笔 (买入{len(tdf[tdf['action']=='BUY']) if not tdf.empty else 0}, "
      f"卖出{len(tdf[tdf['action']=='SELL']) if not tdf.empty else 0})")

# 关键问题1: 2015年泡沫
print(f"\n{'='*70}")
print(f"  关键问题 1: 2015年股灾前，泡沫检测表现")
print(f"{'='*70}")
for kd in key_months_detail:
    if '2015' in kd['date']:
        print(f"\n  [{kd['date']}] 评分={kd['score']:.0f} PE分位={kd['pe_pct']:.0%} "
              f"泡沫={kd['is_bubble']} {kd['bubble_level']}")
        if kd['signals']:
            for cat, sigs in kd['signals'].items():
                if sigs and cat != '触发类别数' and cat != '总信号数':
                    print(f"    {cat}: {sigs}")

# 关键问题2: 2016年冰点
print(f"\n{'='*70}")
print(f"  关键问题 2: 2016年初冰点，买入端表现")
print(f"{'='*70}")
for kd in key_months_detail:
    if '2016' in kd['date']:
        print(f"\n  [{kd['date']}] 评分={kd['score']:.0f} PE分位={kd['pe_pct']:.0%} "
              f"泡沫={kd['is_bubble']}")
        # 打印分维度
        for dim, det in kd['details'].items():
            if isinstance(det, dict) and '得分' in det:
                print(f"    {dim}: {det.get('等级','')} 得分={det['得分']} {det.get('PE分位','')} {det.get('PB分位','')} {det.get('ERP','')} {det.get('ERP分位','')} {det.get('回撤幅度','')}")

# 关键问题3: 2018年冰点
print(f"\n{'='*70}")
print(f"  关键问题 3: 2018年熊市冰点")
print(f"{'='*70}")
for kd in key_months_detail:
    if '2018' in kd['date']:
        print(f"\n  [{kd['date']}] 评分={kd['score']:.0f} PE分位={kd['pe_pct']:.0%}")
        for dim, det in kd['details'].items():
            if isinstance(det, dict) and '得分' in det:
                print(f"    {dim}: {det.get('等级','')} 得分={det['得分']}")

# 逐年
print(f"\n  {'年份':<6} {'策略':>8} {'B&H':>8} {'α':>8} {'买入':>4} {'卖出':>4}")
print(f"  {'─'*48}")
vals['year'] = vals['date'].dt.year
yearly_data = []
for year, grp in vals.groupby('year'):
    sv, ev = grp['total'].iloc[0], grp['total'].iloc[-1]
    strat = ev/sv - 1
    f3 = idx300[idx300['date'] >= grp['date'].iloc[0]]
    l3 = idx300[(idx300['date'] >= grp['date'].iloc[0]) & (idx300['date'] <= grp['date'].iloc[-1])]
    r3 = float(l3['close'].iloc[-1])/float(f3['close'].iloc[0]) - 1 if not f3.empty and not l3.empty else 0
    b_count = len(tdf[(tdf['date'].dt.year==year)&(tdf['action']=='BUY')]) if not tdf.empty else 0
    s_count = len(tdf[(tdf['date'].dt.year==year)&(tdf['action']=='SELL')]) if not tdf.empty else 0
    marker = '✅' if strat - r3 > 0 else '  '
    print(f'  {year:<6} {strat:>+7.2%} {r3:>+7.1%}% {strat-r3:>+7.2%} {b_count:>4} {s_count:>4}  {marker}')
    yearly_data.append({'year': int(year), 'strategy': round(strat, 4), 'csi300': round(r3, 4)})

# --- 4. 交易记录 ---
if not tdf.empty:
    print(f"\n  交易明细:")
    for _, t in tdf.iterrows():
        act = '买入' if t['action']=='BUY' else '卖出'
        extra = f"评分={t['score']}" if t['action']=='BUY' else f"{t.get('level','')}"
        print(f"    {str(t['date'].date())}  {act}  {t['units']}单位 @ {t['price']:.0f}点  "
              f"¥{t['value']:,.0f}  {extra}")

# --- 5. HTML 图表 ---
out_dir = Path('/Users/hyj/PycharmProjects/StockMarket/backtest')
gen_date = datetime.now().strftime('%Y-%m-%d %H:%M')

strat_nav = [round(v/100000, 4) for v in vals['total'].tolist()]
strat_dates = [str(d.date()) for d in vals['date']]
bm_nav = [round(float(idx300[idx300['date']<=d]['close'].iloc[-1])/bm_start, 4) for d in vals['date']]
price_dates = [str(d.date()) for d in vals['date']]
prices = [round(v, 2) for v in vals['price'].tolist()]
scores = [round(v, 1) if not pd.isna(v) else None for v in vals['score'].tolist()]

buy_dates, buy_vals = [], []
sell_dates, sell_vals = [], []
if not tdf.empty:
    for _, t in tdf.iterrows():
        vrow = vals[vals['date'] <= t['date']]
        if vrow.empty: continue
        nav = float(vrow['total'].iloc[-1]) / 100000
        if t['action'] == 'BUY':
            buy_dates.append(str(t['date'].date())); buy_vals.append(nav)
        else:
            sell_dates.append(str(t['date'].date())); sell_vals.append(nav)

# 预计算 JSON（避免 f-string 花括号冲突）
_json_buy = json.dumps({'d': buy_dates, 'v': buy_vals})
_json_sell = json.dumps({'d': sell_dates, 'v': sell_vals})
_json_strat_dates = json.dumps(strat_dates)
_json_strat_nav = json.dumps(strat_nav)
_json_bm_nav = json.dumps(bm_nav)
_json_price_dates = json.dumps(price_dates)
_json_prices = json.dumps(prices)
_json_scores = json.dumps(scores)
_json_yearly = json.dumps(yearly_data)

html = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CSI300 2015泡沫专项测试</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
:root {{
  --bg:#ffffff; --text:#1a1a2e; --text2:#6b7280; --card:#f8f9fa; --border:#e5e7eb;
  --up:#16a34a; --down:#dc2626; --blue:#2563eb; --orange:#ea580c; --green:#16a34a; --red:#dc2626;
  --grid:rgba(128,128,128,0.15);
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    --bg:#0f172a; --text:#e2e8f0; --text2:#94a3b8; --card:#1e293b; --border:#334155;
    --up:#22c55e; --down:#ef4444; --blue:#60a5fa; --orange:#fb923c; --green:#4ade80; --red:#f87171;
    --grid:rgba(148,163,184,0.12);
  }}
}}
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Noto Sans SC",sans-serif;
  background:var(--bg); color:var(--text); padding:40px 24px; max-width:1100px; margin:0 auto; }}
h1 {{ font-size:28px; font-weight:700; margin-bottom:6px; }}
.subtitle {{ color:var(--text2); font-size:14px; margin-bottom:32px; }}
.section {{ background:var(--card); border:1px solid var(--border); border-radius:12px; padding:24px; margin-bottom:28px; }}
.section h2 {{ font-size:18px; font-weight:600; margin-bottom:16px; }}
.stats-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(140px,1fr)); gap:12px; margin-bottom:18px; }}
.stat-card {{ background:var(--bg); border:1px solid var(--border); border-radius:8px; padding:12px 14px; }}
.stat-card .label {{ font-size:11px; color:var(--text2); margin-bottom:4px; }}
.stat-card .value {{ font-size:20px; font-weight:700; }}
.up {{ color:var(--up); }} .down {{ color:var(--down); }}
.plot-box {{ width:100%; height:440px; }}
table {{ width:100%; border-collapse:collapse; font-size:13px; }}
th,td {{ padding:8px 12px; text-align:right; border-bottom:1px solid var(--border); }}
th:first-child,td:first-child {{ text-align:left; }}
th {{ color:var(--text2); font-weight:500; font-size:11px; }}
tr:last-child td {{ border-bottom:none; }}
.buy {{ color:var(--up); font-weight:600; }} .sell {{ color:var(--down); font-weight:600; }}
.footnote {{ color:var(--text2); font-size:12px; margin-top:24px; text-align:center; }}
</style>
</head>
<body>
<h1>🧊 CSI300 专项: 2015 泡沫卖出验证</h1>
<p class="subtitle">2013-01 ~ 2018-12 · 使用项目实际模型 (冰点评分 + 三分类泡沫检测) · 生成于 {gen_date}</p>

<div class="section">
  <h2>策略净值 vs 买入持有</h2>
  <div class="stats-grid">
    <div class="stat-card"><div class="label">策略累计收益</div><div class="value {'up' if total_ret>=0 else 'down'}">{total_ret:+.2%}</div></div>
    <div class="stat-card"><div class="label">年化收益</div><div class="value {'up' if annual_ret>=0 else 'down'}">{annual_ret:+.2%}</div></div>
    <div class="stat-card"><div class="label">最大回撤</div><div class="value down">{max_dd:.2%}</div></div>
    <div class="stat-card"><div class="label">B&H 收益</div><div class="value {'up' if bm_ret>=0 else 'down'}">{bm_ret:+.2%}</div></div>
    <div class="stat-card"><div class="label">α</div><div class="value {'up' if (total_ret-bm_ret)>=0 else 'down'}">{total_ret-bm_ret:+.2%}</div></div>
    <div class="stat-card"><div class="label">交易次数</div><div class="value">{len(tdf)}笔</div></div>
  </div>
  <div id="chartNav" class="plot-box"></div>
</div>

<div class="section">
  <h2>冰点评分走势（≥50触发买入）</h2>
  <div id="chartScore" class="plot-box"></div>
</div>

<div class="section">
  <h2>逐年收益</h2>
  <div id="chartYearly" class="plot-box"></div>
</div>

<div class="section">
  <h2>交易记录</h2>
  <table>
    <thead><tr><th>日期</th><th>操作</th><th>份额</th><th>指数点位</th><th>金额</th><th>备注</th></tr></thead>
    <tbody>
'''

if not tdf.empty:
    for _, t in tdf.iterrows():
        act = 'buy' if t['action']=='BUY' else 'sell'
        act_label = '买入' if t['action']=='BUY' else '卖出'
        note = f"评分={t['score']:.0f}" if t['action']=='BUY' else t.get('level','')
        html += f'''      <tr><td>{str(t['date'].date())}</td>
        <td class="{act}">{act_label}</td><td>{t['units']}</td><td>{t['price']:.0f}</td>
        <td>¥{t['value']:,.0f}</td><td style="font-size:11px;color:var(--text2)">{note}</td></tr>\n'''
else:
    html += '      <tr><td colspan="6">无交易</td></tr>\n'

html += f'''    </tbody>
  </table>
</div>

<p class="footnote">模型: models/csi300.py (7维评分) + macro/sell_signal.py (三分类泡沫) · 不含分红滑点 · 历史不代表未来</p>

<script>
const D = {{
  stratDates: {_json_strat_dates},
  stratNav: {_json_strat_nav},
  bmNav: {_json_bm_nav},
  priceDates: {_json_price_dates},
  prices: {_json_prices},
  scores: {_json_scores},
  buy: {_json_buy},
  sell: {_json_sell},
  yearly: {_json_yearly}
}};

function gc() {{
  const s = getComputedStyle(document.documentElement);
  return {{
    t2: s.getPropertyValue('--text2').trim(),
    blue: s.getPropertyValue('--blue').trim(),
    orange: s.getPropertyValue('--orange').trim(),
    green: s.getPropertyValue('--green').trim(),
    up: s.getPropertyValue('--up').trim(),
    down: s.getPropertyValue('--down').trim(),
    grid: s.getPropertyValue('--grid').trim(),
  }};
}}

function baseLayout(c) {{
  return {{
    margin:{{l:55,r:30,t:10,b:40}}, paper_bgcolor:'transparent', plot_bgcolor:'transparent',
    xaxis:{{showgrid:true,gridcolor:c.grid,gridwidth:1,zeroline:false,tickfont:{{size:12,color:c.t2}}}},
    hovermode:'x unified', showlegend:true,
    legend:{{orientation:'h',y:1.12,font:{{size:12,color:c.t2}}}}, dragmode:false
  }};
}}

function plotNav(c) {{
  const tr = [
    {{x:D.stratDates,y:D.stratNav,type:'scatter',mode:'lines',name:'策略净值',
      line:{{color:c.blue,width:3}},hovertemplate:'策略: <b>%{{y:.4f}}</b><extra></extra>'}},
    {{x:D.stratDates,y:D.bmNav,type:'scatter',mode:'lines',name:'CSI300 买入持有',
      line:{{color:c.orange,width:1.8,dash:'dot'}},hovertemplate:'B&H: <b>%{{y:.4f}}</b><extra></extra>'}},
  ];
  if (D.buy.d.length) tr.push({{x:D.buy.d,y:D.buy.v,type:'scatter',mode:'markers',name:'冰点买入',
    marker:{{color:c.up,size:11,symbol:'triangle-up'}},hovertemplate:'🧊 冰点买入<extra></extra>'}});
  if (D.sell.d.length) tr.push({{x:D.sell.d,y:D.sell.v,type:'scatter',mode:'markers',name:'泡沫卖出',
    marker:{{color:c.down,size:11,symbol:'triangle-down'}},hovertemplate:'⚠️ 泡沫卖出<extra></extra>'}});
  const ly = Object.assign({{}}, baseLayout(c), {{
    yaxis:{{showgrid:true,gridcolor:c.grid,gridwidth:1,zeroline:true,zerolinecolor:c.grid,zerolinewidth:1.5,
      tickformat:'.2f',tickfont:{{size:12,color:c.t2}},title:{{text:'净值',font:{{size:13,color:c.t2}}}}}}
  }});
  Plotly.newPlot('chartNav', tr, ly, {{responsive:true,displayModeBar:false}});
}}

function plotScore(c) {{
  const tr = [
    {{x:D.priceDates,y:D.prices,type:'scatter',mode:'lines',name:'指数点位',yaxis:'y2',
      line:{{color:c.orange,width:1.5}},hovertemplate:'点位: <b>%{{y:.0f}}</b><extra></extra>'}},
    {{x:D.priceDates,y:D.scores,type:'scatter',mode:'lines',name:'冰点评分',
      line:{{color:c.blue,width:2.5}},hovertemplate:'评分: <b>%{{y:.1f}}</b><extra></extra>'}},
  ];
  const ly = Object.assign({{}}, baseLayout(c), {{
    yaxis:{{showgrid:true,gridcolor:c.grid,gridwidth:1,zeroline:true,zerolinecolor:c.grid,zerolinewidth:1.5,
      range:[0,100],tickfont:{{size:12,color:c.t2}},title:{{text:'评分(0-100)',font:{{size:13,color:c.t2}}}}}},
    yaxis2:{{overlaying:'y',side:'right',showgrid:false,tickfont:{{size:12,color:c.t2}},
      title:{{text:'指数点位',font:{{size:13,color:c.t2}}}}}},
    shapes:[{{type:'line',x0:D.priceDates[0],x1:D.priceDates[D.priceDates.length-1],y0:50,y1:50,
      line:{{color:c.green,dash:'dash',width:1.5}}}}]
  }});
  Plotly.newPlot('chartScore', tr, ly, {{responsive:true,displayModeBar:false}});
}}

function plotYearly(c) {{
  const years = D.yearly.map(d=>d.year);
  const tr = [
    {{x:years,y:D.yearly.map(d=>d.strategy),type:'bar',name:'策略',
      marker:{{color:c.blue,opacity:0.85}},
      text:D.yearly.map(d=>(d.strategy>=0?'+':'')+(d.strategy*100).toFixed(1)+'%'),
      textposition:'outside',textfont:{{size:11,color:c.t2}},
      hovertemplate:'策略: <b>%{{y:.2%}}</b><extra></extra>'}},
    {{x:years,y:D.yearly.map(d=>d.csi300),type:'bar',name:'CSI300 B&H',
      marker:{{color:c.orange,opacity:0.7}},
      hovertemplate:'B&H: <b>%{{y:.2%}}</b><extra></extra>'}},
  ];
  const ly = Object.assign({{}}, baseLayout(c), {{
    barmode:'group',
    yaxis:{{showgrid:true,gridcolor:c.grid,gridwidth:1,zeroline:true,zerolinecolor:c.grid,zerolinewidth:1.5,
      tickformat:'.0%',tickfont:{{size:12,color:c.t2}},title:{{text:'年度收益率',font:{{size:13,color:c.t2}}}}}}
  }});
  Plotly.newPlot('chartYearly', tr, ly, {{responsive:true,displayModeBar:false}});
}}

const c = gc();
plotNav(c); plotScore(c); plotYearly(c);
window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => {{
  Plotly.purge('chartNav'); Plotly.purge('chartScore'); Plotly.purge('chartYearly');
  const nc = gc(); plotNav(nc); plotScore(nc); plotYearly(nc);
}});
</script>
</body>
</html>'''

out_path = out_dir / 'csi300_2015_bubble_test.html'
out_path.write_text(html, encoding='utf-8')
print(f"\n✅ 图表已保存: {out_path}")
print(f"   open {out_path}")
