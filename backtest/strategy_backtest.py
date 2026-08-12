"""
策略回测：冰点买入 + 泡沫卖出（使用项目实际量化模型）

- CSI300: 估值驱动冰点模型（score≥50 → 买入）
- CSI2000: 流动性+动量冰点模型（score≥50 → 买入）
- 泡沫检测: PE>60%分位 + 三分类确认 ≥2类 → 卖出
- 月频信号，每次冰点最多用25%现金买入
- 使用指数点位追踪，不影响相对收益率
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

# --- 1. 加载数据 ---
from data.fetcher import fetch_all_data
from macro.fetcher import fetch_all_macro
from strategy import detect_ice_point, detect_bubble

print("=" * 60)
print("  加载策略数据...")
print("=" * 60)

a = fetch_all_data(force=True)
macro = fetch_all_macro(force=True)

# --- 2. 准备价格数据 ---
# CSI300: 直接用指数点位（用 fetch_all_data 拉取的缓存，东财失败时自动走缓存）
idx300 = a['csi300_daily'].copy()
print(f"CSI300 指数: {len(idx300)}行, {idx300['date'].min().date()} ~ {idx300['date'].max().date()}")

# CSI2000: 用 full 文件（csindex + ETF 反推合成）
full_path = Path('/Users/hyj/PycharmProjects/StockMarket/data/csi2000_daily_full.parquet')
if full_path.exists():
    idx2000 = pd.read_parquet(full_path)
    print(f"CSI2000 full: {len(idx2000)}行, {idx2000['date'].min().date()} ~ {idx2000['date'].max().date()}")
else:
    idx2000 = a['csi2000_daily'].copy()
    print(f"CSI2000: {len(idx2000)}行, {idx2000['date'].min().date()} ~ {idx2000['date'].max().date()}")

# ETF 用于价格换算（指数点到实际交易价格）
# ETF 净值 ≈ 指数点位 / 1000（粗略），精确比例从缓存计算
etf300_cache = a.get('etf_159330')
if etf300_cache is not None and not etf300_cache.empty:
    m = idx300.merge(etf300_cache[['date','close']].rename(columns={'close':'e'}), on='date', how='inner')
    r300 = (m['e'] / m['close']).median() if not m.empty else 1/1000
else:
    r300 = 1/1000
print(f"CSI300 价格因子 (ETF/Index): {r300:.6f}")

# CSI2000 价格：直接用指数点位的 1/1000 作为参考价
r2000 = 1/1000

# --- 3. 逐月回测 ---
start_date = pd.Timestamp('2015-01-01')
end_date = pd.Timestamp('2026-08-01')

months = pd.date_range(start_date, end_date, freq='MS')

cash = 100000.0
# 持仓用 "指数单位" 追踪：1单位 = 买入时按指数换算
pos300_units = 0.0   # CSI300 指数单位
pos2000_units = 0.0  # CSI2000 指数单位

trades = []
monthly_records = []

# 防止同一冰点反复买入
last_buy_300 = None
last_buy_2000 = None
cooldown_days = 90  # 同一标的冰点买入冷却期

print(f"\n{'='*60}")
print(f"  开始逐月扫描: {start_date.date()} ~ {end_date.date()}")
print(f"  共 {len(months)} 个月")
print(f"{'='*60}")

signal_count = {'buy_300': 0, 'buy_2000': 0, 'sell': 0, 'skip_data': 0}

for i, dt in enumerate(months):
    # 获取当日指数点位
    p300_rows = idx300[idx300['date'] <= dt]
    p2000_rows = idx2000[idx2000['date'] <= dt]
    if p300_rows.empty:
        signal_count['skip_data'] += 1
        continue

    p300_idx = float(p300_rows['close'].iloc[-1])

    if p2000_rows.empty:
        p2000_idx = 0
    else:
        p2000_idx = float(p2000_rows['close'].iloc[-1])

    # 检测信号
    try:
        ice = detect_ice_point(a, macro, dt)
        bubble = detect_bubble(a, macro, dt)
    except Exception as e:
        signal_count['skip_data'] += 1
        continue

    # --- 泡沫卖出 ---
    if bubble['is_bubble']:
        sell_pct = bubble.get('sell_pct', 0.5)
        for code, units, idx_price, r in [
            ('300', pos300_units, p300_idx, r300),
            ('2000', pos2000_units, p2000_idx, r2000),
        ]:
            if units > 0 and idx_price > 0:
                price = idx_price * r
                sell_units = units * sell_pct
                proceeds = sell_units * idx_price * r * 0.9998  # 扣除手续费
                cash += proceeds
                if code == '300':
                    pos300_units -= sell_units
                else:
                    pos2000_units -= sell_units
                trades.append({
                    'date': dt, 'action': 'SELL', 'code': code,
                    'units': round(sell_units, 2), 'price': round(price, 4),
                    'value': round(proceeds, 2),
                    'reason': '; '.join(bubble.get('reasons', [])[:2])
                })
                signal_count['sell'] += 1

    # --- 冰点买入 ---
    for code, is_ice, idx_price, r, last_buy_ref, units_ref in [
        ('300', ice['csi300'], p300_idx, r300, 'last_buy_300', 'pos300_units'),
        ('2000', ice['csi2000'], p2000_idx, r2000, 'last_buy_2000', 'pos2000_units'),
    ]:
        if is_ice and cash > 5000 and idx_price > 0:
            # 检查冷却期
            lb = last_buy_300 if code == '300' else last_buy_2000
            if lb is not None and (dt - lb).days < cooldown_days:
                continue

            # 每次用 25% 现金买入
            amount = cash * 0.25
            price = idx_price * r
            units = amount / price
            cost = amount * 1.0002  # 含手续费

            if cost <= cash and units > 0:
                cash -= cost
                if code == '300':
                    pos300_units += units
                    last_buy_300 = dt
                else:
                    pos2000_units += units
                    last_buy_2000 = dt

                score_key = f'score_{code}'
                trades.append({
                    'date': dt, 'action': 'BUY', 'code': code,
                    'units': round(units, 2), 'price': round(price, 4),
                    'value': round(cost, 2),
                    'score': ice.get(score_key, 0)
                })
                if code == '300':
                    signal_count['buy_300'] += 1
                else:
                    signal_count['buy_2000'] += 1

    # 记录月末净值
    total = cash + pos300_units * p300_idx * r300 + pos2000_units * p2000_idx * r2000
    monthly_records.append({
        'date': dt,
        'total': total,
        'cash': cash,
        'pos300_val': pos300_units * p300_idx * r300,
        'pos2000_val': pos2000_units * p2000_idx * r2000,
        'p300': p300_idx,
        'p2000': p2000_idx,
    })

    # 进度
    if (i+1) % 24 == 0:
        print(f"  进度: {i+1}/{len(months)} ({dt.date()})  净值: {total:.0f}  信号: 买300={signal_count['buy_300']} 买2000={signal_count['buy_2000']} 卖={signal_count['sell']}")

vals = pd.DataFrame(monthly_records)
tdf = pd.DataFrame(trades) if trades else pd.DataFrame()

print(f"\n  扫描完成! 买入信号: CSI300={signal_count['buy_300']}, CSI2000={signal_count['buy_2000']}, 卖出={signal_count['sell']}, 跳过={signal_count['skip_data']}")

# --- 4. 统计 ---
if vals.empty:
    print("错误: 无有效回测数据")
    sys.exit(1)

final_total = float(vals['total'].iloc[-1])
total_ret = final_total / 100000 - 1
years = (vals['date'].iloc[-1] - vals['date'].iloc[0]).days / 365.25
annual_ret = ((final_total / 100000) ** (1/years) - 1) if years > 0 else 0

peak = vals['total'].expanding().max()
dd = (vals['total'] - peak) / peak
max_dd = float(dd.min())

# 基准
start_300_row = idx300[idx300['date'] >= start_date]
bm300_start = float(start_300_row.iloc[0]['close']) if not start_300_row.empty else 1
bm300_end_row = idx300[idx300['date'] <= vals['date'].iloc[-1]]
bm300_end = float(bm300_end_row['close'].iloc[-1]) if not bm300_end_row.empty else bm300_start
bm300_ret = bm300_end / bm300_start - 1

start_2000_row = idx2000[idx2000['date'] >= idx2000['date'].min()]
bm2000_start = float(start_2000_row.iloc[0]['close']) if not start_2000_row.empty else 1
bm2000_end_row = idx2000[idx2000['date'] <= vals['date'].iloc[-1]]
bm2000_end = float(bm2000_end_row['close'].iloc[-1]) if not bm2000_end_row.empty else bm2000_start
bm2000_ret = bm2000_end / bm2000_start - 1
bm6040_ret = 0.6 * bm300_ret + 0.4 * bm2000_ret

# --- 5. 输出 ---
print(f"\n{'='*70}")
print(f"  🧊 策略回测: 冰点买入 + 泡沫卖出")
print(f"     回测区间: {vals['date'].iloc[0].date()} ~ {vals['date'].iloc[-1].date()} ({years:.1f}年)")
print(f"{'='*70}")
print(f"  策略累计收益: {total_ret:+.2%}")
print(f"  策略年化收益: {annual_ret:+.2%}")
print(f"  策略最大回撤: {max_dd:.2%}")
print(f"  基准 CSI300 B&H: {bm300_ret:+.2%}")
print(f"  基准 CSI2000 B&H: {bm2000_ret:+.2%}")
print(f"  基准 60/40 B&H: {bm6040_ret:+.2%}")
print(f"  α vs CSI300: {total_ret - bm300_ret:+.2%}")
print(f"  α vs 60/40:  {total_ret - bm6040_ret:+.2%}")
print(f"  最终现金: ¥{cash:,.0f}")
print(f"  交易次数: {len(tdf)}笔 (买入{len(tdf[tdf['action']=='BUY']) if not tdf.empty else 0}, 卖出{len(tdf[tdf['action']=='SELL']) if not tdf.empty else 0})")

# 逐年
print(f"\n  {'年份':<6} {'策略':>8} {'CSI300':>8} {'CSI2000':>8} {'60/40':>8} {'α':>8} {'买入':>4} {'卖出':>4}")
print(f"  {'─'*65}")

vals['year'] = vals['date'].dt.year
yearly_data = []
for year, grp in vals.groupby('year'):
    sv, ev = grp['total'].iloc[0], grp['total'].iloc[-1]
    strat = ev/sv - 1

    yr_start = grp['date'].iloc[0]
    yr_end = grp['date'].iloc[-1]

    f3 = idx300[(idx300['date'] >= yr_start)]
    l3 = idx300[(idx300['date'] >= yr_start) & (idx300['date'] <= yr_end)]
    r3 = float(l3['close'].iloc[-1]) / float(f3['close'].iloc[0]) - 1 if not f3.empty and not l3.empty else 0

    f2 = idx2000[(idx2000['date'] >= yr_start)]
    l2 = idx2000[(idx2000['date'] >= yr_start) & (idx2000['date'] <= yr_end)]
    r2 = float(l2['close'].iloc[-1]) / float(f2['close'].iloc[0]) - 1 if not f2.empty and not l2.empty else 0

    bm = 0.6*r3 + 0.4*r2
    alpha = strat - bm
    b_count = len(tdf[(tdf['date'].dt.year == year) & (tdf['action'] == 'BUY')]) if not tdf.empty else 0
    s_count = len(tdf[(tdf['date'].dt.year == year) & (tdf['action'] == 'SELL')]) if not tdf.empty else 0
    marker = '✅' if alpha > 0 else '  '
    print(f'  {year:<6} {strat:>+7.2%} {r3:>+7.1%}% {r2:>+7.1%}% {bm:>+7.2%} {alpha:>+7.2%} {b_count:>4} {s_count:>4}  {marker}')

    yearly_data.append({
        'year': int(year), 'strategy': round(strat, 4),
        'csi300': round(r3, 4), 'csi2000': round(r2, 4)
    })

# --- 6. 生成 HTML ---
out_dir = Path('/Users/hyj/PycharmProjects/StockMarket/backtest')
out_dir.mkdir(parents=True, exist_ok=True)
gen_date = datetime.now().strftime('%Y-%m-%d %H:%M')

# 净值数据
strat_nav = [round(v/100000, 4) for v in vals['total'].tolist()]
strat_dates = [str(d.date()) for d in vals['date']]

# 基准净值
def get_nav_at(df, dt):
    row = df[df['date'] <= dt]
    if row.empty: return None
    base = float(df.iloc[0]['close'])
    return round(float(row['close'].iloc[-1]) / base, 4)

bm300_nav = [get_nav_at(idx300, dt) for dt in vals['date']]
bm2000_nav = [get_nav_at(idx2000, dt) for dt in vals['date']]

# 交易标记
buy_dates_300, buy_vals_300 = [], []
buy_dates_2000, buy_vals_2000 = [], []
sell_dates, sell_vals = [], []
if not tdf.empty:
    for _, t in tdf.iterrows():
        vrow = vals[vals['date'] <= t['date']]
        if vrow.empty: continue
        nav = float(vrow['total'].iloc[-1]) / 100000
        if t['action'] == 'BUY':
            if t['code'] == '300':
                buy_dates_300.append(str(t['date'].date()))
                buy_vals_300.append(nav)
            else:
                buy_dates_2000.append(str(t['date'].date()))
                buy_vals_2000.append(nav)
        else:
            sell_dates.append(str(t['date'].date()))
            sell_vals.append(nav)

# HTML 模板
html = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>策略回测：冰点买入 + 泡沫卖出</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
:root {{
  --bg: #ffffff; --text: #1a1a2e; --text2: #6b7280;
  --card: #f8f9fa; --border: #e5e7eb;
  --up: #16a34a; --down: #dc2626;
  --blue: #2563eb; --blueFill: rgba(37,99,235,0.08);
  --orange: #ea580c; --orangeFill: rgba(234,88,12,0.08);
  --green: #16a34a; --red: #dc2626;
  --grid: rgba(128,128,128,0.15);
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    --bg: #0f172a; --text: #e2e8f0; --text2: #94a3b8;
    --card: #1e293b; --border: #334155;
    --up: #22c55e; --down: #ef4444;
    --blue: #60a5fa; --blueFill: rgba(96,165,250,0.10);
    --orange: #fb923c; --orangeFill: rgba(251,146,60,0.10);
    --green: #4ade80; --red: #f87171;
    --grid: rgba(148,163,184,0.12);
  }}
}}
:root[data-theme="dark"] {{
  --bg: #0f172a; --text: #e2e8f0; --text2: #94a3b8;
  --card: #1e293b; --border: #334155;
  --up: #22c55e; --down: #ef4444;
  --blue: #60a5fa; --blueFill: rgba(96,165,250,0.10);
  --orange: #fb923c; --orangeFill: rgba(251,146,60,0.10);
  --green: #4ade80; --red: #f87171;
  --grid: rgba(148,163,184,0.12);
}}
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Noto Sans SC", sans-serif;
  background: var(--bg); color: var(--text);
  padding: 40px 24px; max-width: 1100px; margin: 0 auto;
}}
h1 {{ font-size: 28px; font-weight: 700; margin-bottom: 6px; }}
.subtitle {{ color: var(--text2); font-size: 14px; margin-bottom: 32px; }}
.section {{
  background: var(--card); border: 1px solid var(--border);
  border-radius: 12px; padding: 24px; margin-bottom: 28px;
}}
.section h2 {{ font-size: 18px; font-weight: 600; margin-bottom: 16px; }}
.stats-grid {{
  display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
  gap: 12px; margin-bottom: 18px;
}}
.stat-card {{
  background: var(--bg); border: 1px solid var(--border);
  border-radius: 8px; padding: 12px 14px;
}}
.stat-card .label {{ font-size: 11px; color: var(--text2); margin-bottom: 4px; }}
.stat-card .value {{ font-size: 20px; font-weight: 700; }}
.up {{ color: var(--up); }} .down {{ color: var(--down); }}
.plot-box {{ width: 100%; height: 480px; }}
table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
th, td {{ padding: 8px 12px; text-align: right; border-bottom: 1px solid var(--border); }}
th:first-child, td:first-child {{ text-align: left; }}
th {{ color: var(--text2); font-weight: 500; font-size: 11px; }}
tr:last-child td {{ border-bottom: none; }}
.buy {{ color: var(--up); font-weight: 600; }}
.sell {{ color: var(--down); font-weight: 600; }}
.footnote {{ color: var(--text2); font-size: 12px; margin-top: 24px; text-align: center; }}
</style>
</head>
<body>

<h1>🧊 冰点买入 + 泡沫卖出 · 策略回测</h1>
<p class="subtitle">
  CSI300估值驱动 + CSI2000流动性动量 · 月频信号 · 冰点score≥50买入 · 泡沫≥2类确认卖出
  <br>回测区间: {vals["date"].iloc[0].date()} ~ {vals["date"].iloc[-1].date()} · 生成于 {gen_date}
</p>

<!-- 净值图 -->
<div class="section">
  <h2>策略净值 vs 基准指数（买入持有）</h2>
  <div class="stats-grid">
    <div class="stat-card"><div class="label">策略累计收益</div><div class="value {'up' if total_ret>=0 else 'down'}">{total_ret:+.2%}</div></div>
    <div class="stat-card"><div class="label">策略年化收益</div><div class="value {'up' if annual_ret>=0 else 'down'}">{annual_ret:+.2%}</div></div>
    <div class="stat-card"><div class="label">最大回撤</div><div class="value down">{max_dd:.2%}</div></div>
    <div class="stat-card"><div class="label">α vs CSI300</div><div class="value {'up' if (total_ret-bm300_ret)>=0 else 'down'}">{total_ret-bm300_ret:+.2%}</div></div>
    <div class="stat-card"><div class="label">α vs 60/40</div><div class="value {'up' if (total_ret-bm6040_ret)>=0 else 'down'}">{total_ret-bm6040_ret:+.2%}</div></div>
    <div class="stat-card"><div class="label">交易次数</div><div class="value">{len(tdf)}笔</div></div>
  </div>
  <div id="chartNav" class="plot-box"></div>
</div>

<!-- 逐年对比 -->
<div class="section">
  <h2>逐年收益对比</h2>
  <div id="chartYearly" class="plot-box"></div>
</div>

<!-- 对比表 -->
<div class="section">
  <h2>策略 vs 基准 汇总对比</h2>
  <table>
    <thead><tr><th>指标</th><th>策略</th><th>CSI300 B&H</th><th>CSI2000 B&H</th><th>60/40 B&H</th></tr></thead>
    <tbody>
      <tr><td>累计收益率</td><td style="color:{'var(--up)' if total_ret>=0 else 'var(--down)'}">{total_ret:+.2%}</td><td style="color:{'var(--up)' if bm300_ret>=0 else 'var(--down)'}">{bm300_ret:+.2%}</td><td style="color:{'var(--up)' if bm2000_ret>=0 else 'var(--down)'}">{bm2000_ret:+.2%}</td><td style="color:{'var(--up)' if bm6040_ret>=0 else 'var(--down)'}">{bm6040_ret:+.2%}</td></tr>
      <tr><td>年化收益率</td><td>{annual_ret:+.2%}</td><td colspan="3">—</td></tr>
      <tr><td>最大回撤</td><td>{max_dd:.2%}</td><td colspan="3">—</td></tr>
      <tr><td>超额收益(α)</td><td>—</td><td>{total_ret-bm300_ret:+.2%}</td><td>{total_ret-bm2000_ret:+.2%}</td><td>{total_ret-bm6040_ret:+.2%}</td></tr>
      <tr><td>买入/卖出</td><td>{signal_count["buy_300"]+signal_count["buy_2000"]}/{signal_count["sell"]}笔</td><td colspan="3">1/0笔</td></tr>
    </tbody>
  </table>
</div>

<!-- 交易记录 -->
<div class="section">
  <h2>交易记录</h2>
  <table>
    <thead><tr><th>日期</th><th>操作</th><th>标的</th><th>份额</th><th>价格</th><th>金额</th><th>备注</th></tr></thead>
    <tbody>
'''

if not tdf.empty:
    for _, t in tdf.iterrows():
        act = 'buy' if t['action'] == 'BUY' else 'sell'
        act_label = '买入' if t['action'] == 'BUY' else '卖出'
        code_label = '沪深300' if t['code'] == '300' else '中证2000'
        note = ''
        if t['action'] == 'BUY' and 'score' in t:
            note = f"评分:{t['score']}"
        elif t['action'] == 'SELL' and 'reason' in t:
            note = str(t['reason'])[:50]
        html += f'''      <tr>
        <td>{str(t['date'].date())}</td>
        <td class="{act}">{act_label}</td>
        <td>{code_label}</td>
        <td>{t['units']}</td>
        <td>¥{t['price']:.4f}</td>
        <td>¥{t['value']:,.0f}</td>
        <td style="font-size:11px;color:var(--text2)">{note}</td>
      </tr>\n'''
else:
    html += '      <tr><td colspan="7">暂无交易记录</td></tr>\n'

html += f'''    </tbody>
  </table>
</div>

<p class="footnote">
  策略逻辑：冰点检测(score≥50) → 每次用25%现金买入 · 泡沫确认(≥2类触发) → 卖出50%持仓 · 同标的买入冷却90天
  <br>不含分红和滑点 · 历史表现不代表未来收益 · 数据来源: akshare/csindex/Sina
</p>

<script>
const D = {{
  stratDates: {json.dumps(strat_dates)},
  stratNav: {json.dumps(strat_nav)},
  bm300Nav: {json.dumps(bm300_nav)},
  bm2000Nav: {json.dumps(bm2000_nav)},
  buy300: {json.dumps({'d': buy_dates_300, 'v': buy_vals_300})},
  buy2000: {json.dumps({'d': buy_dates_2000, 'v': buy_vals_2000})},
  sell: {json.dumps({'d': sell_dates, 'v': sell_vals})},
  yearly: {json.dumps(yearly_data)}
}};

function gc() {{
  const s = getComputedStyle(document.documentElement);
  return {{
    t2: s.getPropertyValue('--text2').trim(),
    blue: s.getPropertyValue('--blue').trim(),
    blueFill: s.getPropertyValue('--blueFill').trim(),
    orange: s.getPropertyValue('--orange').trim(),
    green: s.getPropertyValue('--green').trim(),
    red: s.getPropertyValue('--red').trim(),
    up: s.getPropertyValue('--up').trim(),
    down: s.getPropertyValue('--down').trim(),
    grid: s.getPropertyValue('--grid').trim(),
  }};
}}

function baseLayout(c) {{
  return {{
    margin: {{l:55, r:30, t:10, b:40}},
    paper_bgcolor:'transparent', plot_bgcolor:'transparent',
    xaxis: {{showgrid:true, gridcolor:c.grid, gridwidth:1, zeroline:false, tickfont:{{size:12,color:c.t2}}}},
    hovermode:'x unified', showlegend:true,
    legend: {{orientation:'h', y:1.12, font:{{size:12,color:c.t2}}}},
    dragmode: false
  }};
}}

function plotNav(c) {{
  const tr = [
    {{x:D.stratDates, y:D.stratNav, type:'scatter', mode:'lines', name:'策略净值',
      line:{{color:c.blue, width:3}},
      hovertemplate:'策略: <b>%{{y:.4f}}</b><extra></extra>'}},
    {{x:D.stratDates, y:D.bm300Nav, type:'scatter', mode:'lines', name:'沪深300 B&H',
      line:{{color:c.orange, width:1.8, dash:'dot'}},
      hovertemplate:'CSI300: <b>%{{y:.4f}}</b><extra></extra>'}},
    {{x:D.stratDates, y:D.bm2000Nav, type:'scatter', mode:'lines', name:'中证2000 B&H',
      line:{{color:c.green, width:1.8, dash:'dot'}},
      hovertemplate:'CSI2000: <b>%{{y:.4f}}</b><extra></extra>'}},
  ];
  if (D.buy300.d.length) tr.push({{
    x:D.buy300.d, y:D.buy300.v, type:'scatter', mode:'markers', name:'买入(CSI300)',
    marker:{{color:c.up, size:10, symbol:'triangle-up'}},
    hovertemplate:'🧊 买入 CSI300<extra></extra>'}});
  if (D.buy2000.d.length) tr.push({{
    x:D.buy2000.d, y:D.buy2000.v, type:'scatter', mode:'markers', name:'买入(CSI2000)',
    marker:{{color:c.up, size:10, symbol:'triangle-up', opacity:0.7}},
    hovertemplate:'🧊 买入 CSI2000<extra></extra>'}});
  if (D.sell.d.length) tr.push({{
    x:D.sell.d, y:D.sell.v, type:'scatter', mode:'markers', name:'泡沫卖出',
    marker:{{color:c.down, size:10, symbol:'triangle-down'}},
    hovertemplate:'⚠️ 泡沫卖出<extra></extra>'}});

  const ly = Object.assign({{}}, baseLayout(c), {{
    yaxis: {{showgrid:true, gridcolor:c.grid, gridwidth:1, zeroline:true, zerolinecolor:c.grid, zerolinewidth:1.5,
      tickformat:'.2f', tickfont:{{size:12,color:c.t2}},
      title:{{text:'净值 (初始=1.00)', font:{{size:13,color:c.t2}}}}}}
  }});
  Plotly.newPlot('chartNav', tr, ly, {{responsive:true, displayModeBar:false}});
}}

function plotYearly(c) {{
  const years = D.yearly.map(d => d.year);
  const tr = [
    {{x:years, y:D.yearly.map(d=>d.strategy), type:'bar', name:'策略',
      marker:{{color:c.blue, opacity:0.85}},
      text:D.yearly.map(d=>(d.strategy>=0?'+':'')+(d.strategy*100).toFixed(1)+'%'),
      textposition:'outside', textfont:{{size:11,color:c.t2}},
      hovertemplate:'策略: <b>%{{y:.2%}}</b><extra></extra>'}},
    {{x:years, y:D.yearly.map(d=>d.csi300), type:'bar', name:'沪深300',
      marker:{{color:c.orange, opacity:0.7}},
      hovertemplate:'CSI300: <b>%{{y:.2%}}</b><extra></extra>'}},
    {{x:years, y:D.yearly.map(d=>d.csi2000), type:'bar', name:'中证2000',
      marker:{{color:c.green, opacity:0.7}},
      hovertemplate:'CSI2000: <b>%{{y:.2%}}</b><extra></extra>'}},
  ];
  const ly = Object.assign({{}}, baseLayout(c), {{
    barmode:'group',
    yaxis: {{showgrid:true, gridcolor:c.grid, gridwidth:1, zeroline:true, zerolinecolor:c.grid, zerolinewidth:1.5,
      tickformat:'.0%', tickfont:{{size:12,color:c.t2}},
      title:{{text:'年度收益率', font:{{size:13,color:c.t2}}}}}}
  }});
  Plotly.newPlot('chartYearly', tr, ly, {{responsive:true, displayModeBar:false}});
}}

const c = gc();
plotNav(c); plotYearly(c);
window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => {{
  Plotly.purge('chartNav'); Plotly.purge('chartYearly');
  const nc = gc(); plotNav(nc); plotYearly(nc);
}});
</script>
</body>
</html>'''

out_path = out_dir / 'strategy_backtest.html'
out_path.write_text(html, encoding='utf-8')
print(f"\n✅ 图表已保存: {out_path}")
print(f"   open {out_path}")
