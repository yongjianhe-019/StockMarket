"""
CSI300 vs CSI2000 指数投资收益率回测对比

- CSI300: 2015年至今
- CSI2000: 从最早可用数据开始
- 分别买入持有，画两张图
"""
import json
import sys
sys.path.insert(0, '/Users/hyj/PycharmProjects/StockMarket')

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# --- 1. 拉取最新数据 ---
from data.fetcher import fetch_csi300_daily

print("拉取 CSI300 数据...")
df300 = fetch_csi300_daily(force=True)
print(f"  CSI300: {len(df300)}行, {df300['date'].min().date()} ~ {df300['date'].max().date()}")

# CSI2000: 使用 full 文件（csindex 历史 + ETF 反推合成最新数据）
# csindex 接口只有到 2024-06, 东财接口不稳定，full 文件已合并 ETF 反推数据到最新
print("加载 CSI2000 数据 (full, 含 ETF 反推)...")
full_path = Path('/Users/hyj/PycharmProjects/StockMarket/data/csi2000_daily_full.parquet')
if full_path.exists():
    df2000 = pd.read_parquet(full_path)
    print(f"  CSI2000: {len(df2000)}行, {df2000['date'].min().date()} ~ {df2000['date'].max().date()}")
else:
    from data.fetcher import fetch_csi2000_daily
    df2000 = fetch_csi2000_daily(force=True)
    print(f"  CSI2000: {len(df2000)}行, {df2000['date'].min().date()} ~ {df2000['date'].max().date()}")

# --- 2. 过滤时间范围 ---
start_300 = pd.Timestamp('2015-01-01')
df300 = df300[df300['date'] >= start_300].copy()
df300 = df300.sort_values('date').reset_index(drop=True)

df2000 = df2000.sort_values('date').reset_index(drop=True)

# --- 3. 计算买入持有收益率 ---
def calc_cumulative_return(df):
    base = float(df['close'].iloc[0])
    df = df.copy()
    df['cum_return'] = (df['close'] / base - 1) * 100
    df['nav'] = df['close'] / base
    return df, base

df300, base300 = calc_cumulative_return(df300)
df2000, base2000 = calc_cumulative_return(df2000)

# --- 4. 统计摘要 ---
def compute_stats(df, name, base):
    first_date = df['date'].iloc[0]
    last_date = df['date'].iloc[-1]
    first_close = base
    last_close = float(df['close'].iloc[-1])
    cum_ret = float(df['cum_return'].iloc[-1])
    days = len(df)
    years = (last_date - first_date).days / 365.25
    annual_ret = ((last_close / first_close) ** (1 / years) - 1) * 100 if years > 0 else 0

    peak = df['nav'].expanding().max()
    dd = (df['nav'] - peak) / peak * 100
    max_dd = float(dd.min())
    max_ret = float(df['cum_return'].max())
    min_ret = float(df['cum_return'].min())

    print(f"\n{'='*60}")
    print(f"  {name} 买入持有回测")
    print(f"{'='*60}")
    print(f"  起止日期: {first_date.date()} ~ {last_date.date()}")
    print(f"  起始点位: {first_close:.2f}")
    print(f"  最新点位: {last_close:.2f}")
    print(f"  累计收益: {cum_ret:+.2f}%")
    print(f"  年化收益: {annual_ret:+.2f}%")
    print(f"  最高收益: {max_ret:+.2f}%")
    print(f"  最低收益: {min_ret:+.2f}%")
    print(f"  最大回撤: {max_dd:.2f}%")
    print(f"  交易天数: {days}")
    print(f"  投资年限: {years:.1f}年")

    return dict(
        name=name, first_date=str(first_date.date()), last_date=str(last_date.date()),
        first_close=f"{first_close:.2f}", last_close=f"{last_close:.2f}",
        cum_ret=f"{cum_ret:+.2f}%", cum_cls='up' if cum_ret >= 0 else 'down',
        cum_color='var(--up)' if cum_ret >= 0 else 'var(--down)',
        annual_ret=f"{annual_ret:+.2f}%", ann_cls='up' if annual_ret >= 0 else 'down',
        max_dd=f"{max_dd:.2f}%", years=f"{years:.1f}",
        max_ret=f"{max_ret:+.2f}%", min_ret=f"{min_ret:+.2f}%",
    )

s300 = compute_stats(df300, 'CSI300 (沪深300)', base300)
s2000 = compute_stats(df2000, 'CSI2000 (中证2000)', base2000)

# --- 5. 生成 HTML ---
out_dir = Path('/Users/hyj/PycharmProjects/StockMarket/backtest')
out_dir.mkdir(parents=True, exist_ok=True)

# 准备 JSON 数据（小数形式的收益率）
s300_chart = {
    'dates': [str(d.date()) for d in df300['date']],
    'values': [round(v/100, 6) for v in df300['cum_return'].tolist()],
}
s2000_chart = {
    'dates': [str(d.date()) for d in df2000['date']],
    'values': [round(v/100, 6) for v in df2000['cum_return'].tolist()],
}

gen_date = datetime.now().strftime('%Y-%m-%d %H:%M')
out_path = out_dir / 'index_return_compare.html'

# 分块拼接，避免 Python .format() 和 JS {} 冲突
parts = []

parts.append(f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>沪深300 vs 中证2000 投资收益率回测</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
:root {{
  --bg: #ffffff; --text: #1a1a2e; --text-secondary: #6b7280;
  --card-bg: #f8f9fa; --border: #e5e7eb;
  --up: #16a34a; --down: #dc2626;
  --line: #2563eb; --fill: rgba(37,99,235,0.10);
  --grid: rgba(128,128,128,0.18);
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    --bg: #0f172a; --text: #e2e8f0; --text-secondary: #94a3b8;
    --card-bg: #1e293b; --border: #334155;
    --up: #22c55e; --down: #ef4444;
    --line: #60a5fa; --fill: rgba(96,165,250,0.12);
    --grid: rgba(148,163,184,0.15);
  }}
}}
:root[data-theme="dark"] {{
  --bg: #0f172a; --text: #e2e8f0; --text-secondary: #94a3b8;
  --card-bg: #1e293b; --border: #334155;
  --up: #22c55e; --down: #ef4444;
  --line: #60a5fa; --fill: rgba(96,165,250,0.12);
  --grid: rgba(148,163,184,0.15);
}}
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Noto Sans SC", sans-serif;
  background: var(--bg); color: var(--text);
  padding: 40px 24px; max-width: 1100px; margin: 0 auto;
}}
h1 {{ font-size: 28px; font-weight: 700; margin-bottom: 8px; }}
.subtitle {{ color: var(--text-secondary); font-size: 15px; margin-bottom: 32px; }}
.chart-container {{
  background: var(--card-bg); border: 1px solid var(--border);
  border-radius: 12px; padding: 24px; margin-bottom: 32px;
}}
.chart-container h2 {{ font-size: 20px; font-weight: 600; margin-bottom: 16px; }}
.stats-grid {{
  display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
  gap: 16px; margin-bottom: 20px;
}}
.stat-card {{
  background: var(--bg); border: 1px solid var(--border);
  border-radius: 8px; padding: 14px 16px;
}}
.stat-card .label {{ font-size: 12px; color: var(--text-secondary); margin-bottom: 4px; }}
.stat-card .value {{ font-size: 22px; font-weight: 700; }}
.stat-card .value.up {{ color: var(--up); }}
.stat-card .value.down {{ color: var(--down); }}
.plot-box {{ width: 100%; height: 500px; }}
.comparison {{
  background: var(--card-bg); border: 1px solid var(--border);
  border-radius: 12px; padding: 24px;
}}
.comparison h2 {{ font-size: 20px; font-weight: 600; margin-bottom: 16px; }}
.comparison table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
.comparison th, .comparison td {{
  padding: 10px 14px; text-align: right; border-bottom: 1px solid var(--border);
}}
.comparison th:first-child, .comparison td:first-child {{ text-align: left; }}
.comparison th {{ color: var(--text-secondary); font-weight: 500; font-size: 12px; }}
.comparison tr:last-child td {{ border-bottom: none; }}
.footnote {{ color: var(--text-secondary); font-size: 12px; margin-top: 24px; text-align: center; }}
</style>
</head>
<body>

<h1>沪深300 vs 中证2000 指数投资收益率回测</h1>
<p class="subtitle">
  买入持有策略 · 忽略交易费用 · 数据来源: akshare/csindex · 生成于 {gen_date}
</p>

<!-- CSI300 -->
<div class="chart-container">
  <h2>沪深300 (CSI300) — {s300["first_date"]} 至今</h2>
  <div class="stats-grid">
    <div class="stat-card"><div class="label">累计收益</div><div class="value {s300["cum_cls"]}">{s300["cum_ret"]}</div></div>
    <div class="stat-card"><div class="label">年化收益</div><div class="value {s300["ann_cls"]}">{s300["annual_ret"]}</div></div>
    <div class="stat-card"><div class="label">最大回撤</div><div class="value down">{s300["max_dd"]}</div></div>
    <div class="stat-card"><div class="label">最高收益</div><div class="value up">{s300["max_ret"]}</div></div>
    <div class="stat-card"><div class="label">投资年限</div><div class="value">{s300["years"]}年</div></div>
    <div class="stat-card"><div class="label">起始点位</div><div class="value">{s300["first_close"]}</div></div>
    <div class="stat-card"><div class="label">最新点位</div><div class="value">{s300["last_close"]}</div></div>
  </div>
  <div id="chart300" class="plot-box"></div>
</div>

<!-- CSI2000 -->
<div class="chart-container">
  <h2>中证2000 (CSI2000) — {s2000["first_date"]} 至今</h2>
  <div class="stats-grid">
    <div class="stat-card"><div class="label">累计收益</div><div class="value {s2000["cum_cls"]}">{s2000["cum_ret"]}</div></div>
    <div class="stat-card"><div class="label">年化收益</div><div class="value {s2000["ann_cls"]}">{s2000["annual_ret"]}</div></div>
    <div class="stat-card"><div class="label">最大回撤</div><div class="value down">{s2000["max_dd"]}</div></div>
    <div class="stat-card"><div class="label">最高收益</div><div class="value up">{s2000["max_ret"]}</div></div>
    <div class="stat-card"><div class="label">投资年限</div><div class="value">{s2000["years"]}年</div></div>
    <div class="stat-card"><div class="label">起始点位</div><div class="value">{s2000["first_close"]}</div></div>
    <div class="stat-card"><div class="label">最新点位</div><div class="value">{s2000["last_close"]}</div></div>
  </div>
  <div id="chart2000" class="plot-box"></div>
</div>

<!-- 对比表 -->
<div class="comparison">
  <h2>指标对比</h2>
  <table>
    <thead><tr><th>指标</th><th>沪深300</th><th>中证2000</th></tr></thead>
    <tbody>
      <tr><td>回测区间</td><td>{s300["first_date"]} ~ {s300["last_date"]}</td><td>{s2000["first_date"]} ~ {s2000["last_date"]}</td></tr>
      <tr><td>投资年限</td><td>{s300["years"]}年</td><td>{s2000["years"]}年</td></tr>
      <tr><td>累计收益率</td><td style="color:{s300["cum_color"]}">{s300["cum_ret"]}</td><td style="color:{s2000["cum_color"]}">{s2000["cum_ret"]}</td></tr>
      <tr><td>年化收益率</td><td style="color:{s300["cum_color"]}">{s300["annual_ret"]}</td><td style="color:{s2000["cum_color"]}">{s2000["annual_ret"]}</td></tr>
      <tr><td>最大回撤</td><td style="color:var(--down)">{s300["max_dd"]}</td><td style="color:var(--down)">{s2000["max_dd"]}</td></tr>
    </tbody>
  </table>
</div>

<p class="footnote">注：以上为纯指数买入持有收益，不含分红、交易费用和滑点。历史收益不代表未来表现。</p>
''')

# JavaScript: 读取 CSS 变量构建 Plotly 配置
parts.append(f'''
<script>
const DATA_300 = {json.dumps(s300_chart)};
const DATA_2000 = {json.dumps(s2000_chart)};

function getThemeColors() {{
  const style = getComputedStyle(document.documentElement);
  return {{
    textSecondary: style.getPropertyValue('--text-secondary').trim(),
    line: style.getPropertyValue('--line').trim(),
    fill: style.getPropertyValue('--fill').trim(),
    grid: style.getPropertyValue('--grid').trim(),
  }};
}}

function makeLayout(colors) {{
  return {{
    margin: {{ l: 50, r: 30, t: 10, b: 40 }},
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'transparent',
    xaxis: {{
      showgrid: true, gridcolor: colors.grid, gridwidth: 1,
      zeroline: false, showline: false,
      tickfont: {{ size: 12, color: colors.textSecondary }},
    }},
    yaxis: {{
      showgrid: true, gridcolor: colors.grid, gridwidth: 1,
      zeroline: true, zerolinecolor: colors.grid, zerolinewidth: 1.5,
      showline: false,
      tickformat: '.0%',
      tickfont: {{ size: 12, color: colors.textSecondary }},
      title: {{ text: '累计收益率', font: {{ size: 13, color: colors.textSecondary }} }},
      hoverformat: '.2%'
    }},
    hovermode: 'x unified',
    showlegend: false,
    dragmode: false
  }};
}}

function makeTrace(dates, values, colors) {{
  return {{
    x: dates, y: values,
    type: 'scatter', mode: 'lines',
    line: {{ color: colors.line, width: 2.5 }},
    fill: 'tozeroy',
    fillcolor: colors.fill,
    hovertemplate: '%{{x|%Y-%m-%d}}<br>累计收益: <b>%{{y:.2%}}</b><extra></extra>'
  }};
}}

function plotAll(colors) {{
  Plotly.newPlot('chart300', [makeTrace(DATA_300.dates, DATA_300.values, colors)],
                 makeLayout(colors), {{ responsive: true, displayModeBar: false }});
  Plotly.newPlot('chart2000', [makeTrace(DATA_2000.dates, DATA_2000.values, colors)],
                 makeLayout(colors), {{ responsive: true, displayModeBar: false }});
}}

// 初始绘制
plotAll(getThemeColors());

// 监听主题切换
const mq = window.matchMedia('(prefers-color-scheme: dark)');
mq.addEventListener('change', () => {{
  Plotly.purge('chart300');
  Plotly.purge('chart2000');
  plotAll(getThemeColors());
}});
</script>
</body>
</html>''')

out_path.write_text('\n'.join(parts), encoding='utf-8')
print(f"\n✅ 图表已保存: {out_path}")
print(f"   在浏览器中打开: open {out_path}")
