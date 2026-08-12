"""
研究脚本（不改模型）: 验证「两融出清」加分维度的假设

问题: 如果给 CSI2000 冰点模型加"两融出清"维度，能否:
  1. 捕捉 2024-02 小微盘崩盘底 ✅
  2. 捕捉 2026-07 急跌反弹底 ✅
  3. 不在 2022/2023 阴跌中接飞刀 ✅

候选信号定义（待验证）:
  A. 两融距6月高点收缩 >10%
  B. 两融20日收缩 >5% (急清)
  C. 两融3月收缩 >10%
  + 指数站回MA20 (反弹确认) + 成交额20日均量回升

评估: 每次信号触发后 20/60/120 日前向收益，判断是好底还是飞刀
"""
import sys
sys.path.insert(0, '/Users/hyj/PycharmProjects/StockMarket')

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

DATA = '/Users/hyj/PycharmProjects/StockMarket/data'

mb = pd.read_parquet(f'{DATA}/margin_balance.parquet').sort_values('date').reset_index(drop=True)
idx = pd.read_parquet(f'{DATA}/csi2000_daily_full.parquet').sort_values('date').reset_index(drop=True)
idx300 = pd.read_parquet(f'{DATA}/csi300_daily.parquet').sort_values('date').reset_index(drop=True)

mb['date'] = pd.to_datetime(mb['date'])
idx['date'] = pd.to_datetime(idx['date'])

# 合并两融到指数交易日
merged = idx.merge(mb[['date', 'margin_balance']], on='date', how='left')
merged['margin_balance'] = merged['margin_balance'].ffill()

# 计算信号
merged['mb_6m_high'] = merged['margin_balance'].rolling(126, min_periods=60).max()
merged['mb_dd_from_high'] = merged['margin_balance'] / merged['mb_6m_high'] - 1  # 负值=收缩
merged['mb_20d_chg'] = merged['margin_balance'].pct_change(20)
merged['mb_63d_chg'] = merged['margin_balance'].pct_change(63)
merged['ma20'] = merged['close'].rolling(20).mean()
merged['above_ma20'] = merged['close'] > merged['ma20']
merged['vol_ma5'] = merged['volume'].rolling(5).mean()
merged['vol_ma20'] = merged['volume'].rolling(20).mean()
merged['vol_recovering'] = merged['vol_ma5'] > merged['vol_ma20']

# 三组信号定义
merged['sigA'] = (merged['mb_dd_from_high'] < -0.10) & merged['above_ma20'] & merged['vol_recovering']
merged['sigB'] = (merged['mb_20d_chg'] < -0.05) & merged['above_ma20'] & merged['vol_recovering']
merged['sigC'] = (merged['mb_63d_chg'] < -0.10) & merged['above_ma20'] & merged['vol_recovering']

# 只统计信号首次触发（连续触发合并为一次事件）
def find_events(df, sig_col):
    events = []
    in_event = False
    for i, row in df.iterrows():
        if row[sig_col] and not in_event:
            events.append(i)
            in_event = True
        elif not row[sig_col]:
            in_event = False
    return events

# 前向收益
for horizon in [20, 60, 120]:
    merged[f'fwd_{horizon}d'] = merged['close'].shift(-horizon) / merged['close'] - 1

print('=' * 70)
print('  两融出清信号研究: 触发事件 + 前向收益')
print('=' * 70)

for sig_name in ['sigA', 'sigB', 'sigC']:
    events = find_events(merged, sig_name)
    print(f'\n--- {sig_name} 共{len(events)}次触发 ---')
    if not events:
        continue
    print(f'  {"触发日期":<12} {"点位":>7} {"20日":>8} {"60日":>8} {"120日":>8}  判定')
    for i in events:
        row = merged.iloc[i]
        f20 = row['fwd_20d']; f60 = row['fwd_60d']; f120 = row['fwd_120d']
        # 判定: 120日内 >10% = 好底; 继续跌>10% = 飞刀; 其他=中性
        if not np.isnan(f120):
            if f120 > 0.10: verdict = '✅好底'
            elif f120 < -0.10: verdict = '❌飞刀'
            else: verdict = '➖中性'
        else:
            verdict = '近期(数据不足)'
        f20s = f'{f20:+.1%}' if not np.isnan(f20) else '  —'
        f60s = f'{f60:+.1%}' if not np.isnan(f60) else '  —'
        f120s = f'{f120:+.1%}' if not np.isnan(f120) else '  —'
        print(f'  {str(row["date"].date()):<12} {row["close"]:>7.0f} {f20s:>8} {f60s:>8} {f120s:>8}  {verdict}')

# 关键期验证
print('\n' + '=' * 70)
print('  关键期验证')
print('=' * 70)
key_periods = {
    '2022-01~2022-12 阴跌年': ('2022-01-01', '2022-12-31'),
    '2023-01~2023-12 阴跌年': ('2023-01-01', '2023-12-31'),
    '2024-02 小微盘崩盘底': ('2024-01-15', '2024-03-15'),
    '2026-07 急跌反弹底': ('2026-06-25', '2026-08-12'),
}
for period, (s, e) in key_periods.items():
    sub = merged[(merged['date'] >= s) & (merged['date'] <= e)]
    triggers = {sig: find_events(sub, sig) for sig in ['sigA', 'sigB', 'sigC']}
    # 期间指数表现
    p0 = sub['close'].iloc[0]; p1 = sub['close'].iloc[-1]
    ts = ', '.join([f'{sig}:{len(v)}次' for sig, v in triggers.items()])
    print(f'  {period}: 指数{p0:.0f}→{p1:.0f} ({(p1/p0-1)*100:+.1f}%) | 信号 {ts}')
