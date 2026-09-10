"""
择时策略：冰点买入 + 泡沫卖出

- 冰点：CSI300分数≥50 or CSI2000分数≥50 → 分批买入
- 泡沫：PE>60%分位 + 宏观恶化 → 减仓
- 其他：持有不动
"""

import logging
from datetime import datetime
import pandas as pd
import numpy as np

from data.fetcher import fetch_all_data, _load
from macro.fetcher import fetch_all_macro
from strategy import generate_signal, detect_ice_point, detect_bubble

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

if __name__ == "__main__":
    print(f"╔══════════════════════════════════════════════════════╗")
    print(f"║  择时策略 · 冰点买入 + 泡沫卖出                       ║")
    print(f"║  {datetime.now().strftime('%Y-%m-%d %H:%M'):<50}║")
    print(f"╚══════════════════════════════════════════════════════╝")

    try:
        # 数据
        a_data = fetch_all_data(force=False)
        macro_df = fetch_all_macro(force=False)

        # 补全 CSI2000（近期段缺失时用 ETF 159531 反推指数）
        # ratio = ETF价格/指数点位（重叠日实测）；反推指数 = ETF价格 / ratio（2026-08-28 修复: 曾误用 *ratio 产生4700倍断层）
        idx2000 = a_data['csi2000_daily'].copy()
        etf531 = a_data['etf_159531'].copy()
        nd = etf531[~etf531['date'].isin(idx2000['date'])]
        ratio_2000 = None
        if len(nd) > 0:
            ol = idx2000.merge(etf531[['date','close']].rename(columns={'close':'e'}), on='date', how='inner')
            if ol.empty:
                raise ValueError("CSI2000拼接: 与ETF无重叠日，无法计算换算比率")
            ol = ol.assign(day_ratio=ol['e']/ol['close']).sort_values('date')
            # 断言1: 相邻日换算比率变化应平稳（ETF折溢价漂移，日变化通常<1%；
            # 全期CV含趋势漂移不适用，故用相邻日变化std）
            day_ratio_chg = ol['day_ratio'].pct_change().dropna()
            if day_ratio_chg.std() > 0.02:
                raise ValueError(f"CSI2000拼接: 换算比率日变化异常 (std={day_ratio_chg.std():.2%})")
            # ratio 取最近10个重叠日均值（贴近补丁段，实测反推误差<0.3%）
            ratio_2000 = float(ol['day_ratio'].tail(10).mean())
            nr = pd.DataFrame({'date':nd['date'],'open':nd['open']/ratio_2000,'close':nd['close']/ratio_2000,
                               'high':nd['high']/ratio_2000,'low':nd['low']/ratio_2000,'volume':nd['volume']})
            a_data['csi2000_daily'] = pd.concat([idx2000,nr]).sort_values('date').reset_index(drop=True)
            # 断言2: 拼接点无跳变（换算正确时相邻两日对数收益应远小于50%）
            jc = a_data['csi2000_daily']['close']
            jump = float(abs(np.log(jc.iloc[len(idx2000)] / jc.iloc[len(idx2000)-1])))
            if jump > 0.5:
                raise ValueError(f"CSI2000拼接: 拼接点跳变 {jump:.0%}，换算比率疑似错误")

        # ═══════════════════════════════════
        # 当前信号
        # ═══════════════════════════════════
        print(f"\n{'='*60}")
        print(f"  📡 当前信号")
        print(f"{'='*60}")

        signal = generate_signal(a_data, macro_df)

        print(f"\n  CSI300 分数: {signal['score_300']:.0f}/100 {'⚠️ 距冰点还差'+str(max(0,50-signal['score_300']))+'分' if signal['score_300']<50 else '✅ 冰点！'}")
        print(f"  CSI2000 分数: {signal['score_2000']:.0f}/100 {'⚠️ 距冰点还差'+str(max(0,50-signal['score_2000']))+'分' if signal['score_2000']<50 else '✅ 冰点！'}")
        pe_pct = signal.get('pe_pct_300')
        pe_str = f"{pe_pct:.0%}" if pe_pct is not None else "N/A"
        print(f"  PE分位: {pe_str}  {'🟢 便宜' if pe_pct and pe_pct < 0.3 else '🟡 合理' if pe_pct and pe_pct < 0.6 else '🔴 偏贵' if pe_pct else ''}")
        # v6 加分维度状态（两融出清确认）——冰点分数分项
        fl_2000 = signal.get('details_2000', {}).get('两融出清确认(+13)', {})
        if fl_2000:
            trig = fl_2000.get('触发', '')
            dd = fl_2000.get('两融距6月高', '')
            print(f"  [2000加分] 两融出清确认: {trig}" + (f" (两融{dd})" if dd else ""))

        if signal['ice_300'] or signal['ice_2000']:
            targets = []
            if signal['ice_300']: targets.append('沪深300')
            if signal['ice_2000']: targets.append('中证2000')
            print(f"\n  🧊 冰点买入: {' + '.join(targets)} → 当前就是播种季节！")
        else:
            print(f"\n  🧊 冰点: 无 → 等待回调")

        if signal.get('leg_300') or signal.get('leg_2000'):
            for name, leg in [('沪深300', signal.get('leg_300')), ('中证2000', signal.get('leg_2000'))]:
                if leg is None:
                    continue
                if leg.get('is_bubble'):
                    print(f"  ⚠️ {name}: {leg['level']}（PE分位{leg['pe_pct']:.0%}）")
                    for r in leg['reasons'][:3]:
                        print(f"     - {r}")
                elif leg.get('recovery'):
                    # v6 回补阶梯: 档位/累计回补比例（档1=1/3 → 档2=2/3 → 档3=3/3）
                    pct = leg.get('recovery_pct', 0)
                    print(f"  ↩️ {name}: {leg['level']} → 回补（累计{pct:.0%}，分批买回）")
                else:
                    print(f"  ✅ {name}: 无卖出信号（{leg['level']}）")
        elif signal['bubble']:
            print(f"  ⚠️ 泡沫: {signal['bubble_level']}")
            for r in signal['bubble_reasons']:
                print(f"     - {r}")
        else:
            print(f"  ✅ 泡沫: 无")

        print(f"\n  >>> {signal['position_advice']}")
        print(f"      CSI300: {signal['action_300']}  |  CSI2000: {signal['action_2000']}")

        # ═══════════════════════════════════
        # 机会雷达（卫星模型：核心之外的行业机会，独立不影响 300/2000）
        # ═══════════════════════════════════
        try:
            from models.opportunity_radar import (
                SATELLITE_SINGLE_PCT, SATELLITE_TOTAL_PCT, radar_scan,
            )
            radar = radar_scan()
            print(f"\n{'='*60}")
            print(f"  🛰️ 机会雷达（行业机会 · 单标≤{SATELLITE_SINGLE_PCT:.1%} · 总卫星≤{SATELLITE_TOTAL_PCT:.0%}）")
            print(f"{'='*60}")
            if radar['opportunities']:
                for o in radar['opportunities']:
                    tag = "持仓中" if o.get('held') else "新信号"
                    print(f"  🎯 [{tag}] {o['name']}({o['code']}) 收{o['price']:.3f} 仓位≤{o['pct']:.1%}")
                    print(f"     风控: 止损{o['stop']:.3f}({-o['stop_pct']:.2%}) 止盈{o['tp']:.3f}(+{o['tp_pct']:.2%}) 最长60交易日")
                    if o.get('held'):
                        continue
                    print(f"     买区: {o['ref_buy_zone']}")
                    d = o['detail'].get('因子窗口', {})
                    if d:
                        print(f"     因子: 近20日{d.get('近20日涨幅','—')} 窗口{d.get('窗口','')}")
                    d2 = o['detail'].get('行情确认', {})
                    if d2:
                        print(f"     行情: {d2.get('企稳','')} 量能{d2.get('量能5/20','')} {d2.get('距250日低点','')} {d2.get('启动区','')}")
            else:
                print(f"  😴 无机会——因子窗口+行情确认共振才会报")

            pos = radar.get('positions', [])
            if pos:
                used = sum(p['pct'] for p in pos)
                print(f"  📌 卫星持仓 {len(pos)} 笔 / 占用 {used:.1%} / 上限 15%")
                for p in pos:
                    print(f"     {p['name']}({p['code']}) {p['entry_date']}@{p['entry']:.3f} "
                          f"止损{p['stop']:.3f} 止盈{p['tp']:.3f} 持有{p['bars_held']}/60日")
            for e in radar.get('exits', []):
                print(f"  🚪 退出: {e['name']}({e['code']}) {e['reason']} @{e['exit_price']:.3f} "
                      f"收益{e['ret']:+.2%}（{e['entry_date']}@{e['entry']:.3f} 入）")
            for w in radar['watching']:
                if any(('因子窗口未开' in m) or ('行情未确认' in m) for m in w['missing']):
                    print(f"  👀 观察: {w['name']} 缺{'/'.join(w['missing'])}")
        except Exception as e:
            print(f"  ⚠️ 机会雷达: {str(e)[:60]}")

        if a_data.get('bond_yield_10y') is None:
            print(f"  ⚠️ 国债收益率数据缺失（所有新鲜源失效），ERP维度已跳过——宁缺毋滥，未使用停更数据")

        # ═══════════════════════════════════
        # 历史冰点回顾
        # ═══════════════════════════════════
        print(f"\n{'='*60}")
        print(f"  2020-2026 冰点买入回顾（持有至今收益）")
        print(f"{'='*60}")

        idx300 = a_data['csi300_daily'].copy()
        etf300 = a_data['etf_159330'].copy()
        m = idx300.merge(etf300[['date','close']].rename(columns={'close':'e'}), on='date', how='inner')
        r300 = (m['e']/m['close']).mean()

        p300_now = float(etf300['close'].iloc[-1])
        p531_now = float(etf531['close'].iloc[-1])

        # 按月扫描冰点
        months = pd.date_range('2020-01-01', '2026-08-01', freq='MS')
        ice_points = []
        for dt in months:
            ice = detect_ice_point(a_data, macro_df, dt)
            idx_p300 = float(idx300[idx300['date']<=dt]['close'].iloc[-1]) * r300
            # CSI2000 ETF价格近似
            idx2 = a_data['csi2000_daily']
            p2 = float(etf531[etf531['date']<=dt]['close'].tail(1).iloc[0]) if len(etf531[etf531['date']<=dt])>0 else None
            if p2 is None and len(idx2[idx2['date']<=dt])>0:
                # ETF未上市前用指数×实测换算比率近似（ratio_2000=ETF/指数；r300*0.8为旧兜底近似）
                p2 = float(idx2[idx2['date']<=dt]['close'].iloc[-1]) * (ratio_2000 or r300*0.8)

            if ice['csi300'] or ice['csi2000']:
                ice_points.append({
                    'date': dt,
                    'ice_300': ice['csi300'],
                    'ice_2000': ice['csi2000'],
                    's300': ice['score_300'],
                    's2000': ice['score_2000'],
                    'p300': idx_p300,
                    'p2000': p2,
                    'pe_pct': ice['pe_pct_300'],
                })

        # 去重：连续冰点合并
        merged_ice = []
        prev_year = None
        for ip in ice_points:
            year = ip['date'].year
            if year != prev_year or not merged_ice:
                merged_ice.append(ip)
            elif ip['s300'] > merged_ice[-1]['s300'] or ip['s2000'] > merged_ice[-1]['s2000']:
                # 保留该年分数最高的
                merged_ice[-1] = ip
            prev_year = year

        print(f"\n  {'年份':<6} {'ETF':<8} {'买入价':>7} {'当前':>7} {'收益':>7}  {'年化':>7}  {'分数':>5}")
        print(f"  {'─'*60}")
        for ip in merged_ice:
            year = ip['date'].year
            if ip['ice_300']:
                ret = (p300_now/ip['p300'] - 1) * 100
                years = (datetime.now() - ip['date'].to_pydatetime()).days / 365
                ann = ((1+ret/100)**(1/years)-1)*100 if years > 0 else 0
                print(f"  {year:<6} {'CSI300':<8} {ip['p300']:>7.4f} {p300_now:>7.4f} {ret:>+6.1f}% {ann:>+6.1f}% {ip['s300']:>5.0f}")
            if ip['ice_2000']:
                if ip['p2000']:
                    ret = (p531_now/ip['p2000'] - 1) * 100
                    years = (datetime.now() - ip['date'].to_pydatetime()).days / 365
                    ann = ((1+ret/100)**(1/years)-1)*100 if years > 0 else 0
                    print(f"  {year:<6} {'CSI2000':<8} {ip['p2000']:>7.4f} {p531_now:>7.4f} {ret:>+6.1f}% {ann:>+6.1f}% {ip['s2000']:>5.0f}")

        # 泡沫回顾
        print(f"\n{'='*60}")
        print(f"  泡沫卖出信号回顾")
        print(f"{'='*60}")
        bubble_count = 0
        for dt in months:
            b = detect_bubble(a_data, macro_df, dt)
            if b['is_bubble']:
                bubble_count += 1
                print(f"  {dt.date()}  {b['level']}  PE分位{b['pe_pct']:.0%}")
                for r in b['reasons']:
                    print(f"    - {r}")
        if bubble_count == 0:
            print(f"  2020-2026: 无泡沫信号（PE从未超过60%分位）")

        print(f"\n✅ 完成 ({datetime.now().strftime('%H:%M:%S')})")

    except Exception as e:
        print(f"\n❌ 失败: {e}")
        import traceback
        traceback.print_exc()
