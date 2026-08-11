"""回测报告"""
from __future__ import annotations


def print_report(result: dict) -> None:
    metrics = result["metrics"]
    trades = result["trades"]
    snapshots = result["snapshots"]

    if not metrics:
        print("无回测数据")
        return

    alpha = metrics["alpha"]
    total_ret = metrics["total_return"]
    bench_ret = metrics["benchmark_return"]

    print("\n" + "=" * 70)
    print("  回测报告")
    print("=" * 70)

    # ---- 总览（核心指标） ----
    symbol = "🚀" if alpha > 0 else "📉"
    print(f"\n  {symbol} 核心指标")
    print(f"  {'─' * 50}")
    print(f"  策略总收益:    {total_ret:>+8.2%}")
    print(f"  基准收益:      {bench_ret:>+8.2%}")
    print(f"  {'超额 α:':<16} {alpha:>+8.2%}  {'✅ 跑赢' if alpha > 0 else '❌ 跑输'}")
    print(f"  年化收益:      {metrics['annual_return']:>+8.2%}")
    print(f"  最大回撤:      {metrics['max_drawdown']:>8.2%}")
    print(f"  总交易:        {metrics['total_trades']:>8}  (买{metrics['buy_trades']}/卖{metrics['sell_trades']})")
    print(f"  期末现金:      {metrics['final_cash']:>10,.0f}")

    # ---- 年度分解 ----
    yearly = metrics.get("yearly", [])
    if yearly:
        print(f"\n  📅 年度分解")
        print(f"  {'─' * 65}")
        header = f"  {'年份':<6} {'策略收益':>8} {'基准收益':>8} {'超额α':>8} {'最大回撤':>8} {'交易':>6}"
        print(header)
        for y in yearly:
            strat_r = float(y['return'].rstrip('%')) / 100
            bench_r = float(y['benchmark'].rstrip('%')) / 100
            y_alpha = strat_r - bench_r
            marker = " ✅" if y_alpha > 0 else "  "
            print(f"  {y['year']:<6} {y['return']:>8} {y['benchmark']:>8} "
                  f"{y_alpha:>+7.2%}{marker} {y['max_drawdown']:>8} {y['trades']:>6}")

    # ---- 期末持仓 ----
    positions = metrics.get("final_positions", {})
    print(f"\n  📦 期末持仓")
    print(f"  {'─' * 40}")
    for code, pos in positions.items():
        name = {"159330": "沪深300ETF", "159531": "中证2000ETF"}.get(code, code)
        print(f"  {name}: {pos['shares']}股  均价{pos['avg_cost']:.3f}  成本{pos['total_cost']:,.0f}")

    # ---- 最近交易 ----
    if trades:
        print(f"\n  📋 最近 15 笔交易")
        print(f"  {'─' * 80}")
        print(f"  {'日期':<12} {'ETF':<8} {'操作':<4} {'数量':>7} {'价格':>7} {'金额':>10} {'分数':>5}  原因")
        for t in trades[-15:]:
            print(f"  {str(t.date.date()):<12} {t.etf:<8} {t.action:<4} {t.shares:>7,} "
                  f"{t.price:>7.3f} {t.amount:>10,.0f} {t.score:>5.0f}  {t.reason}")

    print("\n" + "=" * 70 + "\n")


def print_optimization(top_results: list[dict]) -> None:
    """打印优化扫描的 top 结果。"""
    print(f"\n{'='*80}")
    print("  参数优化 — Top 结果（按 α 超额收益排序）")
    print(f"{'='*80}")
    print(f"  {'排名':<4} {'α':>7} {'策略':>7} {'基准':>7} {'回撤':>7} {'交易':>5}  core   enter  翻倍止盈  熊市减仓")
    print(f"  {'─'*80}")
    for i, r in enumerate(top_results):
        p = r["params"]
        print(f"  {i+1:<4} {r['alpha']:>+6.2%} {r['total_return']:>+6.2%} "
              f"{r['benchmark_return']:>+6.2%} {r['max_drawdown']:>6.2%} "
              f"{r['trades']:>5}  "
              f"{p['core_pct']:.0%}    {p['buy_score_enter']:.0f}     "
              f"{'✅' if p.get('profit_double_sell') else '❌':<6}    "
              f"{'✅' if p.get('bear_market_exit') else '❌'}")
    print()


def print_yearly(results: list[dict]) -> None:
    """打印年度独立回测结果。"""
    print(f"{'='*80}")
    print("  年度独立回测 — 每年 10 万独立运作，年内最优")
    print(f"{'='*80}")
    print(f"  {'年份':<6} {'策略最优':>8} {'买入持有':>8} {'超额α':>8} {'最大回撤':>8} {'交易':>5}  "
          f"core   enter  数据范围")
    print(f"  {'─'*100}")

    for r in results:
        if r is None:
            continue
        alpha = r["alpha"]
        marker = " ✅" if alpha > 0 else "  "
        p = r["params"]
        dr = r.get("etf300_dates", r.get("etf531_dates", ""))[:21]
        print(f"  {r['year']:<6} {r['total_return']:>+7.2%} "
              f"{r['benchmark_return']:>+7.2%} "
              f"{alpha:>+7.2%}{marker} "
              f"{r['max_drawdown']:>7.2%} "
              f"{r['total_trades']:>5}  "
              f"{p['core_pct']:.0%}    {p['buy_score_enter']:.0f}     "
              f"{dr}")

    print()
