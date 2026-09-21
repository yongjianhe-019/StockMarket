#!/usr/bin/env python
"""机会扫描器 —— 只回答一件事：现在有没有"值得博弈"的冰点买点？

逻辑见 models/opportunity_entry.py（R:R + 条件胜率 + EV + 企稳确认）。
止盈由用户自行处理，本工具只负责发现买点。

用法:
    .venv/bin/python find_opportunity.py          # 当前机会
    .venv/bin/python find_opportunity.py --recent # 附最近 90 天触发记录
退出码: 1 = 有值得博弈的机会（可接通知/cron）
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from data.fetcher import fetch_all_data, patch_csi2000_index  # noqa: E402
from macro.fetcher import fetch_all_macro  # noqa: E402
from models.opportunity_entry import (  # noqa: E402
    RR_MIN, missing_reasons, prep, scan, scan_history,
)


def main() -> int:
    a = fetch_all_data(force=False)
    _ = fetch_all_macro(force=False)
    patch_csi2000_index(a)

    print("=" * 88)
    print("  🎯 机会扫描器 · 值得博弈的冰点买点")
    print("=" * 88)

    any_opp = False
    for e in scan(a):
        tag = "🟢 值得博弈" if e["eligible"] else "😴 暂不满足"
        print(f"\n  【{e['name']}】{str(e['date'])[:10]}  收{e['close']:.3f}  "
              f"回撤{e['dd']:.1%}  {e['regime']}  →  {tag}")
        print(f"    止损{e['stop']:.3f}(-{e['risk']/e['close']:.1%})  "
              f"目标{e['target']:.3f}(+{e['reward']/e['close']:.1%})  R:R={e['rr']:.2f}")
        print(f"    条件胜率{e['win']:.0%}(Wilson下界{e['wilson']:.0%}, n={e['n']})  "
              f"EV={e['ev']:.1f}  →  建议仓位 {e['position_pct']:.0%}")
        ma20 = prep(a[e["code"]]).iloc[-1]["ma20"]
        miss = missing_reasons(e, ma20)
        if miss:
            print(f"    卡在: {'、'.join(miss)}")
        any_opp = any_opp or e["eligible"]

    if "--recent" in sys.argv:
        print(f"\n{'='*88}\n  最近 90 天触发记录")
        for name, key in [("沪深300", "csi300_daily"), ("中证2000", "csi2000_daily")]:
            hits = scan_history(a, name, key, tail=90)
            txt = " ".join(f"{str(h['date'])[5:10]}(R:R{h['rr']:.1f},胜率{h['win']:.0%})" for h in hits)
            print(f"    {name}: {txt if txt else '无'}")

    print(f"\n  提示: 止盈请自行处理；本工具只负责发现买点。历史表为全样本，实盘以样本外复核为准。")
    return 1 if any_opp else 0


if __name__ == "__main__":
    sys.exit(main())
