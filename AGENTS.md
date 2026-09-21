# AGENTS.md

A股 ETF 左侧择时研究系统（冰点买入 / 泡沫卖出）。两个标的：沪深300 `159330`、中证2000 `159531`。纯研究，产出月频信号与回测。

## 环境与依赖

- 一律用仓库自带 venv：`.venv/bin/python`（Python 3.13）。不要用系统 `python`。
- `pyproject.toml` 的 `dependencies = []` 是空的；真实依赖（akshare / pandas / numpy / pyarrow / efinance）只装在 `.venv` 里。不要据此以为没有依赖。
- 未安装 pytest。测试全部是 `unittest`。
- `data/fetcher.py` / `macro/fetcher.py` 通过 akshare 走网络（eastmoney/sina/csindex），部分源需要代理；抓取慢，`force=False` 时优先读本地 parquet 缓存。

## 常用命令

```bash
.venv/bin/python dashboard.py                          # 主报告：月频信号+机会雷达+历史回顾（走网络，约5-6分钟）
.venv/bin/python -m unittest discover -s tests         # 全部测试
.venv/bin/python -m unittest tests.test_sell_signal -v # 单个测试模块
.venv/bin/python check_2000_ice.py                     # 只答"中证2000离冰点差多远"，exit 1=触发
.venv/bin/python backtest.py                           # 历史回测
```

## 架构要点

- `strategy.py` 是核心决策文件（`generate_signal` / `detect_ice_point` / `detect_bubble`）。README「项目结构」里提到的生产引擎 `signal.py` **不存在**，真实入口是 `dashboard.py`。
- `strategy/`（目录）是设计文档（markdown），**不是代码**；别和根目录的 `strategy.py` 混淆。
- `macro/sell_signal.py`：泡沫/趋势破坏/回补信号，且**分标判断**（300 用 `csi300_daily`，2000 用 `etf_159531`）。
- 打分：`models/csi300.py`、`models/csi2000.py`；卫星雷达：`models/opportunity_radar.py`，状态持久化在 `data/radar_state.json`。
- 数据缓存：`data/*.parquet`，由 `data/fetcher._load` 管理，缓存寿命 1 个日历日。**注意这些 parquet 已提交进 git**（虽然 `.gitignore` 写了 `*.parquet`），跑一次脚本后 `git status` 会显示大量 `data/*.parquet` 变更，提交前想清楚是否要带上。
- 数据新鲜度铁律（不可静默违反）：≤7 天正常，7–20 天告警，>20 天直接抛弃并告警；停更数据源永久移除。

## 测试注意事项

- 5 个测试会因**当前行情数据**而失败（不是代码回归，别为了让它绿去改模型）：
  - `tests/test_csi2000_channel.py::TestChannel::test_2026_08_04_channel_A_50`
  - `tests/test_opportunity_radar.py::TestCooldownPersistence` 下 4 个（`test_cooldown_survives_across_scans`、`test_next_day_still_in_cooldown`、`test_returns_open_positions`、`test_close_position_reconciles_without_buying`）
- 这些测试断言"某信号必须触发"或固定历史日期的通道结果，随行情/滚动分位窗口变化会翻红。
- 多数测试用本地 parquet 重建宏观序列（见 `tests/test_csi2000_channel.py` 的 `_macro_from_local_files`），所以可离线跑；少数走 `radar_scan()` 会读实时数据。

## 约定

- 文档、docstring、注释、提交信息均为中文，保持一致。
- **每次提交前必须回测看效果**（用户铁律）：模型/信号/数据管道改动都要跑回测，用数据证明不劣化或改进；不达标不提交。回测脚本见 `backtest/`。
- 模型逻辑改动要记入 README「模型优化日志」（v3…v8）和 `strategy/00-project-progress.md`，不要静默改。提交前缀用 `修复 …` / `新增 …` / `mod …` 并带版本号（v6/v7/v8）。
