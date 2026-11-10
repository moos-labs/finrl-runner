# finrl-runner

Runs a **FinRL** deep-reinforcement-learning stock-trading pilot via GitHub
Actions — part of a one-public-repo-per-tool pattern: each tool gets its own
repo, the commands to execute live in a repository variable, runs are
triggered manually, and outputs are collected as artifacts.

The pilot trains an **A2C** agent (FinRL's `DRLAgent` on top of
stable-baselines3) on a single ticker and backtests it on a following
trade window, reporting FinRL's own account-value output.

## How to run an analysis

1. **Edit the commands**: repo **Settings → Secrets and variables → Actions →
   Variables** → set `COMMANDS` to the shell commands you want executed, e.g.
   ```
   mkdir -p results
   set -o pipefail; python scripts/finrl_a2c.py --ticker AAPL 2>&1 | tee results/run.log
   ```
   (`set -o pipefail;` matters: piping through `tee` would otherwise mask a
   script failure and show the run as "success".)
2. **Trigger manually**: **Actions** tab → *Run analysis* → *Run workflow*
   (or dispatch it through the GitHub API).
3. **Collect results**: download the `results` artifact from the finished run.
   It contains `result.md` (ticker, windows, timesteps, initial vs final
   portfolio value and total return, from FinRL's own backtest output),
   `account_value.csv` (FinRL's per-step account value over the trade
   window), and `run.log`.

## Pilot

`scripts/finrl_a2c.py` — generic; ticker and parameters are CLI arguments:

| Argument | Default |
|---|---|
| `--ticker` | `AAPL` |
| `--timesteps` | `8000` |
| `--train-start` / `--train-end` | `2022-01-03` / `2023-12-30` (~2-year train window) |
| `--trade-start` / `--trade-end` | `2023-12-30` / `2024-06-29` (~6-month trade window) |
| `--initial-amount` | `1000000` |

Flow (FinRL 0.3.7, the classic StockTradingEnv tutorial): yfinance download
→ FinRL `FeatureEngineer` with `finrl.config.INDICATORS` → `data_split` →
`StockTradingEnv` → `DRLAgent` A2C `train_model()` → `DRL_prediction()` on
the trade env.

## Honest notes

- **FinRL 0.3.7 declares no dependencies of its own** (its wheel metadata
  has no `Requires-Dist`), and merely *importing* `finrl` pulls in its
  alpaca / WRDS data-processor chain. `requirements.txt` therefore pins the
  whole tree explicitly: CPU `torch==2.3.1+cpu`,
  `stable-baselines3==2.3.2`, `gymnasium==0.29.1`, `numpy==1.26.4`,
  `pandas==2.2.2`, `stockstats`, plus `alpaca-trade-api`,
  `exchange-calendars` and `wrds` for those import chains (the alpaca/WRDS
  processors are imported, never used). `alpaca-trade-api` pins
  `msgpack==1.0.3`, which has no Python 3.11 wheel and builds from source
  on the runner.
- **FinRL's `YahooDownloader` is not used** — it cannot work with any
  yfinance that still talks to Yahoo today: yfinance ≥0.2.54 returns
  MultiIndex columns and no separate `Adj Close` by default, so
  `fetch_data()`'s fixed flat column assignment crashes, while
  yfinance ≤0.2.43 (the old flat format) downloads nothing from Yahoo any
  more. The script therefore downloads with yfinance directly
  (`auto_adjust=False, multi_level_index=False`, pinned
  `yfinance==0.2.55`) and shapes the result into the exact dataframe
  format `YahooDownloader.fetch_data()` produced (close = adjusted close),
  then uses FinRL's own `FeatureEngineer` / `data_split` / env / agent.
- **This is a smoke test, not research.** 8,000 A2C timesteps on one stock
  barely trains the agent; the numbers prove the pipeline runs end to end,
  nothing more. DRL trading results are highly seed- and window-sensitive.
- **CPU only.** Torch comes from the PyTorch CPU wheel index.
- yfinance is fine here because the pilot ticker is a US listing; it is
  *not* used for Cairo/EGX tickers in this fleet (yfinance fails on
  Africa/Cairo DST timestamps there — see the other runners).
