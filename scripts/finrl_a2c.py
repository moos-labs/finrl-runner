#!/usr/bin/env python3
"""FinRL pilot: train an A2C agent (stable-baselines3 via FinRL's DRLAgent)
on one ticker and backtest it on a following trade window.

Standard FinRL flow (FinRL 0.3.7, the classic StockTradingEnv tutorial):
  1. Download OHLCV data (yahooquery primary, yfinance fallback) and shape
     it into the exact dataframe FinRL's preprocessor expects
     (date/open/high/low/close/volume/tic/day, close = adjusted close) --
     the same output format as
     finrl.meta.preprocessor.yahoodownloader.YahooDownloader.fetch_data().
     YahooDownloader itself is NOT called: with every yfinance version that
     still works against Yahoo today (>=0.2.54), fetch_data() crashes --
     yfinance now returns MultiIndex columns and (by default) no
     "Adj Close" column, while fetch_data() assigns a fixed flat column
     list and only catches NotImplementedError. yfinance <=0.2.43 returns
     the old flat format but no longer downloads anything from Yahoo.
  2. Feature engineering with FinRL's own FeatureEngineer
     (finrl.meta.preprocessor.preprocessors) using finrl.config.INDICATORS.
  3. data_split() into the train and trade windows.
  4. StockTradingEnv (finrl.meta.env_stock_trading.env_stocktrading)
     for both windows.
  5. DRLAgent (finrl.agents.stablebaselines3.models) with model "a2c",
     train_model(), then DRL_prediction() on the trade env, which returns
     FinRL's own account-value (asset) memory.
  6. Write results/result.md and results/account_value.csv.

Generic: ticker / windows / timesteps / capital are CLI arguments, nothing
is tied to a specific market or exchange.
"""
from __future__ import annotations

import argparse
import os

import pandas as pd
import yfinance as yf

from finrl.config import INDICATORS
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split

INITIAL_AMOUNT = 1_000_000


def _finrl_frame(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Shape a raw OHLCV frame into FinRL's YahooDownloader output format.

    Input columns (already lower-cased, spaces removed): date, open, high,
    low, close, volume and optionally adjclose. Following FinRL's
    YahooDownloader convention, the adjusted close becomes `close`.
    """
    df = df.copy()
    if "adjclose" in df.columns:
        df["close"] = df["adjclose"]
        df = df.drop(labels="adjclose", axis=1)
    df["tic"] = ticker
    df["day"] = pd.to_datetime(df["date"]).dt.dayofweek
    df["date"] = pd.to_datetime(df["date"]).apply(lambda x: x.strftime("%Y-%m-%d"))
    df = df.dropna().reset_index(drop=True)
    df = df.sort_values(by=["date", "tic"]).reset_index(drop=True)
    return df[["date", "open", "high", "low", "close", "volume", "tic", "day"]]


def _download_yahooquery(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Primary route: yahooquery's history endpoint.

    Verified 2026-11-10 (moos-labs/yahooquery-runner): yahooquery fetched
    full AAPL and POUL.CA history from GitHub-runner IPs at times when
    both yfinance and the raw Yahoo chart API were returning HTTP 429.
    """
    from yahooquery import Ticker

    hist = Ticker(ticker).history(start=start, end=end)
    if hist is None or len(hist) == 0:
        raise ValueError("yahooquery returned no rows")
    df = hist.reset_index()
    df.columns = [str(c).lower().replace(" ", "") for c in df.columns]
    cols = ["date", "open", "high", "low", "close", "volume"]
    if "adjclose" in df.columns:
        cols.append("adjclose")
    return _finrl_frame(df[cols], ticker)


def download_finrl_frame(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download OHLCV data and return FinRL's YahooDownloader format.

    Route order: yahooquery first (see _download_yahooquery), then
    yfinance as a fallback. Yahoo intermittently rate-limits
    GitHub-runner IPs (YFRateLimitError / HTTP 429) -- pilot runs 2, 3
    and 5 died on yfinance 429s alone -- so both routes retry with
    backoff, and a fresh dispatch also lands on a different runner IP.
    (A curl_cffi Chrome-impersonating session does NOT help the
    yfinance route: yfinance 0.2.55 cannot handle curl_cffi sessions --
    "'str' object has no attribute 'name'" -- and the yfinance versions
    that can, >=0.2.60, need websockets>=13, which conflicts with
    alpaca-trade-api's websockets<11 pin. Verified on the runner,
    2026-11-10.)
    """
    import time

    errors: list[str] = []
    for attempt in range(1, 4):
        try:
            df = _download_yahooquery(ticker, start, end)
            if len(df) > 0:
                print(f"data source: yahooquery (attempt {attempt})")
                return df
        except Exception as e:
            errors.append(f"yahooquery attempt {attempt}: {e!r}")
        print(f"yahooquery attempt {attempt} returned no data; retrying...")
        time.sleep(15 * attempt)

    raw = None
    last_err = None
    for attempt in range(1, 4):
        try:
            raw = yf.download(
                ticker,
                start=start,
                end=end,
                auto_adjust=False,
                multi_level_index=False,
                progress=False,
            )
        except Exception as e:
            last_err = e
            raw = None
        if raw is not None and len(raw) > 0:
            break
        print(f"download attempt {attempt} returned no data; retrying...")
        time.sleep(20 * attempt)
    if raw is None or len(raw) == 0:
        raise ValueError(
            f"no data fetched for {ticker} {start}..{end} "
            f"(yfinance last error: {last_err}; yahooquery errors: {errors})"
        )
    print("data source: yfinance (fallback)")
    df = raw.reset_index()
    # yfinance columns: Date, Open, High, Low, Close, Adj Close, Volume
    df.columns = [str(c).lower().replace(" ", "") for c in df.columns]
    cols = ["date", "open", "high", "low", "close", "volume"]
    if "adjclose" in df.columns:
        cols.append("adjclose")
    return _finrl_frame(df[cols], ticker)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ticker", default="AAPL")
    ap.add_argument("--timesteps", type=int, default=8000)
    ap.add_argument("--train-start", default="2022-01-03")
    ap.add_argument("--train-end", default="2023-12-30")
    ap.add_argument("--trade-start", default="2023-12-30")
    ap.add_argument("--trade-end", default="2024-06-29")
    ap.add_argument("--initial-amount", type=float, default=INITIAL_AMOUNT)
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    ticker_list = [args.ticker]

    print(f"Ticker: {args.ticker}")
    print(f"Train window: {args.train_start} .. {args.train_end}")
    print(f"Trade window: {args.trade_start} .. {args.trade_end}")
    print(f"Total timesteps: {args.timesteps}")
    print(f"Initial amount: {args.initial_amount}")

    # 1-2. Download + FinRL feature engineering (download a little extra
    # history before the train start is not needed: indicators are filled
    # with ffill/bfill by FeatureEngineer, as in the FinRL tutorials).
    df = download_finrl_frame(args.ticker, args.train_start, args.trade_end)
    print("Downloaded shape:", df.shape)

    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=INDICATORS,
        use_vix=False,
        use_turbulence=False,
        user_defined_feature=False,
    )
    processed = fe.preprocess_data(df)
    print("Processed shape:", processed.shape)

    # 3. Split into train / trade windows (data_split is [start, end) ).
    train = data_split(processed, args.train_start, args.train_end)
    trade = data_split(processed, args.trade_start, args.trade_end)
    print("Train rows (days):", len(train.index.unique()))
    print("Trade rows (days):", len(trade.index.unique()))

    # 4. Environments (FinRL tutorial env_kwargs for a single stock).
    stock_dimension = len(processed.tic.unique())
    state_space = 1 + 2 * stock_dimension + len(INDICATORS) * stock_dimension
    env_kwargs = {
        "hmax": 100,
        "initial_amount": args.initial_amount,
        "num_stock_shares": [0] * stock_dimension,
        "buy_cost_pct": [0.001] * stock_dimension,
        "sell_cost_pct": [0.001] * stock_dimension,
        "state_space": state_space,
        "stock_dim": stock_dimension,
        "tech_indicator_list": INDICATORS,
        "action_space": stock_dimension,
        "reward_scaling": 1e-4,
        "print_verbosity": 5,
    }

    from finrl.agents.stablebaselines3.models import DRLAgent

    # Train
    e_train_gym = StockTradingEnv(df=train, **env_kwargs)
    env_train, _ = e_train_gym.get_sb_env()
    agent = DRLAgent(env=env_train)
    model_a2c = agent.get_model("a2c")
    trained_a2c = DRLAgent.train_model(
        model=model_a2c, tb_log_name="a2c", total_timesteps=args.timesteps
    )
    print("Training finished.")

    # 5. Trade / backtest on the trade window.
    e_trade_gym = StockTradingEnv(
        df=trade, turbulence_threshold=None, risk_indicator_col="vix", **env_kwargs
    )
    df_account_value, df_actions = DRLAgent.DRL_prediction(
        model=trained_a2c, environment=e_trade_gym
    )
    print("Backtest finished.")
    print(df_account_value.head())

    account = df_account_value.copy()
    account.to_csv(os.path.join(args.outdir, "account_value.csv"), index=False)

    initial = float(account["account_value"].iloc[0])
    final = float(account["account_value"].iloc[-1])
    total_return_pct = (final / initial - 1.0) * 100.0

    result_md = f"""# FinRL A2C pilot — result

- **Tool:** FinRL 0.3.7 (`finrl`), A2C via FinRL's `DRLAgent` (stable-baselines3)
- **Ticker:** {args.ticker}
- **Train window:** {args.train_start} to {args.train_end} (data_split end-exclusive: last train day before {args.train_end})
- **Trade (backtest) window:** {args.trade_start} to {args.trade_end} (data_split end-exclusive: last trade day before {args.trade_end})
- **Total training timesteps:** {args.timesteps}
- **Technical indicators (finrl.config.INDICATORS):** {", ".join(INDICATORS)}
- **hmax:** 100 shares per step; **buy/sell cost:** 0.1% each; **reward scaling:** 1e-4

## Backtest outcome (FinRL's own account-value output)

- **Initial portfolio value:** {initial:,.2f}
- **Final portfolio value:** {final:,.2f}
- **Total return:** {total_return_pct:.2f}%

Rows in `account_value.csv`: {len(account)} (one per trade-window step, from
FinRL `DRL_prediction`'s asset memory, paired with the trade env's dates).

> This is a short smoke-test training run (a few thousand timesteps on a
> single stock), not investment research: the agent is barely trained and
> the result mostly shows the pipeline works end to end.
"""
    with open(os.path.join(args.outdir, "result.md"), "w") as f:
        f.write(result_md)
    print(result_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
