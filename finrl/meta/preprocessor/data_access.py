from __future__ import annotations

import pandas as pd

from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader


TRAIN_START_DATE = "2014-01-01"
TRAIN_END_DATE = "2023-12-31"

TEST_START_DATE = "2024-01-01"
TEST_END_DATE = "2025-12-31"

TRADE_START_DATE = "2024-01-01"
TRADE_END_DATE = "2025-12-31"

INDICATORS = [
    "macd",
    "boll_ub",
    "boll_lb",
    "rsi_30",
    "cci_30",
    "dx_30",
    "close_30_sma",
    "close_60_sma",
]


try:
    from finrl.config_tickers import SSE50_20_TICKER as RAW_TICKER_LIST
except Exception:
    RAW_TICKER_LIST = ['601012.SS', '600030.SS', '603259.SS', '600887.SS', '600031.SS', '601318.SS', '601668.SS', '601688.SS', '600276.SS', '600703.SS', '600048.SS', '601166.SS', '600519.SS', '600036.SS', '601888.SS', '600196.SS']

def to_yahoo_ticker(ticker: str) -> str:
    """
    Convert common China A-share ticker format to Yahoo Finance format.

    Examples:
        600519.SH -> 600519.SS
        000001.SZ -> 000001.SZ
        600519    -> 600519.SS
    """
    ticker = str(ticker).strip().upper()

    if ticker.endswith(".SH"):
        return ticker[:-3] + ".SS"
    if ticker.endswith(".SS"):
        return ticker
    if ticker.endswith(".SZ"):
        return ticker

    code = ticker[:6]
    if code.startswith(("600", "601", "603", "605", "688", "689")):
        return code + ".SS"
    return code + ".SZ"


TICKER_LIST = [to_yahoo_ticker(ticker) for ticker in RAW_TICKER_LIST]


def download_raw_data(ticker_list: list[str]) -> pd.DataFrame:
    print("Downloading raw data from Yahoo Finance...")
    df_raw = YahooDownloader(
        start_date=TRAIN_START_DATE,
        end_date=TRADE_END_DATE,
        ticker_list=ticker_list,
    ).fetch_data()

    df_raw = df_raw.sort_values(["date", "tic"]).reset_index(drop=True)
    print(f"Raw data shape: {df_raw.shape}")
    print(f"Number of tickers downloaded: {df_raw['tic'].nunique()}")
    return df_raw


def process_data(df_raw: pd.DataFrame) -> pd.DataFrame:
    print("Running feature engineering...")
    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=INDICATORS,
        use_vix=False,
        use_turbulence=True,
        user_defined_feature=False,
    )

    df_processed = fe.preprocess_data(df_raw)
    df_processed = df_processed.sort_values(["date", "tic"]).reset_index(drop=True)

    print(f"Processed data shape: {df_processed.shape}")
    print(f"Number of tickers after cleaning: {df_processed['tic'].nunique()}")
    return df_processed


def save_splits(df_processed: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    print("Splitting train/trade data...")
    train = data_split(df_processed, TRAIN_START_DATE, TRAIN_END_DATE)
    trade = data_split(df_processed, TRADE_START_DATE, TRADE_END_DATE)

    train.to_csv("train_data.csv", index=False)
    trade.to_csv("trade_data.csv", index=False)

    print(f"Train shape: {train.shape}")
    print(f"Trade shape: {trade.shape}")
    return train, trade


print("=" * 60)
print("Ticker list used in Yahoo Finance:")
print(TICKER_LIST)
print("=" * 60)

df_raw = download_raw_data(TICKER_LIST)
df_raw.to_csv("raw_data.csv", index=False)

df_processed = process_data(df_raw)
df_processed.to_csv("processed_data.csv", index=False)

train, trade = save_splits(df_processed)

print("=" * 60)
print("Data preparation finished.")
print("Saved files:")
print("- raw_data.csv")
print("- processed_data.csv")
print("- train_data1.csv")
print("- trade_data1.csv")
print("=" * 60)

print("Train tickers:", sorted(train["tic"].unique().tolist()))
print("Trade tickers:", sorted(trade["tic"].unique().tolist()))