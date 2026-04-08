from __future__ import annotations

import numpy as np
import pandas as pd

from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader


# =========================================================
# 1. 日期配置
# =========================================================
TRAIN_START_DATE = "2014-01-01"
TRAIN_END_DATE = "2023-12-31"

TEST_START_DATE = "2024-01-01"
TEST_END_DATE = "2025-12-31"

TRADE_START_DATE = "2024-01-01"
TRADE_END_DATE = "2025-12-31"


# =========================================================
# 2. FinRL 原始技术指标
# =========================================================
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

# 你新增的自定义特征
CUSTOM_FEATURES = [
    "r1",
    "r5",
    "r10",
    "r20",
    "intraday",
    "overnight",
    "range",
    "vol_ratio20",
]


# =========================================================
# 3. 股票池
# =========================================================
try:
    from finrl.config_tickers import SSE50_20_TICKER as RAW_TICKER_LIST
except Exception:
    RAW_TICKER_LIST = [
        "600009.SS", "601088.SS", "601236.SS", "600019.SS", "601818.SS",
        "601800.SS", "601111.SS", "601939.SS", "601628.SS", "601658.SS",
        "601006.SS", "601816.SS", "600606.SS", "601319.SS", "601229.SS"
    ]


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


# =========================================================
# 4. 下载原始数据
# =========================================================
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


# =========================================================
# 5. FinRL 自带特征工程
# =========================================================
def process_data(df_raw: pd.DataFrame) -> pd.DataFrame:
    print("Running FinRL feature engineering...")
    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=INDICATORS,
        use_vix=False,
        use_turbulence=True,
        user_defined_feature=False,
    )

    df_processed = fe.preprocess_data(df_raw)
    df_processed = df_processed.sort_values(["date", "tic"]).reset_index(drop=True)

    print(f"Processed data shape (before custom features): {df_processed.shape}")
    print(f"Number of tickers after cleaning: {df_processed['tic'].nunique()}")
    return df_processed


# =========================================================
# 6. 安全对数比值函数
# =========================================================
def safe_log_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """
    Return log(numerator / denominator) only when both > 0, otherwise NaN.
    """
    out = pd.Series(np.nan, index=numerator.index, dtype="float64")
    mask = (numerator > 0) & (denominator > 0)
    out.loc[mask] = np.log(numerator.loc[mask] / denominator.loc[mask])
    return out


# =========================================================
# 7. 增加自定义特征
# =========================================================
def add_custom_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add:
      - r1, r5, r10, r20: log returns
      - intraday: log(close / open)
      - overnight: log(open / prev_close)
      - range: log(high / low)
      - vol_ratio20: log(volume / MA20(volume))
    """
    print("Adding custom features...")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["tic", "date"]).reset_index(drop=True)

    g = df.groupby("tic", group_keys=False)

    # ---- lagged close for returns ----
    close_lag1 = g["close"].shift(1)
    close_lag5 = g["close"].shift(5)
    close_lag10 = g["close"].shift(10)
    close_lag20 = g["close"].shift(20)

    # ---- log returns ----
    df["r1"] = safe_log_ratio(df["close"], close_lag1)
    df["r5"] = safe_log_ratio(df["close"], close_lag5)
    df["r10"] = safe_log_ratio(df["close"], close_lag10)
    df["r20"] = safe_log_ratio(df["close"], close_lag20)

    # ---- intraday / overnight / range ----
    df["intraday"] = safe_log_ratio(df["close"], df["open"])
    df["overnight"] = safe_log_ratio(df["open"], close_lag1)
    df["range"] = safe_log_ratio(df["high"], df["low"])

    # ---- volume ratio 20 ----
    vol_ma20 = g["volume"].transform(lambda s: s.rolling(window=20, min_periods=20).mean())
    df["vol_ratio20"] = safe_log_ratio(df["volume"], vol_ma20)

    print("Custom features added:")
    print(CUSTOM_FEATURES)

    return df


# =========================================================
# 8. 过滤不完整日期，保证每个交易日所有股票都齐全
# =========================================================
def filter_complete_panel(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only dates where:
      1) all tickers are present
      2) all required features are non-null
    This is important for RL environments expecting a complete panel.
    """
    print("Filtering incomplete dates for a complete panel...")

    df = df.copy()
    expected_n_tickers = df["tic"].nunique()

    required_cols = [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ] + INDICATORS + CUSTOM_FEATURES

    # 如果 turbulence 存在，也要求非空
    if "turbulence" in df.columns:
        required_cols.append("turbulence")

    def is_valid_day(day_df: pd.DataFrame) -> bool:
        if len(day_df) != expected_n_tickers:
            return False
        return day_df[required_cols].notna().all().all()

    valid_dates = (
        df.groupby("date")
        .apply(is_valid_day)
    )

    valid_dates = valid_dates[valid_dates].index
    before_rows = len(df)
    before_dates = df["date"].nunique()

    df = df[df["date"].isin(valid_dates)].copy()
    df = df.sort_values(["date", "tic"]).reset_index(drop=True)

    after_rows = len(df)
    after_dates = df["date"].nunique()

    print(f"Dates before filter: {before_dates}")
    print(f"Dates after  filter: {after_dates}")
    print(f"Rows  before filter: {before_rows}")
    print(f"Rows  after  filter: {after_rows}")

    return df


# =========================================================
# 9. 划分数据并保存
# =========================================================
def save_splits(df_processed: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    print("Splitting train/trade data...")

    train = data_split(df_processed, TRAIN_START_DATE, TRAIN_END_DATE)
    trade = data_split(df_processed, TRADE_START_DATE, TRADE_END_DATE)

    train.to_csv("train_data1.csv", index=False)
    trade.to_csv("trade_data1.csv", index=False)

    print(f"Train shape: {train.shape}")
    print(f"Trade shape: {trade.shape}")

    return train, trade


# =========================================================
# 10. 主程序
# =========================================================
if __name__ == "__main__":
    print("=" * 80)
    print("Ticker list used in Yahoo Finance:")
    print(TICKER_LIST)
    print("=" * 80)

    # 1) 下载原始数据
    df_raw = download_raw_data(TICKER_LIST)
    df_raw.to_csv("raw_data.csv", index=False)

    # 2) FinRL 自带特征
    df_processed = process_data(df_raw)

    # 3) 加自定义特征
    df_processed = add_custom_features(df_processed)

    # 4) 过滤成完整面板
    df_processed = filter_complete_panel(df_processed)

    # 5) 保存完整特征数据
    df_processed.to_csv("processed_data.csv", index=False)

    # 6) train / trade 划分
    train, trade = save_splits(df_processed)

    print("=" * 80)
    print("Data preparation finished.")
    print("Saved files:")
    print("- raw_data.csv")
    print("- processed_data.csv")
    print("- train_data1.csv")
    print("- trade_data1.csv")
    print("=" * 80)

    print("Train tickers:", sorted(train["tic"].unique().tolist()))
    print("Trade tickers:", sorted(trade["tic"].unique().tolist()))

    print("\nColumns in processed_data.csv:")
    print(df_processed.columns.tolist())