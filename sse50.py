import os
import time
import akshare as ak
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ===================== Core Configuration =====================
SYMBOL = "000016"          # SSE 50 Index
SYMBOL_SH = "sh000016"     # Sina/Tencent APIs require market prefix
START_DATE = "20140101"
END_DATE = "20251231"


# ===================== Network Configuration =====================
# Keep these lines only if you are sure no proxy is needed.
os.environ["HTTP_PROXY"] = ""
os.environ["HTTPS_PROXY"] = ""
os.environ["http_proxy"] = ""
os.environ["https_proxy"] = ""

print("AKShare version:", ak.__version__)

# ===================== Data Download Utilities =====================
def standardize_df_from_eastmoney(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={
        "日期": "Date",
        "开盘": "Open",
        "收盘": "Close",
        "最高": "High",
        "最低": "Low",
        "成交量": "Volume",
        "成交额": "Amount",
    })
    df["Date"] = pd.to_datetime(df["Date"])
    num_cols = ["Open", "Close", "High", "Low", "Volume", "Amount"]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.sort_values("Date").reset_index(drop=True)


def standardize_df_from_tx(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={
        "date": "Date",
        "open": "Open",
        "close": "Close",
        "high": "High",
        "low": "Low",
        "amount": "Volume",
    })
    df["Date"] = pd.to_datetime(df["Date"])
    num_cols = ["Open", "Close", "High", "Low", "Volume"]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.sort_values("Date").reset_index(drop=True)


def standardize_df_from_sina(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={
        "date": "Date",
        "open": "Open",
        "close": "Close",
        "high": "High",
        "low": "Low",
        "volume": "Volume",
    })
    df["Date"] = pd.to_datetime(df["Date"])
    num_cols = ["Open", "Close", "High", "Low", "Volume"]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.sort_values("Date").reset_index(drop=True)


def slice_date(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)
    return df[(df["Date"] >= start_dt) & (df["Date"] <= end_dt)].copy()


def download_index_data(symbol="000016", start="20140101", end="20251231", retries=2):
    errors = []

    # ---------- Source 1: Eastmoney ----------
    for i in range(retries):
        try:
            print(f"[Eastmoney] Attempt {i + 1}...")
            df = ak.index_zh_a_hist(
                symbol=symbol,
                period="daily",
                start_date=start,
                end_date=end,
            )
            if df is not None and not df.empty:
                print("[Eastmoney] Download succeeded")
                return standardize_df_from_eastmoney(df), "Eastmoney"
        except Exception as e:
            errors.append(f"Eastmoney attempt {i + 1}: {repr(e)}")
            print(f"[Eastmoney] Attempt {i + 1} failed: {e}")
            time.sleep(2)

    # ---------- Source 2: Tencent ----------
    try:
        print("[Tencent] Falling back...")
        df = ak.stock_zh_index_daily_tx(
            symbol=f"sh{symbol}" if not symbol.startswith(("sh", "sz")) else symbol
        )
        df = standardize_df_from_tx(df)
        df = slice_date(df, start, end)
        if df is not None and not df.empty:
            print("[Tencent] Download succeeded")
            return df, "Tencent"
    except Exception as e:
        errors.append(f"Tencent fallback: {repr(e)}")
        print(f"[Tencent] Failed: {e}")

    # ---------- Source 3: Sina ----------
    try:
        print("[Sina] Falling back...")
        df = ak.stock_zh_index_daily(
            symbol=f"sh{symbol}" if not symbol.startswith(("sh", "sz")) else symbol
        )
        df = standardize_df_from_sina(df)
        df = slice_date(df, start, end)
        if df is not None and not df.empty:
            print("[Sina] Download succeeded")
            return df, "Sina"
    except Exception as e:
        errors.append(f"Sina fallback: {repr(e)}")
        print(f"[Sina] Failed: {e}")

    raise RuntimeError("All data sources failed:\n" + "\n".join(errors))


# ===================== Download Data =====================
df, source = download_index_data(symbol=SYMBOL, start=START_DATE, end=END_DATE)

print(f"\nFinal data source: {source}")
print(f"Download completed: {len(df)} trading days")
print(df.head())
print(df.tail())

# ===================== Plot Styling =====================
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.sans-serif": ["SimHei", "Microsoft YaHei", "DejaVu Sans"],
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,

    "figure.dpi": 120,
    "savefig.dpi": 400,

    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 15,
    "axes.titleweight": "semibold",
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,

    "axes.linewidth": 0.9,
    "grid.linewidth": 0.5,
    "grid.alpha": 0.22,
    "lines.linewidth": 2.0,
})

# ===================== Plot Figure =====================
fig, ax = plt.subplots(figsize=(12.5, 5.6))

ax.plot(
    df["Date"],
    df["Close"],
    color="#4C72B0",
    linewidth=2.0,
    label=f"SSE 50 Close",
    zorder=3
)

ax.set_title("SSE 50 Index Trend from 2014 to 2025", pad=10)
ax.set_xlabel("Date")
ax.set_ylabel("Index Level")

ax.xaxis.set_major_locator(mdates.YearLocator(2))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
plt.setp(ax.get_xticklabels(), rotation=0, ha="center")

ax.grid(True, axis="y", linestyle="--", color="#B0B0B0")
ax.set_axisbelow(True)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("#444444")
ax.spines["bottom"].set_color("#444444")
ax.tick_params(axis="both", which="major", length=3.5, width=0.8, color="#444444")

ax.legend(
    loc="upper left",
    frameon=True,
    framealpha=0.95,
    facecolor="white",
    edgecolor="#D0D0D0",
    borderpad=0.4,
    handlelength=2.2
)

fig.subplots_adjust(left=0.09, right=0.98, top=0.90, bottom=0.12)


plt.savefig("sse50_2014_2025.png", dpi=400)
plt.show()
