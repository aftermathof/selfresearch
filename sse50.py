import os
import time
import akshare as ak
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ===================== 核心配置 =====================
SYMBOL = "000016"          # 上证50
SYMBOL_SH = "sh000016"     # 新浪/腾讯接口需要带市场前缀
START_DATE = "20140101"
END_DATE = "20251231"

# ===================== 网络环境处理 =====================
# 如果你明确不走代理，可以保留；如果你本来就需要科学网络/公司代理，就删掉这几行
os.environ["HTTP_PROXY"] = ""
os.environ["HTTPS_PROXY"] = ""
os.environ["http_proxy"] = ""
os.environ["https_proxy"] = ""

print("AKShare version:", ak.__version__)

# ===================== 数据下载函数 =====================
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
        "amount": "Volume",   # 腾讯文档里这里是 amount，单位是手
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

    # ---------- 方案1：东方财富 ----------
    for i in range(retries):
        try:
            print(f"[Eastmoney] 尝试第 {i+1} 次...")
            df = ak.index_zh_a_hist(
                symbol=symbol,
                period="daily",
                start_date=start,
                end_date=end,
            )
            if df is not None and not df.empty:
                print("[Eastmoney] 下载成功")
                return standardize_df_from_eastmoney(df), "Eastmoney"
        except Exception as e:
            errors.append(f"Eastmoney attempt {i+1}: {repr(e)}")
            print(f"[Eastmoney] 第 {i+1} 次失败: {e}")
            time.sleep(2)

    # ---------- 方案2：腾讯 ----------
    try:
        print("[Tencent] 开始回退...")
        df = ak.stock_zh_index_daily_tx(symbol=f"sh{symbol}" if not symbol.startswith(("sh", "sz")) else symbol)
        df = standardize_df_from_tx(df)
        df = slice_date(df, start, end)
        if df is not None and not df.empty:
            print("[Tencent] 下载成功")
            return df, "Tencent"
    except Exception as e:
        errors.append(f"Tencent fallback: {repr(e)}")
        print(f"[Tencent] 失败: {e}")

    # ---------- 方案3：新浪 ----------
    try:
        print("[Sina] 开始回退...")
        df = ak.stock_zh_index_daily(symbol=f"sh{symbol}" if not symbol.startswith(("sh", "sz")) else symbol)
        df = standardize_df_from_sina(df)
        df = slice_date(df, start, end)
        if df is not None and not df.empty:
            print("[Sina] 下载成功")
            return df, "Sina"
    except Exception as e:
        errors.append(f"Sina fallback: {repr(e)}")
        print(f"[Sina] 失败: {e}")

    raise RuntimeError("所有数据源都失败了：\n" + "\n".join(errors))


# ===================== 下载数据 =====================
df, source = download_index_data(symbol=SYMBOL, start=START_DATE, end=END_DATE)

print(f"\n最终使用数据源: {source}")
print(f"数据下载完成！共 {len(df)} 条交易日数据")
print(df.head())
print(df.tail())

# ===================== 绘图 =====================
plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(df["Date"], df["Close"], linewidth=1.2, label=f"上证50收盘价 ({source})")

ax.set_title("上证50指数 2014-2025 走势", fontsize=16, pad=20)
ax.set_xlabel("日期")
ax.set_ylabel("指数点位")

ax.xaxis.set_major_locator(mdates.YearLocator(2))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
plt.xticks(rotation=45)

ax.grid(alpha=0.3, linestyle="--")
ax.legend()
plt.tight_layout()

plt.savefig("sse50_2014_2025.png", dpi=300, bbox_inches="tight")
print("图表已保存为：sse50_2014_2025.png")
plt.show()