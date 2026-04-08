import akshare as ak
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import os

for k in [
    "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
    "http_proxy", "https_proxy", "all_proxy",
]:
    os.environ.pop(k, None)

os.environ["NO_PROXY"] = "*"
os.environ["no_proxy"] = "*"

# ===================== 核心配置 =====================
SYMBOL = "000016"   # 上证50
START_DATE = "20140101"
END_DATE = "20251231"

# ===================== 1. 下载数据 =====================
print("正在下载上证50指数数据...")
df = ak.index_zh_a_hist(
    symbol=SYMBOL,
    period="daily",
    start_date=START_DATE,
    end_date=END_DATE,
)

if df.empty:
    raise ValueError("AKShare 返回空数据，请检查日期范围或网络。")

# 统一列名
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
df = df.sort_values("Date").reset_index(drop=True)

print(f"数据下载完成！共 {len(df)} 条交易日数据")
print(df.head())

# ===================== 2. 绘图 =====================
plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(df["Date"], df["Close"], linewidth=1.2, label="上证50收盘价")

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