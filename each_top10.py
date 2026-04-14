import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.font_manager import FontProperties

from finrl.meta.preprocessor.yahoodownloader import YahooDownloader


os.environ["HTTP_PROXY"] = "http://127.0.0.1:7897"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7897"


# =========================
# 1. 股票列表
# =========================
TOP_BRL = [
    "601012.SS",
    "600887.SS",
    "600030.SS",
    "600031.SS",
    "600036.SS",
    "600276.SS",
    "600519.SS",
    "601318.SS",
    "600703.SS",
    "601166.SS",
]

START_DATE = "2014-01-01"
END_DATE = "2025-12-31"


# =========================
# 2. 字体与样式
# =========================
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.sans-serif": ["SimSun", "Songti SC", "STSong", "SimHei", "Microsoft YaHei", "DejaVu Sans"],
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,
    "figure.dpi": 120,
    "savefig.dpi": 400,
    "font.size": 10.5,
    "axes.labelsize": 11.5,
    "axes.titlesize": 12,
    "axes.titleweight": "semibold",
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.linewidth": 0.85,
    "grid.linewidth": 0.5,
    "grid.alpha": 0.18,
    "lines.linewidth": 1.8,
})

TNR_FONT = FontProperties(family="Times New Roman")
SONGTI_FONT = FontProperties(family="SimSun")


def has_chinese(text) -> bool:
    text = str(text)
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


def choose_font(text):
    return SONGTI_FONT if has_chinese(text) else TNR_FONT


def apply_font_to_text(text_obj) -> None:
    if text_obj is None:
        return
    text_obj.set_fontproperties(choose_font(text_obj.get_text()))


def style_axis(ax, grid_axis="y") -> None:
    ax.grid(True, axis=grid_axis, linestyle=(0, (3, 3)), color="#B8B8B8")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#444444")
    ax.spines["bottom"].set_color("#444444")
    ax.tick_params(axis="both", which="major", length=3.2, width=0.75, color="#444444")


def apply_fonts_to_axis(ax) -> None:
    apply_font_to_text(ax.title)
    apply_font_to_text(ax.xaxis.label)
    apply_font_to_text(ax.yaxis.label)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        apply_font_to_text(label)
    legend = ax.get_legend()
    if legend is not None:
        for text in legend.get_texts():
            apply_font_to_text(text)
        if legend.get_title() is not None:
            apply_font_to_text(legend.get_title())


# =========================
# 3. 下载数据
# =========================
raw_df = YahooDownloader(
    start_date=START_DATE,
    end_date=END_DATE,
    ticker_list=TOP_BRL,
).fetch_data()

df = raw_df[["date", "tic", "close"]].copy()
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["date", "tic"])

# print(df.head())
# print(df["tic"].value_counts())


# =========================
# 4. 转为宽表
# =========================
close_wide = df.pivot(index="date", columns="tic", values="close").sort_index()
close_wide = close_wide.ffill()

# print(close_wide.head())


# =========================
# 4.1 描述性统计
# =========================
desc_stats = (
    close_wide.describe()
    .T[["mean", "std", "min", "max"]]
    .round(3)
)

print("\n" + "=" * 72)
print("10支股票收盘价描述性统计（均值、标准差、最小值、最大值）")
print("=" * 72)
print(desc_stats.to_string())

locator = mdates.YearLocator(2)
formatter = mdates.DateFormatter("%Y")


# =========================
# 5. 图1：10只股票原始收盘价小 multiples
# =========================
fig, axes = plt.subplots(5, 2, figsize=(13.2, 15.2), sharex=True)
axes = axes.flatten()

line_color = "#4C72B0"

for i, tic in enumerate(TOP_BRL):
    ax = axes[i]
    ax.plot(
        close_wide.index,
        close_wide[tic],
        color=line_color,
        linewidth=1.7,
    )
    ax.set_title(tic, pad=6)
    ax.set_ylabel("Close Price")
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    style_axis(ax, grid_axis="y")
    apply_fonts_to_axis(ax)

for ax in axes[-2:]:
    ax.set_xlabel("Date")
    apply_font_to_text(ax.xaxis.label)

plt.setp(axes[-1].get_xticklabels(), rotation=0)
plt.setp(axes[-2].get_xticklabels(), rotation=0)

fig.suptitle(
    "Close Price Time Series of the Selected 10 Stocks",
    fontsize=16,
    y=0.992,
    fontproperties=TNR_FONT,
)
fig.subplots_adjust(left=0.08, right=0.98, bottom=0.06, top=0.95, hspace=0.34, wspace=0.18)
fig.savefig("top10_close_timeseries.png", bbox_inches="tight")
# plt.show()


# =========================
# 6. 图2：归一化后收盘价对比
# =========================
normalized_close = close_wide / close_wide.iloc[0]

fig, ax = plt.subplots(figsize=(12.8, 6.2))
palette = [
    "#4C72B0", "#C44E52", "#55A868", "#8172B2", "#CCB974",
    "#64B5CD", "#E17C05", "#76B7B2", "#B07AA1", "#8C8C8C",
]

for color, tic in zip(palette, TOP_BRL):
    ax.plot(
        normalized_close.index,
        normalized_close[tic],
        linewidth=1.7,
        color=color,
        alpha=0.96,
        label=tic,
    )

ax.set_title("Normalized Close Price Comparison (Start = 1)", pad=8)
ax.set_xlabel("Date")
ax.set_ylabel("Normalized Close Price")
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(formatter)

style_axis(ax, grid_axis="both")

legend = ax.legend(
    loc="upper left",
    bbox_to_anchor=(1.01, 1.00),
    frameon=False,
    ncol=1,
    handlelength=1.8,
    borderaxespad=0.0,
)

apply_fonts_to_axis(ax)
for text in legend.get_texts():
    apply_font_to_text(text)

fig.subplots_adjust(left=0.08, right=0.80, bottom=0.13, top=0.90)
fig.savefig("each_top10.png", bbox_inches="tight")
# plt.show()
