import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
import scienceplots
import akshare as ak


# 1. 下载上证50指数数据
df_raw = ak.stock_zh_index_daily(symbol="sh000016")
df_raw["date"] = pd.to_datetime(df_raw["date"])
df_raw = df_raw.set_index("date")
df = df_raw[["close"]].rename(columns={"close": "Close"})
df = df[(df.index >= "2023-01-01") & (df.index <= "2025-12-31")].dropna()

print(f"数据范围: {df.index[0].date()} -> {df.index[-1].date()}，共 {len(df)} 个交易日")


# 2. 按半年划分并计算区间收益与波动率
periods = [
    ("2023H1", "2023-01-01", "2023-06-30"),
    ("2023H2", "2023-07-01", "2023-12-31"),
    ("2024H1", "2024-01-01", "2024-06-30"),
    ("2024H2", "2024-07-01", "2024-12-31"),
    ("2025H1", "2025-01-01", "2025-06-30"),
    ("2025H2", "2025-07-01", "2025-12-31"),
]

print("\n各半年区间累计收益率:")
print("-" * 52)

period_stats = []
for name, start, end in periods:
    seg = df[(df.index >= start) & (df.index <= end)]["Close"]
    if len(seg) < 2:
        print(f"{name}: 数据不足，跳过")
        continue

    ret = (seg.iloc[-1] / seg.iloc[0]) - 1.0
    daily_ret = seg.pct_change().dropna()
    vol = daily_ret.std() * np.sqrt(252)

    period_stats.append({
        "name": name,
        "calendar_start": pd.to_datetime(start),
        "calendar_end": pd.to_datetime(end),
        "start": seg.index[0],
        "end": seg.index[-1],
        "return": ret,
        "volatility": vol,
        "start_price": seg.iloc[0],
        "end_price": seg.iloc[-1],
    })

    print(
        f"{name}: 收益率={ret:+.2%}，年化波动率={vol:.2%}，"
        f"{seg.index[0].date()} -> {seg.index[-1].date()}"
    )


# 3. 根据收益率划分市场状态
BULL_THRESHOLD = 0.05
BEAR_THRESHOLD = -0.05

regime_colors = {
    "Bull": "#C44E52",
    "Bear": "#4C72B0",
    "Sideways": "#DD8452",
}

print("\n市场状态划分:")
print("-" * 52)
for p in period_stats:
    if p["return"] > BULL_THRESHOLD:
        regime = "Bull"
    elif p["return"] < BEAR_THRESHOLD:
        regime = "Bear"
    else:
        regime = "Sideways"

    p["regime"] = regime
    print(f"{p['name']}: {regime} (收益率={p['return']:+.2%})")


# 4. 绘图
with plt.style.context(["science", "no-latex"]):
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
        "axes.titlesize": 14,
        "axes.titleweight": "semibold",
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 9.5,
        "axes.linewidth": 0.9,
        "grid.linewidth": 0.5,
        "grid.alpha": 0.22,
        "lines.linewidth": 2.0,
    })

    fig, ax = plt.subplots(figsize=(12.6, 5.4))

    y_min = df["Close"].min()
    y_max = df["Close"].max()
    y_span = y_max - y_min if y_max > y_min else 1.0
    label_y = y_max - 0.04 * y_span

    for p in period_stats:
        color = regime_colors[p["regime"]]
        ax.axvspan(
            p["calendar_start"],
            p["calendar_end"] + pd.Timedelta(days=1),
            alpha=0.10,
            color=color,
            zorder=0,
        )

        mid = p["start"] + (p["end"] - p["start"]) / 2
        ax.text(
            mid,
            label_y,
            f"{p['name']}\n{p['regime']}\n{p['return']:+.1%}",
            ha="center",
            va="top",
            fontsize=8.5,
            color=color,
            fontweight="semibold",
            zorder=3,
            bbox=dict(
                boxstyle="round,pad=0.25",
                facecolor="white",
                edgecolor="none",
                alpha=0.80,
            ),
        )

    ax.plot(
        df.index,
        df["Close"],
        color="#2F3E46",
        linewidth=2.1,
        alpha=0.98,
        label="SSE 50 Index",
        zorder=2,
    )

    for p in period_stats[1:]:
        ax.axvline(
            p["start"],
            color="#8C8C8C",
            linewidth=0.8,
            linestyle="--",
            alpha=0.7,
            zorder=1,
        )

    legend_handles = [
        plt.Line2D([], [], color="#2F3E46", linewidth=2.1, label="SSE 50 Index"),
        mpatches.Patch(color=regime_colors["Bull"], alpha=0.5, label="Bull (> +5%)"),
        mpatches.Patch(color=regime_colors["Bear"], alpha=0.5, label="Bear (< -5%)"),
        mpatches.Patch(color=regime_colors["Sideways"], alpha=0.5, label="Sideways (|r| ≤ 5%)"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper left",
        frameon=True,
        framealpha=0.95,
        facecolor="white",
        edgecolor="#D0D0D0",
        borderpad=0.4,
        handlelength=2.2,
    )

    ax.set_title("SSE 50 Market Regime Segmentation (2023-2025)", pad=10)
    ax.set_xlabel("Date")
    ax.set_ylabel("Index Level")

    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=25, ha="right")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#444444")
    ax.spines["bottom"].set_color("#444444")
    ax.minorticks_off()
    ax.tick_params(
        axis="both",
        which="major",
        length=5.5,
        width=0.8,
        color="#444444",
        top=False,
        right=False,
        bottom=True,
        left=True,
    )

    ax.grid(True, axis="y", linestyle="--", color="#B0B0B0")
    ax.set_axisbelow(True)

    fig.subplots_adjust(left=0.08, right=0.98, top=0.89, bottom=0.16)
    fig.savefig("sse50_divide.png", bbox_inches="tight")
    plt.show()


# 5. 汇总表
print("\n汇总表:")
print(f"{'区间':<8} {'状态':<10} {'收益率':>10} {'年化波动率':>12} {'起始点位':>10} {'结束点位':>10}")
print("-" * 70)
for p in period_stats:
    print(
        f"{p['name']:<8} {p['regime']:<10} "
        f"{p['return']:>+10.2%} {p['volatility']:>12.2%} "
        f"{float(p['start_price']):>10.1f} {float(p['end_price']):>10.1f}"
    )
