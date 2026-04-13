from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

try:
    from scipy.optimize import minimize
except ImportError:
    minimize = None

output_dir = "baseline_results"
os.makedirs(output_dir, exist_ok=True)

# =========================
# 0. 配置
# =========================
TRAIN_CSV = "train_data1.csv"
TRADE_CSV = "trade_data1.csv"

INITIAL_CASH = 1_000_000
RISK_FREE_RATE = 0.0
PERIODS_PER_YEAR = 252


# =========================
# 1. 工具函数：识别列名
# =========================
def infer_column(df: pd.DataFrame, candidates: list[str], name: str) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(
        f"找不到{name}列。当前列名: {list(df.columns)}；尝试候选: {candidates}"
    )


def load_price_matrix(csv_path: str) -> pd.DataFrame:
    """
    把长表转成价格矩阵:
    index = date
    columns = tic
    values = close
    """
    df = pd.read_csv(csv_path)

    date_col = infer_column(df, ["date", "datetime", "Date"], "日期")
    tic_col = infer_column(df, ["tic", "ticker", "symbol", "stock"], "股票代码")
    price_col = infer_column(
        df,
        ["close", "adjcp", "adj_close", "adjusted_close", "Close"],
        "收盘价",
    )

    df[date_col] = pd.to_datetime(df[date_col])
    df = df[[date_col, tic_col, price_col]].copy()
    df = df.dropna()
    df = df.sort_values([date_col, tic_col])

    # 同一天同一股票若重复，保留最后一条
    df = df.drop_duplicates(subset=[date_col, tic_col], keep="last")

    prices = df.pivot(index=date_col, columns=tic_col, values=price_col)
    prices = prices.sort_index()

    # 前向填充
    prices = prices.ffill()

    # 只保留完整列
    prices = prices.dropna(axis=1, how="any")
    prices = prices.dropna(axis=0, how="any")

    if prices.shape[0] < 2:
        raise ValueError(f"{csv_path} 有效交易日少于2天。")
    if prices.shape[1] < 1:
        raise ValueError(f"{csv_path} 没有可用股票。")

    return prices


# =========================
# 2. 绩效指标
# =========================
def calculate_metrics(
    value_series: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> dict:
    values = value_series.astype(float).values
    returns = values[1:] / values[:-1] - 1.0

    total_return = values[-1] / values[0] - 1.0

    n_periods = len(returns)
    if n_periods > 0:
        annual_return = (values[-1] / values[0]) ** (periods_per_year / n_periods) - 1.0
    else:
        annual_return = np.nan

    rf_per_period = (1 + risk_free_rate) ** (1 / periods_per_year) - 1

    if len(returns) > 1 and np.std(returns, ddof=1) > 0:
        sharpe = (
            np.sqrt(periods_per_year)
            * (np.mean(returns) - rf_per_period)
            / np.std(returns, ddof=1)
        )
    else:
        sharpe = np.nan

    running_max = np.maximum.accumulate(values)
    drawdown = values / running_max - 1.0
    max_drawdown = np.min(drawdown)

    return {
        "final_value": values[-1],
        "total_return": total_return,
        "annual_return": annual_return,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
    }


# =========================
# 3. Baseline 1: Buy and Hold
#    trade期初始等权买入，之后不调仓
# =========================
def run_buy_and_hold(
    trade_prices: pd.DataFrame,
    initial_cash: float = 1_000_000,
) -> tuple[pd.Series, pd.DataFrame]:
    n_assets = trade_prices.shape[1]
    init_weights = np.full(n_assets, 1.0 / n_assets)

    first_prices = trade_prices.iloc[0].values
    shares = (initial_cash * init_weights) / first_prices

    portfolio_values = (trade_prices.values * shares).sum(axis=1)
    weights = (trade_prices.values * shares) / portfolio_values.reshape(-1, 1)

    value_series = pd.Series(
        portfolio_values,
        index=trade_prices.index,
        name="Buy_and_Hold",
    )
    weight_df = pd.DataFrame(
        weights,
        index=trade_prices.index,
        columns=trade_prices.columns,
    )

    return value_series, weight_df


# =========================
# 4. Baseline 2: EW
#    trade期每一期都再平衡到 1/N
# =========================
def run_equal_weight_rebalanced(
    trade_prices: pd.DataFrame,
    initial_cash: float = 1_000_000,
) -> tuple[pd.Series, pd.DataFrame]:
    returns = trade_prices.pct_change().fillna(0.0)
    n_assets = trade_prices.shape[1]
    target_weights = np.full(n_assets, 1.0 / n_assets)

    portfolio_values = np.zeros(len(trade_prices))
    portfolio_values[0] = initial_cash

    for t in range(1, len(trade_prices)):
        period_ret = np.dot(target_weights, returns.iloc[t].values)
        portfolio_values[t] = portfolio_values[t - 1] * (1.0 + period_ret)

    weights = np.tile(target_weights, (len(trade_prices), 1))

    value_series = pd.Series(
        portfolio_values,
        index=trade_prices.index,
        name="Equal_Weight",
    )
    weight_df = pd.DataFrame(
        weights,
        index=trade_prices.index,
        columns=trade_prices.columns,
    )

    return value_series, weight_df


# =========================
# 5. Baseline 3: Best Stock
#    用 train_data1 选股，再到 trade_data1 测试
# =========================
def select_best_stock_from_train(train_prices: pd.DataFrame) -> str:
    """
    在训练集上选累计收益最高的股票
    """
    cumulative_returns = train_prices.iloc[-1] / train_prices.iloc[0] - 1.0
    best_ticker = cumulative_returns.idxmax()
    return best_ticker


def run_best_stock_out_of_sample(
    train_prices: pd.DataFrame,
    trade_prices: pd.DataFrame,
    initial_cash: float = 1_000_000,
) -> tuple[pd.Series, pd.DataFrame, str]:
    """
    训练集选冠军股，交易集全仓持有
    """
    common_tickers = sorted(set(train_prices.columns).intersection(set(trade_prices.columns)))
    if len(common_tickers) == 0:
        raise ValueError("train 和 trade 没有共同股票，无法做 Best Stock。")

    train_sub = train_prices[common_tickers].copy()
    trade_sub = trade_prices[common_tickers].copy()

    best_ticker = select_best_stock_from_train(train_sub)

    best_prices = trade_sub[best_ticker]
    shares = initial_cash / best_prices.iloc[0]
    portfolio_values = shares * best_prices.values

    weights = np.zeros((len(trade_sub), trade_sub.shape[1]))
    best_idx = list(trade_sub.columns).index(best_ticker)
    weights[:, best_idx] = 1.0

    value_series = pd.Series(
        portfolio_values,
        index=trade_sub.index,
        name="Best_Stock",
    )
    weight_df = pd.DataFrame(
        weights,
        index=trade_sub.index,
        columns=trade_sub.columns,
    )

    return value_series, weight_df, best_ticker


# =========================
# 6. Baseline 4: Minimum Variance
#    用 train_data1 估计协方差，在 trade_data1 固定权重持有
# =========================
def solve_min_variance_weights(train_prices: pd.DataFrame) -> pd.Series:
    returns = train_prices.pct_change().dropna()
    cov = returns.cov().values
    n_assets = cov.shape[0]

    if n_assets == 0:
        raise ValueError("训练集为空，无法估计最小方差组合。")

    if minimize is not None:
        def portfolio_variance(weights: np.ndarray) -> float:
            return float(weights @ cov @ weights)

        constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
        bounds = [(0.0, 1.0)] * n_assets
        x0 = np.full(n_assets, 1.0 / n_assets)

        result = minimize(
            portfolio_variance,
            x0=x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )
        if result.success:
            weights = result.x
        else:
            weights = x0
    else:
        inv_cov = np.linalg.pinv(cov)
        ones = np.ones(n_assets)
        weights = inv_cov @ ones
        weights = np.clip(weights, 0.0, None)
        if weights.sum() == 0:
            weights = np.full(n_assets, 1.0 / n_assets)
        else:
            weights = weights / weights.sum()

    return pd.Series(weights, index=train_prices.columns, name="Minimum_Variance")


def run_min_variance_out_of_sample(
    train_prices: pd.DataFrame,
    trade_prices: pd.DataFrame,
    initial_cash: float = 1_000_000,
) -> tuple[pd.Series, pd.DataFrame, pd.Series]:
    common_tickers = sorted(set(train_prices.columns).intersection(set(trade_prices.columns)))
    if len(common_tickers) == 0:
        raise ValueError("train 和 trade 没有共同股票，无法做 Minimum Variance。")

    train_sub = train_prices[common_tickers].copy()
    trade_sub = trade_prices[common_tickers].copy()

    weights = solve_min_variance_weights(train_sub).reindex(trade_sub.columns)

    first_prices = trade_sub.iloc[0].values
    shares = (initial_cash * weights.values) / first_prices

    portfolio_values = (trade_sub.values * shares).sum(axis=1)
    drifted_weights = (trade_sub.values * shares) / portfolio_values.reshape(-1, 1)

    value_series = pd.Series(
        portfolio_values,
        index=trade_sub.index,
        name="Minimum_Variance",
    )
    weight_df = pd.DataFrame(
        drifted_weights,
        index=trade_sub.index,
        columns=trade_sub.columns,
    )

    return value_series, weight_df, weights


# =========================
# 7. 主程序
# =========================
train_prices = load_price_matrix(TRAIN_CSV)
trade_prices = load_price_matrix(TRADE_CSV)

# 对齐股票池：只保留 train 和 trade 都存在的股票
common_tickers = sorted(set(train_prices.columns).intersection(set(trade_prices.columns)))
if len(common_tickers) == 0:
    raise ValueError("train_data1.csv 和 trade_data1.csv 没有共同股票代码。")

train_prices = train_prices[common_tickers].copy()
trade_prices = trade_prices[common_tickers].copy()

print("共同股票数:", len(common_tickers))
print("训练集日期区间:", train_prices.index.min().date(), "->", train_prices.index.max().date())
print("交易集日期区间:", trade_prices.index.min().date(), "->", trade_prices.index.max().date())
print()

# 运行 baseline
bh_curve, bh_weights = run_buy_and_hold(trade_prices, INITIAL_CASH)
ew_curve, ew_weights = run_equal_weight_rebalanced(trade_prices, INITIAL_CASH)
bs_curve, bs_weights, best_ticker = run_best_stock_out_of_sample(
    train_prices, trade_prices, INITIAL_CASH
)
mv_curve, mv_weights, mv_target_weights = run_min_variance_out_of_sample(
    train_prices, trade_prices, INITIAL_CASH
)

# 计算绩效
bh_metrics = calculate_metrics(bh_curve, RISK_FREE_RATE, PERIODS_PER_YEAR)
ew_metrics = calculate_metrics(ew_curve, RISK_FREE_RATE, PERIODS_PER_YEAR)
bs_metrics = calculate_metrics(bs_curve, RISK_FREE_RATE, PERIODS_PER_YEAR)
mv_metrics = calculate_metrics(mv_curve, RISK_FREE_RATE, PERIODS_PER_YEAR)

result_df = pd.DataFrame(
    {
        "Buy_and_Hold": bh_metrics,
        "Equal_Weight": ew_metrics,
        "Best_Stock": bs_metrics,
        "Minimum_Variance": mv_metrics,
    }
).T

print("Best Stock（由训练集选出）:", best_ticker)
print("Minimum Variance 权重（前10项）:")
print(mv_target_weights.sort_values(ascending=False).head(10))
print()
print(result_df)
print()

# 保存结果
result_df.to_csv(os.path.join(output_dir, "baseline_metrics.csv"), encoding="utf-8-sig")
bh_curve.to_csv(os.path.join(output_dir, "buy_and_hold_curve.csv"), encoding="utf-8-sig")
ew_curve.to_csv(os.path.join(output_dir, "equal_weight_curve.csv"), encoding="utf-8-sig")
bs_curve.to_csv(os.path.join(output_dir, "best_stock_curve.csv"), encoding="utf-8-sig")
mv_curve.to_csv(os.path.join(output_dir, "minimum_variance_curve.csv"), encoding="utf-8-sig")

bh_weights.to_csv(os.path.join(output_dir, "buy_and_hold_weights.csv"), encoding="utf-8-sig")
ew_weights.to_csv(os.path.join(output_dir, "equal_weight_weights.csv"), encoding="utf-8-sig")
bs_weights.to_csv(os.path.join(output_dir, "best_stock_weights.csv"), encoding="utf-8-sig")
mv_weights.to_csv(os.path.join(output_dir, "minimum_variance_weights.csv"), encoding="utf-8-sig")
mv_target_weights.to_csv(os.path.join(output_dir, "minimum_variance_target_weights.csv"), encoding="utf-8-sig")

# 画资金曲线
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

fig, ax = plt.subplots(figsize=(12.5, 5.6))

ax.plot(
    bh_curve.index,
    bh_curve.values,
    color="#4C72B0",
    label="Buy and Hold",
    zorder=3,
)
ax.plot(
    ew_curve.index,
    ew_curve.values,
    color="#55A868",
    label="Equal Weight",
    zorder=3,
)
ax.plot(
    bs_curve.index,
    bs_curve.values,
    color="#C44E52",
    label=f"Best Stock ({best_ticker})",
    zorder=3,
)
ax.plot(
    mv_curve.index,
    mv_curve.values,
    color="#562626",
    label="Minimum Variance",
    zorder=3,
)

ax.set_title("Baseline Portfolio Value Comparison", pad=10)
ax.set_xlabel("Date")
ax.set_ylabel("Portfolio Value")

ax.xaxis.set_major_locator(mdates.AutoDateLocator())
ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))

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
    handlelength=2.2,
)

fig.subplots_adjust(left=0.09, right=0.98, top=0.90, bottom=0.13)
fig.savefig(os.path.join(output_dir, "baseline_curves.png"), bbox_inches="tight")
plt.show()

