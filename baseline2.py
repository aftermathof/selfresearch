import pandas as pd
import numpy as np
import scipy.optimize as sco

# ======================
RISK_FREE_RATE_ANNUAL = 0
RISK_FREE_RATE_DAILY = RISK_FREE_RATE_ANNUAL / 252
TRADING_DAYS = 252
INITIAL_CAPITAL = 1000000

# ======================
# 加载价格数据
# ======================
def load_price_matrix(file_path):
    df = pd.read_csv(file_path)
    # 构造：行=日期，列=股票，值=收盘价
    price_df = df.pivot(index="date", columns="tic", values="close")
    price_df = price_df.dropna()
    return price_df

# ======================
# 组合指标计算
# ======================
def portfolio_perf(weights, mean_ret, cov_mat):
    port_ret = np.sum(mean_ret * weights)
    port_std = np.sqrt(np.dot(weights.T, np.dot(cov_mat, weights)))
    # 夏普率：rf=0，和 FinRL 一致
    sharpe = port_ret / port_std * np.sqrt(TRADING_DAYS) if port_std > 0 else 0
    return port_ret, port_std, sharpe

# 最大化夏普率
def max_sharpe(weights, mean_ret, cov_mat):
    return -portfolio_perf(weights, mean_ret, cov_mat)[2]

# 马科维茨优化
def mpt_optimize(prices):
    daily_ret = prices.pct_change().dropna()
    mean_ret = daily_ret.mean()
    cov_mat = daily_ret.cov()
    n = len(mean_ret)

    # 约束：权重和=1，0~1之间（不做空）
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds = tuple((0, 1) for _ in range(n))
    init_weights = np.ones(n) / n

    res = sco.minimize(
        max_sharpe,
        init_weights,
        args=(mean_ret, cov_mat),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    return res.x

# ======================
# 回测 & 指标计算
# ======================
def backtest(prices, weights):
    daily_ret = prices.pct_change().dropna()
    port_ret = daily_ret.dot(weights)

    # 资产曲线
    cum_ret = (1 + port_ret).cumprod()
    final_val = INITIAL_CAPITAL * cum_ret.iloc[-1]
    total_ret = cum_ret.iloc[-1]

    # 最大回撤
    peak = cum_ret.cummax()
    dd = (cum_ret - peak) / peak
    max_dd = dd.min()

    # 夏普率（rf=0）
    sharpe = port_ret.mean() / port_ret.std() * np.sqrt(TRADING_DAYS) if port_ret.std() > 0 else 0

    return {
        "Initial portfolio value":1000000,
        "Final portfolio value": round(final_val, 2),
        "Final accumulative portfolio value": round(total_ret, 4),
        "max_drawdown": round(max_dd, 4),
        "sharpe ratio": round(sharpe, 4)
    }

# ======================
# 主运行
# ======================
if __name__ == "__main__":
    prices = load_price_matrix("trade_data.csv")
    optimal_weights = mpt_optimize(prices)
    mpt_result = backtest(prices, optimal_weights)

    print("=" * 60)
    print("📊 马科维茨 MPT Baseline")
    print("=" * 60)
    for k, v in mpt_result.items():
        print(f"{k}: {v}")
    print("=" * 60)