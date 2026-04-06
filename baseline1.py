import pandas as pd
import numpy as np
from finrl.meta.env_portfolio_optimization.env_portfolio_optimization import PortfolioOptimizationEnv

# =========================
# 1. 读数据
# =========================
trade_df = pd.read_csv("trade_data.csv")

FEATURES = ["close", "high", "low", "open", "boll_ub", "boll_lb", "close_30_sma"]
use_cols = ["date", "tic"] + FEATURES
trade_df = trade_df[use_cols].copy()
trade_df["date"] = pd.to_datetime(trade_df["date"]).dt.strftime("%Y-%m-%d")

assert trade_df.groupby("date")["tic"].nunique().min() == trade_df["tic"].nunique()

# =========================
# 2. 建测试环境
# =========================
TIME_WINDOW = 50

trade_env = PortfolioOptimizationEnv(
    df=trade_df,
    initial_amount=1_000_000,
    order_df=True,
    return_last_action=False,
    normalize_df="by_previous_time",
    reward_scaling=1,
    comission_fee_model="trf",
    comission_fee_pct=0.001,
    features=FEATURES,
    valuation_feature="close",
    time_column="date",
    time_format="%Y-%m-%d",
    tic_column="tic",
    tics_in_portfolio="all",
    time_window=TIME_WINDOW,
    cwd="./",
    new_gym_api=False,
)

# =========================
# 3. 等权重 baseline
# =========================
def calc_metrics_from_env(env):
    vals = np.array(env._asset_memory["final"], dtype=float)
    rets = np.diff(vals) / vals[:-1]
    sharpe = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else 0.0
    peak = np.maximum.accumulate(vals)
    max_dd = ((vals - peak) / peak).min()
    return {
        "final_value": float(vals[-1]),
        "accumulative_return": float(vals[-1] / vals[0]),
        "sharpe": float(sharpe),
        "max_drawdown": float(max_dd),
    }

def run_equal_weight_baseline(env):
    obs = env.reset()
    done = False

    n_assets = env.portfolio_size
    weights = np.array([0.0] + [1.0 / n_assets] * n_assets, dtype=np.float32)

    while not done:
        obs, reward, done, info = env.step(weights)

    return calc_metrics_from_env(env)

ew_result = run_equal_weight_baseline(trade_env)
print("Equal Weight Baseline:", ew_result)