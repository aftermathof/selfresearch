from __future__ import annotations

import os
import io
import copy
import contextlib
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from finrl.meta.env_portfolio_optimization.env_portfolio_optimization import (
    PortfolioOptimizationEnv,
)
from finrl.agents.portfolio_optimization.models import DRLAgent
from finrl.agents.portfolio_optimization.architectures import EIIE
import finrl.agents.portfolio_optimization.algorithms as algo_mod


# =========================
# 0. 基本配置
# =========================
INITIAL_AMOUNT = 1_000_000
TIME_WINDOW = 50
EPISODES = 5
LR = 1e-3
BATCH_SIZE = 64
COMISSION_FEE_PCT = 0.001

FEATURES = [
    "close",
    "high",
    "low",
    "open",
    "boll_ub",
    "boll_lb",
    "macd",
    "rsi_30",
    "cci_30",
    "dx_30",
    "close_30_sma",
    "close_60_sma",
    "vol_ratio20",
    "r1",
    "r5",
    "r10",
    "r20",
    "intraday",
    "overnight",
    "range",
]

SAVE_DIR = "./plots"
MODEL_PATH = "eiie_pg_policy.pth"
os.makedirs(SAVE_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# 1. 关闭 tqdm 进度条
# =========================
algo_mod.tqdm = lambda x: x


# =========================
# 2. 读已有数据
# =========================
train_df_raw = pd.read_csv("train_data1.csv")
trade_df_raw = pd.read_csv("trade_data1.csv")

use_cols = ["date", "tic"] + FEATURES
train_df_raw = train_df_raw[use_cols].copy()
trade_df_raw = trade_df_raw[use_cols].copy()

train_df_raw["date"] = pd.to_datetime(train_df_raw["date"])
trade_df_raw["date"] = pd.to_datetime(trade_df_raw["date"])

# 合并成全样本，后面再按年份切
all_df = pd.concat([train_df_raw, trade_df_raw], axis=0, ignore_index=True)
all_df = all_df.drop_duplicates(subset=["date", "tic"]).sort_values(["date", "tic"]).reset_index(drop=True)


# =========================
# 3. 数据清洗
# =========================
# def clean_panel(df: pd.DataFrame, name: str) -> pd.DataFrame:
#     df = df.copy()

#     for col in FEATURES:
#         df[col] = pd.to_numeric(df[col], errors="coerce")

#     df = df.replace([np.inf, -np.inf], np.nan)

#     # 原始价格必须 > 0
#     for col in ["close", "high", "low", "open"]:
#         df.loc[df[col] <= 0, col] = np.nan

#     n_tic = df["tic"].nunique()
#     bad_dates = []

#     for d, g in df.groupby("date"):
#         if len(g) != n_tic:
#             bad_dates.append(d)
#             continue
#         if not g[FEATURES].notna().all().all():
#             bad_dates.append(d)
#             continue

#     if bad_dates:
#         print(f"[{name}] drop bad dates: {len(bad_dates)}")
#         df = df[~df["date"].isin(bad_dates)].copy()

#     assert df.groupby("date")["tic"].nunique().min() == df["tic"].nunique(), \
#         f"{name} 中存在某些日期股票池不完整"

#     df = df.sort_values(["date", "tic"]).reset_index(drop=True)
#     return df


# all_df = clean_panel(all_df, "all")


# =========================
# 4. 按年份切分：训练 2014-2022，回测 2023/2024/2025
# =========================
def select_period(df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)
    out = df[(df["date"] >= start_date) & (df["date"] <= end_date)].copy()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    return out


train_df = select_period(all_df, "2014-01-01", "2022-12-31")
df_2023 = select_period(all_df, "2023-01-01", "2023-12-31")
df_2024 = select_period(all_df, "2024-01-01", "2024-12-31")
df_2025 = select_period(all_df, "2025-01-01", "2025-12-31")

for name, df in [
    ("train_df", train_df),
    ("df_2023", df_2023),
    ("df_2024", df_2024),
    ("df_2025", df_2025),
]:
    if df.empty:
        raise ValueError(f"{name} 为空，请检查日期范围或数据文件。")

print("=" * 60)
print("Dataset summary")
print(f"Train dates: {train_df['date'].nunique()} | range: {train_df['date'].min()} -> {train_df['date'].max()}")
print(f"2023  dates: {df_2023['date'].nunique()} | range: {df_2023['date'].min()} -> {df_2023['date'].max()}")
print(f"2024  dates: {df_2024['date'].nunique()} | range: {df_2024['date'].min()} -> {df_2024['date'].max()}")
print(f"2025  dates: {df_2025['date'].nunique()} | range: {df_2025['date'].min()} -> {df_2025['date'].max()}")
print("=" * 60)


# =========================
# 5. 自定义归一化
# =========================
PRICE_LIKE = ["close", "high", "low", "open", "boll_ub", "boll_lb"]
STATIONARY = [
    "macd",
    "rsi_30",
    "cci_30",
    "dx_30",
    "close_30_sma",
    "close_60_sma",
    "vol_ratio20",
    "r1",
    "r5",
    "r10",
    "r20",
    "intraday",
    "overnight",
    "range",
]


def mixed_normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["tic", "date"]).copy()

    # 价格型做 x_t / x_{t-1}
    for col in PRICE_LIKE:
        prev = df.groupby("tic")[col].shift(1)
        df[col] = df[col] / prev
        df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(1.0)

    # 已经平稳/收益型特征不再除以前一期
    for col in STATIONARY:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return df


# =========================
# 6. 构造环境：完全按 demo 风格，分段单独实例化
# =========================
def make_env(df: pd.DataFrame, cwd: str = "./") -> PortfolioOptimizationEnv:
    # 静默创建，屏蔽环境内部 print
    with contextlib.redirect_stdout(io.StringIO()):
        env = PortfolioOptimizationEnv(
            df=df.copy(),
            initial_amount=INITIAL_AMOUNT,
            order_df=True,
            return_last_action=False,
            normalize_df=mixed_normalize,
            reward_scaling=100,
            comission_fee_model="trf",
            comission_fee_pct=COMISSION_FEE_PCT,
            features=FEATURES,
            valuation_feature="close",
            time_column="date",
            time_format="%Y-%m-%d",
            tic_column="tic",
            tics_in_portfolio="all",
            time_window=TIME_WINDOW,
            cwd=cwd,
            new_gym_api=False,
        )
    return env


train_env = make_env(train_df)
environment_2023 = make_env(df_2023)
environment_2024 = make_env(df_2024)
environment_2025 = make_env(df_2025)


# =========================
# 7. 工具函数
# =========================
def get_env_portfolio_curve(env) -> list[float]:
    asset_memory = env._asset_memory
    if isinstance(asset_memory, dict):
        curve = asset_memory["final"]
    else:
        curve = asset_memory
    curve = list(curve)
    if len(curve) == 0:
        raise ValueError("Portfolio value curve is empty.")
    return curve


def get_env_action_memory(env):
    if hasattr(env, "_asset_memory"):
        asset_memory = getattr(env, "_asset_memory")
        if isinstance(asset_memory, dict) and "actions" in asset_memory:
            actions = asset_memory["actions"]
            if actions is not None:
                return actions

    candidate_names = [
        "_actions_memory",
        "_action_memory",
        "actions_memory",
        "action_memory",
    ]
    for name in candidate_names:
        if hasattr(env, name):
            memory = getattr(env, name)
            if memory is not None:
                return memory

    return None


def calc_metrics(curve: list[float]) -> dict:
    arr = np.asarray(curve, dtype=float)

    if len(arr) < 2:
        return {
            "final_value": float(arr[-1]) if len(arr) else 0.0,
            "cum_return": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
        }

    rets = np.diff(arr) / arr[:-1]
    sharpe = float(rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else 0.0
    peak = np.maximum.accumulate(arr)
    max_dd = float(((arr - peak) / peak).min())

    return {
        "final_value": float(arr[-1]),
        "cum_return": float(arr[-1] / arr[0]),
        "sharpe": sharpe,
        "max_drawdown": max_dd,
    }


def summarize_curve(name: str, curve: list[float]):
    m = calc_metrics(curve)
    print(f"{name}:")
    print(f"  Initial portfolio value: {curve[0]:.4f}")
    print(f"  Final portfolio value:   {m['final_value']:.4f}")
    print(f"  Cumulative return:       {m['cum_return']:.6f}")
    print(f"  Sharpe ratio:            {m['sharpe']:.6f}")
    print(f"  Maximum DrawDown:        {m['max_drawdown']:.6f}")
    print("-" * 50)


def run_drl_on_existing_env(env: PortfolioOptimizationEnv, model) -> list[float]:
    with contextlib.redirect_stdout(io.StringIO()):
        DRLAgent.DRL_validation(
            model=model,
            test_env=env,
            policy=None,
            online_training_period=10**18,
            learning_rate=0.0,
            optimizer=None,
            batch_size=BATCH_SIZE,
        )
    return get_env_portfolio_curve(env)


def run_equal_weight_on_existing_env(env: PortfolioOptimizationEnv, n_assets: int) -> list[float]:
    action = [0.0] + [1.0 / n_assets] * n_assets
    terminated = False

    with contextlib.redirect_stdout(io.StringIO()):
        env.reset()
        while not terminated:
            _, _, terminated, _ = env.step(action)

    return get_env_portfolio_curve(env)


def plot_performance_curve(baseline_curve, drl_curve, title, save_path):
    plt.figure(figsize=(10, 6))
    plt.plot(baseline_curve, label="Buy and Hold")
    plt.plot(drl_curve, label="EIIE-PG")
    plt.xlabel("Days")
    plt.ylabel("Portfolio Value")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_actions_memory(env, title, save_path):
    actions_memory = get_env_action_memory(env)
    if actions_memory is None:
        print(f"[Warning] Cannot find action memory for: {title}")
        return

    actions = np.array(actions_memory)
    if actions.ndim != 2:
        print(f"[Warning] action memory has unexpected shape: {actions.shape}")
        return

    plt.figure(figsize=(12, 7))
    for i in range(actions.shape[1]):
        plt.plot(actions[:, i], linewidth=1)

    plt.xlabel("Time")
    plt.ylabel("Weight")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


# =========================
# 8. 建 agent 和 model（不加验证集）
# =========================
agent = DRLAgent(env=train_env)

model = agent.get_model(
    model_name="pg",
    device=device,
    model_kwargs={
        "lr": LR,
        "batch_size": BATCH_SIZE,
        "policy": EIIE,
        "action_noise": 0.05,
    },
    policy_kwargs={
        "initial_features": len(FEATURES),
        "time_window": TIME_WINDOW,
    },
)


# =========================
# 9. 训练：每个 episode 只输出 Train EIIE-PG
# =========================
trained_model = model

for ep in range(EPISODES):
    with contextlib.redirect_stdout(io.StringIO()):
        trained_model = agent.train_model(model=trained_model, episodes=1)

    train_curve_ep = get_env_portfolio_curve(train_env)

    print(f"\nEpisode {ep + 1}/{EPISODES}")
    print("-" * 50)
    summarize_curve("Train EIIE-PG", train_curve_ep)

torch.save(trained_model.train_policy.state_dict(), MODEL_PATH)
print(f"\n训练结束，最终策略参数已保存到 {MODEL_PATH}")


# =========================
# 10. 分年份单独回测 + plot
# =========================
def backtest_one_year(year: int, env: PortfolioOptimizationEnv, raw_df: pd.DataFrame, model):
    n_assets = raw_df["tic"].nunique()

    bh_curve = run_equal_weight_on_existing_env(env, n_assets)
    drl_curve = run_drl_on_existing_env(env, model)

    print()
    summarize_curve(f"{year} Buy and Hold", bh_curve)
    summarize_curve(f"{year} EIIE-PG", drl_curve)

    plot_performance_curve(
        baseline_curve=bh_curve,
        drl_curve=drl_curve,
        title=f"Performance in {year}",
        save_path=os.path.join(SAVE_DIR, f"performance_{year}.png"),
    )

    plot_actions_memory(
        env=env,
        title=f"Actions performed in {year}",
        save_path=os.path.join(SAVE_DIR, f"actions_{year}.png"),
    )


backtest_one_year(2023, environment_2023, df_2023, trained_model)
backtest_one_year(2024, environment_2024, df_2024, trained_model)
backtest_one_year(2025, environment_2025, df_2025, trained_model)