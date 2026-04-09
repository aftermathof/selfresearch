from __future__ import annotations

import os
import io
import copy
import random
import contextlib
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from gym import spaces

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
LR = 1e-5
BATCH_SIZE = 64
COMISSION_FEE_PCT = 0.001
SEED = 42

FEATURES = [
    "close",
    "macd",
    "rsi_30",
    "cci_30",
    "dx_30",
    "vol_ratio20",
    "r1",
    "r5",
    "r10",
    "r20",
    "intraday",
    "overnight",
    "range",
]

# 这里不再把 last_action 打包进 observation
# 因为 PolicyGradient 已经会单独传 last_action 给 EIIE
POLICY_INITIAL_FEATURES = len(FEATURES)

SAVE_DIR = "./plots"
LAST_MODEL_PATH = "last_eiie_pg_policy.pth"
BEST_MODEL_PATH = "best_eiie_pg_policy.pth"
HISTORY_CSV_PATH = "training_history.csv"

os.makedirs(SAVE_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# 1. 可复现设置
# =========================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)


# =========================
# 2. 关闭 tqdm 进度条
# =========================
algo_mod.tqdm = lambda x: x


# =========================
# 3. 读已有数据
# =========================
train_df_raw = pd.read_csv("train_data1.csv")
trade_df_raw = pd.read_csv("trade_data1.csv")

use_cols = ["date", "tic"] + FEATURES
train_df_raw = train_df_raw[use_cols].copy()
trade_df_raw = trade_df_raw[use_cols].copy()

train_df_raw["date"] = pd.to_datetime(train_df_raw["date"])
trade_df_raw["date"] = pd.to_datetime(trade_df_raw["date"])

all_df = pd.concat([train_df_raw, trade_df_raw], axis=0, ignore_index=True)
all_df = (
    all_df.drop_duplicates(subset=["date", "tic"])
    .sort_values(["date", "tic"])
    .reset_index(drop=True)
)


# =========================
# 4. 按年份切分
# train: 2014-2021
# val  : 2022
# test : 2023/2024/2025
# =========================
def select_period(df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)
    out = df[(df["date"] >= start_date) & (df["date"] <= end_date)].copy()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    return out


train_df = select_period(all_df, "2016-01-01", "2020-12-31")
val_df = select_period(all_df, "2021-01-01", "2021-12-31")
df_2023 = select_period(all_df, "2023-01-01", "2023-12-31")
df_2024 = select_period(all_df, "2024-01-01", "2024-12-31")
df_2025 = select_period(all_df, "2025-01-01", "2025-12-31")

for name, df in [
    ("train_df", train_df),
    ("val_df", val_df),
    ("df_2023", df_2023),
    ("df_2024", df_2024),
    ("df_2025", df_2025),
]:
    if df.empty:
        raise ValueError(f"{name} 为空，请检查日期范围或数据文件。")

print("=" * 60)
print("Dataset summary")
print(
    f"Train dates: {train_df['date'].nunique()} | range: {train_df['date'].min()} -> {train_df['date'].max()}"
)
print(
    f"Val   dates: {val_df['date'].nunique()} | range: {val_df['date'].min()} -> {val_df['date'].max()}"
)
print(
    f"2023  dates: {df_2023['date'].nunique()} | range: {df_2023['date'].min()} -> {df_2023['date'].max()}"
)
print(
    f"2024  dates: {df_2024['date'].nunique()} | range: {df_2024['date'].min()} -> {df_2024['date'].max()}"
)
print(
    f"2025  dates: {df_2025['date'].nunique()} | range: {df_2025['date'].min()} -> {df_2025['date'].max()}"
)
print("=" * 60)


# =========================
# 5. 自定义归一化
# 只对 close 做相对变化，其余保持原值
# =========================
PRICE_LIKE = ["close"]
STATIONARY = [
    "macd",
    "rsi_30",
    "cci_30",
    "dx_30",
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

    # close -> x_t / x_{t-1}
    for col in PRICE_LIKE:
        prev = df.groupby("tic")[col].shift(1)
        df[col] = df[col] / prev
        df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(1.0)

    # 其余指标仅数值清洗
    for col in STATIONARY:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return df


# =========================
# 6. 包装环境：只返回纯 state(float32 Box)
# 内部仍保留 return_last_action=True
# 这样 PolicyGradient 还能单独使用 last_action，
# 但 observation 不再重复携带它。
# =========================
class PackedStateEnv:
    """
    将 FinRL 原生 Dict observation:
        {"state": (F, N, T), "last_action": (N+1,)}
    打包成纯 float32 Box observation:
        (F, N, T)

    注意：
    - last_action 不再放进 observation
    - 算法仍会通过 PVM 单独传 last_action 给策略网络
    """

    def __init__(self, base_env: PortfolioOptimizationEnv):
        self.base_env = base_env
        self.action_space = base_env.action_space

        n_assets = base_env.portfolio_size
        time_window = base_env._time_window
        packed_shape = (len(FEATURES), n_assets, time_window)

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=packed_shape,
            dtype=np.float32,
        )

    def _pack_obs(self, obs):
        if isinstance(obs, tuple):
            obs = obs[0]

        if not isinstance(obs, dict):
            return np.asarray(obs, dtype=np.float32)

        state = np.asarray(obs["state"], dtype=np.float32)
        return state

    def reset(self):
        obs = self.base_env.reset()
        return self._pack_obs(obs)

    def step(self, action):
        out = self.base_env.step(action)
        if len(out) == 4:
            obs, reward, done, info = out
            return self._pack_obs(obs), reward, done, info
        elif len(out) == 5:
            obs, reward, done, truncated, info = out
            return self._pack_obs(obs), reward, done, truncated, info
        else:
            raise ValueError(f"Unexpected step output length: {len(out)}")

    def render(self, mode="human"):
        return self.base_env.render(mode=mode)

    def seed(self, seed=None):
        if hasattr(self.base_env, "_seed"):
            return self.base_env._seed(seed)
        return None

    def __getattr__(self, item):
        return getattr(self.base_env, item)


# =========================
# 7. 构造环境
# =========================
def make_env(df: pd.DataFrame, cwd: str = "./") -> PackedStateEnv:
    with contextlib.redirect_stdout(io.StringIO()):
        base_env = PortfolioOptimizationEnv(
            df=df.copy(),
            initial_amount=INITIAL_AMOUNT,
            order_df=True,
            return_last_action=True,   # 仍保留，给算法内部单独使用
            normalize_df=mixed_normalize,
            reward_scaling=1,
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
    env = PackedStateEnv(base_env)
    return env


train_env = make_env(train_df)

tmp_obs = train_env.reset()
# print("Packed observation shape:", tmp_obs.shape)
# print("Packed observation dtype:", tmp_obs.dtype)


# =========================
# 8. 工具函数
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


def run_drl_on_existing_env(env, model) -> list[float]:
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


def run_equal_weight_on_existing_env(env, n_assets: int) -> list[float]:
    """
    注意：这不是严格意义上的 Buy-and-Hold，
    而是每一步都执行相同目标权重的等权重再平衡策略。
    """
    action = [0.0] + [1.0 / n_assets] * n_assets
    terminated = False

    with contextlib.redirect_stdout(io.StringIO()):
        env.reset()
        while not terminated:
            _, _, terminated, _ = env.step(action)

    return get_env_portfolio_curve(env)


def evaluate_on_df(name: str, df: pd.DataFrame, model):
    env = make_env(df)
    curve = run_drl_on_existing_env(env, model)
    summarize_curve(name, curve)
    metrics = calc_metrics(curve)
    return curve, metrics


def plot_performance_curve(baseline_curve, drl_curve, title, save_path):
    plt.figure(figsize=(10, 6))
    plt.plot(baseline_curve, label="Equal Weight Rebalance")
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

    actions = np.array(actions_memory, dtype=np.float32)
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


def plot_training_history(history_df: pd.DataFrame, save_dir: str):
    if history_df.empty:
        return

    # Val Sharpe
    plt.figure(figsize=(10, 6))
    plt.plot(history_df["episode"], history_df["train_sharpe"], label="Train Sharpe")
    plt.plot(history_df["episode"], history_df["val_sharpe"], label="Val Sharpe")
    plt.xlabel("Episode")
    plt.ylabel("Sharpe")
    plt.title("Train / Val Sharpe by Episode")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "history_sharpe.png"), dpi=300)
    plt.close()

    # Final value
    plt.figure(figsize=(10, 6))
    plt.plot(history_df["episode"], history_df["train_final"], label="Train Final Value")
    plt.plot(history_df["episode"], history_df["val_final"], label="Val Final Value")
    plt.xlabel("Episode")
    plt.ylabel("Final Portfolio Value")
    plt.title("Train / Val Final Value by Episode")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "history_final_value.png"), dpi=300)
    plt.close()


# =========================
# 9. 建 agent 和 model
# 注意：这里不传 validation_env
# 因为我们在训练循环外部用 fresh env 独立验证
# =========================
agent = DRLAgent(env=train_env)

model = agent.get_model(
    model_name="pg",
    device=device,
    model_kwargs={
        "lr": LR,
        "batch_size": BATCH_SIZE,
        "policy": EIIE,
        "action_noise": 0.01,
    },
    policy_kwargs={
        "initial_features": POLICY_INITIAL_FEATURES,
        "time_window": TIME_WINDOW,
    },
)


# =========================
# 10. 训练 + 每轮独立验证 + 保存 best checkpoint
# =========================
trained_model = model

best_val_sharpe = -np.inf
best_val_final = -np.inf
best_episode = -1

history = []

print("\nStart training...")
for ep in range(EPISODES):
    # 训练 1 个 episode
    with contextlib.redirect_stdout(io.StringIO()):
        trained_model = agent.train_model(model=trained_model, episodes=1)

    print(f"\nEpisode {ep + 1}/{EPISODES}")
    print("-" * 50)

    # 独立训练集评估：重新 make_env，不看训练轨迹本身
    _, train_metrics = evaluate_on_df("Train-Eval EIIE-PG", train_df, trained_model)

    # 独立验证集评估
    _, val_metrics = evaluate_on_df("Val-2022 EIIE-PG", val_df, trained_model)

    history.append(
        {
            "episode": ep + 1,
            "train_final": train_metrics["final_value"],
            "train_cum_return": train_metrics["cum_return"],
            "train_sharpe": train_metrics["sharpe"],
            "train_mdd": train_metrics["max_drawdown"],
            "val_final": val_metrics["final_value"],
            "val_cum_return": val_metrics["cum_return"],
            "val_sharpe": val_metrics["sharpe"],
            "val_mdd": val_metrics["max_drawdown"],
        }
    )

    improved = (
        (val_metrics["sharpe"] > best_val_sharpe)
        or (
            np.isclose(val_metrics["sharpe"], best_val_sharpe)
            and val_metrics["final_value"] > best_val_final
        )
    )

    if improved:
        best_val_sharpe = val_metrics["sharpe"]
        best_val_final = val_metrics["final_value"]
        best_episode = ep + 1
        torch.save(trained_model.train_policy.state_dict(), BEST_MODEL_PATH)
        print(
            f"[Best checkpoint updated] "
            f"episode={best_episode}, "
            f"val_sharpe={best_val_sharpe:.6f}, "
            f"val_final={best_val_final:.4f}"
        )

# 保存最后一轮
torch.save(trained_model.train_policy.state_dict(), LAST_MODEL_PATH)
print(f"\n最后一轮策略参数已保存到 {LAST_MODEL_PATH}")

# 保存 history
history_df = pd.DataFrame(history)
history_df.to_csv(HISTORY_CSV_PATH, index=False, encoding="utf-8-sig")
plot_training_history(history_df, SAVE_DIR)

print(f"训练历史已保存到 {HISTORY_CSV_PATH}")
print(f"最佳模型来自 Episode {best_episode}, best val sharpe = {best_val_sharpe:.6f}")


# =========================
# 11. 加载最佳 checkpoint
# =========================
best_state_dict = torch.load(BEST_MODEL_PATH, map_location=device)
trained_model.train_policy.load_state_dict(best_state_dict)
trained_model.train_policy.to(device)
trained_model.train_policy.eval()

print(f"已加载验证集最佳策略参数: {BEST_MODEL_PATH}")


# =========================
# 12. 分年份回测 + plot
# 只用 best checkpoint 跑 2023/2024/2025
# =========================
def backtest_one_year(year: int, raw_df: pd.DataFrame, model):
    n_assets = raw_df["tic"].nunique()

    # baseline 和 drl 分开建环境，避免相互覆盖 memory
    bh_env = make_env(raw_df)
    drl_env = make_env(raw_df)

    bh_curve = run_equal_weight_on_existing_env(bh_env, n_assets)
    drl_curve = run_drl_on_existing_env(drl_env, model)

    print()
    summarize_curve(f"{year} Equal Weight Rebalance", bh_curve)
    summarize_curve(f"{year} EIIE-PG", drl_curve)

    plot_performance_curve(
        baseline_curve=bh_curve,
        drl_curve=drl_curve,
        title=f"Performance in {year}",
        save_path=os.path.join(SAVE_DIR, f"performance_{year}.png"),
    )

    plot_actions_memory(
        env=drl_env,
        title=f"Actions performed in {year}",
        save_path=os.path.join(SAVE_DIR, f"actions_{year}.png"),
    )


backtest_one_year(2023, df_2023, trained_model)
backtest_one_year(2024, df_2024, trained_model)
backtest_one_year(2025, df_2025, trained_model)

print("\nAll done.")