from __future__ import annotations

import os
import copy
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from finrl.meta.env_portfolio_optimization.env_portfolio_optimization import (
    PortfolioOptimizationEnv,
)
from finrl.agents.portfolio_optimization.models import DRLAgent
from finrl.agents.portfolio_optimization.architectures import EIIE

import swanlab


# =========================
# 0. 基本配置
# =========================
INITIAL_AMOUNT = 1_000_000
TIME_WINDOW = 50
EPISODES = 20
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
BEST_MODEL_PATH = "eiie_pg_best.pth"
os.makedirs(SAVE_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# 1. SwanLab 初始化
# =========================
swanlab.init(
    project="sse50-portfolio",
    experiment_name="EIIE-PG-train-val-trade",
    mode="cloud",
    config={
        "lr": LR,
        "batch_size": BATCH_SIZE,
        "episodes": EPISODES,
        "time_window": TIME_WINDOW,
        "features": FEATURES,
        "comission_fee_pct": COMISSION_FEE_PCT,
        "device": device,
    },
)


# =========================
# 2. 读数据
# =========================
train_df = pd.read_csv("train_data1.csv")
trade_df = pd.read_csv("trade_data1.csv")

use_cols = ["date", "tic"] + FEATURES
train_df = train_df[use_cols].copy()
trade_df = trade_df[use_cols].copy()

train_df["date"] = pd.to_datetime(train_df["date"]).dt.strftime("%Y-%m-%d")
trade_df["date"] = pd.to_datetime(trade_df["date"]).dt.strftime("%Y-%m-%d")

# 检查每个交易日股票池是否完整
assert train_df.groupby("date")["tic"].nunique().min() == train_df["tic"].nunique(), \
    "train_df 中存在某些日期股票池不完整"
assert trade_df.groupby("date")["tic"].nunique().min() == trade_df["tic"].nunique(), \
    "trade_df 中存在某些日期股票池不完整"


# =========================
# 3. 从训练数据中切出验证集
# =========================
def split_train_val_by_date(
    df: pd.DataFrame,
    val_ratio: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    按日期切分 train/val，保证时间顺序，不打乱。
    """
    unique_dates = sorted(df["date"].unique())
    n_dates = len(unique_dates)

    if n_dates <= TIME_WINDOW + 5:
        raise ValueError(
            f"训练集日期数过少: {n_dates}，不足以支持 time_window={TIME_WINDOW} 的训练/验证划分。"
        )

    split_idx = int(n_dates * (1 - val_ratio))
    split_idx = max(split_idx, TIME_WINDOW + 1)
    split_idx = min(split_idx, n_dates - (TIME_WINDOW + 1))

    train_dates = unique_dates[:split_idx]
    val_dates = unique_dates[split_idx:]

    train_sub_df = df[df["date"].isin(train_dates)].copy()
    val_df = df[df["date"].isin(val_dates)].copy()

    if train_sub_df.empty or val_df.empty:
        raise ValueError("train/val 划分失败，请检查数据量或 val_ratio。")

    return train_sub_df, val_df


train_sub_df, val_df = split_train_val_by_date(train_df, val_ratio=0.2)

print("=" * 60)
print("Data split summary")
print(f"Full train dates: {train_df['date'].nunique()}")
print(f"Train dates:      {train_sub_df['date'].nunique()}")
print(f"Val dates:        {val_df['date'].nunique()}")
print(f"Trade dates:      {trade_df['date'].nunique()}")
print("=" * 60)


# =========================
# 4. 构造环境函数
# =========================
def make_env(df: pd.DataFrame, cwd: str = "./") -> PortfolioOptimizationEnv:
    return PortfolioOptimizationEnv(
        df=df.copy(),
        initial_amount=INITIAL_AMOUNT,
        order_df=True,
        return_last_action=False,
        normalize_df="by_previous_time",
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


train_env = make_env(train_sub_df)
val_env = make_env(val_df)
trade_env = make_env(trade_df)


# =========================
# 5. 工具函数（注意：必须放在训练循环前）
# =========================
def get_env_portfolio_curve(env) -> list[float]:
    """
    提取组合价值曲线。
    优先从 env._asset_memory["final"] 中取。
    """
    if not hasattr(env, "_asset_memory"):
        raise AttributeError("Environment has no attribute '_asset_memory'.")

    asset_memory = env._asset_memory

    if isinstance(asset_memory, dict):
        if "final" in asset_memory:
            curve = asset_memory["final"]
        else:
            raise KeyError("env._asset_memory exists, but key 'final' was not found.")
    else:
        curve = asset_memory

    curve = list(curve)
    if len(curve) == 0:
        raise ValueError("Portfolio value curve is empty.")

    return curve


def get_env_action_memory(env):
    """
    提取动作权重历史。
    优先尝试:
      1) env._asset_memory["actions"]
      2) env._actions_memory / env._action_memory / ...
    """
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


def run_equal_weight_baseline(df: pd.DataFrame) -> tuple[list[float], object]:
    """
    等权 Buy-and-Hold 基线:
    action = [0] + [1/N] * N
    第一个维度通常是 cash。
    """
    env = make_env(df)

    n_assets = df["tic"].nunique()
    action = [0.0] + [1.0 / n_assets] * n_assets

    terminated = False
    env.reset()

    while not terminated:
        _, _, terminated, _ = env.step(action)

    curve = get_env_portfolio_curve(env)
    return curve, env


def run_drl_validation(df: pd.DataFrame, model) -> tuple[list[float], object]:
    """
    在给定数据上跑一次验证/评估，拿到组合价值曲线。
    """
    env = make_env(df)

    DRLAgent.DRL_validation(
        model=model,
        test_env=env,
        policy=None,
        online_training_period=999999,  # 近似关闭 online training
        learning_rate=0,
        optimizer=None,
    )

    curve = get_env_portfolio_curve(env)
    return curve, env


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
    plt.show()


def plot_actions_memory(env, title, save_path):
    """
    画动作权重图。
    如果环境里取不到动作记忆，就跳过。
    """
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
    plt.show()


def summarize_curve(name: str, curve: list[float]):
    m = calc_metrics(curve)
    print(f"{name}:")
    print(f"  Initial portfolio value: {curve[0]:.4f}")
    print(f"  Final portfolio value:   {m['final_value']:.4f}")
    print(f"  Cumulative return:       {m['cum_return']:.6f}")
    print(f"  Sharpe ratio:            {m['sharpe']:.6f}")
    print(f"  Maximum DrawDown:        {m['max_drawdown']:.6f}")
    print("-" * 50)


# =========================
# 6. 建 agent 和 model
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
        "validation_env": val_env,   # 若你的 finrl 版本不支持，可删掉这一行
    },
    policy_kwargs={
        "initial_features": len(FEATURES),
        "time_window": TIME_WINDOW,
    },
)


# =========================
# 7. 训练（逐episode + 最优checkpoint）
# =========================
best_val_value = -np.inf
best_episode = -1
best_state_dict = None

trained_model = model

for ep in range(EPISODES):
    # 每次只训练 1 个 episode，并继续在上一次参数基础上训练
    trained_model = agent.train_model(model=trained_model, episodes=1)

    # 在验证集上评估
    val_curve_ep, _ = run_drl_validation(val_df, trained_model)
    val_final = float(val_curve_ep[-1])

    is_best = val_final > best_val_value

    if is_best:
        best_val_value = val_final
        best_episode = ep + 1
        best_state_dict = copy.deepcopy(trained_model.train_policy.state_dict())

        # 保存完整 checkpoint
        torch.save(
            {
                "episode": best_episode,
                "val_portfolio_value": best_val_value,
                "policy_state_dict": best_state_dict,
            },
            BEST_MODEL_PATH,
        )

    swanlab.log({
        "episode": ep + 1,
        "val_portfolio_value": val_final,
        "best_val_portfolio_value": best_val_value,
    })

    print(
        f"Episode {ep + 1}/{EPISODES} | "
        f"val_value={val_final:.2f} | "
        f"best={best_val_value:.2f}"
    )

    if is_best:
        print(f"  ✓ New best at episode {best_episode}, saved to {BEST_MODEL_PATH}")

if best_state_dict is None:
    raise RuntimeError("训练结束后没有得到有效的 best checkpoint，请检查训练过程。")

print(
    f"\n训练结束，最优 checkpoint 来自 episode {best_episode}，"
    f"val_value={best_val_value:.2f}"
)

# 训练结束后，将最优参数加载回当前模型
trained_model.train_policy.load_state_dict(best_state_dict)
print("已将验证集最优参数加载回当前模型。")

# 再单独保存一份纯策略参数
torch.save(best_state_dict, MODEL_PATH)
print(f"最终最优策略参数已保存到 {MODEL_PATH}")


# =========================
# 8. 训练集表现图：EIIE vs Buy-and-Hold
# =========================
train_bh_curve, train_bh_env = run_equal_weight_baseline(train_sub_df)
train_drl_curve, train_eval_env = run_drl_validation(train_sub_df, trained_model)

plot_performance_curve(
    baseline_curve=train_bh_curve,
    drl_curve=train_drl_curve,
    title="Performance in training period",
    save_path=os.path.join(SAVE_DIR, "train_performance.png"),
)

plot_actions_memory(
    env=train_eval_env,
    title="Actions performed in training period",
    save_path=os.path.join(SAVE_DIR, "train_actions.png"),
)

summarize_curve("Train Buy and Hold", train_bh_curve)
summarize_curve("Train EIIE-PG", train_drl_curve)


# =========================
# 9. 验证集表现图：EIIE vs Buy-and-Hold
# =========================
val_bh_curve, val_bh_env = run_equal_weight_baseline(val_df)
val_drl_curve, val_eval_env = run_drl_validation(val_df, trained_model)

plot_performance_curve(
    baseline_curve=val_bh_curve,
    drl_curve=val_drl_curve,
    title="Performance in validation period",
    save_path=os.path.join(SAVE_DIR, "val_performance.png"),
)

plot_actions_memory(
    env=val_eval_env,
    title="Actions performed in validation period",
    save_path=os.path.join(SAVE_DIR, "val_actions.png"),
)

summarize_curve("Val Buy and Hold", val_bh_curve)
summarize_curve("Val EIIE-PG", val_drl_curve)


# =========================
# 10. 交易集表现图：EIIE vs Buy-and-Hold
# =========================
trade_bh_curve, trade_bh_env = run_equal_weight_baseline(trade_df)
trade_drl_curve, trade_eval_env = run_drl_validation(trade_df, trained_model)

plot_performance_curve(
    baseline_curve=trade_bh_curve,
    drl_curve=trade_drl_curve,
    title="Performance in trading period",
    save_path=os.path.join(SAVE_DIR, "trade_performance.png"),
)

plot_actions_memory(
    env=trade_eval_env,
    title="Actions performed in trading period",
    save_path=os.path.join(SAVE_DIR, "trade_actions.png"),
)

summarize_curve("Trade Buy and Hold", trade_bh_curve)
summarize_curve("Trade EIIE-PG", trade_drl_curve)


# =========================
# 11. SwanLab 记录
# =========================
train_metrics = calc_metrics(train_drl_curve)
val_metrics = calc_metrics(val_drl_curve)
trade_metrics = calc_metrics(trade_drl_curve)

swanlab.log({
    "best_episode": best_episode,
    "best_val_portfolio_value": best_val_value,

    "train_bh_final_value": float(train_bh_curve[-1]),
    "train_drl_final_value": float(train_drl_curve[-1]),
    "val_bh_final_value": float(val_bh_curve[-1]),
    "val_drl_final_value": float(val_drl_curve[-1]),
    "trade_bh_final_value": float(trade_bh_curve[-1]),
    "trade_drl_final_value": float(trade_drl_curve[-1]),

    "train_drl_cum_return": train_metrics["cum_return"],
    "train_drl_sharpe": train_metrics["sharpe"],
    "train_drl_max_drawdown": train_metrics["max_drawdown"],

    "val_drl_cum_return": val_metrics["cum_return"],
    "val_drl_sharpe": val_metrics["sharpe"],
    "val_drl_max_drawdown": val_metrics["max_drawdown"],

    "trade_drl_cum_return": trade_metrics["cum_return"],
    "trade_drl_sharpe": trade_metrics["sharpe"],
    "trade_drl_max_drawdown": trade_metrics["max_drawdown"],
})

swanlab.finish()