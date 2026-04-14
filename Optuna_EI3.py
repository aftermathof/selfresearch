import os
import json
import random
import warnings
import logging
import gc
import optuna
import os
warnings.filterwarnings("ignore")
logging.getLogger("matplotlib").setLevel(logging.ERROR)

import numpy as np
import pandas as pd
import torch
import optuna
import matplotlib.pyplot as plt
from sklearn.preprocessing import MaxAbsScaler

# ── 根据你的项目路径调整这些 import ──────────────────────────
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import GroupByScaler
from finrl.meta.env_portfolio_optimization.env_portfolio_optimization import (
    PortfolioOptimizationEnv,
)
from finrl.agents.portfolio_optimization.models import DRLAgent
from finrl.agents.portfolio_optimization.architectures_raw import EI3
# ──────────────────────────────────────────────────────────────

# =========================
# 0. 可调的全局配置
# =========================
SEED = 42
INITIAL_AMOUNT = 100000
COMISSION_FEE_PCT = 0.002
FEATURES = ["close", "high", "low"]
NORMALIZE_DF = None
PLOT_GRAPHS = False

TRAIN_START = "2014-01-01"
TRAIN_END   = "2020-01-01"
VAL_START   = "2020-01-01"
VAL_END     = "2022-01-01"

MAX_EPISODES = 15
PATIENCE = 3

STUDY_NAME = "raw_ei3_sse50"
RESULT_CSV = "optuna_results_ei3.csv"
BEST_PARAM_JSON = "best_params_ei3.json"
BEST_MODEL_PATH = "best_ei3_optuna.pt"

N_TRIALS = 50
N_JOBS = 1

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

# 如果你需要代理，再打开下面两行
os.environ["HTTP_PROXY"] = "http://127.0.0.1:7897"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7897"


# =========================
# 1. 随机种子
# =========================
def set_seed(seed: int = 42, deterministic: bool = True):
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass


# =========================
# 2. 数据获取
# =========================
def get_data():
    portfolio_raw_df = YahooDownloader(
        start_date="2014-01-01",
        end_date="2025-12-31",
        ticker_list=TOP_BRL,
    ).fetch_data()

    portfolio_norm_df = GroupByScaler(
        by="tic", scaler=MaxAbsScaler
    ).fit_transform(portfolio_raw_df)

    df = portfolio_norm_df[["date", "tic", "close", "high", "low"]].copy()
    df["date"] = pd.to_datetime(df["date"])

    df_train = df[(df["date"] >= TRAIN_START) & (df["date"] < TRAIN_END)].copy()
    df_val   = df[(df["date"] >= VAL_START)   & (df["date"] < VAL_END)].copy()

    return df_train, df_val


# =========================
# 3. 环境构造
# =========================
def make_env(df, time_window: int):
    env = PortfolioOptimizationEnv(
        df,
        initial_amount=INITIAL_AMOUNT,
        comission_fee_pct=COMISSION_FEE_PCT,
        time_window=time_window,
        features=FEATURES,
        normalize_df=NORMALIZE_DF,
        plot_graphs=PLOT_GRAPHS,
    )
    return env


def cleanup_memory():
    plt.close("all")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# =========================
# 4. Sharpe 计算
# =========================
def calculate_sharpe(value_list, risk_free_rate=0.0, periods_per_year=252):
    values = np.asarray(value_list, dtype=float)

    if len(values) < 2:
        return 0.0

    daily_returns = np.diff(values) / values[:-1]

    if len(daily_returns) < 2:
        return 0.0

    std = daily_returns.std(ddof=1)
    if std == 0 or np.isnan(std):
        return 0.0

    rf_per_period = risk_free_rate / periods_per_year
    excess = daily_returns.mean() - rf_per_period
    sharpe = (excess / std) * np.sqrt(periods_per_year)

    if np.isnan(sharpe) or np.isinf(sharpe):
        return 0.0

    return float(sharpe)


# =========================
# 5. Optuna objective
# =========================
def objective(trial, df_train, df_val):
    """
    每个 trial 返回：
    验证集最优 Sharpe Ratio
    """
    trial_seed = SEED + trial.number
    set_seed(trial_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- 5.1 超参数搜索空间 ----
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    k_short = trial.suggest_categorical("k_short", [3,5,7])
    k_medium = trial.suggest_categorical("k_medium", [15, 21, 30])
    conv_mid_features = trial.suggest_categorical("conv_mid_features", [2, 4, 8])
    conv_final_features = trial.suggest_categorical("conv_final_features", [16, 20, 32])

    # ---- 5.2 构建环境 ----
    env_train = make_env(df_train, time_window=50)

    # ---- 5.3 构建模型 ----
    model_kwargs = {
        "lr": lr,
        "batch_size": batch_size,
        "policy": EI3,
        "action_noise": 0,
    }

    policy_kwargs = {
        "initial_features": len(FEATURES),
        "k_short": k_short,
        "k_medium": k_medium,
        "conv_mid_features": conv_mid_features,
        "conv_final_features": conv_final_features,
        "time_window": 50,
    }

    model = DRLAgent(env_train).get_model(
        model_name="pg",
        device=device,
        model_kwargs=model_kwargs,
        policy_kwargs=policy_kwargs,
    )

    # ---- 5.4 early stopping on validation ----
    best_sharpe = -np.inf
    best_episode = 0
    patience_count = 0

    val_sharpe_history = []
    val_return_history = []

    for ep in range(1, MAX_EPISODES + 1):
        # 训练 1 个 episode
        DRLAgent.train_model(model, episodes=1)

        # 每轮都重建验证环境，避免状态污染
        env_val = make_env(df_val, time_window=50)
        DRLAgent.DRL_validation(model, env_val, policy=model.train_policy)

        val_values = np.asarray(env_val._asset_memory["final"], dtype=float)
        val_sharpe = calculate_sharpe(val_values)
        val_return = float(val_values[-1] / val_values[0] - 1.0)

        val_sharpe_history.append(val_sharpe)
        val_return_history.append(val_return)

        # 向 Optuna 报告中间结果
        trial.report(val_sharpe, ep)

        # 剪枝
        if trial.should_prune():
            del env_val
            cleanup_memory()
            raise optuna.exceptions.TrialPruned()

        # 更新最优
        if val_sharpe > best_sharpe:
            best_sharpe = val_sharpe
            best_episode = ep
            patience_count = 0
        else:
            patience_count += 1

        # early stop
        if patience_count >= PATIENCE:
            del env_val
            cleanup_memory()
            break

        del env_val
        cleanup_memory()

    # 把额外信息挂到 trial 上
    trial.set_user_attr("best_episode", int(best_episode))
    trial.set_user_attr("val_sharpe_history", val_sharpe_history)
    trial.set_user_attr("val_return_history", val_return_history)

    del env_train
    del model
    cleanup_memory()
    return best_sharpe


# =========================
# 6. 用最优参数重新训练并保存模型
# =========================
def retrain_best_model(best_params, df_train, df_val):
    """
    用最优参数重新训练一个最终模型，并保存验证集最佳 checkpoint
    """
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env_train = make_env(df_train, time_window=best_params["time_window"])

    model_kwargs = {
        "lr": best_params["lr"],
        "batch_size": best_params["batch_size"],
        "policy": EI3,
        "action_noise": best_params["action_noise"],
    }

    policy_kwargs = {
        "initial_features": len(FEATURES),
        "k_short": best_params["k_short"],
        "k_medium": best_params["k_medium"],
        "conv_mid_features": best_params["conv_mid_features"],
        "conv_final_features": best_params["conv_final_features"],
        "time_window": best_params["time_window"],
    }

    model = DRLAgent(env_train).get_model(
        model_name="pg",
        device=device,
        model_kwargs=model_kwargs,
        policy_kwargs=policy_kwargs,
    )

    best_sharpe = -np.inf
    best_episode = 0
    patience_count = 0

    history = {
        "val_sharpe": [],
        "val_return": [],
    }

    print("\n开始用最优参数重新训练最终模型...")

    for ep in range(1, MAX_EPISODES + 1):
        DRLAgent.train_model(model, episodes=1)

        env_val = make_env(df_val, time_window=best_params["time_window"])
        DRLAgent.DRL_validation(model, env_val, policy=model.train_policy)

        val_values = np.asarray(env_val._asset_memory["final"], dtype=float)
        val_sharpe = calculate_sharpe(val_values)
        val_return = float(val_values[-1] / val_values[0] - 1.0)

        history["val_sharpe"].append(val_sharpe)
        history["val_return"].append(val_return)

        print(
            f"Episode {ep:02d} | "
            f"Val Sharpe = {val_sharpe:.4f} | "
            f"Val Return = {val_return:.4f}"
        )

        if val_sharpe > best_sharpe:
            best_sharpe = val_sharpe
            best_episode = ep
            patience_count = 0
            torch.save(model.train_policy.state_dict(), BEST_MODEL_PATH)
            print(f"  ✓ 保存当前最优模型到 {BEST_MODEL_PATH}")
        else:
            patience_count += 1

        if patience_count >= PATIENCE:
            print(f"Early stopping at episode {ep}")
            del env_val
            cleanup_memory()
            break

        del env_val
        cleanup_memory()

    print("\n最终模型训练结束")
    print(f"Best episode = {best_episode}")
    print(f"Best validation Sharpe = {best_sharpe:.4f}")

    del env_train
    del model
    cleanup_memory()
    return history


# =========================
# 7. 运行 study
# =========================
def run_study(n_trials=50, n_jobs=1):
    set_seed(SEED)
    df_train, df_val = get_data()

    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=3
    )
    sampler = optuna.samplers.TPESampler(seed=SEED)

    study = optuna.create_study(
        direction="maximize",
        study_name=STUDY_NAME,
        pruner=pruner,
        sampler=sampler,
    )

    study.optimize(
        lambda trial: objective(trial, df_train, df_val),
        n_trials=n_trials,
        n_jobs=n_jobs,
        show_progress_bar=True,
    )

    # ---- 结果汇总 ----
    print("\n" + "=" * 60)
    print("最优 Trial:")
    best = study.best_trial
    print(f"  验证集 Sharpe: {best.value:.4f}")
    print("  超参数：")
    for k, v in best.params.items():
        print(f"    {k}: {v}")

    # 保存所有 trial 结果
    df_results = study.trials_dataframe()
    df_results.to_csv(RESULT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n所有 trial 结果已保存到 {RESULT_CSV}")

    # 保存最佳参数
    with open(BEST_PARAM_JSON, "w", encoding="utf-8") as f:
        json.dump(best.params, f, ensure_ascii=False, indent=2)
    print(f"最优参数已保存到 {BEST_PARAM_JSON}")

    # 可视化（需要 plotly）
    try:
        fig1 = optuna.visualization.plot_optimization_history(study)
        fig1.write_html("optuna_history_ei3.html")

        fig2 = optuna.visualization.plot_param_importances(study)
        fig2.write_html("optuna_importance_ei3.html")

        print("可视化已保存：optuna_history_ei3.html, optuna_importance_ei3.html")
    except Exception as e:
        print(f"可视化保存失败：{e}")

    # ---- 用最优参数重训最终模型 ----
    retrain_history = retrain_best_model(best.params, df_train, df_val)

    return study, retrain_history


# =========================
# 8. 主程序
# =========================
if __name__ == "__main__":
    study, retrain_history = run_study(
        n_trials=N_TRIALS,
        n_jobs=N_JOBS,
    )
