"""
Optuna 超参数搜索 —— EIIE + DPG (PortfolioOptimizationEnv)
目标：最大化验证集 Sharpe Ratio
搜索空间：lr, batch_size, time_window, k_size, conv_mid_features, conv_final_features
"""

import warnings
warnings.filterwarnings("ignore")
import logging
logging.getLogger("matplotlib").setLevel(logging.ERROR)

import numpy as np
import torch
import optuna
from sklearn.preprocessing import MaxAbsScaler

# ── 根据你的项目路径调整这些 import ──────────────────────────
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.env_portfolio_optimization.env_portfolio_optimization import PortfolioOptimizationEnv
from finrl.agents.portfolio_optimization.models import DRLAgent
from finrl.agents.portfolio_optimization.architectures_raw import EIIE
# ──────────────────────────────────────────────────────────────

# ── 你的数据（直接复用你 notebook 里已有的 df） ────────────────
# 如果在 notebook 里调用，把这段替换成你已有的变量
from finrl.meta.preprocessor.preprocessors import GroupByScaler

import os
os.environ["HTTP_PROXY"] = "http://127.0.0.1:7897"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7897"

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

def get_data():
    portfolio_raw_df = YahooDownloader(
        start_date="2014-01-01", end_date="2025-12-31",
        ticker_list=TOP_BRL
    ).fetch_data()
    portfolio_norm_df = GroupByScaler(
        by="tic", scaler=MaxAbsScaler
    ).fit_transform(portfolio_raw_df)
    df = portfolio_norm_df[["date", "tic", "close", "high", "low"]]
    df_train = df[(df["date"] >= "2014-01-01") & (df["date"] < "2020-01-01")]
    df_val   = df[(df["date"] >= "2020-01-01") & (df["date"] < "2022-01-01")]
    return df_train, df_val


def calculate_sharpe(value_list, risk_free_rate=0, periods_per_year=252):
    values = np.array(value_list)
    daily_returns = np.diff(values) / values[:-1]
    if daily_returns.std() == 0:
        return 0.0
    excess = daily_returns.mean() - risk_free_rate / periods_per_year
    return float((excess / daily_returns.std()) * np.sqrt(periods_per_year))


def objective(trial, df_train, df_val):
    """Optuna objective：每次 trial 返回验证集 Sharpe"""

    # ── 1. 超参采样 ───────────────────────────────────────────
    lr            = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    batch_size    = trial.suggest_categorical("batch_size", [32, 64, 100, 128])
    time_window   = trial.suggest_categorical("time_window", [30, 50, 70])
    k_size        = trial.suggest_categorical("k_size", [2, 3, 5])
    conv_mid      = trial.suggest_categorical("conv_mid_features", [2, 4, 8])
    conv_final    = trial.suggest_categorical("conv_final_features", [10, 20, 32])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── 2. 构建环境 ───────────────────────────────────────────
    env_train = PortfolioOptimizationEnv(
        df_train,
        initial_amount=100000,
        comission_fee_pct=0.002,
        time_window=time_window,
        features=["close", "high", "low"],
        normalize_df=None,
    )
    env_val = PortfolioOptimizationEnv(
        df_val,
        initial_amount=100000,
        comission_fee_pct=0.002,
        time_window=time_window,
        features=["close", "high", "low"],
        normalize_df=None,
    )

    # ── 3. 构建模型 ───────────────────────────────────────────
    model_kwargs = {
        "lr": lr,
        "batch_size": batch_size,
        "policy": EIIE,
    }
    policy_kwargs = {
        "k_size": k_size,
        "conv_mid_features": conv_mid,
        "conv_final_features": conv_final,
        "time_window": time_window,
    }
    model = DRLAgent(env_train).get_model("pg", device, model_kwargs, policy_kwargs)

    # ── 4. 带 early-stop 的训练 ───────────────────────────────
    MAX_EPISODES   = 15
    PATIENCE       = 3
    best_sharpe    = -np.inf
    best_weights   = None
    patience_count = 0

    for ep in range(1, MAX_EPISODES + 1):
        DRLAgent.train_model(model, episodes=1)
        DRLAgent.DRL_validation(model, env_val, policy=model.train_policy)
        val_sharpe = calculate_sharpe(env_val._asset_memory["final"])

        # Optuna pruning（中间结果不好就提前剪枝）
        trial.report(val_sharpe, ep)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        if val_sharpe > best_sharpe:
            best_sharpe    = val_sharpe
            best_weights   = {k: v.clone() for k, v in model.train_policy.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                break

    return best_sharpe


def run_study(n_trials=50, n_jobs=1):
    df_train, df_val = get_data()

    # MedianPruner：前5个 trial 作为 warmup，之后开启剪枝
    pruner  = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3)
    sampler = optuna.samplers.TPESampler(seed=42)

    study = optuna.create_study(
        direction="maximize",
        study_name="raw_eiie_sse50",
        pruner=pruner,
        sampler=sampler,
        # 如果想持久化到磁盘，取消注释下一行：
        # storage="sqlite:///eiie_optuna.db",
        # load_if_exists=True,
    )

    study.optimize(
        lambda trial: objective(trial, df_train, df_val),
        n_trials=n_trials,
        n_jobs=n_jobs,           # 如果有多卡/多核可以设 >1
        show_progress_bar=True,
    )

    # ── 结果汇总 ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("最优 Trial:")
    best = study.best_trial
    print(f"  验证集 Sharpe: {best.value:.4f}")
    print("  超参：")
    for k, v in best.params.items():
        print(f"    {k}: {v}")

    # 保存所有 trial 结果
    import pandas as pd
    df_results = study.trials_dataframe()
    df_results.to_csv("optuna_results.csv", index=False)
    print("\n所有 trial 结果已保存到 optuna_results.csv")

    # 可视化（需要 plotly）
    import plotly
    fig1 = optuna.visualization.plot_optimization_history(study)
    fig1.write_html("optuna_history.html")

    fig2 = optuna.visualization.plot_param_importances(study)
    fig2.write_html("optuna_importance.html")

    print("可视化已保存：optuna_history.html, optuna_importance.html")


    return study


if __name__ == "__main__":
    study = run_study(n_trials=50)