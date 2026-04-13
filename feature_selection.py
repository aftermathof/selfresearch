"""
Feature Selection: Pearson Correlation + PCA
===========================================
流程：
  1. 从训练集与交易集读取原始特征
  2. 计算皮尔逊相关系数矩阵并绘制热图
  3. 剔除与其他特征高度相关的冗余特征
  4. 对保留特征做标准化后拟合 PCA
  5. 选取累计解释方差 >= 90% 的主成分数量
  6. 输出每个主成分的载荷与 Top-3 贡献特征
  7. 保存降维后的 train_pca.csv / trade_pca.csv 供 FinRL 使用
"""

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


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
    "lines.linewidth": 1.8,
})

sns.set_theme(style="white")


# --------------------------------------------------
# 0. Configuration
# --------------------------------------------------
TRAIN_FILE = "train_data1.csv"
TRADE_FILE = "trade_data1.csv"
OUTPUT_DIR = "./"

CORR_THRESHOLD = 0.90
VARIANCE_TARGET = 0.90

FEATURE_COLS = [
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
    "volume",
    "r1",
    "r5",
    "r10",
    "r20",
    "intraday",
    "overnight",
    "range",
]

META_COLS = ["date", "tic"]

BLUE = "#4C72B0"
RED = "#C44E52"
GREEN = "#55A868"
GRID = "#B0B0B0"
SPINE = "#444444"


def save_png(fig: plt.Figure, filename: str) -> None:
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, bbox_inches="tight")
    print(f"图像已保存: {path}")


def style_axis(ax: plt.Axes, grid_axis: str = "both") -> None:
    ax.grid(True, axis=grid_axis, linestyle="--", color=GRID)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(SPINE)
    ax.spines["bottom"].set_color(SPINE)
    ax.tick_params(axis="both", which="major", length=3.5, width=0.8, color=SPINE)


def style_heatmap_ticks(ax: plt.Axes) -> None:
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), rotation=0)


def remove_high_corr_features(corr_mat: pd.DataFrame, threshold: float) -> list[str]:
    """
    贪心剔除：遍历相关矩阵上三角，
    遇到 |r| > threshold 的特征对则删除后出现的那个。
    """
    cols = list(corr_mat.columns)
    to_drop = set()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            if abs(corr_mat.iloc[i, j]) > threshold:
                to_drop.add(cols[j])
    return [c for c in cols if c not in to_drop]


print("=" * 60)
print("Step 1 - 读取数据")
print("=" * 60)

train_df = pd.read_csv(TRAIN_FILE)
trade_df = pd.read_csv(TRADE_FILE)

actual_features = [c for c in FEATURE_COLS if c in train_df.columns]
missing = [c for c in FEATURE_COLS if c not in train_df.columns]
if missing:
    print(f"[警告] 以下特征在数据中不存在，已跳过: {missing}")

FEATURE_COLS = actual_features
print(f"实际使用特征数: {len(FEATURE_COLS)}")
print(f"特征列表: {FEATURE_COLS}\n")

X_train = train_df[FEATURE_COLS].dropna()


print("=" * 60)
print("Step 2 - 皮尔逊相关系数分析")
print("=" * 60)

corr_matrix = X_train.corr(method="pearson")

fig, ax = plt.subplots(figsize=(11.5, 9.5))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

sns.heatmap(
    corr_matrix,
    mask=mask,
    cmap=sns.diverging_palette(240, 15, as_cmap=True),
    vmin=-1.0,
    vmax=1.0,
    center=0,
    square=True,
    linewidths=0.45,
    linecolor="white",
    annot=True,
    fmt=".2f",
    annot_kws={"size": 7.2},
    cbar_kws={"shrink": 0.78, "pad": 0.02, "label": "Pearson Correlation"},
    ax=ax,
)
ax.set_title("Pearson Correlation Heatmap of Raw Features", pad=12)
style_heatmap_ticks(ax)
fig.tight_layout(pad=0.6)
save_png(fig, "feature_correlation_heatmap.png")
plt.close(fig)


print(f"\n高相关阈值 |r| > {CORR_THRESHOLD}，开始剔除冗余特征")

kept_features = remove_high_corr_features(corr_matrix, CORR_THRESHOLD)
dropped_features = [c for c in FEATURE_COLS if c not in kept_features]

print(f"原始特征数: {len(FEATURE_COLS)}")
print(f"剔除高相关特征: {dropped_features}")
print(f"剔除后保留特征数: {len(kept_features)}")
print(f"保留特征: {kept_features}\n")

X_filtered = X_train[kept_features]


print("=" * 60)
print("Step 3 - PCA 降维")
print("=" * 60)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_filtered)

pca_full = PCA(n_components=len(kept_features), random_state=42)
pca_full.fit(X_scaled)

explained_ratio = pca_full.explained_variance_ratio_
cumulative_ratio = np.cumsum(explained_ratio)
n_components = int(np.argmax(cumulative_ratio >= VARIANCE_TARGET) + 1)

print("各主成分解释方差比例:")
for i, (ratio, cumulative) in enumerate(zip(explained_ratio, cumulative_ratio), start=1):
    marker = " <-- 选取截止" if i == n_components else ""
    print(f"  PC{i:02d}: {ratio:.4f}  累计: {cumulative:.4f}{marker}")

print(f"\n达到 {VARIANCE_TARGET * 100:.0f}% 解释方差需要 {n_components} 个主成分")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 4.8))
components = np.arange(1, len(explained_ratio) + 1)

ax1.bar(
    components,
    explained_ratio,
    color=BLUE,
    alpha=0.88,
    edgecolor="white",
    linewidth=0.6
)
ax1.axvline(
    n_components + 0.5,
    color=RED,
    linestyle="--",
    linewidth=1.3,
    label=f"Cutoff: PC{n_components}"
)
ax1.set_xlabel("Principal Component")
ax1.set_ylabel("Explained Variance Ratio")
ax1.set_title("Scree Plot", pad=8)
ax1.set_xticks(components)
ax1.legend(
    loc="upper right",
    frameon=True,
    framealpha=0.95,
    facecolor="white",
    edgecolor="#D0D0D0",
)
style_axis(ax1, grid_axis="y")

ax2.plot(
    components,
    cumulative_ratio,
    marker="o",
    markersize=4.8,
    color=BLUE,
    linewidth=1.8
)
ax2.axhline(
    VARIANCE_TARGET,
    color=RED,
    linestyle="--",
    linewidth=1.3,
    label=f"Target = {VARIANCE_TARGET * 100:.0f}%"
)
ax2.axvline(
    n_components,
    color=RED,
    linestyle="--",
    linewidth=1.3,
    label=f"n = {n_components}"
)
ax2.fill_between(
    components[:n_components],
    cumulative_ratio[:n_components],
    alpha=0.12,
    color=GREEN
)
ax2.set_xlabel("Number of Principal Components")
ax2.set_ylabel("Cumulative Explained Variance")
ax2.set_title("Cumulative Variance Explained", pad=8)
ax2.set_ylim(0, 1.02)
ax2.set_xticks(components)
ax2.legend(
    loc="lower right",
    frameon=True,
    framealpha=0.95,
    facecolor="white",
    edgecolor="#D0D0D0",
)
style_axis(ax2, grid_axis="both")

fig.tight_layout(pad=0.8, w_pad=1.3)
save_png(fig, "pca_variance_plot.png")
plt.close(fig)


print("\n" + "=" * 60)
print("Step 4 - 主成分载荷分析（Top-3 贡献特征）")
print("=" * 60)

loadings = pd.DataFrame(
    pca_full.components_[:n_components].T,
    index=kept_features,
    columns=[f"PC{i + 1}" for i in range(n_components)]
)

print("\n各主成分载荷矩阵（绝对值越大 = 贡献越大）:")
print(loadings.round(4).to_string())

print("\n每个主成分 Top-3 贡献特征:")
for pc in loadings.columns:
    top3 = loadings[pc].abs().nlargest(3)
    summary = ", ".join(f"{feat}({value:.3f})" for feat, value in top3.items())
    print(f"  {pc}: {summary}")

fig_width = max(8.2, n_components * 0.88)
fig_height = max(6.0, len(kept_features) * 0.47)
fig, ax = plt.subplots(figsize=(fig_width, fig_height))

sns.heatmap(
    loadings,
    cmap=sns.diverging_palette(240, 15, as_cmap=True),
    center=0,
    vmin=-1.0,
    vmax=1.0,
    linewidths=0.35,
    linecolor="white",
    annot=True,
    fmt=".2f",
    annot_kws={"size": 8},
    cbar_kws={"shrink": 0.85, "pad": 0.02, "label": "Loading"},
    ax=ax,
)
ax.set_title(f"PCA Loading Heatmap for the First {n_components} Components", pad=12)
style_heatmap_ticks(ax)
fig.tight_layout(pad=0.6)
save_png(fig, "pca_loadings_heatmap.png")
plt.close(fig)


print("\n" + "=" * 60)
print("Step 5 - 生成 PCA 降维后的数据集")
print("=" * 60)

pca_final = PCA(n_components=n_components, random_state=42)

X_train_filtered = train_df[kept_features].copy().ffill().bfill()
X_train_scaled = scaler.transform(X_train_filtered)
X_train_pca = pca_final.fit_transform(X_train_scaled)

pc_cols = [f"PC{i + 1}" for i in range(n_components)]
train_pca_df = pd.DataFrame(X_train_pca, columns=pc_cols, index=train_df.index)

for col in META_COLS:
    if col in train_df.columns:
        train_pca_df.insert(0, col, train_df[col].values)

train_pca_df.to_csv(f"{OUTPUT_DIR}train_pca.csv", index=False)
print(f"训练集降维完成: train_pca.csv  shape={train_pca_df.shape}")

X_trade_filtered = trade_df[kept_features].copy().ffill().bfill()
X_trade_scaled = scaler.transform(X_trade_filtered)
X_trade_pca = pca_final.transform(X_trade_scaled)

trade_pca_df = pd.DataFrame(X_trade_pca, columns=pc_cols, index=trade_df.index)
for col in META_COLS:
    if col in trade_df.columns:
        trade_pca_df.insert(0, col, trade_df[col].values)

trade_pca_df.to_csv(f"{OUTPUT_DIR}trade_pca.csv", index=False)
print(f"测试集降维完成: trade_pca.csv  shape={trade_pca_df.shape}")


print("\n" + "=" * 60)
print("汇总报告")
print("=" * 60)
print(f"原始特征数:              {len(FEATURE_COLS)}")
print(f"皮尔逊剔除高相关后:      {len(kept_features)}  (阈值 |r| > {CORR_THRESHOLD})")
print(f"PCA 保留主成分数:        {n_components}  (解释方差 >= {VARIANCE_TARGET * 100:.0f}%)")
print(f"实际累计解释方差:        {cumulative_ratio[n_components - 1] * 100:.2f}%")
print(f"\n降维后特征名: {pc_cols}")
print(f"\n在 FinRL 环境中将 features 改为: {pc_cols}")
print("=" * 60)
print("\n生成文件:")
print("  feature_correlation_heatmap.png  — 皮尔逊热图")
print("  pca_variance_plot.png            — 碎石图 + 累计方差曲线")
print("  pca_loadings_heatmap.png         — 主成分载荷热图")
print("  train_pca.csv                    — 降维后训练集")
print("  trade_pca.csv                    — 降维后测试集")
