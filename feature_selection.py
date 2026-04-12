"""
Feature Selection: Pearson Correlation + PCA
==========================================
流程：
  1. 从 processed_data.csv 读取 20 个原始特征
  2. 计算皮尔逊相关系数矩阵并绘制热图
  3. 剔除与其他特征高度相关(|r| > threshold)的冗余特征
  4. 对保留特征做标准化后拟合 PCA
  5. 选取累计解释方差 >= 90% 的主成分数量
  6. 输出每个主成分的载荷（贡献最大的原始特征）
  7. 保存降维后的 train_pca.csv / trade_pca.csv 供 FinRL 使用
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

plt.rcParams["font.sans-serif"] = ["SimHei"]      # 黑体
plt.rcParams["axes.unicode_minus"] = False        # 解决负号显示为方块
# ──────────────────────────────────────────
# 0. 配置
# ──────────────────────────────────────────
TRAIN_FILE  = "train_data1.csv"
TRADE_FILE  = "trade_data1.csv"
OUTPUT_DIR  = "./"

# 皮尔逊高相关阈值：两个特征 |r| 超过此值则视为冗余，保留其中一个
CORR_THRESHOLD = 0.90

# PCA 目标方差解释率
VARIANCE_TARGET = 0.90

# 原始 20 个特征（根据你的 processed_data.csv 实际列名调整）
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

META_COLS = ["date", "tic"]   # 非特征列，保留用于重构

# ──────────────────────────────────────────
# 1. 读取数据
# ──────────────────────────────────────────
print("=" * 60)
print("Step 1 — 读取数据")
print("=" * 60)

train_df = pd.read_csv(TRAIN_FILE)
trade_df = pd.read_csv(TRADE_FILE)

# 检查实际存在的特征列
actual_features = [c for c in FEATURE_COLS if c in train_df.columns]
missing = [c for c in FEATURE_COLS if c not in train_df.columns]
if missing:
    print(f"[警告] 以下特征在数据中不存在，已跳过: {missing}")

FEATURE_COLS = actual_features
print(f"实际使用特征数: {len(FEATURE_COLS)}")
print(f"特征列表: {FEATURE_COLS}\n")

X_train = train_df[FEATURE_COLS].dropna()

# ──────────────────────────────────────────
# 2. 皮尔逊相关系数矩阵 + 热图
# ──────────────────────────────────────────
print("=" * 60)
print("Step 2 — 皮尔逊相关系数分析")
print("=" * 60)

corr_matrix = X_train.corr(method="pearson")

# 绘制热图
fig, ax = plt.subplots(figsize=(12, 10))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)   # 只显示下三角

cmap = sns.diverging_palette(220, 20, as_cmap=True)
sns.heatmap(
    corr_matrix,
    mask=mask,
    cmap=cmap,
    vmax=1.0, vmin=-1.0, center=0,
    square=True,
    linewidths=0.4,
    annot=True,
    fmt=".2f",
    annot_kws={"size": 7},
    cbar_kws={"shrink": 0.6},
    ax=ax,
)
ax.set_title("特征皮尔逊相关系数热图", fontsize=14, pad=16)
plt.xticks(rotation=45, ha="right", fontsize=9)
plt.yticks(rotation=0, fontsize=9)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}feature_correlation_heatmap.png", dpi=150, bbox_inches="tight")
print("热图已保存: feature_correlation_heatmap.png")
plt.close()

# ──────────────────────────────────────────
# 3. 剔除高相关冗余特征
# ──────────────────────────────────────────
print(f"\n高相关阈值 |r| > {CORR_THRESHOLD}，剔除冗余特征")

def remove_high_corr_features(corr_mat: pd.DataFrame, threshold: float) -> list:
    """
    贪心剔除：遍历相关矩阵上三角，
    遇到 |r| > threshold 的特征对则删除后出现的那个（列索引更大的）。
    返回保留的特征名列表。
    """
    cols = list(corr_mat.columns)
    to_drop = set()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            if abs(corr_mat.iloc[i, j]) > threshold:
                to_drop.add(cols[j])
    return [c for c in cols if c not in to_drop]

kept_features = remove_high_corr_features(corr_matrix, CORR_THRESHOLD)
dropped_features = [c for c in FEATURE_COLS if c not in kept_features]

print(f"原始特征数: {len(FEATURE_COLS)}")
print(f"剔除高相关特征: {dropped_features}")
print(f"剔除后保留特征数: {len(kept_features)}")
print(f"保留特征: {kept_features}\n")

X_filtered = X_train[kept_features]

# ──────────────────────────────────────────
# 4. 标准化 + PCA 拟合
# ──────────────────────────────────────────
print("=" * 60)
print("Step 3 — PCA 降维")
print("=" * 60)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_filtered)

# 先拟合全量 PCA，获取方差解释率曲线
pca_full = PCA(n_components=len(kept_features), random_state=42)
pca_full.fit(X_scaled)

explained_ratio    = pca_full.explained_variance_ratio_
cumulative_ratio   = np.cumsum(explained_ratio)

# 找到累计方差 >= 90% 所需的主成分数
n_components = int(np.argmax(cumulative_ratio >= VARIANCE_TARGET) + 1)

print(f"各主成分解释方差比例:")
for i, (r, c) in enumerate(zip(explained_ratio, cumulative_ratio)):
    marker = " <-- 选取截止" if i + 1 == n_components else ""
    print(f"  PC{i+1:02d}: {r:.4f}  累计: {c:.4f}{marker}")

print(f"\n达到 {VARIANCE_TARGET*100:.0f}% 解释方差需要 {n_components} 个主成分")

# 绘制碎石图 (Scree Plot) + 累计方差曲线
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# 碎石图
ax1.bar(range(1, len(explained_ratio) + 1), explained_ratio,
        color="#378ADD", alpha=0.85, edgecolor="white", linewidth=0.4)
ax1.axvline(x=n_components + 0.5, color="#E24B4A", linestyle="--", linewidth=1.2,
            label=f"截止: PC{n_components}")
ax1.set_xlabel("主成分", fontsize=11)
ax1.set_ylabel("解释方差比例", fontsize=11)
ax1.set_title("碎石图 (Scree Plot)", fontsize=12)
ax1.legend(fontsize=10)
ax1.set_xticks(range(1, len(explained_ratio) + 1))

# 累计方差曲线
ax2.plot(range(1, len(cumulative_ratio) + 1), cumulative_ratio,
         marker="o", markersize=5, color="#378ADD", linewidth=1.5)
ax2.axhline(y=VARIANCE_TARGET, color="#E24B4A", linestyle="--", linewidth=1.2,
            label=f"{VARIANCE_TARGET*100:.0f}% 方差")
ax2.axvline(x=n_components, color="#E24B4A", linestyle="--", linewidth=1.2,
            label=f"n={n_components} 个主成分")
ax2.fill_between(range(1, n_components + 1),
                 cumulative_ratio[:n_components],
                 alpha=0.12, color="#378ADD")
ax2.set_xlabel("主成分数量", fontsize=11)
ax2.set_ylabel("累计解释方差比例", fontsize=11)
ax2.set_title("累计方差解释率", fontsize=12)
ax2.legend(fontsize=10)
ax2.set_ylim(0, 1.05)
ax2.set_xticks(range(1, len(cumulative_ratio) + 1))

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}pca_variance_plot.png", dpi=150, bbox_inches="tight")
print("\n方差解释率图已保存: pca_variance_plot.png")
plt.close()

# ──────────────────────────────────────────
# 5. 各主成分的载荷分析（哪些原始特征贡献最大）
# ──────────────────────────────────────────
print("\n" + "=" * 60)
print("Step 4 — 主成分载荷分析（Top-3 贡献特征）")
print("=" * 60)

loadings = pd.DataFrame(
    pca_full.components_[:n_components].T,
    index=kept_features,
    columns=[f"PC{i+1}" for i in range(n_components)]
)

print("\n各主成分载荷矩阵（绝对值越大 = 贡献越大）:")
print(loadings.round(4).to_string())

print("\n每个主成分 Top-3 贡献特征:")
for pc in loadings.columns:
    top3 = loadings[pc].abs().nlargest(3)
    print(f"  {pc}: {', '.join([f'{f}({v:.3f})' for f, v in top3.items()])}")

# 绘制载荷热图
fig, ax = plt.subplots(figsize=(max(8, n_components * 0.8), max(6, len(kept_features) * 0.45)))
sns.heatmap(
    loadings,
    cmap="RdBu_r",
    center=0,
    vmax=1.0, vmin=-1.0,
    annot=True,
    fmt=".2f",
    annot_kws={"size": 8},
    linewidths=0.3,
    ax=ax,
)
ax.set_title(f"PCA 载荷热图（前 {n_components} 个主成分）", fontsize=12, pad=12)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}pca_loadings_heatmap.png", dpi=150, bbox_inches="tight")
print("\n载荷热图已保存: pca_loadings_heatmap.png")
plt.close()

# ──────────────────────────────────────────
# 6. 拟合最终 PCA 并转换训练集 / 测试集
# ──────────────────────────────────────────
print("\n" + "=" * 60)
print("Step 5 — 生成 PCA 降维后的数据集")
print("=" * 60)

pca_final = PCA(n_components=n_components, random_state=42)

# --- 训练集 ---
X_train_filtered = train_df[kept_features].copy()
X_train_filtered = X_train_filtered.ffill().bfill()

X_train_scaled  = scaler.transform(X_train_filtered)
X_train_pca     = pca_final.fit_transform(X_train_scaled)

pc_cols = [f"PC{i+1}" for i in range(n_components)]
train_pca_df = pd.DataFrame(X_train_pca, columns=pc_cols, index=train_df.index)

# 保留 meta 列
for col in META_COLS:
    if col in train_df.columns:
        train_pca_df.insert(0, col, train_df[col].values)

train_pca_df.to_csv(f"{OUTPUT_DIR}train_pca.csv", index=False)
print(f"训练集降维完成: train_pca.csv  shape={train_pca_df.shape}")

# --- 测试/交易集 ---
X_trade_filtered = trade_df[kept_features].copy()
X_trade_filtered = X_trade_filtered.ffill().bfill()

X_trade_scaled  = scaler.transform(X_trade_filtered)   # 用 train 的 scaler
X_trade_pca     = pca_final.transform(X_trade_scaled)  # 用 train 的 PCA

trade_pca_df = pd.DataFrame(X_trade_pca, columns=pc_cols, index=trade_df.index)
for col in META_COLS:
    if col in trade_df.columns:
        trade_pca_df.insert(0, col, trade_df[col].values)

trade_pca_df.to_csv(f"{OUTPUT_DIR}trade_pca.csv", index=False)
print(f"测试集降维完成: trade_pca.csv  shape={trade_pca_df.shape}")

# ──────────────────────────────────────────
# 7. 汇总报告
# ──────────────────────────────────────────
print("\n" + "=" * 60)
print("汇总报告")
print("=" * 60)
print(f"原始特征数:              {len(FEATURE_COLS)}")
print(f"皮尔逊剔除高相关后:      {len(kept_features)}  (阈值 |r| > {CORR_THRESHOLD})")
print(f"PCA 保留主成分数:        {n_components}  (解释方差 >= {VARIANCE_TARGET*100:.0f}%)")
print(f"实际累计解释方差:        {cumulative_ratio[n_components-1]*100:.2f}%")
print(f"\n降维后特征名: {pc_cols}")
print(f"\n在 FinRL 环境中将 features 改为: {pc_cols}")
print("=" * 60)
print("\n生成文件:")
print("  feature_correlation_heatmap.png  — 皮尔逊热图")
print("  pca_variance_plot.png            — 碎石图 + 累计方差曲线")
print("  pca_loadings_heatmap.png         — 主成分载荷热图")
print("  train_pca.csv                    — 降维后训练集")
print("  trade_pca.csv                    — 降维后测试集")