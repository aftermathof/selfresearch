import pandas as pd

train_df = pd.read_csv("train_data.csv")
trade_df = pd.read_csv("trade_data.csv")

print("训练集：", train_df["date"].min(), "~", train_df["date"].max())
print("测试集：", trade_df["date"].min(), "~", trade_df["date"].max())

# 检查是否有重叠
overlap = set(train_df["date"]) & set(trade_df["date"])
print("重叠天数：", len(overlap))  # 必须为 0

trade_df = pd.read_csv("trade_data.csv")
pivot = trade_df.pivot(index="date", columns="tic", values="close")
equal_test = (pivot.iloc[-1] / pivot.iloc[0]).mean()
print(f"测试集等权基准：{equal_test:.4f}x")