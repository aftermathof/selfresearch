import warnings
warnings.filterwarnings("ignore")

import time
import numpy as np
import pandas as pd
import yfinance as yf

# =========================
# 1. 参数
# =========================
TRAIN_START = "2014-01-01"
TRAIN_END   = "2024-12-31"

# 这里放“上证50候选池”
# 注意：yfinance 里上交所通常用 .SS
SSE50_TICKERS = [
    "600000.SS",
    "600036.SS",
    "600104.SS",
    "600030.SS",
    "601628.SS",
    "601166.SS",
    "601318.SS",
    "601328.SS",
    "601088.SS",
    "601857.SS",
    "601601.SS",
    "601668.SS",
    "601288.SS",
    "601818.SS",
    "601989.SS",
    "601398.SS",
    "600048.SS",
    "600028.SS",
    "600050.SS",
    "600519.SS",
    "600016.SS",
    "600887.SS",
    "601688.SS",
    "601186.SS",
    "601988.SS",
    "601211.SS",
    "601336.SS",
    "600309.SS",
    "603993.SS",
    "600690.SS",
    "600276.SS",
    "600703.SS",
    "600585.SS",
    "603259.SS",
    "601888.SS",
    "601138.SS",
    "600196.SS",
    "601766.SS",
    "600340.SS",
    "601390.SS",
    "601939.SS",
    "601111.SS",
    "600029.SS",
    "600019.SS",
    "601229.SS",
    "601800.SS",
    "600547.SS",
    "601006.SS",
    "601360.SS",
    "600606.SS",
    "601319.SS",
    "600837.SS",
    "600031.SS",
    "601066.SS",
    "600009.SS",
    "601236.SS",
    "601012.SS",
    "600745.SS",
    "600588.SS",
    "601658.SS",
    "601816.SS",
    "603160.SS",
]


# 然后再从中筛 top16


# =========================
# 2. 拉历史行情
# =========================
def download_history(tickers, start, end):
    """
    批量下载历史数据。
    返回长表：date, tic, close, volume
    """
    data = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
        group_by="ticker",
        threads=True
    )

    frames = []

    # 多股票时 columns 通常是 MultiIndex
    if isinstance(data.columns, pd.MultiIndex):
        for tic in tickers:
            if tic not in data.columns.get_level_values(0):
                continue
            tmp = data[tic].copy()
            if tmp.empty:
                continue
            tmp = tmp.reset_index()
            tmp["tic"] = tic

            # 兼容不同版本字段命名
            close_col = "Close" if "Close" in tmp.columns else None
            volume_col = "Volume" if "Volume" in tmp.columns else None

            if close_col is None or volume_col is None:
                continue

            tmp = tmp.rename(columns={
                "Date": "date",
                close_col: "close",
                volume_col: "volume"
            })
            tmp = tmp[["date", "tic", "close", "volume"]]
            frames.append(tmp)
    else:
        # 单股票时
        tmp = data.reset_index().copy()
        tic = tickers[0]
        tmp["tic"] = tic
        tmp = tmp.rename(columns={
            "Date": "date",
            "Close": "close",
            "Volume": "volume"
        })
        tmp = tmp[["date", "tic", "close", "volume"]]
        frames.append(tmp)

    if not frames:
        raise ValueError("历史行情下载失败，未获取到任何有效数据。")

    hist_df = pd.concat(frames, ignore_index=True)
    hist_df["date"] = pd.to_datetime(hist_df["date"])
    hist_df["close"] = pd.to_numeric(hist_df["close"], errors="coerce")
    hist_df["volume"] = pd.to_numeric(hist_df["volume"], errors="coerce")
    hist_df = hist_df.dropna(subset=["date", "tic", "close", "volume"]).copy()
    return hist_df


# =========================
# 3. 获取股本并构造换手率
# =========================
def get_shares_series(tic, start, end):
    """
    优先用 get_shares_full() 获取历史股本；
    如果失败，则退化到 info['sharesOutstanding'] 作为常数近似。
    返回 Series: index=date, value=shares_outstanding
    """
    tk = yf.Ticker(tic)

    # 3.1 优先：历史股本
    try:
        shares = tk.get_shares_full(start=start, end=end)
        if shares is not None and len(shares) > 0:
            shares = pd.Series(shares).copy()
            shares.index = pd.to_datetime(shares.index).tz_localize(None, nonexistent="shift_forward", ambiguous="NaT")
            shares = shares.sort_index()
            shares = shares[~shares.index.isna()]
            shares.name = "shares_outstanding"
            return shares
    except Exception:
        pass

    # 3.2 退化：当前/静态股本
    try:
        info = tk.get_info()
        shares_outstanding = info.get("sharesOutstanding", None)
        if shares_outstanding is not None and shares_outstanding > 0:
            idx = pd.date_range(start=start, end=end, freq="D")
            shares = pd.Series(float(shares_outstanding), index=idx, name="shares_outstanding")
            return shares
    except Exception:
        pass

    return None


def build_liquidity_table(tickers, start, end):
    """
    对每只股票计算：
    - avg_amount
    - avg_turnover_rate
    并返回汇总表
    """
    hist_df = download_history(tickers, start, end)
    results = []

    for i, tic in enumerate(tickers, 1):
        print(f"[{i}/{len(tickers)}] processing {tic} ...")
        sub = hist_df[hist_df["tic"] == tic].copy()
        if sub.empty:
            continue

        # 成交额近似 = close * volume
        sub["amount"] = sub["close"] * sub["volume"]

        # 获取股本
        shares_series = get_shares_series(tic, start, end)
        if shares_series is None:
            print(f"  skip {tic}: shares_outstanding unavailable")
            continue

        # 对齐到交易日
        shares_df = shares_series.reset_index()
        shares_df.columns = ["date", "shares_outstanding"]
        shares_df["date"] = pd.to_datetime(shares_df["date"]).dt.normalize()

        sub["date"] = pd.to_datetime(sub["date"]).dt.normalize()

        # 用 merge_asof 做向后填充，适合股本这种低频更新数据
        sub = sub.sort_values("date")
        shares_df = shares_df.sort_values("date")

        merged = pd.merge_asof(
            sub,
            shares_df,
            on="date",
            direction="backward"
        )

        # 如果最前面仍为空，再向前填充一次
        merged["shares_outstanding"] = merged["shares_outstanding"].ffill().bfill()

        merged = merged.dropna(subset=["amount", "volume", "shares_outstanding"]).copy()
        merged = merged[merged["shares_outstanding"] > 0].copy()

        if merged.empty:
            print(f"  skip {tic}: merged data empty")
            continue

        merged["turnover_rate"] = merged["volume"] / merged["shares_outstanding"]

        avg_amount = merged["amount"].mean()
        avg_turnover = merged["turnover_rate"].mean()

        results.append({
            "tic": tic,
            "avg_amount": float(avg_amount),
            "avg_turnover_rate": float(avg_turnover),
            "n_obs": int(len(merged))
        })

        # 避免请求过快
        time.sleep(0.2)

    if not results:
        raise ValueError("没有成功构造任何股票的流动性指标。")

    liq = pd.DataFrame(results)

    # 排名：值越大，rank 越靠前（即 rank=1 最好）
    liq["rank_amount"] = liq["avg_amount"].rank(ascending=False, method="min")
    liq["rank_turnover"] = liq["avg_turnover_rate"].rank(ascending=False, method="min")

    # 分数越小越好
    liq["LiquidityScore"] = 0.7 * liq["rank_amount"] + 0.3 * liq["rank_turnover"]

    liq = liq.sort_values(
        by=["LiquidityScore", "avg_amount", "avg_turnover_rate"],
        ascending=[True, False, False]
    ).reset_index(drop=True)

    return liq


# =========================
# 4. 运行并输出后15
# =========================
liq_df = build_liquidity_table(SSE50_TICKERS, TRAIN_START, TRAIN_END)

top15 = liq_df.tail(15).copy()

print("\n===== Bottom 15 Liquidity Stocks =====")
print(top15[[
    "tic", "avg_amount", "avg_turnover_rate",
    "rank_amount", "rank_turnover", "LiquidityScore", "n_obs"
]])

print("\n后15只股票代码：")
print(top15["tic"].tolist())

# 保存
top15.to_csv("top15_liquidity_from_yfinance.csv", index=False, encoding="utf-8-sig")