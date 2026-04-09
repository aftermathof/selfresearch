def clean_panel(df: pd.DataFrame, name: str) -> pd.DataFrame:
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