from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import os
import io
import base64
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
from lightgbm import LGBMRegressor


def _encode_plot_to_base64() -> str:
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    return b64


def run_forecast_pipeline(csv_path: str, horizon_days: int = 0) -> dict:
    # Load
    df = pd.read_csv(csv_path, parse_dates=["date"]) 

    # Basic cleaning
    df = df[(df[["units_sold", "stock_available", "delivered_qty", "delivery_days"]] >= 0).all(axis=1)].copy()
    df = df.sort_values(["sku", "date"]).reset_index(drop=True)

    # Lag features
    for lag in [1, 2, 3, 7, 14, 28]:
        df[f"lag_{lag}"] = df.groupby("sku")["units_sold"].shift(lag)
    df["momentum_7_1"] = df["lag_1"] / (df["lag_7"] + 1e-3)
    df["momentum_14_7"] = df["lag_7"] / (df["lag_14"] + 1e-3)

    # Rolling trend
    def rolling_trend(x: pd.Series) -> float:
        return x.shift(1).rolling(7).apply(lambda z: np.polyfit(range(len(z)), z, 1)[0] if len(z.dropna()) == 7 else np.nan)
    df["trend_7"] = df.groupby("sku")["units_sold"].transform(rolling_trend)

    # Lags for exogenous vars
    df["price_lag_1"] = df.groupby("sku")["price_unit"].shift(1)
    df["promo_lag_1"] = df.groupby("sku")["promotion_flag"].shift(1)
    df["promo_effect"] = df["promotion_flag"] * df["lag_1"]
    df["delivery_lag_7"] = df.groupby("sku")["delivery_days"].shift(7)
    df["stock_lag_1"] = df.groupby("sku")["stock_available"].shift(1)

    # SKU stats
    sku_stats = df.groupby("sku")["units_sold"].agg(["mean", "min", "max", "std"]).reset_index()
    sku_stats.columns = ["sku", "sku_mean", "sku_min", "sku_max", "sku_std"]
    df = df.merge(sku_stats, on="sku", how="left")

    # Date features
    df["is_month_start"] = df["date"].dt.is_month_start.astype(int)
    df["is_month_end"]   = df["date"].dt.is_month_end.astype(int)
    df["dayofweek"]      = df["date"].dt.dayofweek
    df["month"]          = df["date"].dt.month
    df["year"]           = df["date"].dt.year
    df["is_weekend"]     = df["dayofweek"].isin([5, 6]).astype(int)

    # Rolling by category and exogenous rolling
    df["category_rolling_mean_7"] = df.groupby("category")["units_sold"].transform(lambda x: x.shift(1).rolling(7).mean())
    df["category_rolling_std_7"]  = df.groupby("category")["units_sold"].transform(lambda x: x.shift(1).rolling(7).std())
    df["price_rolling_mean_7"]    = df.groupby("sku")["price_unit"].transform(lambda x: x.shift(1).rolling(7).mean())
    df["promo_rolling_7"]         = df.groupby("sku")["promotion_flag"].transform(lambda x: x.shift(1).rolling(7).sum())

    # Stockouts and timers
    df["stockout_flag"] = ((df["stock_available"] == 0) & (df["delivered_qty"] > 0)).astype(int)
    df["rolling_stockouts_7"] = df.groupby("sku")["stockout_flag"].transform(lambda x: x.shift(1).rolling(7).sum())
    def time_since_event(series: pd.Series) -> pd.Series:
        return series[::-1].cumsum()[::-1] * series
    df["time_since_promo"] = df.groupby("sku")["promotion_flag"].transform(lambda x: time_since_event(x == 0))
    df["time_since_delivery"] = df.groupby("sku")["delivery_days"].transform(lambda x: time_since_event(x == 0))

    # Seasonality + interactions
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["dow_sin"]   = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"]   = np.cos(2 * np.pi * df["dayofweek"] / 7)
    df["promo_dow"] = df["promotion_flag"] * df["dayofweek"]
    df["price_x_stock"] = df["price_unit"] * df["stock_available"]

    # Encode categories
    enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    df[["sku_cat", "segment_cat", "category_cat"]] = enc.fit_transform(df[["sku", "segment", "category"]])

    # Drop NaNs from feature generation
    df = df.dropna()

    exclude_cols = ["date", "sku", "brand", "segment", "channel", "region", "pack_type", "delivered_qty", "units_sold", "category"]
    features = [c for c in df.columns if c not in exclude_cols and not np.issubdtype(df[c].dtype, np.datetime64)]
    categorical_features = [
        "sku_cat", "segment_cat", "category_cat",
        "dayofweek", "month", "is_weekend",
        "promotion_flag", "is_month_start", "is_month_end"
    ]
    target = "units_sold"

    # Train/test split by time
    cutoff_date = df["date"].max() - pd.Timedelta(days=42)
    train = df[df["date"] <= cutoff_date].copy()
    test  = df[df["date"] > cutoff_date].copy()

    category_models = {}
    category_preds = []
    category_metrics = {}

    # Train per category
    for cat in df["category"].unique():
        df_cat = df[df["category"] == cat].copy()
        cutoff = df_cat["date"].max() - pd.Timedelta(days=42)
        train_c = df_cat[df_cat["date"] <= cutoff]
        test_c  = df_cat[df_cat["date"] > cutoff]
        if len(train_c) < 30 or len(test_c) < 7:
            continue

        base_model = LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42, verbose=-1)
        base_model.fit(train_c[features], train_c[target], categorical_feature=[f for f in categorical_features if f in features])
        fi = pd.Series(base_model.feature_importances_, index=features).sort_values(ascending=False)

        best_score = float("inf")
        best_features = features
        for k in range(10, len(features) + 1, 5):
            top_k = fi.head(k).index.tolist()
            cat_features_k = [f for f in categorical_features if f in top_k]
            model_k = LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42, verbose=-1)
            model_k.fit(train_c[top_k], train_c[target], categorical_feature=cat_features_k)
            preds_k = np.clip(model_k.predict(test_c[top_k]), 0, None)
            rmse_k = mean_squared_error(test_c[target], preds_k)
            if np.sqrt(rmse_k) < best_score:
                best_score = np.sqrt(rmse_k)
                best_features = top_k

        model = LGBMRegressor(
            n_estimators=500,
            learning_rate=0.03,
            max_depth=7,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1,
        )
        model.fit(
            train_c[best_features], train_c[target],
            categorical_feature=[f for f in categorical_features if f in best_features]
        )
        test_c = test_c.copy()
        test_c["prediction"] = np.clip(model.predict(test_c[best_features]), 0, None)
        category_preds.append(test_c)
        category_models[cat] = {"model": model, "features": best_features}

        rmse = mean_squared_error(test_c[target], test_c["prediction"])
        mape = mean_absolute_percentage_error(test_c[target][test_c[target] != 0], test_c["prediction"][test_c[target] != 0])
        smape = np.mean(2 * np.abs(test_c[target] - test_c["prediction"]) / (np.abs(test_c[target]) + np.abs(test_c["prediction"]) + 1e-8))
        r2 = r2_score(test_c[target], test_c["prediction"])
        category_metrics[cat] = {"RMSE": np.sqrt(rmse), "MAPE": mape, "SMAPE": smape, "R2": r2, "N_features": len(best_features)}

    # Aggregate outputs
    df_all = pd.concat(category_preds, axis=0) if category_preds else pd.DataFrame()
    metrics_by_category = (
        pd.DataFrame(category_metrics).T.reset_index().rename(columns={"index": "category"})
        if category_metrics else pd.DataFrame()
    )

    # Produce one example plot (weekly agg) to display in UI
    img_b64 = None
    actual_plots = []
    if not df_all.empty:
        df_all["week"] = df_all["date"] - pd.to_timedelta(df_all["date"].dt.dayofweek, unit="d")
        agg = df_all.groupby(["category", "week"])[["units_sold", "prediction"]].sum().reset_index()
        # one global sample
        cat0 = agg["category"].unique()[0]
        df_cat0 = agg[agg["category"] == cat0]
        plt.figure(figsize=(10, 4))
        plt.plot(df_cat0["week"], df_cat0["units_sold"], label="Actual", marker="o")
        plt.plot(df_cat0["week"], df_cat0["prediction"], label="Predicted", marker="x")
        plt.title(f"{cat0} – Weekly Prediction vs Actual")
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.legend()
        img_b64 = _encode_plot_to_base64()

        # per-category plots
        for cat in agg["category"].unique():
            df_cat = agg[agg["category"] == cat]
            plt.figure(figsize=(10, 4))
            plt.plot(df_cat["week"], df_cat["units_sold"], label="Actual", marker="o")
            plt.plot(df_cat["week"], df_cat["prediction"], label="Predicted", marker="x")
            plt.title(f"{cat} – Weekly Prediction vs Actual")
            plt.xticks(rotation=45)
            plt.grid(True)
            plt.legend()
            actual_plots.append({
                "category": cat,
                "plot_b64": _encode_plot_to_base64(),
            })

    # Optional: future forecasting for N days by iterative rollout
    forecasts_by_category = []
    forecast_plot_b64 = None
    forecast_plots = []
    if horizon_days and category_models:
        future_frames = []
        for cat, bundle in category_models.items():
            model = bundle["model"]
            used_features = bundle["features"]
            df_cat_full = df[df["category"] == cat].copy()
            last_date = df_cat_full["date"].max()
            # Work on a rolling frame that we will extend
            roll = df_cat_full.sort_values("date").copy()
            for step in range(1, horizon_days + 1):
                next_date = last_date + pd.Timedelta(days=step)
                # Build a new row by copying last known exogenous values
                last_row = roll.iloc[-1].copy()
                new_row = last_row.copy()
                new_row["date"] = next_date
                # carry-forward exogenous vars
                # if business-specific calendars are available, replace these with realistic schedules
                for col in ["price_unit", "promotion_flag", "delivery_days", "stock_available", "is_month_start", "is_month_end", "dayofweek", "month", "year", "is_weekend"]:
                    if col in new_row.index:
                        # recompute calendar columns from date
                        if col == "is_month_start":
                            new_row[col] = int(next_date.is_month_start)
                        elif col == "is_month_end":
                            new_row[col] = int(next_date.is_month_end)
                        elif col == "dayofweek":
                            new_row[col] = next_date.dayofweek
                        elif col == "month":
                            new_row[col] = next_date.month
                        elif col == "year":
                            new_row[col] = next_date.year
                        elif col == "is_weekend":
                            new_row[col] = int(next_date.dayofweek in [5,6])
                        else:
                            # keep previous value
                            pass
                # update seasonality
                new_row["month_sin"] = np.sin(2 * np.pi * new_row["month"] / 12)
                new_row["month_cos"] = np.cos(2 * np.pi * new_row["month"] / 12)
                new_row["dow_sin"]   = np.sin(2 * np.pi * new_row["dayofweek"] / 7)
                new_row["dow_cos"]   = np.cos(2 * np.pi * new_row["dayofweek"] / 7)

                # update counters depending on events (promotion active == 1 resets time_since_promo to 0)
                if "time_since_promo" in new_row.index and "promotion_flag" in new_row.index:
                    new_row["time_since_promo"] = 0 if new_row["promotion_flag"] == 1 else (roll.iloc[-1]["time_since_promo"] + 1)
                if "time_since_delivery" in new_row.index and "delivery_days" in new_row.index:
                    new_row["time_since_delivery"] = 0 if new_row["delivery_days"] == 0 else (roll.iloc[-1]["time_since_delivery"] + 1)

                # compute lag-based features from current rolling frame
                # append placeholder to compute lags easily
                temp = pd.concat([roll, new_row.to_frame().T], ignore_index=True)
                for lag in [1, 2, 3, 7, 14, 28]:
                    temp.loc[temp.index[-1], f"lag_{lag}"] = temp["units_sold"].shift(lag).iloc[-1]
                temp.loc[temp.index[-1], "momentum_7_1"] = temp.loc[temp.index[-1], "lag_1"] / (temp.loc[temp.index[-1], "lag_7"] + 1e-3)
                temp.loc[temp.index[-1], "momentum_14_7"] = temp.loc[temp.index[-1], "lag_7"] / (temp.loc[temp.index[-1], "lag_14"] + 1e-3)
                # simple rolling trend using last 7 actual/pred values
                hist = temp["units_sold"].shift(1).iloc[-7:]
                temp.loc[temp.index[-1], "trend_7"] = np.polyfit(range(len(hist)), hist.fillna(0), 1)[0] if hist.notna().sum() == 7 else np.nan

                # promo and other lags
                temp.loc[temp.index[-1], "price_lag_1"] = temp["price_unit"].shift(1).iloc[-1]
                temp.loc[temp.index[-1], "promo_lag_1"] = temp["promotion_flag"].shift(1).iloc[-1]
                temp.loc[temp.index[-1], "promo_effect"] = temp.loc[temp.index[-1], "promotion_flag"] * temp.loc[temp.index[-1], "lag_1"]
                temp.loc[temp.index[-1], "delivery_lag_7"] = temp["delivery_days"].shift(7).iloc[-1]
                temp.loc[temp.index[-1], "stock_lag_1"] = temp["stock_available"].shift(1).iloc[-1]

                # category rolling (approximate with previous window values)
                # for forecasting per category, use the same category's history
                temp.loc[temp.index[-1], "category_rolling_mean_7"] = temp["units_sold"].shift(1).rolling(7).mean().iloc[-1]
                temp.loc[temp.index[-1], "category_rolling_std_7"] = temp["units_sold"].shift(1).rolling(7).std().iloc[-1]
                temp.loc[temp.index[-1], "price_rolling_mean_7"] = temp["price_unit"].shift(1).rolling(7).mean().iloc[-1]
                temp.loc[temp.index[-1], "promo_rolling_7"] = temp["promotion_flag"].shift(1).rolling(7).sum().iloc[-1]

                # stockout derived
                if "stockout_flag" in temp.columns:
                    temp.loc[temp.index[-1], "stockout_flag"] = int((temp.loc[temp.index[-1], "stock_available"] == 0) and (temp.loc[temp.index[-1], "delivered_qty"] > 0))
                    temp.loc[temp.index[-1], "rolling_stockouts_7"] = temp["stockout_flag"].shift(1).rolling(7).sum().iloc[-1]

                # predict next units_sold
                X_next = temp.iloc[[-1]].copy()
                # Ensure full feature alignment and numeric dtypes
                X_next = X_next.reindex(columns=used_features, fill_value=0)
                X_next = X_next.apply(pd.to_numeric, errors='coerce').fillna(0)
                y_next = float(np.clip(model.predict(X_next)[0], 0, None))
                temp.loc[temp.index[-1], "units_sold"] = y_next
                temp.loc[temp.index[-1], "prediction"] = y_next
                roll = temp

            future = roll.tail(horizon_days).copy()
            future["horizon_day"] = range(1, horizon_days + 1)
            future_frames.append(future[["date", "category", "sku", "units_sold", "horizon_day"]])

        if future_frames:
            df_future = pd.concat(future_frames, ignore_index=True)
            # build a small plot for one category
            sample_cat = df_future["category"].unique()[0]
            df_cat = df_future[df_future["category"] == sample_cat]
            plt.figure(figsize=(10, 4))
            plt.plot(df_cat["date"], df_cat["units_sold"], marker='o')
            plt.title(f"{sample_cat} – {horizon_days}-day Forecast")
            plt.xticks(rotation=45)
            plt.grid(True)
            forecast_plot_b64 = _encode_plot_to_base64()

            # per-category forecast plots
            for cat in df_future["category"].unique():
                fc = df_future[df_future["category"] == cat]
                plt.figure(figsize=(10, 4))
                plt.plot(fc["date"], fc["units_sold"], marker='o')
                plt.title(f"{cat} – {horizon_days}-day Forecast")
                plt.xticks(rotation=45)
                plt.grid(True)
                forecast_plots.append({
                    "category": cat,
                    "plot_b64": _encode_plot_to_base64(),
                })
            # Per-category aggregates
            agg_future = (
                df_future.groupby(["category", "date"], as_index=False)["units_sold"].sum()
                .rename(columns={"units_sold": "forecast"})
            )
            # Ensure JSON-serializable types
            agg_future["date"] = agg_future["date"].astype(str)
            agg_future["forecast"] = agg_future["forecast"].astype(float)
            forecasts_by_category = agg_future.to_dict(orient='records')

    # Prepare metrics as JSON-serializable
    metrics_records = []
    if not metrics_by_category.empty:
        for rec in metrics_by_category.to_dict(orient='records'):
            safe = {k: (float(v) if isinstance(v, (np.floating, np.float32, np.float64)) else v) for k, v in rec.items()}
            metrics_records.append(safe)

    return {
        "metrics": metrics_records,
        "plot_b64": img_b64,
        "actual_plots": actual_plots,
        "forecasts": forecasts_by_category,
        "forecast_plot_b64": forecast_plot_b64,
        "forecast_plots": forecast_plots,
    }


