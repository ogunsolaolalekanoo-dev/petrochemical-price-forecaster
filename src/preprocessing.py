# src/preprocessing.py

import os
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings("ignore")


# ────────────────────────────────────────────────────────────────────
# LOAD
# ────────────────────────────────────────────────────────────────────
def load_data(path="data/raw/master_dataset.csv"):
    print("📂 Loading master dataset...")
    df = pd.read_csv(
        path, index_col="date", parse_dates=True
    )
    print(f"   ✅ {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"   Range: {df.index.min().date()} → "
          f"{df.index.max().date()}")
    return df


# ────────────────────────────────────────────────────────────────────
# MISSING VALUES
# ────────────────────────────────────────────────────────────────────
def report_missing(df, label=""):
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        print(f"   ✅ {label} No missing values")
    else:
        pct    = (missing / len(df) * 100).round(2)
        report = pd.DataFrame({
            "missing": missing, "pct": pct
        })
        print(f"\n🔍 Missing Values {label}:")
        print(report.to_string())
    return missing


def handle_missing(df):
    print("\n🧹 Handling missing values...")
    before = df.isnull().sum().sum()

    # GDP is quarterly — forward fill is correct
    df["gdp_growth"] = df["gdp_growth"].ffill()

    # All others: interpolate then fill edges
    df = df.interpolate(
        method="linear", limit_direction="both"
    )
    df = df.ffill().bfill()

    after = df.isnull().sum().sum()
    print(f"   Before: {before} | After: {after}")
    print("   ✅ Missing values resolved")
    return df


# ────────────────────────────────────────────────────────────────────
# STATIONARITY
# ────────────────────────────────────────────────────────────────────
def run_adf_test(series, name):
    result  = adfuller(series.dropna())
    p_value = result[1]
    status  = ("✅ Stationary" if p_value < 0.05
               else "⚠️  Non-Stationary")
    print(f"   {name:<38} p={p_value:.4f}  {status}")
    return p_value


def check_stationarity(df):
    print("\n📊 Stationarity Report:")
    print("   " + "-" * 62)

    raw_cols = [
        "resin_ppi", "thermosetting_ppi",
        "wti_crude", "natural_gas_price",
        "housing_starts", "dollar_index",
        "crude_brent", "chemical_production",
        "crude_inventory"
    ]
    pct_cols = [
        "resin_ppi_pct_change", "crude_pct_change",
        "gas_pct_change", "housing_pct_change",
        "dollar_pct_change", "commodity_pct_change",
        "chem_prod_pct_change", "inventory_pct_change"
    ]

    results = {}
    print("   Raw price levels:")
    for col in raw_cols:
        if col in df.columns:
            results[col] = run_adf_test(df[col], col)

    print("\n   Percentage change series:")
    for col in pct_cols:
        if col in df.columns:
            results[col] = run_adf_test(df[col], col)

    return results


# ────────────────────────────────────────────────────────────────────
# WINSORIZATION
# ────────────────────────────────────────────────────────────────────
def winsorize(df, cols, lower=0.02, upper=0.98):
    """
    Caps extreme values at percentile boundaries.
    Reduces distorting effect of black swan events
    without deleting real data points.
    """
    print("\n✂️  Winsorizing outliers...")
    df_w = df.copy()
    for col in cols:
        if col in df_w.columns:
            lo      = df_w[col].quantile(lower)
            hi      = df_w[col].quantile(upper)
            clipped = df_w[col].clip(lo, hi)
            n       = (df_w[col] != clipped).sum()
            if n > 0:
                print(f"   {col}: {n} values clipped "
                      f"[{lo:.2f} → {hi:.2f}]")
            df_w[col] = clipped
    print("   ✅ Winsorization complete")
    return df_w


# ────────────────────────────────────────────────────────────────────
# PREPARE PROPHET DATASET
# ────────────────────────────────────────────────────────────────────
def prepare_prophet_data(df):
    """
    Prophet format: ds (date) + y (target) + regressors.
    Prophet handles non-stationarity internally through
    trend decomposition — use raw prices as target.
    """
    print("\n🔮 Preparing Prophet dataset...")

    regressor_cols = [
        "crude_lag_2m", "crude_lag_3m",
        "natural_gas_price", "all_commodity_ppi",
        "housing_lag_3m", "manufacturing_pmi",
        "dollar_index", "crude_ma_3m", "crude_vol_6m",
        "crude_x_housing", "crude_x_commodity",
        "gas_x_crude", "resin_rsi",
        "is_covid", "is_post_covid_surge",
        "is_construction_season", "is_2026_shock",
    ]

    available = [c for c in regressor_cols
                 if c in df.columns]

    data = {"ds": df.index, "y": df["resin_ppi"].values}
    for col in available:
        data[col] = df[col].values

    prophet_df = pd.DataFrame(data)
    prophet_df = prophet_df.dropna(
        subset=["y", "crude_lag_2m"]
    ).reset_index(drop=True)

    print(f"   ✅ Prophet ready: {prophet_df.shape[0]} rows, "
          f"{prophet_df.shape[1]} columns")
    print(f"   Regressors: {available}")
    return prophet_df


# ────────────────────────────────────────────────────────────────────
# PREPARE ARIMAX DATASET
# ────────────────────────────────────────────────────────────────────
def prepare_arimax_data(df):
    """
    ARIMAX works on stationary data (% changes).
    Target:    resin_ppi_pct_change
    Exogenous: all stationary features + dummies
    """
    print("\n📈 Preparing ARIMAX dataset...")

    y = df["resin_ppi_pct_change"]

    exog_cols = {
        "crude_pct_change":       df["crude_pct_change"],
        "crude_lag_2m_pct":       df["crude_lag_2m_pct_change"],
        "gas_pct_change":         df["gas_pct_change"],
        "commodity_pct_change":   df["commodity_pct_change"],
        "housing_pct_change":     df["housing_pct_change"],
        "dollar_pct_change":      df["dollar_pct_change"],
        "crude_vol_3m":           df["crude_vol_3m"],
        "resin_acceleration":     df["resin_acceleration"],
        "crude_acceleration":     df["crude_acceleration"],
        "is_covid":               df["is_covid"],
        "is_post_covid_surge":    df["is_post_covid_surge"],
        "is_construction_season": df["is_construction_season"],
        "is_2026_shock":          df["is_2026_shock"],
    }

    # Add new features if available
    for col in ["chem_prod_pct_change", "inventory_pct_change"]:
        if col in df.columns:
            exog_cols[col] = df[col]

    exog     = pd.DataFrame(exog_cols)
    combined = pd.concat([y, exog], axis=1).dropna()

    y_clean    = combined["resin_ppi_pct_change"]
    exog_clean = combined.drop(
        columns=["resin_ppi_pct_change"]
    )

    print(f"   ✅ ARIMAX ready: {len(y_clean)} obs, "
          f"{exog_clean.shape[1]} exog variables")
    return y_clean, exog_clean


# ────────────────────────────────────────────────────────────────────
# PREPARE LIGHTGBM DATASET
# ────────────────────────────────────────────────────────────────────
def prepare_lgb_data(df):
    """
    LightGBM handles non-stationarity natively.
    Use raw price levels as target with full feature set.
    """
    print("\n🌲 Preparing LightGBM dataset...")

    features = [
        # Lag features
        "crude_lag_1m", "crude_lag_2m", "crude_lag_3m",
        "crude_lag_4m", "crude_lag_6m",
        "brent_lag_1m", "brent_lag_2m",
        "gas_lag_1m", "gas_lag_2m",
        "housing_lag_3m", "housing_lag_6m",
        "chem_prod_lag_2m", "chem_prod_lag_3m",
        "inventory_lag_1m", "inventory_lag_2m",

        # Rolling features
        "crude_ma_3m", "crude_ma_6m", "crude_ma_12m",
        "crude_vol_3m", "crude_vol_6m",
        "resin_ma_3m", "resin_ma_6m", "resin_vol_3m",
        "wti_brent_spread",
        "chem_prod_ma_3m", "inventory_ma_3m",

        # Level features
        "natural_gas_price", "all_commodity_ppi",
        "manufacturing_pmi", "dollar_index",
        "housing_starts", "unemployment",
        "chemical_production", "crude_inventory",

        # Interaction features
        "crude_x_housing", "crude_x_commodity",
        "gas_x_crude", "crude_x_chem_prod",
        "inventory_x_crude",

        # Acceleration features
        "resin_acceleration", "crude_acceleration",
        "resin_jerk",

        # Technical indicator
        "resin_rsi", "vol_ratio",

        # Calendar features
        "month", "quarter", "year",
        "is_construction_season", "is_q1",

        # Structural breaks
        "is_covid", "is_post_covid_surge",
        "is_gfc", "is_2026_shock",
    ]

    available = [f for f in features if f in df.columns]
    X         = df[available].dropna()
    y         = df["resin_ppi"].loc[X.index]

    print(f"   ✅ LightGBM ready: {len(X)} rows, "
          f"{len(available)} features")
    print(f"   Features ({len(available)}):")
    for i, f in enumerate(available, 1):
        print(f"   {i:>2}. {f}")
    return X, y, available


# ────────────────────────────────────────────────────────────────────
# SAVE ALL
# ────────────────────────────────────────────────────────────────────
def save_all(df, prophet_df, y_arimax,
             exog_arimax, X_lgb, y_lgb):
    os.makedirs("data/processed", exist_ok=True)

    df.to_csv("data/processed/clean_master.csv")
    prophet_df.to_csv(
        "data/processed/prophet_ready.csv", index=False
    )
    y_arimax.to_csv("data/processed/arimax_target.csv")
    exog_arimax.to_csv("data/processed/arimax_exog.csv")
    X_lgb.to_csv("data/processed/xgb_features.csv")
    y_lgb.to_csv("data/processed/xgb_target.csv")

    print("\n💾 Saved all processed datasets:")
    print("   → data/processed/clean_master.csv")
    print("   → data/processed/prophet_ready.csv")
    print("   → data/processed/arimax_target.csv")
    print("   → data/processed/arimax_exog.csv")
    print("   → data/processed/xgb_features.csv")
    print("   → data/processed/xgb_target.csv")


# ────────────────────────────────────────────────────────────────────
# RUN
# ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    df = load_data()
    report_missing(df, "BEFORE:")
    df = handle_missing(df)
    report_missing(df, "AFTER:")
    check_stationarity(df)

    # Winsorize percentage change columns only
    pct_cols = [
        "resin_ppi_pct_change", "crude_pct_change",
        "gas_pct_change", "housing_pct_change",
        "dollar_pct_change", "commodity_pct_change",
        "chem_prod_pct_change", "inventory_pct_change",
        "resin_acceleration", "crude_acceleration",
    ]
    df = winsorize(df, pct_cols, lower=0.02, upper=0.98)

    prophet_df            = prepare_prophet_data(df)
    y_arimax, exog_arimax = prepare_arimax_data(df)
    X_lgb, y_lgb, feats   = prepare_lgb_data(df)

    save_all(
        df, prophet_df, y_arimax,
        exog_arimax, X_lgb, y_lgb
    )

    print("\n✅ Preprocessing complete — ready for modeling!")