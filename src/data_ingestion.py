# src/data_ingestion.py

import os
import numpy as np
import pandas as pd
import requests
from fredapi import Fred
from dotenv import load_dotenv

# ── Load API keys ────────────────────────────────────────────────────
load_dotenv()
FRED_API_KEY = os.getenv("FRED_API_KEY")
EIA_API_KEY  = os.getenv("EIA_API_KEY")

fred = Fred(api_key=FRED_API_KEY)

# ── FRED SERIES ──────────────────────────────────────────────────────
FRED_SERIES = {
    # ── Core Resin Price Indices
    "resin_ppi":           "WPU066",
    "thermosetting_ppi":   "PCU3252113252114",
    "thermoplastic_ppi":   "PCU325211325211P",

    # ── Macro Baselines
    "all_commodity_ppi":   "PPIACO",
    "natural_gas_price":   "MHHNGSP",
    "manufacturing_pmi":   "MANEMP",

    # ── Demand Drivers
    "housing_starts":      "HOUST",
    "construction_spend":  "TTLCONS",

    # ── Macroeconomic Context
    "gdp_growth":          "A191RL1Q225SBEA",
    "unemployment":        "UNRATE",

    # ── Currency Effect
    "dollar_index":        "DTWEXBGS",

    # ── Energy Market
    "crude_brent":         "DCOILBRENTEU",

    # ── Chemical Industry Production
    # High production = high feedstock demand = higher prices
    "chemical_production": "IPG325S",

    # ── Crude Oil Inventory
    # More inventory = more supply = lower prices
    "crude_inventory":     "WCESTUS1",
}

EIA_SERIES = {
    "wti_crude": "PET.RWTC.M",
}


# ────────────────────────────────────────────────────────────────────
# FUNCTION 1: Pull FRED data
# ────────────────────────────────────────────────────────────────────
def pull_fred_data(start_date="2000-01-01", end_date=None):
    print("📥 Pulling FRED data...")
    frames = {}

    for name, series_id in FRED_SERIES.items():
        try:
            print(f"   → Fetching {name} ({series_id})...")
            series = fred.get_series(
                series_id,
                observation_start=start_date,
                observation_end=end_date
            )
            frames[name] = series
            print(f"   ✅ {name}: {len(series)} observations")
        except Exception as e:
            print(f"   ❌ Failed {name}: {e}")

    df = pd.DataFrame(frames)
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"

    # Resample to month-start frequency
    df = df.resample("MS").mean()

    # GDP is quarterly — forward fill missing months
    if "gdp_growth" in df.columns:
        df["gdp_growth"] = df["gdp_growth"].ffill()

    print(f"\n✅ FRED pulled: {df.shape[0]} rows x "
          f"{df.shape[1]} cols")
    print(f"   Range: {df.index.min().date()} → "
          f"{df.index.max().date()}\n")
    return df


# ────────────────────────────────────────────────────────────────────
# FUNCTION 2: Pull EIA crude oil data
# ────────────────────────────────────────────────────────────────────
def pull_eia_data(start_date="2000-01-01"):
    print("📥 Pulling EIA crude oil data...")

    url = (
        f"https://api.eia.gov/v2/seriesid/{EIA_SERIES['wti_crude']}"
        f"?api_key={EIA_API_KEY}&frequency=monthly&data[0]=value"
        f"&start={start_date[:7]}"
        f"&sort[0][column]=period&sort[0][direction]=asc"
    )

    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        data    = response.json()
        records = data["response"]["data"]

        df_eia = pd.DataFrame(records)[["period", "value"]]
        df_eia.columns = ["date", "wti_crude"]
        df_eia["date"] = pd.to_datetime(df_eia["date"])
        df_eia["wti_crude"] = pd.to_numeric(
            df_eia["wti_crude"], errors="coerce"
        )
        df_eia = df_eia.set_index("date").sort_index()
        df_eia.index = df_eia.index + pd.offsets.MonthBegin(0)

        print(f"✅ EIA pulled: {len(df_eia)} observations")
        print(f"   Range: {df_eia.index.min().date()} → "
              f"{df_eia.index.max().date()}\n")
        return df_eia

    except Exception as e:
        print(f"❌ EIA failed: {e}")
        return pd.DataFrame()


# ────────────────────────────────────────────────────────────────────
# FUNCTION 3: Engineer all features
# ────────────────────────────────────────────────────────────────────
def engineer_features(df):
    """
    Creates all derived features on top of raw pulled data.
    All column existence checks prevent crashes if a FRED
    series failed to pull.

    Categories:
    1. Lag features        — capture supply chain delay
    2. Rolling features    — capture momentum and volatility
    3. Percentage changes  — stationarity for ARIMAX
    4. Interaction features — compound market effects
    5. Acceleration        — rate of change signals
    6. Technical indicators — RSI overbought/oversold
    7. Calendar features   — seasonality
    8. Structural dummies  — flag extreme events
    """
    print("⚙️  Engineering features...")

    # ── Lag Features ─────────────────────────────────────────────────

    # WTI crude oil lags — test multiple lag lengths
    if "wti_crude" in df.columns:
        for lag in [1, 2, 3, 4, 6]:
            df[f"crude_lag_{lag}m"] = df["wti_crude"].shift(lag)

    # Brent crude lags — international benchmark
    if "crude_brent" in df.columns:
        for lag in [1, 2, 3]:
            df[f"brent_lag_{lag}m"] = df["crude_brent"].shift(lag)

    # Natural gas lags — energy cost delay
    if "natural_gas_price" in df.columns:
        df["gas_lag_1m"] = df["natural_gas_price"].shift(1)
        df["gas_lag_2m"] = df["natural_gas_price"].shift(2)

    # Housing starts lags — demand takes time to flow through
    if "housing_starts" in df.columns:
        df["housing_lag_3m"] = df["housing_starts"].shift(3)
        df["housing_lag_6m"] = df["housing_starts"].shift(6)

    # Chemical production lags — only if pull succeeded
    if "chemical_production" in df.columns:
        df["chem_prod_lag_2m"] = (
            df["chemical_production"].shift(2)
        )
        df["chem_prod_lag_3m"] = (
            df["chemical_production"].shift(3)
        )
    else:
        print("   ⚠️  chemical_production not available — skipping")

    # Crude inventory lags — only if pull succeeded
    if "crude_inventory" in df.columns:
        df["inventory_lag_1m"] = df["crude_inventory"].shift(1)
        df["inventory_lag_2m"] = df["crude_inventory"].shift(2)
    else:
        print("   ⚠️  crude_inventory not available — skipping")

    # ── Rolling Features ─────────────────────────────────────────────

    if "wti_crude" in df.columns:
        df["crude_ma_3m"]  = df["wti_crude"].rolling(3).mean()
        df["crude_ma_6m"]  = df["wti_crude"].rolling(6).mean()
        df["crude_ma_12m"] = df["wti_crude"].rolling(12).mean()
        df["crude_vol_3m"] = df["wti_crude"].rolling(3).std()
        df["crude_vol_6m"] = df["wti_crude"].rolling(6).std()

    if "resin_ppi" in df.columns:
        df["resin_ma_3m"]  = df["resin_ppi"].rolling(3).mean()
        df["resin_ma_6m"]  = df["resin_ppi"].rolling(6).mean()
        df["resin_vol_3m"] = df["resin_ppi"].rolling(3).std()

    if ("wti_crude" in df.columns and
            "crude_brent" in df.columns):
        df["wti_brent_spread"] = (
            df["wti_crude"] - df["crude_brent"]
        )

    if "chemical_production" in df.columns:
        df["chem_prod_ma_3m"] = (
            df["chemical_production"].rolling(3).mean()
        )

    if "crude_inventory" in df.columns:
        df["inventory_ma_3m"] = (
            df["crude_inventory"].rolling(3).mean()
        )

    # ── Percentage Changes ───────────────────────────────────────────
    # Used for stationarity in ARIMAX model
    pct_map = {
        "resin_ppi_pct_change":    "resin_ppi",
        "crude_pct_change":        "wti_crude",
        "crude_lag_2m_pct_change": "crude_lag_2m",
        "gas_pct_change":          "natural_gas_price",
        "housing_pct_change":      "housing_starts",
        "commodity_pct_change":    "all_commodity_ppi",
        "dollar_pct_change":       "dollar_index",
        "chem_prod_pct_change":    "chemical_production",
        "inventory_pct_change":    "crude_inventory",
    }
    for new_col, source_col in pct_map.items():
        if source_col in df.columns:
            df[new_col] = (
                df[source_col].pct_change(fill_method=None) * 100
            )
        else:
            print(f"   ⚠️  Skipping {new_col} "
                  f"({source_col} not available)")

    # ── Interaction Features ─────────────────────────────────────────
    # Captures compound effects — when two drivers move together
    # the impact is stronger than either alone

    # High crude + high housing = maximum resin price pressure
    if ("crude_lag_2m" in df.columns and
            "housing_lag_3m" in df.columns):
        df["crude_x_housing"] = (
            df["crude_lag_2m"] * df["housing_lag_3m"] / 1000
        )

    # Crude + broad commodity inflation amplifies resin moves
    if ("crude_lag_2m" in df.columns and
            "all_commodity_ppi" in df.columns):
        df["crude_x_commodity"] = (
            df["crude_lag_2m"] * df["all_commodity_ppi"] / 100
        )

    # Natural gas + crude both rising = double energy cost pressure
    if ("natural_gas_price" in df.columns and
            "wti_crude" in df.columns):
        df["gas_x_crude"] = (
            df["natural_gas_price"] * df["wti_crude"] / 100
        )

    # Crude + chemical production compound effect
    if ("crude_lag_2m" in df.columns and
            "chemical_production" in df.columns):
        df["crude_x_chem_prod"] = (
            df["crude_lag_2m"] *
            df["chemical_production"].fillna(
                df["chemical_production"].mean()
            ) / 100
        )

    # Inventory × crude — low inventory + high crude = price spike
    if ("inventory_lag_1m" in df.columns and
            "wti_crude" in df.columns):
        df["inventory_x_crude"] = (
            df["inventory_lag_1m"] * df["wti_crude"] / 1000
        )

    # ── Acceleration Features ────────────────────────────────────────
    # Is the RATE of price change speeding up or slowing down?
    # Positive acceleration = momentum building
    # Negative acceleration = momentum reversing

    if "resin_ppi_pct_change" in df.columns:
        df["resin_acceleration"] = (
            df["resin_ppi_pct_change"].diff()
        )
        # Second derivative — is acceleration itself changing?
        df["resin_jerk"] = df["resin_acceleration"].diff()

    if "crude_pct_change" in df.columns:
        df["crude_acceleration"] = (
            df["crude_pct_change"].diff()
        )

    # ── Volatility Ratio ─────────────────────────────────────────────
    # Short-term vs long-term volatility ratio
    # Ratio > 1 = market becoming more volatile = more uncertainty
    if ("crude_vol_3m" in df.columns and
            "crude_vol_6m" in df.columns):
        df["vol_ratio"] = (
            df["crude_vol_3m"] /
            df["crude_vol_6m"].replace(0, np.nan)
        )

    # ── RSI — Relative Strength Index ───────────────────────────────
    # RSI > 70 = overbought → price likely to pull back
    # RSI < 30 = oversold  → price likely to bounce up
    # Standard 14-period RSI used in commodity markets
    if "resin_ppi" in df.columns:
        delta = df["resin_ppi"].diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        rs    = gain / loss.replace(0, np.nan)
        df["resin_rsi"] = 100 - (100 / (1 + rs))

    # ── Calendar Features ────────────────────────────────────────────
    df["month"]   = df.index.month
    df["quarter"] = df.index.quarter
    df["year"]    = df.index.year

    # Construction season March–August
    # Peak building activity = peak resin demand
    df["is_construction_season"] = (
        df["month"].isin([3, 4, 5, 6, 7, 8]).astype(int)
    )

    # Q1 flag — typically slow demand, prices soften
    df["is_q1"] = (df["quarter"] == 1).astype(int)

    # ── Structural Break Dummies ─────────────────────────────────────
    # Tell the model to treat extreme periods differently
    # These are binary flags — 1 during the event, 0 otherwise

    # COVID crash: March–September 2020
    df["is_covid"] = (
        (df.index >= "2020-03-01") &
        (df.index <= "2020-09-01")
    ).astype(int)

    # Post-COVID inflation surge: unprecedented resin spike
    df["is_post_covid_surge"] = (
        (df.index >= "2021-01-01") &
        (df.index <= "2022-12-01")
    ).astype(int)

    # 2008 Global Financial Crisis
    df["is_gfc"] = (
        (df.index >= "2008-09-01") &
        (df.index <= "2009-06-01")
    ).astype(int)

    # 2026 Middle East crude oil shock
    df["is_2026_shock"] = (
        df.index >= "2026-01-01"
    ).astype(int)

    print(f"   ✅ Features engineered: {df.shape[1]} total columns")
    return df


# ────────────────────────────────────────────────────────────────────
# FUNCTION 4: Build master dataset
# ────────────────────────────────────────────────────────────────────
def build_master_dataset(start_date="2000-01-01"):
    df_fred = pull_fred_data(start_date=start_date)
    df_eia  = pull_eia_data(start_date=start_date)

    # Merge FRED + EIA on date index
    df = df_fred.join(df_eia, how="left")

    # Engineer all features
    df = engineer_features(df)

    # Save
    os.makedirs("data/raw", exist_ok=True)
    path = "data/raw/master_dataset.csv"
    df.to_csv(path)

    print(f"\n💾 Saved → {path}")
    print(f"   Shape: {df.shape}")
    print(f"\n📋 All columns ({df.shape[1]}):")
    for i, col in enumerate(df.columns, 1):
        print(f"   {i:>2}. {col}")

    # Preview key columns
    key_cols = [
        c for c in [
            "resin_ppi", "wti_crude", "crude_lag_2m",
            "housing_starts", "dollar_index",
            "crude_x_housing", "crude_x_commodity",
            "resin_rsi", "crude_vol_6m"
        ] if c in df.columns
    ]
    print(f"\n🔍 Last 3 rows (key columns):")
    print(df[key_cols].tail(3))
    return df


if __name__ == "__main__":
    df = build_master_dataset(start_date="2000-01-01")