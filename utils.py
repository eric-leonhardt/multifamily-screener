"""
utils.py — Shared utilities for the rent model pipeline.

Haversine distance, data cleaning, feature encoding, imputation,
and column auto-classification for interactive setup.
"""

import numpy as np
import pandas as pd


# ─── Distance ────────────────────────────────────────────────

def haversine_miles(lat1, lon1, lat2, lon2):
    """Vectorized haversine distance in miles."""
    R = 3958.8
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def add_landmark_distances(df, landmarks, lat_col="Latitude", lon_col="Longitude"):
    """Add a Dist_<name> column for each landmark. Returns df and new col names."""
    dist_cols = []
    for lm in landmarks:
        col = f"Dist_{lm['name'].replace(' ', '_')}"
        df[col] = haversine_miles(
            df[lat_col].values, df[lon_col].values, lm["lat"], lm["lon"]
        )
        dist_cols.append(col)
    return df, dist_cols


# ─── Parsing ─────────────────────────────────────────────────

def safe_numeric(series):
    """Convert to numeric, coercing errors to NaN."""
    return pd.to_numeric(series, errors="coerce")


def normalize_yn(val):
    """Map yes/no/y/n variants to 1/0, else NaN."""
    if pd.isna(val):
        return np.nan
    s = str(val).strip().lower()
    if s in ("yes", "y", "1", "true", "t"):
        return 1.0
    if s in ("no", "n", "0", "false", "f"):
        return 0.0
    return np.nan


# ─── Column classification ───────────────────────────────────

def classify_columns(df, col_map, rent_prefix):
    """
    Auto-classify DataFrame columns into binary_yn, numeric, and categorical.
    Excludes ID/key columns and rent columns.

    Returns dict with keys: binary_yn, numeric, categorical
    """
    # Columns to skip (IDs, keys, rent, coordinates)
    skip = set()
    for v in col_map.values():
        if v:
            skip.add(v)
    skip.update(c for c in df.columns if c.startswith(rent_prefix))
    skip.update(["LatestRent", "AvgSqFt", "RentPerSqFt", "ParkingPerUnit"])

    binary_yn = []
    numeric = []
    categorical = []

    for col in df.columns:
        if col in skip:
            continue

        series = df[col].dropna()
        if len(series) == 0:
            continue

        # Check if it's a Yes/No column
        vals_lower = series.astype(str).str.strip().str.lower().unique()
        yn_vals = {"yes", "no", "y", "n"}
        if len(vals_lower) <= 4 and set(vals_lower).issubset(yn_vals | {"nan", ""}):
            binary_yn.append(col)
            continue

        # Check if numeric
        numeric_converted = pd.to_numeric(series, errors="coerce")
        pct_numeric = numeric_converted.notna().mean()
        if pct_numeric > 0.8:
            numeric.append(col)
            continue

        # Categorical: string with limited unique values
        n_unique = series.nunique()
        if 2 <= n_unique <= 20:
            categorical.append(col)

    return {
        "binary_yn": sorted(binary_yn),
        "numeric": sorted(numeric),
        "categorical": sorted(categorical),
    }


# ─── Feature encoding ────────────────────────────────────────

def encode_binary_yn(df, columns):
    """Convert Yes/No columns to binary 0/1. Returns df and new column names."""
    new_cols = []
    for col in columns:
        if col not in df.columns:
            continue
        new_name = f"{col}_bin"
        df[new_name] = df[col].apply(normalize_yn)
        new_cols.append(new_name)
    return df, new_cols


def encode_categorical_dummies(df, columns):
    """One-hot encode categoricals (drop first). Returns df and new column names."""
    new_cols = []
    for col in columns:
        if col not in df.columns:
            continue
        df[col] = df[col].fillna("Unknown").astype(str)
        dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
        df = pd.concat([df, dummies], axis=1)
        new_cols.extend(dummies.columns.tolist())
    return df, new_cols


def encode_custom_interactive(df, custom_config):
    """
    Apply custom encoding rules from config. Simplified for interactive mode.

    Supported types:
      - binary_match: 1 if value matches
      - one_hot: dummies for specific values
      - contains: substring match
      - binary_any: 1 if value in list
    """
    new_cols = []
    if not custom_config:
        return df, new_cols

    for col_name, spec in custom_config.items():
        if col_name not in df.columns:
            continue

        col_lower = df[col_name].astype(str).str.lower().str.strip()
        enc_type = spec.get("type", "")

        if enc_type == "binary_match":
            out = spec["output_name"]
            df[out] = (col_lower == spec["match_value"].lower()).astype(float)
            new_cols.append(out)

        elif enc_type == "one_hot":
            for val in spec["values"]:
                out = f"{spec['prefix']}_{val.title()}"
                df[out] = (col_lower == val.lower()).astype(float)
                new_cols.append(out)

        elif enc_type == "contains":
            for out_name, pattern in spec["patterns"].items():
                df[out_name] = col_lower.str.contains(pattern.lower(), na=False).astype(float)
                new_cols.append(out_name)

        elif enc_type == "binary_any":
            out = spec["output_name"]
            match_vals = [v.lower() for v in spec["match_values"]]
            df[out] = col_lower.isin(match_vals).astype(float)
            new_cols.append(out)

    return df, new_cols


# ─── Imputation ──────────────────────────────────────────────

def impute_columns(df, columns, numeric_method="mean"):
    """Impute NaN: mean/median for numeric, mode for others."""
    for col in columns:
        if col not in df.columns:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            fill = df[col].mean() if numeric_method == "mean" else df[col].median()
            df[col] = df[col].fillna(fill if not pd.isna(fill) else 0)
        else:
            mode_vals = df[col].mode()
            df[col] = df[col].fillna(mode_vals.iloc[0] if len(mode_vals) > 0 else "Unknown")
    return df


# ─── Latest rent ─────────────────────────────────────────────

def get_latest_rent(df, prefix="RentYearQuarter"):
    """Find most recent non-NaN rent per row across quarterly columns."""
    rent_cols = sorted([c for c in df.columns if c.startswith(prefix)])
    for c in rent_cols:
        df[c] = safe_numeric(df[c])
    df["LatestRent"] = df[rent_cols[::-1]].bfill(axis=1).iloc[:, 0]
    return df, rent_cols


# ─── Zero-variance removal ───────────────────────────────────

def drop_zero_variance(X, feature_names):
    """Remove columns with zero variance."""
    var = np.var(X, axis=0)
    mask = var > 0
    return X[:, mask], [f for f, m in zip(feature_names, mask) if m]
