"""
tools.py — Pipeline functions exposed as Claude tool calls.

Each function takes simple arguments and returns plain data (dicts/lists)
that Claude can interpret and explain to the user.
"""

import numpy as np
import pandas as pd
from utils import (
    safe_numeric,
    haversine_miles,
    add_landmark_distances,
    encode_binary_yn,
    encode_categorical_dummies,
    impute_columns,
    get_latest_rent,
    drop_zero_variance,
    classify_columns,
)
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
import io
import base64
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# ─── Tool: Inspect uploaded data ─────────────────────────────

def inspect_data(df):
    """Return basic info about the uploaded dataset."""
    return {
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": df.columns.tolist(),
        "sample_rows": df.head(3).to_dict(orient="records"),
    }


# ─── Tool: List submarkets ──────────────────────────────────

def list_submarkets(df, submarket_col="Submarket"):
    """List all unique submarkets with property counts."""
    if submarket_col not in df.columns:
        # Try to auto-detect
        candidates = [c for c in df.columns if "submarket" in c.lower() or "market" in c.lower()]
        if candidates:
            submarket_col = candidates[0]
        else:
            return {"error": f"No submarket column found. Available columns: {df.columns.tolist()[:20]}"}

    counts = df[submarket_col].value_counts().reset_index()
    counts.columns = ["submarket", "property_count"]
    return {
        "submarket_col": submarket_col,
        "total_submarkets": len(counts),
        "submarkets": counts.head(30).to_dict(orient="records"),
    }


# ─── Tool: Describe columns ─────────────────────────────────

def describe_columns(df, submarket_col="Submarket", rent_prefix="RentYearQuarter"):
    """Auto-classify columns into binary, numeric, categorical."""
    col_map = {
        "property_id": next((c for c in ["PropertyID", "property_id", "ID"] if c in df.columns), None),
        "property_name": next((c for c in ["propertyname", "PropertyName", "Name"] if c in df.columns), None),
        "address": next((c for c in ["Property.Address", "Address"] if c in df.columns), None),
        "latitude": next((c for c in ["Latitude", "latitude", "lat"] if c in df.columns), None),
        "longitude": next((c for c in ["Longitude", "longitude", "lon"] if c in df.columns), None),
        "total_sqft": next((c for c in ["TotalSqFt", "total_sqft"] if c in df.columns), None),
        "num_units": next((c for c in ["Property.NoUnits", "Units"] if c in df.columns), None),
        "parking": next((c for c in ["TotalParkingSpaces", "Parking"] if c in df.columns), None),
    }
    classified = classify_columns(df, col_map, rent_prefix)
    return {
        "detected_columns": col_map,
        "binary_yn_columns": classified["binary_yn"],
        "numeric_columns": classified["numeric"],
        "categorical_columns": classified["categorical"],
        "rent_columns_found": len([c for c in df.columns if c.startswith(rent_prefix)]),
    }


# ─── Tool: Run full analysis ────────────────────────────────

def run_analysis(
    df,
    submarket_filter,
    landmarks=None,
    binary_yn_cols=None,
    numeric_cols=None,
    categorical_cols=None,
    submarket_col="Submarket",
    rent_prefix="RentYearQuarter",
    cv_folds=10,
    top_n=10,
):
    """
    Run the full LASSO pipeline on a filtered submarket.

    Parameters
    ----------
    df : pd.DataFrame — the full dataset
    submarket_filter : str — substring to match in the submarket column
    landmarks : list of dicts with 'name', 'lat', 'lon' (optional)
    binary_yn_cols : list of column names to treat as Yes/No binary
    numeric_cols : list of column names to treat as numeric features
    categorical_cols : list of column names to one-hot encode
    submarket_col : str — column containing submarket labels
    rent_prefix : str — prefix for quarterly rent columns
    cv_folds : int — cross-validation folds
    top_n : int — number of acquisition targets to return

    Returns
    -------
    dict with model results, variable importance, and targets
    """
    # Auto-detect column mappings
    id_col = next((c for c in ["PropertyID", "property_id", "ID"] if c in df.columns), None)
    name_col = next((c for c in ["propertyname", "PropertyName", "Name"] if c in df.columns), None)
    addr_col = next((c for c in ["Property.Address", "Address"] if c in df.columns), None)
    lat_col = next((c for c in ["Latitude", "latitude", "lat"] if c in df.columns), None)
    lon_col = next((c for c in ["Longitude", "longitude", "lon"] if c in df.columns), None)
    sqft_col = next((c for c in ["TotalSqFt", "total_sqft"] if c in df.columns), None)
    units_col = next((c for c in ["Property.NoUnits", "Units"] if c in df.columns), None)
    parking_col = next((c for c in ["TotalParkingSpaces", "Parking"] if c in df.columns), None)

    if not all([lat_col, lon_col, sqft_col, units_col]):
        return {"error": "Could not auto-detect required columns (lat, lon, sqft, units). Please check your data."}

    # Filter to submarket
    sub_df = df[df[submarket_col].str.contains(submarket_filter, case=False, na=False)].copy()
    if len(sub_df) == 0:
        return {"error": f"No properties found matching '{submarket_filter}'"}

    # Parse core columns
    for col in [sqft_col, units_col, parking_col, lat_col, lon_col]:
        if col and col in sub_df.columns:
            sub_df[col] = safe_numeric(sub_df[col])

    if numeric_cols:
        for col in numeric_cols:
            if col in sub_df.columns:
                sub_df[col] = safe_numeric(sub_df[col])

    # Latest rent
    sub_df, _ = get_latest_rent(sub_df, prefix=rent_prefix)

    # Clean
    n_before = len(sub_df)
    sub_df = sub_df[
        (sub_df["LatestRent"] > 0) & (sub_df[sqft_col] > 0) & (sub_df[units_col] > 0)
    ].copy()

    if len(sub_df) < 5:
        return {"error": f"Only {len(sub_df)} valid properties after cleaning. Need at least 5."}

    # Feature engineering
    sub_df["AvgSqFt"] = sub_df[sqft_col] / sub_df[units_col]
    sub_df["RentPerSqFt"] = sub_df["LatestRent"] / sub_df["AvgSqFt"]
    if parking_col and parking_col in sub_df.columns:
        sub_df["ParkingPerUnit"] = sub_df[parking_col] / sub_df[units_col]

    # Landmark distances
    dist_cols = []
    if landmarks and lat_col and lon_col:
        sub_df, dist_cols = add_landmark_distances(sub_df, landmarks, lat_col=lat_col, lon_col=lon_col)

    # Extra numeric
    extra_num = []
    if "Dist_to_city_center" in sub_df.columns:
        sub_df["Dist_to_city_center"] = safe_numeric(sub_df["Dist_to_city_center"])
        extra_num.append("Dist_to_city_center")

    # Binary
    yn_new = []
    if binary_yn_cols:
        sub_df, yn_new = encode_binary_yn(sub_df, binary_yn_cols)

    # Categorical
    cat_new = []
    if categorical_cols:
        sub_df, cat_new = encode_categorical_dummies(sub_df, categorical_cols)

    # Assemble predictors
    pred_cols = (
        (numeric_cols or [])
        + ["AvgSqFt"]
        + (["ParkingPerUnit"] if "ParkingPerUnit" in sub_df.columns else [])
        + extra_num + dist_cols + yn_new + cat_new
    )
    pred_cols = [c for c in pred_cols if c in sub_df.columns]

    if len(pred_cols) < 2:
        return {"error": "Too few features to build a model. Try including more columns."}

    # Impute
    sub_df = impute_columns(sub_df, pred_cols, numeric_method="mean")

    # LASSO
    y = sub_df["RentPerSqFt"].values
    X_raw = sub_df[pred_cols].values.astype(float)
    fnames = list(pred_cols)
    X_raw, fnames = drop_zero_variance(X_raw, fnames)

    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)
    n_folds = min(cv_folds, max(3, len(y) // 5))

    lasso = LassoCV(alphas=None, cv=n_folds, random_state=42, max_iter=20000)
    lasso.fit(X, y)
    yhat = lasso.predict(X)
    resid = y - yhat

    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - np.sum(resid ** 2) / ss_tot
    rmse = np.sqrt(np.mean(resid ** 2))

    # Coefficients
    coef_df = pd.DataFrame({"variable": fnames, "beta": lasso.coef_})
    nonzero = coef_df[coef_df["beta"] != 0].sort_values("beta", key=abs, ascending=False)

    # Variable importance (partial R²)
    importance_rows = []
    for var in nonzero["variable"]:
        idx = fnames.index(var)
        Xd = np.delete(X, idx, axis=1)
        ld = LassoCV(alphas=None, cv=n_folds, random_state=42, max_iter=20000)
        ld.fit(Xd, y)
        r2d = 1 - np.sum((y - ld.predict(Xd)) ** 2) / ss_tot
        importance_rows.append({"variable": var, "partial_r2": round(r2 - r2d, 4)})

    importance_df = pd.DataFrame(importance_rows).sort_values("partial_r2", ascending=False)
    importance_df = nonzero.merge(importance_df, on="variable")

    # Results
    result_cols = [c for c in [id_col, name_col, addr_col, "RentPerSqFt", "AvgSqFt",
                                lat_col, lon_col] if c and c in sub_df.columns]
    results = sub_df[result_cols].copy()
    results["predicted"] = np.round(yhat, 4)
    results["residual"] = np.round(resid, 4)
    results["pct_below_market"] = np.round(resid / yhat * 100, 1)

    # Targets
    targets = results.nsmallest(top_n, "residual")

    # Generate map
    map_b64 = _generate_map_b64(results, landmarks or [], lat_col, lon_col, name_col, submarket_filter)

    return {
        "submarket": submarket_filter,
        "n_properties": len(sub_df),
        "n_cleaned_from": n_before,
        "n_predictors": len(fnames),
        "r_squared": round(r2, 4),
        "rmse": round(rmse, 4),
        "lambda": round(lasso.alpha_, 6),
        "n_nonzero_coefficients": len(nonzero),
        "mean_rent_per_sqft": round(float(y.mean()), 2),
        "variable_importance": importance_df[["variable", "beta", "partial_r2"]].round(4).to_dict(orient="records"),
        "acquisition_targets": targets.to_dict(orient="records"),
        "map_image_base64": map_b64,
    }


def _generate_map_b64(results, landmarks, lat_col, lon_col, name_col, title):
    """Generate map and return as base64 PNG string."""
    try:
        fig, ax = plt.subplots(figsize=(10, 8))
        norm = mcolors.TwoSlopeNorm(
            vmin=results["residual"].min(), vcenter=0, vmax=results["residual"].max()
        )
        sc = ax.scatter(
            results[lon_col], results[lat_col], c=results["residual"],
            cmap=plt.cm.RdYlGn, norm=norm, s=80, alpha=0.85,
            edgecolors="gray", linewidth=0.3, zorder=5,
        )
        for lm in landmarks:
            ax.plot(lm["lon"], lm["lat"], "^", color="black", markersize=8, zorder=6)
            ax.annotate(lm["name"], (lm["lon"], lm["lat"]),
                        textcoords="offset points", xytext=(6, 6),
                        fontsize=7, fontweight="bold", zorder=7)
        if name_col and name_col in results.columns:
            for _, row in results.nsmallest(3, "residual").iterrows():
                ax.annotate(row[name_col], (row[lon_col], row[lat_col]),
                            textcoords="offset points", xytext=(-8, -12),
                            fontsize=7, fontstyle="italic", color="darkred", zorder=7)
        plt.colorbar(sc, ax=ax, shrink=0.7, pad=0.02, label="Residual ($/SqFt)")
        ax.set_title(f"{title}\nRed = under-rented (targets) | Green = over-rented", fontsize=11)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_aspect("equal")

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")
    except Exception as e:
        return None


# ─── Tool definitions for Claude API ─────────────────────────

TOOL_DEFINITIONS = [
    {
        "name": "inspect_data",
        "description": "Get basic info about the uploaded dataset: row count, column count, column names, and a few sample rows. Call this first when the user uploads a file.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "list_submarkets",
        "description": "List all unique submarkets in the dataset with property counts. Helps the user pick which submarket to analyze.",
        "input_schema": {
            "type": "object",
            "properties": {
                "submarket_col": {
                    "type": "string",
                    "description": "Column name containing submarket labels. Default: 'Submarket'",
                },
            },
            "required": [],
        },
    },
    {
        "name": "describe_columns",
        "description": "Auto-classify dataset columns into binary (Yes/No), numeric, and categorical. Helps decide which features to include in the model.",
        "input_schema": {
            "type": "object",
            "properties": {
                "submarket_col": {"type": "string", "description": "Submarket column name"},
                "rent_prefix": {"type": "string", "description": "Prefix for quarterly rent columns. Default: 'RentYearQuarter'"},
            },
            "required": [],
        },
    },
    {
        "name": "run_analysis",
        "description": "Run the full LASSO regression analysis on a submarket. Identifies rent drivers and acquisition targets. Returns model stats, variable importance, top undervalued properties, and a map.",
        "input_schema": {
            "type": "object",
            "properties": {
                "submarket_filter": {
                    "type": "string",
                    "description": "Substring to match in the submarket column (e.g. 'Midtown West')",
                },
                "landmarks": {
                    "type": "array",
                    "description": "List of local landmarks for distance features",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "lat": {"type": "number"},
                            "lon": {"type": "number"},
                        },
                        "required": ["name", "lat", "lon"],
                    },
                },
                "binary_yn_cols": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Columns to treat as Yes/No binary features",
                },
                "numeric_cols": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Columns to treat as numeric features",
                },
                "categorical_cols": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Columns to one-hot encode as categorical features",
                },
                "submarket_col": {"type": "string"},
                "rent_prefix": {"type": "string"},
                "cv_folds": {"type": "integer", "description": "Cross-validation folds (default 10)"},
                "top_n": {"type": "integer", "description": "Number of acquisition targets to return (default 10)"},
            },
            "required": ["submarket_filter"],
        },
    },
]


def execute_tool(tool_name, tool_input, df):
    """Dispatch a tool call to the right function."""
    if tool_name == "inspect_data":
        return inspect_data(df)
    elif tool_name == "list_submarkets":
        return list_submarkets(df, **tool_input)
    elif tool_name == "describe_columns":
        return describe_columns(df, **tool_input)
    elif tool_name == "run_analysis":
        return run_analysis(df, **tool_input)
    else:
        return {"error": f"Unknown tool: {tool_name}"}
