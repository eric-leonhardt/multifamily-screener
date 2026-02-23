#!/usr/bin/env python3
"""
run_analysis.py — Interactive LASSO rent model pipeline.

Usage:
    python run_analysis.py                  # Interactive mode (prompts for everything)
    python run_analysis.py --config my.yaml # Config mode (skip prompts, use saved config)

On first run, the CLI walks you through selecting a CSV, submarket, features,
and landmarks. You can optionally save your choices as a YAML config for reuse.
"""

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler

from utils import (
    safe_numeric,
    add_landmark_distances,
    encode_binary_yn,
    encode_categorical_dummies,
    encode_custom_interactive,
    impute_columns,
    get_latest_rent,
    drop_zero_variance,
    classify_columns,
)

warnings.filterwarnings("ignore")


# ─── Interactive helpers ─────────────────────────────────────

def prompt(msg, default=None):
    suffix = f" [{default}]" if default else ""
    val = input(f"  {msg}{suffix}: ").strip()
    return val if val else default


def prompt_yn(msg, default="y"):
    val = prompt(msg, default).lower()
    return val in ("y", "yes", "1", "true")


def prompt_int(msg, default):
    val = prompt(msg, str(default))
    try:
        return int(val)
    except ValueError:
        return default


def prompt_float(msg, default):
    val = prompt(msg, str(default))
    try:
        return float(val)
    except ValueError:
        return default


def pick_from_list(items, msg="Select items", multi=True):
    """Display numbered list, return selected items."""
    for i, item in enumerate(items, 1):
        print(f"    {i:3d}. {item}")

    if multi:
        sel = prompt(f"{msg} (comma-separated numbers, 'all', or 'none')", "all")
        if sel.lower() == "all":
            return list(items)
        if sel.lower() == "none":
            return []
        try:
            indices = [int(x.strip()) - 1 for x in sel.split(",")]
            return [items[i] for i in indices if 0 <= i < len(items)]
        except (ValueError, IndexError):
            print("    Invalid selection, using all.")
            return list(items)
    else:
        sel = prompt(f"{msg} (enter number)")
        try:
            return items[int(sel.strip()) - 1]
        except (ValueError, IndexError):
            print("    Invalid, using first option.")
            return items[0]


# ─── Interactive setup ───────────────────────────────────────

def interactive_setup():
    """Walk user through CSV → submarket → features → landmarks."""
    print("\n" + "=" * 60)
    print("  RENT MODEL — Interactive Setup")
    print("=" * 60)

    # ── CSV path ─────────────────────────────────────────────
    print("\n── Step 1: Data ──")
    while True:
        csv_path = prompt("Path to your CSV file")
        if csv_path and os.path.isfile(csv_path):
            break
        print(f"    File not found: {csv_path}")

    df = pd.read_csv(csv_path, low_memory=False)
    print(f"    Loaded {len(df)} rows, {len(df.columns)} columns")

    # ── Submarket ────────────────────────────────────────────
    print("\n── Step 2: Submarket ──")
    sub_candidates = [c for c in df.columns if "submarket" in c.lower() or "market" in c.lower()]
    if sub_candidates:
        print("    Detected possible submarket columns:")
        sub_col = pick_from_list(sub_candidates, "Which column?", multi=False)
    else:
        sub_col = prompt("Column name containing submarket labels")

    unique_subs = sorted(df[sub_col].dropna().unique())
    print(f"\n    {len(unique_subs)} submarkets found:")
    selected_sub = pick_from_list(unique_subs, "Which submarket?", multi=False)

    df_sub = df[df[sub_col].str.contains(selected_sub, case=False, na=False)].copy()
    print(f"    → {len(df_sub)} properties in '{selected_sub}'")

    # ── Key columns ──────────────────────────────────────────
    print("\n── Step 3: Column Mapping ──")
    print("    Press Enter to accept defaults.\n")

    guesses = {
        "property_id": ["PropertyID", "property_id", "ID"],
        "property_name": ["propertyname", "PropertyName", "property_name", "Name"],
        "address": ["Property.Address", "Address", "address"],
        "latitude": ["Latitude", "latitude", "lat"],
        "longitude": ["Longitude", "longitude", "lon"],
        "total_sqft": ["TotalSqFt", "total_sqft", "SqFt"],
        "num_units": ["Property.NoUnits", "Units", "NoUnits"],
        "parking": ["TotalParkingSpaces", "Parking", "ParkingSpaces"],
    }
    col_map = {}
    for key, candidates in guesses.items():
        found = next((c for c in candidates if c in df.columns), None)
        col_map[key] = prompt(key, found)

    rent_prefix = prompt("Rent column prefix", "RentYearQuarter")

    # ── Features ─────────────────────────────────────────────
    print("\n── Step 4: Features ──")
    print("    Auto-classifying columns...\n")

    classified = classify_columns(df_sub, col_map, rent_prefix)

    yn_cols = []
    if classified["binary_yn"]:
        print("  Binary (Yes/No) columns:")
        yn_cols = pick_from_list(classified["binary_yn"], "Include?")

    num_cols = []
    if classified["numeric"]:
        print("\n  Numeric columns:")
        num_cols = pick_from_list(classified["numeric"], "Include?")

    cat_cols = []
    if classified["categorical"]:
        print("\n  Categorical columns:")
        cat_cols = pick_from_list(classified["categorical"], "Include?")

    # ── Landmarks ────────────────────────────────────────────
    print("\n── Step 5: Landmarks ──")
    print("    Enter local landmarks for distance features.")
    print("    Tip: right-click in Google Maps to copy lat/lon.")
    print("    Type 'done' when finished.\n")

    landmarks = []
    while True:
        name = prompt("Landmark name (or 'done')")
        if not name or name.lower() == "done":
            break
        lat = prompt_float(f"  {name} — latitude", 0)
        lon = prompt_float(f"  {name} — longitude", 0)
        if lat != 0 and lon != 0:
            landmarks.append({"name": name, "lat": lat, "lon": lon})
            print(f"    ✓ {name}")

    print(f"    {len(landmarks)} landmarks added")

    # ── Model settings ───────────────────────────────────────
    print("\n── Step 6: Model Settings ──")
    cv_folds = prompt_int("CV folds", 10)
    top_n = prompt_int("Acquisition targets to show", 15)

    # ── Map FIPS ─────────────────────────────────────────────
    print("\n── Step 7: Map Background (optional) ──")
    print("    Census block outlines require state + county FIPS codes.")
    state_fips = prompt("State FIPS (or Enter to skip)", "")
    county_fips = prompt("County FIPS (or Enter to skip)", "")

    # ── Assemble config ──────────────────────────────────────
    cfg = {
        "submarket": {
            "name": selected_sub,
            "filter_pattern": selected_sub,
            "state_fips": state_fips,
            "county_fips": county_fips,
        },
        "data": {
            "path": csv_path,
            "submarket_col": sub_col,
            "property_id_col": col_map["property_id"],
            "property_name_col": col_map["property_name"],
            "address_col": col_map["address"],
            "lat_col": col_map["latitude"],
            "lon_col": col_map["longitude"],
            "sqft_col": col_map["total_sqft"],
            "units_col": col_map["num_units"],
            "parking_col": col_map["parking"],
            "rent_col_prefix": rent_prefix,
        },
        "landmarks": landmarks,
        "features": {
            "numeric": num_cols,
            "binary_yn": yn_cols,
            "numeric_amenities": [],
            "categorical": cat_cols,
            "custom": {},
        },
        "model": {
            "alpha": 1.0,
            "cv_folds": cv_folds,
            "lambda_rule": "1se",
            "standardize": True,
            "random_seed": 42,
        },
        "imputation": {"numeric": "mean", "categorical": "mode"},
        "output": {
            "dir": "output",
            "targets_top_n": top_n,
            "map_filename": "submarket_map.png",
            "results_filename": "results.csv",
            "importance_filename": "variable_importance.csv",
            "targets_filename": "acquisition_targets.csv",
        },
    }

    # ── Save config ──────────────────────────────────────────
    print()
    if prompt_yn("Save config as YAML for reuse?", "y"):
        safe_name = selected_sub.replace(" ", "_").replace("/", "_").lower()
        save_path = prompt("Filename", f"config_{safe_name}.yaml")
        with open(save_path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
        print(f"    ✓ Saved to {save_path}")
        print(f"    Rerun anytime: python run_analysis.py --config {save_path}")

    return cfg


# ─── Pipeline ────────────────────────────────────────────────

def run(cfg):
    print(f"\n{'='*60}")
    print(f"  Rent Model: {cfg['submarket']['name']}")
    print(f"{'='*60}\n")

    data_cfg = cfg["data"]
    feat_cfg = cfg["features"]
    model_cfg = cfg["model"]
    imp_cfg = cfg["imputation"]
    out_cfg = cfg["output"]
    os.makedirs(out_cfg["dir"], exist_ok=True)

    # 1. Load & filter
    print("[1/8] Loading data...")
    df_raw = pd.read_csv(data_cfg["path"], low_memory=False)
    pattern = cfg["submarket"]["filter_pattern"]
    df = df_raw[
        df_raw[data_cfg["submarket_col"]].str.contains(pattern, case=False, na=False)
    ].copy()
    print(f"      {len(df)} rows in submarket")

    # 2. Parse
    print("[2/8] Parsing columns...")
    for col in [data_cfg["sqft_col"], data_cfg["units_col"],
                data_cfg["parking_col"], data_cfg["lat_col"], data_cfg["lon_col"]]:
        if col in df.columns:
            df[col] = safe_numeric(df[col])
    for col in feat_cfg.get("numeric", []):
        if col in df.columns:
            df[col] = safe_numeric(df[col])
    df, _ = get_latest_rent(df, prefix=data_cfg["rent_col_prefix"])

    # 3. Clean
    print("[3/8] Cleaning...")
    sqft, units = data_cfg["sqft_col"], data_cfg["units_col"]
    n0 = len(df)
    df = df[(df["LatestRent"] > 0) & (df[sqft] > 0) & (df[units] > 0)].copy()
    print(f"      {len(df)} of {n0} kept")
    if len(df) < 10:
        print("      ⚠ Very few observations — interpret with caution")

    # 4. Features
    print("[4/8] Engineering features...")
    df["AvgSqFt"] = df[sqft] / df[units]
    df["RentPerSqFt"] = df["LatestRent"] / df["AvgSqFt"]
    df["ParkingPerUnit"] = df[data_cfg["parking_col"]] / df[units]

    df, dist_cols = add_landmark_distances(
        df, cfg.get("landmarks", []),
        lat_col=data_cfg["lat_col"], lon_col=data_cfg["lon_col"]
    )
    extra_num = []
    if "Dist_to_city_center" in df.columns:
        df["Dist_to_city_center"] = safe_numeric(df["Dist_to_city_center"])
        extra_num.append("Dist_to_city_center")

    df, yn_cols = encode_binary_yn(df, feat_cfg.get("binary_yn", []))
    num_amenity_cols = []
    for col in feat_cfg.get("numeric_amenities", []):
        if col in df.columns:
            df[col] = safe_numeric(df[col])
            num_amenity_cols.append(col)
    df, custom_cols = encode_custom_interactive(df, feat_cfg.get("custom", {}))
    df, cat_cols = encode_categorical_dummies(df, feat_cfg.get("categorical", []))

    pred_cols = (
        feat_cfg.get("numeric", []) + ["AvgSqFt", "ParkingPerUnit"]
        + extra_num + dist_cols + yn_cols + num_amenity_cols + custom_cols + cat_cols
    )
    pred_cols = [c for c in pred_cols if c in df.columns]
    print(f"      {len(pred_cols)} predictors")

    # 5. Impute
    print("[5/8] Imputing...")
    df = impute_columns(df, pred_cols, numeric_method=imp_cfg.get("numeric", "mean"))

    # 6. LASSO
    print("[6/8] Fitting LASSO...")
    y = df["RentPerSqFt"].values
    X_raw = df[pred_cols].values.astype(float)
    fnames = list(pred_cols)
    X_raw, fnames = drop_zero_variance(X_raw, fnames)

    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)
    n_folds = min(model_cfg.get("cv_folds", 10), max(3, len(y) // 5))

    lasso = LassoCV(alphas=None, cv=n_folds,
                    random_state=model_cfg.get("random_seed", 42), max_iter=20000)
    lasso.fit(X, y)
    yhat = lasso.predict(X)
    resid = y - yhat

    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - np.sum(resid ** 2) / ss_tot
    rmse = np.sqrt(np.mean(resid ** 2))
    print(f"      R² = {r2:.4f} | RMSE = ${rmse:.4f}/sqft | λ = {lasso.alpha_:.6f}")

    coef_df = pd.DataFrame({"Variable": fnames, "Beta": lasso.coef_})
    nonzero = coef_df[coef_df["Beta"] != 0].sort_values("Beta", key=abs, ascending=False)
    print(f"      {len(nonzero)} non-zero coefficients")

    # 7. Importance
    print("[7/8] Variable importance (partial R²)...")
    rows = []
    for var in nonzero["Variable"]:
        idx = fnames.index(var)
        Xd = np.delete(X, idx, axis=1)
        ld = LassoCV(alphas=None, cv=n_folds,
                     random_state=model_cfg.get("random_seed", 42), max_iter=20000)
        ld.fit(Xd, y)
        r2d = 1 - np.sum((y - ld.predict(Xd)) ** 2) / ss_tot
        rows.append({"Variable": var, "Partial_R2": r2 - r2d})

    imp_df = nonzero.merge(pd.DataFrame(rows), on="Variable")
    psum = imp_df.loc[imp_df["Partial_R2"] > 0, "Partial_R2"].sum()
    imp_df["Pct"] = (imp_df["Partial_R2"] / psum * 100).round(1) if psum > 0 else 0
    imp_df = imp_df.sort_values("Partial_R2", ascending=False)
    print(imp_df.to_string(index=False))

    # 8. Output
    print("\n[8/8] Saving results...")
    name_col = data_cfg["property_name_col"]
    addr_col = data_cfg["address_col"]
    keep = [c for c in [data_cfg["property_id_col"], name_col, addr_col,
                         "RentPerSqFt", "AvgSqFt", "CompletedYear",
                         data_cfg["lat_col"], data_cfg["lon_col"]] if c in df.columns]
    results = df[keep].copy()
    results["Predicted"] = yhat
    results["Residual"] = resid
    results["Pct_Below_Market"] = (resid / yhat * 100).round(1)

    top_n = out_cfg.get("targets_top_n", 15)
    targets = results.nsmallest(top_n, "Residual")

    print(f"\n{'='*60}")
    print(f"  TOP {top_n} ACQUISITION TARGETS")
    print(f"{'='*60}")
    show = [c for c in [name_col, addr_col, "RentPerSqFt", "Predicted",
                         "Residual", "Pct_Below_Market"] if c in results.columns]
    print(targets[show].to_string(index=False))

    results.to_csv(os.path.join(out_cfg["dir"], out_cfg["results_filename"]), index=False)
    imp_df.to_csv(os.path.join(out_cfg["dir"], out_cfg["importance_filename"]), index=False)
    targets.to_csv(os.path.join(out_cfg["dir"], out_cfg["targets_filename"]), index=False)
    print(f"\n      CSVs saved to {out_cfg['dir']}/")

    generate_map(cfg, results, out_cfg)

    print(f"\n{'='*60}")
    print(f"  Done. R² = {r2:.4f} | RMSE = ${rmse:.4f}/sqft")
    print(f"{'='*60}")
    return results, imp_df


# ─── Map ─────────────────────────────────────────────────────

def generate_map(cfg, results, out_cfg):
    data_cfg = cfg["data"]
    lat_col, lon_col = data_cfg["lat_col"], data_cfg["lon_col"]
    name_col = data_cfg["property_name_col"]

    if lat_col not in results.columns or lon_col not in results.columns:
        print("      Skipping map (no coordinates)")
        return

    fig, ax = plt.subplots(figsize=(10, 8))

    # Census block background (optional)
    sf, cf = cfg["submarket"].get("state_fips", ""), cfg["submarket"].get("county_fips", "")
    if sf and cf:
        try:
            import geopandas as gpd
            from shapely.geometry import box
            url = f"https://www2.census.gov/geo/tiger/TIGER2020/TABBLOCK20/tl_2020_{sf}{cf}_tabblock20.zip"
            blocks = gpd.read_file(url)
            pad = 0.01
            bbox = box(results[lon_col].min() - pad, results[lat_col].min() - pad,
                        results[lon_col].max() + pad, results[lat_col].max() + pad)
            gpd.clip(blocks, bbox).plot(ax=ax, color="whitesmoke", edgecolor="lightgray", linewidth=0.3)
            print("      Census block background loaded")
        except Exception as e:
            print(f"      No map background ({e})")

    norm = mcolors.TwoSlopeNorm(vmin=results["Residual"].min(), vcenter=0,
                                 vmax=results["Residual"].max())
    sc = ax.scatter(results[lon_col], results[lat_col], c=results["Residual"],
                    cmap=plt.cm.RdYlGn, norm=norm, s=80, alpha=0.85,
                    edgecolors="gray", linewidth=0.3, zorder=5)

    for lm in cfg.get("landmarks", []):
        ax.plot(lm["lon"], lm["lat"], "^", color="black", markersize=8, zorder=6)
        ax.annotate(lm["name"], (lm["lon"], lm["lat"]),
                    textcoords="offset points", xytext=(6, 6),
                    fontsize=7, fontweight="bold", zorder=7)

    if name_col in results.columns:
        for _, row in results.nsmallest(3, "Residual").iterrows():
            ax.annotate(row[name_col], (row[lon_col], row[lat_col]),
                        textcoords="offset points", xytext=(-8, -12),
                        fontsize=7, fontstyle="italic", color="darkred", zorder=7)

    plt.colorbar(sc, ax=ax, shrink=0.7, pad=0.02, label="Residual ($/SqFt)")
    ax.set_title(f"{cfg['submarket']['name']}\nRed = under-rented | Green = over-rented", fontsize=11)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal")

    path = os.path.join(out_cfg["dir"], out_cfg["map_filename"])
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"      Map saved to {path}")


# ─── Entry point ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Interactive LASSO rent model")
    parser.add_argument("--config", help="Saved YAML config (skips interactive setup)")
    args = parser.parse_args()

    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        print(f"Loaded config: {args.config}")
    else:
        cfg = interactive_setup()

    run(cfg)


if __name__ == "__main__":
    main()
