# streamlit_app_complete.py
"""
Complete Streamlit app for the Predictive Analytics Hackathon.
- Loads dataset from /mnt/data/Synthetic_Farming_Dataset_With_Seasonality_And_Challenge.csv
- Creates statistically-derived synthetic inputs (fertilizer_kg_per_ha, irrigation_mm, pesticide_l)
- Preprocesses data (impute, winsorize, seasonal features)
- Fits a regression pipeline (imputer -> scaler -> Ridge)
- Provides SHAP/PDP explainability (if shap installed)
- Optimizes input multipliers with a constrained grid-search per row under budget & env constraints
- Produces downloadable CSV with recommendations
"""

import math
import itertools
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# Optional SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

# ---------------------------------------------------------
# Config / constants
# ---------------------------------------------------------
CSV_PATH = "F:\Predictive Analytics\Haackathon\Synthetic_Farming_Dataset_With_Seasonality_And_Challenge.csv"
TARGET_COL = "yield_kg_per_ha"
ENV_COL = "environmental_score"
COST_COL = "input_cost_total"

# realistic ranges for synthetic inputs (domain-informed)
FERT_RANGE = (40, 250)      # kg/ha
IRRI_RANGE = (30, 300)      # mm per season (example)
PEST_RANGE = (0.2, 4.0)     # liters/ha

# multiplier grid for optimizer (pragmatic)
DEFAULT_MULTIPLIERS = [0.6, 0.8, 1.0, 1.2, 1.4]

# ---------------------------------------------------------
# Utilities: load with encoding fallback
# ---------------------------------------------------------
def load_data(path: str = CSV_PATH) -> pd.DataFrame:
    for enc in ("utf-8", "cp1252", "latin1"):
        try:
            df = pd.read_csv(path, encoding=enc, parse_dates=["date"])
            st.sidebar.write(f"Loaded CSV with encoding: {enc}")
            return df
        except Exception:
            continue
    # final fallback
    df = pd.read_csv(path, encoding="latin1")
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    st.sidebar.write("Loaded CSV with fallback latin1")
    return df

# ---------------------------------------------------------
# Statistical synthetic input generation (Option 2 style)
# ---------------------------------------------------------
def detect_columns_by_keywords(cols: List[str], keywords: List[str]) -> List[str]:
    found = []
    for c in cols:
        low = c.lower()
        for k in keywords:
            if k in low:
                found.append(c)
                break
    return found

def compute_soil_score(df: pd.DataFrame) -> pd.Series:
    """
    Compute a 'soil deficit' proxy from columns likely representing soil nutrients.
    Heuristics: look for 'nitro', 'n_', 'phos', 'p_', 'potass', 'k_', 'organic', 'soil'
    Falls back to mean of numeric columns that contain 'soil' or 'nutr' or none->zeros.
    """
    cols = df.columns.tolist()
    keywords = ["nitro", "n_", "nitrogen", "phosph", "phos", "potass", "k_", "organic", "soil", "nutr"]
    soil_cols = detect_columns_by_keywords(cols, keywords)
    if len(soil_cols) == 0:
        # fallback: columns containing 'ph' or 'pH' or 'clay' or 'silt' or 'sand'
        soil_cols = detect_columns_by_keywords(cols, ["ph", "clay", "silt", "sand"])
    if len(soil_cols) == 0:
        # fallback to numeric columns that contain 'soil' in name or just numeric columns that are plausible
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(num_cols) == 0:
            return pd.Series(0.0, index=df.index)
        # use subset of numeric columns as proxy (but de-emphasize yield & cost & env)
        exclude = {TARGET_COL, COST_COL, ENV_COL}
        candidate = [c for c in num_cols if c not in exclude]
        # if still many, take first 3
        soil_cols = candidate[:3] if len(candidate) >= 1 else candidate

    soil_score = df[soil_cols].mean(axis=1, skipna=True)
    # invert soil_score into 'deficit' by subtracting relative to median
    soil_deficit = soil_score.median() - soil_score
    # normalize to 0..1
    sd = soil_deficit.fillna(0)
    if sd.max() == sd.min():
        return pd.Series(0.0, index=df.index)
    return (sd - sd.min()) / (sd.max() - sd.min())

def compute_weather_stress(df: pd.DataFrame) -> pd.Series:
    """
    Compute a weather 'stress' proxy (higher => need more irrigation).
    Heuristics: high temperature and low precipitation => stress.
    We search for temp/rain/humidity columns by keywords.
    """
    cols = df.columns.tolist()
    temp_cols = detect_columns_by_keywords(cols, ["temp", "temperature", "max_temp", "tmean"])
    rain_cols = detect_columns_by_keywords(cols, ["precip", "rain", "rainfall", "mm", "rain_mm"])
    humidity_cols = detect_columns_by_keywords(cols, ["humid", "rh", "relative_humidity"])
    temp = df[temp_cols].mean(axis=1, skipna=True) if len(temp_cols) else pd.Series(np.nan, index=df.index)
    rain = df[rain_cols].sum(axis=1, skipna=True) if len(rain_cols) else pd.Series(np.nan, index=df.index)
    # z-score normalize (robust)
    temp_z = (temp - np.nanmedian(temp)) / (np.nanstd(temp) + 1e-9) if len(temp.dropna())>0 else pd.Series(0.0, index=df.index)
    rain_z = (rain - np.nanmedian(rain)) / (np.nanstd(rain) + 1e-9) if len(rain.dropna())>0 else pd.Series(0.0, index=df.index)
    # stress: high temp (positive) and low rain (negative) -> combine
    # scale to 0..1
    combined = temp_z - rain_z
    # replace NaN with 0
    combined = combined.fillna(0.0)
    if combined.max() == combined.min():
        return pd.Series(0.0, index=df.index)
    return (combined - combined.min()) / (combined.max() - combined.min())

def compute_crop_sensitivity(df: pd.DataFrame) -> pd.Series:
    """
    Estimate per-crop pesticide sensitivity using correlation between env_score and yield loss.
    Higher sensitivity -> more pesticide needed to protect yield when env_score indicates pest pressure.
    If crop_type missing, return zeros.
    """
    if "crop_type" not in df.columns or ENV_COL not in df.columns or TARGET_COL not in df.columns:
        # fallback: use env_norm directly
        if ENV_COL in df.columns:
            env_norm = (df[ENV_COL] - df[ENV_COL].min()) / (df[ENV_COL].max() - df[ENV_COL].min() + 1e-9)
            return env_norm.fillna(0.0)
        return pd.Series(0.0, index=df.index)

    # compute within-crop correlation between env and yield (if env high -> yield low => positive sensitivity)
    df_tmp = df[[ "crop_type", ENV_COL, TARGET_COL]].copy()
    crop_sens = {}
    for crop, sub in df_tmp.groupby("crop_type"):
        if sub[ENV_COL].nunique() < 2 or sub[TARGET_COL].nunique() < 2:
            crop_sens[crop] = 0.0
            continue
        try:
            corr = np.corrcoef(sub[ENV_COL].fillna(sub[ENV_COL].median()), sub[TARGET_COL].fillna(sub[TARGET_COL].median()))[0,1]
        except Exception:
            corr = 0.0
        # if env negatively correlates with yield (corr < 0), pests/stress lower yield -> higher sensitivity
        sens = max(0.0, -corr)  # positive value in [0,1] range roughly
        crop_sens[crop] = sens
    # map sensitivities to series
    sens_series = df["crop_type"].map(crop_sens).fillna(0.0)
    # normalize
    if sens_series.max() == sens_series.min():
        return sens_series * 0.0
    return (sens_series - sens_series.min()) / (sens_series.max() - sens_series.min())

def create_synthetic_inputs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create fertilizer_kg_per_ha, irrigation_mm, pesticide_l from dataset using statistical heuristics.
    """
    df = df.copy()

    # Normalize cost and env
    if COST_COL in df.columns:
        cost = df[COST_COL].astype(float)
        cost_norm = (cost - cost.min()) / (cost.max() - cost.min() + 1e-9)
    else:
        # create synthetic cost norm from other numeric columns
        numeric = df.select_dtypes(include=[np.number]).drop(columns=[TARGET_COL] if TARGET_COL in df.columns else [])
        if numeric.shape[1] == 0:
            cost_norm = pd.Series(0.5, index=df.index)
        else:
            cost_norm = (numeric.mean(axis=1) - numeric.mean(axis=1).min()) / (numeric.mean(axis=1).max() - numeric.mean(axis=1).min() + 1e-9)

    if ENV_COL in df.columns:
        env = df[ENV_COL].astype(float)
        env_norm = (env - env.min()) / (env.max() - env.min() + 1e-9)
    else:
        env_norm = pd.Series(0.5, index=df.index)

    soil_deficit = compute_soil_score(df)            # 0..1, higher -> more fertilizer
    weather_stress = compute_weather_stress(df)      # 0..1, higher -> more irrigation
    crop_sens = compute_crop_sensitivity(df)         # 0..1, higher -> more pesticide

    # Base splits of cost into inputs (weights)
    # We'll allocate cost to fertilizer & irrigation primarily; pesticide linked to env_norm and crop_sens
    # base fractions (sum <=1)
    base_fert_frac = 0.55
    base_irri_frac = 0.35
    base_pest_frac = 0.10

    # adjust fractions per row using soil_deficit & weather_stress & crop_sens
    fert_frac = base_fert_frac + 0.25 * soil_deficit - 0.1 * weather_stress
    irri_frac = base_irri_frac + 0.35 * weather_stress - 0.15 * soil_deficit
    pest_frac = base_pest_frac + 0.4 * crop_sens + 0.2 * env_norm

    # normalize fractions to sum to 1 (avoid zero)
    fracs = np.vstack([fert_frac.fillna(0).values, irri_frac.fillna(0).values, pest_frac.fillna(0).values]).T
    fracs_sum = fracs.sum(axis=1, keepdims=True)
    fracs_sum[fracs_sum == 0] = 1.0
    fracs = fracs / fracs_sum

    fert_frac_n = fracs[:,0]
    irri_frac_n = fracs[:,1]
    pest_frac_n = fracs[:,2]

    # Map cost_norm to physical units using realistic scaling
    # We map cost_norm to a "budget scale" where 1.0 corresponds to a higher-end input mix.
    # Use linear mapping: fertilizer range maps to FERT_RANGE scaled by cost_norm and soil_deficit
    fert = FERT_RANGE[0] + (FERT_RANGE[1] - FERT_RANGE[0]) * (0.4*cost_norm + 0.6*soil_deficit * cost_norm)
    irri = IRRI_RANGE[0] + (IRRI_RANGE[1] - IRRI_RANGE[0]) * (0.4*cost_norm + 0.6*weather_stress * cost_norm)
    pest = PEST_RANGE[0] + (PEST_RANGE[1] - PEST_RANGE[0]) * (0.5*env_norm + 0.5*crop_sens * env_norm)

    # Final synthetic inputs adjusted by fractions (to respect cost allocation)
    # Convert fraction-weighted adjustment: scale fert by fert_frac_n normalized
    fert_final = fert * (fert_frac_n / (fert_frac_n + 1e-9))
    irri_final = irri * (irri_frac_n / (irri_frac_n + 1e-9))
    pest_final = pest * (pest_frac_n / (pest_frac_n + 1e-9))

    # For safety, clip to ranges
    fert_final = np.clip(fert_final, FERT_RANGE[0], FERT_RANGE[1])
    irri_final = np.clip(irri_final, IRRI_RANGE[0], IRRI_RANGE[1])
    pest_final = np.clip(pest_final, PEST_RANGE[0], PEST_RANGE[1])

    # Attach to df
    df["fertilizer_kg_per_ha"] = fert_final
    df["irrigation_mm"] = irri_final
    df["pesticide_l"] = pest_final

    # Also compute synthetic cost_estimated from these inputs (simple linear cost model)
    # Assume unit costs (example): fertilizer ₹15/kg, irrigation ₹10 per mm (per ha aggregated), pesticide ₹500 per L
    fert_unit_cost = 15.0
    irri_unit_cost = 10.0
    pest_unit_cost = 500.0
    df["synthetic_input_cost"] = df["fertilizer_kg_per_ha"] * fert_unit_cost + df["irrigation_mm"] * irri_unit_cost + df["pesticide_l"] * pest_unit_cost

    return df

# ---------------------------------------------------------
# Preprocessing: imputation, winsorization, seasonal features
# ---------------------------------------------------------
def winsorize(df: pd.DataFrame, q_low=0.01, q_high=0.99) -> pd.DataFrame:
    df = df.copy()
    num_cols = df.select_dtypes(include=[np.number]).columns
    for c in num_cols:
        low = df[c].quantile(q_low)
        high = df[c].quantile(q_high)
        df[c] = df[c].clip(lower=low, upper=high)
    return df

def add_seasonal_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "date" in df.columns:
        df["month"] = df["date"].dt.month
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)
    else:
        df["month"] = np.nan
        df["month_sin"] = 0.0
        df["month_cos"] = 0.0
    return df

# ---------------------------------------------------------
# Modeling: pipeline and fit
# ---------------------------------------------------------
def build_pipeline_and_fit(df: pd.DataFrame, target_col: str = TARGET_COL):
    # Separate features and target
    df = df.copy()
    # Drop rows lacking target
    df = df[~df[target_col].isna()].reset_index(drop=True)
    # Identify numeric and categorical features
    numeric_feats = df.select_dtypes(include=[np.number]).columns.tolist()
    # remove index-like columns (we'll keep synthetic_input_cost etc.)
    if target_col in numeric_feats:
        numeric_feats.remove(target_col)
    # We'll include fertilizer/irrigation/pesticide as numeric features
    cat_feats = df.select_dtypes(include=["object", "category"]).columns.tolist()

    # Preprocessing pipelines
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocessor = ColumnTransformer([
        ("num", num_pipeline, numeric_feats),
        ("cat", cat_pipeline, cat_feats),
    ], remainder="drop")

    model = Ridge(alpha=1.0)

    pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("model", model),
    ])

    X = df.drop(columns=[target_col])
    y = df[target_col].values

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline.fit(X_train, y_train)

    # predict & metrics
    y_pred = pipeline.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = math.sqrt(mean_squared_error(y_test, y_pred))

    # to compute feature importance in original feature names we need the pipeline internals
    # We'll extract numeric feature names and categorical OHE names
    feature_names = []
    # numeric names
    feature_names += numeric_feats
    # categorical OHE names
    try:
        ohe = pipeline.named_steps["preprocess"].named_transformers_["cat"].named_steps["ohe"]
        cat_ohe_names = []
        for i, col in enumerate(cat_feats):
            cats = ohe.categories_[i]
            cat_ohe_names.extend([f"{col}__{v}" for v in cats])
        feature_names += cat_ohe_names
    except Exception:
        # fallback: no categorical
        pass

    # coefficients from ridge
    try:
        coefs = pipeline.named_steps["model"].coef_
        coef_series = pd.Series(coefs, index=feature_names)
    except Exception:
        coef_series = pd.Series([], dtype=float)

    return {
        "pipeline": pipeline,
        "r2": r2,
        "rmse": rmse,
        "feature_names": feature_names,
        "coef_series": coef_series,
        "X_test": X_test,
        "y_test": y_test,
        "X": X,
        "df_model": df,
    }

# ---------------------------------------------------------
# Optimizer: grid search on multipliers for synthetic input columns
# ---------------------------------------------------------
def detect_input_cols(df: pd.DataFrame) -> List[str]:
    candidates = []
    for c in df.columns:
        low = c.lower()
        if any(key in low for key in ["fertil", "irrig", "pestic", "input_"]):
            if pd.api.types.is_numeric_dtype(df[c]):
                candidates.append(c)
    # Force include our created columns if present
    for c in ["fertilizer_kg_per_ha", "irrigation_mm", "pesticide_l"]:
        if c in df.columns and c not in candidates:
            candidates.append(c)
    return candidates

def optimize_row_grid(row: pd.Series, pipeline, feature_cols: List[str], input_cols: List[str],
                      multipliers=DEFAULT_MULTIPLIERS, budget_limit=12000.0, env_threshold=10000.0):
    """
    For a single row, try combinations of multipliers applied to its input_cols to maximize predicted yield.
    Returns best dict.
    """
    best = {"predicted_yield": -np.inf, "multipliers": None, "feasible": False}
    # baseline row as DataFrame single row
    base = row.copy()
    # iterate combos
    for combo in itertools.product(multipliers, repeat=len(input_cols)):
        tmp = base.copy()
        for i, col in enumerate(input_cols):
            tmp[col] = tmp[col] * combo[i]
        # if budget column exists we can recompute synthetic cost or check existing cost
        raw_cost_val = tmp.get(COST_COL, np.nan)
        raw_env_val = tmp.get(ENV_COL, np.nan)
        # Build feature vector as DataFrame (same columns order as training X)
        X_tmp = tmp[feature_cols].to_frame().T
        try:
            pred = float(pipeline.predict(X_tmp)[0])
        except Exception:
            # skip combos that break
            continue
        feasible = True
        # Safely coerce cost/env to numeric to avoid isfinite() and float() type errors
        cost_num = pd.to_numeric(pd.Series([raw_cost_val]), errors="coerce").iloc[0]
        env_num = pd.to_numeric(pd.Series([raw_env_val]), errors="coerce").iloc[0]
        if not pd.isna(cost_num) and np.isfinite(cost_num):
            feasible = feasible and (cost_num <= budget_limit)
        if not pd.isna(env_num) and np.isfinite(env_num):
            feasible = feasible and (env_num <= env_threshold)
        # update best if feasible & better prediction
        if feasible and pred > best["predicted_yield"]:
            best.update({
                "predicted_yield": pred,
                "multipliers": {input_cols[i]: combo[i] for i in range(len(input_cols))},
                "feasible": True,
                "cost_val": float(cost_num) if not pd.isna(cost_num) and np.isfinite(cost_num) else None,
                "env_val": float(env_num) if not pd.isna(env_num) and np.isfinite(env_num) else None,
            })
        # keep best even if not feasible if nothing feasible exists
        if (not best["feasible"]) and pred > best["predicted_yield"]:
            best.update({
                "predicted_yield": pred,
                "multipliers": {input_cols[i]: combo[i] for i in range(len(input_cols))},
                "feasible": False,
            })
    return best


def optimize_global_multipliers(df: pd.DataFrame,
                                pipeline,
                                feature_cols: List[str],
                                input_cols: List[str],
                                multipliers=DEFAULT_MULTIPLIERS,
                                budget_limit: float = 12000.0,
                                env_threshold: float = 10000.0):
    """Grid-search a *single* set of multipliers applied to all rows.

    Objective:
      - Maximize mean predicted yield across the dataset
      - While keeping mean cost <= budget_limit (when available)
      - And mean environmental_score <= env_threshold (when available)

    Returns (best_summary_dict, baseline_mean_yield).
    """
    df = df.copy()
    feature_cols = list(feature_cols)

    # Baseline predicted yield (no change to inputs)
    try:
        baseline_preds = pipeline.predict(df[feature_cols])
        baseline_mean = float(np.mean(baseline_preds))
    except Exception:
        baseline_mean = None

    # Focus multipliers on up to three key input columns
    key_order = ["fertilizer_kg_per_ha", "irrigation_mm", "pesticide_l"]
    cols_for_mult = [c for c in key_order if c in input_cols]
    if not cols_for_mult:
        cols_for_mult = input_cols[:3]

    best = {
        "predicted_mean_yield": -np.inf,
        "multipliers": None,
        "feasible": False,
        "mean_cost": None,
        "mean_env": None,
        "baseline_mean_yield": baseline_mean,
    }

    for combo in itertools.product(multipliers, repeat=len(cols_for_mult)):
        tmp = df.copy()
        # apply multipliers
        for col, m in zip(cols_for_mult, combo):
            tmp[col] = tmp[col] * m

        # cost approximation
        if COST_COL in tmp.columns:
            cost_series = pd.to_numeric(tmp[COST_COL], errors="coerce")
        elif "synthetic_input_cost" in tmp.columns:
            cost_series = pd.to_numeric(tmp["synthetic_input_cost"], errors="coerce")
        else:
            cost_series = pd.Series(np.nan, index=tmp.index)
        mean_cost = float(cost_series.mean()) if cost_series.notna().any() else None

        # env approximation
        if ENV_COL in tmp.columns:
            env_series = pd.to_numeric(tmp[ENV_COL], errors="coerce")
        else:
            env_series = pd.Series(np.nan, index=tmp.index)
        mean_env = float(env_series.mean()) if env_series.notna().any() else None

        # predictions
        try:
            preds = pipeline.predict(tmp[feature_cols])
            mean_yield = float(np.mean(preds))
        except Exception:
            continue

        feasible = True
        if mean_cost is not None:
            feasible = feasible and (mean_cost <= budget_limit)
        if mean_env is not None:
            feasible = feasible and (mean_env <= env_threshold)

        update = False
        if feasible and mean_yield > best["predicted_mean_yield"]:
            update = True
        elif (not best["feasible"]) and mean_yield > best["predicted_mean_yield"]:
            # If we don't yet have any feasible solution, keep track of the best overall
            update = True

        if update:
            best.update({
                "predicted_mean_yield": mean_yield,
                "multipliers": {cols_for_mult[i]: combo[i] for i in range(len(cols_for_mult))},
                "feasible": feasible,
                "mean_cost": mean_cost,
                "mean_env": mean_env,
            })

    return best, baseline_mean


def recommend_inputs_rule_based(df: pd.DataFrame,
                                model_info: Dict[str, Any],
                                budget_limit: float = 12000.0,
                                env_threshold: float = 10000.0,
                                multipliers = None) -> pd.DataFrame:
    """Rule-based, interpretable recommendations for fertilizer, irrigation, and pesticide.

    For each record, we:
      - Start from synthetic inputs (fertilizer_kg_per_ha, irrigation_mm, pesticide_l)
      - Try a grid of multipliers around the current levels
      - Recompute input cost using simple unit prices
      - Respect per-hectare budget and a soft environmental threshold
      - Choose the combination with highest predicted yield

    Returns a DataFrame with baseline vs recommended inputs and yields.
    """
    if multipliers is None or len(multipliers) == 0:
        multipliers = [0.8, 1.0, 1.2]

    # Keep multiplier grid small for speed: choose up to 3 values, centered around 1.0
    try:
        vals = sorted({float(m) for m in multipliers})
    except Exception:
        vals = [0.8, 1.0, 1.2]
    if 1.0 not in vals:
        vals.append(1.0)
        vals = sorted(vals)
    if len(vals) > 3:
        vals = sorted(vals, key=lambda v: abs(v - 1.0))[:3]
    multipliers = vals

    df = df.copy()
    pipeline = model_info["pipeline"]
    feature_cols = model_info["X"].columns.tolist()

    # Unit costs (same as used to build synthetic_input_cost)
    fert_unit_cost = 15.0
    irri_unit_cost = 10.0
    pest_unit_cost = 500.0

    results = []
    for idx, row in df.iterrows():
        # Baseline inputs
        fert_base = row.get("fertilizer_kg_per_ha", np.nan)
        irri_base = row.get("irrigation_mm", np.nan)
        pest_base = row.get("pesticide_l", np.nan)

        # Skip if we don't have valid baseline inputs
        if any(pd.isna([fert_base, irri_base, pest_base])):
            continue

        # Baseline cost and prediction
        cost_base = float(fert_base * fert_unit_cost + irri_base * irri_unit_cost + pest_base * pest_unit_cost)
        env_val = row.get(ENV_COL, np.nan)

        try:
            baseline_pred = float(pipeline.predict(row[feature_cols].to_frame().T)[0])
        except Exception:
            baseline_pred = None

        best_yield = -np.inf
        best_combo = None
        best_cost = None

        for fm in multipliers:
            for im in multipliers:
                for pm in multipliers:
                    # If environment already high, do not allow increasing fertilizer/pesticide
                    if not pd.isna(env_val) and env_val >= env_threshold:
                        if fm > 1.0 or pm > 1.0:
                            continue

                    fert_new = fert_base * fm
                    irri_new = irri_base * im
                    pest_new = pest_base * pm

                    cost_new = float(
                        fert_new * fert_unit_cost
                        + irri_new * irri_unit_cost
                        + pest_new * pest_unit_cost
                    )
                    if cost_new > budget_limit:
                        continue

                    tmp = row.copy()
                    tmp["fertilizer_kg_per_ha"] = fert_new
                    tmp["irrigation_mm"] = irri_new
                    tmp["pesticide_l"] = pest_new

                    try:
                        pred = float(pipeline.predict(tmp[feature_cols].to_frame().T)[0])
                    except Exception:
                        continue

                    if pred > best_yield:
                        best_yield = pred
                        best_combo = (fm, im, pm, fert_new, irri_new, pest_new)
                        best_cost = cost_new

        # If no better feasible combo found, fall back to baseline
        if best_combo is None or baseline_pred is not None and baseline_pred >= best_yield:
            fert_rec, irri_rec, pest_rec = fert_base, irri_base, pest_base
            rec_yield = baseline_pred
            cost_rec = cost_base
            fm, im, pm = 1.0, 1.0, 1.0
        else:
            fm, im, pm, fert_rec, irri_rec, pest_rec = best_combo
            rec_yield = best_yield
            cost_rec = best_cost

        improvement = None
        if baseline_pred is not None and rec_yield is not None:
            improvement = rec_yield - baseline_pred

        results.append({
            "index": int(idx),
            "crop_type": row.get("crop_type"),
            "baseline_fertilizer_kg_per_ha": fert_base,
            "baseline_irrigation_mm": irri_base,
            "baseline_pesticide_l": pest_base,
            "baseline_cost_estimated": cost_base,
            "baseline_predicted_yield": baseline_pred,
            "recommended_fertilizer_kg_per_ha": fert_rec,
            "recommended_irrigation_mm": irri_rec,
            "recommended_pesticide_l": pest_rec,
            "recommended_cost_estimated": cost_rec,
            "optimized_predicted_yield": rec_yield,
            "yield_improvement": improvement,
            "mult_fertilizer": fm,
            "mult_irrigation": im,
            "mult_pesticide": pm,
            "env_value": env_val,
            "cost_within_budget": cost_rec <= budget_limit if cost_rec is not None else None,
            "env_below_threshold": env_val <= env_threshold if not pd.isna(env_val) else None,
        })

    if not results:
        return pd.DataFrame()

    return pd.DataFrame(results)

# ---------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------
def main():
    st.set_page_config(page_title="Predictive Analytics Hackathon — Complete App", layout="wide", page_icon="🌾")

    # Global CSS to improve visual appearance
    st.markdown(
        """
        <style>
        /* App background */
        .stApp {
            background: radial-gradient(circle at top left, #e0f2fe 0, #f9fafb 45%, #f1f5f9 100%);
        }

        /* Tabs styling */
        div[data-baseweb="tab-list"] {
            gap: 0.25rem;
        }
        div[data-baseweb="tab"] {
            padding: 0.4rem 0.9rem;
            border-radius: 0.75rem 0.75rem 0 0;
            background-color: #e5e7eb;
            font-weight: 500;
            border: 1px solid #d4d4d8;
            border-bottom: none;
        }
        div[data-baseweb="tab"][aria-selected="true"] {
            background: linear-gradient(90deg, #14532d, #16a34a);
            color: #ffffff;
        }

        /* Generic card container */
        .section-card {
            padding: 1.25rem 1.5rem;
            border-radius: 0.9rem;
            background-color: #ffffff;
            box-shadow: 0 18px 30px rgba(15,23,42,0.08);
            margin-bottom: 1.25rem;
            border-left: 4px solid #16a34a;
        }

        /* Hero block at top */
        .hero-block {
            padding: 1.75rem 1.75rem 1.5rem 1.75rem;
            border-radius: 1rem;
            background: linear-gradient(120deg, #052e16, #166534, #65a30d);
            color: #f9fafb;
            margin-bottom: 1.5rem;
            box-shadow: 0 22px 40px rgba(15,23,42,0.35);
        }
        .hero-title {
            font-size: 1.7rem;
            font-weight: 700;
            margin-bottom: 0.4rem;
        }
        .hero-subtitle {
            font-size: 0.95rem;
            opacity: 0.9;
            margin-bottom: 0.75rem;
        }
        .hero-tags {
            display: flex;
            flex-wrap: wrap;
            gap: 0.4rem;
        }
        .pill-badge {
            font-size: 0.8rem;
            padding: 0.2rem 0.7rem;
            border-radius: 999px;
            background-color: rgba(15,23,42,0.35);
            border: 1px solid rgba(248,250,252,0.25);
        }

        /* Small stat cards under hero */
        .stat-card {
            padding: 0.8rem 0.9rem;
            border-radius: 0.75rem;
            background-color: #ffffff;
            box-shadow: 0 10px 18px rgba(15,23,42,0.06);
            border: 1px solid #e5e7eb;
            font-size: 0.85rem;
        }
        .stat-card h4 {
            font-size: 0.9rem;
            margin: 0 0 0.25rem 0;
        }
        .stat-card p {
            margin: 0;
            color: #4b5563;
        }

        /* Optional: slightly rounded dataframes */
        .dataframe-container {
            border-radius: 0.75rem;
            overflow: hidden;
            border: 1px solid #e5e7eb;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Initialize session state for model and optimization results
    if "model_info" not in st.session_state:
        st.session_state["model_info"] = None
    if "coef_df" not in st.session_state:
        st.session_state["coef_df"] = None
    if "opt_results" not in st.session_state:
        st.session_state["opt_results"] = None

    # Hero section
    st.markdown(
        """<div class="hero-block">
        <div class="hero-title">Farm Input Optimization Console</div>
        <div class="hero-subtitle">
            Data-driven recommendations for fertilizer, irrigation, and pesticide that balance
            yield, input budget, and environmental impact.
        </div>
        <div class="hero-tags">
            <span class="pill-badge">ML model: Ridge regression</span>
            <span class="pill-badge">Budget ≤ ₹12,000 / ha</span>
            <span class="pill-badge">Environmental_score &lt; 10,000 (soft)</span>
            <span class="pill-badge">Rule-based optimization</span>
        </div>
        </div>""",
        unsafe_allow_html=True,
    )

    # Quick summary stat cards
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown(
            """<div class="stat-card">
            <h4>Step 1 – Explore data</h4>
            <p>Preview your farming dataset, synthetic inputs and data quality.</p>
            </div>""",
            unsafe_allow_html=True,
        )
    with col_b:
        st.markdown(
            """<div class="stat-card">
            <h4>Step 2 – Train model</h4>
            <p>Fit a regression model to estimate yield from features and inputs.</p>
            </div>""",
            unsafe_allow_html=True,
        )
    with col_c:
        st.markdown(
            """<div class="stat-card">
            <h4>Step 3 – Optimize inputs</h4>
            <p>Generate rule-based recommendations and download them as CSV.</p>
            </div>""",
            unsafe_allow_html=True,
        )

    # Sidebar configuration
    st.sidebar.header("Configuration")
    multipliers = st.sidebar.multiselect("Multiplier grid to try (per input)", DEFAULT_MULTIPLIERS, default=DEFAULT_MULTIPLIERS)
    use_winsor = st.sidebar.checkbox("Apply winsorization to numeric cols (1%-99%)", value=True)
    apply_season = st.sidebar.checkbox("Add seasonal features (month sin/cos)", value=True)
    use_shap = st.sidebar.checkbox("Attempt SHAP explanations (requires shap installed)", value=SHAP_AVAILABLE)
    env_soft = st.sidebar.number_input("Max environmental_score (soft)", min_value=0.0, value=10000.0, step=500.0)
    budget_limit = st.sidebar.number_input("Max input_cost_total (₹/ha)", min_value=0.0, value=12000.0, step=500.0)
    run_train = st.sidebar.button("Run full pipeline & train model")

    # Load dataset
    try:
        df_raw = load_data(CSV_PATH)
    except Exception as e:
        st.error(f"Failed to load CSV: {e}")
        return

    # Dataset preview card
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Dataset preview")
    st.write("Path used:", CSV_PATH)
    st.dataframe(df_raw.head(8))
    with st.expander("Show column names"):
        st.write(list(df_raw.columns))
    st.markdown('</div>', unsafe_allow_html=True)

    # Synthetic inputs card
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Creating synthetic inputs (statistical decomposition)")
    df = create_synthetic_inputs(df_raw)
    if apply_season:
        df = add_seasonal_features(df)
    if use_winsor:
        df = winsorize(df)

    st.write("Synthetic input columns added: `fertilizer_kg_per_ha`, `irrigation_mm`, `pesticide_l`, `synthetic_input_cost`")
    st.dataframe(df[["fertilizer_kg_per_ha", "irrigation_mm", "pesticide_l", "synthetic_input_cost"]].head(10))
    st.markdown('</div>', unsafe_allow_html=True)

    # Data quality card
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Data quality snapshot")
    mv = df.isnull().sum()
    mv_pct = (df.isnull().mean() * 100).round(2)
    mv_report = pd.DataFrame({"missing_count": mv, "missing_pct": mv_pct})
    st.dataframe(mv_report.sort_values("missing_pct", ascending=False).head(20))
    st.markdown('</div>', unsafe_allow_html=True)

    # Train model when user clicks (store in session state)
    if run_train:
        with st.spinner("Training model..."):
            model_info_local = build_pipeline_and_fit(df, target_col=TARGET_COL)
        if "error" in model_info_local:
            st.error("Model training failed.")
            return
        st.session_state["model_info"] = model_info_local
        # Reset downstream cached artifacts when model changes
        st.session_state["coef_df"] = None
        st.session_state["opt_results"] = None
        st.success("Model trained.")

    # Pull latest model objects from session
    model_info = st.session_state["model_info"]
    coef_df = st.session_state["coef_df"]

    # Layout for results: tabs for model, explainability, and optimization
    tab_model, tab_explain, tab_opt = st.tabs([
        "Model Performance",
        "Explainability & Effects",
        "Optimization & Recommendations",
    ])

    with tab_model:
        if model_info is None:
            st.info("Click 'Run full pipeline & train model' in the sidebar to see model performance.")
        else:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.subheader("Model performance")
            st.write(f"R² (test): {model_info['r2']:.3f} — RMSE: {model_info['rmse']:.3f}")
            st.markdown("Top coefficients (Ridge):")
            try:
                coef_df = model_info["coef_series"].abs().sort_values(ascending=False).to_frame("abs_coef")
                coef_df = coef_df.join(model_info["coef_series"].to_frame("coef"))
                st.session_state["coef_df"] = coef_df
                st.dataframe(coef_df.head(20))
                # Bar chart of top absolute coefficients
                coef_top = coef_df.head(20).reset_index().rename(columns={"index": "feature"})
                coef_chart = alt.Chart(coef_top).mark_bar().encode(
                    x=alt.X("abs_coef:Q", title="|Coefficient|"),
                    y=alt.Y("feature:N", sort="-x", title="Feature"),
                    color=alt.value("#4c78a8"),
                ).properties(height=300)
                st.altair_chart(coef_chart, use_container_width=True)
            except Exception:
                st.write("Could not extract coefficients (categorical OHE may change shape).")
            st.markdown('</div>', unsafe_allow_html=True)

    with tab_explain:
        if model_info is None:
            st.info("Train the model to see SHAP explanations and partial dependence plots.")
        else:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            # SHAP
            if use_shap and SHAP_AVAILABLE:
                st.subheader("SHAP explanations (sample)")
                try:
                    pipeline = model_info["pipeline"]
                    # build background from training X (use model_info['X'])
                    X = model_info["X"]
                    # use small background and pass through preprocessing so SHAP sees only numeric data
                    background_df = X.sample(n=min(100, len(X)), random_state=1)
                    preprocessor = pipeline.named_steps["preprocess"]
                    background = preprocessor.transform(background_df)
                    # create explainer on the underlying model using preprocessed numeric features
                    model = pipeline.named_steps["model"]
                    explainer = shap.Explainer(model.predict, background)
                    # SHAP values for some rows (also preprocessed)
                    sample_df = X.sample(n=min(50, len(X)), random_state=2)
                    sample = preprocessor.transform(sample_df)
                    shap_values = explainer(sample)
                    st.write("SHAP summary (sample):")
                    try:
                        shap.plots.beeswarm(shap_values, show=False)
                        st.pyplot(bbox_inches="tight")
                    except Exception:
                        st.write("SHAP plot rendering is not available in this environment.")
                except Exception as e:
                    st.write("SHAP failed:", e)
            elif use_shap and not SHAP_AVAILABLE:
                st.info("SHAP not installed in environment. Install with `pip install shap` to enable.")

            # PDP-like for top numeric features (approx)
            st.subheader("Partial Dependence (approx) for top numeric features")
            try:
                feature_names = model_info["feature_names"]
                # choose top numeric features: those in coef_df present in numeric columns
                numeric_candidates = df.select_dtypes(include=[np.number]).columns.tolist()
                if coef_df is not None:
                    ranked_feats = [f for f in coef_df.index if f in numeric_candidates]
                    pdp_features = ranked_feats[:3]
                else:
                    pdp_features = [f for f in feature_names if f in numeric_candidates][:3]
                for feat in pdp_features:
                    lo, hi = df[feat].quantile(0.05), df[feat].quantile(0.95)
                    vals = np.linspace(lo, hi, 25)
                    preds = []
                    for v in vals:
                        # create copy and set feat to v across all rows then predict mean
                        tmp = df.copy()
                        tmp[feat] = v
                        Xtmp = tmp.drop(columns=[TARGET_COL])
                        preds.append(float(model_info["pipeline"].predict(Xtmp).mean()))
                    pdp_df = pd.DataFrame({"feature_value": vals, "predicted_mean": preds})
                    chart = alt.Chart(pdp_df).mark_line().encode(
                        x=alt.X("feature_value", title=f"{feat} value"),
                        y=alt.Y("predicted_mean", title="Predicted yield (mean)"),
                    ).properties(title=f"PDP-like for {feat}", height=200)
                    st.altair_chart(chart, use_container_width=True)
            except Exception:
                st.write("PDP computation skipped due to internal error.")
            st.markdown('</div>', unsafe_allow_html=True)

    with tab_opt:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Recommend optimal inputs under budget & environmental constraints")
        st.markdown(
            "- **Goal:** Use the trained model to recommend fertilizer, irrigation, and pesticide levels.\n"
            f"- **Budget constraint:** Estimated input cost ≤ ₹{budget_limit:,.0f} per hectare.\n"
            f"- **Environmental constraint (soft):** environmental_score ≤ {env_soft:,.0f} (do not increase inputs when already high)."
        )
        if model_info is None:
            st.info("Train the model to enable optimization.")
        else:
            st.markdown("This rule-based optimizer searches multipliers around current inputs and keeps only combinations that respect the budget and environmental rules.")

            run_opt = st.button("Generate rule-based recommendations & download CSV")
            if run_opt:
                with st.spinner("Computing recommendations across all records..."):
                    rec_df = recommend_inputs_rule_based(
                        df,
                        model_info,
                        budget_limit=budget_limit,
                        env_threshold=env_soft,
                        multipliers=multipliers or DEFAULT_MULTIPLIERS,
                    )
                if rec_df.empty:
                    st.warning("No recommendations could be generated. Check that synthetic input columns are present and non-empty.")
                else:
                    st.session_state["opt_results"] = rec_df
                    st.success("Recommendations generated.")

            opt_results = st.session_state.get("opt_results")
            if opt_results is None or opt_results.empty:
                st.info("Click the button above to generate recommendations and enable CSV download.")
            else:
                res_df = opt_results
                # Summary metrics
                valid = res_df.dropna(subset=["baseline_predicted_yield", "optimized_predicted_yield"])
                if not valid.empty:
                    avg_base = valid["baseline_predicted_yield"].mean()
                    avg_opt = valid["optimized_predicted_yield"].mean()
                    avg_imp = valid["yield_improvement"].mean()
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Avg baseline yield", f"{avg_base:.2f} kg/ha")
                    with col2:
                        st.metric("Avg recommended yield", f"{avg_opt:.2f} kg/ha")
                    with col3:
                        st.metric("Avg improvement", f"{avg_imp:.2f} kg/ha")

                # Improvement chart
                try:
                    top_imp = (
                        res_df.dropna(subset=["yield_improvement"])
                        .sort_values("yield_improvement", ascending=False)
                        .head(20)
                        .copy()
                    )
                    top_imp["index_str"] = top_imp["index"].astype(str)
                    imp_chart = (
                        alt.Chart(top_imp)
                        .mark_bar()
                        .encode(
                            x=alt.X("index_str:N", title="Row index (farm / record)"),
                            y=alt.Y("yield_improvement:Q", title="Predicted yield improvement (kg/ha)"),
                            color="cost_within_budget:N",
                            tooltip=[
                                "index_str",
                                "crop_type",
                                "baseline_predicted_yield",
                                "optimized_predicted_yield",
                                "baseline_cost_estimated",
                                "recommended_cost_estimated",
                                "env_value",
                            ],
                        )
                        .properties(height=300)
                    )
                    st.altair_chart(imp_chart, use_container_width=True)
                except Exception:
                    pass

                st.subheader("Sample of recommended input plan (top 50 rows)")
                st.dataframe(res_df.head(50))
                csv_bytes = res_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download recommendations CSV",
                    data=csv_bytes,
                    file_name="recommended_inputs_rule_based.csv",
                    mime="text/csv",
                )
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("Notes: \n- Synthetic inputs were derived using statistical heuristics tied to soil/weather/crop features and normalized to realistic ranges. \n- Optimization uses a pragmatic grid-search on multipliers for these inputs and enforces budget & environmental thresholds. \n- For a formal constrained optimization (LP/nonlinear), we can replace the grid search with PuLP/SciPy-based solver if you want higher fidelity modeling.")

if __name__ == "__main__":
    main()
