"""
instructor_tool/train.py
========================
Trains a regression model to predict G3 (final grade) from the UCI
Student Performance dataset stored in the local `database/` folder.

Saves three artefacts in the same directory:
    model.pkl   – best fitted regression model
    scaler.pkl  – fitted StandardScaler
    columns.pkl – ordered list of feature columns after one-hot encoding

School grading system
----------------------
  G1  →  0-25   (first exam)
  G2  →  0-25   (second exam)
  G3  →  0-50   (final exam  ← what the model predicts)
  Total = G1 + G2 + G3  (max 100)
  PASS  → Total >= 60

The UCI dataset stores G1, G2, G3 on a 0-20 scale.
We rescale them in preprocessing so the model learns and
predicts entirely in the school's native scale:
    G1, G2  ×  1.25  →  0-25
    G3      ×  2.50  →  0-50

No runtime conversion is ever needed in the Streamlit apps.

Usage:
    python train.py
"""

import os
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

DATABASE_DIR    = "database"
FILE_MATH       = "student-mat.csv"
FILE_PORTUGUESE = "student-por.csv"

MODEL_PATH   = "model.pkl"
SCALER_PATH  = "scaler.pkl"
COLUMNS_PATH = "columns.pkl"

RAW_TARGET = "G3"   # column name in UCI dataset (raw 0-20)

# ── School grade scale factors ──────────────────────────────────────────────
# Applied once during preprocessing so everything downstream uses the
# school's native scale (0-25 / 0-50) with no further conversion needed.
G1G2_FACTOR    = 25 / 20   # 1.25  →  raw 0-20  becomes school 0-25
G3_FACTOR      = 50 / 20   # 2.50  →  raw 0-20  becomes school 0-50
PASS_THRESHOLD = 60         # G1 + G2 + G3 (school scale) >= 60 → PASS
G3_MAX         = 50         # clamp for predictions

# Columns to DROP: PII, sensitive lifestyle/personal attributes
DROP_COLUMNS = [
    "school",     # school identifier
    "address",    # location data
    "famsize",    # family structure
    "Pstatus",    # parent cohabitation
    "romantic",   # relationship status
    "Dalc",       # weekday alcohol
    "Walc",       # weekend alcohol
    "goout",      # social going-out frequency
    "nursery",    # attended nursery school
    "paid",       # extra paid tuition
]

# Numeric features to standardise (G1, G2 are already on school scale after
# rescaling and are NOT in this list — they stay as interpretable inputs)
NUMERIC_TO_SCALE = [
    "age", "Medu", "Fedu", "traveltime", "studytime",
    "failures", "famrel", "freetime", "health", "absences",
]


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    """
    Load and merge student-mat.csv + student-por.csv from database/.
    Tags each row with its subject, deduplicates students that appear
    in both files (keeping the Math record), and returns the combined df.
    """
    frames = []

    for fname, subject in [(FILE_MATH, "math"), (FILE_PORTUGUESE, "portuguese")]:
        path = os.path.join(DATABASE_DIR, fname)
        if os.path.exists(path):
            df = pd.read_csv(path, sep=";")
            df["subject"] = subject
            frames.append(df)
            print(f"  📂 Loaded {fname:<22} → {len(df):>4} records")
        else:
            print(f"  ⚠️  {path} not found — skipping.")

    if not frames:
        raise FileNotFoundError(
            f"No CSV files found in '{DATABASE_DIR}/'. "
            "Expected student-mat.csv and/or student-por.csv."
        )

    df = pd.concat(frames, ignore_index=True)

    # Deduplicate students enrolled in both courses; keep Math record first.
    demo_cols = [c for c in [
        "sex", "age", "Medu", "Fedu", "Mjob", "Fjob",
        "reason", "guardian", "traveltime", "schoolsup",
        "famsup", "activities", "higher", "internet",
    ] if c in df.columns]

    before = len(df)
    df = df.drop_duplicates(subset=demo_cols, keep="first").reset_index(drop=True)
    removed = before - len(df)
    if removed:
        print(f"  🔗 Removed {removed} duplicate student(s) across datasets.")

    print(f"  ✅ Total records: {len(df)}\n")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# GRADE RESCALING  ← applied as the very first preprocessing step
# ─────────────────────────────────────────────────────────────────────────────

def rescale_grades(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert UCI raw grades (0-20) to the school's native scale:
        G1, G2  →  0-25   (multiply by 1.25)
        G3      →  0-50   (multiply by 2.50)

    This is done once here so every downstream step — including the
    model itself — works entirely in the school's grading system.
    The Streamlit apps therefore receive and output values directly
    in the school scale with no further conversion required.
    """
    df = df.copy()
    for col in ["G1", "G2"]:
        if col in df.columns:
            df[col] = (df[col] * G1G2_FACTOR).round(2)
    if RAW_TARGET in df.columns:
        df[RAW_TARGET] = (df[RAW_TARGET] * G3_FACTOR).round(2)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# PREPROCESSING  (single source of truth — imported verbatim in student.py)
# ─────────────────────────────────────────────────────────────────────────────

def drop_sensitive(df: pd.DataFrame) -> pd.DataFrame:
    """Remove PII and sensitive columns."""
    return df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns])


def impute(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values: median for numeric, mode for categorical."""
    for col in df.select_dtypes(include="number").columns:
        df[col] = df[col].fillna(df[col].median())
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].fillna(df[col].mode()[0])
    return df


def encode(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode all remaining object/categorical columns."""
    return pd.get_dummies(df, drop_first=False)


def preprocess_training(df: pd.DataFrame):
    """
    Full training-time preprocessing pipeline.

    Steps
    -----
    1. Rescale G1, G2 → 0-25  |  G3 → 0-50  (school native scale)
    2. Extract target (G3 on 0-50 scale)
    3. Drop sensitive + subject-tag columns
    4. Impute missing values
    5. One-hot encode categoricals
    6. StandardScale selected numeric columns

    Returns
    -------
    X               : processed feature DataFrame
    y               : G3 Series on school scale (0-50)
    scaler          : fitted StandardScaler
    feature_columns : list of column names in X
    """
    df = df.copy()

    # Step 1 – rescale to school scale FIRST
    df = rescale_grades(df)

    # Step 2 – extract target (now 0-50)
    y = df.pop(RAW_TARGET)

    # Step 3 – drop sensitive + helper columns
    df = drop_sensitive(df)
    df = df.drop(columns=["subject"], errors="ignore")

    # Step 4 – impute
    df = impute(df)

    # Step 5 – one-hot encode
    df = encode(df)

    # Step 6 – standardise selected numeric columns
    scaler = StandardScaler()
    cols_to_scale = [c for c in NUMERIC_TO_SCALE if c in df.columns]
    df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])

    feature_columns = list(df.columns)
    return df, y, scaler, feature_columns


def preprocess_inference(df_raw: pd.DataFrame,
                         scaler: StandardScaler,
                         feature_columns: list) -> pd.DataFrame:
    """
    Inference-time preprocessing: identical logic to training.

    Expects G1, G2 already on the school scale (0-25) as entered by
    the user or read from the uploaded CSV — no further conversion needed.

    Parameters
    ----------
    df_raw          : input DataFrame with school-scale G1/G2 (0-25)
    scaler          : fitted StandardScaler from scaler.pkl
    feature_columns : column list from columns.pkl

    Returns
    -------
    Aligned, scaled DataFrame ready for model.predict()
    The model's output will be G3 on the school scale (0-50).
    """
    df = df_raw.copy()

    # Drop any sensitive, target, or helper columns present in the input
    drop = [c for c in DROP_COLUMNS + ["subject", RAW_TARGET, "pass"]
            if c in df.columns]
    df.drop(columns=drop, inplace=True)

    # Impute → encode → align → scale  (same order as training)
    df = impute(df)
    df = encode(df)
    df = df.reindex(columns=feature_columns, fill_value=0)

    cols_to_scale = [c for c in NUMERIC_TO_SCALE if c in df.columns]
    df[cols_to_scale] = scaler.transform(df[cols_to_scale])

    return df


# ─────────────────────────────────────────────────────────────────────────────
# VERDICT HELPER  (used by student.py)
# ─────────────────────────────────────────────────────────────────────────────

def compute_verdict(g1: float, g2: float, g3: float):
    """
    Compute total score and pass/fail verdict.

    All inputs must already be on the school scale:
        g1  →  0-25
        g2  →  0-25
        g3  →  0-50   (predicted by the model)

    Returns (total: float, passed: bool)
    """
    total  = round(g1 + g2 + g3, 1)
    passed = total >= PASS_THRESHOLD
    return total, passed


# ─────────────────────────────────────────────────────────────────────────────
# MODEL TRAINING & EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def train_and_evaluate(X_train, X_test, y_train, y_test) -> dict:
    """
    Train Linear Regression and Random Forest Regressor.
    Evaluate with RMSE and MAE (both in the school's G3 scale: 0-50).
    """
    candidates = {
        "Linear Regression":       LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=200, random_state=42
        ),
    }

    results = {}
    sep = "═" * 50

    for name, reg in candidates.items():
        reg.fit(X_train, y_train)
        preds = reg.predict(X_test)
        rmse  = float(np.sqrt(mean_squared_error(y_test, preds)))
        mae   = float(mean_absolute_error(y_test, preds))

        results[name] = {"model": reg, "rmse": rmse, "mae": mae}

        print(f"\n{sep}")
        print(f"  {name}")
        print(sep)
        print(f"  RMSE : {rmse:.4f}  (school scale, out of 50)")
        print(f"  MAE  : {mae:.4f}  (school scale, out of 50)")

    return results


def select_best(results: dict) -> tuple:
    """Return (name, result_dict) for the model with lowest RMSE."""
    best = min(results, key=lambda k: results[k]["rmse"])
    return best, results[best]


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    sep = "═" * 50
    print(f"\n{sep}")
    print("  📊  Instructor Tool — Model Training")
    print(f"{sep}\n")

    # 1. Load raw data
    df_raw = load_data()
    print(f"  Raw G3 range  : {df_raw[RAW_TARGET].min()} – {df_raw[RAW_TARGET].max()}  (UCI 0-20)")

    # 2. Preprocess (includes grade rescaling to school scale)
    X, y, scaler, feature_columns = preprocess_training(df_raw)
    print(f"  G3 after rescale : {y.min():.1f} – {y.max():.1f}  (school 0-50)")
    print(f"  G3 mean          : {y.mean():.2f}")
    print(f"  Features         : {X.shape[1]}")
    print(f"  First 6          : {list(X.columns[:6])}\n")

    # 3. Train / test split
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    print(f"  Train: {len(X_tr)}  |  Test: {len(X_te)}\n")

    # 4. Train & evaluate
    print("  Training models …")
    results = train_and_evaluate(X_tr, X_te, y_tr, y_te)

    # 5. Best model
    best_name, best = select_best(results)
    print(f"\n  🏆 Best model : {best_name}")
    print(f"      RMSE      : {best['rmse']:.4f}  (out of 50)")
    print(f"      MAE       : {best['mae']:.4f}  (out of 50)\n")

    # 6. Save artefacts
    joblib.dump(best["model"],   MODEL_PATH)
    joblib.dump(scaler,          SCALER_PATH)
    joblib.dump(feature_columns, COLUMNS_PATH)

    print(f"  💾 Saved → {MODEL_PATH}")
    print(f"     Saved → {SCALER_PATH}")
    print(f"     Saved → {COLUMNS_PATH}")
    print(f"\n  ✅ Done! Run:  streamlit run instructor.py\n")


if __name__ == "__main__":
    main()
