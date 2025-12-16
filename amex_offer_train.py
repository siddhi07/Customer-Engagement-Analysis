# amex_offer_train_fixed.py
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder
import joblib, json, os
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# --------------------------- CONFIG -------------------------------- #
CFG = {
    "PATHS": {
        "TRAIN": "data/train_data.csv",
        "TRANS": "data/add_trans.csv",
        "EVENT": "data/add_event.csv",
        "OFFER_META": "data/offer_metadata.csv",
        "DICT": "data/data_dictionary.csv",
        "MODEL_OUT": "amex_pipeline.pkl",
        "SUB_OUT": "r2_submission_file_teamname.csv"
    },
    "TARGET": "clicked",
    "ID_COLS": ["id1", "id2", "id3", "id5"],
    "VALID_SPLIT_DATE": "2023-11-03",
    "SEED": 42
}
np.random.seed(CFG["SEED"])


# ------------------------ UTILITY FUNCTIONS ------------------------ #
def safe_datetime_convert(series, format_str=None):
    """Safely convert to datetime and extract useful features"""
    try:
        if format_str:
            dt_series = pd.to_datetime(series, format=format_str, errors='coerce')
        else:
            dt_series = pd.to_datetime(series, errors='coerce')
        return dt_series
    except:
        return pd.to_datetime(series, errors='coerce')


def extract_time_features(dt_series, prefix="time"):
    """Extract useful time features from datetime column"""
    features = {}
    features[f"{prefix}_hour"] = dt_series.dt.hour
    features[f"{prefix}_day"] = dt_series.dt.day
    features[f"{prefix}_month"] = dt_series.dt.month
    features[f"{prefix}_dayofweek"] = dt_series.dt.dayofweek
    features[f"{prefix}_is_weekend"] = (dt_series.dt.dayofweek >= 5).astype(int)
    return pd.DataFrame(features)


def preprocess_data_types(df):
    """Clean and preprocess all data types for ML models"""
    df_clean = df.copy()

    # Store original columns for reference
    original_cols = df_clean.columns.tolist()

    # Handle datetime columns
    datetime_cols = []
    for col in df_clean.columns:
        if df_clean[col].dtype == 'datetime64[ns]':
            datetime_cols.append(col)
            # Extract time features
            time_features = extract_time_features(df_clean[col], prefix=f"{col}")
            df_clean = pd.concat([df_clean, time_features], axis=1)
            # Drop original datetime column
            df_clean = df_clean.drop(columns=[col])

    # Handle object columns that might be numeric
    label_encoders = {}
    for col in df_clean.columns:
        if df_clean[col].dtype == 'object':
            # Try to convert to numeric first
            numeric_converted = pd.to_numeric(df_clean[col], errors='coerce')
            if not numeric_converted.isna().all():
                # If mostly numeric, use numeric version
                if numeric_converted.notna().sum() / len(df_clean) > 0.5:
                    df_clean[col] = numeric_converted
                    continue

            # Otherwise, use label encoding
            le = LabelEncoder()
            # Handle NaN values
            mask = df_clean[col].notna()
            if mask.sum() > 0:  # If there are non-null values
                df_clean.loc[mask, col] = le.fit_transform(df_clean.loc[mask, col].astype(str))
                df_clean[col] = df_clean[col].fillna(-1)
                label_encoders[col] = le
            else:
                df_clean[col] = -1

    # Ensure all columns are numeric
    for col in df_clean.columns:
        if df_clean[col].dtype == 'object':
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(-1)

    # Fill any remaining NaNs
    df_clean = df_clean.fillna(0)

    return df_clean, label_encoders, datetime_cols


# ------------------------ DATA LOAD -------------------------------- #
print("Loading data files...")
train = pd.read_csv(CFG["PATHS"]["TRAIN"])
trans = pd.read_csv(CFG["PATHS"]["TRANS"])
event = pd.read_csv(CFG["PATHS"]["EVENT"])
meta = pd.read_csv(CFG["PATHS"]["OFFER_META"])

print(f"Train shape: {train.shape}")
print(f"Problematic columns: {[col for col in train.columns if train[col].dtype in ['object', 'datetime64[ns]']]}")

# ------------------------ OFFER META PREP -------------------------- #
# Handle date columns with specific format
date_columns = ['f374', 'id13', 'id12']
for col in date_columns:
    if col in meta.columns:
        # Try different date formats based on your sample data
        meta[col] = safe_datetime_convert(meta[col])

today = pd.Timestamp("2023-11-05").normalize()

# Active flag
if 'id12' in meta.columns and 'id13' in meta.columns:
    meta["is_currently_active"] = (
            (meta["id12"].isna() | (meta["id12"] <= today)) &
            (meta["id13"].isna() | (meta["id13"] >= today))
    ).astype("int8")
else:
    meta["is_currently_active"] = 1


# Discount bucketing
def bucket_val(val):
    if pd.isna(val): return "unk"
    if val == 0: return "zero"
    if val < 5: return "<5"
    if val < 20: return "5-20"
    if val < 50: return "20-50"
    return "50+"


if 'f376' in meta.columns:
    meta["discount_bucket"] = meta["f376"].apply(bucket_val)
else:
    meta["discount_bucket"] = "unk"

# Keep useful meta columns
KEEP_META = ["id3", "is_currently_active", "discount_bucket"]
for col in ["id9", "f375", "f376", "id10", "id11"]:
    if col in meta.columns:
        KEEP_META.append(col)

meta = meta[KEEP_META]

# ------------------------ MAIN MERGE ------------------------------- #
print("Merging offer metadata...")
train = train.merge(meta, on="id3", how="left")

# ------------------------ TRANSACTION FEATURES -------------------- #
print("Creating transaction features...")
if 'id2' in trans.columns and 'f367' in trans.columns:
    # Convert f367 to numeric if it's not
    trans['f367'] = pd.to_numeric(trans['f367'], errors='coerce').fillna(0)

    agg = (trans.groupby("id2")
           .agg(trans_cnt=("f367", "size"),
                trans_amt_sum=("f367", "sum"),
                trans_amt_mean=("f367", "mean"))
           ).reset_index()
    train = train.merge(agg, on="id2", how="left")

    train.fillna({
        "trans_cnt": 0,
        "trans_amt_sum": 0,
        "trans_amt_mean": 0,
    }, inplace=True)
else:
    print("Warning: Transaction features skipped")
    train["trans_cnt"] = 0
    train["trans_amt_sum"] = 0
    train["trans_amt_mean"] = 0

# ------------------------ TARGET VARIABLE CHECK ------------------- #
if 'clicked' in train.columns:
    target_col = 'clicked'
elif 'y' in train.columns:
    target_col = 'y'
    CFG["TARGET"] = 'y'
else:
    print("Error: No target variable found!")
    exit(1)

print(f"Target variable: {target_col}")
print(f"Target distribution:\n{train[target_col].value_counts()}")

# ------------------------ DATA TYPE PREPROCESSING ------------------- #
print("Preprocessing data types...")

# Separate target and IDs before preprocessing
target = train[target_col].copy()
id_cols_present = [col for col in CFG["ID_COLS"] if col in train.columns]
id_data = train[id_cols_present].copy()

# Remove target and ID columns for preprocessing
feature_data = train.drop(columns=[target_col] + id_cols_present)

# Preprocess all features
feature_data_clean, label_encoders, datetime_cols = preprocess_data_types(feature_data)

print(f"Original features: {len(feature_data.columns)}")
print(f"Processed features: {len(feature_data_clean.columns)}")
print(f"Datetime columns processed: {datetime_cols}")

# Combine back
train_processed = pd.concat([id_data, feature_data_clean, target], axis=1)

# ------------------------ TRAIN / VALID SPLIT ---------------------- #
print("Creating train/validation split...")

if 'id5' in train_processed.columns:
    train_processed['id5'] = safe_datetime_convert(train_processed['id5'])
    train_processed = train_processed.sort_values('id5')
    split_date = pd.to_datetime(CFG["VALID_SPLIT_DATE"])

    train_set = train_processed[train_processed['id5'] < split_date]
    valid_set = train_processed[train_processed['id5'] >= split_date]
else:
    train_set, valid_set = train_test_split(train_processed, test_size=0.2, random_state=CFG["SEED"])

print(f"Train set size: {len(train_set)}")
print(f"Valid set size: {len(valid_set)}")

# Prepare final feature set
feature_cols = [col for col in train_processed.columns
                if col not in [target_col] + id_cols_present]

X_train = train_set[feature_cols]
y_train = train_set[target_col]
X_valid = valid_set[feature_cols]
y_valid = valid_set[target_col]

print(f"Using {len(feature_cols)} features")
print(f"Train target rate: {y_train.mean():.4f}")
print(f"Valid target rate: {y_valid.mean():.4f}")

# Final data type check
print(f"X_train dtypes: {X_train.dtypes.value_counts()}")

# ------------------------ MODEL 1: LightGBM ------------------------ #
print("Training LightGBM...")
lgb_params = dict(
    objective="binary",
    metric="binary_logloss",
    learning_rate=0.05,
    num_leaves=64,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=4,
    seed=CFG["SEED"],
    verbose=-1,
    force_row_wise=True  # Handle potential threading issues
)

lgb_train = lgb.Dataset(X_train, y_train)
lgb_valid = lgb.Dataset(X_valid, y_valid, reference=lgb_train)

lgb_model = lgb.train(
    lgb_params,
    lgb_train,
    num_boost_round=1000,
    valid_sets=[lgb_train, lgb_valid],
    callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)]
)

# ------------------------ MODEL 2: XGBoost ------------------------- #
print("Training XGBoost...")
# ------------------------ MODEL 2: XGBoost ------------------------- #
print("Training XGBoost...")

# Check XGBoost version and adjust accordingly
import xgboost as xgb

try:
    # For newer XGBoost versions (>= 1.6.0)
    xgb_model = xgb.XGBClassifier(
        objective='binary:logistic',
        learning_rate=0.05,
        max_depth=6,
        n_estimators=1000,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=CFG["SEED"],
        eval_metric='logloss',
        early_stopping_rounds=100
    )

    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=100
    )

except TypeError:
    # For older XGBoost versions (< 1.6.0)
    xgb_model = xgb.XGBClassifier(
        objective='binary:logistic',
        learning_rate=0.05,
        max_depth=6,
        n_estimators=1000,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=CFG["SEED"],
        eval_metric='logloss'
    )

    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        early_stopping_rounds=100,
        verbose=100
    )

# ------------------------ PREDICTIONS & EVALUATION ----------------- #
print("Generating predictions...")
lgb_pred = lgb_model.predict(X_valid, num_iteration=lgb_model.best_iteration)
xgb_pred = xgb_model.predict_proba(X_valid)[:, 1]

# Ensemble prediction
valid_pred = 0.7 * lgb_pred + 0.3 * xgb_pred
logloss = log_loss(y_valid, valid_pred)

print(f"\nValidation Results:")
print(f"LightGBM LogLoss: {log_loss(y_valid, lgb_pred):.5f}")
print(f"XGBoost LogLoss: {log_loss(y_valid, xgb_pred):.5f}")
print(f"Ensemble LogLoss: {logloss:.5f}")

# ------------------------ SAVE PIPELINE ---------------------------- #
PIPE = {
    "lgb_model": lgb_model,
    "xgb_model": xgb_model,
    "feature_cols": feature_cols,
    "label_encoders": label_encoders,
    "datetime_cols": datetime_cols,
    "target_col": target_col,
    "config": CFG,
    "meta_columns": KEEP_META
}

joblib.dump(PIPE, CFG["PATHS"]["MODEL_OUT"])
print(f"\nPipeline saved ➜ {CFG['PATHS']['MODEL_OUT']}")
print("Training completed successfully!")

# ------------------------ FEATURE IMPORTANCE ----------------------- #
print("\nTop 20 Feature Importances (LightGBM):")
feature_imp = pd.DataFrame({
    'feature': feature_cols,
    'importance': lgb_model.feature_importance(importance_type='gain')
})
feature_imp = feature_imp.sort_values('importance', ascending=False).head(20)
print(feature_imp.to_string(index=False))
