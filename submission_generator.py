# standalone_submission_generator.py
import pandas as pd
import numpy as np
import joblib
import sys
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings('ignore')


def bucket_val(val):
    """Discount bucketing function - standalone version"""
    if pd.isna(val): return "unk"
    if val == 0: return "zero"
    if val < 5: return "<5"
    if val < 20: return "5-20"
    if val < 50: return "20-50"
    return "50+"


def safe_datetime_convert(series):
    """Safe datetime conversion - standalone version"""
    return pd.to_datetime(series, errors='coerce')


def safe_label_encode(series, label_encoder=None):
    """Safe label encoding handling unseen categories - standalone version"""
    if label_encoder is None:
        le = LabelEncoder()
        return le.fit_transform(series.fillna("missing").astype(str))

    series_str = series.fillna("missing").astype(str)
    known_classes = set(label_encoder.classes_)

    if len(known_classes) > 0:
        default_category = list(known_classes)[0]
    else:
        default_category = "unknown"

    # Map unseen categories to default
    series_mapped = series_str.apply(
        lambda x: x if x in known_classes else default_category
    )

    # Extend encoder if needed
    if default_category not in known_classes:
        label_encoder.classes_ = np.append(label_encoder.classes_, default_category)

    return label_encoder.transform(series_mapped)


def generate_submission(test_file_path, output_file_path="r2_submission_file_teamname.csv"):
    """Completely standalone submission generator"""

    print("=== Standalone American Express Submission Generator ===")

    # Load trained pipeline
    try:
        PIPE = joblib.load("amex_pipeline.pkl")
        print("✓ Pipeline loaded successfully!")
    except FileNotFoundError:
        print("✗ Error: amex_pipeline.pkl not found. Please train the model first.")
        return None

    # Extract components
    lgb_model = PIPE["lgb_model"]
    xgb_model = PIPE.get("xgb_model")
    feature_cols = PIPE["feature_cols"]
    label_encoders = PIPE.get("label_encoders", {})
    CFG = PIPE["config"]

    # Load test data
    print(f"Loading test data from: {test_file_path}")
    test = pd.read_csv(test_file_path)
    print(f"✓ Test data loaded: {test.shape}")

    # Process offer metadata if available
    try:
        meta = pd.read_csv("data/offer_metadata.csv")
        print("✓ Loading offer metadata...")

        # Process dates
        for col in ['f374', 'id13', 'id12']:
            if col in meta.columns:
                meta[col] = safe_datetime_convert(meta[col])

        today = pd.Timestamp("2023-11-05").normalize()

        # Create active flag
        if 'id12' in meta.columns and 'id13' in meta.columns:
            meta["is_currently_active"] = (
                    (meta["id12"].isna() | (meta["id12"] <= today)) &
                    (meta["id13"].isna() | (meta["id13"] >= today))
            ).astype("int8")
        else:
            meta["is_currently_active"] = 1

        # Discount bucketing
        if 'f376' in meta.columns:
            meta["discount_bucket"] = meta["f376"].apply(bucket_val)
        else:
            meta["discount_bucket"] = "unk"

        # Select useful columns
        keep_cols = ["id3", "is_currently_active", "discount_bucket"]
        for col in ["id9", "f375", "f376", "id10", "id11"]:
            if col in meta.columns:
                keep_cols.append(col)

        meta = meta[keep_cols]
        test = test.merge(meta, on="id3", how="left")
        print("✓ Offer metadata merged successfully")

    except Exception as e:
        print(f"⚠ Warning: Could not process offer meta {e}")
        test["is_currently_active"] = 1
        test["discount_bucket"] = "unk"

    # Add transaction features if available
    try:
        trans = pd.read_csv("data/add_trans.csv")
        print("✓ Processing transaction data...")

        if 'id2' in trans.columns and 'f367' in trans.columns:
            trans['f367'] = pd.to_numeric(trans['f367'], errors='coerce').fillna(0)
            agg = (trans.groupby("id2")
                   .agg(trans_cnt=("f367", "size"),
                        trans_amt_sum=("f367", "sum"),
                        trans_amt_mean=("f367", "mean"))
                   ).reset_index()
            test = test.merge(agg, on="id2", how="left")
            test.fillna({"trans_cnt": 0, "trans_amt_sum": 0, "trans_amt_mean": 0}, inplace=True)
            print("✓ Transaction features added")
        else:
            test["trans_cnt"] = 0
            test["trans_amt_sum"] = 0
            test["trans_amt_mean"] = 0
    except:
        test["trans_cnt"] = 0
        test["trans_amt_sum"] = 0
        test["trans_amt_mean"] = 0

    # Handle categorical encoding
    cat_cols = []
    for col in ["discount_bucket", "id9", "id10", "id11"]:
        if col in test.columns:
            cat_cols.append(col)
            test[col] = test[col].fillna("missing").astype(str)

            if col in label_encoders:
                test[col] = safe_label_encode(test[col], label_encoders[col])
            else:
                test[col] = safe_label_encode(test[col], None)

    print(f"✓ Processed categorical columns: {cat_cols}")

    # Convert all object columns to numeric
    for col in test.columns:
        if test[col].dtype == 'object':
            test[col] = pd.to_numeric(test[col], errors='coerce')
        test[col] = test[col].fillna(0)

    # Ensure all required features exist
    missing_features = [col for col in feature_cols if col not in test.columns]
    if missing_features:
        print(f"⚠ Adding {len(missing_features)} missing features with default values")
        for col in missing_features:
            test[col] = 0

    # Select features in correct order
    X_test = test[feature_cols]
    print(f"✓ Feature matrix ready: {X_test.shape}")

    # Generate predictions
    print("🔮 Generating predictions...")
    lgb_pred = lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration)

    if xgb_model is not None:
        try:
            xgb_pred = xgb_model.predict_proba(X_test)[:, 1]
            final_pred = 0.7 * lgb_pred + 0.3 * xgb_pred
            print("✓ Using LightGBM + XGBoost ensemble")
        except:
            final_pred = lgb_pred
            print("✓ Using LightGBM only")
    else:
        final_pred = lgb_pred
        print("✓ Using LightGBM only")

    # Create submission with correct format
    print("📝 Formatting submission...")
    submission = pd.DataFrame()

    # Format id1 according to competition template
    if 'id1' in test.columns and not (test['id1'] == 0).all():
        submission['id1'] = test['id1'].astype(str)
        # Replace any 0.0 values
        submission['id1'] = submission['id1'].replace(['0.0', '0'],
                                                      test['id2'].astype(str) + "_" +
                                                      test['id3'].astype(str) + "_" +
                                                      "16-23_2023-11-05")
    else:
        # Create composite id1
        submission['id1'] = (test['id2'].astype(str) + "_" +
                             test['id3'].astype(str) + "_" +
                             "16-23_2023-11-05")

    submission['id2'] = test['id2'].astype(int)
    submission['id3'] = test['id3'].astype(int)

    # Format id5 as date
    if 'id5' in test.columns and not (test['id5'] == 0).all():
        test['id5_dt'] = safe_datetime_convert(test['id5'])
        submission['id5'] = test['id5_dt'].dt.strftime('%m/%d/%Y')
        submission['id5'] = submission['id5'].fillna("11/05/2023")
    else:
        submission['id5'] = "11/05/2023"

    submission['pred'] = final_pred

    # Final validation and save
    print("💾 Saving submission...")
    submission.to_csv(output_file_path, index=False)

    print(f"✅ SUCCESS! Submission saved to: {output_file_path}")
    print(f"📊 Submission shape: {submission.shape}")
    print(f"🎯 Prediction range: {final_pred.min():.6f} - {final_pred.max():.6f}")
    print(f"📋 Sample rows:")
    print(submission.head(3).to_string(index=False))

    return submission


def robust_template_match(template_file, output_file, final_file):
    """Robust template matching using dictionary lookup"""

    template = pd.read_csv(template_file)
    output = pd.read_csv(output_file)

    print(f"Template shape: {template.shape}")
    print(f"Output shape: {output.shape}")

    # Create lookup dictionary from output: (id2, id3) -> pred
    pred_lookup = {}
    for _, row in output.iterrows():
        key = (int(row['id2']), int(row['id3']))
        pred_lookup[key] = row['pred']

    print(f"Created lookup for {len(pred_lookup)} predictions")

    # Apply predictions to template
    predictions = []
    matches = 0

    for _, row in template.iterrows():
        key = (int(row['id2']), int(row['id3']))
        if key in pred_lookup:
            predictions.append(pred_lookup[key])
            matches += 1
        else:
            predictions.append(0.0)  # Default for missing matches

    # Add predictions to template
    template['pred'] = predictions

    print(f"Successfully matched {matches}/{len(template)} predictions")

    # Save final submission
    template.to_csv(final_file, index=False)
    print(f"✅ Final submission saved: {final_file}")

    return template




if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python standalone_submission_generator.py <test_file> [output_file]")
        print("Example: python standalone_submission_generator.py data/test_data.csv")
    else:
        test_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else "r2_submission_file_teamname.csv"
        # generate_submission(test_file, output_file)
        # quick_template_match("data/submission_template.csv", "r2_submission_file_yourteamname.csv", "final_submission.csv")
        # Usage
        robust_template_match(
            "data/submission_template.csv",
            output_file,
            "final_submission.csv"
        )
