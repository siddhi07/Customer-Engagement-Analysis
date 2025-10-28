import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Load data
df_train = pd.read_csv("train_features.csv")
df_test = pd.read_csv("test_features.csv")
submission = pd.read_csv("685404e30cfdb_submission_template.csv")

# Add CTR per offer
ctr_offer = df_train.groupby("id3")["y"].mean().reset_index().rename(columns={"y": "ctr_offer"})
df_train = df_train.merge(ctr_offer, on="id3", how="left")
df_test  = df_test.merge(ctr_offer, on="id3", how="left")

# Add CTR per user
ctr_user = df_train.groupby("id2")["y"].mean().reset_index().rename(columns={"y": "ctr_user"})
df_train = df_train.merge(ctr_user, on="id2", how="left")
df_test  = df_test.merge(ctr_user, on="id2", how="left")

# Prepare training features
target_col = "y"
id_col = "id1"
drop_cols = [target_col, id_col, "id2", "id3", "id4"]

X = df_train.drop(columns=drop_cols, errors="ignore")
X = X.select_dtypes(include=["number"])
y = df_train[target_col]

# Prepare test features
X_test = df_test.drop(columns=["id1", "id2", "id3", "id4"], errors="ignore")
X_test = X_test.select_dtypes(include=["number"])

# Train model
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model.fit(X, y)

# Predict
y_pred = model.predict_proba(X_test)[:, 1]

# Prepare submission
submission["pred"] = y_pred
submission["pred"] = (submission["pred"] > 0.5).astype(int)  # optional: make hard 0/1
submission.to_csv("submission.csv", index=False)

print("✅ Submission file saved as 'submission.csv'")
