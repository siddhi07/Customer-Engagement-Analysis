import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import VarianceThreshold
import warnings
warnings.filterwarnings("ignore")

# ========== 1. LOAD DATA ==========
train = pd.read_csv("train_features.csv")    # You must provide this file!
test = pd.read_csv("test_features.csv")      # You must provide this file!

# ========== 2. FEATURE & TARGET DEFINITION ==========
# Your files are of the form: id1, id2, id3, id4, ..., id13, f1, ..., fN, target(y)
ID_COLS = ['id1', 'id2', 'id3', 'id5']
TARGET = 'y'  # adjust if your target column is named differently
DROP_COLS = [c for c in train.columns if c.startswith('id')] + [TARGET]

X = train.drop(DROP_COLS, axis=1, errors='ignore')
y = train[TARGET]
X_test = test[X.columns]

# ========== 3. HANDLE CATEGORICALS ==========
# Label encode all categorical (object/string) columns so model can use them
for col in X.columns:
    if X[col].dtype == 'object' or pd.api.types.is_string_dtype(X[col]):
        le = LabelEncoder()
        all_vals = pd.concat([X[col], X_test[col]], axis=0).astype(str).fillna('missing')
        le.fit(all_vals)
        X[col] = le.transform(X[col].astype(str).fillna('missing'))
        X_test[col] = le.transform(X_test[col].astype(str).fillna('missing'))

# ========== 4. FEATURE SELECTION ==========
# Remove almost-constant/zero-variance features
sel = VarianceThreshold(threshold=1e-5)
X = pd.DataFrame(sel.fit_transform(X), columns=X.columns[sel.get_support()])
X_test = pd.DataFrame(sel.transform(X_test), columns=X.columns)

# ========== 5. TRAIN/VALIDATION SPLITTING ==========
# Use GroupKFold to avoid customer leakage (grouping by 'id2')
groups = train['id2']
folds = GroupKFold(n_splits=5)
for trn_idx, val_idx in folds.split(X, y, groups):
    break  # use only the first fold for illustration
X_tr, X_val = X.iloc[trn_idx], X.iloc[val_idx]
y_tr, y_val = y.iloc[trn_idx], y.iloc[val_idx]

# ========== 6. LIGHTGBM RANKER TRAINING ==========
train_group = train.iloc[trn_idx].groupby('id2').size().values
val_group = train.iloc[val_idx].groupby('id2').size().values

lgb_train = lgb.Dataset(X_tr, y_tr, group=train_group)
lgb_val = lgb.Dataset(X_val, y_val, group=val_group, reference=lgb_train)

params = {
    'objective': 'lambdarank',
    'metric': 'map',
    'boosting_type': 'gbdt',
    'num_leaves': 63,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 3,
    'seed': 42,
    'verbosity': -1,
}

model = lgb.train(
    params,
    lgb_train,
    valid_sets=[lgb_train, lgb_val],
    num_boost_round=500,
    early_stopping_rounds=40,
    verbose_eval=50
)

# ========== 7. PREDICTION & NORMALIZATION ==========
test_pred = model.predict(X_test, num_iteration=model.best_iteration)
# Normalize per-customer if your metric rewards within-customer ranking
test['pred'] = test_pred
test['rank_score'] = test.groupby('id2')['pred'].transform(lambda x: (x - x.min()) / (x.max() - x.min() + 1e-9))

# ========== 8. SAVE SUBMISSION ==========
submission = test[ID_COLS].copy()
submission['pred'] = test['rank_score']
submission.to_csv("submission.csv", index=False)
