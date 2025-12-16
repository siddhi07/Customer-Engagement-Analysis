Customer Engagement Analysis – AMEX Offer Recommendation System

This project builds a customer engagement and offer recommendation system using transaction behavior, event interactions, and offer metadata.
The goal is to predict whether a customer will engage (click) with a given offer, and to rank offers per customer using machine learning models.

The project is designed as a modular, reproducible ML pipeline, starting from raw Parquet data and ending with trained models and submission files.

📂 Why the Raw Data Is Not in This Repository

The original dataset consists of large Parquet files (several GBs), which are not suitable for GitHub.

Therefore:

❌ Raw Parquet files are excluded from this repository

✅ The repository contains:

ETL scripts

Feature engineering code

Model training pipelines

Instructions to regenerate all intermediate files

This follows industry best practices for ML repositories.

🧠 End-to-End Data Pipeline
Raw Parquet Data
   ↓
read.py
   → train_features.csv
   → test_features.csv
   ↓
ETL / Feature Engineering
   ↓
amex_offer_train.py  (main ML pipeline)
OR
etl_round2.py / model.py (baseline & ranking models)
   ↓
submission.csv / amex_pipeline.pkl

##📁 Repository Structure

├── read.py                  # Converts raw Parquet files into feature CSVs

├── etl_round2.py            # Feature selection + ranking model (LightGBM)

├── amex_offer_train.py      # Main ML pipeline (LightGBM + XGBoost ensemble)

├── model.py                 # Baseline RandomForest model + submission

├── data_dictionary.csv      # Column definitions

├── README.md

└── .gitignore               # Excludes Parquet & large data files

🔄 Step-by-Step Pipeline Explanation
1️⃣ read.py — Raw Data → Feature CSVs

Purpose:
Transforms raw Parquet data into model-ready CSV files.

Inputs (NOT included in GitHub):

train_data.parquet

test_data.parquet

add_event.parquet

add_trans.parquet

offer_metadata.parquet

Key Operations:

Data type normalization

Offer metadata joins

Event-based interaction features

RFM features:

Recency

Frequency

Monetary value

Outputs:

train_features.csv

test_features.csv

python read.py

2️⃣ Feature Engineering & Modeling Options

You can proceed with either of the following paths.

🔹 Option A: amex_offer_train.py (Main Production Pipeline)

Purpose:
End-to-end ML pipeline for high-quality predictions.

Key Features:

Advanced feature engineering

Datetime feature extraction

Transaction aggregations

Ensemble of:

LightGBM

XGBoost

Outputs:

amex_pipeline.pkl → saved trained pipeline

Validation metrics (LogLoss)

python amex_offer_train.py

🔹 Option B: etl_round2.py + model.py (Baselines & Ranking)
etl_round2.py

Performs:

Label encoding

Variance thresholding

GroupKFold splitting (by customer)

Trains a LightGBM LambdaRank model

Outputs:

submission.csv

python etl_round2.py

model.py

Trains a RandomForest baseline

Adds:

CTR per user

CTR per offer

Outputs:

submission.csv

python model.py

📤 Final Outputs
File	Description
train_features.csv	Engineered training features
test_features.csv	Engineered test features
submission.csv	Final predictions
amex_pipeline.pkl	Serialized trained model pipeline
🛠️ Requirements

Install dependencies using:

pip install pandas numpy scikit-learn lightgbm xgboost catboost joblib

🚀 Key Highlights

Scalable ETL design

Customer-aware modeling (prevents data leakage)

Ranking + classification approaches

Production-ready pipeline saving

GitHub-friendly structure (no large data committed)

📌 Notes

To fully reproduce results, place the raw Parquet files in the project root before running read.py

Sample data can be used for testing if full data is unavailable

All scripts are modular and can be run independently
