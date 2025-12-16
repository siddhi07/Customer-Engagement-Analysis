import pandas as pd

# 1. Read all files
df_train  = pd.read_parquet("train_data.parquet")
df_test   = pd.read_parquet("test_data.parquet")
df_events = pd.read_parquet("add_event.parquet")
df_trans  = pd.read_parquet("add_trans.parquet")
df_offers = pd.read_parquet("offer_metadata.parquet")
df_dict   = pd.read_csv("data_dictionary.csv")

# 2. Ensure consistent types before any merge
df_train["id2"] = df_train["id2"].astype(str)
df_test["id2"] = df_test["id2"].astype(str)
df_events["id2"] = df_events["id2"].astype(str)
df_trans["id2"] = df_trans["id2"].astype(str)

df_train["id3"] = df_train["id3"].astype("int64")
df_test["id3"] = df_test["id3"].astype("int64")
df_events["id3"] = df_events["id3"].astype("int64")
df_offers["id3"] = df_offers["id3"].astype("int64")

# 3. Merge Offer Metadata
df_train = df_train.merge(df_offers, on="id3", how="left")
df_test = df_test.merge(df_offers, on="id3", how="left")

# 4. Feature: past impression count
evt_cnt = (
    df_events
    .groupby(["id2", "id3"])
    .size()
    .reset_index(name="past_impression_count")
)

# Ensure types match before merging again
evt_cnt["id2"] = evt_cnt["id2"].astype(str)
evt_cnt["id3"] = evt_cnt["id3"].astype("int64")

df_train = df_train.merge(evt_cnt, on=["id2", "id3"], how="left")
df_test = df_test.merge(evt_cnt, on=["id2", "id3"], how="left")

# Fill NA after merge
df_train["past_impression_count"].fillna(0, inplace=True)
df_test["past_impression_count"].fillna(0, inplace=True)

# 5. RFM Features
df_trans["f370"] = pd.to_datetime(df_trans["f370"])

rfm = (
    df_trans
    .groupby("id2")
    .agg(
        recency=("f370", lambda x: (pd.Timestamp.today() - x.max()).days),
        frequency=("id2", "count"),
        monetary=("f367", "sum")
    )
    .reset_index()
)

rfm["id2"] = rfm["id2"].astype(str)

df_train = df_train.merge(rfm, on="id2", how="left")
df_test = df_test.merge(rfm, on="id2", how="left")

# Fill RFM NAs
df_train[["recency", "frequency", "monetary"]] = df_train[["recency", "frequency", "monetary"]].fillna(0)
df_test[["recency", "frequency", "monetary"]] = df_test[["recency", "frequency", "monetary"]].fillna(0)

# 6. Write to CSV
df_train.to_csv("train_features.csv", index=False)
df_test.to_csv("test_features.csv", index=False)

print("✅ train_features.csv and test_features.csv generated successfully.")
