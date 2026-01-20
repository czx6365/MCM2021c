import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

DATASET_PATH = r"data\\raw\\2021MCMProblemC_DataSet.xlsx"


# -----------------------------
# 工具：Haversine 距离（km）
# -----------------------------
def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


# -----------------------------
# Step 1: 读取并清洗数据
# -----------------------------
df = pd.read_excel(DATASET_PATH)

# 仅使用已被实验室处理的数据
df = df[df["Lab Status"].isin(["Positive ID", "Negative ID"])].copy()

# 标签：1 = 误报（Negative），0 = 真阳性（Positive）
df["y"] = (df["Lab Status"] == "Negative ID").astype(int)

# 日期处理
df["Detection Date"] = pd.to_datetime(df["Detection Date"], errors="coerce")
df["Submission Date"] = pd.to_datetime(df["Submission Date"], errors="coerce")

df = df.dropna(subset=["Detection Date", "Latitude", "Longitude"])

# -----------------------------
# Step 2: 时间特征
# -----------------------------
df["month"] = df["Detection Date"].dt.month
df["delay_days"] = (df["Submission Date"] - df["Detection Date"]).dt.days
df["delay_days"] = df["delay_days"].fillna(0).clip(lower=0)


# -----------------------------
# Step 3: 空间特征
# 到最近 Positive ID 的距离
# -----------------------------
positives = df[df["Lab Status"] == "Positive ID"]

pos_lats = positives["Latitude"].to_numpy()
pos_lons = positives["Longitude"].to_numpy()

def dist_to_nearest_positive(row):
    if len(pos_lats) == 0:
        return 300.0
    dists = haversine_km(
        row["Latitude"], row["Longitude"],
        pos_lats, pos_lons
    )
    return float(np.min(dists))

df["dist_to_positive"] = df.apply(dist_to_nearest_positive, axis=1)


# -----------------------------
# Step 4: 文本特征（极简）
# -----------------------------
df["note"] = df["Notes"].fillna("").astype(str).str.lower()

df["note_length"] = df["note"].str.len()
df["kw_large"] = df["note"].str.contains("large", na=False).astype(int)
df["kw_yellow"] = df["note"].str.contains("yellow", na=False).astype(int)
df["kw_queen"] = df["note"].str.contains("queen", na=False).astype(int)


# -----------------------------
# Step 5: 逻辑回归（概率模型）
# -----------------------------
feature_cols = [
    "month",
    "delay_days",
    "dist_to_positive",
    "note_length",
    "kw_large",
    "kw_yellow",
    "kw_queen"
]

X = df[feature_cols]
y = df["y"]

model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# 误报概率
df["mistake_prob"] = model.predict_proba(X)[:, 1]


# -----------------------------
# Step 6: 简单验证（排序是否有用）
# -----------------------------
df_sorted = df.sort_values("mistake_prob")

top_10 = df_sorted.head(int(0.1 * len(df_sorted)))
overall_positive_rate = (df["y"] == 0).mean()
top_positive_rate = (top_10["y"] == 0).mean()

print("Overall Positive Rate:", overall_positive_rate)
print("Top 10% (Lowest Mistake Prob) Positive Rate:", top_positive_rate)

# 保存结果
df_sorted.to_csv("outputs_t2_ranked_reports.csv", index=False)
print("Saved ranked reports to outputs_t2_ranked_reports.csv")
