import os
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

DATASET_PATH = r"data\\raw\\2021MCMProblemC_DataSet.xlsx"
OUTPUT_PATH = r"data\\processed\\outputs_t2_without_images.csv"
RANDOM_STATE = 42


@dataclass
class RiskPriorModel:
    time_decay_days: float = 180.0
    bandwidth_km: float = 50.0


def load_reports(path: str = DATASET_PATH) -> pd.DataFrame:
    df = pd.read_excel(path)
    keep_cols = [
        "GlobalID", "Detection Date", "Submission Date",
        "Latitude", "Longitude", "Notes", "Lab Status"
    ]
    df = df[keep_cols].copy()
    df["Detection Date"] = pd.to_datetime(df["Detection Date"], errors="coerce")
    df["Submission Date"] = pd.to_datetime(df["Submission Date"], errors="coerce")
    df = df.dropna(subset=["Detection Date", "Latitude", "Longitude"])
    df["Latitude"] = df["Latitude"].astype(float)
    df["Longitude"] = df["Longitude"].astype(float)
    return df


def _haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def compute_risk_prior(train_pos: pd.DataFrame,
                       query_dates: np.ndarray,
                       query_lats: np.ndarray,
                       query_lons: np.ndarray,
                       model: RiskPriorModel) -> np.ndarray:
    if len(train_pos) == 0:
        return np.zeros(len(query_dates))

    t_train = train_pos["Detection Date"].values.astype("datetime64[D]")
    lat_train = train_pos["Latitude"].to_numpy()
    lon_train = train_pos["Longitude"].to_numpy()

    q_dates = pd.to_datetime(query_dates).values.astype("datetime64[D]")
    dt_days = (q_dates[:, None] - t_train[None, :]).astype("timedelta64[D]").astype(int)
    dt_days = np.maximum(dt_days, 0)
    time_weight = np.exp(-dt_days / model.time_decay_days)

    dist_km = _haversine_km(
        query_lats[:, None], query_lons[:, None],
        lat_train[None, :], lon_train[None, :]
    )
    space_weight = np.exp(-(dist_km ** 2) / (2.0 * model.bandwidth_km ** 2))

    return (time_weight * space_weight).sum(axis=1)


def add_text_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["note"] = df["Notes"].fillna("").astype(str).str.lower()
    df["note_length"] = df["note"].str.len()
    df["kw_large"] = df["note"].str.contains("large", na=False).astype(int)
    df["kw_yellow"] = df["note"].str.contains("yellow", na=False).astype(int)
    df["kw_queen"] = df["note"].str.contains("queen", na=False).astype(int)
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["month"] = df["Detection Date"].dt.month
    df["delay_days"] = (df["Submission Date"] - df["Detection Date"]).dt.days
    df["delay_days"] = df["delay_days"].fillna(0).clip(lower=0)
    return df


def get_time_splits(df: pd.DataFrame, min_train_periods: int = 3):
    df = df.copy()
    df["period"] = df["Detection Date"].dt.to_period("M")
    periods = sorted(df["period"].unique())
    splits = []
    for i in range(min_train_periods, len(periods)):
        train_idx = df[df["period"].isin(periods[:i])].index
        test_idx = df[df["period"] == periods[i]].index
        if len(train_idx) > 0 and len(test_idx) > 0:
            splits.append((train_idx, test_idx))
    return splits


def evaluate_model(df, feature_cols, risk_model):
    splits = get_time_splits(df)
    aucs, auprcs = [], []
    uplifts = {1: [], 5: [], 10: []}

    for train_idx, test_idx in splits:
        train_df = df.loc[train_idx].copy()
        test_df = df.loc[test_idx].copy()

        train_pos = train_df[train_df["y"] == 1]
        if train_df["y"].nunique() < 2:
            continue

        for part in [train_df, test_df]:
            part["risk_prior"] = compute_risk_prior(
                train_pos,
                part["Detection Date"].to_numpy(),
                part["Latitude"].to_numpy(),
                part["Longitude"].to_numpy(),
                risk_model
            )
            part["risk_prior_log"] = np.log1p(part["risk_prior"])

        model = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE
        )
        model.fit(train_df[feature_cols], train_df["y"])
        prob = model.predict_proba(test_df[feature_cols])[:, 1]

        y_true = test_df["y"].to_numpy()
        if len(np.unique(y_true)) < 2:
            continue

        aucs.append(roc_auc_score(y_true, prob))
        auprcs.append(average_precision_score(y_true, prob))

        overall_pos = (y_true == 1).mean()
        for k in [1, 5, 10]:
            n_top = max(1, int(len(prob) * k / 100))
            top_pos = (y_true[np.argsort(prob)[-n_top:]] == 1).mean()
            uplifts[k].append(top_pos / overall_pos if overall_pos > 0 else np.nan)

    return {
        "auroc_mean": float(np.mean(aucs)),
        "auprc_mean": float(np.mean(auprcs)),
        "uplift_1_mean": float(np.mean(uplifts[1])),
        "uplift_5_mean": float(np.mean(uplifts[5])),
        "uplift_10_mean": float(np.mean(uplifts[10])),
        "num_splits": len(splits),
    }


def main():
    warnings.filterwarnings("ignore")

    df = load_reports()
    df = add_text_features(df)
    df = add_time_features(df)
    df["y"] = (df["Lab Status"] == "Positive ID").astype(int)

    # 无图片：prob_img 为整体正样本率
    base_rate = df["y"].mean()
    df["prob_img"] = base_rate
    df["has_image"] = 0
    df["num_images"] = 0

    risk_model = RiskPriorModel()

    feature_cols = [
        "prob_img",
        "risk_prior_log",
        "note_length",
        "kw_large",
        "kw_yellow",
        "kw_queen",
        "delay_days",
        "month",
        "Latitude",
        "Longitude",
    ]

    labeled = df[df["Lab Status"].isin(["Positive ID", "Negative ID"])].copy()
    metrics = evaluate_model(labeled, feature_cols, risk_model)
    print("Evaluation metrics (without images):", metrics)

    # final model
    labeled_pos = labeled[labeled["y"] == 1]
    for part in [labeled, df]:
        part["risk_prior"] = compute_risk_prior(
            labeled_pos,
            part["Detection Date"].to_numpy(),
            part["Latitude"].to_numpy(),
            part["Longitude"].to_numpy(),
            risk_model
        )
        part["risk_prior_log"] = np.log1p(part["risk_prior"])

    final_model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_STATE
    )
    final_model.fit(labeled[feature_cols], labeled["y"])

    df["prob_positive"] = final_model.predict_proba(df[feature_cols])[:, 1]
    df["mistake_prob"] = 1.0 - df["prob_positive"]

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
    print(f"Saved results to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
