import os
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, confusion_matrix

DATASET_PATH = r"data\\raw\\2021MCMProblemC_DataSet.xlsx"
IMAGES_BY_ID_PATH = r"data\\raw\\2021MCM_ProblemC_ Images_by_GlobalID.xlsx"
IMAGES_DIRS = [
    r"data\\raw\\images",
    r"data\\raw\\2021MCM_ProblemC_Files",
]
OUTPUT_PATH = r"data\\processed\\outputs_t2_ranked_reports.csv"
RANDOM_STATE = 42


@dataclass
class RiskPriorModel:
    time_decay_days: float = 180.0
    bandwidth_km: float = 50.0


def load_reports(path: str = DATASET_PATH) -> pd.DataFrame:
    # 读取原始报告数据，保留关键字段并清洗时间与坐标
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


def load_images_by_globalid(path: str = IMAGES_BY_ID_PATH) -> pd.DataFrame:
    # 读取图片索引表，统一列名并去除无文件名记录
    df = pd.read_excel(path)
    df = df.rename(columns=str.strip)
    df = df[df["FileName"].notna()].copy()
    df["FileName"] = df["FileName"].astype(str)
    return df


def _haversine_km(lat1, lon1, lat2, lon2):
    # 计算地球表面两点之间的大圆距离（单位：公里）
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
    # 基于历史阳性记录，按时间衰减与空间核计算风险先验
    if len(train_pos) == 0:
        return np.zeros(len(query_dates))

    t_train = train_pos["Detection Date"].values.astype("datetime64[D]")
    lat_train = train_pos["Latitude"].to_numpy()
    lon_train = train_pos["Longitude"].to_numpy()

    q_dates = pd.to_datetime(query_dates).values.astype("datetime64[D]")
    dt_days = (q_dates[:, None] - t_train[None, :]).astype("timedelta64[D]").astype(int)
    dt_days = np.maximum(dt_days, 0)
    time_weight = np.exp(-dt_days / model.time_decay_days)

    # 距离越远权重越小
    dist_km = _haversine_km(query_lats[:, None], query_lons[:, None], lat_train[None, :], lon_train[None, :])
    space_weight = np.exp(-(dist_km ** 2) / (2.0 * model.bandwidth_km ** 2))

    return (time_weight * space_weight).sum(axis=1)


def _find_image_path(filename: str) -> str:
    # 在多个图片目录中查找文件路径
    for base in IMAGES_DIRS:
        candidate = os.path.join(base, filename)
        if os.path.exists(candidate):
            return candidate
    return ""


def build_image_index(img_df: pd.DataFrame) -> Dict[str, List[str]]:
    # 过滤支持的图像扩展名，并建立 GlobalID 到图片路径列表的映射
    valid_ext = {".jpg", ".jpeg", ".png"}
    img_df = img_df.copy()
    img_df["ext"] = img_df["FileName"].str.lower().str.extract(r"(\.[a-z0-9]+)$", expand=False)
    img_df = img_df[img_df["ext"].isin(valid_ext)]

    mapping: Dict[str, List[str]] = {}
    for _, row in img_df.iterrows():
        gid = str(row["GlobalID"]).strip()
        fname = row["FileName"].strip()
        path = _find_image_path(fname)
        if not path:
            continue
        mapping.setdefault(gid, []).append(path)
    return mapping


def _sobel_edges(gray: np.ndarray) -> np.ndarray:
    # 简化 Sobel 边缘检测，输出梯度幅值图
    if gray.shape[0] < 3 or gray.shape[1] < 3:
        return np.zeros_like(gray, dtype=float)
    g = gray.astype(float)
    gx = (
        -1 * g[:-2, :-2] + 1 * g[:-2, 2:] +
        -2 * g[1:-1, :-2] + 2 * g[1:-1, 2:] +
        -1 * g[2:, :-2] + 1 * g[2:, 2:]
    )
    gy = (
        -1 * g[:-2, :-2] + -2 * g[:-2, 1:-1] + -1 * g[:-2, 2:] +
        1 * g[2:, :-2] + 2 * g[2:, 1:-1] + 1 * g[2:, 2:]
    )
    mag = np.sqrt(gx ** 2 + gy ** 2)
    return mag


def extract_image_features(path: str) -> Dict[str, float]:
    # 提取单张图片的灰度统计与边缘强度特征
    try:
        with Image.open(path) as img:
            img = img.convert("L")
            arr = np.array(img, dtype=float)
    except Exception:
        return {}

    h, w = arr.shape
    mean_bright = float(arr.mean())
    std_bright = float(arr.std())
    aspect = float(w / h) if h > 0 else 0.0

    edges = _sobel_edges(arr)
    edge_density = float((edges > 50.0).mean()) if edges.size > 0 else 0.0
    edge_mean = float(edges.mean()) if edges.size > 0 else 0.0

    return {
        "img_w": float(w),
        "img_h": float(h),
        "img_aspect": aspect,
        "img_mean": mean_bright,
        "img_std": std_bright,
        "img_edge_density": edge_density,
        "img_edge_mean": edge_mean,
    }


def aggregate_image_features(paths: List[str]) -> Dict[str, float]:
    # 对同一 GlobalID 的多张图片进行统计汇总
    feats = [extract_image_features(p) for p in paths]
    feats = [f for f in feats if f]
    if not feats:
        return {}

    df = pd.DataFrame(feats)
    agg = {}
    for col in df.columns:
        agg[f"{col}_mean"] = float(df[col].mean())
        agg[f"{col}_max"] = float(df[col].max())
    agg["num_images"] = float(len(feats))
    return agg


def add_text_features(df: pd.DataFrame) -> pd.DataFrame:
    # 从 Notes 文本中提取简单关键词与长度特征
    df = df.copy()
    df["note"] = df["Notes"].fillna("").astype(str).str.lower()
    df["note_length"] = df["note"].str.len()
    df["kw_large"] = df["note"].str.contains("large", na=False).astype(int)
    df["kw_yellow"] = df["note"].str.contains("yellow", na=False).astype(int)
    df["kw_queen"] = df["note"].str.contains("queen", na=False).astype(int)
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    # 生成月份与报告延迟（提交-发现）特征
    df = df.copy()
    df["month"] = df["Detection Date"].dt.month
    df["delay_days"] = (df["Submission Date"] - df["Detection Date"]).dt.days
    df["delay_days"] = df["delay_days"].fillna(0).clip(lower=0)
    return df


def build_features(df: pd.DataFrame, image_index: Dict[str, List[str]]) -> pd.DataFrame:
    # 合并图像路径、统计图像特征并补全缺失值
    df = df.copy()
    df["image_paths"] = df["GlobalID"].astype(str).map(image_index).fillna("")
    df["image_paths"] = df["image_paths"].apply(lambda x: x if isinstance(x, list) else [])
    df["has_image"] = df["image_paths"].apply(lambda x: 1 if len(x) > 0 else 0)

    img_rows = []
    for paths in df["image_paths"]:
        img_rows.append(aggregate_image_features(paths))

    img_feat_df = pd.DataFrame(img_rows)
    df = pd.concat([df.reset_index(drop=True), img_feat_df.reset_index(drop=True)], axis=1)
    df["num_images"] = df["num_images"].fillna(0)
    return df


def train_image_model(train_df: pd.DataFrame, image_feature_cols: List[str]) -> Tuple[LogisticRegression, float]:
    # 仅在有图片的样本上训练图像子模型，并返回均值作为回退概率
    df = train_df[train_df["has_image"] == 1].copy()
    if len(df) < 5:
        return None, float(train_df["y"].mean())

    X = df[image_feature_cols].fillna(0.0)
    y = df["y"]

    model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    model.fit(X, y)
    return model, float(y.mean())


def predict_prob_img(df: pd.DataFrame, model: LogisticRegression, fallback: float,
                     image_feature_cols: List[str]) -> np.ndarray:
    # 预测图像概率；若无模型则使用回退概率
    if model is None:
        return np.full(len(df), fallback, dtype=float)

    X = df[image_feature_cols].fillna(0.0)
    prob = model.predict_proba(X)[:, 1]
    return prob


def get_time_splits(df: pd.DataFrame, min_train_periods: int = 3) -> List[Tuple[pd.Index, pd.Index]]:
    # 生成按月滚动的训练/测试划分，用于时序评估
    df = df.copy()
    df["period"] = df["Detection Date"].dt.to_period("M")
    periods = sorted(df["period"].unique())
    splits = []
    for i in range(min_train_periods, len(periods)):
        train_periods = periods[:i]
        test_period = periods[i]
        train_idx = df[df["period"].isin(train_periods)].index
        test_idx = df[df["period"] == test_period].index
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        splits.append((train_idx, test_idx))
    return splits


def evaluate_model(df: pd.DataFrame,
                   feature_cols: List[str],
                   image_feature_cols: List[str],
                   risk_model: RiskPriorModel) -> Dict[str, float]:
    # 在滚动时间窗口上评估模型的 AUROC/AUPRC 及 Top-K uplift
    splits = get_time_splits(df)
    if not splits:
        return {}

    aucs = []
    auprcs = []
    uplifts = {1: [], 5: [], 10: []}

    for train_idx, test_idx in splits:
        train_df = df.loc[train_idx].copy()
        test_df = df.loc[test_idx].copy()

        train_pos = train_df[train_df["Lab Status"] == "Positive ID"]
        if train_df["y"].nunique() < 2:
            continue
        train_df["risk_prior"] = compute_risk_prior(
            train_pos,
            train_df["Detection Date"].to_numpy(),
            train_df["Latitude"].to_numpy(),
            train_df["Longitude"].to_numpy(),
            risk_model,
        )
        test_df["risk_prior"] = compute_risk_prior(
            train_pos,
            test_df["Detection Date"].to_numpy(),
            test_df["Latitude"].to_numpy(),
            test_df["Longitude"].to_numpy(),
            risk_model,
        )
        train_df["risk_prior_log"] = np.log1p(train_df["risk_prior"])
        test_df["risk_prior_log"] = np.log1p(test_df["risk_prior"])

        image_model, img_fallback = train_image_model(train_df, image_feature_cols)
        train_df["prob_img"] = predict_prob_img(train_df, image_model, img_fallback, image_feature_cols)
        test_df["prob_img"] = predict_prob_img(test_df, image_model, img_fallback, image_feature_cols)

        model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
        model.fit(train_df[feature_cols].fillna(0.0), train_df["y"])
        prob = model.predict_proba(test_df[feature_cols].fillna(0.0))[:, 1]

        y_true = test_df["y"].to_numpy()
        if len(np.unique(y_true)) < 2:
            continue

        aucs.append(roc_auc_score(y_true, prob))
        auprcs.append(average_precision_score(y_true, prob))

        overall_pos = (y_true == 1).mean()
        for k in [1, 5, 10]:
            n_top = max(1, int(len(prob) * (k / 100.0)))
            top_idx = np.argsort(prob)[-n_top:]
            top_pos = (y_true[top_idx] == 1).mean()
            uplift = top_pos / overall_pos if overall_pos > 0 else np.nan
            uplifts[k].append(uplift)

    metrics = {
        "auroc_mean": float(np.mean(aucs)) if aucs else np.nan,
        "auprc_mean": float(np.mean(auprcs)) if auprcs else np.nan,
        "uplift_1_mean": float(np.mean(uplifts[1])) if uplifts[1] else np.nan,
        "uplift_5_mean": float(np.mean(uplifts[5])) if uplifts[5] else np.nan,
        "uplift_10_mean": float(np.mean(uplifts[10])) if uplifts[10] else np.nan,
        "num_splits": len(splits),
    }
    return metrics


def main():
    warnings.filterwarnings("ignore")
    # 读取原始报告数据，包含时间、地点、文本与标签等字段
    df = load_reports()

    # 读取图片索引表并建立 GlobalID -> 图片路径列表的映射
    img_df = load_images_by_globalid()
    image_index = build_image_index(img_df)
    if not image_index:
        print("Warning: no image files found. Running without image features.")

    # 构造文本特征与时间特征，并拼接图片统计特征
    df = add_text_features(df)
    df = add_time_features(df)
    df = build_features(df, image_index)

    # 构造训练标签：Positive ID 记为 1，其余为 0
    df["y"] = (df["Lab Status"] == "Positive ID").astype(int)

    # 配置时空风险先验的超参数
    risk_model = RiskPriorModel()

    # 选择用于图像模型的特征列（若缺失则剔除）
    image_feature_cols = [
        "img_mean_mean",
        "img_std_mean",
        "img_edge_density_mean",
        "img_edge_mean_mean",
        "img_aspect_mean",
    ]
    image_feature_cols = [c for c in image_feature_cols if c in df.columns]

    if not image_feature_cols:
        image_feature_cols = []

    # 主模型的特征列：图像先验 + 时空先验 + 文本/时间/地理信息
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
        "has_image",
        "num_images",
    ]

    # 仅使用已有标签的数据进行滚动时间评估
    labeled = df[df["Lab Status"].isin(["Positive ID", "Negative ID"])].copy()
    splits = get_time_splits(labeled)
    if splits:
        metrics = evaluate_model(labeled, feature_cols, image_feature_cols, risk_model)
        print("Evaluation metrics:", metrics)
    else:
        print("Not enough time periods for rolling evaluation.")

    # Train final models on all labeled data
    # 在全部有标签数据上训练最终模型并对全体样本输出概率
    labeled_pos = labeled[labeled["Lab Status"] == "Positive ID"]
    df["risk_prior"] = compute_risk_prior(
        labeled_pos,
        df["Detection Date"].to_numpy(),
        df["Latitude"].to_numpy(),
        df["Longitude"].to_numpy(),
        risk_model,
    )
    df["risk_prior_log"] = np.log1p(df["risk_prior"])

    # 训练图像子模型，预测每条样本的图像风险概率
    image_model, img_fallback = train_image_model(labeled, image_feature_cols)
    df["prob_img"] = predict_prob_img(df, image_model, img_fallback, image_feature_cols)

    # 训练最终逻辑回归模型并输出正样本概率与错误概率
    labeled = df[df["Lab Status"].isin(["Positive ID", "Negative ID"])].copy()
    final_model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    final_model.fit(labeled[feature_cols].fillna(0.0), labeled["y"])

    df["prob_positive"] = final_model.predict_proba(df[feature_cols].fillna(0.0))[:, 1]
    df["mistake_prob"] = 1.0 - df["prob_positive"]

    # 保存结果到指定路径，包含概率与关键特征
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    output_cols = [
        "GlobalID", "Lab Status", "Detection Date", "Submission Date",
        "Latitude", "Longitude", "Notes",
        "prob_img", "risk_prior", "risk_prior_log", "prob_positive", "mistake_prob",
        "has_image", "num_images",
    ]
    df[output_cols].to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
    print(f"Saved ranked reports to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
