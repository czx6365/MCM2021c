import numpy as np
import pandas as pd
from dataclasses import dataclass

# ===============================
# 数据路径
# ===============================
DATASET_PATH = r"data/raw/2021MCMProblemC_DataSet.xlsx"


# ===============================
# 模型定义
# ===============================
@dataclass
class SpatiotemporalModel:
    train_df: pd.DataFrame
    time_decay_days: float
    bandwidth_km: float


# ===============================
# 数据加载
# ===============================
def load_positive_cases(path: str = DATASET_PATH,
                        freq: str = "M") -> pd.DataFrame:
    df = pd.read_excel(path)

    df = df[df["Lab Status"] == "Positive ID"].copy()
    df = df[["Detection Date", "Latitude", "Longitude"]].dropna()

    df["detection_date"] = pd.to_datetime(df["Detection Date"], errors="coerce")
    df = df.dropna(subset=["detection_date"])

    df["latitude"] = df["Latitude"].astype(float)
    df["longitude"] = df["Longitude"].astype(float)

    df["period"] = df["detection_date"].dt.to_period(freq)

    return df[["detection_date", "latitude", "longitude", "period"]]


# ===============================
# Haversine 距离
# ===============================
def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    lat1, lon1 = np.radians(lat1), np.radians(lon1)
    lat2, lon2 = np.radians(lat2), np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


# ===============================
# 模型拟合（存储）
# ===============================
def fit_spatiotemporal_model(df,
                             time_decay_days=180.0,
                             bandwidth_km=50.0):
    return SpatiotemporalModel(
        train_df=df.reset_index(drop=True),
        time_decay_days=float(time_decay_days),
        bandwidth_km=float(bandwidth_km)
    )


# ===============================
# 风险预测
# ===============================
def predict_risk(model, query_dates, query_lats, query_lons):
    train = model.train_df

    t_train = train["detection_date"].values.astype("datetime64[D]")
    lat_train = train["latitude"].to_numpy()
    lon_train = train["longitude"].to_numpy()

    q_dates = pd.to_datetime(query_dates).values.astype("datetime64[D]")
    q_lats = np.asarray(query_lats, dtype=float)
    q_lons = np.asarray(query_lons, dtype=float)

    # 时间差（负值截断，避免未来信息泄漏）
    dt_days = (q_dates[:, None] - t_train[None, :]).astype("timedelta64[D]").astype(int)
    dt_days = np.maximum(dt_days, 0)

    time_weight = np.exp(-dt_days / model.time_decay_days)

    dist_km = haversine_km(
        q_lats[:, None], q_lons[:, None],
        lat_train[None, :], lon_train[None, :]
    )

    space_weight = np.exp(-(dist_km ** 2) / (2 * model.bandwidth_km ** 2))

    return (time_weight * space_weight).sum(axis=1)


# ===============================
# 空间网格
# ===============================
def make_grid(df, grid_size=50):
    lat_min, lat_max = df["latitude"].min(), df["latitude"].max()
    lon_min, lon_max = df["longitude"].min(), df["longitude"].max()

    lats = np.linspace(lat_min, lat_max, grid_size)
    lons = np.linspace(lon_min, lon_max, grid_size)

    grid_lats, grid_lons = np.meshgrid(lats, lons, indexing="ij")
    return grid_lats.ravel(), grid_lons.ravel()


# ===============================
# T1 核心评估函数
# ===============================
def evaluate_prediction_precision(df,
                                  time_decay_days=180.0,
                                  bandwidth_km=50.0,
                                  grid_size=50,
                                  top_quantile=0.90,
                                  min_train_periods=3,
                                  bootstrap_samples=500,
                                  random_state=42):

    rng = np.random.default_rng(random_state)
    periods = sorted(df["period"].unique())

    hit_rates = []
    all_test_hits = []

    for i in range(min_train_periods, len(periods)):
        train_df = df[df["period"].isin(periods[:i])]
        test_df = df[df["period"] == periods[i]]

        if train_df.empty or test_df.empty:
            continue

        model = fit_spatiotemporal_model(
            train_df,
            time_decay_days,
            bandwidth_km
        )

        # 用网格定义高风险区域
        mid_date = test_df["detection_date"].median()
        grid_lats, grid_lons = make_grid(train_df, grid_size)
        grid_dates = np.full(len(grid_lats), mid_date)

        grid_risk = predict_risk(model, grid_dates, grid_lats, grid_lons)
        threshold = np.quantile(grid_risk, top_quantile)

        test_risk = predict_risk(
            model,
            test_df["detection_date"],
            test_df["latitude"],
            test_df["longitude"]
        )

        hits = (test_risk >= threshold).astype(int)
        hit_rates.append(hits.mean())
        all_test_hits.extend(hits.tolist())

    avg_hit_rate = float(np.mean(hit_rates)) if hit_rates else np.nan

    # Bootstrap CI
    ci_low, ci_high = np.nan, np.nan
    if len(all_test_hits) >= 2:
        boot = [
            np.mean(rng.choice(all_test_hits,
                               size=len(all_test_hits),
                               replace=True))
            for _ in range(bootstrap_samples)
        ]
        ci_low, ci_high = np.quantile(boot, [0.025, 0.975])

    return {
        "avg_hit_rate": avg_hit_rate,
        "hit_rate_ci_95": (float(ci_low), float(ci_high)),
        "num_splits": len(hit_rates),
        "num_test_points": len(all_test_hits)
    }


# ===============================
# 主程序
# ===============================
if __name__ == "__main__":
    df = load_positive_cases()
    result = evaluate_prediction_precision(df)

    print("T1 Evaluation Result")
    for k, v in result.items():
        print(f"{k}: {v}")
