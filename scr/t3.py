import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

# =========================
# 参数设置
# =========================

# T2 输出结果路径
T2_OUTPUT_PATH = r"data\\processed\\outputs_t2_ranked_reports.csv"

# T3 输出文件
OUTPUT_TABLE_PATH = r"data\\processed\\t3_region_priority_table.csv"
OUTPUT_FIG_PATH = r"data\\processed\\t3_region_clusters.png"

# --------- 关键建模参数 ---------

# Level 1：粗聚类（热点区）数量范围
LEVEL1_K_RANGE = range(8, 21)

# Level 2：政府一次行动可接受的最大直径（公里）
MAX_REGION_DIAMETER_KM = 200.0

# 区域优先级中“有效信息量”的上限（避免概率饱和）
K_EFFECTIVE = 10

# Level 2 最大允许的细分聚类数
MAX_K2 = 30


# =========================
# 地理工具函数
# =========================

def haversine_km(lat1, lon1, lat2, lon2):
    """
    使用 Haversine 公式计算地球表面两点之间的大圆距离（公里）
    """
    r = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def max_intra_cluster_distance(coords):
    """
    计算同一聚类（区域）内任意两点的最大地理距离（公里）
    用于判断是否满足“单次政府行动可覆盖”的现实约束
    """
    if len(coords) < 2:
        return 0.0
    max_d = 0.0
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            d = haversine_km(
                coords[i, 0], coords[i, 1],
                coords[j, 0], coords[j, 1]
            )
            max_d = max(max_d, d)
    return max_d


def latlon_to_xy_km(coords):
    """
    将经纬度近似转换为平面坐标（单位：公里）
    用于聚类算法中避免直接使用经纬度欧氏距离的几何失真
    """
    lat = coords[:, 0]
    lon = coords[:, 1]

    lat0 = np.mean(lat)
    lon0 = np.mean(lon)

    x = (lon - lon0) * 111.32 * np.cos(np.radians(lat0))
    y = (lat - lat0) * 110.57
    return np.column_stack([y, x])


# =========================
# Step 1：读取并筛选数据
# =========================

df = pd.read_csv(T2_OUTPUT_PATH)

# 只保留未核实 / 未处理的报告（T3 决策对象）
df_t3 = df[df["Lab Status"].isin(["Unverified", "Unprocessed"])].copy()
df_t3 = df_t3.dropna(subset=["Latitude", "Longitude", "prob_positive"])

coords_latlon = df_t3[["Latitude", "Longitude"]].to_numpy()
coords_xy = latlon_to_xy_km(coords_latlon)

print(f"T3 reports count: {len(df_t3)}")

# =========================
# Step 2：Level 1 粗聚类 —— 空间热点识别
# =========================

best_k1 = None
best_score = -1
best_labels_l1 = None

print("\nLevel 1 clustering (hotspot detection):")

for k in LEVEL1_K_RANGE:
    model = AgglomerativeClustering(
        n_clusters=k,
        linkage="ward"
    )
    labels = model.fit_predict(coords_xy)

    # 采样计算轮廓系数，避免 O(n^2) 内存占用
    sample_size = min(1000, len(coords_xy))
    sil = silhouette_score(coords_xy, labels, sample_size=sample_size, random_state=42)
    print(f"k={k}, silhouette={sil:.3f}")

    if sil > best_score:
        best_score = sil
        best_k1 = k
        best_labels_l1 = labels

print(f"\nSelected Level-1 cluster number: {best_k1}")
df_t3["cluster_l1"] = best_labels_l1

# =========================
# Step 3：Level 2 细聚类 —— 行动区域划分
# =========================

final_cluster_labels = np.full(len(df_t3), -1)
cluster_id = 0

print("\nLevel 2 clustering (actionable regions):")

for l1 in sorted(df_t3["cluster_l1"].unique()):
    idx = df_t3["cluster_l1"] == l1
    sub_coords_latlon = coords_latlon[idx]
    sub_coords_xy = coords_xy[idx]

    # 如果该热点区本身就满足行动直径约束，直接作为一个区域
    if max_intra_cluster_distance(sub_coords_latlon) <= MAX_REGION_DIAMETER_KM:
        final_cluster_labels[idx.to_numpy()] = cluster_id
        print(f"Hotspot {l1}: single region (diameter OK)")
        cluster_id += 1
        continue

    # 否则在该热点内部进一步细分
    found = False
    sub_idx = np.where(idx.to_numpy())[0]

    for k2 in range(2, min(MAX_K2, len(sub_coords_xy))):
        model = AgglomerativeClustering(
            n_clusters=k2,
            linkage="complete"
        )
        labels_l2 = model.fit_predict(sub_coords_xy)

        ok = True
        for c in range(k2):
            if max_intra_cluster_distance(sub_coords_latlon[labels_l2 == c]) > MAX_REGION_DIAMETER_KM:
                ok = False
                break

        if ok:
            print(f"Hotspot {l1}: split into {k2} regions")
            for c in range(k2):
                final_cluster_labels[sub_idx[labels_l2 == c]] = cluster_id
                cluster_id += 1
            found = True
            break

    # 如果仍无法满足，作为“分散风险区”整体处理
    if not found:
        final_cluster_labels[idx.to_numpy()] = cluster_id
        print(f"Hotspot {l1}: fallback as dispersed region")
        cluster_id += 1

df_t3["cluster"] = final_cluster_labels
num_regions = df_t3["cluster"].nunique()

print(f"\nTotal actionable regions: {num_regions}")

# =========================
# Step 4：计算区域优先级（Top-K 有效概率）
# =========================

region_rows = []

for c in sorted(df_t3["cluster"].unique()):
    region_df = df_t3[df_t3["cluster"] == c]

    # 只取区域内概率最高的前 K 条报告，避免并集概率饱和
    probs = region_df["prob_positive"].to_numpy()
    top_probs = np.sort(probs)[-K_EFFECTIVE:] if len(probs) > K_EFFECTIVE else probs

    region_priority = 1.0 - np.prod(1.0 - top_probs)

    center_lat = region_df["Latitude"].mean()
    center_lon = region_df["Longitude"].mean()
    max_dist = max_intra_cluster_distance(
        region_df[["Latitude", "Longitude"]].to_numpy()
    )

    region_rows.append({
        "cluster": c,
        "center_latitude": center_lat,
        "center_longitude": center_lon,
        "num_reports": len(region_df),
        "priority": region_priority,
        "max_diameter_km": max_dist,
    })

region_table = pd.DataFrame(region_rows)
region_table = region_table.sort_values("priority", ascending=False)
region_table["priority_rank"] = np.arange(1, len(region_table) + 1)

# =========================
# Step 5：保存结果
# =========================

os.makedirs(os.path.dirname(OUTPUT_TABLE_PATH), exist_ok=True)
region_table.to_csv(OUTPUT_TABLE_PATH, index=False, encoding="utf-8")

print("\nTop regions by priority:")
print(region_table.head())

# =========================
# Step 6：空间可视化
# =========================

plt.figure(figsize=(8, 6))
plt.scatter(
    df_t3["Longitude"],
    df_t3["Latitude"],
    c=df_t3["cluster"],
    cmap="tab20",
    s=10,
    alpha=0.7
)

plt.scatter(
    region_table["center_longitude"],
    region_table["center_latitude"],
    c="black",
    marker="x",
    s=80,
    label="Region center"
)

plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("T3 Two-stage Spatial Clustering and Region Priority")
plt.legend()
plt.tight_layout()

plt.savefig(OUTPUT_FIG_PATH, dpi=300)
plt.close()

print(f"\nSaved region table to: {OUTPUT_TABLE_PATH}")
print(f"Saved clustering figure to: {OUTPUT_FIG_PATH}")
