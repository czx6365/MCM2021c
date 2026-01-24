import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from dataclasses import dataclass

# ===============================
# 配置与路径
# ===============================
DATASET_PATH = r"data/raw/2021MCMProblemC_DataSet.xlsx"
OUTPUT_DIR = r"data/processed"
GRAPHIC_DIR = r"graphic/1.22 1.0版本画图/T1"

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(GRAPHIC_DIR, exist_ok=True)


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
# 核心算法：Haversine 距离
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
# 模型拟合
# ===============================
def fit_spatiotemporal_model(df,
                             time_decay_days=180.0,
                             bandwidth_km=50.0):
    """
    拟合时空核密度估计模型（非参数模型，仅存储数据）
    """
    return SpatiotemporalModel(
        train_df=df.reset_index(drop=True),
        time_decay_days=float(time_decay_days),
        bandwidth_km=float(bandwidth_km)
    )


# ===============================
# 风险预测（核心公式）
# ===============================
def predict_risk(model, query_dates, query_lats, query_lons):
    train = model.train_df

    t_train = train["detection_date"].values.astype("datetime64[D]")
    lat_train = train["latitude"].to_numpy()
    lon_train = train["longitude"].to_numpy()

    q_dates = pd.to_datetime(query_dates).values.astype("datetime64[D]")
    q_lats = np.asarray(query_lats, dtype=float)
    q_lons = np.asarray(query_lons, dtype=float)

    # 1. 时间权重：单向指数衰减 (Past -> Future)
    dt_days = (q_dates[:, None] - t_train[None, :]).astype("timedelta64[D]").astype(int)
    dt_days = np.maximum(dt_days, 0)
    time_weight = np.exp(-dt_days / model.time_decay_days)

    # 2. 空间权重：高斯核 (Gaussian Kernel)
    dist_km = haversine_km(
        q_lats[:, None], q_lons[:, None],
        lat_train[None, :], lon_train[None, :]
    )
    space_weight = np.exp(-(dist_km ** 2) / (2 * model.bandwidth_km ** 2))

    # 3. 综合风险密度
    return (time_weight * space_weight).sum(axis=1)


# ===============================
# 辅助工具：生成网格
# ===============================
def make_grid(df, grid_size=50):
    lat_min, lat_max = df["latitude"].min(), df["latitude"].max()
    lon_min, lon_max = df["longitude"].min(), df["longitude"].max()
    
    # 向外扩展 10% 边界以防止边缘效应
    lat_w = lat_max - lat_min
    lon_w = lon_max - lon_min
    lat_min -= 0.1 * lat_w; lat_max += 0.1 * lat_w
    lon_min -= 0.1 * lon_w; lon_max += 0.1 * lon_w

    lats = np.linspace(lat_min, lat_max, grid_size)
    lons = np.linspace(lon_min, lon_max, grid_size)

    grid_lats, grid_lons = np.meshgrid(lats, lons, indexing="ij")
    return grid_lats.ravel(), grid_lons.ravel()


# ===============================
# 统计工具：区组自助法 (Block Bootstrap)
# ===============================
def block_bootstrap_hit_rate(all_hits, block_size=5, n_boot=1000, random_state=42):
    """
    针对时间序列相关性的区块自助法，计算置信区间
    """
    rng = np.random.default_rng(random_state)
    if len(all_hits) < block_size:
        return np.nan, np.nan
        
    # 构建块
    blocks = [all_hits[i:i+block_size] for i in range(0, len(all_hits)-block_size+1)]
    
    boot_means = []
    # 每次重采样生成的序列长度约为原始长度
    n_blocks_needed = max(1, len(all_hits) // block_size)
    
    for _ in range(n_boot):
        sample = rng.choice(blocks, size=n_blocks_needed, replace=True)
        # 展平
        sample_flat = [item for sublist in sample for item in sublist]
        boot_means.append(np.mean(sample_flat))
        
    return np.quantile(boot_means, 0.025), np.quantile(boot_means, 0.975)


# ===============================
# T1 任务：时序精度评估 (Precision over Time)
# ===============================
def evaluate_time_series_precision(df,
                                  time_decay_days=180.0,
                                  bandwidth_km=50.0,
                                  grid_size=50,
                                  top_quantile=0.90,
                                  min_train_periods=3):
    """
    按时间滚动评估模型精度，用于回答 "predicted... with what level of precision"
    """
    print("\nRunning T1 Time Series Evaluation...")
    periods = sorted(df["period"].unique())
    
    rows = []
    all_hits_pool = []
    
    for i in range(min_train_periods, len(periods)):
        current_period = periods[i]
        
        # 滚动窗口划分：使用截止到上个月的数据训练，预测本月
        train_df = df[df["period"].isin(periods[:i])]
        test_df = df[df["period"] == current_period]

        if train_df.empty or test_df.empty:
            continue

        model = fit_spatiotemporal_model(train_df, time_decay_days, bandwidth_km)

        # 1. 确定动态阈值（根据历史分布）
        mid_date = test_df["detection_date"].median()
        grid_lats, grid_lons = make_grid(train_df, grid_size)
        grid_dates = np.full(len(grid_lats), mid_date)

        grid_risk = predict_risk(model, grid_dates, grid_lats, grid_lons)
        threshold = np.quantile(grid_risk, top_quantile)

        # 2. 预测测试集
        test_risk = predict_risk(
            model,
            test_df["detection_date"],
            test_df["latitude"],
            test_df["longitude"]
        )

        # 3. 计算命中情况
        hits = (test_risk >= threshold).astype(int)
        hit_rate = hits.mean()
        
        rows.append({
            "period": str(current_period),
            "hit_rate": hit_rate,
            "n_test": len(hits),
            "threshold": threshold
        })
        all_hits_pool.extend(hits.tolist())

    ts_df = pd.DataFrame(rows)
    
    # 保存结果
    ts_df.to_csv(os.path.join(OUTPUT_DIR, "t1_seasonal_trend_data.csv"), index=False)
    
    # 总体统计
    avg_hit_rate = np.mean(all_hits_pool)
    ci_low, ci_high = block_bootstrap_hit_rate(all_hits_pool)
    
    print(f"Overall Hit Rate: {avg_hit_rate:.4f} (95% CI: [{ci_low:.4f}, {ci_high:.4f}])")
    
    # 绘图
    plt.figure(figsize=(10, 5))
    plt.plot(ts_df["period"], ts_df["hit_rate"], marker='o', linestyle='-', label="Monthly Accuracy")
    plt.axhline(avg_hit_rate, color='r', linestyle='--', label=f"Mean: {avg_hit_rate:.2f}")
    plt.xticks(rotation=45)
    plt.ylabel("Prediction Hit Rate (Top 10% Risk)")
    plt.title("Model Prediction Precision Over Time")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPHIC_DIR, "t1_precision_over_time.png"), dpi=300)
    plt.close()
    
    return ts_df


# ===============================
# T5 任务：根除判定证据 (Evidence of Eradication)
# ===============================
def analyze_eradication_criteria(df, 
                               time_decay_days=180.0, 
                               bandwidth_km=50.0, 
                               grid_size=80, 
                               threshold=1e-5, 
                               sustain_periods=3):
    """
    计算全州最大风险值的演变，判定是否根除。
    Args:
        threshold: 根除阈值（当全州最大风险密度低于此值时）
        sustain_periods: 需要持续多少个周期低于阈值才确信
    """
    print("\nRunning T5 Eradication Analysis...")
    periods = sorted(df["period"].unique())
    
    # 为了模拟直到 2021 年甚至未来的情况，我们可以外推几个周期
    # 这里演示仅使用现有数据时段的演变
    
    max_risks = []
    dates = []
    
    for i, p in enumerate(periods):
        # 假设当前时间点是 p 的月末
        current_date = p.to_timestamp(freq='M') + pd.Timedelta(days=1)
        
        train_df = df[df["period"] <= p]
        if train_df.empty:
            continue
            
        model = fit_spatiotemporal_model(train_df, time_decay_days, bandwidth_km)
        
        # 网格覆盖全州
        grid_lats, grid_lons = make_grid(train_df, grid_size)
        query_dates = np.full(len(grid_lats), current_date)
        
        # 计算全州风险表面
        risk_surface = predict_risk(model, query_dates, grid_lats, grid_lons)
        
        # 取最大值作为 "Current Peak Infestation Level"
        current_max_risk = np.max(risk_surface)
        
        max_risks.append(current_max_risk)
        dates.append(str(p))

    # 构建结果表
    erad_df = pd.DataFrame({
        "period": dates,
        "max_risk_intensity": max_risks
    })
    
    # 判定逻辑
    erad_df["below_threshold"] = erad_df["max_risk_intensity"] < threshold
    
    # 计算连续低于阈值的期数 (Run-length encoding style)
    erad_df["consecutive_below"] = 0
    run = 0
    for idx, row in erad_df.iterrows():
        if row["below_threshold"]:
            run += 1
        else:
            run = 0
        erad_df.at[idx, "consecutive_below"] = run
        
    erad_df["is_eradicated"] = erad_df["consecutive_below"] >= sustain_periods
    
    # 保存结果
    erad_df.to_csv(os.path.join(OUTPUT_DIR, "t5_eradication_evidence.csv"), index=False)
    
    # 绘图：根除证据曲线
    plt.figure(figsize=(10, 5))
    plt.plot(erad_df["period"], erad_df["max_risk_intensity"], marker='s', color='purple', label="Max Risk Intensity")
    plt.axhline(threshold, color='green', linestyle='--', linewidth=2, label=f"Eradication Threshold ({threshold})")
    plt.yscale("log") # 风险值通常跨度很大，用对数坐标
    plt.xticks(rotation=45)
    plt.ylabel("Statewide Max Risk (Log Scale)")
    plt.title("Evidence of Eradication: Peak Risk Intensity Over Time")
    plt.legend()
    plt.grid(True, which="both", linestyle='--', alpha=0.3)
    
    # 标记是否达成根除
    if erad_df["is_eradicated"].any():
        first_erad = erad_df[erad_df["is_eradicated"]].iloc[0]
        plt.annotate("Possible Eradication", 
                     xy=(first_erad["period"], threshold), 
                     xytext=(first_erad["period"], threshold * 10),
                     arrowprops=dict(facecolor='green', shrink=0.05))
        print(f"Eradication criteria met at period: {first_erad['period']}")
    else:
        print("Eradication criteria NOT met in the current dataset.")

    plt.tight_layout()
    plt.savefig(os.path.join(GRAPHIC_DIR, "t5_eradication_curve.png"), dpi=300)
    plt.close()
    
    return erad_df


# ===============================
# 辅助函数：单次HIT RATE评估 (用于敏感性分析快速调用)
# ===============================
def evaluate_prediction_precision(df,
                                  time_decay_days=180.0,
                                  bandwidth_km=50.0,
                                  grid_size=50,
                                  top_quantile=0.90,
                                  min_train_periods=3,
                                  bootstrap_samples=100,  # 默认值
                                  random_state=42):
    """
    计算给定参数下的平均击中率 (Average Hit Rate)，用于参数网格搜索。
    这是原函数的简化版，只返回一个核心指标。
    """
    periods = sorted(df["period"].unique())
    hit_rates = []

    for i in range(min_train_periods, len(periods)):
        train_df = df[df["period"].isin(periods[:i])]
        test_df = df[df["period"] == periods[i]]

        if train_df.empty or test_df.empty:
            continue

        model = fit_spatiotemporal_model(train_df, time_decay_days, bandwidth_km)

        # 确定动态阈值
        mid_date = test_df["detection_date"].median()
        grid_lats, grid_lons = make_grid(train_df, grid_size)
        grid_dates = np.full(len(grid_lats), mid_date)

        grid_risk = predict_risk(model, grid_dates, grid_lats, grid_lons)
        threshold = np.quantile(grid_risk, top_quantile)

        # 预测与评估
        test_risk = predict_risk(
            model,
            test_df["detection_date"],
            test_df["latitude"],
            test_df["longitude"]
        )

        hits = (test_risk >= threshold).astype(int)
        hit_rates.append(hits.mean())

    avg_hit_rate = float(np.mean(hit_rates)) if hit_rates else 0.0
    return {"avg_hit_rate": avg_hit_rate}


# ===============================
# 敏感性分析 (Sensitivty Analysis)
# ===============================
def analyze_parameter_sensitivity(df):
    """
    网格搜索最优参数并绘制热力图
    """
    print("\nRunning Parameter Sensitivity Analysis...")
    decay_candidates = [90, 180, 270, 360]
    bw_candidates = [30, 50, 70, 100]
    
    rows = []
    
    for d in decay_candidates:
        for b in bw_candidates:
            # 简化评估（减少 bootstrap 采样以加速）
            res = evaluate_prediction_precision(
                df, time_decay_days=d, bandwidth_km=b, bootstrap_samples=10
            )
            rows.append({
                "decay": d,
                "bandwidth": b,
                "hit_rate": res["avg_hit_rate"]
            })
            
    res_df = pd.DataFrame(rows)
    # 转换为矩阵形式
    pivot_table = res_df.pivot(index="decay", columns="bandwidth", values="hit_rate")
    
    # 绘图
    plt.figure(figsize=(8, 6))
    plt.imshow(pivot_table.values, cmap="viridis", origin="lower", aspect="auto")
    plt.colorbar(label="Average Hit Rate")
    
    # 轴标签
    plt.xticks(range(len(bw_candidates)), bw_candidates)
    plt.yticks(range(len(decay_candidates)), decay_candidates)
    plt.xlabel("Bandwidth (km)")
    plt.ylabel("Time Decay (days)")
    plt.title("Sensitivity Analysis: T1 Model Parameters")
    
    # 在格子里标数值
    for i in range(len(decay_candidates)):
        for j in range(len(bw_candidates)):
            val = pivot_table.values[i, j]
            plt.text(j, i, f"{val:.3f}", ha="center", va="center", color="white")
            
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPHIC_DIR, "t1_param_sensitivity.png"), dpi=300)
    plt.close()
    print("Sensitivity heatmap saved.")
    
    return res_df


# ===============================
# 主程序
# ===============================
def main():
    print("Loading data...")
    df = load_positive_cases()
    print(f"Loaded {len(df)} positive cases.")

    # 1. 参数敏感性分析（确立模型参数选择的合理性）
    analyze_parameter_sensitivity(df)
    
    # 2. 时序精度评估（回答 Accuracy 问题）
    # 使用上一步的经验最优参数 (假设 180, 50 是合理的，或根据上一步结果自动选)
    evaluate_time_series_precision(df, time_decay_days=180.0, bandwidth_km=50.0)

    # 3. 根除判定（T5）
    # 设定一个极小的阈值，例如万分之一的风险密度
    analyze_eradication_criteria(df, threshold=1e-5, sustain_periods=3)

    print("\nAll T1/T5 tasks completed successfully.")

# ===============================
# T5 任务：根除判定证据 (Advanced Population Dynamics Model)
# ===============================
def analyze_eradication_criteria_advanced(positive_df, 
                                        all_reports_path=r"data/raw/2021MCMProblemC_DataSet.xlsx",
                                        model_accuracy_theta=0.95):
    """
    基于种群增长率(gt)、分布范围比(dt)和阳性概率比(pt)的综合评估系数 K
    以及贝叶斯修正。
    
    Formula: K_t = g_t * d_t * p_t (Corrected by Theta)
    Eradication evidence: K_t converges to 0.
    """
    print("\nRunning T5 Advanced Eradication Analysis (Dynamic Indicators)...")
    
    # 1. 加载全量报告数据（用于计算 pt）
    raw_df = pd.read_excel(all_reports_path)
    raw_df["year"] = pd.to_datetime(raw_df["Detection Date"], errors="coerce").dt.year
    raw_df = raw_df.dropna(subset=["year"])
    
    # 2. 准备阳性数据（用于 calculation gt, dt）
    pos_df = positive_df.copy()
    pos_df["year"] = pos_df["detection_date"].dt.year
    
    years = sorted(pos_df["year"].unique())
    # 确保至少有两年数据才能计算比率
    if len(years) < 2:
        print("Not enough years of data to calculate yearly growth ratios.")
        return None
        
    metrics = []
    
    # 辅助：计算年内平均分布距离 (Distribution Range)
    def calc_distribution_range(year_df):
        coords = year_df[["latitude", "longitude"]].to_numpy()
        n = len(coords)
        if n < 2: return 1.0 # 单点无距离，设默认值防止除零
        
        sum_sq_dist = 0.0
        count = 0
        # 计算所有点对距离平方和
        for i in range(n):
            for j in range(i+1, n):
                d = haversine_km(coords[i,0], coords[i,1], coords[j,0], coords[j,1])
                sum_sq_dist += d**2
                count += 1
        return sum_sq_dist / count if count > 0 else 0.0

    # 逐年计算指标 t vs t-1
    for i in range(1, len(years)):
        t_curr = years[i]
        t_prev = years[i-1]
        
        # 数据切片
        pos_curr = pos_df[pos_df["year"] == t_curr]
        pos_prev = pos_df[pos_df["year"] == t_prev]
        
        raw_curr = raw_df[raw_df["year"] == t_curr]
        raw_prev = raw_df[raw_df["year"] == t_prev]
        
        # --- (1) Growth Ratio gt = Nt / Nt-1 ---
        # 引入贝叶斯修正：实际阳性数 ~ 观测阳性数 * theta (粗略近似)
        # 这里假设观测数本身已经经过Lab核实，故直接用数量比，或者乘theta作为修正因子
        # 按照题目公式 (19)，主要想修正 K，这里先算出原始 K
        
        N_t = len(pos_curr)
        N_prev = len(pos_prev)
        
        gt = N_t / N_prev if N_prev > 0 else N_t  # 避免除零
        
        # --- (2) Distribution Range Ratio dt ---
        # 注意：题目公式 (17) 是 Prev / Curr (反比，范围越大意味越稀疏或扩散? 
        # 文中写 "distribution range ... will increase as number grows", 同时 dt 定义是 t-1 在分子
        # 这意味着如果 Range t > Range t-1 (扩散)，dt 会 < 1 ?
        # 让我们仔细阅读：Ratio of current average to average of previous year... 
        # 但公式(17)分母是 Nt (Current)，分子是 Nt-1 (Previous)。
        # 如果范围扩散(变大)，分母变大，dt 变小。这作为一个衰退指标(K减小)似乎有点反直觉（扩散通常不好）。
        # 但原文说 "K converges to 0 -> Eradicated"。 
        # 如果种群变小(gt<1)且收缩，我们希望 K 变小。
        # 此时若收缩，Range变小，分母变小，dt变大。
        # 可能原文 dt 的定义是 "聚集度"？或者是笔误？我们将严格按照公式(17)编写: Prev / Curr
        
        avg_dist_sq_curr = calc_distribution_range(pos_curr)
        avg_dist_sq_prev = calc_distribution_range(pos_prev)
        
        dt = avg_dist_sq_prev / avg_dist_sq_curr if avg_dist_sq_curr > 1e-6 else 1.0
        
        # --- (3) Positive Probability Ratio pt ---
        # pt = (Pos_t / All_t) / (Pos_t-1 / All_t-1)
        
        rate_curr = len(pos_curr) / len(raw_curr) if len(raw_curr) > 0 else 0
        rate_prev = len(pos_prev) / len(raw_prev) if len(raw_prev) > 0 else 0
        
        pt = rate_curr / rate_prev if rate_prev > 0 else 1.0
        
        # --- (4) Evaluation Coefficient Kt ---
        Kt_raw = gt * dt * pt
        
        # --- (5) Bayesian Correction (Simplified) ---
        # 公式 (19) 形式 Pr(K|theta) somewhat complex without defined Priors.
        # 我们采用更直观的修正：由于存在误报被剔除，我们对 Nt 的置信度很高(Lab Confirmed)。
        # 但既然题目特别提到了 Bayesian inference with accuracy theta:
        # 我们可以简单地认为修正后的 Kt_adj = Kt_raw * theta (保守估计)
        # 或者 interpret theta as probability evidence is real.
        
        Kt_final = Kt_raw * model_accuracy_theta 
        
        metrics.append({
            "year": t_curr,
            "N_positive": N_t,
            "gt (Growth)": gt,
            "dt (Range_Inv)": dt,
            "pt (Rate_Ratio)": pt,
            "Kt (Evaluation)": Kt_final,
            "Status": "Decreasing" if Kt_final < 1.0 else "Increasing"
        })
        
    res_df = pd.DataFrame(metrics)
    
    # 保存与绘图
    output_path = os.path.join(OUTPUT_DIR, "t5_advanced_eradication_metrics.csv")
    res_df.to_csv(output_path, index=False)
    
    print("\nCalculated Dynamics Metrics:")
    print(res_df[["year", "Kt (Evaluation)", "Status"]])
    
    if not res_df.empty:
        plt.figure(figsize=(10, 6))
        
        # 双轴图：左轴 Kt，右轴 确诊数
        ax1 = plt.gca()
        line1 = ax1.plot(res_df["year"], res_df["Kt (Evaluation)"], 'o-', color='tab:red', label='Evaluation Coeff K', linewidth=2)
        ax1.axhline(1.0, color='gray', linestyle='--', alpha=0.5, label='Stable Threshold (K=1)')
        ax1.set_xlabel("Year")
        ax1.set_ylabel("Evaluation Model Coefficient K")
        ax1.set_xticks(years[1:])
        
        ax2 = ax1.twinx()
        line2 = ax2.bar(res_df["year"], res_df["N_positive"], alpha=0.3, color='tab:blue', label='Positive Cases', width=0.4)
        ax2.set_ylabel("Number of Positive Cases")
        
        # 合并图例
        lns = line1 + [line2]
        labs = [l.get_label() for l in lns]
        ax1.legend(lns, labs, loc='upper left')
        
        plt.title("Population Dynamics & Eradication Evaluation (K Indicator)")
        plt.tight_layout()
        plt.savefig(os.path.join(GRAPHIC_DIR, "t5_dynamics_K_indicator.png"), dpi=300)
        plt.close()
        
    return res_df


def main():
    print("Loading data...")
    df = load_positive_cases()
    print(f"Loaded {len(df)} positive cases.")

    # 1. 参数敏感性分析
    analyze_parameter_sensitivity(df)
    
    # 2. 时序精度评估
    evaluate_time_series_precision(df, time_decay_days=180.0, bandwidth_km=50.0)

    # 3. 根除判定（T5 Advanced）
    # 设定 theta = 0.95 (假设图像识别与人工复核的综合准确率很高)
    analyze_eradication_criteria_advanced(df, model_accuracy_theta=0.95)

    print("\nAll T1/T5 tasks completed successfully.")

if __name__ == "__main__":

    main()
