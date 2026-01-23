# =========================
# T4：Optimal Updating Strategy
# Based on Numerical Derivative
# =========================

import argparse
import os
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

# =========================
# 一、全局配置
# =========================

# T2 输出文件（包含模型预测概率、真实标签、时间信息）
DEFAULT_INPUT = r"data\\processed\\outputs_t2_ranked_reports.csv"

# T4 数值结果输出目录
DEFAULT_OUT_DIR = r"data\\processed"

# T4 图像输出目录
DEFAULT_FIG_DIR = r"graphic\\t4"

# 固定随机种子，保证结果可复现
RANDOM_STATE = 42


# =========================
# 二、时间窗口数据结构
# =========================
@dataclass
class Window:
    """
    表示一个时间窗口：
    - start: 窗口起始时间
    - end:   窗口结束时间
    - idx:   属于该时间窗口的数据行索引
    """
    start: pd.Timestamp
    end: pd.Timestamp
    idx: pd.Index


# =========================
# 三、构建时间窗口
# =========================
def build_time_windows(
    df: pd.DataFrame,
    date_col: str,
    window_days: int = 30
) -> List[Window]:
    """
    将数据按时间划分为连续的固定长度时间窗口（rolling windows）

    Parameters
    ----------
    df : DataFrame
        原始数据
    date_col : str
        时间列名（Submission Date / Detection Date）
    window_days : int
        每个时间窗口的天数（如 30 天）

    Returns
    -------
    windows : List[Window]
        按时间顺序排列的时间窗口列表
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])

    if df.empty:
        return []

    # 获取整体时间范围
    min_date = df[date_col].min().normalize()
    max_date = df[date_col].max().normalize()

    windows: List[Window] = []
    start = min_date

    # 逐段构建窗口
    while start <= max_date:
        end = start + pd.Timedelta(days=window_days)
        mask = (df[date_col] >= start) & (df[date_col] < end)
        idx = df[mask].index
        windows.append(Window(start=start, end=end, idx=idx))
        start = end

    return windows


# =========================
# 四、特征选择（沿用 T2）
# =========================
def _select_feature_cols(df: pd.DataFrame) -> List[str]:
    """
    从数据中选择可用于 Logistic Regression 的特征列
    （与 T2 模型保持一致）
    """
    candidates = [
        "prob_img",          # 图像模型输出
        "risk_prior_log",    # 风险先验（对数）
        "risk_prior",        # 风险先验
        "Latitude",          # 纬度
        "Longitude",         # 经度
        "has_image",         # 是否有图片
        "num_images",        # 图片数量
    ]
    return [c for c in candidates if c in df.columns]


# =========================
# 五、模型训练与预测
# =========================
def _fit_model(
    train_df: pd.DataFrame,
    feature_cols: List[str]
) -> LogisticRegression:
    """
    使用 Logistic Regression 训练 T2 分类模型
    """
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",  # 处理类别极度不平衡
        random_state=RANDOM_STATE,
    )
    model.fit(train_df[feature_cols].fillna(0.0), train_df["y"])
    return model


def _predict_prob(
    model: LogisticRegression,
    df: pd.DataFrame,
    feature_cols: List[str]
) -> np.ndarray:
    """
    输出样本为 Positive ID 的预测概率
    """
    return model.predict_proba(df[feature_cols].fillna(0.0))[:, 1]


# =========================
# 六、核心：模拟“不同更新频率 Δ”的性能
# =========================
def train_and_evaluate(
    df: pd.DataFrame,
    windows: List[Window],
    feature_cols: List[str],
    update_interval: int = 1,
) -> pd.DataFrame:
    """
    模拟：模型每 update_interval 个时间窗口更新一次

    在每个时间窗口：
    - 用历史所有窗口训练（或复用旧模型）
    - 在当前窗口测试
    - 记录性能指标

    Returns
    -------
    DataFrame：
        每个时间窗口对应的 AUROC / AUPRC / Top10% Positive Rate
    """
    records = []

    if len(windows) < 2:
        return pd.DataFrame()

    model = None
    last_fit_at = None

    # 从第二个窗口开始（第一个没有历史数据）
    for i in range(1, len(windows)):

        # 构造训练集（历史窗口）
        train_idx = pd.Index([])
        for w in windows[:i]:
            train_idx = train_idx.union(w.idx)

        test_idx = windows[i].idx

        train_df = df.loc[train_idx].copy()
        test_df = df.loc[test_idx].copy()

        if train_df.empty or test_df.empty:
            continue
        if train_df["y"].nunique() < 2:
            continue

        # 是否需要重新训练模型
        if (last_fit_at is None) or ((i - last_fit_at) >= update_interval):
            model = _fit_model(train_df, feature_cols)
            last_fit_at = i

        # 模型预测
        prob = _predict_prob(model, test_df, feature_cols)
        y_true = test_df["y"].to_numpy()

        # 计算 AUROC / AUPRC
        auroc = np.nan
        auprc = np.nan
        if len(np.unique(y_true)) >= 2:
            auroc = roc_auc_score(y_true, prob)
            auprc = average_precision_score(y_true, prob)

        # Top 10% 报告中的真实阳性率（政策效用指标）
        n_top = max(1, int(len(prob) * 0.10))
        top_idx = np.argsort(prob)[-n_top:]
        top_pos_rate = float((y_true[top_idx] == 1).mean())

        records.append(
            {
                "window_start": windows[i].start,
                "window_end": windows[i].end,
                "auroc": auroc,
                "auprc": auprc,
                "top10_pos_rate": top_pos_rate,
            }
        )

    return pd.DataFrame(records)


# =========================
# 七、数值导数（一阶有限差分）
# =========================
def compute_numerical_derivative(series: pd.Series) -> pd.Series:
    """
    用一阶差分近似数值导数：
        D(t) = I(t) - I(t-1)
    """
    return series.diff()


# =========================
# 八、选择最优更新间隔 Δ*
# =========================
def select_optimal_update_interval(
    results_by_delta: Dict[int, pd.DataFrame],
    metric: str = "auroc",
    epsilon: float = 0.02,
    increase_ratio: float = 1.5,
) -> Dict[str, object]:
    """
    结合两种准则选择最优更新频率：

    1）epsilon 准则：性能下降不超过阈值
    2）导数突变准则：|D(t)| 明显增大

    Returns
    -------
    dict : 推荐更新间隔及依据
    """
    deltas = sorted(results_by_delta.keys())
    if not deltas:
        return {"delta_star": None}

    # 以最频繁更新（delta 最小）作为 baseline
    baseline = results_by_delta[deltas[0]]
    baseline_series = baseline.set_index("window_end")[metric]

    best_delta = deltas[0]

    # epsilon 判据
    for d in deltas[1:]:
        df = results_by_delta[d]
        series = df.set_index("window_end")[metric]
        aligned = pd.concat(
            [baseline_series, series],
            axis=1,
            keys=["base", "cand"]
        ).dropna()

        if aligned.empty:
            continue

        max_drop = (aligned["base"] - aligned["cand"]).max()
        if max_drop <= epsilon:
            best_delta = d
        else:
            break

    # 数值导数突变判据
    max_abs_deriv = []
    for d in deltas:
        df = results_by_delta[d]
        if df.empty:
            max_abs_deriv.append(np.nan)
            continue
        max_abs_deriv.append(
            compute_numerical_derivative(df[metric]).abs().max()
        )

    deriv_delta = None
    for i in range(1, len(deltas)):
        if (
            np.isfinite(max_abs_deriv[i - 1])
            and np.isfinite(max_abs_deriv[i])
            and max_abs_deriv[i] > max_abs_deriv[i - 1] * increase_ratio
        ):
            deriv_delta = deltas[i - 1]
            break

    if deriv_delta is None:
        deriv_delta = deltas[-1]

    return {
        "delta_star": best_delta,
        "epsilon": epsilon,
        "metric": metric,
        "derivative_delta": deriv_delta,
    }


# =========================
# 九、绘图函数
# =========================
def _plot_metric_series(results_by_delta, metric, out_path):
    plt.figure(figsize=(10, 5))
    for d, df in results_by_delta.items():
        if df.empty:
            continue
        plt.plot(df["window_end"], df[metric], marker="o", label=f"Δ={d}")
    plt.title(f"{metric} over time")
    plt.xlabel("Time")
    plt.ylabel(metric)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _plot_derivative_series(results_by_delta, metric, out_path):
    plt.figure(figsize=(10, 5))
    for d, df in results_by_delta.items():
        if df.empty:
            continue
        deriv = compute_numerical_derivative(df[metric]).abs()
        plt.plot(df["window_end"], deriv, marker="o", label=f"Δ={d}")
    plt.title(f"|D(t)| of {metric}")
    plt.xlabel("Time")
    plt.ylabel(f"|Δ{metric}|")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# =========================
# 十、主函数
# =========================
def main():
    """
    T4 主流程：
    - 读取 T2 输出
    - 构建时间窗口
    - 模拟不同更新频率
    - 计算数值导数
    - 给出最优更新建议
    - 输出图表和 CSV
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--window_days", type=int, default=90)
    parser.add_argument("--deltas", default="1,2,3,4")
    parser.add_argument("--epsilon", type=float, default=0.02)
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--fig_dir", default=DEFAULT_FIG_DIR)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.fig_dir, exist_ok=True)

    df = pd.read_csv(args.input)

    # 构造真实标签 y
    if "y" not in df.columns:
        df["y"] = (df["Lab Status"] == "Positive ID").astype(int)

    # 选择事件时间
    if "Submission Date" in df.columns and "Detection Date" in df.columns:
        df["event_date"] = df["Submission Date"].fillna(df["Detection Date"])
    elif "Submission Date" in df.columns:
        df["event_date"] = df["Submission Date"]
    else:
        df["event_date"] = df["Detection Date"]

    df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce")
    df = df.dropna(subset=["event_date", "y"])

    # 特征列
    feature_cols = _select_feature_cols(df)

    # 构建时间窗口
    windows = build_time_windows(df, "event_date", args.window_days)

    deltas = [int(x) for x in args.deltas.split(",")]
    results_by_delta = {}

    for d in deltas:
        res = train_and_evaluate(df, windows, feature_cols, update_interval=d)
        res["delta"] = d
        results_by_delta[d] = res

    # 保存性能结果
    perf_df = pd.concat(results_by_delta.values(), ignore_index=True)
    perf_df.to_csv(
        os.path.join(args.out_dir, "t4_update_strategy_performance.csv"),
        index=False,
        encoding="utf-8"
    )

    # 选择最优更新频率
    rec = select_optimal_update_interval(
        results_by_delta,
        metric="auroc",
        epsilon=args.epsilon
    )

    with open(
        os.path.join(args.out_dir, "t4_update_strategy_recommendation.txt"),
        "w",
        encoding="utf-8"
    ) as f:
        f.write(str(rec))

    # 绘图
    for metric in ["auroc", "auprc", "top10_pos_rate"]:
        _plot_metric_series(
            results_by_delta,
            metric,
            os.path.join(args.fig_dir, f"t4_{metric}.png")
        )
        _plot_derivative_series(
            results_by_delta,
            metric,
            os.path.join(args.fig_dir, f"t4_{metric}_derivative.png")
        )

    print("T4 finished successfully.")


if __name__ == "__main__":
    main()
