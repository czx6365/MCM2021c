import os
import sys
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

# ================== 【环境路径修复】 ==================
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

project_root = os.path.dirname(current_dir)
if os.path.basename(os.getcwd()) == "scr":
    os.chdir(project_root)
    print(f"Working directory changed to: {project_root}")
# =================================================

try:
    import t2_train_predict_with_images as t2_with
except ImportError as e:
    print(f"Import Error: {e}")
    exit()

# 输出路径
OUTPUT_DIR = os.path.join(project_root, r"graphic\1.22 1.0版本画图\T2")
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE = 42

def generate_2_1_model_comparison(df, image_index):
    print("Generating data for 2.1 Model Comparison...")
    risk_model = t2_with.RiskPriorModel()
    
    # 这里的特征列表必须与您的 t2_train_predict_with_images.py 训练时的列表一致
    features_with = [
        "prob_img", "risk_prior_log", "note_length", "kw_large", 
        "kw_yellow", "kw_queen", "delay_days", "month", 
        "Latitude", "Longitude", "has_image", "num_images"
    ]
    img_cols = [c for c in df.columns if c.startswith("cnn_emb_")]

    # 模拟无图模式的数据状态
    df_no_img = df.copy()
    df_no_img["prob_img"] = df["y"].mean()
    df_no_img["has_image"] = 0
    df_no_img["num_images"] = 0
    
    # 无图模式特征集（去除 has_image 和 num_images）
    features_without = [f for f in features_with if f not in ["has_image", "num_images"]]
    
    metrics_with = t2_with.evaluate_model(df, features_with, img_cols, risk_model)
    metrics_without = t2_with.evaluate_model(df_no_img, features_without, [], risk_model)
    
    data = []
    for m in ["auroc_mean", "auprc_mean", "uplift_1_mean", "uplift_5_mean", "uplift_10_mean"]:
        data.append({"Metric": m, "Model": "With Images", "Value": metrics_with.get(m, 0)})
        data.append({"Metric": m, "Model": "Without Images", "Value": metrics_without.get(m, 0)})
        
    pd.DataFrame(data).to_csv(os.path.join(OUTPUT_DIR, "plot_2_1_model_comparison.csv"), index=False)

def generate_2_2_feature_importance(df, image_feature_cols):
    print("Generating data for 2.2 Feature Importance...")
    risk_model = t2_with.RiskPriorModel()
    labeled_pos = df[df["Lab Status"] == "Positive ID"]
    
    df["risk_prior"] = t2_with.compute_risk_prior(
        labeled_pos, df["Detection Date"].to_numpy(),
        df["Latitude"].to_numpy(), df["Longitude"].to_numpy(), risk_model
    )
    df["risk_prior_log"] = np.log1p(df["risk_prior"])
    
    image_model, img_fallback = t2_with.train_image_model(df, image_feature_cols)
    df["prob_img"] = t2_with.predict_prob_img(df, image_model, img_fallback, image_feature_cols)
    
    feature_cols = [
        "prob_img", "risk_prior_log", "note_length", "kw_large", 
        "kw_yellow", "kw_queen", "delay_days", "month", 
        "Latitude", "Longitude", "has_image", "num_images"
    ]
    
    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)
    X = df[feature_cols].fillna(0.0)
    model.fit(X, df["y"])
    
    imp_df = pd.DataFrame({
        "Feature": feature_cols,
        "Coefficient": model.coef_[0],
        "Abs_Coefficient": np.abs(model.coef_[0])
    }).sort_values("Abs_Coefficient", ascending=False)
    
    imp_df.to_csv(os.path.join(OUTPUT_DIR, "plot_2_2_feature_importance.csv"), index=False)
    return model, feature_cols

def main():
    print("Loading data...")
    df = t2_with.load_reports()
    img_df = t2_with.load_images_by_globalid()
    image_index = t2_with.build_image_index(img_df)
    
    print("Preprocessing basic counts...")
    # 【修复重点】：在调用 build_features 之前，手动建立 num_images 和 has_image 列
    # 模拟 t2_train_predict_with_images.py 内部可能缺失的初始化步骤
    image_counts = img_df.groupby("GlobalID").size().to_dict()
    df["num_images"] = df["GlobalID"].map(image_counts).fillna(0)
    df["has_image"] = (df["num_images"] > 0).astype(int)

    print("Processing NLP and Time features...")
    df = t2_with.add_text_features(df)
    df = t2_with.add_time_features(df)
    
    print("Extracting CNN features (this may take time)...")
    # 现在调用 build_features 不会报错了，因为列已存在
    df = t2_with.build_features(df, image_index)
    df["y"] = (df["Lab Status"] == "Positive ID").astype(int)
    
    labeled_df = df[df["Lab Status"].isin(["Positive ID", "Negative ID"])].copy()
    
    # 2.1 对比
    generate_2_1_model_comparison(labeled_df, image_index)
    
    # 2.2 特征重要性
    img_feat_cols = [c for c in labeled_df.columns if c.startswith("cnn_emb_")]
    final_model, f_cols = generate_2_2_feature_importance(labeled_df, img_feat_cols)
    
    # 2.3 & 2.4 概率分布与混淆矩阵
    X = labeled_df[f_cols].fillna(0.0)
    labeled_df["prob_positive"] = final_model.predict_proba(X)[:, 1]
    
    labeled_df[["GlobalID", "Lab Status", "prob_positive"]].to_csv(
        os.path.join(OUTPUT_DIR, "plot_2_3_prob_dist.csv"), index=False
    )
    
    y_pred = (labeled_df["prob_positive"] >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(labeled_df["y"], y_pred).ravel()
    cm_df = pd.DataFrame([
        {"Type": "TN", "Actual": 0, "Predicted": 0, "Count": tn},
        {"Type": "FP", "Actual": 0, "Predicted": 1, "Count": fp},
        {"Type": "FN", "Actual": 1, "Predicted": 0, "Count": fn},
        {"Type": "TP", "Actual": 1, "Predicted": 1, "Count": tp}
    ])
    cm_df.to_csv(os.path.join(OUTPUT_DIR, "plot_2_4_confusion_matrix.csv"), index=False)
    
    print(f"\nSuccess! All files saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()