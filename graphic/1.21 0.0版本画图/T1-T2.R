# ==============================================================================
# 0. 环境设置与风格定义 (Setup & Style)
# ==============================================================================

library(ggplot2)
library(dplyr)
library(readxl)
library(sf)
library(waffle)
library(ggdist)
library(patchwork)
library(showtext)
library(maps)
library(scales)

# ------------------------------------------------------------------------------
# 1. 莫兰迪配色方案 (Morandi Palette from your image)
# ------------------------------------------------------------------------------
morandi_cols <- c(
  purple_dark  = "#B6B3D6",
  purple_light = "#CFCCE3",
  grey_purple  = "#D5D3DE",
  grey_light   = "#D5D1D1",
  nude         = "#F6DFD6",
  peach        = "#F8B2A2",
  coral        = "#F1837A",
  pink_dark    = "#E9687A"
)

# 核心配色映射
color_positive <- morandi_cols["pink_dark"]  # 阳性/高风险
color_negative <- morandi_cols["grey_light"] # 阴性/背景
color_mistaken <- morandi_cols["purple_dark"] # 误报
color_highlight <- morandi_cols["coral"]      # 强调

# ------------------------------------------------------------------------------
# 2. 期刊风格主题 (Journal Theme)
# ------------------------------------------------------------------------------
# 启用字体 (如果没有Roboto，可能会回退到默认sans)
font_add_google("Roboto", "roboto")
showtext_auto()

theme_journal <- function() {
  theme_classic(base_size = 14, base_family = "roboto") +
    theme(
      # 边框与背景
      panel.border = element_rect(colour = "black", fill = NA, linewidth = 1), # 四周黑框
      axis.line = element_blank(), # 移除默认的L型轴线，使用panel.border代替
      
      # 坐标轴刻度向外
      axis.ticks.length = unit(0.2, "cm"), 
      axis.ticks = element_line(color = "black"),
      
      # 标题与标签
      plot.title = element_text(face = "bold", size = 16, hjust = 0.5, margin = margin(b = 10)),
      plot.subtitle = element_text(size = 12, color = "grey30", hjust = 0.5, margin = margin(b = 10)),
      axis.title = element_text(face = "bold", size = 12),
      axis.text = element_text(color = "black"),
      
      # 图例
      legend.position = "top",
      legend.background = element_blank(),
      legend.key = element_blank()
    )
}

# ==============================================================================
# 3. 数据加载 (Data Loading)
# ==============================================================================

# 定义路径
base_dir <- "E:\\美赛\\2021C\\code\\MCM2021c\\graphic"
raw_data_path <- "E:\\美赛\\2021C\\data\\2021_MCM_Problem_C_Data\\2021MCMProblemC_DataSet.xlsx"
file_pred <- file.path(base_dir, "model_predictions_ranked.csv") # 假设是csv，如果是xlsx请改为read_excel
file_data <- file.path(base_dir, "processed_labeled_data.csv")

# 尝试读取数据，如果不存在则生成模拟数据 (Demo Mode)
if (file.exists(file_pred) && file.exists(file_data)) {
  message("Loading local files...")
  df_pred <- read.csv(file_pred)
  df_data <- read.csv(file_data)
} else {
  message("Warning: Files not found. Generating DEMO DATA for visualization...")
  
  # 模拟数据：Processed Data (用于图2, 3)
  set.seed(2021)
  n_samples <- 3000
  df_data <- data.frame(
    GlobalID = 1:n_samples,
    Latitude = c(rnorm(50, 49.0, 0.1), runif(n_samples-50, 48.5, 49.5)), # 50个聚集的阳性
    Longitude = c(rnorm(50, -122.5, 0.1), runif(n_samples-50, -123.0, -122.0)),
    Label = c(rep("Positive", 50), rep("Negative", n_samples-50)),
    # 特征：阳性样本距离更近，Delay分布不同
    dist_to_positive = c(rgamma(50, shape=1, scale=0.5), rgamma(n_samples-50, shape=5, scale=5)),
    delay_days = c(rnorm(50, 5, 2), rnorm(n_samples-50, 20, 10))
  )

  # 模拟数据：Predictions (用于图1, 4)
  # 假设模型打分后，True Positive主要集中在前部
  df_pred <- data.frame(
    rank = 1:n_samples,
    score = sort(runif(n_samples), decreasing = TRUE),
    True_Label = c(rep(1, 40), rep(0, 100), rep(1, 10), rep(0, n_samples-150)) # 简单的模拟
  )
  # 重新混洗一下模拟真实感
  df_pred$True_Label <- df_pred$True_Label[order(df_pred$score + rnorm(n_samples, 0, 0.2), decreasing = TRUE)]
}

# ==============================================================================
# 3.5 标签列统一 (Standardize label column)
# ==============================================================================
# 真实数据：y (0/1) -> Label
if ("y" %in% names(df_data)) {
  df_data$Label <- ifelse(df_data$y == 1, "Positive", "Negative")
  df_data$Label <- factor(df_data$Label, levels = c("Negative", "Positive"))
} else if ("Label" %in% names(df_data)) {
  # demo 数据已经有 Label
  df_data$Label <- factor(df_data$Label, levels = c("Negative", "Positive"))
} else {
  stop("df_data 中既没有 y 也没有 Label 列，请检查 names(df_data)")
}


# ==============================================================================
# Chart 1: The Challenge (Waffle Chart)
# ==============================================================================
# 数据准备：计算比例
count_pos <- sum(df_pred$True_Label == 1)
count_neg <- nrow(df_pred) - count_pos
# 为了华夫图好看，我们将数据归一化到 10x10 = 100个格子
# 如果阳性率极低(<1%)，我们强制显示1个格子并标注
waffle_data <- c("Positive" = 1, "Negative" = 99) 

p1 <- waffle(waffle_data, rows = 10, size = 1, 
             colors = c(color_positive, color_negative),
             legend_pos = "bottom") +
  theme_journal() +
  theme(axis.text = element_blank(), 
        axis.ticks = element_blank(),
        panel.border = element_blank()) + # 华夫图不需要边框
  labs(title = "Figure 1: The Needle in the Haystack",
       subtitle = "Severe Class Imbalance (Target < 1%)") +
  # 添加标注箭头
  annotate("segment", x = 1.5, xend = 2.5, y = 9.5, yend = 9.5, 
           colour = "black", size = 0.5, arrow = arrow(length = unit(0.2, "cm"))) +
  annotate("text", x = 3, y = 9.5, label = "True Positive\n(Extremely Rare)", 
           hjust = 0, family = "roboto", size = 3.5)

# ==============================================================================
# Chart 2: The Spatial Signal (Contour + Scatter)
# ==============================================================================
# 准备地图底图 (Washington State for MCM C)
wa_map <- map_data("state", region = "washington")

p2 <- ggplot() +
  geom_polygon(data = wa_map, aes(x = long, y = lat, group = group),
               fill = "white", color = morandi_cols["grey_purple"], linewidth = 0.5) +
  stat_density_2d(
    data = subset(df_data, Label == "Positive"),
    aes(x = Longitude, y = Latitude, fill = after_stat(level)),
    geom = "polygon", alpha = 0.4, color = NA
  ) +
  scale_fill_gradient(low = morandi_cols["purple_light"], high = morandi_cols["purple_dark"]) +
  geom_point(data = subset(df_data, Label == "Negative"),
             aes(x = Longitude, y = Latitude),
             color = color_negative, alpha = 0.3, size = 0.8) +
  geom_point(data = subset(df_data, Label == "Positive"),
             aes(x = Longitude, y = Latitude),
             color = "#E6B422", fill = "#E6B422", shape = 24, size = 2.5) +
  coord_fixed(
    1.3,
    xlim = c(min(df_data$Longitude) - 0.5, max(df_data$Longitude) + 0.5),
    ylim = c(min(df_data$Latitude)  - 0.5, max(df_data$Latitude)  + 0.5)
  ) +
  theme_journal() +
  labs(
    title = "Figure 2: Spatial Clustering of True Reports",
    subtitle = "Positives (Stars) align with High-Risk Contours",
    x = "Longitude", y = "Latitude"
  ) +
  theme(legend.position = "none")

# ==============================================================================
# Chart 3: The Discriminators (Raincloud Plot)
# ==============================================================================
# 数据转换：为了画图美观，取对数或者标准化特征
# 假设我们要对比 "dist_to_positive" (距离最近阳性点的距离)
df_plot3 <- df_data
df_plot3$Log_Dist <- log1p(df_plot3$dist_to_positive) # Log scale for visualization

p3 <- ggplot(df_plot3, aes(x = Log_Dist, y = Label, fill = Label, color = Label)) +
  # 1. 云 (Density)
  stat_halfeye(adjust = 0.5, justification = -0.2, .width = 0, point_colour = NA) +
  # 2. 箱线 (Boxplot)
  geom_boxplot(width = 0.12, outlier.color = NA, alpha = 0.5) +
  # 3. 雨 (Strip dots)
  stat_dots(side = "bottom", justification = 1.2, scale = 0.5, alpha = 0.6) +
  # 配色与修饰
  scale_fill_manual(values = c("Negative" = color_mistaken, "Positive" = color_positive)) +
  scale_color_manual(values = c("Negative" = color_mistaken, "Positive" = color_positive)) +
  theme_journal() +
  labs(title = "Figure 3: Feature Discrimination Analysis",
       subtitle = "Distribution of 'Distance to Nearest Positive' by Class",
       x = "Log(Distance to Nearest Positive)", y = "Class Label") +
  coord_flip() # 翻转让类别在X轴可能更好看，或者保持Y轴为类别(Raincloud标准是Y为类别)

# ==============================================================================
# Chart 4: The Solution Impact (Cumulative Gain / Lift)
# ==============================================================================
# 计算累计增益
df_gain <- df_pred %>%
  mutate(score = prob, True_Label = y )%>%
  arrange(desc(score)) %>%
  mutate(
    rank_pct = row_number() / n(),
    cum_positive = cumsum(True_Label),
    total_positive = sum(True_Label),
    recall = cum_positive / total_positive,
    random_recall = rank_pct # Baseline
  ) %>%
  filter(row_number() %% 10 == 0) # 降采样减少绘图点数

p4 <- ggplot(df_gain, aes(x = rank_pct, y = recall)) +
  # Baseline (对角虚线)
  geom_segment(aes(x = 0, y = 0, xend = 1, yend = 1), 
               linetype = "dashed", color = "grey50") +
  # Model Curve
  geom_line(color = color_positive, size = 1.2) +
  # 阴影填充 (Area under curve feel)
  geom_area(fill = color_positive, alpha = 0.1) +
  # 标注 Top 10% 点
  geom_vline(xintercept = 0.1, linetype = "dotted", color = "black") +
  geom_point(data = filter(df_gain, abs(rank_pct - 0.1) < 0.005)[1,], 
             aes(x = rank_pct, y = recall), color = color_highlight, size = 3) +
  # 文字标注
  annotate("text", x = 0.15, y = 0.6, 
           label = paste0("Top 10% Priority:\nRecall ≈ ", round(filter(df_gain, abs(rank_pct - 0.1) < 0.005)[1,"recall"]*100, 1), "%"), 
           hjust = 0, color = "black", family = "roboto", size = 3.5) +
  # 轴设置
  scale_x_continuous(labels = scales::percent, expand = c(0,0), limits = c(0, 1)) +
  scale_y_continuous(labels = scales::percent, expand = c(0,0), limits = c(0, 1.05)) +
  theme_journal() +
  labs(title = "Figure 4: Model Efficacy (Cumulative Gain)",
       subtitle = "Rapid Accumulation of True Positives in Top Ranked Reports",
       x = "Percentage of Reports Investigated",
       y = "Percentage of True Positives Found (Recall)")

# ==============================================================================
# 4. 保存输出 (Saving)
# ==============================================================================
# 组合保存预览 (可选)
# combined_plot <- (p1 | p2) / (p3 | p4)
# ggsave(file.path(base_dir, "combined_overview.png"), combined_plot, width = 12, height = 10, dpi = 300)

# 单独保存为 SVG (矢量图，适合论文)
ggsave(file.path(base_dir, "Fig1_Waffle.svg"), p1, width = 6, height = 5)
ggsave(file.path(base_dir, "Fig2_Spatial.svg"), p2, width = 6, height = 5)
ggsave(file.path(base_dir, "Fig3_Raincloud.svg"), p3, width = 6, height = 5)
ggsave(file.path(base_dir, "Fig4_GainCurve.svg"), p4, width = 6, height = 5)

message("All charts saved to: ", base_dir)
