library(ggplot2)
library(svglite)
library(dplyr)
library(tidyr)
library(ggdist)

# --- 1. 定义莫兰迪配色方案 ---
my_pal <- c(
  purple_dark = "#B6B3D6",
  purple_light = "#CFCCE3",
  coral = "#F1837A",
  peach = "#F8B2A2",
  nude = "#F6DFD6",
  pink_dark = "#E9687A",
  grey_purple = "#D5D3DE",
  grey_light = "#D5D1D1"
)

# --- 2. 通用专业主题设置 ---
theme_journal <- function() {
  theme_bw() +
    theme(
      panel.border = element_rect(colour = "black", fill=NA, linewidth=1), # 四边框
      axis.ticks = element_line(colour = "black"),
      axis.ticks.length = unit(-0.15, "cm"), # 刻度向外 (通过负值配合margin)
      axis.text.x = element_text(margin = margin(t = 10)),
      axis.text.y = element_text(margin = margin(r = 10)),
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),
      legend.position = "top",
      legend.title = element_blank(),
      text = element_text(family = "sans", size = 12)
    )
}

# 假设数据读取路径 (请根据实际修改)
# setwd("E:/美赛/2021C/code/MCM2021c/graphic/1.22 1.0版本画图/T2")

# =================================================================
# 2.1 模型性能对比 (分组柱状图 + 差值标注)
# =================================================================
df_21 <- read.csv("plot_2_1_model_comparison.csv")
# 过滤关键指标进行展示
df_21_plot <- df_21 %>% filter(Metric %in% c("auprc_mean", "uplift_5_mean", "auroc_mean"))

p1 <- ggplot(df_21_plot, aes(x = Metric, y = Value, fill = Model)) +
  geom_bar(stat = "identity", position = position_dodge(0.8), width = 0.7, color = "white") +
  scale_fill_manual(values = c("With Images" = my_pal["coral"], "Without Images" = my_pal["purple_dark"])) +
  labs(x = "Evaluation Metrics", y = "Score") +
  theme_journal() +
  # 留出标注空位，添加指示箭头示例 (LaTeX后续可标注具体提升%)
  annotate("segment", x = 1.8, xend = 2.2, y = 0.8, yend = 0.8, arrow = arrow(length = unit(0.2, "cm")))

ggsave("Chart_2_1_Comparison.svg", p1, width = 8, height = 6, device = "svglite")

# =================================================================
# 2.2 特征重要性 (水平条形图 - 莫兰迪渐变)
# =================================================================
df_22 <- read.csv("plot_2_2_feature_importance.csv")

p2 <- ggplot(df_22, aes(x = reorder(Feature, Abs_Coefficient), y = Coefficient, fill = Coefficient > 0)) +
  geom_bar(stat = "identity", width = 0.7) +
  coord_flip() +
  scale_fill_manual(values = c("TRUE" = my_pal["coral"], "FALSE" = my_pal["purple_light"])) +
  labs(x = "Model Features", y = "Logistic Regression Coefficient") +
  theme_journal() +
  geom_hline(yintercept = 0, linetype = "dashed", color = "grey50")

ggsave("Chart_2_2_Importance.svg", p2, width = 8, height = 7, device = "svglite")

# =================================================================
# 2.3 预测概率分布 (Raincloud Plot - 现代杂志风)
# =================================================================
# 这种画法比单纯的KDE更专业，包含密度、散点和箱线图
df_23 <- read.csv("plot_2_3_prob_dist.csv")

p3 <- ggplot(df_23, aes(x = Lab_Status, y = prob_positive, fill = Lab_Status)) +
  ggdist::stat_halfeye(adjust = .5, width = .6, .width = 0, justification = -.3, point_colour = NA) +
  geom_boxplot(width = .15, outlier.shape = NA, alpha = 0.5) +
  ggdist::stat_dots(side = "left", justification = 1.1, binwidth = .02, color = my_pal["grey_purple"]) +
  scale_fill_manual(values = c("Positive ID" = my_pal["pink_dark"], "Negative ID" = my_pal["purple_light"])) +
  labs(x = "Actual Label", y = "Predicted Probability") +
  theme_journal()

ggsave("Chart_2_3_Distribution.svg", p3, width = 8, height = 6, device = "svglite")

# =================================================================
# 2.4 混淆矩阵 (热力图 - 莫兰迪配色)
# =================================================================
df_24 <- read.csv("plot_2_4_confusion_matrix.csv")

p4 <- ggplot(df_24, aes(x = factor(Predicted), y = factor(Actual), fill = Count)) +
  geom_tile(color = "white", size = 1) +
  geom_text(aes(label = Count), size = 6, color = "black", family = "sans") +
  scale_fill_gradient(low = my_pal["nude"], high = my_pal["coral"]) +
  scale_x_discrete(labels = c("0" = "Pred Negative", "1" = "Pred Positive")) +
  scale_y_discrete(labels = c("0" = "Actual Negative", "1" = "Actual Positive")) +
  labs(x = "Predicted Class", y = "True Class") +
  theme_journal() +
  theme(legend.position = "right")

ggsave("Chart_2_4_Confusion.svg", p4, width = 6, height = 5, device = "svglite")