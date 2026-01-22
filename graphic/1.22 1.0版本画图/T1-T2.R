# --- 1. 准备工作：安装并加载所需的包 ---
# 如果您是第一次使用这些包，请取消下面的注释并运行以安装它们
# install.packages("ggplot2")
# install.packages("svglite")
# install.packages("dplyr")
# install.packages("rnaturalearth")
# install.packages("sf")

library(ggplot2)
library(svglite)
library(dplyr)
library(rnaturalearth) # 用于获取地图背景
library(sf)            # 用于处理地理空间数据

# --- 2. 定义您的专属SCI期刊风格配色方案 ---
# 将您提供的颜色方案定义为一个命名的向量，方便后续调用
morandi_colors <- c(
  purple_dark = "#B6B3D6",
  purple_light = "#CFCCE3",
  grey_purple = "#D5D3DE",
  grey_light = "#D5D1D1",
  nude = "#F6DFD6",
  peach = "#F8B2A2",
  coral = "#F1837A",
  pink_dark = "#E9687A"
)
spatial_data_path <- "E:/美赛/2021C/code/MCM2021c/graphic/1.22 1.0版本画图/t1_spatiotemporal_data.csv"
spatial_output_path <- "E:/美赛/2021C/code/MCM2021c/graphic/1.22 1.0版本画图/figure_1_spatiotemporal_map.svg"
seasonal_data_path <- "E:/美赛/2021C/code/MCM2021c/graphic/1.22 1.0版本画图/t1_seasonal_trend_data.csv"
seasonal_output_path <- "E:/美赛/2021C/code/MCM2021c/graphic/1.22 1.0版本画图/figure_2_seasonal_trend.svg"


# --- 3. 定义明亮的配色方案 ---
vibrant_colors <- c(
  vibrant_blue = "#0077B6",
  vibrant_orange = "#F77F00",
  map_bg = "#E9ECEF"
)


# --- 4. 定义无标题的图形主题 ---
theme_sci_journal <- function() {
  theme_classic() + 
    theme(
      plot.title = element_blank(),
      plot.subtitle = element_blank(),
      panel.border = element_rect(colour = "black", fill = NA, size = 1),
      axis.ticks.length = unit(-0.2, "cm"), 
      axis.ticks = element_line(colour = "black", size = 0.5),
      axis.text = element_text(color = "black", size = 10),
      axis.title = element_text(color = "black", size = 12, face = "bold"),
      legend.title = element_text(size = 11, face = "bold"),
      legend.text = element_text(size = 10),
      legend.position = "top",
      plot.margin = margin(0.5, 0.5, 0.5, 0.5, "cm")
    )
}


# --- 5. 绘制图1：时空分布图 (已修正) ---
# 读取数据
spatial_data <- read.csv(spatial_data_path)

#
# ******************** 关键修正 ********************
#
# 强制将 'year' 列转换为干净的、无空格的因子类型
# trimws() 会移除任何可能存在的空格，factor() 则将其转为分类变量
spatial_data$year <- factor(trimws(spatial_data$year))
#
# *************************************************
#

# 获取地图数据
north_america <- ne_countries(scale = "medium", continent = "North America", returnclass = "sf")
coord_limits_x <- c(-125, -121.5)
coord_limits_y <- c(48, 49.5)

# 绘图
ggplot() +
  geom_sf(data = north_america, fill = vibrant_colors["map_bg"], color = "white") +
  geom_point(
    data = spatial_data, 
    aes(x = longitude, y = latitude, color = year), 
    size = 4,
    alpha = 0.8
  ) +
  # 现在，这里的 "2019" 和 "2020" 将能正确匹配数据
  scale_color_manual(
    name = "Detection Year",
    values = c("2019" = vibrant_colors["vibrant_orange"], "2020" = vibrant_colors["vibrant_blue"])
  ) +
  coord_sf(xlim = coord_limits_x, ylim = coord_limits_y, expand = FALSE) +
  labs(
    x = "Longitude",
    y = "Latitude"
  ) +
  theme_sci_journal()

# 保存
ggsave(spatial_output_path, width = 10, height = 8)
print(paste("已修正的图1已成功保存到:", spatial_output_path))


# --- 6. 绘制图2：季节性趋势图 (代码不变) ---
# 读取并处理数据
seasonal_data <- read.csv(seasonal_data_path)
all_months <- data.frame(month = 1:12)
seasonal_data_full <- all_months %>%
  left_join(seasonal_data, by = "month") %>%
  mutate(case_count = ifelse(is.na(case_count), 0, case_count))
seasonal_data_full$month_name <- factor(month.abb[seasonal_data_full$month], levels = month.abb)

# 绘图
ggplot(seasonal_data_full, aes(x = month_name, y = case_count)) +
  geom_segment(
    aes(x = month_name, xend = month_name, y = 0, yend = case_count),
    color = vibrant_colors["vibrant_orange"], 
    size = 1.5
  ) +
  geom_point(
    color = vibrant_colors["vibrant_orange"], 
    size = 6, 
    alpha = 0.9
  ) +
  geom_text(aes(label = case_count), color = "white", size = 3.5, fontface = "bold") +
  labs(
    x = "Month",
    y = "Number of Confirmed Cases"
  ) +
  theme_sci_journal() +
  theme(legend.position = "none")

# 保存
ggsave(seasonal_output_path, width = 12, height = 7)
print(paste("已修正的图2已成功保存到:", seasonal_output_path))

