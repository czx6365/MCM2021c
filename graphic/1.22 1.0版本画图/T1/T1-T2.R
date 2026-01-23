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

# --- 2. 定义文件路径 (使用您提供的最新路径) ---
spatial_data_path <- "E:/美赛/2021C/code/MCM2021c/graphic/1.22 1.0版本画图/t1_spatiotemporal_data.csv"
spatial_output_path <- "E:/美赛/2021C/code/MCM2021c/graphic/1.22 1.0版本画图/figure_1_spatiotemporal_map_v3.svg"
seasonal_data_path <- "E:/美赛/2021C/code/MCM2021c/graphic/1.22 1.0版本画图/t1_seasonal_trend_data.csv"
seasonal_output_path <- "E:/美赛/2021C/code/MCM2021c/graphic/1.22 1.0版本画图/figure_2_seasonal_trend_v3.svg"


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

# --- 5. 绘制图1：时空分布图 (已应用最终修正) ---
# 读取数据
spatial_data <- read.csv(spatial_data_path, stringsAsFactors = FALSE)


#
# ******************** 最终关键修正 ********************
#
#  我们采取三步确保数据纯净：
#  1. 强制将 year 列转为字符，防止任何非预期的格式。
#  2. 移除所有 year 不是 "2019" 或 "2020" 的行（关键的防御性步骤）。
#  3. 最后，将清理干净的列转为因子。
#
spatial_data_cleaned <- spatial_data %>%
  mutate(year = as.character(year)) %>%
  filter(year %in% c("2019", "2020")) %>%
  mutate(year = factor(year))
#
# ******************************************************
#

# 获取地图数据
north_america <- ne_countries(scale = "medium", continent = "North America", returnclass = "sf")
coord_limits_x <- c(-125, -121.5)
coord_limits_y <- c(48, 49.5)

# 绘图 (现在使用清理过的数据 `spatial_data_cleaned`)
ggplot() +
  geom_sf(data = north_america, fill = vibrant_colors["map_bg"], color = "white") +
  geom_point(
    data = spatial_data_cleaned, # <-- 使用清理后的数据
    aes(x = longitude, y = latitude, color = year), 
    size = 4,
    alpha = 0.8
  ) +
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

# 保存 (使用 v3 后缀)
ggsave(spatial_output_path, width = 10, height = 8)
print(paste("已应用最终修正的图1已保存到:", spatial_output_path))


# --- 6. 绘制图2：季节性趋势图 (代码不变) ---
# (这部分代码通常不会有问题，保持原样)
seasonal_data <- read.csv(seasonal_data_path)
all_months <- data.frame(month = 1:12)
seasonal_data_full <- all_months %>%
  left_join(seasonal_data, by = "month") %>%
  mutate(case_count = ifelse(is.na(case_count), 0, case_count))
seasonal_data_full$month_name <- factor(month.abb[seasonal_data_full$month], levels = month.abb)

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

# 保存 (使用 v3 后缀)
ggsave(seasonal_output_path, width = 12, height = 7)
print(paste("已应用最终修正的图2已保存到:", seasonal_output_path))
