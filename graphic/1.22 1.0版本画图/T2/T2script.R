library(ggplot2)
library(dplyr)
library(svglite)
library(showtext)

# 1. Font Setup (Crucial for the "Journal" look)
# Load a serif font that mimics Times New Roman
font_add_google("Merriweather", "journal_font") 
showtext_auto()

# 2. Refined Color Palette & Theme
# Using a "Rich" palette: slightly more vibrant for contrast
my_colors <- c(
  red = "#E9687A",       # Coral Red
  blue = "#6A9EB5",      # Muted Blue (better contrast than purple)
  grey_light = "#F0F0F0",
  text_dark = "#2C3E50"
)

# Enhanced Theme
journal_theme <- theme_minimal(base_family = "journal_font") +
  theme(
    # Clean borders and grid
    panel.border = element_rect(colour = "black", fill = NA, linewidth = 1),
    panel.grid.major = element_line(color = "grey90", linetype = "dashed"),
    panel.grid.minor = element_blank(),
    
    # Text styling
    text = element_text(color = "black"),
    axis.text = element_text(size = 11, color = "black"),
    axis.title = element_text(size = 12, face = "bold"),
    plot.title = element_text(size = 14, face = "bold", hjust = 0.5),
    
    # Legend styling
    legend.position = "top",
    legend.background = element_blank(),
    legend.box.background = element_blank(),
    
    # Ticks pointing out
    axis.ticks = element_line(color = "black"),
    axis.ticks.length = unit(0.2, "cm")
  )

# Set Working Directory
setwd("E:/美赛/2021C/code/MCM2021c/graphic/1.22 1.0版本画图/T2") 

# --- 2.1 Bar Chart (Model Comparison) ---
df_21 <- read.csv("plot_2_1_model_comparison.csv")
df_21 <- df_21 %>% filter(Metric %in% c("auroc_mean", "auprc_mean", "uplift_5_mean"))

p1 <- ggplot(df_21, aes(x = Metric, y = Value, fill = Model)) +
  geom_bar(stat = "identity", position = position_dodge(0.7), width = 0.6, 
           color = "black", size = 0.3) +  # Add black border
  
  # Add Text Labels on top of bars (Rich detail)
  geom_text(aes(label = sprintf("%.2f", Value)), 
            position = position_dodge(0.7), vjust = -0.5, 
            family = "journal_font", size = 3.5) +
  
  scale_fill_manual(values = c("With Images" = my_colors[["red"]], 
                               "Without Images" = my_colors[["blue"]])) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.15)), limits = c(0, 1)) +
  labs(x = "Evaluation Metrics", y = "Performance Score", title = "Model Performance Comparison") +
  journal_theme

ggsave("Chart_2_1_Comparison_Rich.svg", p1, width = 8, height = 6)


# --- 2.2 Lollipop Chart (Feature Importance) ---
# *CHANGED* from Bar Chart to Lollipop for a cleaner, modern look
df_22 <- read.csv("plot_2_2_feature_importance.csv")

p2 <- ggplot(df_22, aes(x = reorder(Feature, Abs_Coefficient), y = Coefficient)) +
  # Vertical line at 0
  geom_hline(yintercept = 0, color = "black", size = 0.5) +
  
  # The Stick
  geom_segment(aes(xend = Feature, yend = 0), color = "grey50", size = 0.8) +
  
  # The Pop (Point)
  geom_point(aes(fill = Coefficient > 0), shape = 21, size = 4, color = "black") +
  
  coord_flip() +
  scale_fill_manual(values = c("TRUE" = my_colors[["red"]], 
                               "FALSE" = my_colors[["blue"]]), labels=c("Negative", "Positive")) +
  labs(x = NULL, y = "Model Coefficient (Influence)", title = "Feature Importance Analysis") +
  journal_theme +
  theme(legend.position = "none") # Legend often redundant here if color implies sign

ggsave("Chart_2_2_Importance_Rich.svg", p2, width = 7, height = 7)


# --- 2.3 KDE Plot (Distribution) ---
df_23 <- read.csv("plot_2_3_prob_dist.csv")

p3 <- ggplot(df_23, aes(x = prob_positive, fill = Lab.Status, color = Lab.Status)) +
  # Density with transparency and border
  geom_density(alpha = 0.4, size = 1) +
  
  # Add Rug Plot at bottom (Shows actual data density)
  geom_rug(alpha = 0.5, length = unit(0.05, "npc")) +
  
  scale_fill_manual(values = c("Positive ID" = my_colors[["red"]], 
                               "Negative ID" = my_colors[["blue"]])) +
  scale_color_manual(values = c("Positive ID" = my_colors[["red"]], 
                                "Negative ID" = my_colors[["blue"]])) +
  labs(x = "Predicted Probability", y = "Density", title = "Probability Density Distribution") +
  journal_theme +
  theme(legend.position = c(0.85, 0.85)) # Inset legend

ggsave("Chart_2_3_Distribution_Rich.svg", p3, width = 8, height = 6)


# --- 2.4 Confusion Matrix (Rich Heatmap) ---
df_24 <- read.csv("plot_2_4_confusion_matrix.csv")

# Pre-calculate Percentages
total_count <- sum(df_24$Count)
df_24 <- df_24 %>% 
  mutate(Percent = Count / total_count * 100,
         Label = paste0(Count, "\n(", sprintf("%.1f", Percent), "%)"))

p4 <- ggplot(df_24, aes(x = factor(Predicted), y = factor(Actual), fill = Count)) +
  geom_tile(color = "white", lwd = 2) + # Thick white borders
  
  # Rich Text: Count + Percentage
  geom_text(aes(label = Label), size = 5, family = "journal_font", fontface = "bold") +
  
  scale_fill_gradient(low = "#FFF5F5", high = my_colors[["red"]]) +
  scale_x_discrete(labels = c("0" = "Negative", "1" = "Positive"), position = "top") +
  scale_y_discrete(labels = c("0" = "Negative", "1" = "Positive")) +
  labs(x = "Predicted Class", y = "Actual Class", title = "Confusion Matrix") +
  journal_theme +
  theme(
    axis.title = element_text(size = 13, face="bold"),
    axis.ticks = element_blank(),
    panel.border = element_blank() # Heatmaps look better without outer border
  )

ggsave("Chart_2_4_Confusion_Rich.svg", p4, width = 6, height = 6)
