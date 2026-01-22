# Title: 3D Spatiotemporal Scatter Plot
# Author: Joy
# Date: 2026-1-22
# Description: This script creates a 3D scatter plot to visualize spatiotemporal data,
#              incorporating seasonal trends through point size, and styled for
#              scientific publication.

# --- 1. Setup and Configuration ---

# Install required packages if they are not already installed.
# plot3D is for creating the 3D plot.
# svglite is for saving the plot in high-quality SVG format.
if (!require("plot3D")) install.packages("plot3D")
if (!require("svglite")) install.packages("svglite")

# Load the libraries
library(plot3D)
library(svglite)

spatial_data_path <- "E:/美赛/2021C/code/MCM2021c/graphic/1.22 1.0版本画图/t1_spatiotemporal_data.csv"
seasonal_data_path <- "E:/美赛/2021C/code/MCM2021c/graphic/1.22 1.0版本画图/t1_seasonal_trend_data.csv"

# Define the output path and filename for the plot.
output_path <- "." # Current directory
output_filename <- "3D_Spatiotemporal_Scatter_Plot.svg"

# Define the color palette based on your preference.
# We will use a subset of these for a clean, 2-3 color scheme.
color_palette <- list(
  purple_dark = "#B6B3D6",
  coral = "#F1837A",
  grey_light = "#D5D1D1"
)

# Assign colors for plot elements
color_year_2019 <- color_palette$coral
color_year_2020 <- color_palette$purple_dark
color_projection <- color_palette$grey_light

# --- 2. Data Loading and Preparation ---

# Read the spatiotemporal and seasonal data
spatiotemporal_data <- read.csv(spatial_data_path)
seasonal_data <- read.csv(seasonal_data_path)

# Convert the 'detection_date' column to Date objects for manipulation.
spatiotemporal_data$detection_date <- as.Date(spatiotemporal_data$detection_date)

# Create a numeric Z-axis for time: "Days Since First Detection".
# This makes the time axis quantitative and easy to interpret.
min_date <- min(spatiotemporal_data$detection_date)
spatiotemporal_data$time_days <- as.numeric(spatiotemporal_data$detection_date - min_date)

# Extract the month number to join with the seasonal data.
spatiotemporal_data$month <- as.numeric(format(spatiotemporal_data$detection_date, "%m"))

# Merge the spatial data with seasonal data to get 'case_count' for each point.
merged_data <- merge(spatiotemporal_data, seasonal_data, by = "month", all.x = TRUE)

# Handle potential missing values in case_count after the merge (if any).
merged_data$case_count[is.na(merged_data$case_count)] <- 1

# Map 'case_count' to point size ('cex'). We scale the sizes for better visual representation.
# Formula: base_size + scaled_size. This ensures all points are visible.
merged_data$point_size <- 0.8 + (merged_data$case_count / max(merged_data$case_count)) * 2

# Create a color vector based on the detection year.
merged_data$year_factor <- as.factor(merged_data$year)
point_colors <- ifelse(merged_data$year_factor == "2019", color_year_2019, color_year_2020)

# --- 3. Plot Generation ---

# Set up the SVG output device. This will save the plot to a file.
svglite(file.path(output_path, output_filename), width = 10, height = 8)

# Define plot boundaries, adding some padding for aesthetics.
x_lim <- range(merged_data$longitude) + c(-0.1, 0.1)
y_lim <- range(merged_data$latitude) + c(-0.1, 0.1)
z_lim <- c(0, max(merged_data$time_days) + 20)

# Set the viewing angle for the 3D plot.
phi <- 25
theta <- -45

# Create the main 3D scatter plot.
# This first call to scatter3D sets up the entire plot area.
scatter3D(
  x = merged_data$longitude,
  y = merged_data$latitude,
  z = merged_data$time_days,
  
  # --- Aesthetics ---
  col = point_colors,
  cex = merged_data$point_size,
  pch = 19, # Solid circles
  
  # --- Box and Axes Style ---
  bty = "g", # Box with grey background and white grid lines.
  ticktype = "detailed", # Ticks point outwards, matching journal style.
  
  # --- Labels and Title ---
  xlab = "\n\nLongitude", # Add newlines for spacing
  ylab = "\n\nLatitude",
  zlab = "\n\nDays Since First Detection",
  main = NULL, # No title on the plot itself, as requested.
  
  # --- Plot and Viewport ---
  xlim = x_lim, ylim = y_lim, zlim = z_lim,
  phi = phi, theta = theta,
  
  # --- Legend ---
  colkey = FALSE # Turn off the default continuous color key.
)

# Add projections onto the planes to replicate the example's style.
# These are added as smaller, semi-transparent points.
# Projection on XY plane (bottom)
scatter3D(
  x = merged_data$longitude, y = merged_data$latitude, z = rep(z_lim[1], nrow(merged_data)),
  add = TRUE, col = color_projection, pch = 19, cex = 0.5
)
# Projection on XZ plane (back wall)
scatter3D(
  x = merged_data$longitude, y = rep(y_lim[2], nrow(merged_data)), z = merged_data$time_days,
  add = TRUE, col = color_projection, pch = 19, cex = 0.5
)
# Projection on YZ plane (left wall)
scatter3D(
  x = rep(x_lim[1], nrow(merged_data)), y = merged_data$latitude, z = merged_data$time_days,
  add = TRUE, col = color_projection, pch = 19, cex = 0.5
)

# Add a legend to explain the colors for the years.
legend(
  "topright",
  inset = c(0.05, 0.1), # Adjust inset to position legend nicely.
  legend = c("2019", "2020"),
  col = c(color_year_2019, color_year_2020),
  pch = 19,
  bty = "n", # No box around the legend.
  cex = 1.0,
  title = "Detection Year",
  title.adj = 0.1 # Adjust title position
)

# Close the SVG device, which saves the file.
dev.off()

print(paste("Plot saved to:", file.path(output_path, output_filename)))

