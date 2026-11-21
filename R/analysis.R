# --- STEP 1: LOAD LIBRARIES ---
# Use install.packages(c("tidyverse", "randomForest", "caret")) if not installed
library(tidyverse)
library(randomForest)
library(caret)

# --- STEP 2: DATA PREPROCESSING ---
# Loading data with ';' separator
df <- read.csv("cardio_train.csv", sep = ";")

# Cleaning and feature construction
df <- df %>%
  mutate(
    age_years = as.integer(round(age / 365)),
    bmi = weight / ((height / 100) ^ 2)
  )

# Selecting Independent Variables (IVs) and Dependent Variable (DV)
features <- c("age_years", "bmi", "cholesterol", "gluc", "active")
target <- "ap_hi"

# Filter unrealistic systolic blood pressure values and keep a reproducible sample
set.seed(42)
df_clean <- df %>%
  filter(ap_hi > 80 & ap_hi < 200) %>%
  slice_sample(n = 100)

final_df <- df_clean %>%
  select(all_of(features), all_of(target))

# --- STEP 3: SUMMARY STATISTICS (Task-02) ---
mode_value <- function(x) {
  freq <- table(x, useNA = "no")
  if (length(freq) == 0 || max(freq) <= 1) {
    return(NA_character_)
  }
  paste(names(freq[freq == max(freq)]), collapse = "; ")
}

summary_stats <- tibble(
  variable = names(final_df),
  count = sapply(final_df, function(x) sum(!is.na(x))),
  mean = sapply(final_df, mean, na.rm = TRUE),
  std = sapply(final_df, sd, na.rm = TRUE),
  min = sapply(final_df, min, na.rm = TRUE),
  q25 = sapply(final_df, quantile, probs = 0.25, na.rm = TRUE),
  median = sapply(final_df, median, na.rm = TRUE),
  mode = sapply(final_df, mode_value),
  q75 = sapply(final_df, quantile, probs = 0.75, na.rm = TRUE),
  max = sapply(final_df, max, na.rm = TRUE)
)

cat("\n--- Task-02: Summary Statistics ---\n")
print(summary_stats)
write.csv(summary_stats, "summary_statistics.csv", row.names = FALSE)

# --- STEP 4: OUTLIER IDENTIFICATION + BOX PLOT (Task-03) ---
long_df <- final_df %>%
  pivot_longer(cols = everything(), names_to = "variable", values_to = "value")

outlier_report <- long_df %>%
  group_by(variable) %>%
  summarise(
    q1 = quantile(value, 0.25, na.rm = TRUE),
    q3 = quantile(value, 0.75, na.rm = TRUE),
    iqr = IQR(value, na.rm = TRUE),
    lower_bound = q1 - 1.5 * iqr,
    upper_bound = q3 + 1.5 * iqr,
    outlier_count = sum(value < lower_bound | value > upper_bound, na.rm = TRUE),
    outlier_values = {
      vals <- value[value < lower_bound | value > upper_bound]
      if (length(vals) == 0) "None" else paste(round(vals, 2), collapse = "; ")
    },
    .groups = "drop"
  )

cat("\n--- Task-03: Outlier Report ---\n")
print(outlier_report)
write.csv(outlier_report, "outlier_report.csv", row.names = FALSE)

box_plot <- ggplot(long_df, aes(x = variable, y = value, fill = variable)) +
  geom_boxplot(alpha = 0.8, outlier.color = "red", outlier.shape = 16) +
  theme_minimal(base_size = 12) +
  labs(
    title = "Task-03: Box and Whisker Plots for All Variables",
    x = "Variables",
    y = "Values"
  ) +
  theme(legend.position = "none")

ggsave("box_plot.png", plot = box_plot, width = 12, height = 7, dpi = 300)

# --- STEP 5: SCATTER PLOT GRID + INTERPRETATION SUPPORT (Task-04) ---
png("scatter_grid.png", width = 1600, height = 1600, res = 200)
pairs(
  final_df,
  main = "Task-04: Scatter Plot Grid for All Variables",
  pch = 19,
  col = rgb(0.15, 0.39, 0.65, 0.45)
)
dev.off()

cor_long <- as.data.frame(as.table(round(cor(final_df), 3))) %>%
  rename(var_x = Var1, var_y = Var2, correlation = Freq) %>%
  filter(var_x != var_y)

write.csv(cor_long, "scatter_grid_correlations.csv", row.names = FALSE)

interpretation_lines <- apply(cor_long, 1, function(row) {
  corr <- as.numeric(row[["correlation"]])
  strength <- case_when(
    abs(corr) >= 0.70 ~ "strong",
    abs(corr) >= 0.40 ~ "moderate",
    abs(corr) >= 0.20 ~ "weak",
    TRUE ~ "very weak"
  )
  direction <- ifelse(corr > 0, "positive", ifelse(corr < 0, "negative", "no"))
  paste0(row[["var_x"]], " vs ", row[["var_y"]], ": ", strength, " ", direction, " relationship (r = ", corr, ").")
})

writeLines(
  c(
    "Task-04 Scatter Grid Interpretation Notes",
    "Each line below can be used to discuss the relevant panel in the scatter grid.",
    "",
    interpretation_lines
  ),
  "scatter_grid_interpretation.txt"
)

# --- STEP 6: REGRESSION MODELS (Task-05) ---
set.seed(42)
trainIndex <- createDataPartition(final_df[[target]], p = 0.8, list = FALSE)
train_data <- final_df[trainIndex, ]
test_data <- final_df[-trainIndex, ]

# 1. Multiple Linear Regression Model
lr_model <- lm(ap_hi ~ ., data = train_data)
lr_pred <- predict(lr_model, test_data)

# 2. Suitable Machine Learning Model: Random Forest Regressor
rf_model <- randomForest(ap_hi ~ ., data = train_data, ntree = 100)
rf_pred <- predict(rf_model, test_data)

# --- STEP 7: EVALUATION + MODEL COMPARISON OUTPUTS ---
eval_metrics <- function(actual, predicted, model_name) {
  mse <- mean((actual - predicted) ^ 2)
  rmse <- sqrt(mse)
  r2 <- cor(actual, predicted) ^ 2

  cat(paste0("\n--- ", model_name, " Performance ---"))
  cat(paste0("\nMSE: ", round(mse, 4)))
  cat(paste0("\nRMSE: ", round(rmse, 4)))
  cat(paste0("\nR2 Score: ", round(r2, 4), "\n"))

  tibble(
    Model = model_name,
    MSE = mse,
    RMSE = rmse,
    R2 = r2
  )
}

metrics_df <- bind_rows(
  eval_metrics(test_data[[target]], lr_pred, "Linear Regression"),
  eval_metrics(test_data[[target]], rf_pred, "Random Forest")
)

print(metrics_df)
write.csv(metrics_df, "model_metrics.csv", row.names = FALSE)

prediction_comparison <- tibble(
  actual = test_data[[target]],
  linear_regression = lr_pred,
  random_forest = rf_pred
)

write.csv(prediction_comparison, "prediction_comparison.csv", row.names = FALSE)

comparison_long <- prediction_comparison %>%
  mutate(observation = row_number()) %>%
  pivot_longer(
    cols = c(linear_regression, random_forest),
    names_to = "model",
    values_to = "predicted"
  )

model_comparison_plot <- ggplot(comparison_long, aes(x = actual, y = predicted, color = model)) +
  geom_point(size = 2.8, alpha = 0.75) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "black") +
  theme_minimal(base_size = 12) +
  labs(
    title = "Actual vs Predicted Values: Model Comparison",
    x = "Original Values (ap_hi)",
    y = "Predicted Values",
    color = "Model"
  )

ggsave("model_comparison.png", plot = model_comparison_plot, width = 10, height = 6, dpi = 300)

metrics_plot_df <- metrics_df %>%
  pivot_longer(cols = c(MSE, RMSE, R2), names_to = "Metric", values_to = "Value")

metrics_plot <- ggplot(metrics_plot_df, aes(x = Metric, y = Value, fill = Model)) +
  geom_col(position = "dodge") +
  theme_minimal(base_size = 12) +
  labs(
    title = "Regression Model Performance Metrics",
    x = "Metric",
    y = "Value"
  )

ggsave("model_performance_metrics.png", plot = metrics_plot, width = 10, height = 6, dpi = 300)

# Linear regression coefficients can be used in the report's model explanation section
lr_coefficients <- as.data.frame(summary(lr_model)$coefficients)
lr_coefficients$term <- rownames(lr_coefficients)
lr_coefficients <- lr_coefficients %>%
  relocate(term)
write.csv(lr_coefficients, "linear_regression_coefficients.csv", row.names = FALSE)

cat("\nAll R outputs generated successfully.\n")
