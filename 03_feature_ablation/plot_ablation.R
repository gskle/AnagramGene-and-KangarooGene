# Plot the plant ablation accuracy comparison.
#
# The input folder must contain the prediction CSVs produced by the ablation
# scripts whose names match the pattern below, plus the plain plant paired
# input prediction file. Accuracy is computed per orthogroup fold and then
# summarised as mean +/- standard error per ablation type.
#
# Usage:
#   Rscript plot_ablation.R <input_dir> <plant_pred_csv> <output_svg>

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(readr)
})

args <- commandArgs(trailingOnly = TRUE)
input_dir <- if (length(args) >= 1) args[1] else "."
plant_pred_csv <- if (length(args) >= 2) args[2] else "plant18_intronshuffle_plant_2input_pred_out.csv"
output_svg <- if (length(args) >= 3) args[3] else "plant_regions_contri_accuracy_plot.svg"

file_pattern <- "i1_00_e0_70|i1_00_e0_00|i0_00_e1_00|i1_e2|intron_model_only.*\\.csv$"
file_list <- list.files(input_dir, pattern = file_pattern, full.names = TRUE)
if (length(file_list) == 0) stop("No ablation prediction files found in ", input_dir)

df_all <- lapply(file_list, function(file) {
  df <- read_csv(file, show_col_types = FALSE)
  df$type <- dplyr::case_when(
    grepl("i1_00_e0_70", file) ~ "i1_00_e0_70",
    grepl("i1_00_e0_00", file) ~ "i1_00_e0_00",
    grepl("i0_00_e1_00", file) ~ "i0_00_e1_00",
    grepl("i1_e2", file) ~ "i1_e2",
    grepl("intron_model_only", file) ~ "only_intron",
    TRUE ~ "unknown"
  )
  df
}) %>% bind_rows()

plant_acc <- read.csv(plant_pred_csv, stringsAsFactors = FALSE)
plant_acc$type <- "no_annotated"
df_all <- df_all[, colnames(plant_acc)]
df_all <- rbind(df_all, plant_acc)

acc_fun <- function(df) {
  df %>%
    group_by(group_type, type) %>%
    summarise(
      acc = (sum(left_out > 0.5) + sum(right_out > 0.5)) / (n() * 2),
      .groups = "drop"
    )
}

acc_sum <- acc_fun(df_all)
error_df <- acc_sum %>%
  group_by(type) %>%
  summarise(
    mean_acc = mean(acc),
    se = sd(acc) / sqrt(n()),
    .groups = "drop"
  )

p_acc <- ggplot(error_df, aes(y = mean_acc, x = type, fill = type)) +
  geom_col(width = 0.7) +
  geom_errorbar(aes(ymin = mean_acc - se, ymax = mean_acc + se), width = 0.2) +
  geom_jitter(data = acc_sum, aes(y = acc, x = type),
              width = 0.15, height = 0, size = 1, color = "black", alpha = 0.8) +
  scale_y_continuous(limits = c(0, 1), expand = c(0, 0)) +
  labs(x = "Training Model", y = "Accuracy") +
  theme_bw() +
  theme(
    panel.border = element_rect(fill = NA, color = "black", size = 1.8),
    panel.background = element_blank(),
    axis.text = element_text(size = 12),
    axis.title = element_text(size = 14),
    text = element_text(size = 15, color = "black")
  )

ggsave(output_svg, p_acc, width = 8, height = 4)
print(p_acc)

# ===================== d/i/a region-label ablation =====================
# Keep the region-label summary separate from the i/e summary.
region_file_pattern <- paste0(
  "d1_00_i1_00_a1_00|",
  "d1_00_i1_00_a0_00|",
  "d0_00_i1_00_a1_00|",
  "d1_00_i0_00_a0_00|",
  "d0_00_i0_00_a1_00|",
  "d0_00_i1_00_a0_00.*\\.csv$"
)
region_files <- list.files(
  input_dir, pattern = region_file_pattern, full.names = TRUE
)
if (length(region_files) == 0) {
  stop("No d/i/a region-label prediction files found in ", input_dir)
}

region_df <- lapply(region_files, function(file) {
  df <- read_csv(file, show_col_types = FALSE)
  df$type <- case_when(
    grepl("d1_00_i1_00_a1_00", file) ~ "d1_00_i1_00_a1_00",
    grepl("d1_00_i1_00_a0_00", file) ~ "d1_00_i1_00_a0_00",
    grepl("d0_00_i1_00_a1_00", file) ~ "d0_00_i1_00_a1_00",
    grepl("d1_00_i0_00_a0_00", file) ~ "d1_00_i0_00_a0_00",
    grepl("d0_00_i0_00_a1_00", file) ~ "d0_00_i0_00_a1_00",
    grepl("d0_00_i1_00_a0_00", file) ~ "d0_00_i1_00_a0_00",
    TRUE ~ "unknown"
  )
  df
}) %>% bind_rows()

region_acc_fun <- function(df) {
  df %>%
    group_by(group_type, type) %>%
    summarise(
      acc = (sum(left_out >= 0.5) + sum(right_out > 0.5)) / (n() * 2),
      .groups = "drop"
    ) %>%
    filter(acc != 0.5)
}

region_acc_sum <- region_acc_fun(region_df)
region_error_df <- region_acc_sum %>%
  group_by(type) %>%
  summarise(
    mean_acc = mean(acc),
    se = sd(acc) / sqrt(n()),
    .groups = "drop"
  )

p_region_acc <- ggplot(
  region_error_df, aes(y = mean_acc, x = type, fill = type)
) +
  geom_col(width = 0.7) +
  geom_errorbar(
    aes(ymin = mean_acc - se, ymax = mean_acc + se), width = 0.2
  ) +
  geom_jitter(
    data = region_acc_sum, aes(y = acc, x = type),
    width = 0.15, height = 0, size = 1, color = "black", alpha = 0.8
  ) +
  scale_y_continuous(limits = c(0, 1), expand = c(0, 0)) +
  labs(x = "Training Model", y = "Accuracy") +
  theme_bw() +
  theme(
    panel.border = element_rect(fill = NA, color = "black", size = 1.8),
    panel.background = element_blank(),
    axis.text = element_text(size = 12),
    axis.title = element_text(size = 14),
    text = element_text(size = 15, color = "black")
  )

region_output_svg <- file.path(
  input_dir, "plant_intron_flank_contri_accuracy_plot.svg"
)
ggsave(region_output_svg, p_region_acc, width = 8, height = 4)
print(p_region_acc)
