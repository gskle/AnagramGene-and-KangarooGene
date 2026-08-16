# Plot three-kingdom paired-input accuracy and AUC.
#
# The paired-input model is evaluated twice per transcript: once with
# (native, shuffled) and once with (shuffled, native). A transcript counts as
# correct when the probability of the native-order input is above 0.5 in
# either presentation. The script then summarises accuracy and AUC per fold,
# per training kingdom and per test kingdom and draws bar charts.
#
# Usage:
#   Rscript plot_pi.R <input_dir> <output_svg>

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(pROC)
})

args <- commandArgs(trailingOnly = TRUE)
input_dir <- if (length(args) >= 1) args[1] else "."
output_svg <- if (length(args) >= 2) args[2] else "threekingdom_accuracy_plot.svg"

read_pred <- function(file) {
  path <- file.path(input_dir, file)
  if (!file.exists(path)) stop("Missing prediction file: ", path)
  read.csv(path, stringsAsFactors = FALSE)
}

# Prediction files are named <data_file>_<model_kingdom>_2input_pred_out.csv.
data_files <- c(
  plant = "plant18_intronshuffle.csv",
  animal = "animal14_intronshuffle.csv",
  fungi = "fungi6_intronshuffle.csv"
)

pred_file <- function(data_key, model_kingdom) {
  base <- sub("\\.csv$", "", data_files[[data_key]])
  paste0(base, "_", model_kingdom, "_2input_pred_out.csv")
}

# Nine prediction files: self predictions and all cross-kingdom transfers.
pred_files <- c(
  plant = pred_file("plant", "plant"),
  fungi = pred_file("fungi", "fungi"),
  animal = pred_file("animal", "animal"),
  animaltoplant = pred_file("plant", "animal"),
  animaltofungi = pred_file("fungi", "animal"),
  planttoanimal = pred_file("animal", "plant"),
  planttofungi = pred_file("fungi", "plant"),
  fungitoplant = pred_file("plant", "fungi"),
  fungitoanimal = pred_file("animal", "fungi")
)

pred_list <- lapply(pred_files, function(f) read_pred(f))

# Accuracy per orthogroup fold: the native-order input must score above 0.5
# in the (native, shuffled) and (shuffled, native) presentations.
acc_fun <- function(df) {
  df %>%
    group_by(group_type) %>%
    summarise(
      acc = (sum(left_out > 0.5) + sum(right_out > 0.5)) / (n() * 2),
      .groups = "drop"
    )
}

model_label <- c(
  plant = "Plant model", fungi = "Fungi model", animal = "Animal model",
  animaltoplant = "Animal model", animaltofungi = "Animal model",
  planttoanimal = "Plant model", planttofungi = "Plant model",
  fungitoplant = "Fungi model", fungitoanimal = "Fungi model"
)

test_label <- c(
  plant = "Plant data", fungi = "Fungi data", animal = "Animal data",
  animaltoplant = "Plant data", animaltofungi = "Fungi data",
  planttoanimal = "Animal data", planttofungi = "Fungi data",
  fungitoplant = "Animal data", fungitoanimal = "Plant data"
)

all_df <- do.call(rbind, lapply(names(pred_list), function(key) {
  df <- acc_fun(pred_list[[key]])
  df$model <- model_label[[key]]
  df$testkingdom <- test_label[[key]]
  df
}))

error_df <- all_df %>%
  group_by(model, testkingdom) %>%
  summarise(
    mean_acc = mean(acc),
    se = sd(acc) / sqrt(n()),
    .groups = "drop"
  )

kingdom_levels <- c("Plant data", "Animal data", "Fungi data")
error_df$testkingdom <- factor(error_df$testkingdom, levels = kingdom_levels)
all_df$testkingdom <- factor(all_df$testkingdom, levels = kingdom_levels)

p_acc <- ggplot(error_df, aes(y = mean_acc, x = model, fill = model)) +
  geom_col(width = 0.7) +
  geom_errorbar(aes(ymin = mean_acc - se, ymax = mean_acc + se), width = 0.2) +
  geom_jitter(data = all_df, aes(y = acc, x = model),
              width = 0.15, height = 0, size = 1, color = "black", alpha = 0.8) +
  scale_y_continuous(limits = c(0, 1), expand = c(0, 0)) +
  scale_fill_manual(values = c(
    "Animal model" = "cadetblue3",
    "Plant model" = "chartreuse3",
    "Fungi model" = "gold3"
  )) +
  facet_wrap(~testkingdom, scales = "free_x") +
  labs(x = "Training Model", y = "Accuracy") +
  theme_bw() +
  theme(
    panel.border = element_rect(fill = NA, color = "black", size = 1.8),
    panel.background = element_blank(),
    axis.text = element_text(size = 12),
    axis.title = element_text(size = 14),
    text = element_text(size = 15, color = "black")
  )

ggsave(output_svg, p_acc, width = 10, height = 4)
print(p_acc)

# ===================== AUC =====================
make_roc_df <- function(df) {
  data.frame(
    gene = df$gene,
    acc = c(df$left_out, 1 - df$right_out),
    type = c(rep("positive", nrow(df)), rep("negative", nrow(df))),
    div = df$group_type
  )
}

auc_fold <- function(df) {
  roc_df <- make_roc_df(df)
  auc <- vapply(1:5, function(i) {
    sub <- roc_df[roc_df$div == i, ]
    if (nrow(sub) == 0 || length(unique(sub$type)) < 2) return(NA_real_)
    as.numeric(auc(roc(sub$type, sub$acc, quiet = TRUE)))
  }, numeric(1))
  data.frame(group = 1:5, auc = auc)
}

auc_df <- do.call(rbind, lapply(names(pred_list), function(key) {
  df <- auc_fold(pred_list[[key]])
  df$model <- model_label[[key]]
  df$testkingdom <- test_label[[key]]
  df
}))

auc_summary <- auc_df %>%
  group_by(model, testkingdom) %>%
  summarise(
    mean_auc = mean(auc, na.rm = TRUE),
    se = sd(auc, na.rm = TRUE) / sqrt(sum(!is.na(auc))),
    .groups = "drop"
  )

auc_summary$testkingdom <- factor(auc_summary$testkingdom, levels = kingdom_levels)
auc_df$testkingdom <- factor(auc_df$testkingdom, levels = kingdom_levels)

p_auc <- ggplot(auc_summary, aes(y = mean_auc, x = model, fill = model)) +
  geom_col(width = 0.7) +
  geom_errorbar(aes(ymin = mean_auc - se, ymax = mean_auc + se), width = 0.2) +
  geom_jitter(data = auc_df, aes(y = auc, x = model),
              width = 0.15, height = 0, size = 1, color = "black", alpha = 0.8) +
  scale_y_continuous(limits = c(0, 1), expand = c(0, 0)) +
  scale_fill_manual(values = c(
    "Animal model" = "cadetblue3",
    "Plant model" = "chartreuse3",
    "Fungi model" = "gold3"
  )) +
  facet_wrap(~testkingdom, scales = "free_x") +
  labs(x = "Training Model", y = "AUC") +
  theme_bw() +
  theme(
    panel.border = element_rect(fill = NA, color = "black", size = 1.8),
    panel.background = element_blank(),
    axis.text = element_text(size = 12),
    axis.title = element_text(size = 14),
    text = element_text(size = 15, color = "black")
  )

auc_svg <- file.path(input_dir, "threekingdom_AUC_plot.svg")
ggsave(auc_svg, p_auc, width = 10, height = 4)
print(p_auc)
