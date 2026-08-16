# Plot the deletion-model accuracy and score-decrease comparisons.
#
# Two panels are produced:
# 1. Classification accuracy of each deletion model (raw, LM5 and
#    KangarooGene-EN) on held-out animal and plant genes.
# 2. The fraction of genes whose native score exceeds the deleted score,
#    i.e. how often in silico intron deletion lowers authenticity.
#
# Input files are the deletion prediction tables written by the training
# scripts and the two linear ensemble scripts.
#
# Usage:
#   Rscript plot_deletion.R

suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
})

MODEL_DIR <- "/data1/lty/intron_pred/new_model/final_model"
ACC_DIR <- "/data1/lty/intron_pred/acc_out"
PHOTO_DIR <- "/data1/lty/intron_pred/photo"

ANIMAL_ACC_CSV <- file.path(MODEL_DIR, "animal_deleted_1intron_newmodel.csv")
PLANT_ACC_CSV <- file.path(MODEL_DIR, "plant_deleted_1model_predout.csv")
ANIMAL_TO_PLANT_CSV <- file.path(MODEL_DIR, "animaltoplant_deleted_1model.csv")
PLANT_TO_ANIMAL_CSV <- file.path(MODEL_DIR, "planttoanimal_deleted_1model_predout.csv")

PLANT_LINEAR_CSV <- file.path(ACC_DIR, "plant_dele_linear_predout.csv")
PLANT_TO_ANIMAL_LINEAR_CSV <- file.path(ACC_DIR, "planttoanimal_dele_linear_predout.csv")
ANIMAL_LINEAR_CSV <- file.path(ACC_DIR, "animal_dele_linear_predout.csv")
ANIMAL_TO_PLANT_LINEAR_CSV <- file.path(ACC_DIR, "animaltoplant_dele_linear_predout.csv")


# Make a data frame with pos_score/neg_score columns, resolving both the
# named layout and the positional layout.
normalise_score_columns <- function(df, pos_col = "pos_score",
                                    neg_col = "neg_score",
                                    pos_index = NULL, neg_index = NULL) {
  df <- as.data.frame(df)
  if (all(c(pos_col, neg_col) %in% colnames(df))) {
    return(df[, c("group_type", pos_col, neg_col)])
  }
  if (!is.null(pos_index) && !is.null(neg_index)) {
    return(df[, c("group_type", pos_index, neg_index)])
  }
  stop("cannot locate pos/neg score columns in ",
       deparse(substitute(df)))
}


# Accuracy: fraction of native rows > 0.5 plus deleted rows < 0.5.
accuracy_summary <- function(df) {
  df %>%
    group_by(group_type) %>%
    summarise(
      acc = (sum(pos_score > 0.5) + sum(neg_score < 0.5)) / (n() * 2),
      .groups = "drop"
    )
}


# Score decrease: fraction of genes whose native score beats the deleted one.
compare_summary <- function(df) {
  df %>%
    group_by(group_type) %>%
    summarise(
      acc = sum(pos_score > neg_score) / n(),
      .groups = "drop"
    )
}


# Same summaries for the paired real/false predictions from a linear model.
linear_accuracy_summary <- function(df) {
  df %>%
    group_by(group_type) %>%
    summarise(
      acc = (sum(pred_add_real > 0.5) + sum(pred_add_fal < 0.5)) /
        (n() * 2),
      .groups = "drop"
    )
}


# Accuracy for the exon-length-only LM5 predictions.
linear_basic_accuracy_summary <- function(df) {
  df %>%
    group_by(group_type) %>%
    summarise(
      acc = (sum(pred_basic_real > 0.5) + sum(pred_basic_fal < 0.5)) /
        (n() * 2),
      .groups = "drop"
    )
}


linear_compare_summary <- function(df) {
  df %>%
    group_by(group_type) %>%
    summarise(
      acc = sum(pred_add_real > pred_add_fal) / n(),
      .groups = "drop"
    )
}


# Score decrease for the exon-length-only LM5 predictions.
linear_basic_compare_summary <- function(df) {
  df %>%
    group_by(group_type) %>%
    summarise(
      acc = sum(pred_basic_real > pred_basic_fal) / n(),
      .groups = "drop"
    )
}


# Attach model and test-kingdom labels, then combine every table.
label_and_bind <- function(tables, models, test_kingdoms) {
  out <- do.call(rbind, lapply(seq_along(tables), function(i) {
    d <- tables[[i]]
    d$model <- models[[i]]
    d$testkingdom <- test_kingdoms[[i]]
    d
  }))
  out$testkingdom <- factor(
    out$testkingdom, levels = c("Plant data", "Animal data")
  )
  out
}


# Mean and standard error per model/testkingdom, then bar plot with jitter.
plot_summary <- function(all_df, y_limits, out_file) {
  error_df <- all_df %>%
    group_by(model, testkingdom) %>%
    summarise(
      mean_acc = mean(acc),
      se = sd(acc) / sqrt(n()),
      .groups = "drop"
    )

  p <- ggplot(error_df, aes(y = mean_acc, x = model, fill = model)) +
    geom_col(width = 0.7) +
    geom_errorbar(
      aes(ymin = mean_acc - se, ymax = mean_acc + se), width = 0.2
    ) +
    geom_jitter(
      data = all_df, aes(y = acc, x = model),
      width = 0.15, height = 0, size = 1, color = "black", alpha = 0.8
    ) +
    scale_y_continuous(limits = y_limits, expand = c(0, 0)) +
    facet_wrap(~testkingdom, scales = "free_x") +
    labs(x = "Training Model", y = "Accuracy") +
    theme(
      text = element_text(size = 15, color = "black"),
      panel.border = element_rect(
        fill = NA, color = "black", size = 1.8, linetype = "solid"
      ),
      panel.background = element_blank()
    )

  ggsave(out_file, p, width = 10, height = 4)
  cat("saved:", out_file, "\n")
}


animal_acc <- normalise_score_columns(
  read.csv(ANIMAL_ACC_CSV), pos_index = 9, neg_index = 10
)
plant_acc <- normalise_score_columns(read.csv(PLANT_ACC_CSV))
ap_acc <- normalise_score_columns(read.csv(ANIMAL_TO_PLANT_CSV))
pa_acc <- normalise_score_columns(read.csv(PLANT_TO_ANIMAL_CSV))

plant_linacc <- read.csv(PLANT_LINEAR_CSV)
pa_linacc <- read.csv(PLANT_TO_ANIMAL_LINEAR_CSV)
animal_linacc <- read.csv(ANIMAL_LINEAR_CSV)
ap_linacc <- read.csv(ANIMAL_TO_PLANT_LINEAR_CSV)

accuracy_tables <- list(
  accuracy_summary(plant_acc),
  accuracy_summary(animal_acc),
  accuracy_summary(ap_acc),
  accuracy_summary(pa_acc),
  linear_accuracy_summary(plant_linacc),
  linear_accuracy_summary(animal_linacc),
  linear_accuracy_summary(ap_linacc),
  linear_accuracy_summary(pa_linacc),
  linear_basic_accuracy_summary(plant_linacc),
  linear_basic_accuracy_summary(animal_linacc),
  linear_basic_accuracy_summary(ap_linacc),
  linear_basic_accuracy_summary(pa_linacc)
)
accuracy_models <- c(
  "Plant model", "Animal model", "Animal model", "Plant model",
  "Plant l+scoremodel", "Animal l+scoremodel",
  "Animal l+scoremodel", "Plant l+scoremodel",
  "Plant linearmodel", "Animal linearmodel",
  "Animal linearmodel", "Plant linearmodel"
)
accuracy_kingdoms <- c(
  "Plant data", "Animal data", "Plant data", "Animal data",
  "Plant data", "Animal data", "Plant data", "Animal data",
  "Plant data", "Animal data", "Plant data", "Animal data"
)

accuracy_all <- label_and_bind(
  accuracy_tables, accuracy_models, accuracy_kingdoms
)
plot_summary(
  accuracy_all, c(0, 0.8),
  file.path(PHOTO_DIR, "twokingdom_accuracy_deleteplot.svg")
)

compare_tables <- list(
  compare_summary(plant_acc),
  compare_summary(animal_acc),
  compare_summary(ap_acc),
  compare_summary(pa_acc),
  linear_compare_summary(plant_linacc),
  linear_compare_summary(animal_linacc),
  linear_compare_summary(ap_linacc),
  linear_compare_summary(pa_linacc),
  linear_basic_compare_summary(plant_linacc),
  linear_basic_compare_summary(animal_linacc),
  linear_basic_compare_summary(ap_linacc),
  linear_basic_compare_summary(pa_linacc)
)
compare_all <- label_and_bind(
  compare_tables, accuracy_models, accuracy_kingdoms
)
plot_summary(
  compare_all, c(0, 1),
  file.path(PHOTO_DIR, "twokingdom_decrease_delete_scorechangeplot.svg")
)
