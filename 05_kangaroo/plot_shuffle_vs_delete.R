# Compare AnagramGene-SI scores on shuffled transcripts with KangarooGene
# scores on one-intron-deleted transcripts.
#
# The shuffle-for-deletion prediction files are written by
# `04_anagram_si/predict_on_deleted_sequences.py`; the deletion prediction
# files are written by the KangarooGene training scripts. Two plots show
# classification accuracy and the frequency of score decrease.
#
# Usage:
#   Rscript plot_shuffle_vs_delete.R

suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
})

MODEL_DIR <- "/data1/lty/intron_pred/new_model/final_model"
ACC_DIR <- "/data1/lty/intron_pred/acc_out"
PHOTO_DIR <- "/data1/lty/intron_pred/photo"

PLANT_SHUFFLE_CSV <- file.path(PHOTO_DIR, "plant_shuffle_for_dele_pred_out.csv")
ANIMAL_SHUFFLE_CSV <- file.path(PHOTO_DIR, "animal_shuffle_for_dele_pred_out.csv")
PLANT_DELETE_CSV <- file.path(MODEL_DIR, "plant_deleted_1model_predout.csv")
ANIMAL_DELETE_CSV <- file.path(MODEL_DIR, "animal_deleted_1intron_newmodel.csv")


# Normalise the prediction tables to group_type/pos_score/neg_score.
normalise_score_columns <- function(df) {
  df <- as.data.frame(df)
  if (all(c("pos_score", "neg_score") %in% colnames(df))) {
    return(df[, c("group_type", "pos_score", "neg_score")])
  }
  if (all(c("pred_out", "pred_negout") %in% colnames(df))) {
    out <- df[, c("group_type", "pred_out", "pred_negout")]
    colnames(out)[2:3] <- c("pos_score", "neg_score")
    return(out)
  }
  stop("cannot locate pos/neg score columns in ", deparse(substitute(df)))
}


# Accuracy: fraction of native rows > 0.5 plus shuffled/deleted rows < 0.5.
shuffle_accuracy_summary <- function(df) {
  df %>%
    group_by(group_type) %>%
    summarise(
      acc = (sum(pos_score > 0.5) + sum(neg_score < 0.5)) / (n() * 2),
      .groups = "drop"
    )
}


# Score decrease: fraction of genes whose native score beats the negative one.
shuffle_compare_summary <- function(df) {
  df %>%
    group_by(group_type) %>%
    summarise(
      acc = sum(pos_score > neg_score) / n(),
      .groups = "drop"
    )
}


# Add the comparison key and plot mean +/- SE with per-fold jitter.
plot_comparison <- function(tables, keys, y_limits, out_file) {
  all_df <- do.call(rbind, lapply(seq_along(tables), function(i) {
    d <- tables[[i]]
    d$key <- keys[[i]]
    d
  }))

  error_df <- all_df %>%
    group_by(key) %>%
    summarise(
      mean_acc = mean(acc),
      se = sd(acc) / sqrt(n()),
      .groups = "drop"
    )

  p <- ggplot(error_df, aes(y = mean_acc, x = key, fill = key)) +
    geom_col(width = 0.7) +
    geom_errorbar(
      aes(ymin = mean_acc - se, ymax = mean_acc + se), width = 0.2
    ) +
    geom_jitter(
      data = all_df, aes(y = acc, x = key),
      width = 0.15, height = 0, size = 1, color = "black", alpha = 0.8
    ) +
    scale_y_continuous(limits = y_limits, expand = c(0, 0)) +
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


plant_shuffle <- normalise_score_columns(read.csv(PLANT_SHUFFLE_CSV))
animal_shuffle <- normalise_score_columns(read.csv(ANIMAL_SHUFFLE_CSV))
plant_delete <- normalise_score_columns(read.csv(PLANT_DELETE_CSV))
animal_delete <- normalise_score_columns(read.csv(ANIMAL_DELETE_CSV))

accuracy_tables <- list(
  shuffle_accuracy_summary(plant_shuffle),
  shuffle_accuracy_summary(plant_delete),
  shuffle_accuracy_summary(animal_shuffle),
  shuffle_accuracy_summary(animal_delete)
)
accuracy_keys <- c(
  "plant_shuffle", "plant_dele", "animal_shuffle", "animal_dele"
)
plot_comparison(
  accuracy_tables, accuracy_keys, c(0, 0.8),
  file.path(PHOTO_DIR, "model_real_accuracy_deleteplot.svg")
)

compare_tables <- list(
  shuffle_compare_summary(plant_shuffle),
  shuffle_compare_summary(plant_delete),
  shuffle_compare_summary(animal_shuffle),
  shuffle_compare_summary(animal_delete)
)
plot_comparison(
  compare_tables, accuracy_keys, c(0, 1),
  file.path(PHOTO_DIR, "model_compare_accuracy_deleteplot.svg")
)
