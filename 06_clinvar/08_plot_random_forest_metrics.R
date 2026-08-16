# Plot ClinVar random-forest metrics by model and held-out fold.
# Accuracy, Precision, Recall and auPRC are read from the files written by
# 07_train_clinvar_random_forest.py. F1 and MCC are calculated from predictions.

suppressPackageStartupMessages({
  library(data.table)
  library(ggplot2)
  library(dplyr)
})

CLINVAR_DIR <- "/data1/lty/clinvar2024/new_mut/new_allscan"
PRED_CSV <- file.path(CLINVAR_DIR, "clinvar2024_random_forest_pred_out.csv")
PLOT_DIR <- file.path(CLINVAR_DIR, "random_forest_plots")
SUMMARY_CSV <- file.path(CLINVAR_DIR, "random_forest_f1_mcc.csv")

dir.create(PLOT_DIR, recursive = TRUE, showWarnings = FALSE)

model_id <- c(
  "shuffle5", "dele_en5", "dele_en5+shuffle5", "spliceai", "pangolin",
  "pangolin+spliceai", "dele_en5+shuffle5+pangolin+spliceai"
)

calc_metrics <- function(label, score) {
  keep <- !is.na(label) & !is.na(score)
  label <- as.integer(label[keep])
  score <- as.numeric(score[keep])
  pred <- as.integer(score >= 0.5)

  tp <- sum(label == 1 & pred == 1)
  tn <- sum(label == 0 & pred == 0)
  fp <- sum(label == 0 & pred == 1)
  fn <- sum(label == 1 & pred == 0)
  precision <- if ((tp + fp) == 0) NA_real_ else tp / (tp + fp)
  recall <- if ((tp + fn) == 0) NA_real_ else tp / (tp + fn)
  f1 <- if (is.na(precision) || is.na(recall) || precision + recall == 0) {
    NA_real_
  } else {
    2 * precision * recall / (precision + recall)
  }
  denominator <- sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
  mcc <- if (denominator == 0) NA_real_ else (tp * tn - fp * fn) / denominator

  data.frame(
    F1 = f1,
    MCC = mcc
  )
}

read_python_metric <- function(filename, metric) {
  path <- file.path(CLINVAR_DIR, "final_modelnew", filename)
  data <- fread(path, header = FALSE, sep = "\t")
  colnames(data) <- c("id", metric)
  parts <- tstrsplit(as.character(data$id), "_", fixed = TRUE)
  data$group_type <- as.integer(parts[[length(parts)]])
  data$model_type <- sub("_[^_]+$", "", as.character(data$id))
  data[, c("model_type", "group_type", metric), with = FALSE]
}

pred_out <- as.data.frame(fread(PRED_CSV))
pred_out <- pred_out[pred_out$model_type %in% model_id, ]

result <- pred_out %>%
  group_by(model_type, group_type) %>%
  group_modify(~ calc_metrics(.x$label, .x$pred_out)) %>%
  ungroup()

python_metrics <- rbindlist(list(
  read_python_metric("random_forest_modelacc_multi_5foldup", "Accuracy"),
  read_python_metric("random_forest_modelprec_multi_5foldup", "Precision"),
  read_python_metric("random_forest_modelrecall_multi_5foldup", "Recall"),
  read_python_metric("random_forest_modelauprc_multi_5foldup", "auPRC")
), fill = TRUE)

f1_mcc <- result[, c("model_type", "group_type", "F1", "MCC")]
all_metrics <- merge(
  python_metrics,
  f1_mcc,
  by = c("model_type", "group_type"),
  all = TRUE
)
write.csv(all_metrics, file.path(CLINVAR_DIR, "random_forest_metrics.csv"), row.names = FALSE)
write.csv(f1_mcc, SUMMARY_CSV, row.names = FALSE)

plot_metric <- function(metric, filename, y_limits = NULL) {
  metric_data <- if (metric %in% c("F1", "MCC")) result else python_metrics
  summary <- metric_data %>%
    group_by(model_type) %>%
    summarise(
      mean_value = mean(.data[[metric]], na.rm = TRUE),
      sd_value = sd(.data[[metric]], na.rm = TRUE),
      .groups = "drop"
    )
  summary$model_type <- factor(summary$model_type, levels = model_id)

  p <- ggplot(summary, aes(x = model_type, y = mean_value, fill = model_type)) +
    geom_bar(
      stat = "identity",
      position = position_dodge(width = 0.5),
      width = 0.5,
      colour = "black",
      linewidth = 1
    ) +
    geom_jitter(
      data = metric_data,
      aes(x = model_type, y = .data[[metric]]),
      position = position_jitterdodge(dodge.width = 0.5, jitter.width = 0.3),
      size = 1,
      shape = 21,
      inherit.aes = FALSE
    ) +
    geom_errorbar(
      aes(ymin = mean_value - sd_value, ymax = mean_value + sd_value),
      position = position_dodge(width = 0.5),
      width = 0.25
    ) +
    coord_cartesian(ylim = y_limits) +
    theme(
      axis.text = element_text(size = 11),
      panel.border = element_rect(fill = NA, color = "black", linewidth = 0.7),
      panel.background = element_blank(),
      axis.text.x = element_text(angle = 45, hjust = 1)
    )
  ggsave(file.path(PLOT_DIR, filename), p, width = 10, height = 5)
}

plot_metric("Accuracy", "model_deleterious_pred_acc.svg", c(0, 1))
plot_metric("Precision", "model_deleterious_pred_prec.svg", c(0, 1))
plot_metric("Recall", "model_deleterious_pred_recall.svg", c(0, 1))
plot_metric("auPRC", "model_deleterious_pred_auprc.svg", c(0, 1))
plot_metric("F1", "model_deleterious_pred_f1.svg", c(0, 1))
plot_metric("MCC", "model_deleterious_pred_mcc.svg")

cat("saved:", SUMMARY_CSV, "\n")
cat("saved plots:", PLOT_DIR, "\n")
