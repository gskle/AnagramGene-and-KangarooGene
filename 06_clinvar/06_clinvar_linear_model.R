# Fit the KangarooGene-EN logistic calibration model on the animal deletion
# summary, then apply it to the ClinVar variants.
#
# The animal exon-length summary (animal_deleted_1cds_summ.csv) is generated
# by linear_ensemble_animal.R in 05_kangaroo and already contains the real/false
# exon features and the orthogroup partition. The KangarooGene deletion
# scores are attached from animal_deleted_1intron_newmodel.csv. For every
# fold the model is fitted on the remaining folds, and the wild-type and
# mutant ClinVar scores are converted into calibrated probabilities. The
# per-fold wild - mutant differences are saved as dele_en_change1..5, and
# the score differences as change1..5.
#
# Usage:
#   Rscript 06_clinvar_linear_model.R
# Output:
#   clinvar2024_deleteEN+delete+shuffle+pangolin+spliceai_shuffle_score_filterall.csv

suppressPackageStartupMessages({
  library(data.table)
})

ACC_DIR <- "/data1/lty/intron_pred/acc_out"
MODEL_DIR <- "/data1/lty/intron_pred/new_model/final_model"
CLINVAR_DIR <- "/data1/lty/clinvar2024"
SCAN_DIR <- file.path(CLINVAR_DIR, "new_mut/new_allscan")

ANIMAL_SUMM_CSV <- file.path(ACC_DIR, "animal_deleted_1cds_summ.csv")
ANIMAL_SCORE_CSV <- file.path(MODEL_DIR, "animal_deleted_1intron_newmodel.csv")
CLINVAR_IN_CSV <- file.path(
  SCAN_DIR,
  "clinvar2024_delete+shuffle+pangolin+spliceai_shuffle_score_filterall.csv"
)
CLINVAR_OUT_CSV <- file.path(
  SCAN_DIR,
  "clinvar2024_deleteEN+delete+shuffle+pangolin+spliceai_shuffle_score_filterall.csv"
)


# Turn one summary row into two training rows: native (label 1) and deleted
# (label 0), sharing the same gene but using the real/false exon features.
build_pair_frame <- function(summary_df) {
  real <- summary_df[, c(
    "gene_id", "real_num", "real_len", "real_meanlen", "real_minlen",
    "real_maxlen", "real_midlen"
  )]
  fal <- summary_df[, c(
    "gene_id", "fal_num", "fal_len", "fal_meanlen", "fal_minlen",
    "fal_maxlen", "fal_midlen"
  )]
  colnames(real) <- colnames(fal) <- c(
    "gene_id", "num", "len", "mean_len", "min_len", "max_len", "mid_len"
  )
  real$label <- 1
  fal$label <- 0
  real$score <- summary_df$pos_score
  fal$score <- summary_df$neg_score
  rbind(real, fal)
}


# Load the summary and attach the KangarooGene positive/negative scores.
animal_summary <- as.data.frame(fread(ANIMAL_SUMM_CSV))
pred_score <- as.data.frame(fread(ANIMAL_SCORE_CSV))
colnames(pred_score)[9:10] <- c("pos_score", "neg_score")
pred_score <- pred_score[, c("gene", "pos_score", "neg_score")]
animal_summary <- merge(
  animal_summary, pred_score, by.x = "gene_id", by.y = "gene"
)
animal_summary <- animal_summary[
  !is.na(animal_summary$pos_score) & !is.na(animal_summary$neg_score),
]

# Human exon-length features come from the animal summary by gene_id,
# by gene_id.
clinvar_pred <- as.data.frame(fread(CLINVAR_IN_CSV))
exon_feats <- animal_summary[, c(
  "gene_id", "real_num", "real_len", "real_meanlen",
  "real_minlen", "real_maxlen", "real_midlen"
)]
clinvar_pred <- merge(clinvar_pred, exon_feats, by = "gene_id")

for (fold in 1:5) {
  train_df <- build_pair_frame(
    animal_summary[animal_summary$group_type != fold, ]
  )
  add_model <- glm(
    label ~ len + mean_len + min_len + max_len + mid_len + score,
    data = train_df, family = binomial
  )
  basic_model <- glm(
    label ~ len + mean_len + min_len + max_len + mid_len,
    data = train_df, family = binomial
  )

  # Calibrate the wild-type KangarooGene score with real exon features.
  wild_use <- clinvar_pred[, c(
    "real_num", "real_len", "real_meanlen", "real_minlen", "real_maxlen",
    "real_midlen", paste0("wild_score", fold)
  )]
  colnames(wild_use) <- c(
    "num", "len", "mean_len", "min_len", "max_len", "mid_len", "score"
  )
  clinvar_pred[, paste0("dele_en_wildscore", fold)] <- predict(
    add_model, newdata = wild_use, type = "response"
  )

  # Calibrate the mutant KangarooGene score the same way.
  mut_use <- clinvar_pred[, c(
    "real_num", "real_len", "real_meanlen", "real_minlen", "real_maxlen",
    "real_midlen", paste0("score_mut", fold)
  )]
  colnames(mut_use) <- c(
    "num", "len", "mean_len", "min_len", "max_len", "mid_len", "score"
  )
  clinvar_pred[, paste0("dele_en_mutscore", fold)] <- predict(
    add_model, newdata = mut_use, type = "response"
  )

  clinvar_pred[, paste0("dele_en_change", fold)] <-
    clinvar_pred[, paste0("dele_en_wildscore", fold)] -
    clinvar_pred[, paste0("dele_en_mutscore", fold)]
  clinvar_pred[, paste0("change", fold)] <-
    clinvar_pred[, paste0("wild_score", fold)] -
    clinvar_pred[, paste0("score_mut", fold)]
}

write.csv(clinvar_pred, CLINVAR_OUT_CSV, row.names = FALSE)
cat("saved:", CLINVAR_OUT_CSV, "\n")
