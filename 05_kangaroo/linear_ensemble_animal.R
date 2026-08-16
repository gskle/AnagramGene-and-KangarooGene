# Fit the animal exon-length linear model (LM5) and the KangarooGene-EN
# ensemble used for the deletion experiment.
#
# LM5 predicts native-vs-deleted status from five exon-length features
# (total, mean, minimum, maximum, median). KangarooGene-EN adds the
# deletion-model score to the same features. Every orthogroup fold is held
# out in turn; the fitted models are applied to the held-out animal genes
# and to plant genes scored with the animal deletion model.
#
# Usage:
#   Rscript linear_ensemble_animal.R

suppressPackageStartupMessages({
  library(data.table)
})

ANIMAL_DELETED_CSV <- "/data1/lty/intron_pred/data/animal_deleted_all.csv"
PLANT_DELETED_CSV <- "/data1/lty/intron_pred/data/plant_deleted_1intron.csv"
ORTHO_CSV <- "/data1/lty/intron_pred/data/orth40spe_out.csv"

OUT_DIR <- "/data1/lty/intron_pred/acc_out"
ANIMAL_SCORE_CSV <- "/data1/lty/intron_pred/new_model/final_model/animal_deleted_1intron_newmodel.csv"
PLANT_SCORE_CSV <- "/data1/lty/intron_pred/new_model/final_model/animaltoplant_deleted_1model.csv"
PLANT_SUMM_CSV <- file.path(OUT_DIR, "plant_deleted_1cds_summ.csv")
ANIMAL_SUMM_CSV <- file.path(OUT_DIR, "animal_deleted_1cds_summ.csv")
ANIMAL_LINEAR_CSV <- file.path(OUT_DIR, "animal_dele_linear_predout.csv")
PLANT_LINEAR_CSV <- file.path(OUT_DIR, "animaltoplant_dele_linear_predout.csv")

ANIMAL_SUFFIX_SPECIES <- c(
  "Bta", "Chi", "Cin", "Dre", "Fca", "Gga", "Homo",
  "Mmu", "Nna", "Ptro", "Ssa", "Xtr"
)


# Summarise the exon segments encoded in a whole_type string.
# The segments are concatenated with lowercase "i" as separator.
exon_stats_from_whole_type <- function(whole_type) {
  parts <- strsplit(whole_type, "i", fixed = TRUE)
  parts <- lapply(parts, function(x) x[x != ""])
  out <- lapply(parts, function(x) {
    len <- nchar(x)
    if (length(len) == 0) {
      return(data.frame(
        num = 0L, len = 0L, mean_len = NA_real_, min_len = NA_real_,
        max_len = NA_real_, mid_len = NA_real_
      ))
    }
    data.frame(
      num = length(len), len = sum(len), mean_len = mean(len),
      min_len = min(len), max_len = max(len), mid_len = median(len)
    )
  })
  do.call(rbind, out)
}


# Build the exon-length summary for one deletion table and save it.
build_exon_summary <- function(deleted_df, out_csv) {
  fal_stats <- exon_stats_from_whole_type(deleted_df$whole_type_false)
  real_stats <- exon_stats_from_whole_type(deleted_df$whole_type_real)
  colnames(fal_stats) <- paste0("fal_", colnames(fal_stats))
  colnames(real_stats) <- paste0("real_", colnames(real_stats))

  summary_df <- cbind(
    data.frame(gene_id = deleted_df$gene_id, stringsAsFactors = FALSE),
    fal_stats,
    real_stats
  )
  # Keep only transcripts whose native and deleted sequences
  # have the same total exon length.
  summary_df <- summary_df[summary_df$fal_len == summary_df$real_len, ]

  ortho <- load_ortho()
  summary_df <- merge(summary_df, ortho, by.x = "gene_id", by.y = "gene")
  write.csv(summary_df, out_csv, row.names = FALSE)
  summary_df
}


# Return the orthogroup table with animal gene names normalised.
# Animal gene ids in the ortho table carry a transcript suffix; the deletion
# table stores the unsuffixed gene id.
load_ortho <- function() {
  ortho <- as.data.frame(fread(ORTHO_CSV))
  ortho <- ortho[, c("gene", "spe", "group_type")]
  mask <- ortho$spe %in% ANIMAL_SUFFIX_SPECIES
  ortho$gene[mask] <- vapply(
    strsplit(ortho$gene[mask], ".", fixed = TRUE),
    `[[`, character(1), 1
  )
  ortho[, c("gene", "group_type")]
}


# Attach deletion-model scores to an exon-length summary.
attach_scores <- function(summary_df, score_df) {
  score_df <- as.data.frame(score_df)
  score_df <- score_df[, c("gene", "pos_score", "neg_score")]
  merged <- merge(summary_df, score_df, by.x = "gene_id", by.y = "gene")
  merged[!is.na(merged$pos_score) & !is.na(merged$neg_score), ]
}


# Turn one summary row into two training rows: native (label 1) and deleted
# (label 0), sharing the same gene but using real/false exon features.
build_pair_frame <- function(summary_df) {
  real <- summary_df[, c(
    "gene_id", "real_num", "real_len", "real_meanlen", "real_minlen",
    "real_maxlen", "real_midlen", "group_type"
  )]
  fal <- summary_df[, c(
    "gene_id", "fal_num", "fal_len", "fal_meanlen", "fal_minlen",
    "fal_maxlen", "fal_midlen", "group_type"
  )]
  colnames(real) <- c(
    "gene_id", "num", "len", "mean_len", "min_len", "max_len", "mid_len",
    "group_type"
  )
  colnames(fal) <- colnames(real)
  real$label <- 1
  fal$label <- 0
  real$score <- summary_df$pos_score
  fal$score <- summary_df$neg_score
  rbind(real, fal)
}


# Predict one held-out fold and return real-vs-false pairs per gene.
score_fold <- function(fold_summary, add_model, basic_model) {
  test_df <- build_pair_frame(fold_summary)
  test_df$pred_add <- predict(add_model, newdata = test_df, type = "response")
  test_df$pred_basic <- predict(basic_model, newdata = test_df, type = "response")

  test_real <- test_df[test_df$label == 1,
                       c("gene_id", "group_type", "pred_add", "pred_basic")]
  test_fal <- test_df[test_df$label == 0,
                      c("gene_id", "pred_add", "pred_basic")]
  colnames(test_real)[3:4] <- c("pred_add_real", "pred_basic_real")
  colnames(test_fal)[2:3] <- c("pred_add_fal", "pred_basic_fal")
  merge(test_real, test_fal, by = "gene_id")
}


# Fit both models on every fold and save animal and plant prediction tables.
run_folds <- function(animal_summary, plant_summary) {
  animal_out <- data.frame()
  plant_out <- data.frame()

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

    animal_out <- rbind(
      animal_out,
      score_fold(animal_summary[animal_summary$group_type == fold, ],
                 add_model, basic_model)
    )
    plant_out <- rbind(
      plant_out,
      score_fold(plant_summary[plant_summary$group_type == fold, ],
                 add_model, basic_model)
    )
  }

  write.csv(animal_out, ANIMAL_LINEAR_CSV, row.names = FALSE)
  write.csv(plant_out, PLANT_LINEAR_CSV, row.names = FALSE)
  cat("saved:", ANIMAL_LINEAR_CSV, "and", PLANT_LINEAR_CSV, "\n")
}


animal_df <- as.data.frame(fread(ANIMAL_DELETED_CSV))
animal_summary <- build_exon_summary(animal_df, ANIMAL_SUMM_CSV)
animal_summary <- attach_scores(
  animal_summary, fread(ANIMAL_SCORE_CSV)
)

plant_df <- as.data.frame(fread(PLANT_DELETED_CSV))
plant_summary <- build_exon_summary(plant_df, PLANT_SUMM_CSV)
plant_summary <- attach_scores(plant_summary, fread(PLANT_SCORE_CSV))

run_folds(animal_summary, plant_summary)
