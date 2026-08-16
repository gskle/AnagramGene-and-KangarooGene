# Merge all ClinVar annotation scores into one table.
#
# The deletion, shuffle, Pangolin and SpliceAI score tables are joined on
# the variant key. Gene-name consistency is checked through GENEINFO in the
# INFO field: a row is kept only when the ClinVar gene symbol matches the
# Ensembl gene symbol mapped from the representative transcript. One row per
# variant-gene pair is kept.
#
# Usage:
#   Rscript 04_merge_annotation_scores.R
# Output:
#   clinvar2024_delete+shuffle+pangolin+spliceai_shuffle_score_filterall.csv

suppressPackageStartupMessages({
  library(data.table)
})

CLINVAR_DIR <- "/data1/lty/clinvar2024"
SCAN_DIR <- file.path(CLINVAR_DIR, "new_mut/new_allscan")
ORTHO_CSV <- "/data1/lty/orthofinder_test/orth40spe_out.csv"

DELETE_CSV <- file.path(SCAN_DIR, "clinvar2024_delete_newanimal_div1_step2_annot_photo.csv")
SHUFFLE_CSV <- file.path(SCAN_DIR, "clinvar2024_shuffleintron_newanimal_div1_step2_annot_photo.csv")
PANGOLIN_CSV <- "/data1/lty/clinvar2024/intron_model/clinvar_2026_pangolin/clinvar2026_24intron_predout_pangolinall_step2.csv"
SPLICEAI_CSV <- "/data1/lty/clinvar2024/intron_model/clinvar_2026_spliceai/clinvar2026_24intron_predout_spliceaiall_step2.csv"
TX_INFO_CSV <- file.path(CLINVAR_DIR, "homo_tx_info.csv")
GENE_NAME_CSV <- file.path(CLINVAR_DIR, "homo_geneid_name.csv")
OUT_CSV <- file.path(
  SCAN_DIR,
  "clinvar2024_delete+shuffle+pangolin+spliceai_shuffle_score_filterall.csv"
)


# Extract the gene symbol stored in the GENEINFO INFO field.
extract_geneinfo <- function(info) {
  gene <- sub(".*GENEINFO=([^;:]+).*", "\\1", info)
  gene[!grepl("GENEINFO=", info)] <- NA
  gene
}


# Build the variant key used by every score table.
make_snp <- function(chrom, pos, ref, alt) {
  paste(chrom, pos, ref, alt, sep = "_")
}


# Rename a column only when it exists; the gene column may have different
# names across intermediate tables.
rename_col <- function(df, old, new) {
  if (old %in% colnames(df)) {
    colnames(df)[colnames(df) == old] <- new
  }
  df
}


delete_df <- as.data.frame(fread(DELETE_CSV))
shuffle_df <- as.data.frame(fread(SHUFFLE_CSV))
pangolin_df <- as.data.frame(fread(PANGOLIN_CSV))
spliceai_df <- as.data.frame(fread(SPLICEAI_CSV))
pangolin_df <- rename_col(pangolin_df, "pgolin_gene", "model_gene")
spliceai_df <- rename_col(spliceai_df, "spliceai_gene", "model_gene")

# Keep only variants scored by all four model families.
delete_df$snp <- make_snp(delete_df$`#CHROM`, delete_df$POS, delete_df$REF, delete_df$ALT)
shuffle_df$snp <- make_snp(shuffle_df$`#CHROM`, shuffle_df$POS, shuffle_df$REF, shuffle_df$ALT)
pangolin_df$snp <- make_snp(pangolin_df$`#CHROM`, pangolin_df$POS, pangolin_df$REF, pangolin_df$ALT)
spliceai_df$snp <- make_snp(spliceai_df$`#CHROM`, spliceai_df$POS, spliceai_df$REF, spliceai_df$ALT)

use_snp <- Reduce(
  intersect,
  list(delete_df$snp, shuffle_df$snp, pangolin_df$snp, spliceai_df$snp)
)
delete_df <- delete_df[delete_df$snp %in% use_snp, ]
shuffle_df <- shuffle_df[shuffle_df$snp %in% use_snp, ]
pangolin_df <- pangolin_df[pangolin_df$snp %in% use_snp, ]
spliceai_df <- spliceai_df[spliceai_df$snp %in% use_snp, ]

# Map representative transcripts and genes to gene symbols.
tx_info <- as.data.frame(fread(TX_INFO_CSV))
tx_info <- tx_info[, c("tx_name", "gene_id")]
id_name <- as.data.frame(fread(GENE_NAME_CSV))
colnames(id_name)[1] <- "gene_id"
id_name <- merge(tx_info, id_name, by = "gene_id")

delete_df$mut_gene_name <- extract_geneinfo(delete_df$INFO)
shuffle_df$mut_gene_name <- extract_geneinfo(shuffle_df$INFO)
pangolin_df$mut_gene_name <- extract_geneinfo(pangolin_df$INFO)
spliceai_df$mut_gene_name <- extract_geneinfo(spliceai_df$INFO)

delete_df <- merge(delete_df, id_name, by = "tx_name")
shuffle_df <- merge(shuffle_df, id_name, by.x = "tx_id", by.y = "tx_name")
gene_name <- id_name[!duplicated(id_name$gene_id), ]
pangolin_df <- merge(
  pangolin_df, gene_name[, c("gene_id", "gene_name")],
  by.x = "model_gene", by.y = "gene_id"
)

delete_df <- delete_df[delete_df$mut_gene_name == delete_df$gene_name, ]
shuffle_df <- shuffle_df[shuffle_df$mut_gene_name == shuffle_df$gene_name, ]
pangolin_df <- pangolin_df[pangolin_df$mut_gene_name == pangolin_df$gene_name, ]
spliceai_df <- spliceai_df[spliceai_df$mut_gene_name == spliceai_df$model_gene, ]

# Deduplicate on variant + gene symbol.
delete_df$snp_gene <- paste(delete_df$snp, delete_df$mut_gene_name, sep = "_")
shuffle_df$snp_gene <- paste(shuffle_df$snp, shuffle_df$mut_gene_name, sep = "_")
pangolin_df$snp_gene <- paste(pangolin_df$snp, pangolin_df$mut_gene_name, sep = "_")
spliceai_df$snp_gene <- paste(spliceai_df$snp, spliceai_df$mut_gene_name, sep = "_")

delete_df <- delete_df[!duplicated(delete_df$snp_gene), ]
shuffle_df <- shuffle_df[!duplicated(shuffle_df$snp_gene), ]
pangolin_df <- pangolin_df[!duplicated(pangolin_df$snp_gene), ]
spliceai_df <- spliceai_df[!duplicated(spliceai_df$snp_gene), ]

# Rename shuffle change columns before building the final table.
shuffle_cols <- c(
  "change1", "change2", "change3", "change4", "change5", "score_change"
)
shuffle_new <- c(
  "shuffle_change1", "shuffle_change2", "shuffle_change3",
  "shuffle_change4", "shuffle_change5", "shuffle_score"
)
rename_idx <- match(shuffle_cols, colnames(shuffle_df))
if (!any(is.na(rename_idx))) {
  colnames(shuffle_df)[rename_idx] <- shuffle_new
}

shuffle_use <- shuffle_df[, c(
  "snp_gene",
  "shuffle_change1", "shuffle_change2", "shuffle_change3",
  "shuffle_change4", "shuffle_change5", "shuffle_score"
)]
pangolin_use <- pangolin_df[, c("snp_gene", "score_change")]
colnames(pangolin_use)[2] <- "pangolin_score"
spliceai_use <- spliceai_df[, c("snp_gene", "score_change")]
colnames(spliceai_use)[2] <- "spliceai_score"

merged <- merge(delete_df, shuffle_use, by = "snp_gene", all.x = TRUE)
merged <- merge(merged, pangolin_use, by = "snp_gene", all.x = TRUE)
merged <- merge(merged, spliceai_use, by = "snp_gene", all.x = TRUE)

# Plain variant key (#CHROM_POS_REF_ALT) used by the downstream PI scoring
# scripts after the score-table merge.
merged$snp <- paste(
  merged$`#CHROM`, merged$POS, merged$REF, merged$ALT, sep = "_"
)

# Add the orthogroup partition used by the five-fold random forest.
gene_family <- as.data.frame(fread(ORTHO_CSV))
gene_family <- gene_family[, c("gene", "group_type")]
gene_family$gene <- vapply(
  strsplit(as.character(gene_family$gene), ".", fixed = TRUE),
  `[[`, character(1), 1
)
merged <- merge(merged, gene_family, by.x = "gene_id", by.y = "gene")

# Recreate the binary ClinVar labels when the score tables do not carry them.
if (!"benign" %in% colnames(merged) || !"Pathogenic" %in% colnames(merged)) {
  merged$benign <- as.integer(grepl("CLNSIG=Benign", merged$INFO))
  merged$Pathogenic <- as.integer(grepl("CLNSIG=Pathogenic", merged$INFO))
  merged <- merged[(merged$benign == 1) | (merged$Pathogenic == 1), ]
}
if (!"pheno" %in% colnames(merged)) {
  merged$pheno <- "benign"
  merged$pheno[merged$Pathogenic == 1] <- "pathogenic"
}
if (!"mut_type" %in% colnames(merged)) {
  # snpEff stores the annotation after ";ANN="; the first pipe field is the
  # consequence type and the transcript match is handled by gene filtering.
  ann <- sub("^.*;ANN=", "", merged$INFO)
  first_consequence <- function(x) {
    x <- strsplit(x, ",", fixed = TRUE)[[1]][1]
    strsplit(x, "|", fixed = TRUE)[[1]][2]
  }
  merged$mut_type <- vapply(ann, first_consequence, character(1))
  merged$mut_type[grepl("stop_gained", merged$INFO)] <- "nonsense"
}

write.csv(merged, OUT_CSV, row.names = FALSE)
cat("saved:", OUT_CSV, "(", nrow(merged), "rows )\n")
