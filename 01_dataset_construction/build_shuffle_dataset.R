# Build shuffled-intron training sequences for one species.
#
# For every gene, the representative transcript (longest CDS) is used. The
# positive sequence is the full transcript sequence from TSS to TTS with the
# introns in native order; the negative sequence is identical except that the
# order of the introns is shuffled. Only transcripts with at least
# INTRON_LEAST introns are kept (default two introns).
#
# Usage:
#   Rscript build_shuffle_dataset.R <species_index> [intron_least]
# Species index is 1-based into the SPECIES / GTF / FASTA vectors below.
# Output:
#   <species>_intron_shuffle.csv

library(GenomicFeatures)
library(BSgenome)
library(dplyr)

args <- commandArgs(trailingOnly = TRUE)
i <- as.integer(args[1])
INTRON_LEAST <- ifelse(length(args) >= 2, as.integer(args[2]), 2)

SPECIES <- c("Ath", "Bdi", "Bvu", "Cre", "Csa", "Gma", "Mtr", "Osa", "Ppa",
             "Ptr", "Sbi", "Sit", "Sly", "Stu", "Svi", "Vvi", "Zma", "Tur")
GTF_PATHS <- c(
  "/home/litianyi/Regulatory_region_translation/0_genomes/annotation/Arabidopsis_thaliana.TAIR10.45.gtf",
  "/home/litianyi/Regulatory_region_translation/0_genomes/annotation/Bdistachyon_314_v3.1.gene_exons.gff3",
  "/home/litianyi/Regulatory_region_translation/0_genomes/annotation/Beta_vulgaris.RefBeet-1.2.2.46.gtf",
  "/home/litianyi/Regulatory_region_translation/0_genomes/annotation/Chlamydomonas_reinhardtii_v5.5.45.gtf",
  "/home/litianyi/Regulatory_region_translation/0_genomes/annotation/Cucumis_sativus.ASM407v2.46.gtf",
  "/home/litianyi/Regulatory_region_translation/0_genomes/annotation/Gmax_508_Wm82.a4.v1.gene_exons.gff3",
  "/home/litianyi/Regulatory_region_translation/0_genomes/annotation/Mtruncatula_285_Mt4.0v1.gene_exons.gff3",
  "/home/litianyi/Regulatory_region_translation/0_genomes/annotation/Osativa_323_v7.0.gene_exons.gff3",
  "/home/litianyi/Regulatory_region_translation/0_genomes/annotation/Physcomitrella_patens.Phypa_V3.45.gtf",
  "/home/litianyi/Regulatory_region_translation/0_genomes/annotation/Ptrichocarpa_533_v4.1.gene_exons.gff3",
  "/home/litianyi/Regulatory_region_translation/0_genomes/annotation/Sbicolor_454_v3.1.1.gene_exons.gff3",
  "/home/litianyi/Regulatory_region_translation/0_genomes/annotation/Sitalica_312_v2.2.gene_exons.gff3",
  "/home/litianyi/Regulatory_region_translation/0_genomes/annotation/Slycopersicum_514_ITAG3.2.gene_exons.gff3",
  "/home/litianyi/Regulatory_region_translation/0_genomes/annotation/Stuberosum_448_v4.03.gene_exons.gff3",
  "/home/litianyi/Regulatory_region_translation/0_genomes/annotation/Sviridis_500_v2.1.gene_exons.gff3",
  "/home/litianyi/Regulatory_region_translation/0_genomes/annotation/Vitis_vinifera.12X.46.gtf",
  "/home/litianyi/Regulatory_region_translation/0_genomes/annotation/Zea_mays.Zm-B73-REFERENCE-NAM-5.0.51.gtf",
  "/home/hai/poaceae_genome/Tur/Triticum_urartu.IGDB.55.gtf"
)
FA_PATHS <- c(
  "/home/litianyi/Regulatory_region_translation/0_genomes/genomic_sequences/Arabidopsis_thaliana.TAIR10.dna.toplevel.fa",
  "/home/litianyi/Regulatory_region_translation/0_genomes/genomic_sequences/Bdistachyon_314_v3.0.fa",
  "/home/litianyi/Regulatory_region_translation/0_genomes/genomic_sequences/Beta_vulgaris.RefBeet-1.2.2.dna.toplevel.fa",
  "/home/litianyi/Regulatory_region_translation/0_genomes/genomic_sequences/Chlamydomonas_reinhardtii_v5.5.dna.toplevel.fa",
  "/home/litianyi/Regulatory_region_translation/0_genomes/genomic_sequences/Cucumis_sativus.ASM407v2.dna.toplevel.fa",
  "/home/litianyi/Regulatory_region_translation/0_genomes/genomic_sequences/Gmax_508_v4.0.fa",
  "/home/litianyi/Regulatory_region_translation/0_genomes/genomic_sequences/Mtruncatula_285_Mt4.0.fa",
  "/home/litianyi/Regulatory_region_translation/0_genomes/genomic_sequences/Osativa_323_v7.0.fa",
  "/home/litianyi/Regulatory_region_translation/0_genomes/genomic_sequences/Physcomitrella_patens.Phypa_V3.dna.toplevel.fa",
  "/home/litianyi/Regulatory_region_translation/0_genomes/genomic_sequences/Ptrichocarpa_533_v4.0.fa",
  "/home/litianyi/Regulatory_region_translation/0_genomes/genomic_sequences/Sbicolor_454_v3.0.1.fa",
  "/home/litianyi/Regulatory_region_translation/0_genomes/genomic_sequences/Sitalica_312_v2.fa",
  "/home/litianyi/Regulatory_region_translation/0_genomes/genomic_sequences/Slycopersicum_514_SL3.0.fa",
  "/home/litianyi/Regulatory_region_translation/0_genomes/genomic_sequences/Stuberosum_448_v4.03.fa",
  "/home/litianyi/Regulatory_region_translation/0_genomes/genomic_sequences/Sviridis_500_v2.0.fa",
  "/home/litianyi/Regulatory_region_translation/0_genomes/genomic_sequences/Vitis_vinifera.12X.dna.toplevel.fa",
  "/home/litianyi/Regulatory_region_translation/0_genomes/genomic_sequences/Zea_mays.Zm-B73-REFERENCE-NAM-5.0.dna.toplevel.fa",
  "/home/hai/poaceae_genome/Tur/Triticum_urartu.IGDB.dna.toplevel.fa"
)

OUT_DIR <- "/home/litianyi/Regulatory_region_translation/intron_shuffle"

fa <- readDNAStringSet(FA_PATHS[i])
names(fa) <- sapply(strsplit(names(fa), " "), "[", 1)
txdb <- makeTxDbFromGFF(GTF_PATHS[i])

# Representative transcript: longest CDS per gene.
cds_list <- cdsBy(txdb, by = "tx", use.names = TRUE)
tx_len <- transcriptLengths(txdb)
tx_gene <- tx_len[, c("tx_name", "gene_id")]
cds_ranges <- unlist(range(cds_list))
cds_len_all <- data.frame(
  len = as.integer(width(cds_ranges)),
  tx_id = names(cds_ranges),
  gene_id = tx_gene$gene_id[match(names(cds_ranges), tx_gene$tx_name)]
)
cds_longest <- as.data.frame(
  cds_len_all %>%
    filter(!is.na(gene_id)) %>%
    group_by(gene_id) %>%
    arrange(desc(len)) %>%
    slice(1)
)

exons <- exonsBy(txdb, use.names = TRUE, by = "tx")
introns <- intronsByTranscript(txdb, use.names = TRUE)

# Interleave exons and introns: exon, intron, exon, intron, ...
interlace <- function(a, b) {
  all_pieces <- c(a, b)
  a_ranks <- seq(from = 1, length.out = length(a), by = 2)
  b_ranks <- seq(from = 2, length.out = length(b), by = 2)
  all_pieces[order(c(a_ranks, b_ranks))]
}

data <- data.frame(tx_rep = cds_longest$tx_id, stringsAsFactors = FALSE)
data$whole_seq_real <- NA_character_
data$whole_seq_false <- NA_character_
data$iranks_shuffled_label <- NA_character_
data$whole_type_real <- NA_character_
data$whole_type_false <- NA_character_
data$intron_counts <- NA_integer_
data$seqs_len <- NA_integer_
rownames(data) <- cds_longest$tx_id

txids <- as.character(data$tx_rep)
for (j in seq_along(txids)) {
  txid <- txids[j]
  es <- exons[[txid]]
  is <- introns[[txid]]
  if (length(is) < INTRON_LEAST) {
    next
  }
  strand <- as.character(strand(es)[1])
  if (strand == "+") {
    es <- sort(es)
    is <- sort(is)
  } else {
    es <- sort(es, decreasing = TRUE)
    is <- sort(is, decreasing = TRUE)
  }

  eseqs <- getSeq(fa, es)
  etypes <- sapply(width(es), function(x) paste(rep("e", x), collapse = ""))
  iseqs <- getSeq(fa, is)
  itypes <- sapply(width(is), function(x) paste(rep("i", x), collapse = ""))
  iranks <- seq_along(iseqs)

  # Shuffle intron order and require the new order to differ from the old one.
  repeat {
    iranks_shuffled <- sample(iranks)
    if (sum(iranks_shuffled != iranks) > 0) {
      break
    }
  }
  iranks_shuffled_label <- paste(iranks_shuffled, collapse = "%")
  iseqs_shuffled <- iseqs[iranks_shuffled]
  itypes_shuffled <- itypes[iranks_shuffled]

  data[txid, "whole_seq_real"] <- paste(unlist(interlace(eseqs, iseqs)), collapse = "")
  data[txid, "whole_type_real"] <- paste(interlace(etypes, itypes), collapse = "")
  data[txid, "whole_seq_false"] <- paste(unlist(interlace(eseqs, iseqs_shuffled)), collapse = "")
  data[txid, "whole_type_false"] <- paste(interlace(etypes, itypes_shuffled), collapse = "")
  data[txid, "iranks_shuffled_label"] <- iranks_shuffled_label
  data[txid, "intron_counts"] <- length(is)
  data[txid, "seqs_len"] <- nchar(data[txid, "whole_seq_real"])
}

data$spe <- SPECIES[i]
write.csv(data, file.path(OUT_DIR, paste0(SPECIES[i], "_intron_shuffle.csv")),
          row.names = FALSE)
