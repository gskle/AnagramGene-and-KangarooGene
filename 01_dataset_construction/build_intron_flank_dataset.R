# Build intron + flank training sequences for one species.
#
# Each intron is extended by FLANK_BP bases on both sides (N-padded at the
# sequence ends). Positive sequences concatenate these units in native
# order with five Ns; negative sequences use a shuffled intron order. The
# whole_type columns carry a/d/i labels used by the region ablation: a =
# acceptor-side region, i = intron interior, d = donor-side region.
#
# Usage:
#   Rscript build_intron_flank_dataset.R <species_index> [flank_bp] [intron_least]
# Output:
#   <species>_intron_with_flank.csv

library(GenomicFeatures)
library(BSgenome)
library(dplyr)

args <- commandArgs(trailingOnly = TRUE)
i <- as.integer(args[1])
FLANK_BP <- ifelse(length(args) >= 2, as.integer(args[2]), 10)
INTRON_LEAST <- ifelse(length(args) >= 3, as.integer(args[3]), 2)

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

OUT_DIR <- "/home/litianyi/Regulatory_region_translation/intron_flank"

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

introns <- intronsByTranscript(txdb, use.names = TRUE)
sep <- paste(rep("N", 5), collapse = "")
sep_type <- paste(rep("n", 5), collapse = "")

data <- data.frame(tx_rep = cds_longest$tx_id, stringsAsFactors = FALSE)
data$whole_seq_real <- NA_character_
data$whole_seq_false <- NA_character_
data$iranks_shuffled_label <- NA_character_
data$whole_type_real <- NA_character_
data$whole_type_false <- NA_character_
data$intron_counts <- NA_integer_
data$seqs_len <- NA_integer_
rownames(data) <- cds_longest$tx_id

# Label one intron unit: a = acceptor flank + first 10 bp of the intron,
# i = intron interior, d = last 10 bp of the intron + donor flank.
build_unit <- function(gr) {
  chr <- as.character(seqnames(gr))
  intron_start <- start(gr)
  intron_end <- end(gr)
  chr_len <- length(fa[[chr]])
  intron_len <- intron_end - intron_start + 1

  flank_start <- max(1, intron_start - FLANK_BP)
  flank_end <- min(chr_len, intron_end + FLANK_BP)

  left_flank_len <- intron_start - flank_start
  if (left_flank_len < FLANK_BP) {
    seq_left <- paste(rep("N", FLANK_BP - left_flank_len), collapse = "")
    if (flank_start <= intron_start - 1) {
      seq_left <- paste0(seq_left,
                         as.character(subseq(fa[[chr]], flank_start, intron_start - 1)))
    }
  } else {
    seq_left <- as.character(subseq(fa[[chr]], intron_start - FLANK_BP,
                                    intron_start - 1))
  }

  seq_intron <- as.character(subseq(fa[[chr]], intron_start, intron_end))

  right_flank_len <- flank_end - intron_end
  if (right_flank_len < FLANK_BP) {
    seq_right <- as.character(subseq(fa[[chr]], intron_end + 1, flank_end))
    seq_right <- paste0(seq_right,
                        paste(rep("N", FLANK_BP - right_flank_len), collapse = ""))
  } else {
    seq_right <- as.character(subseq(fa[[chr]], intron_end + 1,
                                     intron_end + FLANK_BP))
  }

  full_seq <- paste0(seq_left, seq_intron, seq_right)
  if (intron_len > 20) {
    full_type <- paste0(
      paste(rep("a", FLANK_BP), collapse = ""),
      paste(rep("a", 10), collapse = ""),
      paste(rep("i", intron_len - 20), collapse = ""),
      paste(rep("d", 10), collapse = ""),
      paste(rep("d", FLANK_BP), collapse = "")
    )
  } else if (intron_len > 10) {
    full_type <- paste0(
      paste(rep("a", FLANK_BP), collapse = ""),
      paste(rep("a", 10), collapse = ""),
      paste(rep("i", intron_len - 10), collapse = ""),
      paste(rep("d", FLANK_BP), collapse = "")
    )
  } else {
    full_type <- paste0(
      paste(rep("a", FLANK_BP), collapse = ""),
      paste(rep("a", intron_len), collapse = ""),
      paste(rep("d", FLANK_BP), collapse = "")
    )
  }
  list(seq = full_seq, type = full_type)
}

txids <- as.character(data$tx_rep)
for (j in seq_along(txids)) {
  txid <- txids[j]
  is <- introns[[txid]]
  if (length(is) < INTRON_LEAST) {
    next
  }
  strand <- as.character(strand(is)[1])
  if (strand == "+") {
    is <- sort(is)
  } else {
    is <- sort(is, decreasing = TRUE)
  }

  units <- lapply(is, build_unit)
  introns_seq <- sapply(units, function(u) u$seq)
  introns_type <- sapply(units, function(u) u$type)

  iranks <- seq_along(introns_seq)
  repeat {
    iranks_shuffled <- sample(iranks)
    if (sum(iranks_shuffled != iranks) > 0) {
      break
    }
  }

  data[txid, "whole_seq_real"] <- paste(introns_seq, collapse = sep)
  data[txid, "whole_type_real"] <- paste(introns_type, collapse = sep_type)
  data[txid, "whole_seq_false"] <- paste(introns_seq[iranks_shuffled],
                                         collapse = sep)
  data[txid, "whole_type_false"] <- paste(introns_type[iranks_shuffled],
                                          collapse = sep_type)
  data[txid, "iranks_shuffled_label"] <- paste(iranks_shuffled, collapse = "%")
  data[txid, "intron_counts"] <- length(is)
  data[txid, "seqs_len"] <- nchar(data[txid, "whole_seq_real"])
}

data$spe <- SPECIES[i]
write.csv(data, file.path(OUT_DIR, paste0(SPECIES[i], "_intron_with_flank.csv")),
          row.names = FALSE)
