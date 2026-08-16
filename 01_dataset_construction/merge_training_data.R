# Merge kingdom sequence tables, transcript references and orthogroup
# information into one training table.
#
# Outputs:
#   40spe_intronshuffle.csv and the kingdom files used by training:
#   plant18_intronshuffle.csv, animal14_intronshuffle.csv, fungi6_intronshuffle.csv

library(data.table)

ORTHO_CSV <- "/home/hai/Desktop/orth40spe_out.csv"
PLANT_CSV <- "/home/hai/Desktop/plant_data/plant_intron_random.csv"
PLANT_REP_CSV <- "/home/hai/Desktop/plant_data/plant_gene_tx.csv"
ANIMAL_CSV <- "/home/hai/lty_tmp/animal_genome/all_animal_shuffleintron.csv"
ANIMAL_REP_CSV <- "/home/hai/lty_tmp/animal_genome/animal_gene_tx.csv"
FUNGI_CSV <- "/home/hai/Desktop/fungi_allintron_7spe.csv"
FUNGI_REP_CSV <- "/home/hai/Desktop/fungi_data/fungi_gene_tx.csv"
FUNGI_INFO_CSV <- "/home/hai/Desktop/info_tx_len.csv"

OUT_40SPE <- "/home/hai/lty_tmp/intron_pred/40spe_intronshuffle.csv"
OUT_PLANT <- "/data1/lty/intron_pred/data/plant18_intronshuffle.csv"
OUT_ANIMAL <- "/data1/lty/intron_pred/data/animal14_intronshuffle.csv"
OUT_FUNGI <- "/data1/lty/intron_pred/data/fungi6_intronshuffle.csv"

gene_data <- as.data.frame(fread(ORTHO_CSV))
gene_data <- gene_data[, c("group", "gene", "ortho_type", "group_type")]

keep_cols <- c("tx_rep", "whole_seq_real", "whole_seq_false",
               "iranks_shuffled_label", "whole_type_real",
               "whole_type_false", "spe", "intron_counts", "seqs_len", "gene")

# Plant ---------------------------------------------------------------
plant_data <- as.data.frame(fread(PLANT_CSV))
plant_rep <- as.data.frame(fread(PLANT_REP_CSV))[, c("tx_id", "gene_id")]
colnames(plant_rep) <- c("tx_rep", "gene")
plant_data <- merge(plant_data, plant_rep, by = "tx_rep", all.x = TRUE)
plant_data <- plant_data[, keep_cols]
plant_data$kingdom <- "Plant"

# Animal --------------------------------------------------------------
animal_data <- as.data.frame(fread(ANIMAL_CSV))
animal_rep <- as.data.frame(fread(ANIMAL_REP_CSV))[, c("tx_id", "gene_id")]
colnames(animal_rep) <- c("tx_rep", "gene")
animal_data <- merge(animal_data, animal_rep, by = "tx_rep", all.x = TRUE)
animal_data$spe[animal_data$spe == "Ptr"] <- "Ptro"
animal_data$spe[animal_data$spe == "homo"] <- "Homo"
animal_data <- animal_data[, keep_cols]
animal_data$kingdom <- "Animal"

# Fungi ---------------------------------------------------------------
fungi_data <- as.data.frame(fread(FUNGI_CSV))
fungi_rep <- as.data.frame(fread(FUNGI_REP_CSV))[, c("tx_id", "gene_id")]
colnames(fungi_rep) <- c("tx_rep", "gene")
fungi_data <- merge(fungi_data, fungi_rep, by = "tx_rep", all.x = TRUE)

# Four fungi species were annotated with source-specific transcript IDs; their
# transcript IDs point into the tx/gene reference file.
fungi_change <- c("Ani", "Mor", "Ncr", "Ztr")
fungi_changedata <- c("Asp", "Mag", "Neu", "Zym")
fungi_name_map <- c(Asp = "Ani", Mag = "Mor", Puc = "Pgr", Sac = "Sce",
                    Zym = "Ztr", Neu = "Ncr", Rhi = "Rmi", Spo = "Spo")
fungi_info <- as.data.frame(fread(FUNGI_INFO_CSV))
fungi_info <- fungi_info[fungi_info$species %in% fungi_change,
                         c("tx", "gene")]
colnames(fungi_info) <- c("tx_rep", "gene")
rownames(fungi_info) <- fungi_info$tx_rep

is_change <- fungi_data$spe %in% fungi_changedata
fungi_data$tx_rep[is_change] <- sapply(
  strsplit(fungi_data$tx_rep[is_change], ":", fixed = TRUE),
  function(x) x[length(x)]
)
fungi_data$gene[is_change] <- fungi_info[fungi_data$tx_rep[is_change], "gene"]
fungi_data$spe <- unname(fungi_name_map[as.character(fungi_data$spe)])
fungi_data <- fungi_data[, keep_cols]
fungi_data$kingdom <- "Fungi"

# Final merge with orthogroups ----------------------------------------
all_data <- rbind(plant_data, animal_data, fungi_data)

change_spe <- c("Bta", "Chi", "Cin", "Dre", "Fca", "Gga", "Homo", "Mmu",
                "Nna", "Ptro", "Ssa", "Xtr")
strip_version <- function(x) {
  unname(sapply(strsplit(x, ".", fixed = TRUE), "[", 1))
}
all_data$gene[all_data$spe %in% change_spe] <-
  strip_version(all_data$gene[all_data$spe %in% change_spe])

all_data_out <- merge(all_data, gene_data, by = "gene", all.x = TRUE)
write.csv(all_data_out, OUT_40SPE, row.names = FALSE)

# Kingdom training files ----------------------------------------------
spe_map <- c(
  homo = "Homo"
)
all_data_out$spe <- unname(ifelse(all_data_out$spe %in% names(spe_map),
                                  spe_map[all_data_out$spe],
                                  all_data_out$spe))

kingdom_ref <- data.frame(
  spe = c("Ath", "Bdi", "Bvu", "Cre", "Csa", "Gma", "Mtr", "Osa", "Ppa",
          "Ptr", "Sbi", "Sit", "Sly", "Stu", "Svi", "Vvi", "Zma", "Tur",
          "Bta", "Cel", "Chi", "Cin", "Dre", "Dme", "Fca", "Gga", "Homo",
          "Mmu", "Nna", "Ptro", "Ssa", "Xtr", "Ani", "Mor", "Pgr", "Sce",
          "Ztr", "Ncr", "Rmi", "Spo"),
  kingdom = c(rep("plant", 18), rep("animal", 14), rep("fungi", 8)),
  stringsAsFactors = FALSE
)
all_data_out <- merge(all_data_out, kingdom_ref, by = "spe", all.x = TRUE)

write.csv(all_data_out[all_data_out$kingdom == "plant", ], OUT_PLANT,
          row.names = FALSE)
write.csv(all_data_out[all_data_out$kingdom == "animal", ], OUT_ANIMAL,
          row.names = FALSE)
write.csv(all_data_out[all_data_out$kingdom == "fungi", ], OUT_FUNGI,
          row.names = FALSE)
