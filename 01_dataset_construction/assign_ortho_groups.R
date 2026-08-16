# Assign orthogroup membership and family-aware cross-validation groups.
#
# OrthoFinder was run beforehand with:
#   for f in *fa ; do python /data1/lty/OrthoFinder/tools/primary_transcript.py $f ; done
#   orthofinder -f primary_transcripts/
#
# Inputs:
#   ORTHOGROUPS_TSV  : OrthoFinder "Orthogroups.tsv" (gene_id per species per OG)
#   UNASSIGNED_TSV   : OrthoFinder "Orthogroups_UnassignedGenes.tsv"
# Output:
#   OUT_CSV          : one row per gene with group (orthogroup), spe, ortho_type
#                      and group_type (1-5 cross-validation partition)
#


library(data.table)

ORTHOGROUPS_TSV <- "/data1/lty/orthofinder_test/primary_transcripts/OrthoFinder/Results_Sep28/Orthogroups/Orthogroups.tsv"
UNASSIGNED_TSV <- "/data1/lty/orthofinder_test/primary_transcripts/OrthoFinder/Results_Sep28/Orthogroups/Orthogroups_UnassignedGenes.tsv"
OUT_CSV <- "/data1/lty/orthofinder_test/orth40spe_out.csv"

spe_use <- c(
  "Ath", "Ani", "Bdi", "Bvu", "Bta", "Cel", "Chi", "Cre", "Cin", "Csa",
  "Dre", "Dme", "Fca", "Gga", "Gma", "Homo", "Mor", "Mtr", "Mmu", "Nna",
  "Ncr", "Osa", "Ptro", "Ppa", "Ptr", "Pgr", "Rmi", "Sce", "Ssa", "Sbi",
  "Spo", "Sit", "Sly", "Stu", "Svi", "Tur", "Vvi", "Xtr", "Zma", "Ztr"
)

plant_spe <- c("Ath", "Bdi", "Bvu", "Cre", "Csa", "Gma", "Mtr", "Osa",
               "Ppa", "Ptr", "Sbi", "Sit", "Sly", "Stu", "Svi", "Vvi",
               "Zma", "Tur")
animal_spe <- c("Bta", "Cel", "Chi", "Cin", "Dre", "Dme", "Fca", "Gga",
                "Homo", "Mmu", "Nna", "Ptro", "Ssa", "Xtr")
fungi_spe <- c("Ani", "Mor", "Ncr", "Pgr", "Rmi", "Sce", "Spo", "Ztr")

# Convert a species-column table into long form: one row per gene.
melt_ortho_table <- function(tab) {
  tab <- as.data.frame(tab, stringsAsFactors = FALSE)
  og <- as.character(tab[[1]])
  out <- lapply(seq_along(spe_use), function(i) {
    spe <- spe_use[i]
    cell <- as.character(tab[[spe_use[i]]])
    genes <- strsplit(cell, ", ", fixed = TRUE)
    keep <- lengths(genes) > 0 & nchar(cell) > 0
    data.frame(
      group = rep(og[keep], lengths(genes[keep])),
      gene = unlist(genes[keep]),
      spe = spe,
      stringsAsFactors = FALSE
    )
  })
  do.call(rbind, out)
}

ortho_df <- fread(ORTHOGROUPS_TSV, sep = "\t", header = TRUE)
unassign_df <- fread(UNASSIGNED_TSV, sep = "\t", header = TRUE)

all_out <- melt_ortho_table(ortho_df)
all_out$ortho_type <- "assign"

all_unassigned <- melt_ortho_table(unassign_df)
all_unassigned$ortho_type <- "unassign"

gene_data <- rbind(all_out, all_unassigned)
write.csv(gene_data, OUT_CSV, row.names = FALSE)

# ---- Family-aware 5-group partition -------------------------------
# Sort orthogroups by gene count, then assign each family to the group
# with the currently lowest gene count.
gene_data <- as.data.frame(fread(OUT_CSV))

# Genes per orthogroup, ordered from largest family to smallest.
family_genes <- split(as.character(gene_data$gene), gene_data$group)
family_genes <- family_genes[order(-lengths(family_genes))]

grouped_families <- vector("list", 5)
group_counts <- rep(0, 5)

for (genes in family_genes) {
  min_group <- which.min(group_counts)
  grouped_families[[min_group]] <- c(grouped_families[[min_group]], genes)
  group_counts[min_group] <- group_counts[min_group] + length(genes)
}

rownames(gene_data) <- gene_data$gene
gene_data$group_type <- 6
for (g in 1:5) {
  gene_data[grouped_families[[g]], "group_type"] <- g
}

write.csv(gene_data, OUT_CSV, row.names = FALSE)
