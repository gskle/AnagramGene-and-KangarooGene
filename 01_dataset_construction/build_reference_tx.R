# Build representative-transcript references for all species.
#
# For every gene, the representative transcript is the transcript with the
# longest CDS. The output table links each representative transcript
# (tx_rep / tx_id) to its gene and species and is used later to add gene and
# orthogroup information to the sequence datasets.
#
# Usage:
#   Rscript build_reference_tx.R
# Outputs:
#   plant_gene_tx.csv, animal_gene_tx.csv, fungi_gene_tx.csv

library(GenomicFeatures)
library(dplyr)

OUT_DIR <- "/home/hai/Desktop"

# Plants: 18 species, all with a GTF/GFF and a genome FASTA.
plant_species <- c("Ath", "Bdi", "Bvu", "Cre", "Csa", "Gma", "Mtr", "Osa",
                   "Ppa", "Ptr", "Sbi", "Sit", "Sly", "Stu", "Svi", "Vvi",
                   "Zma", "Tur")
plant_gtf <- c(
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
plant_fa <- c(
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

# These are normalised to "Ptro" and "Homo" to match the final training tables.
animal_species <- c("Bta", "Cel", "Chi", "Cin", "Dre", "Dme", "Fca", "Gga",
                    "Mmu", "Nna", "Ptro", "Ssa", "Xtr", "Homo")
animal_gtf <- c(
  "/home/hai/lty_tmp/animal_genome/Bos_taurus.ARS-UCD1.2.110.gtf",
  "/home/hai/lty_tmp/animal_genome/Caenorhabditis_elegans.WBcel235.110.gtf",
  "/home/hai/lty_tmp/animal_genome/Capra_hircus.ARS1.110.gtf",
  "/home/hai/lty_tmp/animal_genome/Ciona_intestinalis.KH.110.gtf",
  "/home/hai/lty_tmp/animal_genome/Danio_rerio.GRCz11.110.gtf",
  "/home/hai/lty_tmp/animal_genome/Drosophila_melanogaster.BDGP6.32.110.gtf",
  "/home/hai/lty_tmp/animal_genome/Felis_catus.Felis_catus_9.0.110.gtf",
  "/home/hai/lty_tmp/animal_genome/Gallus_gallus.bGalGal1.mat.broiler.GRCg7b.110.gtf",
  "/home/hai/lty_tmp/animal_genome/Mus_musculus.GRCm39.110.gtf",
  "/home/hai/lty_tmp/animal_genome/Naja_naja.Nana_v5.110.gtf",
  "/home/hai/lty_tmp/animal_genome/Pan_troglodytes.Pan_tro_3.0.110.gtf",
  "/home/hai/lty_tmp/animal_genome/Salmo_salar.Ssal_v3.1.110.gtf",
  "/home/hai/lty_tmp/animal_genome/Xenopus_tropicalis.UCB_Xtro_10.0.110.gtf",
  "/home/hai/lty_tmp/human_genome/Homo_sapiens.GRCh38.110.gtf"
)
animal_fa <- c(
  "/home/hai/lty_tmp/animal_genome/Bos_taurus.ARS-UCD1.2.dna.toplevel.fa",
  "/home/hai/lty_tmp/animal_genome/Caenorhabditis_elegans.WBcel235.dna.toplevel.fa",
  "/home/hai/lty_tmp/animal_genome/Capra_hircus.ARS1.dna.toplevel.fa",
  "/home/hai/lty_tmp/animal_genome/Ciona_intestinalis.KH.dna.toplevel.fa",
  "/home/hai/lty_tmp/animal_genome/Danio_rerio.GRCz11.dna.toplevel.fa",
  "/home/hai/lty_tmp/animal_genome/Drosophila_melanogaster.BDGP6.32.dna.toplevel.fa",
  "/home/hai/lty_tmp/animal_genome/Felis_catus.Felis_catus_9.0.dna.toplevel.fa",
  "/home/hai/lty_tmp/animal_genome/Gallus_gallus.bGalGal1.mat.broiler.GRCg7b.dna.toplevel.fa",
  "/home/hai/lty_tmp/animal_genome/Mus_musculus.GRCm39.dna.toplevel.fa",
  "/home/hai/lty_tmp/animal_genome/Naja_naja.Nana_v5.dna.toplevel.fa",
  "/home/hai/lty_tmp/animal_genome/Pan_troglodytes.Pan_tro_3.0.dna.toplevel.fa",
  "/home/hai/lty_tmp/animal_genome/Salmo_salar.Ssal_v3.1.dna.toplevel.fa",
  "/home/hai/lty_tmp/animal_genome/Xenopus_tropicalis.UCB_Xtro_10.0.dna.toplevel.fa",
  "/home/hai/lty_tmp/human_genome/Homo_sapiens.GRCh38.cds.all.fa"
)

# Fungi: eight species, using standardized species abbreviations.
fungi_species <- c("Ani", "Mor", "Pgr", "Sce", "Ztr", "Ncr", "Rmi", "Spo")
fungi_gtf <- c(
  "/home/hai/Desktop/fungi_data/Aspergillus_nidulans.ASM1142v1.57.gtf",
  "/home/hai/Desktop/fungi_data/Magnaporthe_oryzae.MG8.57.gff3",
  "/home/hai/Desktop/fungi_data/Puccinia_graminis.ASM14992v1.57.gff3",
  "/home/hai/Desktop/fungi_data/Saccharomyces_cerevisiae.R64-1-1.57.gtf",
  "/home/hai/Desktop/fungi_data/Zymoseptoria_tritici.MG2.57.gff3",
  "/home/hai/Desktop/fungi_data/Neurospora_crassa.NC12.57.gff3",
  "/home/hai/Desktop/fungi_data/Rhizopus_microsporus_gca_000825725.Rmicro_CBS_344.29_Allpaths-LG.57.gff3",
  "/home/hai/Desktop/fungi_data/Schizosaccharomyces_pombe.ASM294v2.57.gtf"
)
fungi_fa <- c(
  "/home/hai/Desktop/fungi_data/Aspergillus_nidulans.ASM1142v1.dna.toplevel.fa",
  "/home/hai/Desktop/fungi_data/Magnaporthe_oryzae.MG8.dna.toplevel.fa",
  "/home/hai/Desktop/fungi_data/Puccinia_graminis.ASM14992v1.dna.toplevel.fa",
  "/home/hai/Desktop/fungi_data/Saccharomyces_cerevisiae.R64-1-1.dna.toplevel.fa",
  "/home/hai/Desktop/fungi_data/Zymoseptoria_tritici.MG2.dna.toplevel.fa",
  "/home/hai/Desktop/fungi_data/Neurospora_crassa.NC12.dna.toplevel.fa",
  "/home/hai/Desktop/fungi_data/Rhizopus_microsporus_gca_000825725.Rmicro_CBS_344.29_Allpaths-LG.dna.toplevel.fa",
  "/home/hai/Desktop/fungi_data/Schizosaccharomyces_pombe.ASM294v2.dna.toplevel.fa"
)

# Pick the longest-CDS transcript per gene for one species.
build_species_table <- function(spe, gtf_path, fa_path, require_multi_intron) {
  txdb <- makeTxDbFromGFF(gtf_path)
  tx_len <- transcriptLengths(txdb)
  tx_gene <- tx_len[, c("tx_name", "gene_id")]

  cds_list <- cdsBy(txdb, by = "tx", use.names = TRUE)
  cds_ranges <- unlist(range(cds_list))
  cds_tx_id <- names(cds_ranges)
  gene_for_tx <- tx_gene$gene_id[match(cds_tx_id, tx_gene$tx_name)]

  out <- data.frame(
    len = as.integer(width(cds_ranges)),
    tx_id = cds_tx_id,
    gene_id = gene_for_tx,
    spe = spe,
    stringsAsFactors = FALSE
  )
  out <- out[!is.na(out$gene_id), , drop = FALSE]

  if (require_multi_intron) {
    introns <- intronsByTranscript(txdb, use.names = TRUE)
    multi <- names(introns)[lengths(introns) > 1]
    out <- out[out$tx_id %in% multi, , drop = FALSE]
  }

  out %>%
    group_by(gene_id) %>%
    arrange(desc(len)) %>%
    slice(1) %>%
    as.data.frame()
}

build_kingdom_table <- function(species, gtf_paths, fa_paths,
                                require_multi_intron = FALSE) {
  all_df <- data.frame()
  for (i in seq_along(species)) {
    tab <- build_species_table(
      species[i], gtf_paths[i], fa_paths[i], require_multi_intron
    )
    all_df <- rbind(all_df, tab)
    message(species[i])
  }
  all_df
}

plant_table <- build_kingdom_table(plant_species, plant_gtf, plant_fa)
write.csv(plant_table, file.path(OUT_DIR, "plant_gene_tx.csv"), row.names = FALSE)

animal_table <- build_kingdom_table(animal_species, animal_gtf, animal_fa)
write.csv(animal_table, file.path(OUT_DIR, "animal_gene_tx.csv"), row.names = FALSE)

fungi_table <- build_kingdom_table(fungi_species, fungi_gtf, fungi_fa,
                                   require_multi_intron = TRUE)
write.csv(fungi_table, file.path(OUT_DIR, "fungi_gene_tx.csv"), row.names = FALSE)
