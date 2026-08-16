# Plot representative-transcript (longest CDS) length distributions.
#
# Produces one combined density plot and three per-species ridge plots.

library(data.table)
library(ggplot2)
library(ggridges)

PLANT_REP_CSV <- "/home/hai/Desktop/plant_data/plant_gene_tx.csv"
ANIMAL_REP_CSV <- "/home/hai/lty_tmp/animal_genome/animal_gene_tx.csv"
FUNGI_REP_CSV <- "/home/hai/Desktop/fungi_data/fungi_gene_tx.csv"
OUT_DIR <- "/home/hai/lty_tmp/intron_pred/photo"

plant_rep <- as.data.frame(fread(PLANT_REP_CSV))
animal_rep <- as.data.frame(fread(ANIMAL_REP_CSV))
fungi_rep <- as.data.frame(fread(FUNGI_REP_CSV))

animal_rep$spe[animal_rep$spe == "Ptr"] <- "Ptro"
animal_rep$spe[animal_rep$spe == "homo"] <- "Homo"

plant_rep$kingdom <- "Plant"
animal_rep$kingdom <- "Animal"
fungi_rep$kingdom <- "Fungi"
all_rep <- rbind(plant_rep, animal_rep, fungi_rep)
all_rep$kingdom <- factor(all_rep$kingdom, levels = c("Fungi", "Plant", "Animal"))

# Combined kingdom density plot.
ggplot(all_rep, aes(x = log10(len), fill = kingdom)) +
  geom_density(alpha = 0.7) +
  theme_ridges(font_size = 13, grid = FALSE) +
  theme(axis.title.y = element_blank())
ggsave(file.path(OUT_DIR, "three_kingdom_tx_width.svg"))

ridge_plot <- function(dat, level_order, out_name) {
  dat$spe <- factor(dat$spe, levels = level_order)
  ggplot(dat, aes(x = log10(len), y = spe, fill = after_stat(x))) +
    geom_density_ridges_gradient(scale = 3, rel_min_height = 0.01,
                                 gradient_lwd = 1) +
    scale_x_continuous(expand = c(0.01, 0)) +
    scale_y_discrete(expand = c(0.01, 0), breaks = level_order) +
    theme_ridges(font_size = 13, grid = FALSE) +
    theme(axis.title.y = element_blank())
  ggsave(file.path(OUT_DIR, out_name))
}

plant_levels <- c("Cre", "Ppa", "Bvu", "Sly", "Stu", "Vvi", "Ath", "Ptr",
                  "Csa", "Gma", "Mtr", "Osa", "Bdi", "Tur", "Svi", "Sit",
                  "Sbi", "Zma")
animal_levels <- c("Ssa", "Dre", "Xtr", "Fca", "Bta", "Chi", "Mmu", "Homo",
                   "Ptro", "Nna", "Gga", "Cin", "Dme", "Cel")

ridge_plot(plant_rep, plant_levels, "plant_tx_width.svg")
ridge_plot(animal_rep, animal_levels, "animal_tx_width.svg")
ridge_plot(fungi_rep, levels(factor(fungi_rep$spe)), "fungi_tx_width.svg")
