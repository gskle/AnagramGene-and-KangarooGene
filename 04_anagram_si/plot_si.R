# Plot single-input model accuracy results.
#
# Two figures are produced: (1) plant and animal SI model accuracy on plant
# and animal shuffled data, and (2) the same models split by dicot and
# monocot plant species.
#
# Usage:
#   Rscript plot_si.R <input_dir> <output_dir>

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
})

args <- commandArgs(trailingOnly = TRUE)
input_dir <- if (length(args) >= 1) args[1] else "."
output_dir <- if (length(args) >= 2) args[2] else "."

read_pred <- function(file) {
  path <- file.path(input_dir, file)
  if (!file.exists(path)) stop("Missing prediction file: ", path)
  read.csv(path, stringsAsFactors = FALSE)
}

plant_acc <- read_pred("plant_1input_pred_out.csv")
animal_acc <- read_pred("animal_1input_pred_out.csv")
animaltoplant_acc <- read_pred("animaltoplant_1input_pred_out.csv")
planttoanimal_acc <- read_pred("planttoanimal_1input_pred_out.csv")

# A transcript is correct when the native-order probability is >= 0.5 or the
# shuffled-order probability is < 0.5.
acc_fun <- function(df) {
  df %>%
    group_by(group_type) %>%
    summarise(
      acc = (sum(pred_out > 0.5) + sum(pred_negout < 0.5)) / (n() * 2),
      .groups = "drop"
    )
}

build_kingdom_df <- function() {
  pp <- acc_fun(plant_acc); pp$model <- "Plant model"; pp$testkingdom <- "Plant data"
  aa <- acc_fun(animal_acc); aa$model <- "Animal model"; aa$testkingdom <- "Animal data"
  ap <- acc_fun(animaltoplant_acc); ap$model <- "Animal model"; ap$testkingdom <- "Plant data"
  pa <- acc_fun(planttoanimal_acc); pa$model <- "Plant model"; pa$testkingdom <- "Animal data"
  rbind(pp, aa, ap, pa)
}

all_df <- build_kingdom_df()
error_df <- all_df %>%
  group_by(model, testkingdom) %>%
  summarise(
    mean_acc = mean(acc),
    se = sd(acc) / sqrt(n()),
    .groups = "drop"
  )

kingdom_levels <- c("Plant data", "Animal data")
error_df$testkingdom <- factor(error_df$testkingdom, levels = kingdom_levels)
all_df$testkingdom <- factor(all_df$testkingdom, levels = kingdom_levels)

p_kingdom <- ggplot(error_df, aes(y = mean_acc, x = model, fill = model)) +
  geom_col(width = 0.7) +
  geom_errorbar(aes(ymin = mean_acc - se, ymax = mean_acc + se), width = 0.2) +
  geom_jitter(data = all_df, aes(y = acc, x = model),
              width = 0.15, height = 0, size = 1, color = "black", alpha = 0.8) +
  scale_y_continuous(limits = c(0, 1), expand = c(0, 0)) +
  scale_fill_manual(values = c(
    "Animal model" = "cadetblue3",
    "Plant model" = "chartreuse3"
  )) +
  facet_wrap(~testkingdom, scales = "free_x") +
  labs(x = "Training Model", y = "Accuracy") +
  theme_bw() +
  theme(
    panel.border = element_rect(fill = NA, color = "black", size = 1.8),
    panel.background = element_blank(),
    axis.text = element_text(size = 12),
    axis.title = element_text(size = 14),
    text = element_text(size = 15, color = "black")
  )

ggsave(file.path(output_dir, "twokingdom_accuracy_plot.svg"),
       p_kingdom, width = 8, height = 4)
print(p_kingdom)

# Dicot vs monocot split on plant data, for the plant model and the animal
# model transferred to plant data.
dicot_species <- c("Bvu", "Sly", "Stu", "Vvi", "Ath", "Ptr", "Csa", "Gma", "Mtr")
mono_species <- c("Osa", "Bdi", "Tur", "Svi", "Sit", "Sbi", "Zma")

build_species_df <- function() {
  amon <- acc_fun(animaltoplant_acc[animaltoplant_acc$spe %in% mono_species, ])
  adict <- acc_fun(animaltoplant_acc[animaltoplant_acc$spe %in% dicot_species, ])
  pmon <- acc_fun(plant_acc[plant_acc$spe %in% mono_species, ])
  pdict <- acc_fun(plant_acc[plant_acc$spe %in% dicot_species, ])
  amon$model <- "Animal model"; amon$testkingdom <- "Monocot data"
  adict$model <- "Animal model"; adict$testkingdom <- "Dicot data"
  pmon$model <- "Plant model"; pmon$testkingdom <- "Monocot data"
  pdict$model <- "Plant model"; pdict$testkingdom <- "Dicot data"
  rbind(pmon, adict, amon, pdict)
}

species_df <- build_species_df()
species_error <- species_df %>%
  group_by(model, testkingdom) %>%
  summarise(
    mean_acc = mean(acc),
    se = sd(acc) / sqrt(n()),
    .groups = "drop"
  )

species_levels <- c("Dicot data", "Monocot data")
species_error$testkingdom <- factor(species_error$testkingdom, levels = species_levels)
species_df$testkingdom <- factor(species_df$testkingdom, levels = species_levels)

p_species <- ggplot(species_error, aes(y = mean_acc, x = testkingdom, fill = testkingdom)) +
  geom_col(width = 0.7) +
  geom_errorbar(aes(ymin = mean_acc - se, ymax = mean_acc + se), width = 0.2) +
  geom_jitter(data = species_df, aes(y = acc, x = testkingdom),
              width = 0.15, height = 0, size = 1, color = "black", alpha = 0.8) +
  scale_y_continuous(limits = c(0, 1), expand = c(0, 0)) +
  facet_wrap(~model, scales = "free_x") +
  labs(x = "Training Model", y = "Accuracy") +
  theme_bw() +
  theme(
    panel.border = element_rect(fill = NA, color = "black", size = 1.8),
    panel.background = element_blank(),
    axis.text = element_text(size = 12),
    axis.title = element_text(size = 14),
    text = element_text(size = 15, color = "black")
  )

ggsave(file.path(output_dir, "dicot_monocot_accuracy_plot.svg"),
       p_species, width = 8, height = 4)
print(p_species)
