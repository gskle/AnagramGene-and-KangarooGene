# Synthetic Negative Learning for Intronic Variant Interpretation

This repository contains the analysis code for a synthetic negative learning framework that distinguishes authentic genes from synthetic intron-based decoys. The framework is designed to learn constraints in intronic sequence space and to score the potential functional effects of intronic variants.

The study introduces two compact models:

- **AnagramGene**: compares natural genes with genes whose introns have been shuffled.
- **KangarooGene**: compares natural genes with genes in which one intron has been deleted.

The models are evaluated across animals, plants, and fungi, and are applied to human ClinVar intronic variants. Their predictions can also be combined with Pangolin and SpliceAI for pathogenicity classification.

## Main analyses

The code covers the following workflow:

1. Construct reference-transcript, shuffled-intron, intron-only, intron-flank, and intron-deletion datasets.
2. Train and evaluate paired-input **AnagramGene-PI** models.
3. Perform feature-ablation experiments to measure the contribution of exonic, intronic, splice-region, and structural features.
4. Train single-input **AnagramGene-SI** models and evaluate in-silico intron deletion effects.
5. Train **KangarooGene** deletion models and combine them with exon-length features in **KangarooGene-EN**.
6. Annotate ClinVar intronic variants, calculate allele-specific scores, and train ensemble classifiers.

## Repository structure

```text
clean_code/
├── 00_common/                  Shared encoders, data generators, and model builders
├── 01_dataset_construction/    Dataset construction and quality-control plots
├── 02_anagram_pi/              AnagramGene paired-input training and evaluation
├── 03_feature_ablation/        Feature-ablation experiments
├── 04_anagram_si/              AnagramGene single-input transfer models
├── 05_kangaroo/                KangarooGene deletion models and ensembles
└── 06_clinvar/                 ClinVar annotation, scoring, and evaluation
```

See [`clean_code/README.md`](clean_code/README.md) for the detailed script map and ClinVar pipeline order.

## Data and models

The study uses 40 eukaryotic species: 14 animals, 18 plants, and 8 fungi. Genes are filtered by kingdom-specific transcript length and are required to contain at least three introns for the standard shuffled datasets. Gene-family-aware five-fold splits are used to reduce train-test leakage.

The repository contains analysis scripts rather than the complete reference genomes, annotations, intermediate tables, and trained model weights. Prepare the required input data and model files before running the workflow.

## Requirements

- Python 3
- R
- TensorFlow / Keras
- NumPy, pandas, and scikit-learn
- R packages required by the plotting and statistical scripts

Install the Python dependencies in an environment suitable for TensorFlow. The exact package versions should be recorded for reproducible model training.

## Usage

The scripts are intended to be run by analysis stage. For example:

```bash
cd clean_code

# Dataset construction
Rscript 01_dataset_construction/build_reference_tx.R
Rscript 01_dataset_construction/build_shuffle_dataset.R
Rscript 01_dataset_construction/merge_training_data.R

# AnagramGene paired-input training
python 02_anagram_pi/train_animal.py 1

# Feature ablation and downstream analyses
python 03_feature_ablation/ablation_intron_only.py
python 04_anagram_si/train_animal_si.py
python 05_kangaroo/train_animal_deletion.py
python 06_clinvar/01_find_intron_variants.py <chromosome>
```

The commands above illustrate the entry points; a complete run requires the corresponding input files and should follow the dependencies between stages. Before running, update the configurable paths near the top of each script, including `DATA_DIR`, `DATA_CSV`, `MODEL_DIR`, and output locations.

## Reproducibility notes

- Standard shuffled datasets use an `intron_least` threshold of 3.
- DNA sequences are one-hot encoded using A/C/G/T; N and unknown bases are encoded as zeros.
- Binary predictions use a threshold of 0.5 where applicable.
- Model probabilities are taken from the final softmax output column.
- ClinVar analysis requires the relevant genome, annotation, ClinVar, Pangolin, and SpliceAI inputs.

## Citation

If you use this code, please cite the associated manuscript:

> *Discriminating functional genomic variants from neutral background variation using synthetic negative learning.*

Full bibliographic information will be added after publication.

## License

No license has been specified yet. Add a `LICENSE` file before distributing this repository for reuse.
