# Pan-Eukaryotic Synthetic Negative Learning Enhances Prediction of Functional Intronic Variants

This repository contains the analysis code for a synthetic negative learning framework that distinguishes authentic genes from synthetic intron-based decoys and scores the potential functional effects of intronic variants.

- **AnagramGene-PI** uses paired natural and intron-shuffled sequences with a shared residual CNN encoder.
- **AnagramGene-SI** initializes its encoder from AnagramGene-PI and jointly fine-tunes the encoder and downstream classification head on individual sequences.
- **KangarooGene** distinguishes natural sequences from sequences with one intron deleted.
- **KangarooGene-EN** combines KangarooGene scores with five exon-length features.

The study uses 40 eukaryotic species (14 animals, 18 plants and 8 fungi), evaluates cross-kingdom predictions, and integrates variant scores with SpliceAI and Pangolin for ClinVar classification.

## Main analyses

The code covers the following workflow:

1. Construct reference-transcript, shuffled-intron, intron-only, intron-flank, and intron-deletion datasets.
2. Train and evaluate paired-input **AnagramGene-PI** models.
3. Perform feature-ablation experiments to measure the contribution of exonic, intronic, splice-region, and structural features.
4. Train single-input **AnagramGene-SI** models and evaluate in-silico intron deletion effects.
5. Train **KangarooGene** deletion models and combine them with exon-length features in **KangarooGene-EN**.
6. Annotate ClinVar intronic variants, calculate allele-specific scores, and train ensemble classifiers.

## Repository structure

The analysis directories are at the repository root:

```text
AnagramGene-and-KangarooGene/
├── 00_common/                  Shared encoders, data generators, and model builders
├── 01_dataset_construction/    Dataset construction and quality-control plots
├── 02_anagram_pi/              AnagramGene paired-input training and evaluation
├── 03_feature_ablation/        Feature-ablation experiments
├── 04_anagram_si/              AnagramGene single-input transfer models
├── 05_kangaroo/                KangarooGene deletion models and ensembles
├── 06_clinvar/                 ClinVar annotation, scoring, and evaluation
└── demo/                       Example inputs, model checkpoints, and prediction demo
```

## Data and model weights

The study datasets, trained AnagramGene-PI, AnagramGene-SI and KangarooGene weights, and ClinVar ensemble prediction scores are available through [Figshare](https://figshare.com/s/bffb0639cbd8d1072503).

For the full analyses, download the required datasets and model weights from Figshare. The prediction demo includes its own example inputs and model checkpoints. Model loading requires the complete saved model, including its variables and metadata, rather than an individual weight component.

The manuscript's Supplementary Tables 1 and 2 describe the genomic sources. Human variant analyses use GRCh38 and the ClinVar release dated 24 June 2024. The external splicing predictors are available from [SpliceAI](https://github.com/Illumina/SpliceAI) and [Pangolin](https://github.com/tkzeng/Pangolin).

## Software environment

The following software versions were used for data processing and analysis. The R data-construction workflow was run on Ubuntu 18.04.5 LTS.

| Component | Version |
|---|---|
| Python | 3.6.5 |
| TensorFlow GPU | 2.6.0 |
| Keras | 2.6.0 |
| NumPy | 1.19.5 |
| pandas | 1.1.5 |
| scikit-learn | 0.23.1 |
| R | 3.6.3 |
| GenomicFeatures | 1.38.2 |
| BSgenome | 1.54.0 |
| dplyr | 1.1.4 |
| GenomicRanges | 1.38.0 |
| Biostrings | 2.54.0 |
| data.table | 1.14.8 |
| ggplot2 | 3.4.4 |

The preprocessing and comparison workflows also used the following tools:

| Tool | Version |
|---|---|
| OrthoFinder | 2.5.5 |
| SpliceAI | 1.3.1 |
| Pangolin | 1.0.2 |
| Mashtree | 1.4.6 |
| RepeatMasker | 4.1.6 |
| Progressive Cactus | 2.8.4 |
| PHAST | 1.5 |

## Usage

The scripts are intended to be run by analysis stage. For example:

```bash
cd AnagramGene-and-KangarooGene

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

## Prediction demo

A small prediction demo is provided in `demo/`. It includes real example inputs, complete plant checkpoints for AnagramGene-PI, AnagramGene-SI and KangarooGene, and expected prediction outputs. The demo runs on a CPU and does not require retraining or a separate model download.

Follow the installation instructions in `demo/README.md`. From the repository root, run the following commands after installing the demo dependencies:

```bash
cd demo
python run_demo.py --check
```

The command writes three prediction CSV files and a runtime report to `demo/results/`, and verifies the predictions against the supplied reference outputs. The reference CPU runs took approximately 12-15 seconds for all three models. See `demo/README.md` for the tested environment, input format, output definitions and instructions for running on your own data.

## License and contact

The source code in this repository is licensed under the GNU General Public License version 3.0 only (GPL-3.0-only). See the [LICENSE](LICENSE) file for the full license text.

For questions about data access or the analysis, contact the corresponding author, Hai Wang, at **wanghai@cau.edu.cn**.
