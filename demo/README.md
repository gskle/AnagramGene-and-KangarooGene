# AnagramGene and KangarooGene prediction demo

This demo runs AnagramGene-PI, AnagramGene-SI and KangarooGene on small real datasets using the supplied plant model checkpoints. It includes the input sequences, model weights, expected outputs and a command-line runner that calls the existing repository prediction functions.

## Contents

- `data/shuffle.csv` contains five native/shuffled transcript pairs from `plant18_intronshuffle.csv`. AnagramGene-PI and AnagramGene-SI use this file.
- `data/deletion.csv` contains five native/single-intron-deleted transcript pairs from `plant_deleted_1intron.csv`. KangarooGene uses this file.
- `models/pi`, `models/si` and `models/kangaroo` contain complete TensorFlow SavedModels for plant fold 1.
- `expected/` contains the three reference prediction CSV files.
- `requirements-lock.txt` lists the tested dependency versions.
- `run_demo.py` runs all three prediction examples and checks their outputs.

The inputs are Arabidopsis thaliana sequences (`spe=Ath`). Five distinct valid genes were selected in source-file order from group 1 for each dataset. Both sequences contain only uppercase A/C/G/T and are shorter than 10,000 bases. The shuffled dataset additionally requires at least three introns. These small examples demonstrate software execution; they do not estimate overall predictive performance.

## System requirements

The demo was tested on Ubuntu 18.04.5 LTS with Python 3.6.5, TensorFlow 2.6.0, Keras 2.6.0, NumPy 1.19.5, pandas 1.1.5 and protobuf 3.18.0. The reference computer had Intel Xeon Silver 4210 CPUs running at 2.20 GHz. The runner uses CPU inference, a batch size of one, two TensorFlow intra-operation threads and one inter-operation thread. No GPU or CUDA installation is required for the CPU demo. Allow at least 2 GB of available RAM; the measured peak process memory was approximately 736 MiB in the newly installed CPU environment. Other operating systems have not been tested.

## Installation

Place this `demo/` folder in the root of the AnagramGene-and-KangarooGene repository, alongside `00_common/`, `02_anagram_pi/`, `04_anagram_si/` and `05_kangaroo/`. Download or clone the complete repository, open a terminal in its root, and run:

```bash
cd demo
python3.6 -m venv .venv
source .venv/bin/activate
python -m pip install pip==21.3.1
python -m pip install -r requirements-lock.txt
```

The dependency file selects the CPU distribution of TensorFlow. `requirements-lock.txt` records the resolved package versions from the successfully tested independent environment. The data and model weights are included, so no separate data or model download is needed.

Dependency installation into a new virtual environment with pip 21.3.1 took approximately 94 seconds on the reference server using the Tsinghua PyPI mirror. This measurement includes dependency downloads and excludes installing Python and upgrading pip. Installation time depends on network speed and computer configuration. To use the same package mirror, run:

```bash
python -m pip install -r requirements-lock.txt \
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

The independent CPU environment passed `python -m pip check` and all three reference prediction comparisons.

## Run the demo

```bash
python run_demo.py --check
```

The command generates `results/pi_predictions.csv`, `results/si_predictions.csv`, `results/kangaroo_predictions.csv` and `results/runtime.json`. It compares the predictions with the supplied reference CSV files and prints `reference check passed` for each model when the comparison succeeds. A failed comparison causes a nonzero exit status. The comparison uses an absolute tolerance of 0.00001 and a relative tolerance of 0.0001 to allow for numerical differences between computing environments.

The three prediction files each contain five rows, retaining the `gene`, `tx_rep` and `spe` identifiers. Their score columns follow the original source code:

| File | Score columns | Meaning |
| --- | --- | --- |
| `pi_predictions.csv` | `left_out`, `right_out` | `left_out` is softmax column 1 for the input pair (native, shuffled). `right_out` is softmax column 0 for the reversed input pair (shuffled, native). |
| `si_predictions.csv` | `pred_out`, `pred_negout` | These are softmax column 1 for the native and shuffled sequences, respectively. |
| `kangaroo_predictions.csv` | `pos_score`, `neg_score` | These are softmax column 1 for the native and intron-deleted sequences, respectively. |

For example, the reference SI prediction for `AT1G01040.1` is `pred_out=0.989586473` and `pred_negout=0.00244282302`. The complete reference values are provided in `expected/`.

The complete demo took 12.6 seconds in the newly installed CPU environment, including Python imports and loading all three models. Model loading and prediction took 4.13 seconds for PI, 3.13 seconds for SI and 3.43 seconds for KangarooGene. Two runs in the original reference environment took 13.2 and 14.3 seconds. Both verification runs passed all reference comparisons. These timings were measured on the CPU server described above; desktop runtime depends on the processor and storage performance. The runner writes timing and peak-memory measurements for each execution to `results/runtime.json`.

## Use your own data

Prepare two CSV files with the required columns `gene`, `tx_rep`, `spe`, `whole_seq_real` and `whole_seq_false`. Each row represents one transcript pair. For the shuffled file, `whole_seq_real` contains the native sequence and `whole_seq_false` contains its intron-shuffled counterpart. For the deletion file, these columns contain the native sequence and the corresponding sequence with one intron deleted. Use the same transcript orientation and sequence-construction conventions as the supplied examples and the manuscript Methods.

Sequences must contain 1-10,000 uppercase A/C/G/T bases. The original encoder uses A/C/G/T channel order and zero-pads shorter sequences at the end to a length of 10,000. The runner rejects missing, empty, ambiguous or overlength sequences. Identifier columns should contain nonempty text. Additional columns are permitted and are not used for prediction.

```bash
python run_demo.py \
  --shuffle-csv /path/to/my_shuffle.csv \
  --deletion-csv /path/to/my_deletion.csv \
  --output-dir my_results
```

Use `--check` only for the bundled inputs. The supplied checkpoints are plant fold-1 models. Predictions on other inputs use these same checkpoints. The demo does not train models or generate shuffled/deleted sequences; the corresponding construction and training scripts are available in the full repository.

## Source code

The runner calls `predict_pairwise` from `02_anagram_pi/predict_and_evaluate.py`, `predict_fold` from `04_anagram_si/predict_on_deleted_sequences.py`, and `predict_scores` from `05_kangaroo/train_plant_deletion.py`. The KangarooGene module is imported only to access its prediction function; its training entry point is not executed. Model loading uses `compile=False` because no optimizer or training configuration is needed for inference. The original source files are unchanged.

Repository: https://github.com/gskle/AnagramGene-and-KangarooGene

Source commit: `9e829708777207f8e73126e63893113794fd736c`.

Full datasets and model resources: https://figshare.com/s/bffb0639cbd8d1072503.
