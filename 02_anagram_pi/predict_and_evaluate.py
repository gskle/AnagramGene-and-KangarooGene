"""Score all five folds with the paired-input AnagramGene models.

For every transcript the model is asked twice: once with (native, shuffled)
and once with (shuffled, native). The first score is left_out and the second
is right_out; both are probabilities from the last softmax column.

Usage:
    python predict_and_evaluate.py <data_csv> <model_prefix> <mrna_limit>
    python predict_and_evaluate.py \
        /data1/lty/intron_pred/data/plant18_intronshuffle.csv \
        /data1/lty/intron_pred/new_model/final_model/plant18spe_1wbp_group 10000

The output CSV is named <data_file>_<model_kingdom>_2input_pred_out.csv,
for example plant18_intronshuffle_plant_2input_pred_out.csv when the plant
model is used.
"""

import math
import os
import sys

import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "00_common"))
from sequence_utils import encode_sequences, filter_agct

THRESHOLD = 0.5


def predict_pairwise(model, df, batch_size, maxlen):
    """Return (left_out, right_out) probabilities for each row."""
    left = []
    right = []
    n_batches = int(math.ceil(len(df) / batch_size))
    for i in range(n_batches):
        chunk = df.iloc[batch_size * i: batch_size * (i + 1)]
        pos = encode_sequences(chunk["whole_seq_real"], maxlen)
        neg = encode_sequences(chunk["whole_seq_false"], maxlen)
        left_out = np.asarray(model.predict([pos, neg]))[:, -1]
        right_out = np.asarray(model.predict([neg, pos]))[:, 0]
        left.extend(left_out)
        right.extend(right_out)
    return np.asarray(left), np.asarray(right)


def infer_model_kingdom(model_prefix):
    """Return the kingdom encoded in a model path prefix."""
    prefix = model_prefix.lower()
    for kingdom in ("plant", "animal", "fungi"):
        if kingdom in prefix:
            return kingdom
    raise ValueError(
        "cannot infer model kingdom from model_prefix; "
        "expected a prefix containing plant/animal/fungi"
    )


def main():
    data_csv, model_prefix, mrna_limit = sys.argv[1], sys.argv[2], int(sys.argv[3])
    batch_size = int(sys.argv[4]) if len(sys.argv) > 4 else 128

    intron_data = pd.read_csv(data_csv)
    intron_data = intron_data[intron_data["intron_counts"] >= 3]
    intron_data = intron_data[intron_data["seqs_len"] < mrna_limit]
    intron_data = filter_agct(intron_data)

    pred_df = pd.DataFrame()
    for fold in range(1, 6):
        model = load_model(f"{model_prefix}{fold}")
        fold_df = intron_data[intron_data["group_type"] == fold].reset_index(drop=True)
        left_out, right_out = predict_pairwise(
            model, fold_df, batch_size, mrna_limit
        )
        fold_df["left_out"] = left_out
        fold_df["right_out"] = right_out
        keep = ["gene", "tx_rep", "spe", "seqs_len", "group_type",
                "left_out", "right_out", "iranks_shuffled_label", "intron_counts"]
        pred_df = pd.concat([pred_df, fold_df[keep]], ignore_index=True)

    data_base = os.path.splitext(os.path.basename(data_csv))[0]
    model_kingdom = infer_model_kingdom(model_prefix)
    out_csv = f"{data_base}_{model_kingdom}_2input_pred_out.csv"
    pred_df.to_csv(out_csv, index=False)

    fold_acc = []
    for fold in range(1, 6):
        fold_df = pred_df[pred_df["group_type"] == fold]
        left_acc = np.mean(fold_df["left_out"] >= THRESHOLD)
        right_acc = np.mean(fold_df["right_out"] >= THRESHOLD)
        both_acc = (left_acc + right_acc) / 2
        fold_acc.append(both_acc)
        print(f"fold {fold}: left {left_acc:.4f} right {right_acc:.4f} "
              f"mean {both_acc:.4f}")
    print(f"overall mean accuracy: {np.mean(fold_acc):.4f}")


if __name__ == "__main__":
    main()
