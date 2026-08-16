"""Score shuffled and one-intron-deleted transcripts with SI models.

Every SI model is applied to native and shuffled sequences from the same
kingdom and across kingdoms, producing the four shuffle prediction files.
The deleted-sequence files are then scored with the same models, giving the
"shuffle for deletion" prediction files used by the KangarooGene comparison.

Usage:
    python predict_on_deleted_sequences.py
"""

import math
import os
import re
import sys

import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "00_common"))
from sequence_utils import encode_sequences


DATA_DIR = "/data1/lty/intron_pred/data"
MODEL_DIR = "/data1/lty/intron_pred/new_model/final_model"
ORTHO_CSV = os.path.join(DATA_DIR, "orth40spe_out.csv")
PLANT_SI = os.path.join(MODEL_DIR, "plant18spe_1input_1wbp_group")
ANIMAL_SI = os.path.join(MODEL_DIR, "animal14spe_1input_30wbp_group")
ANIMAL_SUFFIX_SPECIES = [
    "Bta", "Chi", "Cin", "Dre", "Fca", "Gga", "Homo",
    "Mmu", "Nna", "Ptro", "Ssa", "Xtr",
]

SHUFFLE_OUT_COLS = [
    "gene", "tx_rep", "spe", "seqs_len", "group_type",
    "pred_out", "pred_negout", "iranks_shuffled_label", "intron_counts",
]
DELETE_OUT_COLS = ["gene", "tx_rep", "spe", "group_type", "pred_out", "pred_negout"]


def predict_fold(model, df, maxlen, batch_size=128):
    """Return native-order and shuffled-order probabilities for each row."""
    pred_out = []
    pred_negout = []
    n_batches = int(math.ceil(len(df) / batch_size))
    for i in range(n_batches):
        chunk = df.iloc[batch_size * i: batch_size * (i + 1)]
        pos = encode_sequences(chunk["whole_seq_real"], maxlen)
        neg = encode_sequences(chunk["whole_seq_false"], maxlen)
        pred_out.extend(np.asarray(model.predict(pos))[:, -1])
        pred_negout.extend(np.asarray(model.predict(neg))[:, -1])
    return np.asarray(pred_out), np.asarray(pred_negout)


def score_shuffle(data_csv, model_prefix, maxlen, out_csv, batch_size):
    """Score shuffled datasets with the matching or cross-kingdom SI model."""
    df = pd.read_csv(data_csv)
    df = df[df["seqs_len"] < maxlen]
    df = df[df["intron_counts"] >= 3]
    df = df.reset_index(drop=True)

    pred_df = pd.DataFrame()
    for fold in range(1, 6):
        model = load_model(f"{model_prefix}{fold}")
        fold_df = df[df["group_type"] == fold].reset_index(drop=True)
        pred_out, pred_negout = predict_fold(model, fold_df, maxlen, batch_size)
        fold_df["pred_out"] = pred_out
        fold_df["pred_negout"] = pred_negout
        pred_df = pd.concat(
            [pred_df, fold_df[SHUFFLE_OUT_COLS]], ignore_index=True
        )
    pred_df.to_csv(out_csv, index=False)
    print(f"saved: {out_csv} ({len(pred_df)} rows)")


def load_deleted_data(csv_path, maxlen):
    """Load and filter a deleted-intron dataset."""
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["whole_seq_real"])
    df["real_len"] = df["whole_seq_real"].str.len()
    df = df[df["real_len"] < maxlen]
    df = df[df["whole_seq_real"].str.contains(r"^[AGCT]*$", regex=True)]
    df = df.rename(columns={"gene_id": "gene"})
    return df.reset_index(drop=True)


def load_ortho_groups(normalize_animal=False):
    """Return the orthogroup table with optional animal gene normalisation.

    The transcript suffix is stripped from the ortho table genes (not from the
    deleted-sequence table) before the merge. The
    suffix is only present for genes of the listed animal species.
    """
    ortho = pd.read_csv(ORTHO_CSV, usecols=["gene", "spe", "group_type"])
    if normalize_animal:
        mask = ortho["spe"].isin(ANIMAL_SUFFIX_SPECIES)
        ortho.loc[mask, "gene"] = ortho.loc[mask, "gene"].apply(
            lambda x: str(x).split(".")[0] if "." in str(x) else str(x)
        )
    return ortho[["gene", "group_type"]]


def score_deleted(csv_path, model_prefix, maxlen, out_csv,
                  normalize_ortho_animal, batch_size, require_introns=True):
    """Score one-intron-deleted transcripts with the SI models."""
    df = load_deleted_data(csv_path, maxlen)
    if require_introns:
        df = df[df["intron_counts"] >= 3]

    ortho = load_ortho_groups(normalize_animal=normalize_ortho_animal)
    df = pd.merge(df, ortho, on="gene")

    pred_df = pd.DataFrame()
    for fold in range(1, 6):
        model = load_model(f"{model_prefix}{fold}")
        fold_df = df[df["group_type"] == fold].reset_index(drop=True)
        pred_out, pred_negout = predict_fold(model, fold_df, maxlen, batch_size)
        fold_df["pred_out"] = pred_out
        fold_df["pred_negout"] = pred_negout
        pred_df = pd.concat(
            [pred_df, fold_df[DELETE_OUT_COLS]], ignore_index=True
        )
    pred_df.to_csv(out_csv, index=False)
    print(f"saved: {out_csv} ({len(pred_df)} rows)")


def main():
    shuffle_tasks = [
        (os.path.join(DATA_DIR, "plant18_intronshuffle.csv"),
         PLANT_SI, 10000, os.path.join(MODEL_DIR, "plant_1input_pred_out.csv")),
        (os.path.join(DATA_DIR, "plant18_intronshuffle.csv"),
         ANIMAL_SI, 300000, os.path.join(MODEL_DIR, "animaltoplant_1input_pred_out.csv")),
        (os.path.join(DATA_DIR, "animal14_intronshuffle.csv"),
         ANIMAL_SI, 300000, os.path.join(MODEL_DIR, "animal_1input_pred_out.csv")),
        (os.path.join(DATA_DIR, "animal14_intronshuffle.csv"),
         PLANT_SI, 10000, os.path.join(MODEL_DIR, "planttoanimal_1input_pred_out.csv")),
    ]
    for data_csv, model_prefix, maxlen, out_csv in shuffle_tasks:
        score_shuffle(data_csv, model_prefix, maxlen, out_csv, batch_size=128)

    score_deleted(
        os.path.join(DATA_DIR, "plant_deleted_1intron.csv"),
        PLANT_SI,
        10000,
        os.path.join(MODEL_DIR, "plant_shuffle_for_dele_pred_out.csv"),
        normalize_ortho_animal=False,
        batch_size=128,
        require_introns=True,
    )
    score_deleted(
        os.path.join(DATA_DIR, "animal_deleted_all.csv"),
        ANIMAL_SI,
        300000,
        os.path.join(MODEL_DIR, "animal_shuffle_for_dele_pred_out.csv"),
        normalize_ortho_animal=True,
        batch_size=64,
        require_introns=False,
    )


if __name__ == "__main__":
    main()
