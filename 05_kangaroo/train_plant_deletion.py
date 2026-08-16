"""Train the plant KangarooGene deletion models and score both kingdoms.

The plant transcripts are read directly from the deletion CSV and one-hot
encoded on the fly with the shared 10 kb sequence length. For every
orthogroup fold the training rows are shuffled, one model is trained and the
held-out fold is scored with that model.

Outputs
-------
./plant_random_delete1_group{fold}
    Saved plant deletion model for every fold.
./plant_deleted_1model_predout.csv
    Plant held-out predictions (pos_score/neg_score).
./planttoanimal_deleted_1model_predout.csv
    Animal transcripts scored by the plant models (pos_score/neg_score).

Usage:
    python train_plant_deletion.py
"""

import math
import os
import re
import sys

import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "00_common"))
from model_builders import build_single_input_model
from data_generators import si_generator
from sequence_utils import encode_sequences


DATA_DIR = "/data1/lty/intron_pred/data"
MODEL_DIR = "/data1/lty/intron_pred/new_model/final_model"
ORTHO_CSV = os.path.join(DATA_DIR, "orth40spe_out.csv")
PLANT_DELETED_CSV = os.path.join(DATA_DIR, "plant_deleted_1intron.csv")
ANIMAL_DELETED_CSV = os.path.join(DATA_DIR, "animal_deleted_all.csv")
PLANT_PRED_CSV = os.path.join(
    "/data1/lty/intron_pred/new_model", "plant_deleted_1model_predout.csv"
)
PLANT_TO_ANIMAL_CSV = os.path.join(
    "/data1/lty/intron_pred/new_model",
    "planttoanimal_deleted_1model_predout.csv",
)
MODEL_PREFIX = "plant_random_delete1_group"

MRNA_LIMIT = 10000
TRAIN_BATCH = 128
TEST_BATCH = 128
PATIENCE = 6
EPOCHS = 100
AGCT_ONLY = re.compile(r"^[AGCT]*$")


def load_plant_data():
    """Load, filter and annotate the plant one-intron deletion dataset."""
    df = pd.read_csv(PLANT_DELETED_CSV)
    df["real_len"] = df["whole_seq_real"].str.len()
    df["false_len"] = df["whole_seq_false"].str.len()
    df = df[df["real_len"] < MRNA_LIMIT]
    df = df[df["whole_seq_real"].str.match(AGCT_ONLY)]
    ortho = pd.read_csv(ORTHO_CSV, usecols=["gene", "group_type"])
    df = df.rename(columns={"gene_id": "gene"})
    return pd.merge(df, ortho, on="gene").reset_index(drop=True)


def load_animal_data():
    """Load, filter and annotate the animal deletion table."""
    df = pd.read_csv(ANIMAL_DELETED_CSV)
    df = df.dropna(subset=["whole_seq_real"])
    df["real_len"] = df["whole_seq_real"].str.len()
    df["false_len"] = df["whole_seq_false"].str.len()
    df = df[df["real_len"] < MRNA_LIMIT]
    df = df[df["whole_seq_real"].str.match(AGCT_ONLY)]
    ortho = pd.read_csv(ORTHO_CSV, usecols=["gene", "group_type"])
    df = df.rename(columns={"gene_id": "gene"})
    return pd.merge(df, ortho, on="gene").reset_index(drop=True)


def train_fold(df, test_fold):
    """Train one plant deletion fold and return the saved model path."""
    train_df = df[df["group_type"] != test_fold].sample(frac=1)
    test_df = df[df["group_type"] == test_fold]

    model = build_single_input_model(MRNA_LIMIT)
    model.compile(
        loss="binary_crossentropy",
        optimizer=Adam(learning_rate=0.001),
        metrics=["accuracy"],
    )
    callbacks = [EarlyStopping(
        monitor="val_loss", patience=PATIENCE, verbose=1,
        restore_best_weights=True,
    )]
    model.fit(
        si_generator(train_df, TRAIN_BATCH, MRNA_LIMIT),
        steps_per_epoch=int(math.ceil(len(train_df) * 2 / TRAIN_BATCH)),
        epochs=EPOCHS,
        callbacks=callbacks,
        validation_data=si_generator(test_df, TEST_BATCH, MRNA_LIMIT),
        validation_steps=int(math.ceil(len(test_df) * 2 / TEST_BATCH)),
        max_queue_size=10,
        workers=1,
        use_multiprocessing=False,
    )
    path = os.path.join(MODEL_DIR, f"{MODEL_PREFIX}{test_fold}")
    model.save(path)
    return path


def predict_scores(model, df, seq_col, batch_size):
    """Return the positive-class probability for one sequence column."""
    scores = []
    n_batches = int(math.ceil(len(df) / batch_size))
    for i in range(n_batches):
        chunk = df.iloc[batch_size * i: batch_size * (i + 1)]
        x = encode_sequences(chunk[seq_col], MRNA_LIMIT)
        scores.extend(np.asarray(model.predict(x))[:, -1])
    return np.asarray(scores)


def score_all(df, out_csv):
    """Score every fold with the matching held-out model and save the table."""
    out = pd.DataFrame()
    for fold in range(1, 6):
        model = load_model(os.path.join(MODEL_DIR, f"{MODEL_PREFIX}{fold}"))
        fold_df = df[df["group_type"] == fold].reset_index(drop=True)
        fold_df["pos_score"] = predict_scores(
            model, fold_df, "whole_seq_real", TEST_BATCH
        )
        fold_df["neg_score"] = predict_scores(
            model, fold_df, "whole_seq_false", TEST_BATCH
        )
        fold_df = fold_df.drop(
            columns=["whole_seq_real", "whole_seq_false",
                     "whole_type_real", "whole_type_false"]
        )
        out = pd.concat([out, fold_df], ignore_index=True)
    out.to_csv(out_csv, index=False)
    print(f"saved: {out_csv} ({len(out)} rows)")


def main():
    plant_df = load_plant_data()
    for fold in range(1, 6):
        train_fold(plant_df, fold)

    score_all(plant_df, PLANT_PRED_CSV)
    score_all(load_animal_data(), PLANT_TO_ANIMAL_CSV)


if __name__ == "__main__":
    main()
