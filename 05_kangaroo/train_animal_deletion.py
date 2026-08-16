"""Train the animal KangarooGene deletion models and score both kingdoms.

The animal transcripts are read directly from the deletion CSV and one-hot
encoded on the fly with the shared 300 kb sequence length. One model is
trained per orthogroup fold and the held-out fold is scored with that model.

Outputs
-------
./animal_random_delete1_final_group{fold}
    Saved animal deletion model for every fold.
./animal_deleted_1intron_newmodel.csv
    Animal held-out predictions with predpos/predneg columns.
./animaltoplant_deleted_1model.csv
    Plant transcripts scored by the animal models (pos_score/neg_score).

Usage:
    python train_animal_deletion.py
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
ANIMAL_DELETED_CSV = os.path.join(DATA_DIR, "animal_deleted_all.csv")
PLANT_DELETED_CSV = os.path.join(DATA_DIR, "plant_deleted_1intron.csv")
ANIMAL_PRED_CSV = os.path.join(
    "/data1/lty/intron_pred/new_model", "animal_deleted_1intron_newmodel.csv"
)
ANIMAL_TO_PLANT_CSV = os.path.join(
    "/data1/lty/intron_pred/new_model", "animaltoplant_deleted_1model.csv"
)
MODEL_PREFIX = "animal_random_delete1_final_group"

MRNA_LIMIT = 300000
TRAIN_BATCH = 32
TEST_BATCH = 32
PATIENCE = 4
EPOCHS = 100
AGCT_ONLY = re.compile(r"^[AGCT]*$")

ANIMAL_SPECIES = [
    "Bta", "Chi", "Cin", "Dre", "Fca", "Gga", "Homo",
    "Mmu", "Nna", "Ptro", "Ssa", "Xtr",
]


def load_animal_data():
    """Load, filter and annotate the animal one-intron deletion dataset."""
    df = pd.read_csv(ANIMAL_DELETED_CSV)
    df = df.dropna(subset=["whole_seq_real"])
    df["real_len"] = df["whole_seq_real"].str.len()
    df["false_len"] = df["whole_seq_false"].str.len()
    df = df[df["real_len"] < MRNA_LIMIT]
    df = df[df["whole_seq_real"].str.match(AGCT_ONLY)]

    # Animal gene ids in the ortho table carry a transcript suffix that is
    # removed before joining on the unsuffixed gene id of the deletion table.
    ortho = pd.read_csv(ORTHO_CSV)
    mask = ortho["spe"].isin(ANIMAL_SPECIES)
    ortho.loc[mask, "gene"] = ortho.loc[mask, "gene"].apply(
        lambda x: str(x).split(".")[0] if "." in str(x) else str(x)
    )
    ortho = ortho[["gene", "group_type"]]
    df = df.rename(columns={"gene_id": "gene"})
    return pd.merge(df, ortho, on="gene").reset_index(drop=True)


def load_plant_data():
    """Load, filter and annotate the plant deletion table."""
    df = pd.read_csv(PLANT_DELETED_CSV)
    df["real_len"] = df["whole_seq_real"].str.len()
    df["false_len"] = df["whole_seq_false"].str.len()
    df = df[df["real_len"] < MRNA_LIMIT]
    df = df[df["whole_seq_real"].str.match(AGCT_ONLY)]
    ortho = pd.read_csv(ORTHO_CSV, usecols=["gene", "group_type"])
    df = df.rename(columns={"gene_id": "gene"})
    return pd.merge(df, ortho, on="gene").reset_index(drop=True)


def train_fold(df, test_fold):
    """Train one animal deletion fold and return the saved model path."""
    train_df = df[df["group_type"] != test_fold]
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


def score_animal(df):
    """Score every animal fold with its held-out model."""
    pred_pos = []
    pred_neg = []
    for fold in range(1, 6):
        model = load_model(os.path.join(MODEL_DIR, f"{MODEL_PREFIX}{fold}"))
        fold_df = df[df["group_type"] == fold].reset_index(drop=True)
        pred_pos.extend(predict_scores(model, fold_df, "whole_seq_real",
                                       TEST_BATCH))
        pred_neg.extend(predict_scores(model, fold_df, "whole_seq_false",
                                       TEST_BATCH))

    out = df.copy()
    out["predpos"] = pred_pos
    out["predneg"] = pred_neg
    out = out.drop(
        columns=["whole_seq_real", "whole_seq_false",
                 "whole_type_real", "whole_type_false"]
    )
    out.to_csv(ANIMAL_PRED_CSV, index=False)
    print(f"saved: {ANIMAL_PRED_CSV} ({len(out)} rows)")


def score_plant(df):
    """Score plant transcripts with the animal deletion models."""
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
    out.to_csv(ANIMAL_TO_PLANT_CSV, index=False)
    print(f"saved: {ANIMAL_TO_PLANT_CSV} ({len(out)} rows)")


def main():
    animal_df = load_animal_data()
    animal_df = animal_df.sample(frac=1).reset_index(drop=True)
    for fold in range(1, 6):
        train_fold(animal_df, fold)
    score_animal(animal_df)
    score_plant(load_plant_data())


if __name__ == "__main__":
    main()
