"""Train the plant single-input (SI) transfer model.

The SI model reuses the plant paired-input encoder and adds a two-layer
convolutional head without dropout. The encoder is fine-tuned together with
the head.

Usage:
    python train_plant_si.py [fold]
"""

import math
import os
import sys

import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "00_common"))
from model_builders import build_si_model
from data_generators import si_generator


DATA_CSV = "/data1/lty/intron_pred/data/plant18_intronshuffle.csv"
MODEL_DIR = "/data1/lty/intron_pred/new_model/final_model"
PI_PREFIX = "plant18spe_1wbp_group"
SI_PREFIX = "plant18spe_1input_1wbp_group"

MRNA_LIMIT = 10000
TRAIN_BATCH = 64
TEST_BATCH = 128
INTRON_LEAST = 3
PATIENCE = 4
EPOCHS = 100


def load_data():
    df = pd.read_csv(DATA_CSV)
    df = df[df["seqs_len"] < MRNA_LIMIT]
    df = df[df["intron_counts"] >= INTRON_LEAST]
    return df.reset_index(drop=True)


def main():
    fold = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    intron_data = load_data()
    train_df = intron_data[intron_data["group_type"] != fold].sample(
        frac=1
    ).reset_index(drop=True)
    test_df = intron_data[intron_data["group_type"] == fold].sample(
        frac=1
    ).reset_index(drop=True)

    pi_model = load_model(os.path.join(MODEL_DIR, f"{PI_PREFIX}{fold}"))
    model = build_si_model(
        pi_model,
        MRNA_LIMIT,
        head_kernel_size=4,
        head_dropout=False,
        head_activation=False,
    )
    callbacks = [EarlyStopping(
        monitor="val_loss", patience=PATIENCE, verbose=0,
        restore_best_weights=True
    )]
    model.compile(
        loss="binary_crossentropy",
        optimizer=Adam(learning_rate=0.001),
        metrics=["accuracy"],
    )

    train_gen = si_generator(train_df, TRAIN_BATCH, MRNA_LIMIT)
    test_gen = si_generator(test_df, TEST_BATCH, MRNA_LIMIT)
    model.fit(
        train_gen,
        steps_per_epoch=int(math.ceil(len(train_df) * 2 / TRAIN_BATCH)),
        epochs=EPOCHS,
        callbacks=callbacks,
        validation_data=test_gen,
        validation_steps=int(math.ceil(len(test_df) * 2 / TEST_BATCH)),
        max_queue_size=10,
        workers=1,
        use_multiprocessing=False,
    )
    model.save(os.path.join(MODEL_DIR, f"{SI_PREFIX}{fold}"))


if __name__ == "__main__":
    main()
