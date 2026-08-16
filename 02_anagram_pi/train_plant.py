"""Train the plant paired-input AnagramGene model on all five folds.

Each model is trained on four orthogroup partitions and validated on the
remaining one. Positive examples use the transcript with native intron order,
negative examples use the same transcript with shuffled introns.

Usage:
    python train_plant.py [test_fold]
"""

import math
import os
import sys

import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "00_common"))
from model_builders import build_pi_model
from data_generators import pi_generator
from sequence_utils import filter_agct

DATA_CSV = "/data1/lty/intron_pred/data/plant18_intronshuffle.csv"
MODEL_DIR = "/data1/lty/intron_pred/new_model/final_model"
MODEL_PREFIX = "plant18spe_1wbp_group"

MRNA_LIMIT = 10000
TRAIN_BATCH = 64
TEST_BATCH = 128
INTRON_LEAST = 3
PATIENCE = 6
EPOCHS = 100


def load_data():
    df = pd.read_csv(DATA_CSV)
    df = df[df["seqs_len"] < MRNA_LIMIT]
    df = df[df["intron_counts"] >= INTRON_LEAST]
    df = filter_agct(df)
    return df.reset_index(drop=True)


def main():
    test_fold = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    intron_data = load_data()
    train_df = intron_data[intron_data["group_type"] != test_fold].sample(frac=1).reset_index(drop=True)
    test_df = intron_data[intron_data["group_type"] == test_fold].sample(frac=1).reset_index(drop=True)

    model = build_pi_model(MRNA_LIMIT)
    callbacks = [EarlyStopping(
        monitor="val_loss", patience=PATIENCE, verbose=1,
        restore_best_weights=True
    )]
    model.compile(
        loss="binary_crossentropy",
        optimizer=Adam(learning_rate=0.001),
        metrics=["accuracy"],
    )
    model.summary()

    train_gen = pi_generator(train_df, TRAIN_BATCH, MRNA_LIMIT)
    test_gen = pi_generator(test_df, TEST_BATCH, MRNA_LIMIT)
    steps_train = int(math.ceil(len(train_df) * 2 / TRAIN_BATCH))
    steps_test = int(math.ceil(len(test_df) * 2 / TEST_BATCH))

    model.fit(
        train_gen,
        steps_per_epoch=steps_train,
        epochs=EPOCHS,
        callbacks=callbacks,
        validation_data=test_gen,
        validation_steps=steps_test,
        max_queue_size=10,
        workers=1,
        use_multiprocessing=False,
    )

    model.save(os.path.join(MODEL_DIR, f"{MODEL_PREFIX}{test_fold}"))


if __name__ == "__main__":
    main()
