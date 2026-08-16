"""Train the paired-input model on intron sequences without any flank.

The model receives only the intron-only DNA one-hot matrix, without position
labels or flanking exons. 

Usage:
    python ablation_intron_only.py [--fold 5]
        [--model_dir ./models] [--results_dir ./results]
"""

import argparse
import math
import os
import re

import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "00_common"))
from model_builders import build_pi_model
from data_generators import pi_generator, pi_prediction_generator


DATA_CSV = "/data1/lty/intron_pred/data/plant18_intronshuffle.csv"
MRNA_LIMIT = 10000
TRAIN_BATCH = 64
TEST_BATCH = 128
INTRON_LEAST = 3
PATIENCE = 6
EPOCHS = 100
ALLOWED_BASES = re.compile(r"^[AGCTN]+$")


def load_data():
    """Load plant sequences filtered to clean A/C/G/T/N intron sequences."""
    df = pd.read_csv(DATA_CSV, dtype=str)
    df = df[df["seqs_len"].astype(int) < MRNA_LIMIT]
    df = df[df["intron_counts"].astype(int) >= INTRON_LEAST]
    df = df[df["whole_seq_real"].str.match(ALLOWED_BASES, na=False)]
    return df.reset_index(drop=True)


def predict_test(model, test_df):
    """Score the held-out fold in both pair directions."""
    steps = int(math.ceil(len(test_df) / TEST_BATCH))
    posneg = pi_prediction_generator(
        test_df, TEST_BATCH, MRNA_LIMIT, mode="posneg"
    )
    negpos = pi_prediction_generator(
        test_df, TEST_BATCH, MRNA_LIMIT, mode="negpos"
    )
    pred_posneg = np.asarray(model.predict(posneg, steps=steps, verbose=1))
    pred_negpos = np.asarray(model.predict(negpos, steps=steps, verbose=1))
    return pred_posneg, pred_negpos


def main(args):
    model_name = f"intron_model_only_noflank_test{args.fold}"
    model_path = os.path.join(args.model_dir, f"{model_name}.h5")
    results_path = os.path.join(args.results_dir, f"{model_name}_predictions.csv")
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    intron_data = load_data()
    train_df = intron_data[intron_data["group_type"] != args.fold].sample(
        frac=1
    ).reset_index(drop=True)
    test_df = intron_data[intron_data["group_type"] == args.fold]

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

    train_gen = pi_generator(train_df, TRAIN_BATCH, MRNA_LIMIT)
    test_gen = pi_generator(test_df, TEST_BATCH, MRNA_LIMIT)
    model.fit(
        train_gen,
        steps_per_epoch=int(math.ceil(len(train_df) * 2 / TRAIN_BATCH)),
        epochs=EPOCHS,
        callbacks=callbacks,
        validation_data=test_gen,
        validation_steps=int(math.ceil(len(test_df) * 2 / TEST_BATCH)),
        max_queue_size=50,
        workers=1,
        use_multiprocessing=False,
    )
    model.save(model_path)
    print(f"saved model: {model_path}")

    pred_posneg, pred_negpos = predict_test(model, test_df)
    result_df = test_df.copy()
    result_df["left_out"] = pred_posneg[:, -1]
    result_df["right_out"] = pred_negpos[:, 0]
    result_df.to_csv(results_path, index=False)
    print(f"saved predictions: {results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=5)
    parser.add_argument("--model_dir", default="./models")
    parser.add_argument("--results_dir", default="./results")
    main(parser.parse_args())
