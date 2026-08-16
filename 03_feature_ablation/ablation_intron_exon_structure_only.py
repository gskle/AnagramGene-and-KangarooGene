"""Train paired-input models using only the strict i/e one-hot annotation.

Unlike the sequence-based ablations, this model receives a two-channel
one-hot matrix per position: [1, 0] for intron and [0, 1] for exon. The DNA
sequence itself is not provided, so the model can only use the intron/exon
layout to distinguish native from shuffled transcripts.

Usage:
    python ablation_intron_exon_structure_only.py [--i_value 1] [--e_value 1]
        [--fold 5] [--model_dir ./models] [--results_dir ./results]
"""

import argparse
import math
import os

import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "00_common"))
from model_builders import build_pi_model
from data_generators import pi_ie_generator, pi_prediction_generator


DATA_CSV = "/data1/lty/intron_pred/data/plant18_intronshuffle.csv"
MRNA_LIMIT = 10000
N_CHANNELS = 2
TRAIN_BATCH = 64
TEST_BATCH = 128
INTRON_LEAST = 3
PATIENCE = 6
EPOCHS = 100


def load_data():
    """Load plant sequences with at least three introns and length < 10 kb."""
    df = pd.read_csv(DATA_CSV)
    df = df[df["seqs_len"] < MRNA_LIMIT]
    df = df[df["intron_counts"] >= INTRON_LEAST]
    return df.reset_index(drop=True)


def predict_test(model, test_df):
    """Score the held-out fold in both pair directions."""
    steps = int(math.ceil(len(test_df) / TEST_BATCH))
    posneg = pi_prediction_generator(
        test_df, TEST_BATCH, MRNA_LIMIT, mode="posneg", ie_mode=True
    )
    negpos = pi_prediction_generator(
        test_df, TEST_BATCH, MRNA_LIMIT, mode="negpos", ie_mode=True
    )
    pred_posneg = np.asarray(model.predict(posneg, steps=steps, verbose=1))
    pred_negpos = np.asarray(model.predict(negpos, steps=steps, verbose=1))
    return pred_posneg, pred_negpos


def main(args):
    model_name = (
        f"intron_model_ie_onehot_i{int(args.i_value)}"
        f"_e{int(args.e_value)}_test{args.fold}"
    )
    model_path = os.path.join(args.model_dir, f"{model_name}.h5")
    results_path = os.path.join(args.results_dir, f"{model_name}_predictions.csv")
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    intron_data = load_data()
    train_df = intron_data[intron_data["group_type"] != args.fold].sample(
        frac=1
    ).reset_index(drop=True)
    test_df = intron_data[intron_data["group_type"] == args.fold]

    model = build_pi_model(MRNA_LIMIT, n_channels=N_CHANNELS)
    callbacks = [EarlyStopping(
        monitor="val_loss", patience=PATIENCE, verbose=1,
        restore_best_weights=True
    )]
    model.compile(
        loss="binary_crossentropy",
        optimizer=Adam(learning_rate=0.001),
        metrics=["accuracy"],
    )

    train_gen = pi_ie_generator(train_df, TRAIN_BATCH, MRNA_LIMIT)
    test_gen = pi_ie_generator(test_df, TEST_BATCH, MRNA_LIMIT)
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
    model.save_weights(model_path)
    print(f"saved weights: {model_path}")

    pred_posneg, pred_negpos = predict_test(model, test_df)
    result_df = test_df.copy()
    result_df["left_out"] = pred_posneg[:, -1]
    result_df["right_out"] = pred_negpos[:, 0]
    result_df.to_csv(results_path, index=False)
    print(f"saved predictions: {results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--i_value", type=float, default=1.0)
    parser.add_argument("--e_value", type=float, default=1.0)
    parser.add_argument("--fold", type=int, default=5)
    parser.add_argument("--model_dir", default="./models")
    parser.add_argument("--results_dir", default="./results")
    main(parser.parse_args())
