"""Train paired-input models on intron-flank region annotations.

The dataset marks every position as downstream flank (d), intron (i) or
upstream flank (a). The DNA one-hot matrix is multiplied by the region
label at each position; all three labels default to 1, which keeps the
region annotation without changing the sequence values.

Usage:
    python ablation_region_labels.py [--d_value 1.0] [--i_value 1.0]
        [--a_value 1.0] [--fold 5]
        [--model_dir ./models_dia] [--results_dir ./results_dia]
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
from data_generators import pi_labeled_generator, pi_prediction_generator


DATA_CSV = ("/home/bingxing2/home/scx7008/intron/feature_compare/"
            "plant18spe_onlyintron.csv")
MRNA_LIMIT = 10000
TRAIN_BATCH = 64
TEST_BATCH = 128
INTRON_LEAST = 3
PATIENCE = 3
EPOCHS = 100


def load_data():
    """Load the intron-only dataset with region labels."""
    df = pd.read_csv(DATA_CSV)
    df = df[df["seqs_len"] < MRNA_LIMIT]
    df = df[df["intron_counts"] >= INTRON_LEAST]
    return df.reset_index(drop=True)


def predict_test(model, test_df, label_map):
    """Score the held-out fold in both pair directions."""
    steps = int(math.ceil(len(test_df) / TEST_BATCH))
    posneg = pi_prediction_generator(
        test_df, TEST_BATCH, MRNA_LIMIT, mode="posneg", label_map=label_map
    )
    negpos = pi_prediction_generator(
        test_df, TEST_BATCH, MRNA_LIMIT, mode="negpos", label_map=label_map
    )
    pred_posneg = np.asarray(model.predict(posneg, steps=steps, verbose=1))
    pred_negpos = np.asarray(model.predict(negpos, steps=steps, verbose=1))
    return pred_posneg, pred_negpos


def main(args):
    label_map = {"D": args.d_value, "I": args.i_value, "A": args.a_value}
    model_name = (
        f"plant_only_intron_model_d{args.d_value:.2f}"
        f"_i{args.i_value:.2f}_a{args.a_value:.2f}_test{args.fold}"
    ).replace(".", "_")
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

    train_gen = pi_labeled_generator(train_df, TRAIN_BATCH, MRNA_LIMIT, label_map)
    test_gen = pi_labeled_generator(test_df, TEST_BATCH, MRNA_LIMIT, label_map)
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

    pred_posneg, pred_negpos = predict_test(model, test_df, label_map)
    result_df = test_df.copy()
    result_df["left_out"] = pred_posneg[:, -1]
    result_df["right_out"] = pred_negpos[:, 0]
    result_df.to_csv(results_path, index=False)
    print(f"saved predictions: {results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--d_value", type=float, default=1.0)
    parser.add_argument("--i_value", type=float, default=1.0)
    parser.add_argument("--a_value", type=float, default=1.0)
    parser.add_argument("--fold", type=int, default=5)
    parser.add_argument("--model_dir", default="./models_dia")
    parser.add_argument("--results_dir", default="./results_dia")
    main(parser.parse_args())
