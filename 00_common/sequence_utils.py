"""Shared sequence helpers for the intron models.

The functions in this module are used by every Python script so that
sequence encoding and dataset loading stay identical across analyses.
"""

import numpy as np
import pandas as pd


BASES = ("A", "C", "G", "T")

NTS = {
    "A": 0, "C": 1, "G": 2, "T": 3,
    "a": 0, "c": 1, "g": 2, "t": 3,
}


def one_hot_encoding(seq, seq_max):
    """Encode one DNA string with a fixed-length scheme.

    Returns a matrix of shape (seq_max, 4). Bases are encoded in ACGT column
    order and shorter sequences stay zero-padded.
    """
    mat = np.zeros((seq_max, 4), dtype="float32")
    for i, nt in enumerate(seq):
        mat[i, NTS[nt]] = 1.0
    return mat


def encode_sequences(sequences, maxlen):
    """Encode a list of DNA strings into an (n, maxlen, 4) one-hot array.

    Sequences are expected to contain only A/C/G/T bases; rows containing
    other characters should be removed with filter_agct before encoding.
    """
    encoded = [one_hot_encoding(seq, maxlen) for seq in sequences]
    return np.asarray(encoded, dtype="float32")


def filter_agct(df, seq_cols=("whole_seq_real", "whole_seq_false")):
    """Keep rows whose sequence columns contain only A/C/G/T bases."""
    mask = pd.Series(True, index=df.index)
    for col in seq_cols:
        mask &= df[col].str.contains(r"^[AGCT]*$", regex=True).fillna(False)
    return df[mask].reset_index(drop=True)


def load_sequences(csv_path, seq_col="whole_seq_real", label_col="whole_seq_false"):
    """Load real and shuffled sequences together with the binary label.

    The returned dataframe keeps the original CSV columns; the sequence
    columns are only read as strings and rows with non-AGCT bases are
    removed.
    """
    df = pd.read_csv(csv_path, dtype=str)
    df = df.dropna(subset=[seq_col, label_col])
    df = df[(df[seq_col].str.len() > 0) & (df[label_col].str.len() > 0)]
    return filter_agct(df, [seq_col, label_col])


def read_label_array(values, threshold=0.5):
    """Convert a score vector into 0/1 labels using the specified threshold."""
    return (np.asarray(values, dtype="float32") >= threshold).astype("int32")
