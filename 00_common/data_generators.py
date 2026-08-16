"""Data generators shared by the training and prediction scripts.

The paired-input generators create two input orders for every transcript:
(native sequence, intron-shuffled sequence) and
(intron-shuffled sequence, native sequence). The first order is assigned
integer class 1 and therefore receives the one-hot label [0, 1] from
``to_categorical``; the reversed order is assigned integer class 0 and
receives [1, 0].

The single-input generators provide native and intron-shuffled sequences for
the Anagram-SI and KangarooGene models, with native sequences assigned
class 1 and intron-shuffled sequences assigned class 0.
"""

import numpy as np
import threading
from tensorflow.keras.utils import to_categorical

from sequence_utils import encode_sequences


class threadsafe_iter:
    """Serialize access to an underlying generator with a mutex."""

    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return next(self.it)


def threadsafe_generator(func):
    """Wrap a generator function so concurrent next() calls are serialized."""

    def wrapper(*args, **kwargs):
        return threadsafe_iter(func(*args, **kwargs))

    return wrapper


def label_encoding(strings, maxlen, label_map):
    """Return a per-position label matrix of shape (n, maxlen, 1).

    Characters present in label_map are replaced by their numeric value;
    any other character becomes zero. Sequence one-hots are multiplied by
    these per-position labels.
    """
    n = len(strings)
    result = np.zeros((n, maxlen, 1), dtype="float32")
    for idx, string in enumerate(strings):
        text = str(string).upper()[:maxlen]
        for pos, char in enumerate(text):
            value = label_map.get(char, 0.0)
            if value:
                result[idx, pos, 0] = value
    return result


def ie_onehot(strings, maxlen):
    """Encode whole_type strings as an i/e one-hot (n, maxlen, 2).

    Intron positions are encoded as [1, 0] and exon positions as [0, 1],
    using the strict intron/exon position-label encoding.
    """
    n = len(strings)
    result = np.zeros((n, maxlen, 2), dtype="float32")
    for idx, string in enumerate(strings):
        text = str(string).upper()[:maxlen]
        for pos, char in enumerate(text):
            if char == "I":
                result[idx, pos, 0] = 1.0
            elif char == "E":
                result[idx, pos, 1] = 1.0
    return result


def _pair_arrays(chunk, maxlen, label_map=None, ie_mode=False):
    """Encode a chunk into (seqs_left, seqs_right, labels) pair arrays."""
    if ie_mode:
        pos = ie_onehot(chunk["whole_type_real"], maxlen)
        neg = ie_onehot(chunk["whole_type_false"], maxlen)
    else:
        pos = encode_sequences(chunk["whole_seq_real"], maxlen)
        neg = encode_sequences(chunk["whole_seq_false"], maxlen)
        if label_map is not None:
            pos = pos * label_encoding(chunk["whole_type_real"], maxlen, label_map)
            neg = neg * label_encoding(chunk["whole_type_false"], maxlen, label_map)

    seqs_left = np.concatenate([pos, neg], axis=0)
    seqs_right = np.concatenate([neg, pos], axis=0)
    labels = to_categorical(
        np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    )
    return seqs_left, seqs_right, labels


@threadsafe_generator
def pi_generator(df, batch_size, maxlen):
    """Yield paired-input batches from a dataframe with real/false columns."""
    while True:
        each_batch = int(batch_size / 2)
        n_batches = int(np.ceil(len(df) / each_batch))
        for i in range(n_batches):
            chunk = df.iloc[each_batch * i: each_batch * (i + 1)]
            pos = encode_sequences(chunk["whole_seq_real"], maxlen)
            neg = encode_sequences(chunk["whole_seq_false"], maxlen)

            seqs_left = np.concatenate([pos, neg])
            seqs_right = np.concatenate([neg, pos])
            labels = to_categorical(
                np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
            )
            rand_index = np.random.permutation(len(seqs_left))
            yield (
                {
                    "seqs_left": seqs_left[rand_index],
                    "seqs_right": seqs_right[rand_index],
                },
                labels[rand_index],
            )


@threadsafe_generator
def si_generator(df, batch_size, maxlen):
    """Yield balanced single-input batches: real=1, shuffled=0."""
    while True:
        each_batch = int(batch_size / 2)
        n_batches = int(np.ceil(len(df) / each_batch))
        for i in range(n_batches):
            chunk = df.iloc[each_batch * i: each_batch * (i + 1)]
            pos = encode_sequences(chunk["whole_seq_real"], maxlen)
            neg = encode_sequences(chunk["whole_seq_false"], maxlen)
            seqs = np.concatenate([pos, neg])
            labels = to_categorical(
                np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
            )
            rand_index = np.random.permutation(len(seqs))
            yield seqs[rand_index], labels[rand_index]


@threadsafe_generator
def pi_labeled_generator(df, batch_size, maxlen, label_map):
    """Yield paired inputs weighted by per-position region labels.

    The DNA one-hot inputs are multiplied by the position-wise weights in
    ``label_map``; these region labels are distinct from the sample class
    labels assigned according to the two input orders.
    """
    while True:
        each_batch = int(batch_size / 2)
        n_batches = int(np.ceil(len(df) / each_batch))
        for i in range(n_batches):
            chunk = df.iloc[each_batch * i: each_batch * (i + 1)]
            seqs_left, seqs_right, labels = _pair_arrays(
                chunk, maxlen, label_map=label_map
            )
            rand_index = np.random.permutation(len(seqs_left))
            yield (
                {
                    "seqs_left": seqs_left[rand_index],
                    "seqs_right": seqs_right[rand_index],
                },
                labels[rand_index],
            )


@threadsafe_generator
def pi_ie_generator(df, batch_size, maxlen):
    """Paired-input generator for the strict i/e one-hot ablation."""
    while True:
        each_batch = int(batch_size / 2)
        n_batches = int(np.ceil(len(df) / each_batch))
        for i in range(n_batches):
            chunk = df.iloc[each_batch * i: each_batch * (i + 1)]
            seqs_left, seqs_right, labels = _pair_arrays(
                chunk, maxlen, ie_mode=True
            )
            rand_index = np.random.permutation(len(seqs_left))
            yield (
                {
                    "seqs_left": seqs_left[rand_index],
                    "seqs_right": seqs_right[rand_index],
                },
                labels[rand_index],
            )


@threadsafe_generator
def pi_prediction_generator(df, batch_size, maxlen, mode="posneg",
                            label_map=None, ie_mode=False):
    """Order-preserving paired-input batches for ablation prediction."""
    n_batches = int(np.ceil(len(df) / batch_size))
    for i in range(n_batches):
        chunk = df.iloc[batch_size * i: batch_size * (i + 1)]
        if ie_mode:
            pos = ie_onehot(chunk["whole_type_real"], maxlen)
            neg = ie_onehot(chunk["whole_type_false"], maxlen)
        else:
            pos = encode_sequences(chunk["whole_seq_real"], maxlen)
            neg = encode_sequences(chunk["whole_seq_false"], maxlen)
            if label_map is not None:
                pos = pos * label_encoding(chunk["whole_type_real"], maxlen, label_map)
                neg = neg * label_encoding(chunk["whole_type_false"], maxlen, label_map)
        if mode == "posneg":
            yield {"seqs_left": pos, "seqs_right": neg}
        else:
            yield {"seqs_left": neg, "seqs_right": pos}


@threadsafe_generator
def prediction_generator(df, batch_size, maxlen, seq_col="whole_seq_real"):
    """Yield batches of a single sequence column for prediction."""
    while True:
        n_batches = int(np.ceil(len(df) / batch_size))
        for i in range(n_batches):
            chunk = df.iloc[batch_size * i: batch_size * (i + 1)]
            yield encode_sequences(chunk[seq_col], maxlen)
