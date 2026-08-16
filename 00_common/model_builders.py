"""Model construction for the CNN architectures.

The paired-input AnagramGene (PI), single-input transfer AnagramGene (SI),
and single-input KangarooGene deletion models share the encoder blocks
defined here.
"""

from tensorflow.keras.layers import (
    Activation,
    Add,
    BatchNormalization,
    Concatenate,
    Conv1D,
    Dense,
    Dropout,
    Flatten,
    Input,
    MaxPooling1D,
)
from tensorflow.keras.models import Model

DROP_RATE = 0.2


def convolutional_block(x, kernel_size, filters):
    """Residual block: three convolutions plus a 1x1 shortcut."""
    f1, f2, f3 = filters
    shortcut = x

    x = Conv1D(f1, kernel_size, strides=1, padding="same")(x)
    x = BatchNormalization(axis=2)(x)
    x = Activation("relu")(x)

    x = Conv1D(f2, kernel_size, strides=1, padding="same")(x)
    x = BatchNormalization(axis=2)(x)
    x = Activation("relu")(x)

    x = Conv1D(f3, kernel_size, strides=1, padding="same")(x)
    x = BatchNormalization(axis=2)(x)

    shortcut = Conv1D(f3, 1, strides=1, padding="same")(shortcut)
    shortcut = BatchNormalization(axis=2)(shortcut)
    x = Add()([x, shortcut])
    x = Activation("relu")(x)
    return x


def plant_fungi_encoder(inputs):
    """Encoder used for plant and fungi sequences (length <= 10 kb)."""
    x = Conv1D(256, 10, padding="valid")(inputs)
    x = Activation("relu")(x)
    x = Dropout(DROP_RATE)(x)
    x = MaxPooling1D(pool_size=5, padding="valid")(x)
    x = convolutional_block(x, 4, [64, 64, 128])
    x = MaxPooling1D(pool_size=4, padding="valid")(x)
    x = convolutional_block(x, 4, [32, 32, 64])
    x = MaxPooling1D(pool_size=4, padding="valid")(x)
    x = Dropout(DROP_RATE)(x)
    return x


def animal_encoder(inputs):
    """Encoder used for animal sequences (length <= 300 kb)."""
    x = Conv1D(16, 20, padding="valid")(inputs)
    x = Activation("relu")(x)
    x = Dropout(DROP_RATE)(x)
    x = MaxPooling1D(pool_size=20, padding="valid")(x)
    x = convolutional_block(x, 10, [16, 16, 32])
    x = MaxPooling1D(pool_size=5, padding="valid")(x)
    x = convolutional_block(x, 5, [32, 32, 64])
    x = MaxPooling1D(pool_size=5, padding="valid")(x)
    x = convolutional_block(x, 5, [64, 64, 128])
    x = MaxPooling1D(pool_size=5, padding="valid")(x)
    x = Dropout(DROP_RATE)(x)
    return x


def make_encoder(mrna_limit, n_channels=4):
    """Build the encoder as a standalone model with a stable layer name."""
    enc_input = Input(shape=(mrna_limit, n_channels), name="encoder_input")
    if mrna_limit >= 300000:
        enc_output = animal_encoder(enc_input)
    else:
        enc_output = plant_fungi_encoder(enc_input)
    return Model(inputs=enc_input, outputs=enc_output, name="encoder")


def build_pi_model(mrna_limit, n_channels=4):
    """Build the paired-input AnagramGene model.

    The two inputs receive the same transcript with the introns in native and
    shuffled order. The shared encoder is applied to both inputs, the encoded
    feature maps are concatenated along the time axis and the pair is
    classified by a small CNN head.
    """
    encoder = make_encoder(mrna_limit, n_channels=n_channels)

    input1 = Input(shape=(mrna_limit, n_channels), name="seqs_left")
    input2 = Input(shape=(mrna_limit, n_channels), name="seqs_right")

    x1 = encoder(input1)
    x2 = encoder(input2)
    x = Concatenate(axis=-2)([x1, x2])

    x = Conv1D(64, 5, padding="valid")(x)
    x = Activation("relu")(x)
    x = MaxPooling1D(pool_size=4, padding="valid")(x)

    x = Conv1D(32, 5, padding="valid")(x)
    x = Activation("relu")(x)
    x = Dropout(DROP_RATE)(x)
    x = MaxPooling1D(pool_size=4, padding="valid")(x)

    x = Flatten()(x)
    x = Dense(32, activation="relu")(x)
    output = Dense(2, activation="softmax")(x)
    return Model(inputs=[input1, input2], outputs=output)


def build_single_input_model(mrna_limit, n_channels=4):
    """Build the KangarooGene deletion model (single input).

    The model classifies a native transcript (positive) against the same
    transcript after deleting one intron (negative). The shared encoder is
    followed by a small CNN classification head.
    """
    encoder = make_encoder(mrna_limit, n_channels=n_channels)
    single_input = Input(shape=(mrna_limit, n_channels), name="single_seq")

    x = encoder(single_input)
    x = Conv1D(64, 5, padding="valid")(x)
    x = Activation("relu")(x)
    x = MaxPooling1D(pool_size=4, padding="valid")(x)
    x = Conv1D(32, 5, padding="valid")(x)
    x = Activation("relu")(x)
    x = Dropout(DROP_RATE)(x)
    x = MaxPooling1D(pool_size=4, padding="valid")(x)
    x = Flatten()(x)
    x = Dense(32, activation="relu")(x)
    output = Dense(2, activation="softmax")(x)
    return Model(inputs=single_input, outputs=output)


def build_si_model(pi_model, mrna_limit, head_kernel_size=4, head_dropout=True,
                   head_activation=True):
    """Build the single-input transfer model on top of a PI encoder.

    The encoder remains trainable for fine-tuning. Plant/fungi heads use two
    convolutions without an
    activation between them; animal heads use one convolution followed by
    ReLU, max pooling and dropout.
    """
    try:
        encoder = pi_model.get_layer("encoder")
    except ValueError:
        fallback = "model_4" if mrna_limit < 300000 else "model_2"
        encoder = pi_model.get_layer(fallback)
    encoder.trainable = True

    si_input = Input(shape=(mrna_limit, 4), name="single_seq")
    x = encoder(si_input)
    x = Conv1D(32, head_kernel_size, padding="valid")(x)
    if head_activation:
        x = Activation("relu")(x)
    if mrna_limit < 300000:
        x = Conv1D(32, head_kernel_size, padding="valid")(x)
    x = MaxPooling1D(pool_size=4, padding="valid")(x)
    if head_dropout:
        x = Dropout(DROP_RATE)(x)
    x = Flatten()(x)
    x = Dense(32, activation="relu")(x)
    output = Dense(2, activation="softmax")(x)
    return Model(inputs=si_input, outputs=output)
