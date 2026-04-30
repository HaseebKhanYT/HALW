"""Model definitions and training loops."""

import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv1D,
    Dense,
    Dropout,
    Flatten,
)
from tensorflow.keras.models import Sequential


def build_cnn(n_features, learning_rate=1e-3):
    """1D CNN baseline. Input shape: (n_features, 1)."""
    model = Sequential(
        [
            Conv1D(128, kernel_size=3, activation="relu", input_shape=(n_features, 1)),
            BatchNormalization(),
            Flatten(),
            Dense(256, activation="relu"),
            Dropout(0.4),
            Dense(128, activation="relu"),
            Dropout(0.3),
            Dense(64, activation="relu"),
            Dropout(0.2),
            Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train(model, X_train, y_train, X_val, y_val, epochs=50, batch_size=128, callbacks=None):
    """Fit with ReduceLROnPlateau on val_loss. Pass `callbacks` to extend."""
    cbs = [ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3)]
    if callbacks:
        cbs.extend(callbacks)
    return model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=cbs,
    )
