"""Preprocessing: NaN handling, splitting, and scaling for Conv1D input."""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def drop_sparse_columns(df, min_non_null_fraction=0.5):
    """Drop non-numeric columns and columns whose non-null fraction is below the threshold.

    Non-numeric columns (e.g. a residual `text` column in a cached features CSV) can't be
    fed to StandardScaler / Conv1D, so they are dropped here alongside sparse columns. The
    label column is assumed numeric (0/1) and is preserved.
    """
    numeric = df.select_dtypes(include="number")
    return numeric.dropna(thresh=int(min_non_null_fraction * len(numeric)), axis=1)


def split(X, y, val_size=0.1, test_size=0.1, random_state=42):
    """Stratified 80/10/10 split (defaults). Returns train/val/test splits."""
    temp_size = val_size + test_size
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=temp_size, random_state=random_state, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=test_size / temp_size,
        random_state=random_state,
        stratify=y_temp,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def scale_for_conv1d(X_train, X_val, X_test):
    """Fit StandardScaler on train, transform val/test, reshape to (N, F, 1)."""
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    return (
        X_train[..., np.newaxis],
        X_val[..., np.newaxis],
        X_test[..., np.newaxis],
        scaler,
    )


def drop_nan_rows(X, y):
    """Drop rows where X has any NaN. Works on 2D or 3D X."""
    axes = tuple(range(1, X.ndim))
    mask = ~np.isnan(X).any(axis=axes)
    return X[mask], y[mask]
