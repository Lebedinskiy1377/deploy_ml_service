import numpy as np


EPSILON = np.finfo(float).eps


def _as_float_arrays(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)


def wape(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> np.float64:
    y_true, y_pred = _as_float_arrays(y_true, y_pred)
    denominator = max(float(np.sum(np.abs(y_true))), EPSILON)
    return np.float64(np.sum(np.abs(y_true - y_pred)) / denominator)


def mape(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> np.float64:
    y_true, y_pred = _as_float_arrays(y_true, y_pred)
    denominator = np.maximum(np.abs(y_true), EPSILON)
    return np.float64(np.mean(np.abs(y_true - y_pred) / denominator))


def mpe(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> np.float64:
    y_true, y_pred = _as_float_arrays(y_true, y_pred)
    denominator = np.maximum(np.abs(y_true), EPSILON)
    return np.float64(np.mean((y_true - y_pred) / denominator))


def smape(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> np.float64:
    y_true, y_pred = _as_float_arrays(y_true, y_pred)
    denominator = np.maximum(np.abs(y_true) + np.abs(y_pred), EPSILON)
    return np.float64(np.mean(2 * np.abs(y_true - y_pred) / denominator))


def ape(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> np.ndarray:
    y_true, y_pred = _as_float_arrays(y_true, y_pred)
    denominator = np.maximum(np.abs(y_true), EPSILON)
    return np.abs(y_true - y_pred) / denominator
