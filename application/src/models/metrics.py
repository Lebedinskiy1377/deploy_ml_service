import numpy as np


def wape(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> np.float64:
    return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true))


def mape(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> np.float64:
    return np.mean(np.abs((y_true - y_pred) / y_true))


def mpe(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> np.float64:
    return np.mean((y_true - y_pred) / y_true)


def smape(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> np.float64:
    return np.mean(np.abs(y_true - y_pred) / ((np.abs(y_true) + np.abs(y_pred)) / 2))


def ape(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> np.ndarray:
    return np.abs(y_true - y_pred) / np.abs(y_true)