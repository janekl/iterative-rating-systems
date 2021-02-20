import numpy as np
import pandas as pd

EPS = 1e-8


def accuracy(preds, y, average=True):
    """Accuracy. Note: if there are two or more equal probabilities np.argmax returns always the first one!"""
    scores = y == np.argmax(preds, axis=1)
    if average:
        scores = scores.mean()
    return scores


def logloss(preds, y, eps=EPS, average=True):
    """Logloss metric."""
    preds = np.maximum(preds, eps)
    preds = preds / preds.sum(axis=1, keepdims=True)
    pred_probs = preds[np.array(range(len(preds))), y]
    scores = -np.log(pred_probs)
    if average:
        scores = scores.mean()
    return scores


def indicator(i, n=3):
    """Get base i-th vector, e.g., np.array([0, 0, 1, 0]) for i = 2 and n = 4 dimensions."""
    x = np.zeros(n)
    x[i] = 1.0
    return x


def rps(preds, y, average=True):
    """Rank probability score metric."""
    n_class = preds.shape[1]
    actual_cdf = np.array([indicator(i, n_class) for i in y]).cumsum(axis=1)
    preds_cdf = preds.cumsum(axis=1)
    scores = ((preds_cdf[:, :n_class-1] - actual_cdf[:, :n_class-1])**2).sum(axis=1)
    scores /= (n_class-1)
    if average:
        scores = scores.mean()
    return scores


def brier(preds, y, average=True):
    """Brier score metric."""
    n_class = preds.shape[1]
    actual_res = np.array([indicator(i, n_class) for i in y])
    scores = ((preds - actual_res)**2).mean(axis=1)
    if average:
        scores = scores.mean()
    return scores


def evaluate(preds, y, eval_functions=('accuracy', 'logloss', 'rps', 'brier'), average=True):
    """Compute evaluation metrics for predicitons given results."""
    return pd.Series({eval_fun: globals()[eval_fun](preds, y, average=average) for eval_fun in eval_functions})
