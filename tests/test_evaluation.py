import numpy as np
from utils.evaluation import logloss, rps, brier, accuracy, EPS


preds = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],

    [1 / 3] * 3,
    [1 / 3] * 3,
    [1 / 3] * 3,

    [0.5, 0.2, 0.3],
    [0.5, 0.2, 0.3],
    [0.5, 0.2, 0.3],
])

y = [0, 0, 0] + [0, 1, 2] * 2


def test_logloss():
    res1 = logloss(preds, y, average=False)
    res2 = -np.log((1 / (1 + 2 * EPS), EPS / (1 + 2 * EPS), EPS / (1 + 2 * EPS),
                    1 / 3, 1 / 3, 1 / 3,
                    0.5, 0.2, 0.3))
    assert np.allclose(res1, res2)


def test_rps():
    res1 = rps(preds, y, average=False)
    res2 = 0.5 * np.array((0, 1, 2,
                           (2 / 3)**2 + (1 / 3)**2, (1 / 3)**2 + (1 / 3)**2, (2 / 3)**2 + (1 / 3)**2,
                           0.5**2 + 0.3**2, 0.5**2 + 0.3**2, 0.5**2 + 0.7**2))
    assert np.allclose(res1, res2)


def test_brier():
    res1 = brier(preds, y, average=False)
    res2 = (1 / 3) * np.array((0, 2, 2,
                               2 / 3, 2 / 3, 2 / 3,
                               0.5**2 + 0.2**2 + 0.3**2, 0.5**2 + 0.8**2 + 0.3**2, 0.5**2 + 0.2**2 + 0.7**2))
    assert np.allclose(res1, res2)


def test_accuracy():
    res = accuracy(preds, y)
    assert res == 1 / 3
