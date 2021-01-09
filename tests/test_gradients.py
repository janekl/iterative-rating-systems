import numpy as np
from numpy.testing import assert_allclose

from rating_models import OrdinalLogisticRatings
from optimize_model import train_valid_test_split
from utils.data import get_data
from utils.model import goals2class


*_, seasons_all = train_valid_test_split("TEST", True)
matches = get_data(seasons_all)


def numerical_grad(model, params, args, h=1e-6):
    grad = np.zeros_like(params)
    for i in range(len(params)):
        params1 = np.copy(params)
        params2 = np.copy(params)
        params1[i] += h
        params2[i] -= h
        grad[i] = (model._negative_loglik(params1, *args) - model._negative_loglik(params2, *args)) / (2 * h)
    return grad


def test_ordinal_logistic_ratings():
    model_olr = OrdinalLogisticRatings(lambda_reg=1.0, penalty='l1')
    X = model_olr.get_features(matches[['HomeTeam', 'AwayTeam']])
    y = matches[['FTHG', 'FTAG']].apply(goals2class, axis=1).values
    sample_weight = np.random.uniform(0.5, 2., len(y))
    args = (X, y, sample_weight, 3, 1e-8)
    params = np.concatenate((np.random.randn(X.shape[1]), (-0.5, 1.0)))
    grad_exact = model_olr._grad_negative_loglik(params, *args)
    grad_approx = numerical_grad(model_olr, params, args)
    assert_allclose(grad_exact, grad_approx, rtol=1e-6)
