import numpy as np
from scipy import optimize
from scipy.stats import skellam
from abc import ABC, abstractmethod


class PoissonRegression(ABC):
    """Base implementation of Poisson regression."""

    def __init__(self, penalty='l2', lambda_reg=0, optimizer='BFGS'):
        assert penalty in [None, 'l1', 'l2']
        self.optimizer = optimizer
        self.penalty = penalty
        self.lambda_reg = lambda_reg
        self.c = None
        self.h = None
        self.betas = None

    def _predict_proba(self, X, m):
        Xb = X.dot(self.betas)
        eXb_plus_c = np.exp(Xb) + self.c
        mu1, mu2 = eXb_plus_c[:m] + self.h, eXb_plus_c[m:]
        p3 = skellam.cdf(-1, mu1=mu1, mu2=mu2)
        p23 = skellam.cdf(0, mu1=mu1, mu2=mu2)
        p1 = 1.0 - p23
        p2 = p23 - p3
        preds = np.array([p1, p2, p3]).T
        return preds

    def _get_regularization(self, betas):
        regularization = 0.0
        if self.penalty == 'l1':
            regularization = self.lambda_reg * np.abs(betas).sum()
        elif self.penalty == 'l2':
            regularization = 0.5 * self.lambda_reg * np.square(betas).sum()
        return regularization

    def _negative_loglik(self, params, X, y, sample_weight, m):
        c, h = params[:2]
        betas = params[2:]
        Xb = X.dot(betas) + c
        Xb[:m] += h
        penalty = self._get_regularization(betas)
        if sample_weight is not None:
            return -np.sum((y * Xb - np.exp(Xb)) * sample_weight) + penalty
        else:
            return -np.sum(y * Xb - np.exp(Xb)) + penalty

    def _get_grad_regularization(self, betas):
        regularization = 0.0
        if self.penalty == 'l1':
            regularization = self.lambda_reg * np.sign(betas)
        elif self.penalty == 'l2':
            regularization = self.lambda_reg * betas
        return regularization

    def _grad_negative_loglik(self, params, X, y, sample_weight, m):
        c, h = params[:2]
        betas = params[2:]
        Xb = X.dot(betas) + c
        Xb[:m] += h
        eXb = np.exp(Xb)
        z = y - eXb
        if sample_weight is not None:
            z *= sample_weight
        zs1, zs2 = z[:m].sum(), z[m:].sum()
        grad = -np.concatenate([[zs1 + zs2], [zs1], z.dot(X) - self._get_grad_regularization(betas)])
        return grad

    def _get_c_and_hta(self):
        """TODO: Function to derive home team advantage and intercept."""
        pass

    @abstractmethod
    def get_features(self, X, prediction_phase=False):
        pass

    def fit(self, X, y, sample_weight=None):
        X = self.get_features(X)
        n = X.shape[1]
        m = len(X) // 2
        y = np.reshape(y, -1, 'F')
        if sample_weight is not None:
            sample_weight = np.concatenate([sample_weight, sample_weight])
        # TODO: Better starting values based on y?
        params = np.concatenate([np.array([0.01, 0.3]), np.zeros(n)])
        opt = optimize.minimize(self._negative_loglik, x0=params, args=(X, y, sample_weight, m),
                                method=self.optimizer, jac=self._grad_negative_loglik)
        params = opt.x
        self.c, self.h = params[:2]
        self.betas = params[2:]

    def predict_proba(self, X):
        m = len(X)
        X = self.get_features(X, prediction_phase=True)
        preds = self._predict_proba(X, m)
        return preds
