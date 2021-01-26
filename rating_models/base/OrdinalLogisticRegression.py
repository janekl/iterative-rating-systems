import numpy as np
from scipy import optimize
from scipy.special import expit as logistic
import utils.model
from ..IterativeOLR import IterativeOLR


class OrdinalLogisticRegression:
    """Base implementation of ordinal logistic regression."""

    def __init__(self, penalty=None, lambda_reg=0, weight=None, weight_params=None, eps=1e-8):
        assert penalty in {None, 'l1', 'l2'}
        self.penalty = penalty
        self.lambda_reg = lambda_reg
        self.eps = eps
        if weight is not None:
            self.weight_fun = getattr(utils.model, weight)
            if weight_params is not None:
                self.weight_params = weight_params
                print('Setting weight_params = {}'.format(weight_params))
        self.betas = None
        self.thresholds = None
        self.n_classes = None

    def get_features(self, X, prediction_phase=False):
        return X

    def _get_regularization(self, betas):
        regularization = 0.0
        if self.penalty == 'l1':
            regularization = self.lambda_reg * np.abs(betas).sum()
        elif self.penalty == 'l2':
            regularization = 0.5 * self.lambda_reg * np.square(betas).sum()
        return regularization

    def _negative_loglik(self, params, X, y, sample_weight, n_classes, eps):
        betas = params[:X.shape[1]]
        thresholds = np.cumsum(params[X.shape[1]:])
        preds = self._predict_proba2(X, thresholds, betas, n_classes, eps=eps)
        prob = preds[range(len(preds)), y]
        penalty = self._get_regularization(betas)
        if sample_weight is not None:
            return -np.sum(np.log(prob) * sample_weight) + penalty
        else:
            return -np.sum(np.log(prob)) + penalty

    def _get_grad_regularization(self, betas):
        regularization = 0.0
        if self.penalty == 'l1':
            regularization = self.lambda_reg * np.sign(betas)
        elif self.penalty == 'l2':
            regularization = self.lambda_reg * betas
        return regularization

    def _grad_negative_loglik(self, params, X, y, sample_weight, n_classes, eps):
        betas = params[:X.shape[1]]
        thresholds = np.cumsum(params[X.shape[1]:])
        proba = self._predict_proba2(X, thresholds, betas, n_classes, eps=eps)
        dt1 = np.zeros(len(proba))
        for i, preds in enumerate(proba):
            dt1[i] = -IterativeOLR.get_update(y[i], preds)
        p2 = proba[:, 2]
        p01 = 1 - p2
        dt2 = p01 * p2 / proba[range(len(proba)), y]
        if sample_weight is not None:
            dt1 *= sample_weight
            dt2 *= sample_weight
        dt2[y == 2] *= -1
        dt2[y == 0] = 0
        # Gradient only in the case of a three-outcome model for now
        grad = np.concatenate((dt1.dot(X) + self._get_grad_regularization(betas), (dt1.sum(), -dt2.sum())))
        return grad

    def fit(self, X, y, sample_weight=None):
        X = self.get_features(X)
        n_classes = len(np.unique(y))
        if n_classes != 3:
            raise ValueError("Currently only three-way outcome ordinal logistic regression is supported")
        # Define threshold vectors as -0.5 for the first splitpoint and then their diffs assuring they are positive
        # to restrict all splitpoints in OLR model to be increasing so that predictions make sense:
        params = np.concatenate([np.random.randn(X.shape[1]), [-0.5] + [1.0] * (n_classes - 2)])
        opt = optimize.minimize(self._negative_loglik, x0=params, args=(X, y, sample_weight, n_classes, self.eps),
                                bounds=((None, None),) * (X.shape[1] + 1) + ((0, None),) * (n_classes - 2),
                                method='L-BFGS-B', jac=self._grad_negative_loglik)
        params = opt.x
        self.betas = params[:X.shape[1]]
        self.thresholds = np.cumsum(params[X.shape[1]:])
        if not all(np.diff(self.thresholds) > 0):
            raise ValueError("Optimized thresholds should be non-decreasing")
        self.n_classes = n_classes

    def predict_proba(self, X):
        X = self.get_features(X, prediction_phase=True)
        preds = self._predict_proba2(X, self.thresholds, self.betas, self.n_classes, eps=self.eps)
        return preds

    def _predict_proba2(self, X, thresholds, betas, n_classes, eps):
        Xb = X.dot(betas)
        if not (np.diff(thresholds) > 0).all():
            return np.full((X.shape[0], n_classes), eps)
        preds = np.zeros((X.shape[0], n_classes))
        # Below we use the fact that logistic distribution is symmetric
        for c in range(n_classes - 1):
            z = logistic(thresholds[c] + Xb)
            preds[:, c] = z
            if c > 0:  # Probability of intermediate classes (draw)
                preds[:, c] -= preds[:, c - 1]
        # The last class (away team win)
        preds[:, -1] = 1 - z
        preds = np.maximum(preds, eps)
        return preds
