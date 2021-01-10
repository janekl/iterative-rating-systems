import numpy as np
import pandas as pd
from scipy.special import expit as logistic


class IterativeOLR:
    """Iterative version of OLR model."""

    def __init__(self, c, h, lr, lambda_reg, momentum=0.0):
        self.c = c  # Overall goal scoring rate
        self.h = h  # Home team advantage
        self.lr = lr  # Learning rate
        self.lambda_reg = lambda_reg  # Regularization param
        self.momentum = momentum  # Momentum in SGD
        self.ratings = None

    def predict_proba_single(self, team_i, team_j):
        r_i = self.ratings[team_i]
        r_j = self.ratings[team_j]
        d_ij = r_i - r_j + self.h
        p1 = logistic(-self.c + d_ij)
        p3 = 1.0 - logistic(self.c + d_ij)
        p2 = 1.0 - p1 - p3
        return [p1, p2, p3]

    @staticmethod
    def get_update(result, preds):
        if result == 0:
            return 1 - preds[0]
        if result == 1:
            return preds[2] - preds[0]
        if result == 2:
            return preds[2] - 1

    def fit_predict(self, matches, *args):
        teams = np.unique(matches[['HomeTeam', 'AwayTeam']].values.flatten())
        self.ratings = pd.Series([0.0] * len(teams), index=teams)
        predictions = np.empty((len(matches), 3))
        v_i, v_j = 0.0, 0.0  # Momentum updates
        for k, match in matches.iterrows():
            # Data and ratings
            team_i, team_j, result = match[['HomeTeam', 'AwayTeam', 'FTR']]
            r_i = self.ratings[team_i]
            r_j = self.ratings[team_j]
            # Predictions
            preds = self.predict_proba_single(team_i, team_j)
            predictions[k] = preds
            # Updates
            update = self.get_update(result, preds)
            v_i = self.momentum * v_i + self.lr * (update - self.lambda_reg * r_i)
            v_j = self.momentum * v_j + self.lr * (-update - self.lambda_reg * r_j)
            r_i += v_i
            r_j += v_j
            # Save ratings
            self.ratings[team_i] = r_i
            self.ratings[team_j] = r_j
        return predictions
