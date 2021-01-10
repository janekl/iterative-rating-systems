import numpy as np
import pandas as pd
from scipy.stats import skellam


class IterativePoisson:
    """Iterative version of two-parameter Poisson model."""

    def __init__(self, c, h, lr, lambda_reg, rho, goal_cap=None, momentum=0.0):
        self.c = c  # Overall goal scoring rate
        self.h = h  # Home team advantage
        self.goal_cap = goal_cap  # Maximal numbers of goals for clipping
        self.lr = lr  # Learning rate
        self.lambda_reg = lambda_reg  # Regularization param
        self.rho = rho  # Correlation param
        self.momentum = momentum  # Momentum in SGD
        self.ratings = None
        self._columns = ('att', 'def')  # Column names in rating matrix

    def predict_proba_single(self, team_i, team_j):
        a_i, d_i = self.ratings.loc[team_i]
        a_j, d_j = self.ratings.loc[team_j]
        mu_i = np.exp(self.c + a_i - d_j + self.h)
        mu_j = np.exp(self.c + a_j - d_i)
        x, y = skellam.cdf([-1, 0], mu1=mu_i, mu2=mu_j)
        p1 = 1.0 - y
        p2 = y - x
        p3 = x
        return [p1, p2, p3]

    @staticmethod
    def get_update(goals_i, goals_j, mu_i, mu_j):
        return goals_i - mu_i, goals_j - mu_j

    def fit_predict(self, matches, *args):
        teams = np.unique(matches[['HomeTeam', 'AwayTeam']].values.flatten())
        self.ratings = pd.DataFrame(np.zeros((len(teams), 2)), index=teams, columns=self._columns)  # default dict?
        predictions = np.empty((len(matches), 3))
        va_i, vd_i, va_j, vd_j = 0.0, 0.0, 0.0, 0.0  # Momentum update
        for k, match in matches.iterrows():
            # Data and scoring rates
            team_i, team_j, goals_i, goals_j = match[['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']]
            a_i, d_i = self.ratings.loc[team_i]
            a_j, d_j = self.ratings.loc[team_j]
            mu_i = np.exp(self.c + a_i - d_j + self.h)
            mu_j = np.exp(self.c + a_j - d_i)
            # Predictions
            predictions[k] = self.predict_proba_single(team_i, team_j)
            # Clipping goals to avoid outliers
            if self.goal_cap != -1:
                goals_i = min(goals_i, self.goal_cap)
                goals_j = min(goals_j, self.goal_cap)
            # Updates
            update1, update2 = self.get_update(goals_i, goals_j, mu_i, mu_j)
            va_i = self.momentum * va_i + self.lr * (update1 - self.lambda_reg * (a_i - self.rho * d_i))
            vd_i = self.momentum * vd_i + self.lr * (-update2 - self.lambda_reg * (d_i - self.rho * a_i))
            va_j = self.momentum * va_j + self.lr * (update2 - self.lambda_reg * (a_j - self.rho * d_j))
            vd_j = self.momentum * vd_i + self.lr * (-update1 - self.lambda_reg * (d_j - self.rho * a_j))
            a_i += va_i
            d_i += vd_i
            a_j += va_j
            d_j += vd_j
            self.ratings.loc[team_i] = [a_i, d_i]
            self.ratings.loc[team_j] = [a_j, d_j]
        return predictions
