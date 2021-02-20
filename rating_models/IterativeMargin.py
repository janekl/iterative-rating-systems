import numpy as np
from collections import defaultdict
from utils.model import predict_skellam_1x2


class IterativeMargin:
    """Iterative version of one-parameter Poisson model."""

    def __init__(self, c, h, lr, lambda_reg, goal_cap=None, momentum=0.0):
        self.c = c  # Overall goal scoring rate
        self.h = h  # Home team advantage
        self.goal_cap = goal_cap  # Maximal numbers of goals for clipping
        self.lr = lr  # Learning rate
        self.lambda_reg = lambda_reg  # Regularization param
        self.momentum = momentum  # Momentum in SGD
        self.ratings = defaultdict(float)
        self.ratings_history = None

    def predict_proba_single(self, team_i, team_j):
        r_i = self.ratings[team_i]
        r_j = self.ratings[team_j]
        mu_i = np.exp(self.c + r_i - r_j + self.h)
        mu_j = np.exp(self.c + r_j - r_i)
        return predict_skellam_1x2(mu_i, mu_j)

    @staticmethod
    def get_update(goals_i, goals_j, mu_i, mu_j):
        return (goals_i - goals_j) - (mu_i - mu_j)

    def fit_predict(self, matches, *args):
        predictions = np.empty((len(matches), 3))
        ratings_history = np.empty((len(matches), 2))
        v_i, v_j = 0.0, 0.0  # Momentum update
        for k, match in matches.iterrows():
            # Data and scoring rates
            team_i, team_j, goals_i, goals_j = match[['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']]
            r_i = self.ratings[team_i]
            r_j = self.ratings[team_j]
            mu_i = np.exp(self.c + r_i - r_j + self.h)
            mu_j = np.exp(self.c + r_j - r_i)
            # Predictions
            predictions[k] = self.predict_proba_single(team_i, team_j)
            # Clipping goals to avoid outliers
            if self.goal_cap != -1:
                goals_i = min(goals_i, self.goal_cap)
                goals_j = min(goals_j, self.goal_cap)
            # Updates
            update = self.get_update(goals_i, goals_j, mu_i, mu_j)
            v_i = self.momentum * v_i + self.lr * (update - self.lambda_reg * r_i)
            v_j = self.momentum * v_j + self.lr * (-update - self.lambda_reg * r_j)
            r_i += v_i
            r_j += v_j
            self.ratings[team_i] = r_i
            self.ratings[team_j] = r_j
            ratings_history[k] = [r_i, r_j]
        self.ratings_history = ratings_history
        return predictions
