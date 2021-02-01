from collections import defaultdict

import numpy as np
from . import OrdinalLogisticRegression


class Elo:
    """Implementation of Elo model version proposed by Hvattum and Arntzen (2010)."""

    def __init__(self, k=10., lambda_goals=1., prior=1500., c=10., d=400.):
        self.k = k
        self.lambda_goals = lambda_goals
        self.c = c
        self.d = d
        self.prior = prior
        self.ratings = defaultdict(lambda: prior)
        self.model_olr = None

    def expected_result(self, rating_difference):
        return 1. / (1. + np.power(self.c, -rating_difference / self.d))

    def predict_proba_single(self, team_i, team_j):
        if self.model_olr is not None:
            return self.model_olr.predict_proba(np.array([[team_i - team_j]]))
        else:
            raise RuntimeError('Internal OLR model has not been fitted yet.')

    def estimate_ratings(self, matches, predict=False):
        predictions = []
        rating_differences = []
        for i, match in matches.iterrows():
            team_i, team_j, home_team_goal, away_team_goal = match[['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']]
            delta = home_team_goal - away_team_goal
            result = 0.5 if delta == 0 else float(delta > 0)
            rating_i = self.ratings[team_i]
            rating_j = self.ratings[team_j]
            rating_diff = rating_i - rating_j
            expected = self.expected_result(rating_diff)
            if predict:
                predictions.append(self.predict_proba_single(rating_i, rating_j)[0])
            update = self.k * np.power(1 + np.abs(delta), self.lambda_goals) * (result - expected)
            self.ratings[team_i] += update
            self.ratings[team_j] -= update
            rating_differences.append(rating_diff)
        return np.array(rating_differences), np.array(predictions)

    def fit_predict(self, matches, seasons_train, seasons_valid, seasons_test):
        train_index = matches['Season'].isin(seasons_train)
        valid_index = matches['Season'].isin(seasons_valid)
        test_index = matches['Season'].isin(seasons_test)
        matches_fit = matches[train_index | valid_index]
        predictions, rd = np.full((len(matches), 3), -1.0), np.full(len(matches), -1.0)
        rd[(train_index | valid_index).values], _ = self.estimate_ratings(matches_fit)
        rd_valid = rd[valid_index.values]
        X = rd_valid.reshape((-1, 1))
        y = matches.loc[valid_index, 'FTR'].values
        model_olr = OrdinalLogisticRegression()
        model_olr.fit(X, y)
        predictions[valid_index.values] = model_olr.predict_proba(X)
        self.model_olr = model_olr
        matches_test = matches[test_index]
        _, predictions[test_index.values] = self.estimate_ratings(matches_test, predict=True)
        return predictions
