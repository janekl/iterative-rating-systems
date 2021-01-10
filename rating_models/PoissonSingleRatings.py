import numpy as np
import utils.model
from utils.model import generate_predictions, goals2goals
from . import PoissonRegression


class PoissonSingleRatings(PoissonRegression):
    """Rating model based on one-parameter Poisson model."""

    def __init__(self,  goal_cap=20, weight=None, weight_params=None, **kwargs):
        super(PoissonSingleRatings, self).__init__(**kwargs)  # Specify explicitly?
        self.teams = None
        self.team_encoding = None
        self.goal_cap = goal_cap
        if weight is not None:
            self.weight_fun = getattr(utils.model, weight)
            if weight_params is not None:
                self.weight_params = weight_params

    def _set_ratings_for_new_teams(self, X):
        teams = np.unique(X[['HomeTeam', 'AwayTeam']].values.flatten())
        rating_mean = self.betas.mean()
        for team in teams:
            if team not in self.teams:
                self.team_encoding[team] = len(self.teams)
                self.teams = np.append(self.teams, team)
                self.betas = np.append(self.betas, rating_mean)

    def _indicator_matrix(self, data, prediction_phase=False):
        team_pairs = data[['HomeTeam', 'AwayTeam']]
        if not prediction_phase:
            # All teams to rate
            teams = np.unique(team_pairs.values.flatten())
            # Encoding of teams
            team_encoding = {team: i for i, team in enumerate(teams)}
            self.teams = teams
            self.team_encoding = team_encoding
        # Array to store results
        m = len(data)
        X = np.zeros((2 * m, len(self.teams)))
        for i, team_pair in enumerate(team_pairs.values, 0):
            # Away results follow home team results
            team_enc = [self.team_encoding[j] for j in team_pair]
            X[i, team_enc] = [1, -1]      # Goal rate for home team
            X[i + m, team_enc] = [-1, 1]  # ------||----- away team
        return X

    def get_features(self, X, prediction_phase=False):
        if prediction_phase:
            self._set_ratings_for_new_teams(X)
        return self._indicator_matrix(X, prediction_phase=prediction_phase)

    def fit_predict(self, matches, seasons_train, seasons_valid, seasons_test):
        """Generate predictions for given seasons."""
        return generate_predictions(self, matches, seasons_valid + seasons_test, goals2goals, {'cap': self.goal_cap})