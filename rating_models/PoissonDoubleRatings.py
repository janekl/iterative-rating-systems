import numpy as np
import utils.model
from utils.model import generate_predictions, goals2goals
from . import PoissonRegression


class PoissonDoubleRatings(PoissonRegression):
    """Rating model based on two-parameter Poisson model."""

    def __init__(self,  goal_cap=20, weight=None, weight_params=None, rho=0, **kwargs):
        super(PoissonDoubleRatings, self).__init__(**kwargs)  # Specify explicitly?
        self.teams = None
        self.team_encoding = None
        self.goal_cap = goal_cap
        self.rho = rho
        if weight is not None:
            self.weight_fun = getattr(utils.model, weight)
            if weight_params is not None:
                self.weight_params = weight_params
        else:
            self.weight_fun = None

    def _set_ratings_for_new_teams(self, X):  # TODO
        n = len(self.teams)
        rating_att_mean = self.betas[:n].mean()
        rating_def_mean = self.betas[n:].mean()
        assert len(self.betas[:n]) == len(self.betas[n:])
        teams_all = np.unique(X[['HomeTeam', 'AwayTeam']].values.flatten())
        for team in teams_all:
            if team not in self.teams:
                self.team_encoding[team] = len(self.teams)
                self.teams = np.append(self.teams, team)
                self.betas = np.hstack([self.betas[:n], rating_att_mean,
                                        self.betas[n:], rating_def_mean])

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
        n = len(self.teams)
        X = np.zeros((2 * m, 2 * n))
        for i, team_pair in enumerate(team_pairs.values, 0):
            # Away results follow home team results
            team1 = self.team_encoding[team_pair[0]]
            team2 = self.team_encoding[team_pair[1]]
            X[i, [team1, n + team2]] = [1, -1]
            X[i + m, [team2, n + team1]] = [1, -1]
        return X

    def _get_regularization(self, betas):
        regularization = 0.0
        if self.penalty == 'l1':
            regularization = self.lambda_reg * np.abs(betas).sum()
        elif self.penalty == 'l2':
            n = len(betas) // 2
            corr = betas[:n].dot(betas[n:])
            regularization = self.lambda_reg * (0.5 * np.square(betas).sum() - self.rho * corr)
        return regularization

    def _get_grad_regularization(self, betas):
        regularization = 0.0
        if self.penalty == 'l1':
            regularization = self.lambda_reg * np.sign(betas)
        elif self.penalty == 'l2':
            n = len(betas) // 2
            regularization = self.lambda_reg * (betas - self.rho * np.concatenate((betas[n:], betas[:n])))
        return regularization

    def get_features(self, X, prediction_phase=False):
        if prediction_phase:
            self._set_ratings_for_new_teams(X)
        return self._indicator_matrix(X, prediction_phase=prediction_phase)

    def fit_predict(self, matches, seasons_train, seasons_valid, seasons_test):
        """Generate predictions for given seasons."""
        return generate_predictions(self, matches, seasons_valid + seasons_test, goals2goals, {'cap': self.goal_cap})
