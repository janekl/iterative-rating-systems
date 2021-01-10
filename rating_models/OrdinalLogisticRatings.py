import numpy as np
from utils.model import generate_predictions, goals2class
from . import OrdinalLogisticRegression


class OrdinalLogisticRatings(OrdinalLogisticRegression):
    """Rating model based on ordinal logistic regression."""

    def __init__(self, **kwargs):
        super(OrdinalLogisticRatings, self).__init__(**kwargs)

    def _get_c_and_hta(self):
        """Derive home team advantage and intercept."""
        # TODO
        pass

    def _set_ratings_for_new_teams(self, X):
        teams = np.unique(X[['HomeTeam', 'AwayTeam']].values.flatten())
        rating_mean = self.betas.mean()
        for team in teams:
            if team not in self.teams:
                self.team_encoding[team] = len(self.teams)
                self.teams = np.append(self.teams, team)
                self.betas = np.append(self.betas, rating_mean)

    def _indicator_matrix(self, data, prediction_phase=False):
        matches = data[['HomeTeam', 'AwayTeam']]
        if not prediction_phase:
            # All teams to rate
            teams = np.unique(matches.values.flatten())
            # Encoding of teams
            team_encoding = dict(zip(teams, range(len(teams))))
            self.teams = teams
            self.team_encoding = team_encoding
        # Array to store match fixtures (design matrix)
        X = np.zeros((len(matches), len(self.teams)))
        for i, (home_team, away_team) in enumerate(zip(matches['HomeTeam'], matches['AwayTeam'])):
            X[i, self.team_encoding[home_team]] = 1.0
            X[i, self.team_encoding[away_team]] = -1.0
        return X

    def get_features(self, X, prediction_phase=False):
        if prediction_phase:
            self._set_ratings_for_new_teams(X)
        return self._indicator_matrix(X, prediction_phase=prediction_phase)

    def fit_predict(self, matches, seasons_train, seasons_valid, seasons_test):
        """Generate predictions for given seasons."""
        return generate_predictions(self, matches, seasons_valid + seasons_test, goals2class)
