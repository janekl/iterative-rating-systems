import numpy as np
import pandas as pd
from scipy import optimize
from scipy.stats import skellam
from scipy.special import expit as logistic
import tools
from tools import generate_predictions, goals2class, goals2goals
from abc import ABC, abstractmethod


class Elo:
    """Implementation of Elo model version proposed by Hvattum and Arntzen (2010)."""

    def __init__(self, k=10., lambda_goals=1., prior=1500., c=10., d=400.):
        self.k = k
        self.lambda_goals = lambda_goals
        self.c = c
        self.d = d
        self.prior = prior
        self.ratings = {}
        self.model_olr = None

    def expected_result(self, rating_difference):
        return 1. / (1. + np.power(self.c, -rating_difference / self.d))

    def predict_proba_single(self, team_i, team_j):
        if self.model_olr is not None:
            return self.model_olr.predict_proba(np.array([[team_i - team_j]]))
        else:
            raise RuntimeError('Internal OLR model has not been fitted yet.')

    def _set_ratings_for_new_teams(self, teams):
        for team in teams:
            if team not in self.ratings:
                self.ratings[team] = self.prior

    def estimate_ratings(self, matches, initialize=True, predict=False):
        teams = np.unique(matches[['HomeTeam', 'AwayTeam']].values.flatten())
        if initialize:
            # TODO: Possibly get rid of this and just set ratings for new teams
            self.ratings = {team: self.prior for team in teams}
        else:
            self._set_ratings_for_new_teams(teams)
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
        _, predictions[test_index.values] = self.estimate_ratings(matches_test, initialize=False, predict=True)
        return predictions


class OrdinalLogisticRegression:
    """Base implementation of ordinal logistic regression."""

    def __init__(self, penalty=None, lambda_reg=0, weight=None, weight_params=None, eps=1e-8):
        assert penalty in [None, 'l1', 'l2']
        self.penalty = penalty
        self.lambda_reg = lambda_reg
        self.eps = eps
        if weight is not None:
            self.weight_fun = getattr(tools, weight)
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

    def _neg_loglik(self, params, X, y, sample_weight, n_classes, eps):
        betas = params[:X.shape[1]]
        thresholds = np.cumsum(params[X.shape[1]:])
        preds = self._predict_proba2(X, thresholds, betas, n_classes, eps=eps)
        prob = preds[range(len(preds)), y]  # np.array([preds[i, j] for i, j in enumerate(y)])
        penalty = self._get_regularization(betas)
        if sample_weight is not None:
            return -np.sum(np.log(prob) * sample_weight) + penalty
        else:
            return -np.sum(np.log(prob)) + penalty

    def fit(self, X, y, sample_weight=None):
        X = self.get_features(X)
        n_classes = len(np.unique(y))
        # Define threshold vectors as -0.5 for the first splitpoint
        # and then their differences assuring they are positive to
        # restrict all splitpoints in ORL model to  be increasing:
        params = np.concatenate([np.random.randn(X.shape[1]), [-0.5] + [1.0] * (n_classes - 2)])
        opt = optimize.minimize(self._neg_loglik, x0=params, args=(X, y, sample_weight, n_classes, self.eps),
                                bounds=((None, None),) * (X.shape[1] + 1) + ((0, None),) * (n_classes - 2),
                                method='L-BFGS-B', jac=False, options={'disp': False, 'maxiter': 10 ** 4})
        params = opt.x
        self.betas = params[:X.shape[1]]
        self.thresholds = np.cumsum(params[X.shape[1]:])
        # Predictions make sense only if thresholds are not decreasing
        if not all(np.diff(self.thresholds) > 0):
            msg = 'Optimized thresholds in OrdinalLogisticRegression should be in non-decreasing.'
            raise ValueError(msg)
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
        # Array to store results
        X = np.zeros((len(matches), len(self.teams)))
        for i, (home_team, away_team) in enumerate(zip(matches['HomeTeam'], matches['AwayTeam'])):
            X[i, self.team_encoding[home_team]] = 1.0
            X[i, self.team_encoding[away_team]] = -1.0
        # for i, away_team in enumerate(matches['AwayTeam'], 0):
        #     X[i, self.team_encoding[away_team]] = -1.0
        return X

    def get_features(self, X, prediction_phase=False):
        if prediction_phase:
            self._set_ratings_for_new_teams(X)
        return self._indicator_matrix(X, prediction_phase=prediction_phase)

    def fit_predict(self, matches, seasons_train, seasons_valid, seasons_test):
        """Generate predictions for given seasons."""
        return generate_predictions(self, matches, seasons_valid + seasons_test, goals2class)


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

    def get_update(self, result, preds):
        if result == 0:
            return 1 - preds[0]
        if result == 1:
            return preds[2] - preds[0]
        if result == 2:
            return preds[2] - 1

    def fit_predict(self, matches, *args):
        teams = np.unique(matches[['HomeTeam', 'AwayTeam']].values.flatten())
        self.ratings = pd.Series([0.0] * len(teams), index=teams)
        predictions = []
        v_i, v_j = 0.0, 0.0  # Momentum updates
        for i, match in matches.iterrows():
            # Data and ratings
            team_i, team_j, result = match[['HomeTeam', 'AwayTeam', 'FTR']]
            r_i = self.ratings[team_i]
            r_j = self.ratings[team_j]
            # Predictions
            preds = self.predict_proba_single(team_i, team_j)
            predictions.append(preds)
            # Updates
            update = self.get_update(result, preds)
            v_i = self.momentum * v_i + self.lr * (update - self.lambda_reg * r_i)
            v_j = self.momentum * v_j + self.lr * (-update - self.lambda_reg * r_j)
            r_i += v_i
            r_j += v_j
            # Save ratings
            self.ratings[team_i] = r_i
            self.ratings[team_j] = r_j
        return np.array(predictions)


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
                                method=self.optimizer, jac=self._grad_negative_loglik,
                                options={'disp': False, 'maxiter': None})
        params = opt.x
        self.c, self.h = params[:2]
        self.betas = params[2:]
        
    def predict_proba(self, X):
        m = len(X)
        X = self.get_features(X, prediction_phase=True)
        preds = self._predict_proba(X, m)
        return preds


class PoissonSingleRatings(PoissonRegression):
    """Rating model based on one-parameter Poisson model."""

    def __init__(self,  goal_cap=20, weight=None, weight_params=None, **kwargs):
        super(PoissonSingleRatings, self).__init__(**kwargs)  # Specify explicitly?
        self.teams = None
        self.team_encoding = None
        self.goal_cap = goal_cap
        if weight is not None:
            self.weight_fun = getattr(tools, weight)
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


class PoissonDoubleRatings(PoissonRegression):
    """Rating model based on two-parameter Poisson model."""

    def __init__(self,  goal_cap=20, weight=None, weight_params=None, rho=0, **kwargs):
        super(PoissonDoubleRatings, self).__init__(**kwargs)  # Specify explicitly?
        self.teams = None
        self.team_encoding = None
        self.goal_cap = goal_cap
        self.rho = rho
        if weight is not None:
            self.weight_fun = getattr(tools, weight)
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


class IterativeMargin:
    """Iterative version of one-parameter Poisson model."""

    def __init__(self, c, h, lr, lambda_reg, goal_cap=None, momentum=0.0):
        self.c = c  # Overall goal scoring rate
        self.h = h  # Home team advantage
        self.goal_cap = goal_cap  # Maximal numbers of goals for clipping
        self.lr = lr  # Learning rate
        self.lambda_reg = lambda_reg  # Regularization param
        self.momentum = momentum  # Momentum in SGD
        self.ratings = None
        self.ratings_history = None

    def predict_proba_single(self, team_i, team_j):
        r_i = self.ratings[team_i]
        r_j = self.ratings[team_j]
        mu_i = np.exp(self.c + r_i - r_j + self.h)
        mu_j = np.exp(self.c + r_j - r_i)
        x, y = skellam.cdf([-1, 0], mu1=mu_i, mu2=mu_j)
        p1 = 1.0 - y
        p2 = y - x
        p3 = x
        return [p1, p2, p3]

    def fit_predict(self, matches, *args):
        teams = np.unique(matches[['HomeTeam', 'AwayTeam']].values.flatten())
        self.ratings = pd.Series([0.0] * len(teams), index=teams)
        predictions = []
        ratings_history = []
        v_i, v_j = 0.0, 0.0  # Momentum udpate
        for i, match in matches.iterrows():
            # Data and scoring rates
            team_i, team_j, goals_i, goals_j = match[['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']]
            r_i = self.ratings[team_i]
            r_j = self.ratings[team_j]
            mu_i = np.exp(self.c + r_i - r_j + self.h)
            mu_j = np.exp(self.c + r_j - r_i)
            # Predictions
            predictions.append(self.predict_proba_single(team_i, team_j))
            # Clipping goals to avoid outliers
            if self.goal_cap != -1:
                goals_i = min(goals_i, self.goal_cap)
                goals_j = min(goals_j, self.goal_cap)
            # Updates
            margin_diff = (goals_i - goals_j) - (mu_i - mu_j)
            v_i = self.momentum * v_i + self.lr * (margin_diff - self.lambda_reg * r_i)
            v_j = self.momentum * v_j + self.lr * (-margin_diff - self.lambda_reg * r_j)
            r_i += v_i
            r_j += v_j
            self.ratings[team_i] = r_i
            self.ratings[team_j] = r_j
            ratings_history.append([r_i, r_j])
        self.ratings_history = ratings_history
        return np.array(predictions)


class IterativePoisson:
    """Iterative version of two-parameter Poisson model."""

    def __init__(self, c, h, lr, lambda_reg, rho, goal_cap=None, momentum=0.0):
        self.c = c  # Overall goal scoring rate
        self.h = h  # Home team advantage
        self.goal_cap = goal_cap  # Maximal numbers of goals for clipping
        self.lr = lr  # Learning rate
        self.lambda_reg = lambda_reg  # Regularization param
        self.rho = rho  # Correlation param
        self.columns = ['att', 'def']  # Column names in rating matrix
        self.momentum = momentum  # Momentum in SGD
        self.ratings = None

    def predict_proba_single(self, team_i, team_j):
        a_i, d_i = self.ratings.loc[team_i, self.columns]
        a_j, d_j = self.ratings.loc[team_j, self.columns]
        mu_i = np.exp(self.c + a_i - d_j + self.h)
        mu_j = np.exp(self.c + a_j - d_i)
        x, y = skellam.cdf([-1, 0], mu1=mu_i, mu2=mu_j)
        p1 = 1.0 - y
        p2 = y - x
        p3 = x
        return [p1, p2, p3]

    def fit_predict(self, matches, *args):
        teams = np.unique(matches[['HomeTeam', 'AwayTeam']].values.flatten())
        self.ratings = pd.DataFrame(np.zeros((len(teams), 2)), index=teams, columns=self.columns)
        predictions = []
        va_i, vd_i, va_j, vd_j = 0.0, 0.0, 0.0, 0.0  # Momentum update
        for i, match in matches.iterrows():
            # Data and scoring rates
            team_i, team_j, goals_i, goals_j = match[['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']]
            a_i, d_i = self.ratings.loc[team_i]
            a_j, d_j = self.ratings.loc[team_j]
            mu_i = np.exp(self.c + a_i - d_j + self.h)
            mu_j = np.exp(self.c + a_j - d_i)
            # Predictions
            predictions.append(self.predict_proba_single(team_i, team_j))
            # Clipping goals to avoid outliers
            if self.goal_cap != -1:
                goals_i = min(goals_i, self.goal_cap)
                goals_j = min(goals_j, self.goal_cap)
            # Updates
            va_i = self.momentum * va_i + self.lr * ((goals_i - mu_i) - self.lambda_reg * (a_i - self.rho * d_i))
            vd_i = self.momentum * vd_i + self.lr * ((mu_j - goals_j) - self.lambda_reg * (d_i - self.rho * a_i))
            va_j = self.momentum * va_j + self.lr * ((goals_j - mu_j) - self.lambda_reg * (a_j - self.rho * d_j))
            vd_j = self.momentum * vd_i + self.lr * ((mu_i - goals_i) - self.lambda_reg * (d_j - self.rho * a_j))
            a_i += va_i
            d_i += vd_i
            a_j += va_j
            d_j += vd_j
            self.ratings.loc[team_i] = [a_i, d_i]
            self.ratings.loc[team_j] = [a_j, d_j]
        return np.array(predictions)
