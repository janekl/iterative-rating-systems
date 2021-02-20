import numpy as np
from scipy.stats import skellam


def goals2class(goals):
    """Helper function to convert goals to result label."""
    goal_diff = goals[0] - goals[1]
    if goal_diff > 0:
        return 0
    elif goal_diff < 0:
        return 2
    return 1


def goals2goals(goals, cap=-1):
    if cap != -1:
        goals = goals.map(lambda x: min(x, cap))
    return goals


def exponential_weights(x, a):
    return np.exp(a * x)


def generate_predictions(model, matches, seasons, label_def, label_kwargs=None):
    """Generate predictions for given seasons."""
    predictions = np.zeros((len(matches), 3))
    label_kwargs = label_kwargs or {}
    y = matches[['FTHG', 'FTAG']].apply(label_def, axis=1, **label_kwargs).values
    for season in sorted(seasons):
        previous_seasons = matches['Season'] < season
        current_season = matches['Season'].eq(season)
        stage_last = matches[current_season]['Stage'].max()
        stage_start = 0
        for stage in range(stage_start, stage_last):
            train_index = previous_seasons | (current_season & (matches['Stage'] <= stage))
            matches_train = matches.loc[train_index]
            X_tr = matches_train[['HomeTeam', 'AwayTeam']]
            y_tr = y[train_index.values]
            today = matches_train['Date'].max()
            if model.weight_fun is not None:
                days_back = (matches_train['Date'] - today).apply(lambda x: x.days).values
                w_tr = model.weight_fun(days_back, model.weight_params)
            else:
                w_tr = None
            test_index = current_season & matches["Stage"].eq(stage + 1)
            matches_test = matches.loc[test_index]
            X_ts = matches_test[['HomeTeam', 'AwayTeam']]
            model.fit(X_tr, y_tr, w_tr)
            predictions[test_index.values] = model.predict_proba(X_ts)
    return predictions


def predict_skellam_1x2(mu1, mu2):
    """Get 1x2 probabilities (home, draw, away) for Poisson goal rates (mu1, mu2) using Skellam distribution."""
    p_2 = skellam.cdf(-1, mu1=mu1, mu2=mu2)
    p_x2 = skellam.cdf(0, mu1=mu1, mu2=mu2)
    p_1 = 1.0 - p_x2
    p_x = p_x2 - p_2
    return np.column_stack((p_1, p_x, p_2))
