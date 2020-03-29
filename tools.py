import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


def goals2class(goals):
    """Helper function to convert goals to result label."""
    goal_diff = goals[0] - goals[1]
    if goal_diff > 0:
        return 0
    elif goal_diff < 0:
        return 2
    return 1


def goals2goals(goals, cap=np.nan):
    if cap != -1:
        goals = goals.map(lambda x: min(x, cap))
    return goals


def determine_stages(matches):
    """Determine round of play my clustering dates.

    The solution is an overkill but it works in most cases."""
    k = int(2 * len(matches) / matches['HomeTeam'].nunique())
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=50)
    stage = pd.Series(kmeans.fit_predict(matches[['Date']]))
    rename_rounds = {j: i for i, j in enumerate(stage.unique(), 1)}
    stage = stage.map(rename_rounds)
    matches.insert(1, 'Stage', stage)
    return matches


def accuracy(preds, y, average=True):
    """Accuracy. Note: if there are two or more equal probabilities np.argmax returns always the first one!"""
    scores = y == np.argmax(preds, axis=1)
    if average:
        scores = scores.mean()
    return scores


def logloss(preds, y, eps=1e-8, average=True):
    """Logloss metric."""
    preds = np.maximum(preds, eps)
    preds = preds / preds.sum(axis=1, keepdims=True)
    pred_probs = preds[np.array(range(len(preds))), y]
    scores = -np.log(pred_probs)
    if average:
        scores = scores.mean()
    return scores


def indicator(i, n=3):
    """Get base i-th vector, e.g., np.array([0, 0, 1, 0]) for i = 2 and n = 4 dimensions."""
    x = np.zeros(n)
    x[i] = 1.0
    return x


def rps(preds, y, average=True):
    """Rank probability score metric."""
    n_class = preds.shape[1]
    actual_cdf = np.array([indicator(i, n_class) for i in y]).cumsum(axis=1)
    preds_cdf = preds.cumsum(axis=1)
    scores = ((preds_cdf[:, :n_class-1] - actual_cdf[:, :n_class-1])**2).sum(axis=1)
    scores /= (n_class-1)
    if average:
        scores = scores.mean()
    return scores


def brier(preds, y, average=True):
    """Brier score metric."""
    n_class = preds.shape[1]
    actual_res = np.array([indicator(i, n_class) for i in y])
    scores = ((preds - actual_res)**2).mean(axis=1)
    if average:
        scores = scores.mean()
    return scores


def evaluate(preds, y, eval_functions=['accuracy', 'logloss', 'rps', 'brier'], eps=1e-8, average=True):
    """Compute evaluation metrics for predicitons given results."""
    return pd.Series({eval_fun: globals()[eval_fun](preds, y, average=average) for eval_fun in eval_functions})


def exponential_weights(x, a):
    return np.exp(a * x)


def generate_predictions(model, matches, seasons, label_def, label_kwargs=None):
    """Generate predictions for given seasons."""
    predictions = np.zeros((len(matches), 3))
    print('Running optimization for seasons: {}'.format(seasons))
    if label_kwargs is None:
        label_kwargs = {}
    y = matches[['FTHG', 'FTAG']].apply(label_def, axis=1, **label_kwargs).values
    for season in sorted(seasons):
        current_season = matches['Season'].eq(season)
        stage_last = matches[current_season]['Stage'].max()
        stage_start = 0
        for stage in range(stage_start, stage_last):
            # print(season, stage)
            train_index = (matches['Season'] < season) | (current_season & (matches['Stage'] <= stage))
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


def preprocess_odds(matches, htg_col="home_team_goal", atg_col="away_team_goal",
                    bookies=["B365", "BW", "IW", "LB", "PS", "WH", "SJ", "VC", "GB", "BS"]):
    """Calculate probabilities from bookmaker odds for given matches."""
    y_odds = matches[[htg_col, atg_col]].apply(goals2class, axis=1)
    y_freq = y_odds.value_counts(normalize=True, sort=False).values  # If odds are missing use freq (rare case)
    all_odds_raw = [matches[["%sH" % b, "%sD" % b, "%sA" % b]] for b in bookies]
    all_odds = []
    # Normalize
    for i, odds in enumerate(all_odds_raw):
        # Invert and normalize odds to get proper probability distribution
        odds = 1. / odds
        odds = odds.div(odds.sum(axis=1), axis=0)
        if not (odds.isnull().mean() == 1).all():
            all_odds.append(odds)
    # Average
    pred_odds = []
    for i in range(len(matches)):
        odds_tmp = np.zeros(3)
        k = 0
        for j in range(len(all_odds)):
            odds_match = all_odds[j].iloc[i]
            if odds_match.notnull().all():
                odds_tmp += odds_match.values
                k += 1
        if k == 0:
            pred_odds.append(y_freq)
        else:
            pred_odds.append(odds_tmp / k)
    pred_odds = np.array(pred_odds)
    return pred_odds, y_odds
