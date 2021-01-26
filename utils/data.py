import itertools
import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from config import DATA_PATH
from utils.model import goals2class


def get_data(league_seasons, usecols=('Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR'), include_odds=(),
             stages_method="rounds"):
    """Load data from http://www.football-data.co.uk/."""
    odds_cols = ()
    for b in include_odds:
        odds_cols += ("%sH" % b, "%sD" % b, "%sA" % b)
    matches_list = []
    for season in league_seasons:
        matches = pd.read_csv(os.path.join(DATA_PATH, season + '.csv'), usecols=usecols + odds_cols)
        matches.dropna(subset=usecols, inplace=True)
        matches['Date'] = pd.to_datetime(matches['Date'], dayfirst=True)
        matches = matches.sort_values('Date').reset_index(drop=True)
        matches = determine_stages(matches, stages_method)
        matches.insert(0, 'Season', season)
        matches_list.append(matches)
    matches = pd.concat(matches_list)
    matches = matches.reset_index(drop=True)
    matches.loc[:, ['FTHG', 'FTAG']] = matches[['FTHG', 'FTAG']].astype(int)
    matches['FTR'] = matches['FTR'].map({'H': 0, 'D': 1, 'A': 2})
    return matches


def determine_stages(matches, method="rounds"):
    """Determine round of play by days or by clustering dates to get rounds.

    The solution for rounds is an overkill but it works in most cases."""
    if method == "rounds":
        k = int(2 * len(matches) / matches['HomeTeam'].nunique())
        kmeans = KMeans(n_clusters=k, init='k-means++', n_init=50)
        stage = pd.Series(kmeans.fit_predict(matches[['Date']]))
        rename_rounds = {j: i for i, j in enumerate(stage.unique(), 1)}
        stage = stage.map(rename_rounds)
    elif method == "days":
        assert matches["Date"].is_monotonic_increasing
        stage = []
        for i, (_, days) in enumerate(itertools.groupby(matches["Date"]), 1):
            stage.extend(i for _ in days)
    else:
        raise ValueError("Unknown method = '{}' for determining stages.".format(method))
    matches.insert(1, 'Stage', stage)
    return matches


def preprocess_odds(matches, htg_col="home_team_goal", atg_col="away_team_goal",
                    bookies=("B365", "BW", "IW", "LB", "PS", "WH", "SJ", "VC", "GB", "BS")):
    """Calculate probabilities from bookmaker odds for given matches."""
    y_odds = matches[[htg_col, atg_col]].apply(goals2class, axis=1)
    y_freq = y_odds.value_counts(normalize=True, sort=False).values  # If odds are missing use freq (rare case)
    all_odds_raw = [matches[["%sH" % b, "%sD" % b, "%sA" % b]] for b in bookies]
    all_odds = []
    # Normalize
    for i, odds in enumerate(all_odds_raw):
        if not odds.isnull().all(axis=None):
            # Invert and normalize odds to get proper probability distribution
            odds = 1. / odds
            odds = odds.div(odds.sum(axis=1), axis=0)
            all_odds.append(odds)
    # Average for individual matches
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
