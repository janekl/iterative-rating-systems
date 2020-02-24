import numpy as np
import pandas as pd
import itertools
from collections import OrderedDict
import tools
import rating_models
from config import DATA_PATH
from joblib import Parallel, delayed
from tqdm import tqdm
from time import time
import os
import argparse


def get_parameter_grid(model_name, momentum=False, size=10, randomize=True, seed=1234321):
    """Defines parameter grid for a given model."""
    if model_name == 'Elo':
        param_space = OrderedDict({
            'k': [10., 20., 40.],
            'c': [10.],   # This can be constant
            'd': [400.],  # This can be constant too
            'lambda_goals': [1., 1.25, 1.5, 1.75]})
    elif model_name == 'PoissonSingleRatings':
        param_space = OrderedDict([
            ('penalty', ['l1', 'l2']),
            ('lambda_reg', [2.5, 2.0, 1.75, 1.5, 1.25, 1., 0.75, 0.5, 0.25]),
            ('weight', ['exponential_weights']),
            ('weight_params', np.linspace(0.0, 0.005, 6)),
            ('goal_cap', [4, 5, 6, 7, -1])
        ])
    elif model_name == 'PoissonDoubleRatings':
        param_space = OrderedDict([
            ('penalty', ['l1', 'l2']),
            ('lambda_reg', [2.5, 2.0, 1.75, 1.5, 1.25, 1., 0.75, 0.5, 0.25]),
            ('weight', ['exponential_weights']),
            ('weight_params', np.linspace(0.0, 0.005, 6)),
            ('goal_cap', [4, 5, 6, 7, -1])
        ])
    elif model_name == 'IterativeMargin':
        param_space = OrderedDict([
            ('c', [0.0, 0.0001, 0.0002, 0.001, 0.002, 0.004, 0.01, 0.02, 0.1, 0.2]),
            ('h', [0.2, 0.25, 0.275, 0.3, 0.325, 0.35, 0.375, 0.4]),
            ('lr', [0.005, 0.0075, 0.01, 0.0125, 0.015, 0.02, 0.025, 0.03]),
            ('lambda_reg', [0.00001, 0.00002, 0.00005, 0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.01]),
            ('goal_cap', [4, 5, 6, 7, -1])
        ])
    elif model_name == 'IterativeOLR':
        param_space = OrderedDict([
            ('c', [0.45, 0.5, 0.55, 0.6, 0.65, 0.7]),
            ('h', [0.25, 0.3, 0.325, 0.35, 0.375, 0.4, 0.45]),
            ('lr', [0.01, 0.02, 0.04, 0.06, 0.07, 0.08, 0.09, 0.1, 0.12]),
            ('lambda_reg', [0.01, 0.02, 0.04, 0.05, 0.08, 0.1])
        ])
    elif model_name == 'IterativePoisson':
        # How big should be parameter grid for an extra parameter?
        param_space = OrderedDict([
            ('c', [0.0, 0.0001, 0.0002, 0.001, 0.002, 0.004, 0.01, 0.02, 0.1, 0.2]),
            ('h', [0.2, 0.25, 0.275, 0.3, 0.325, 0.35, 0.375, 0.4]),
            ('lr', [0.005, 0.0075, 0.01, 0.0125, 0.015, 0.02, 0.025, 0.03]),
            ('lambda_reg', [0.00001, 0.00002, 0.00005, 0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.01]),
            ('rho', [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
            ('goal_cap', [4, 5, 6, 7, -1])
        ])
    elif model_name == 'OrdinalLogisticRatings':
        param_space = OrderedDict([
            ('penalty', ['l1', 'l2']),
            ('lambda_reg', [2.5, 2.0, 1.75, 1.5, 1.25, 1., 0.75, 0.5, 0.25]),
            ('weight', ['exponential_weights']),
            ('weight_params', np.linspace(0.0, 0.005, 6))
        ])
    else:
        raise ValueError('Parameter space for model {} is not defined.'.format(model_name))

    values_grid = pd.DataFrame.from_records(itertools.product(*param_space.values()), columns=param_space.keys())
    if momentum:
        values_grid = add_momentum(values_grid, 'once', seed=seed)
    if randomize:
        values_grid = values_grid.sample(frac=1, random_state=seed).reset_index(drop=True)
    return values_grid.head(size)  # Limit number of experiments


def add_momentum(values_grid, how, seed=None):
    """Extend parameter grid with momentum."""
    np.random.seed(seed)
    momentum_space = np.linspace(0.05, 0.95, 19)
    if how == 'once':
        values_grid['momentum'] = np.random.choice(momentum_space, size=values_grid.shape[0])
    elif how == 'everywhere':
        grids = []
        for m in momentum_space:
            new_grid = values_grid.copy()
            new_grid['momentum'] = m
            grids.append(new_grid)
        values_grid = pd.concat(grids).reset_index(drop=True)
    else:
        raise ValueError('Unknown how = {} parameter for setting momentum.'.format(how))
    return values_grid


def get_data(league_seasons, usecols=('Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR'), include_odds=()):
    """Load data from http://www.football-data.co.uk/."""
    odds_cols = ()
    for b in include_odds:
        odds_cols += ("%sH" % b, "%sD" % b, "%sA" % b)
    matches_list = []
    for season in league_seasons:
        matches = pd.read_csv(os.path.join(DATA_PATH, season + '.csv'), usecols=usecols + odds_cols)
        matches['Date'] = pd.to_datetime(matches['Date'], dayfirst=True)
        matches = matches.sort_values('Date').reset_index(drop=True)
        matches = tools.determine_stages(matches)
        matches.insert(0, 'Season', season)
        matches_list.append(matches)
    matches = pd.concat(matches_list)
    matches.dropna(subset=usecols, inplace=True)
    matches = matches.reset_index(drop=True)
    matches.loc[:, ['FTHG', 'FTAG']] = matches[['FTHG', 'FTAG']].astype(int)
    matches['FTR'] = matches['FTR'].map({'H': 0, 'D': 1, 'A': 2})
    return matches


def evaluate(model_class, params, matches, valid_index, test_index,
             seasons_train, seasons_valid, seasons_test, eval_functions):
    """Train and evaluate model for a given parameter setup."""
    model = model_class(**params)
    output = {}
    start = time()
    print(params)
    predictions = model.fit_predict(matches, seasons_train, seasons_valid, seasons_test)
    output['train_time'] = np.round((time() - start) / 60., 4)
    # TODO: Monitor train results
    for eval_set, index in zip(('valid', 'test'), (valid_index, test_index)):
        for eval_fun in eval_functions:
            output['{}_{}'.format(eval_set, eval_fun.__name__)] = np.round(eval_fun(predictions[index],
                                                                                    matches['FTR'][index]), 4)
            output['{}_size'.format(eval_set)] = index.sum()
    output['model'] = model_class.__name__
    return {**params, **output}


def parameter_search(model_class, momentum, matches, seasons_train, seasons_valid, seasons_test,
                     eval_functions, size, n_jobs, seed, test_run):
    """Random search for optimal parameters."""
    params_grid = get_parameter_grid(model_class.__name__, momentum=momentum, size=size, seed=seed)
    valid_index = matches['Season'].isin(seasons_valid).values
    test_index = matches['Season'].isin(seasons_test).values
    # For testing: sequential vs parallel
    if test_run:
        results = []
        for i, params in tqdm(params_grid.iterrows()):
            results.append(evaluate(model_class, params.to_dict(), matches, valid_index, test_index,
                                    seasons_train, seasons_valid, seasons_test, eval_functions))
    else:
        results = Parallel(n_jobs=n_jobs)(delayed(evaluate)(model_class, params.to_dict(), matches,
                                                            valid_index, test_index,
                                                            seasons_train, seasons_valid, seasons_test,
                                                            eval_functions) for i, params in tqdm(params_grid.iterrows()))
    results = pd.DataFrame.from_records(results).sort_values('valid_logloss')
    return results


def train_valid_test_split(league, test_run):
    """Seasons for training, validation and testing for a given league."""
    seasons_all = ['{}_{:0>2}{:0>2}'.format(league, i, i + 1) for i in range(9, 19)]
    if test_run:
        seasons_train = seasons_all[:1]
        seasons_valid = seasons_all[1:2]
        seasons_test = seasons_all[2:3]
        seasons_all = seasons_train + seasons_valid + seasons_test
        # Intersection should be empty!
        # assert len(set(seasons_all)) == len(seasons_train) + len(seasons_valid) + len(seasons_test)
    else:
        seasons_train = seasons_all[:3]
        seasons_valid = seasons_all[3:6]
        seasons_test = seasons_all[6:]
    return seasons_train, seasons_valid, seasons_test, seasons_all


def get_eval_functions():
    eval_functions = [
        tools.logloss,
        tools.rps,
        tools.brier,
        tools.accuracy
    ]
    return eval_functions


def get_args():
    parser = argparse.ArgumentParser(description="Parameter optimization via grid search for different models")
    parser.add_argument("--experiment", help="Experiment name - that is the directory where results are saved",
                        default='test', type=str, required=False)
    parser.add_argument("--model", help="Model to be optimized", required=True)
    parser.add_argument("--league", help="One of the leagues available", required=True)
    parser.add_argument("--n_jobs", help="Number of cores to use to perform random search", type=int, default=1)
    parser.add_argument("--n_grid", help="Maximal limit size of the param space", type=int, default=3)
    parser.add_argument("--momentum", help="Whether to add momentum to parameter grid", action='store_true')
    parser.add_argument("--seed", help="Seed for random search", type=int, default=321)
    parser.add_argument("--test", help="Do a test run", action='store_true')
    args = parser.parse_args()
    results_dir = os.path.join('results', args.experiment)
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)
    return args


def main():
    args = get_args()
    if args.momentum and not args.model.startswith('Iterative'):
        raise ValueError('Momentum is supported only by iterative models based on gradient descent.')
    model_class = getattr(rating_models, args.model)
    # print(vars(args))

    # Evaluation setup & data
    seasons_train, seasons_valid, seasons_test, seasons_all = train_valid_test_split(args.league, args.test)
    matches = get_data(seasons_all)
    eval_functions = get_eval_functions()

    # Random search
    results = parameter_search(model_class, args.momentum, matches,
                               seasons_train, seasons_valid, seasons_test, eval_functions,
                               args.n_grid, args.n_jobs, args.seed, args.test)

    # Results save
    save_file = os.path.join('results', args.experiment, '{}_{}.csv'.format(args.league, args.model))
    results.to_csv(save_file, index=False)


if __name__ == '__main__':
    main()
