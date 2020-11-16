import numpy as np
import pandas as pd
import itertools
from collections import OrderedDict
from utils.evaluation import logloss, rps, brier, accuracy
from utils.data import get_data
import rating_models
from joblib import Parallel, delayed
from tqdm import tqdm
from time import time
import os
import argparse


model_param_space = {
    'Elo': OrderedDict([
        ('k', [10., 20., 40.]),
        ('c', [10.]),   # This can be constant
        ('d', [400.]),  # This can be constant too
        ('lambda_goals', [1., 1.25, 1.5, 1.75])
    ]),
    'PoissonSingleRatings': OrderedDict([
        ('penalty', ['l2']),
        ('lambda_reg', np.linspace(0., 15., 31)),
        ('weight', ['exponential_weights']),
        ('weight_params', np.linspace(0.0, 0.006, 7)),
        ('goal_cap', [-1])
    ]),
    'PoissonDoubleRatings': OrderedDict([
        ('penalty', ['l2']),
        ('lambda_reg', np.linspace(0., 15., 31)),
        ('rho', np.concatenate((np.linspace(0.0, 0.95, 20), [0.99]))),
        ('weight', ['exponential_weights']),
        ('weight_params', np.linspace(0.000, 0.006, 7)),
        ('goal_cap', [-1])
    ]),
    'IterativeMargin': OrderedDict([
        ('c', [0.0, 0.0001, 0.0002, 0.001, 0.002, 0.004, 0.01, 0.02, 0.1, 0.2]),
        ('h', np.linspace(0.2, 0.4, 5)),
        ('lr', [0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1]),
        ('lambda_reg', [0.00001, 0.00002, 0.00005, 0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.01]),
        ('goal_cap', [-1])
    ]),
    'IterativeOLR': OrderedDict([
        ('c', np.linspace(0.4, 0.7, 7)),
        ('h', np.linspace(0.2, 0.5, 7)),
        ('lr', [0.01, 0.02, 0.04, 0.06, 0.08, 0.1]),
        ('lambda_reg', [0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02])
    ]),
    'IterativePoisson': OrderedDict([
        # How big should be parameter grid for an extra parameter?
        ('c', [0.0, 0.0001, 0.0002, 0.001, 0.002, 0.004, 0.01, 0.02, 0.1, 0.2]),
        ('h', np.linspace(0.2, 0.4, 5)),
        ('lr', [0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1]),
        ('lambda_reg', [0.00001, 0.00002, 0.00005, 0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.01]),
        ('rho', np.linspace(0., 1., 21)),
        ('goal_cap', [-1])
    ]),
    'OrdinalLogisticRatings': OrderedDict([
        ('penalty', ['l1', 'l2']),
        ('lambda_reg', np.linspace(0., 15., 31)),
        ('weight', ['exponential_weights']),
        ('weight_params', np.linspace(0.0, 0.006, 7))
    ])
}


def get_parameter_grid(model_name, momentum=False, size=10, randomize=True, seed=1234321):
    """Defines parameter grid for a given model."""
    try:
        param_space = model_param_space[model_name]
    except KeyError:
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


def evaluate(model_class, params, matches, valid_index, test_index,
             seasons_train, seasons_valid, seasons_test, eval_functions):
    """Train and evaluate model for a given parameter setup."""
    model = model_class(**params)
    output = {}
    start = time()
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
    print("params_grid size: {}".format(len(params_grid)))
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
        logloss,
        rps,
        brier,
        accuracy
    ]
    return eval_functions


def get_args():
    parser = argparse.ArgumentParser(description="Parameter optimization via grid search for different models")
    parser.add_argument("--experiment", help="Experiment name - that is the directory where results are saved",
                        default='test', type=str, required=False)
    parser.add_argument("--model", help="Model to be optimized", required=True)
    parser.add_argument("--league", help="One of the leagues available", required=True)
    parser.add_argument("--stages_method", help="Method for determining matchdays for sliding window predictions",
                        default="rounds")
    parser.add_argument("--n_jobs", help="Number of cores to use to perform random search", type=int, default=1)
    parser.add_argument("--n_grid", help="Maximal limit size of the param space", type=int, default=3)
    parser.add_argument("--momentum", help="Whether to add momentum to parameter grid", action='store_true')
    parser.add_argument("--seed", help="Seed for random search", type=int, default=321)
    parser.add_argument("--test", help="Do a test run", action='store_true')
    args = parser.parse_args()
    return args


def main(experiment, model, league, stages_method, n_jobs, n_grid, momentum, seed, test):
    if momentum and not model.startswith('Iterative'):
        raise ValueError('Momentum is supported only by iterative models based on gradient descent.')
    model_class = getattr(rating_models, model)

    # Evaluation setup & data
    seasons_train, seasons_valid, seasons_test, seasons_all = train_valid_test_split(league, test)
    matches = get_data(seasons_all, stages_method=stages_method)
    eval_functions = get_eval_functions()

    # Random search
    results = parameter_search(model_class, momentum, matches, seasons_train, seasons_valid, seasons_test,
                               eval_functions, n_grid, n_jobs, seed, test)

    # Results save
    results_dir = os.path.join('results', experiment)
    os.makedirs(results_dir, exist_ok=True)
    save_file = os.path.join(results_dir, '{}_{}.csv'.format(league, model))
    results.to_csv(save_file, index=False)


if __name__ == '__main__':
    args = get_args()
    main(args.experiment, args.model, args.league, args.stages_method, args.n_jobs,
         args.n_grid, args.momentum, args.seed, args.test)
