import os
import pandas as pd
from optimize_model import main
from config import DATA_PATH


def is_data_available(league):
    # TODO: upload some test data
    files = ['{}_{:0>2}{:0>2}.csv'.format(league, i, i + 1) for i in range(16, 19)]
    return all(os.path.isfile(os.path.join(DATA_PATH, file)) for file in files)


def test_optimize_model():
    league = "SP"
    experiment = "_pytest"
    assert is_data_available(league), "Please download some data first!"
    for model in ("IterativeMargin", "OrdinalLogisticRatings"):
        main(experiment=experiment, model=model, league=league, stages_method="rounds",
             n_jobs=2, n_grid=5, momentum=False, seed=0, test=True)
        result_file = os.path.join("results", experiment, "{}_{}.csv".format(league, model))
        assert os.path.isfile(result_file)
        results = pd.read_csv(result_file)
        assert len(results) == 5
