import os
import pandas as pd
from optimize_model import main


def test_optimize_model():
    league = "TEST"
    experiment = "_pytest"
    for model in ("IterativeMargin", "OrdinalLogisticRatings", "PoissonDoubleRatings"):
        main(experiment=experiment, model=model, league="TEST", stages_method="rounds",
             n_jobs=2, n_grid=5, momentum=False, seed=0, test=True)
        result_file = os.path.join("results", experiment, "{}_{}.csv".format(league, model))
        assert os.path.isfile(result_file)
        results = pd.read_csv(result_file)
        assert len(results) == 5
