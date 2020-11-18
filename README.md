# Iterative Rating Systems

This repository provides an implementation of several new iterative models for rating of sport teams, illustrated by the example of association football (soccer). The presented models are based upon the gradient descent algorithm, which makes the rating update equations easy to interpret, explain, and adjust once new match results are observed. A prominent instance of a rating model that fits into this framework is the Elo rating system.

The iterative versions of the popular football match outcome models are not only easier to maintain but also yield more accurate predictions than their counterparts estimated jointly using a sample of matches.

The figure below concerns the *big six* Premier League teams for three selected seasons. The ratings are obtained through the iterative version of the one-parameter Poisson model which proved to be the most accurate approach in the comparison.

![big six](https://github.com/janekl/iterative-rating-systems/blob/master/misc/big_six_ratings.png?raw=true)

The repository is an integral part the paper by Jan Lasek and Marek Gągolewski *"Interpretable sport team rating models based on the gradient descent algorithm"* (under review).

## Setup

### Environment

The project was developed in Python 3.5+. Package dependencies can be installed with `pip` (in a virtual environment):

```bash
$ pip install -r requirements.txt
```

Perhaps the easiest way to get everyting up and running is to add the project directory to `PYTHONPATH` variable by appending the following line to `~/.bashrc`:

```
export PYTHONPATH="${PYTHONPATH}:/your/path/to/project/dir"
```

Directory `tests` provides some testing routines that can be run using [pytest](https://docs.pytest.org/en/stable/):
```bash
$ ./run_tests.sh
```

### Data

Data for running the computations are available on-line at http://www.football-data.co.uk/. It can be conveniently downloaded using the script located in the `data` directory:

```bash
$ ./download_data.sh
```

In order to inform the other scripts where the data reside, set the  `DATA_PATH` variable in `config.py` .

### Running Experiments

Parameter optimisation was performed by random search using `optimize_model.py`. Shell script `run_optimization.sh` includes all the optimisations performed:

```bash
$ ./run_optimization.sh
```

The results will be saved in the current working directory.

Notebook `demo.ipynb` gives a short demonstration of one of the models proposed.
