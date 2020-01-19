# Iterative rating systems

This project contains implementation of several new iterative rating models for rating sport teams, illustrated by the example of association football (soccer). The presented models are based upon the gradient descent heuristic, which makes the rating update equations easy to interpret and explain as well as to adjust once new match results are observed. An example of a rating models that fits into this framework is the prominent Elo rating system.

The iterative versions of the popular football match outcome models are not only easier to maintain but also yield more accurate predictions than their counterparts estimated jointly using a sample of matches.

Figure below presents for *big six* Premier League teams for three selected seasons. The ratings are obtained from the iterative version of one-parameter Poisson model which proved to be the most accurate modelling approach in the comparison.

![big six](https://github.com/janekl/iterative-rating-systems/blob/master/big_six_ratings.png?raw=true)

The repository is an integral of the paper by Jan Lasek and Marek Gągolewski *"Interpretable sport team rating models based on the gradient descent algorithm"* (under review).

## Setup

### Environment

The project was developed in Python 3.5.6. Package dependencies can be installed with `pip`:
```
$ pip install -r requirements.txt
```

### Data

Data for running computations are available on line at http://www.football-data.co.uk/. It can be conveniently downloaded using the script given in `data` directory:
```
$ ./download_data.sh
```
In order to inform other scripts where the data reside edit `config.py` script and overwrite `DATA_PATH` variable.

### Running experiments

Parameter optimisation was performed by random search using `optimize_model.py` script. Shell script `run_optimization.sh` includes all optimisations performed:
```
$ ./run_optimization.sh
```
The results will be saved to a local directory (as specified by the script arguments).

Notebook `demo.ipynb` is a short presentation of one of the models proposed.
