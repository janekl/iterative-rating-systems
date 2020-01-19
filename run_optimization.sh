n_jobs=40
n_grid=250

for model in Elo OrdinalLogisticRatings IterativeOLR PoissonSingleRatings PoissonDoubleRatings IterativeMargin IterativePoisson
do
  for league in D E F I SP
  do
    echo $model
    python optimize_model.py --model $model --league $league --n_jobs $n_jobs --n_grid $n_grid --experiment full
  done
done
