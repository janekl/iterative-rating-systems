N_JOBS=40
N_GRID=250

for model in Elo OrdinalLogisticRatings IterativeOLR PoissonSingleRatings PoissonDoubleRatings IterativeMargin IterativePoisson
do
  for league in D E F I SP
  do
    echo $model
    python optimize_model.py --model $model --league $league --n_jobs $N_JOBS --n_grid $N_GRID --experiment full
  done
done
