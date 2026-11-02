# Results

Data: UCI Energy Efficiency (ENB2012), a substitute dataset. This is not usim data.
Split: 614 train rows, 154 test rows, seed 42. Heating load is in kWh/m².

## Held-out test set

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Baseline (mean) | 9.272 | 10.238 | -0.0055 |
| Ridge | 2.060 | 2.873 | 0.9208 |
| Random Forest | 0.356 | 0.487 | 0.9977 |

## Nested 5-fold cross-validation on the training set (mean ± std)

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Baseline (mean) | 9.109 ± 0.314 | 10.045 ± 0.339 | -0.0027 ± 0.0027 |
| Ridge | 2.036 ± 0.053 | 2.824 ± 0.062 | 0.9204 ± 0.0071 |
| Random Forest | 0.345 ± 0.043 | 0.508 ± 0.073 | 0.9973 ± 0.0009 |

## Unseen building shapes (one of 12 shapes held out at a time)

| Model | Mean RMSE | Std | Worst shape RMSE |
|---|---|---|---|
| Baseline (mean) | 10.331 | 3.536 | 18.095 |
| Ridge | 3.604 | 2.939 | 9.312 |
| Random Forest | 4.289 | 4.114 | 13.062 |

## Best settings

- Ridge: {'model__alpha': 0.05623413251903491}
- Random Forest: {'max_depth': None, 'max_features': 0.75, 'min_samples_leaf': 1}

Best on test RMSE: **Random Forest**. Best on cross-validated RMSE: **Random Forest**.
