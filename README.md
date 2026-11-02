# Heating load prediction

Predicts a building's heating load (kWh/m²) with Ridge regression and Random Forest, compared against a simple baseline that always guesses the average.

**Data:** the [UCI Energy Efficiency dataset](https://archive.ics.uci.edu/dataset/242/energy+efficiency) (Tsanas & Xifara, 2012): 768 simulated buildings. This is a substitute dataset. It is not "usim" data.

## Run it

Written for Windows PowerShell and tested on Python 3.15.2.

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python download_data.py
.\.venv\Scripts\python train.py
```

`train.py` takes about 4 to 7 minutes. It writes everything to `results/`.

## Predict one building

Paste the 8 values in this order, separated by spaces, commas or tabs (so a row copied from Excel works):

`relative_compactness surface_area wall_area roof_area overall_height orientation glazing_area glazing_distribution`

```powershell
.\.venv\Scripts\python predict.py 0.98 514.5 294 110.25 7 2 0 0
```

Run `predict.py` with no values and it asks you to paste them. From your own code:

```python
from predict import predict_heating_load

predict_heating_load("0.98, 514.5, 294, 110.25, 7, 2, 0, 0")
predict_heating_load([0.98, 514.5, 294, 110.25, 7, 2, 0, 0], model="Ridge")
```

It gives a warning if the values are outside what the models were trained on. Treat those predictions with caution.

## Method

- **Inputs:** the 8 building features. The cooling load column is dropped, because it comes from the same simulation and would leak the answer.
- **Split:** 80% train (614 rows), 20% held-out test (154 rows), seed 42.
- **Ridge:** scaling for numeric features, one-hot encoding for orientation and glazing distribution, alpha tuned with 5-fold cross-validation.
- **Random Forest:** 200 trees, with depth, features per split and leaf size tuned with 5-fold cross-validation.
- **Cross-validation scores:** nested, so tuning is repeated inside each fold and doesn't flatter the score.
- **Unseen shape check:** a random split puts all 12 building shapes on both sides, so each shape is also held out in turn. Settings are reused from the tuning above.
- **Feature importance:** how much test RMSE worsens when a feature is shuffled. Both models are measured the same way.

## Results

| Held-out test set | MAE | RMSE | R² |
|---|---|---|---|
| Baseline (mean) | 9.272 | 10.238 | -0.0055 |
| Ridge (alpha 0.056) | 2.060 | 2.873 | 0.9208 |
| Random Forest | 0.356 | 0.487 | 0.9977 |

| Nested 5-fold CV | MAE | RMSE | R² |
|---|---|---|---|
| Baseline (mean) | 9.109 ± 0.314 | 10.045 ± 0.339 | -0.0027 ± 0.0027 |
| Ridge | 2.036 ± 0.053 | 2.824 ± 0.062 | 0.9204 ± 0.0071 |
| Random Forest | 0.345 ± 0.043 | 0.508 ± 0.073 | 0.9973 ± 0.0009 |

| Unseen building shapes | Mean RMSE | Worst shape RMSE |
|---|---|---|
| Baseline (mean) | 10.331 | 18.095 |
| Ridge | 3.604 | 9.312 |
| Random Forest | 4.289 | 13.062 |

### Which model did better, and why

**Random Forest is better for buildings like the ones in this data.** Its test RMSE is about 6 times lower than Ridge's, and cross-validation agrees. Heating load isn't a straight-line sum of the features: the effect of one feature depends on others. Trees capture that, while Ridge can only give each feature one fixed effect. That shows in the plot, where Ridge's points spread away from the line in bands at higher loads.

**Ridge is better on building shapes it hasn't seen,** by a small margin (3.60 against 4.29). A forest can only predict within what it has seen, while a line can extend the trend. Both have a much higher error here, and the spread between shapes is large, so neither should be trusted far outside this data.

**Features:** relative compactness matters a lot to both models. Ridge also leans heavily on overall height, while the forest spreads importance more evenly and gives glazing area more weight. Orientation does almost nothing in either model. The size features overlap with each other, so credit is shared between them and individual rankings are only rough.

## Python 3.15 crash note

On Python 3.15.2 with scikit-learn 1.9.1, worker processes sometimes crashed with an access violation, which turned Random Forest's cross-validation scores into `nan`. So `train.py` uses threads, not worker processes, and stops with an error if any fold fails. Repeated runs with threads gave identical metric files and no crashes. Threads make it slower: 4 to 7 minutes, up from about 2.
