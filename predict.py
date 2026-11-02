"""Predict heating load for one building from its 8 feature values.

Paste the values in this order, separated by commas, spaces or tabs:
  relative_compactness surface_area wall_area roof_area overall_height
  orientation glazing_area glazing_distribution

  python predict.py 0.98 514.5 294 110.25 7 2 0 0
  python predict.py        (asks you to paste the values)

Run train.py first, because this loads results/models.joblib.
"""
import re
import sys
import warnings
from pathlib import Path

import joblib
import pandas as pd

MODELS_FILE = Path(__file__).resolve().parent / "results" / "models.joblib"


def parse_values(values, features):
    """Turn a pasted string, a list or a dict into a dict of feature values."""
    if isinstance(values, dict):
        missing = [f for f in features if f not in values]
        if missing:
            raise ValueError(f"Missing features: {', '.join(missing)}")
        return {f: float(values[f]) for f in features}
    if isinstance(values, str):
        values = [v for v in re.split(r"[,\s]+", values.strip()) if v]
    if len(values) != len(features):
        raise ValueError(
            f"Expected {len(features)} values but got {len(values)}. "
            f"Order: {', '.join(features)}"
        )
    return dict(zip(features, (float(v) for v in values)))


def check_values(row, bundle):
    """List the reasons this building is unlike the training data."""
    problems = []
    for name, allowed in bundle["categories"].items():
        if row[name] not in allowed:
            problems.append(f"{name} {row[name]:g} is not one of {[int(a) for a in allowed]}")
    for name, (low, high) in bundle["ranges"].items():
        if name not in bundle["categories"] and not low <= row[name] <= high:
            problems.append(f"{name} {row[name]:g} is outside the training range {low:g} to {high:g}")
    return problems


def predict_heating_load(values, model=None):
    """Return the predicted heating load in kWh/m².

    values: a pasted string, a list of 8 numbers, or a dict keyed by feature name.
    model: "Ridge" or "Random Forest". Defaults to the better model from training.
    """
    bundle = joblib.load(MODELS_FILE)
    row = parse_values(values, bundle["features"])
    for problem in check_values(row, bundle):
        warnings.warn(f"Prediction may be unreliable: {problem}", stacklevel=2)
    frame = pd.DataFrame([row], columns=bundle["features"])
    return float(bundle["models"][model or bundle["best_model"]].predict(frame)[0])


def main():
    warnings.formatwarning = lambda message, *args, **kwargs: f"Warning: {message}\n"
    text = " ".join(sys.argv[1:]) or input("Paste the 8 values: ")
    bundle = joblib.load(MODELS_FILE)
    try:
        result = predict_heating_load(text)
        print(f"Predicted heating load: {result:.2f} kWh/m² ({bundle['best_model']})")
        for name in bundle["models"]:
            if name != bundle["best_model"]:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    print(f"{name} says: {predict_heating_load(text, name):.2f} kWh/m²")
    except ValueError as error:
        sys.exit(f"Error: {error}")


if __name__ == "__main__":
    main()
