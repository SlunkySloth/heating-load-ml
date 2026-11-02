"""Predict building heating load with Ridge regression and Random Forest.

Run download_data.py first. Everything this script produces goes in results/.
"""
import json
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    LeaveOneGroupOut,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "data" / "ENB2012_data.xlsx"
RESULTS = ROOT / "results"

SEED = 42
TEST_SIZE = 0.2
TARGET = "heating_load"
COLUMNS = {
    "X1": "relative_compactness",
    "X2": "surface_area",
    "X3": "wall_area",
    "X4": "roof_area",
    "X5": "overall_height",
    "X6": "orientation",
    "X7": "glazing_area",
    "X8": "glazing_distribution",
    "Y1": TARGET,
}
# These are codes for a direction or layout, so their numeric order means nothing.
CATEGORICAL = ["orientation", "glazing_distribution"]
SCORING = {
    "mae": "neg_mean_absolute_error",
    "rmse": "neg_root_mean_squared_error",
    "r2": "r2",
}

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"
COLORS = {"Ridge": "#2a78d6", "Random Forest": "#eb6834"}


def load_data():
    # Y2 (cooling load) is dropped: it is another output of the same simulation,
    # so using it as an input would leak the answer.
    df = pd.read_excel(DATA_FILE).dropna(how="all").dropna(axis=1, how="all")
    df = df[list(COLUMNS)].rename(columns=COLUMNS)
    if df.isna().any().any():
        raise ValueError("Dataset has missing values")
    return df.drop(columns=TARGET), df[TARGET]


def build_searches(features):
    numeric = [c for c in features if c not in CATEGORICAL]
    preprocess = ColumnTransformer([
        ("scale", StandardScaler(), numeric),
        ("onehot", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])
    inner_cv = KFold(n_splits=5, shuffle=True, random_state=SEED)

    ridge = GridSearchCV(
        Pipeline([("preprocess", preprocess), ("model", Ridge())]),
        {"model__alpha": np.logspace(-3, 3, 25)},
        cv=inner_cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )
    forest = GridSearchCV(
        RandomForestRegressor(n_estimators=200, random_state=SEED, n_jobs=1),
        {
            "max_depth": [None, 10],
            "max_features": [0.33, 0.5, 0.75, 1.0],
            "min_samples_leaf": [1, 2, 4],
        },
        cv=inner_cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )
    return {"Ridge": ridge, "Random Forest": forest}


def nested_cv(model, X, y):
    # Tuning runs again inside each fold, so these scores are not flattered by it.
    outer_cv = KFold(n_splits=5, shuffle=True, random_state=SEED + 1)
    scores = cross_validate(model, X, y, cv=outer_cv, scoring=SCORING, error_score="raise")
    row = {}
    for name in SCORING:
        values = scores[f"test_{name}"] * (-1 if name != "r2" else 1)
        row[f"{name}_mean"] = values.mean()
        row[f"{name}_std"] = values.std()
    return row


def unseen_shape_cv(models, X, y):
    # A random split puts every building shape on both sides. Holding out one whole
    # shape at a time tests prediction on a building the model has never seen.
    groups = X["relative_compactness"]
    rows = []
    for name, model in models.items():
        fold_rmse = []
        for train_idx, test_idx in LeaveOneGroupOut().split(X, y, groups):
            fitted = clone(model).fit(X.iloc[train_idx], y.iloc[train_idx])
            predicted = fitted.predict(X.iloc[test_idx])
            fold_rmse.append(root_mean_squared_error(y.iloc[test_idx], predicted))
        rows.append({
            "model": name,
            "rmse_mean": np.mean(fold_rmse),
            "rmse_std": np.std(fold_rmse),
            "rmse_worst_shape": np.max(fold_rmse),
        })
    return pd.DataFrame(rows)


def test_metrics(y_true, predicted):
    return {
        "mae": mean_absolute_error(y_true, predicted),
        "rmse": root_mean_squared_error(y_true, predicted),
        "r2": r2_score(y_true, predicted),
    }


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=MUTED, length=0)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)


def plot_predictions(y_test, predictions, metrics, path):
    low = min(y_test.min(), *(p.min() for p in predictions.values())) - 2
    high = max(y_test.max(), *(p.max() for p in predictions.values())) + 2
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.4), sharex=True, sharey=True, facecolor=SURFACE)
    for ax, (name, predicted) in zip(axes, predictions.items()):
        style(ax)
        ax.plot([low, high], [low, high], color=MUTED, linewidth=1.5, linestyle="--")
        ax.scatter(y_test, predicted, s=44, color=COLORS[name], edgecolor=SURFACE, linewidth=1, alpha=0.85)
        ax.set_xlim(low, high)
        ax.set_ylim(low, high)
        ax.set_aspect("equal")
        ax.set_title(name, color=TEXT, loc="left", fontsize=13, fontweight="bold")
        ax.set_xlabel("Actual heating load (kWh/m²)", color=MUTED)
        ax.text(
            0.04, 0.95,
            f"RMSE {metrics[name]['rmse']:.2f}   R² {metrics[name]['r2']:.3f}",
            transform=ax.transAxes, color=TEXT, va="top",
        )
        ax.text(0.96, 0.04, "Dashed line = perfect prediction", transform=ax.transAxes, color=MUTED, ha="right", fontsize=9)
    axes[0].set_ylabel("Predicted heating load (kWh/m²)", color=MUTED)
    fig.suptitle("Predicted vs actual on the held-out test set", color=TEXT, x=0.06, ha="left", fontsize=15)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_importance(importance, path):
    order = importance[importance["model"] == "Random Forest"].sort_values("rmse_increase")["feature"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True, facecolor=SURFACE)
    for ax, name in zip(axes, COLORS):
        style(ax)
        ax.grid(axis="y", visible=False)
        data = importance[importance["model"] == name].set_index("feature").loc[order]
        ax.barh(
            data.index.str.replace("_", " "), data["rmse_increase"], height=0.55,
            color=COLORS[name], xerr=data["rmse_increase_std"],
            error_kw={"ecolor": MUTED, "linewidth": 1},
        )
        ax.set_title(name, color=TEXT, loc="left", fontsize=13, fontweight="bold")
        ax.set_xlabel("Increase in test RMSE when shuffled (kWh/m²)", color=MUTED)
    fig.suptitle("Which features each model relies on", color=TEXT, x=0.06, ha="left", fontsize=15)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def write_summary(cv_table, test_table, shape_table, best_params, n_train, n_test, path):
    winner = test_table.drop(index="Baseline (mean)")["rmse"].idxmin()
    cv_winner = cv_table.drop(index="Baseline (mean)")["rmse_mean"].idxmin()
    lines = [
        "# Results",
        "",
        "Data: UCI Energy Efficiency (ENB2012), a substitute dataset. This is not usim data.",
        f"Split: {n_train} train rows, {n_test} test rows, seed {SEED}. Heating load is in kWh/m².",
        "",
        "## Held-out test set",
        "",
        "| Model | MAE | RMSE | R² |",
        "|---|---|---|---|",
    ]
    for name, row in test_table.iterrows():
        lines.append(f"| {name} | {row['mae']:.3f} | {row['rmse']:.3f} | {row['r2']:.4f} |")
    lines += [
        "",
        "## Nested 5-fold cross-validation on the training set (mean ± std)",
        "",
        "| Model | MAE | RMSE | R² |",
        "|---|---|---|---|",
    ]
    for name, row in cv_table.iterrows():
        lines.append(
            f"| {name} | {row['mae_mean']:.3f} ± {row['mae_std']:.3f} "
            f"| {row['rmse_mean']:.3f} ± {row['rmse_std']:.3f} "
            f"| {row['r2_mean']:.4f} ± {row['r2_std']:.4f} |"
        )
    lines += [
        "",
        "## Unseen building shapes (one of 12 shapes held out at a time)",
        "",
        "| Model | Mean RMSE | Std | Worst shape RMSE |",
        "|---|---|---|---|",
    ]
    for _, row in shape_table.iterrows():
        lines.append(
            f"| {row['model']} | {row['rmse_mean']:.3f} | {row['rmse_std']:.3f} "
            f"| {row['rmse_worst_shape']:.3f} |"
        )
    lines += [
        "",
        "## Best settings",
        "",
        f"- Ridge: {best_params['Ridge']}",
        f"- Random Forest: {best_params['Random Forest']}",
        "",
        f"Best on test RMSE: **{winner}**. Best on cross-validated RMSE: **{cv_winner}**.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    RESULTS.mkdir(exist_ok=True)
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED)
    print(f"{len(X)} rows: {len(X_train)} train, {len(X_test)} test")

    searches = build_searches(list(X.columns))
    baseline = DummyRegressor(strategy="mean")

    print("Running nested cross-validation")
    cv_rows = {"Baseline (mean)": nested_cv(baseline, X_train, y_train)}
    for name, search in searches.items():
        cv_rows[name] = nested_cv(search, X_train, y_train)
    cv_table = pd.DataFrame(cv_rows).T

    print("Tuning on the full training set")
    best, best_params = {}, {}
    for name, search in searches.items():
        search.fit(X_train, y_train)
        best[name] = search.best_estimator_
        best_params[name] = {k: float(v) if isinstance(v, np.floating) else v for k, v in search.best_params_.items()}
    baseline.fit(X_train, y_train)

    predictions = {name: model.predict(X_test) for name, model in best.items()}
    test_table = pd.DataFrame(
        {"Baseline (mean)": test_metrics(y_test, baseline.predict(X_test))}
        | {name: test_metrics(y_test, p) for name, p in predictions.items()}
    ).T

    print("Checking unseen building shapes")
    shape_table = unseen_shape_cv({"Baseline (mean)": baseline} | best, X, y)

    print("Computing feature importance")
    importance_rows = []
    for name, model in best.items():
        result = permutation_importance(
            model, X_test, y_test, scoring="neg_root_mean_squared_error",
            n_repeats=30, random_state=SEED,
        )
        for feature, mean, std in zip(X.columns, result.importances_mean, result.importances_std):
            importance_rows.append({"model": name, "feature": feature, "rmse_increase": mean, "rmse_increase_std": std})
    importance = pd.DataFrame(importance_rows)

    test_table.to_csv(RESULTS / "test_metrics.csv", index_label="model")
    cv_table.to_csv(RESULTS / "cv_metrics.csv", index_label="model")
    shape_table.to_csv(RESULTS / "unseen_shape_metrics.csv", index=False)
    importance.to_csv(RESULTS / "feature_importance.csv", index=False)
    (RESULTS / "best_params.json").write_text(json.dumps(best_params, indent=2), encoding="utf-8")
    joblib.dump(
        {
            "models": best,
            "best_model": test_table.drop(index="Baseline (mean)")["rmse"].idxmin(),
            "features": list(X.columns),
            "ranges": {c: (float(X_train[c].min()), float(X_train[c].max())) for c in X.columns},
            "categories": {c: sorted(float(v) for v in X_train[c].unique()) for c in CATEGORICAL},
        },
        RESULTS / "models.joblib",
    )

    plot_predictions(y_test, predictions, test_table.to_dict("index"), RESULTS / "predicted_vs_actual.png")
    plot_importance(importance, RESULTS / "feature_importance.png")
    write_summary(cv_table, test_table, shape_table, best_params, len(X_train), len(X_test), RESULTS / "summary.md")
    print((RESULTS / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    # Worker processes crash with access violations on Python 3.15, so use threads.
    # Threads race on the global warning filters, which sets off a scikit-learn config
    # warning thousands of times. Filters can't stop it, so drop it when displayed.
    show_warning = warnings.showwarning

    def show_other_warnings(message, *args, **kwargs):
        if "sklearn.utils.parallel.delayed" not in str(message):
            show_warning(message, *args, **kwargs)

    warnings.showwarning = show_other_warnings
    with joblib.parallel_config(backend="threading"):
        main()
