import argparse
import itertools
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

QUADRANTS = ["Q1", "Q2", "Q3", "Q4"]
PARTS = ["train", "validate", "test"]
SEED = 42
PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURE_DIR = PROJECT_ROOT / "data" / "features"
RESULTS_DIR = PROJECT_ROOT / "results" / "classic"
MODELS_DIR = PROJECT_ROOT / "models"
NON_FEATURE_COLUMNS = ["Song", "Quadrant", "duration_loaded"]

PROTOCOLS = {
    "A": ("tvt_70_15_15", PROJECT_ROOT / "data" / "raw" / "MERGE_Audio_Balanced" / "tvt_dataframes" / "tvt_70_15_15"),
    "B": ("artist_70_15_15", PROJECT_ROOT / "results" / "splits" / "artist_70_15_15"),
}

FEATURE_SETS = {
    "base": ["features_librosa.csv"],
    "tonal": ["features_librosa.csv", "features_tonal.csv"],
}

GRIDS = {
    "svm": {
        "C": [0.1, 1, 10, 100],
        "gamma": ["scale", 0.001, 0.01, 0.1],
    },
    "rf": {
        "n_estimators": [500],
        "max_depth": [None, 20],
        "max_features": ["sqrt", 0.3],
        "min_samples_leaf": [1, 2, 4],
    },
}


def build_model(name, params):
    if name == "svm":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="rbf", C=params["C"], gamma=params["gamma"], random_state=SEED)),
        ])
    return RandomForestClassifier(random_state=SEED, n_jobs=-1, **params)


def load_features(feature_set):
    frames = []
    for name in FEATURE_SETS[feature_set]:
        df = pd.read_csv(FEATURE_DIR / name)
        df["Song"] = df["Song"].astype(str)
        frames.append(df.drop(columns=[c for c in ["Quadrant", "duration_loaded"] if c in df.columns]).set_index("Song"))
    labels = pd.read_csv(FEATURE_DIR / FEATURE_SETS[feature_set][0])
    labels["Song"] = labels["Song"].astype(str)
    data = pd.concat(frames, axis=1, join="inner").copy()
    data["Quadrant"] = labels.set_index("Song")["Quadrant"]
    feature_cols = [c for c in data.columns if c not in NON_FEATURE_COLUMNS]
    return data, feature_cols


def load_split(protocol):
    split_name, split_dir = PROTOCOLS[protocol]
    return split_name, {p: pd.read_csv(split_dir / f"{split_name}_{p}_audio_balanced.csv")["Song"].astype(str) for p in PARTS}


def macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average="macro", labels=QUADRANTS)


def save_confusion(y_true, y_pred, title, path):
    cm = confusion_matrix(y_true, y_pred, labels=QUADRANTS)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay(cm, display_labels=QUADRANTS).plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return cm


def run(name, protocol, feature_set, stamp):
    data, feature_cols = load_features(feature_set)
    split_name, parts = load_split(protocol)
    missing = {p: len(set(ids) - set(data.index)) for p, ids in parts.items()}
    if any(missing.values()):
        raise SystemExit(f"Songs in split but missing from features: {missing}")

    X = {p: data.loc[ids, feature_cols].to_numpy() for p, ids in parts.items()}
    y = {p: data.loc[ids, "Quadrant"].to_numpy() for p, ids in parts.items()}
    tag = f"{name}_{protocol}_{feature_set}"
    print(f"\n=== {tag}: {len(feature_cols)} features | train {len(y['train'])}, validate {len(y['validate'])}, test {len(y['test'])}")

    grid = GRIDS[name]
    keys = list(grid)
    combos = list(itertools.product(*(grid[k] for k in keys)))
    search = []
    start = time.time()
    for values in combos:
        params = dict(zip(keys, values))
        model = build_model(name, params).fit(X["train"], y["train"])
        score = macro_f1(y["validate"], model.predict(X["validate"]))
        search.append({**{k: str(v) for k, v in params.items()}, "val_macro_f1": round(score, 4)})
        print(f"  {params} -> validation macro-F1 {score:.4f}")
    best_params = dict(zip(keys, combos[int(np.argmax([s["val_macro_f1"] for s in search]))]))

    final = build_model(name, best_params).fit(X["train"], y["train"])
    pred = {p: final.predict(X[p]) for p in PARTS}
    elapsed = time.time() - start

    out = RESULTS_DIR / f"{stamp}_{tag}"
    out.mkdir(parents=True, exist_ok=True)
    cm = save_confusion(y["test"], pred["test"], f"{name.upper()} | protocol {protocol} | {feature_set} features | test", out / "confusion_matrix_test.png")
    pd.DataFrame(cm, index=[f"true_{q}" for q in QUADRANTS], columns=[f"pred_{q}" for q in QUADRANTS]).to_csv(out / "confusion_matrix_test.csv")
    pd.DataFrame(search).to_csv(out / "validation_search.csv", index=False)
    pd.DataFrame({"Song": parts["test"].values, "true": y["test"], "pred": pred["test"]}).to_csv(out / "test_predictions.csv", index=False)
    if name == "rf":
        importances = pd.Series(final.feature_importances_, index=feature_cols).sort_values(ascending=False)
        importances.rename("importance").to_csv(out / "feature_importances.csv")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / f"{tag}.joblib"
    joblib.dump({"model": final, "feature_columns": feature_cols, "classes": QUADRANTS, "trained": stamp, "params": best_params}, model_path)

    report = classification_report(y["test"], pred["test"], labels=QUADRANTS, output_dict=True, zero_division=0)
    q3q4 = cm[2, 3] + cm[3, 2]
    metrics = {
        "model": name,
        "protocol": protocol,
        "split": split_name,
        "feature_set": feature_set,
        "feature_files": FEATURE_SETS[feature_set],
        "n_features": len(feature_cols),
        "evaluation": "hyperparameters selected on validation; final model trained on train only; test evaluated once",
        "best_params": {k: str(v) for k, v in best_params.items()},
        "n_samples": {p: int(len(y[p])) for p in PARTS},
        "train_macro_f1": round(macro_f1(y["train"], pred["train"]), 4),
        "validation_macro_f1": round(macro_f1(y["validate"], pred["validate"]), 4),
        "test_macro_f1": round(macro_f1(y["test"], pred["test"]), 4),
        "test_accuracy": round(accuracy_score(y["test"], pred["test"]), 4),
        "test_per_class": {q: {k: round(report[q][k], 4) for k in ["precision", "recall", "f1-score"]} for q in QUADRANTS},
        "test_q3_q4_confusions": int(q3q4),
        "saved_model": str(model_path.relative_to(PROJECT_ROOT)),
        "seed": SEED,
        "elapsed_seconds": round(elapsed, 1),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "sklearn": sklearn.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"  best {best_params}")
    print(f"  macro-F1 train {metrics['train_macro_f1']:.4f} | validation {metrics['validation_macro_f1']:.4f} | test {metrics['test_macro_f1']:.4f} | Q3<->Q4 confusions {q3q4}")
    return {k: metrics[k] for k in ["model", "protocol", "feature_set", "n_features", "train_macro_f1", "validation_macro_f1", "test_macro_f1", "test_accuracy", "test_q3_q4_confusions"]} | {"results": out.name}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["svm", "rf", "all"], default="all")
    parser.add_argument("--protocol", choices=["A", "B", "all"], default="all")
    parser.add_argument("--features", choices=["base", "tonal", "all"], default="all")
    args = parser.parse_args()

    models = ["svm", "rf"] if args.model == "all" else [args.model]
    protocols = ["A", "B"] if args.protocol == "all" else [args.protocol]
    feature_sets = ["base", "tonal"] if args.features == "all" else [args.features]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    rows = [run(m, p, f, stamp) for p in protocols for f in feature_sets for m in models]
    summary = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(RESULTS_DIR / f"{stamp}_summary.csv", index=False)
    print("\n" + summary.drop(columns="results").to_string(index=False))
    print(f"\nSummary saved to {RESULTS_DIR / f'{stamp}_summary.csv'}")


if __name__ == "__main__":
    main()
