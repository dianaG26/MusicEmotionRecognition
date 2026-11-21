import argparse
import itertools
import json
import pickle
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from train_classic import FEATURE_SETS, GRIDS, PROJECT_ROOT, QUADRANTS, SEED, build_model, load_features

PARTS = ["train", "validate", "test"]
OUT_DIR = PROJECT_ROOT / "results" / "experiments"
OFFICIAL_DIR = PROJECT_ROOT / "data" / "raw" / "MERGE_Audio_Balanced" / "tvt_dataframes"
ARTIST_DIR = PROJECT_ROOT / "results" / "splits"
ARTIST_SEEDS = [42, 43, 44, 45, 46]

GRIDS = {
    **GRIDS,
    "logreg": {"C": [0.01, 0.1, 1, 10]},
    "dummy": {"strategy": ["stratified"]},
}

FEATURE_GROUPS = {
    "mfcc": lambda c: c.startswith("mfcc_") and not c.startswith("mfcc_delta"),
    "mfcc_delta": lambda c: c.startswith("mfcc_delta"),
    "spectral_shape": lambda c: c.split("_")[1] in {"centroid", "bandwidth", "rolloff", "flatness"} if c.startswith("spectral_") else False,
    "spectral_contrast": lambda c: c.startswith("spectral_contrast"),
    "energy_noise": lambda c: c.startswith("rms") or c.startswith("zcr"),
    "chroma": lambda c: c.startswith("chroma"),
    "rhythm": lambda c: c.startswith("onset") or c == "tempo",
    "mode_key": lambda c: c.startswith("modality") or c.startswith("key_"),
    "tonnetz_harmonic_change": lambda c: c.startswith("tonnetz") or c.startswith("harmonic_change"),
}


def splits():
    out = {"A": ("tvt_70_15_15", OFFICIAL_DIR / "tvt_70_15_15")}
    for seed in ARTIST_SEEDS:
        name = "artist_70_15_15" if seed == SEED else f"artist_70_15_15_seed{seed}"
        out[f"B{seed}"] = (name, ARTIST_DIR / name)
    return out


def load_split(name, folder):
    return {p: pd.read_csv(folder / f"{name}_{p}_audio_balanced.csv")["Song"].astype(str) for p in PARTS}


def make(name, params):
    if name == "logreg":
        return Pipeline([("scaler", StandardScaler()), ("logreg", LogisticRegression(C=params["C"], max_iter=5000, random_state=SEED))])
    if name == "dummy":
        return DummyClassifier(strategy=params["strategy"], random_state=SEED)
    return build_model(name, params)


def macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average="macro", labels=QUADRANTS)


def select(name, X, y):
    keys = list(GRIDS[name])
    rows, best, best_score = [], None, -1.0
    for values in itertools.product(*(GRIDS[name][k] for k in keys)):
        params = dict(zip(keys, values))
        score = macro_f1(y["validate"], make(name, params).fit(X["train"], y["train"]).predict(X["validate"]))
        rows.append({**{k: str(v) for k, v in params.items()}, "val_macro_f1": score})
        if score > best_score:
            best, best_score = params, score
    return best, best_score, rows


def scores(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)
    return model.decision_function(X)


def majority_vote(preds):
    stacked = np.stack(preds, axis=1)
    out = []
    for row in stacked:
        values, counts = np.unique(row, return_counts=True)
        out.append(row[0] if counts.max() == 1 else values[counts.argmax()])
    return np.array(out)


def main_runs(data_cache, split_defs, out):
    runs, preds, searches, roc = [], [], [], []
    for (protocol, (split_name, folder)), feature_set in itertools.product(split_defs.items(), FEATURE_SETS):
        data, cols = data_cache[feature_set]
        ids = load_split(split_name, folder)
        X = {p: data.loc[ids[p], cols].to_numpy() for p in PARTS}
        y = {p: data.loc[ids[p], "Quadrant"].to_numpy() for p in PARTS}
        test_preds = {}
        for name in ["dummy", "logreg", "svm", "rf"]:
            start = time.time()
            params, val_score, rows = select(name, X, y)
            search_time = time.time() - start
            t0 = time.time()
            model = make(name, params).fit(X["train"], y["train"])
            fit_time = time.time() - t0
            pred_test = model.predict(X["test"])
            test_preds[name] = pred_test
            cm = confusion_matrix(y["test"], pred_test, labels=QUADRANTS)
            runs.append({
                "protocol": protocol, "split": split_name, "feature_set": feature_set, "n_features": len(cols), "model": name,
                "best_params": json.dumps({k: str(v) for k, v in params.items()}),
                "train_macro_f1": macro_f1(y["train"], model.predict(X["train"])),
                "val_macro_f1": val_score, "test_macro_f1": macro_f1(y["test"], pred_test),
                "test_accuracy": float((pred_test == y["test"]).mean()),
                **{f"test_f1_{q}": f for q, f in zip(QUADRANTS, f1_score(y["test"], pred_test, average=None, labels=QUADRANTS))},
                "test_q3_as_q4": int(cm[2, 3]), "test_q4_as_q3": int(cm[3, 2]),
                "fit_seconds": fit_time, "search_seconds": search_time, "model_bytes": len(pickle.dumps(model)),
            })
            searches += [{"protocol": protocol, "feature_set": feature_set, "model": name, **r} for r in rows]
            preds.append(pd.DataFrame({"protocol": protocol, "feature_set": feature_set, "model": name, "Song": ids["test"].values, "true": y["test"], "pred": pred_test}))
            if protocol == "A" and name in {"logreg", "svm", "rf"}:
                s = scores(model, X["test"])
                roc.append(pd.DataFrame(s, columns=[f"score_{q}" for q in QUADRANTS]).assign(protocol=protocol, feature_set=feature_set, model=name, Song=ids["test"].values, true=y["test"]))
            print(f"{protocol:4s} {feature_set:5s} {name:6s} val {val_score:.4f} test {runs[-1]['test_macro_f1']:.4f} ({search_time:.0f}s)", flush=True)
        vote = majority_vote([test_preds["svm"], test_preds["rf"], test_preds["logreg"]])
        val_vote = None
        cm = confusion_matrix(y["test"], vote, labels=QUADRANTS)
        runs.append({
            "protocol": protocol, "split": split_name, "feature_set": feature_set, "n_features": len(cols), "model": "vote",
            "best_params": json.dumps({"members": "svm, rf, logreg", "tie": "svm"}),
            "val_macro_f1": val_vote, "test_macro_f1": macro_f1(y["test"], vote), "test_accuracy": float((vote == y["test"]).mean()),
            **{f"test_f1_{q}": f for q, f in zip(QUADRANTS, f1_score(y["test"], vote, average=None, labels=QUADRANTS))},
            "test_q3_as_q4": int(cm[2, 3]), "test_q4_as_q3": int(cm[3, 2]),
        })
        preds.append(pd.DataFrame({"protocol": protocol, "feature_set": feature_set, "model": "vote", "Song": ids["test"].values, "true": y["test"], "pred": vote}))
    pd.DataFrame(runs).to_csv(out / "runs.csv", index=False)
    pd.concat(preds).to_csv(out / "test_predictions.csv", index=False)
    pd.DataFrame(searches).to_csv(out / "validation_search.csv", index=False)
    pd.concat(roc).to_csv(out / "roc_scores_protocol_A.csv", index=False)
    return pd.DataFrame(runs)


def ensemble_validation(data_cache, split_defs, runs, out):
    rows = []
    for (protocol, (split_name, folder)), feature_set in itertools.product(split_defs.items(), FEATURE_SETS):
        data, cols = data_cache[feature_set]
        ids = load_split(split_name, folder)
        X = {p: data.loc[ids[p], cols].to_numpy() for p in ["train", "validate"]}
        y = {p: data.loc[ids[p], "Quadrant"].to_numpy() for p in ["train", "validate"]}
        sel = runs[(runs.protocol == protocol) & (runs.feature_set == feature_set)].set_index("model")
        members = []
        for name in ["svm", "rf", "logreg"]:
            params = {k: (None if v == "None" else type_cast(v)) for k, v in json.loads(sel.loc[name, "best_params"]).items()}
            members.append(make(name, params).fit(X["train"], y["train"]).predict(X["validate"]))
        rows.append({"protocol": protocol, "feature_set": feature_set, "val_macro_f1_vote": macro_f1(y["validate"], majority_vote(members)),
                     "val_macro_f1_best_single": sel.loc[["svm", "rf", "logreg"], "val_macro_f1"].max()})
    pd.DataFrame(rows).to_csv(out / "ensemble_validation.csv", index=False)


def type_cast(v):
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    return v


def best_params_for(runs, protocol, feature_set, model):
    raw = json.loads(runs[(runs.protocol == protocol) & (runs.feature_set == feature_set) & (runs.model == model)].iloc[0]["best_params"])
    return {k: (None if v == "None" else type_cast(v)) for k, v in raw.items()}


def ablation(data_cache, split_defs, runs, out):
    data, cols = data_cache["tonal"]
    rows = []
    for protocol in ["A", "B42"]:
        split_name, folder = split_defs[protocol]
        ids = load_split(split_name, folder)
        y = {p: data.loc[ids[p], "Quadrant"].to_numpy() for p in ["train", "validate"]}
        for name in ["svm", "rf"]:
            params = best_params_for(runs, protocol, "tonal", name)
            def fit_score(use):
                Xtr, Xva = data.loc[ids["train"], use].to_numpy(), data.loc[ids["validate"], use].to_numpy()
                pred = make(name, params).fit(Xtr, y["train"]).predict(Xva)
                cm = confusion_matrix(y["validate"], pred, labels=QUADRANTS)
                return macro_f1(y["validate"], pred), int(cm[2, 3] + cm[3, 2])
            full, full_q34 = fit_score(cols)
            rows.append({"protocol": protocol, "model": name, "removed_group": "none", "n_removed": 0, "val_macro_f1": full, "val_q3_q4_confusions": full_q34})
            for group, rule in FEATURE_GROUPS.items():
                drop = [c for c in cols if rule(c)]
                score, q34 = fit_score([c for c in cols if c not in drop])
                rows.append({"protocol": protocol, "model": name, "removed_group": group, "n_removed": len(drop), "val_macro_f1": score, "val_q3_q4_confusions": q34})
            print(f"ablation {protocol} {name} done", flush=True)
    pd.DataFrame(rows).to_csv(out / "ablation_validation.csv", index=False)


def learning_curve(data_cache, split_defs, runs, out, repeats=5):
    data, cols = data_cache["tonal"]
    split_name, folder = split_defs["A"]
    ids = load_split(split_name, folder)
    train = data.loc[ids["train"]]
    Xva, yva = data.loc[ids["validate"], cols].to_numpy(), data.loc[ids["validate"], "Quadrant"].to_numpy()
    rng = np.random.default_rng(SEED)
    rows = []
    for name in ["logreg", "svm", "rf"]:
        params = best_params_for(runs, "A", "tonal", name)
        for frac in [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]:
            for r in range(repeats if frac < 1.0 else 1):
                idx = np.concatenate([rng.choice(np.flatnonzero(train["Quadrant"].to_numpy() == q), int(round(frac * (train["Quadrant"] == q).sum())), replace=False) for q in QUADRANTS])
                sub = train.iloc[idx]
                model = make(name, params).fit(sub[cols].to_numpy(), sub["Quadrant"].to_numpy())
                rows.append({"model": name, "train_fraction": frac, "n_train": len(sub), "repeat": r,
                             "train_macro_f1": macro_f1(sub["Quadrant"].to_numpy(), model.predict(sub[cols].to_numpy())),
                             "val_macro_f1": macro_f1(yva, model.predict(Xva))})
        print(f"learning curve {name} done", flush=True)
    pd.DataFrame(rows).to_csv(out / "learning_curve_validation.csv", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-extras", action="store_true")
    args = parser.parse_args()
    out = OUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    start = time.time()
    split_defs = splits()
    missing = [k for k, (n, f) in split_defs.items() if not (f / f"{n}_train_audio_balanced.csv").exists()]
    if missing:
        raise SystemExit(f"Missing splits {missing}: run python src\\splits.py --seed 42 43 44 45 46 first")
    data_cache = {fs: load_features(fs) for fs in FEATURE_SETS}
    runs = main_runs(data_cache, split_defs, out)
    if not args.skip_extras:
        ensemble_validation(data_cache, split_defs, runs, out)
        ablation(data_cache, split_defs, runs, out)
        learning_curve(data_cache, split_defs, runs, out)
    info = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "evaluation": "every choice made on the validation set; test set evaluated once per configuration",
        "protocols": {k: v[0] for k, v in split_defs.items()},
        "models": ["dummy", "logreg", "svm", "rf", "vote (majority of svm, rf, logreg)"],
        "grids": {k: {kk: [str(x) for x in vv] for kk, vv in v.items()} for k, v in GRIDS.items()},
        "feature_groups": list(FEATURE_GROUPS),
        "seed": SEED,
        "elapsed_minutes": round((time.time() - start) / 60, 1),
        "python": sys.version.split()[0], "platform": platform.platform(),
        "sklearn": sklearn.__version__, "numpy": np.__version__, "pandas": pd.__version__,
    }
    (out / "info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"Done in {info['elapsed_minutes']} min; results in {out}")


if __name__ == "__main__":
    main()
