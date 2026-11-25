import itertools
import json
import platform
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import sklearn
from scipy.stats import kruskal
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from experiments_classic import load_split, splits
from train_classic import PROJECT_ROOT, QUADRANTS, SEED, load_features

CNN_DIR = PROJECT_ROOT / "results" / "cnn"
OUT_DIR = PROJECT_ROOT / "results" / "two_stage"
PROTOCOLS = ["A", "B42", "B43", "B44", "B45", "B46"]
FEATURE_SET = "tonal"
TOP_K = [20, 40, 80, None]
WEIGHTS = [round(w, 1) for w in np.arange(0, 1.01, 0.1)]
GRIDS = {
    "logreg": {"C": [0.01, 0.1, 1, 10]},
    "svm": {"C": [0.1, 1, 10], "gamma": ["scale", 0.001, 0.01]},
    "rf": {"min_samples_leaf": [1, 4]},
}
LOW = ["Q3", "Q4"]


def specialist(name, params):
    if name == "logreg":
        return Pipeline([("scaler", StandardScaler()), ("logreg", LogisticRegression(C=params["C"], max_iter=5000, random_state=SEED))])
    if name == "svm":
        svm = Pipeline([("scaler", StandardScaler()), ("svm", SVC(kernel="rbf", C=params["C"], gamma=params["gamma"], random_state=SEED))])
        return CalibratedClassifierCV(svm, method="sigmoid", cv=5, ensemble=False)
    return RandomForestClassifier(n_estimators=500, min_samples_leaf=params["min_samples_leaf"], max_features="sqrt", n_jobs=-1, random_state=SEED)


def cnn_runs():
    rows = []
    for path in CNN_DIR.glob("*_cnn_small_*/metrics.json"):
        m = json.loads(path.read_text(encoding="utf-8"))
        rows.append({"protocol": m["protocol"], "folder": path.parent})
    runs = pd.DataFrame(rows).sort_values("folder")
    return runs.drop_duplicates("protocol", keep="last").set_index("protocol")["folder"]


def cnn_proba(folder, part, songs):
    p = pd.read_csv(folder / f"{part}_predictions.csv", dtype={"Song": str}).set_index("Song").loc[songs]
    return p[[f"prob_{q}" for q in QUADRANTS]].to_numpy()


def rank_features(X, y, cols):
    rows = []
    for j, c in enumerate(cols):
        h = kruskal(X[y == "Q3", j], X[y == "Q4", j]).statistic
        rows.append({"feature": c, "epsilon_squared": (h - 1) / (len(y) - 1)})
    return pd.DataFrame(rows).sort_values("epsilon_squared", ascending=False).reset_index(drop=True)


def two_stage(cnn, p_sad, w):
    high = cnn[:, 0] + cnn[:, 1] >= cnn[:, 2] + cnn[:, 3]
    high_label = np.where(cnn[:, 0] >= cnn[:, 1], "Q1", "Q2")
    ratio = cnn[:, 2] / np.clip(cnn[:, 2] + cnn[:, 3], 1e-9, None)
    sad = w * ratio + (1 - w) * p_sad
    return np.where(high, high_label, np.where(sad >= 0.5, "Q3", "Q4"))


def macro_f1(y, pred):
    return f1_score(y, pred, average="macro", labels=QUADRANTS)


def describe(protocol, name, y, pred, extra):
    f1s = f1_score(y, pred, average=None, labels=QUADRANTS)
    low_true = np.isin(y, LOW)
    low_pred = np.isin(pred, LOW)
    both_low = low_true & low_pred
    return {"protocol": protocol, "model": name, **extra, "test_macro_f1": macro_f1(y, pred), "test_accuracy": float((pred == y).mean()),
            **{f"test_f1_{q}": f for q, f in zip(QUADRANTS, f1s)},
            "test_q3_as_q4": int(((y == "Q3") & (pred == "Q4")).sum()), "test_q4_as_q3": int(((y == "Q4") & (pred == "Q3")).sum()),
            "test_sad_calm_accuracy_when_both_low": float((pred[both_low] == y[both_low]).mean())}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    start = time.time()
    data, cols = load_features(FEATURE_SET)
    folders = cnn_runs()
    missing = [p for p in PROTOCOLS if p not in folders.index]
    if missing:
        raise SystemExit(f"Missing CNN runs for {missing}: run src\\train_cnn.py --config small first")
    rows, grid, preds, rankings = [], [], [], []
    for protocol in PROTOCOLS:
        t0 = time.time()
        split_name, folder = splits()[protocol]
        ids = load_split(split_name, folder)
        X = {p: data.loc[ids[p], cols].to_numpy() for p in ids}
        y = {p: data.loc[ids[p], "Quadrant"].to_numpy() for p in ids}
        cnn = {p: cnn_proba(folders[protocol], p, ids[p]) for p in ["validate", "test"]}
        low = np.isin(y["train"], LOW)
        ranking = rank_features(X["train"][low], y["train"][low], cols)
        rankings.append(ranking.head(20).assign(protocol=protocol, rank=range(1, 21)))

        cnn_val = np.array(QUADRANTS)[cnn["validate"].argmax(1)]
        best = {"name": "cnn", "val": round(macro_f1(y["validate"], cnn_val), 6), "w": 1.0, "spec": None, "k": None, "params": None}
        grid.append({"protocol": protocol, "candidate": "cnn", "val_macro_f1": best["val"]})
        fitted = {}
        for k, (name, g) in itertools.product(TOP_K, GRIDS.items()):
            use = [cols.index(f) for f in (ranking["feature"] if k is None else ranking["feature"].head(k))]
            for values in itertools.product(*g.values()):
                params = dict(zip(g, values))
                model = specialist(name, params).fit(X["train"][low][:, use], y["train"][low])
                sad_col = list(model.classes_).index("Q3")
                p_sad = {p: model.predict_proba(X[p][:, use])[:, sad_col] for p in ["validate", "test"]}
                key = (name, k, json.dumps(params))
                fitted[key] = p_sad
                for w in WEIGHTS:
                    score = round(macro_f1(y["validate"], two_stage(cnn["validate"], p_sad["validate"], w)), 6)
                    grid.append({"protocol": protocol, "candidate": "two_stage", "specialist": name, "top_k": k or len(cols), "params": json.dumps(params),
                                 "weight_cnn": w, "val_macro_f1": score})
                    if score > best["val"]:
                        best = {"name": "two_stage", "val": score, "w": w, "spec": name, "k": k, "params": json.dumps(params)}

        y_test = y["test"]
        variants = {"cnn": (np.array(QUADRANTS)[cnn["test"].argmax(1)], {"weight_cnn": None, "specialist": None, "top_k": None, "params": None, "val_macro_f1": macro_f1(y["validate"], cnn_val)})}
        variants["two_stage_cnn_only"] = (two_stage(cnn["test"], np.zeros(len(y_test)), 1.0),
                                          {"weight_cnn": 1.0, "specialist": None, "top_k": None, "params": None,
                                           "val_macro_f1": macro_f1(y["validate"], two_stage(cnn["validate"], np.zeros(len(y["validate"])), 1.0))})
        if best["name"] == "two_stage":
            p_sad = fitted[(best["spec"], best["k"], best["params"])]
            chosen = two_stage(cnn["test"], p_sad["test"], best["w"])
        else:
            chosen = variants["cnn"][0]
        variants["selected"] = (chosen, {"weight_cnn": best["w"], "specialist": best["spec"], "top_k": (best["k"] or len(cols)) if best["spec"] else None,
                                         "params": best["params"], "val_macro_f1": best["val"]})
        spec_only = max(((key, round(macro_f1(y["validate"], two_stage(cnn["validate"], ps["validate"], 0.0)), 6)) for key, ps in fitted.items()), key=lambda t: t[1])
        (s_name, s_k, s_params), s_val = spec_only
        variants["two_stage_specialist_only"] = (two_stage(cnn["test"], fitted[spec_only[0]]["test"], 0.0),
                                                 {"weight_cnn": 0.0, "specialist": s_name, "top_k": s_k or len(cols), "params": s_params, "val_macro_f1": s_val})
        for name, (pred, extra) in variants.items():
            rows.append(describe(protocol, name, y_test, pred, extra))
            preds.append(pd.DataFrame({"protocol": protocol, "model": name, "Song": ids["test"].values, "true": y_test, "pred": pred}))
        r = {d["model"]: d for d in rows[-4:]}
        print(f"{protocol:4s} selected: {best['name']}" + (f" ({best['spec']}, top {best['k'] or len(cols)}, weight CNN {best['w']})" if best["spec"] else "")
              + f" | test macro-F1 CNN {r['cnn']['test_macro_f1']:.4f} -> {r['selected']['test_macro_f1']:.4f}"
              + f" | Q3<->Q4 {r['cnn']['test_q3_as_q4'] + r['cnn']['test_q4_as_q3']} -> {r['selected']['test_q3_as_q4'] + r['selected']['test_q4_as_q3']} | {time.time() - t0:.0f}s", flush=True)

    runs = pd.DataFrame(rows)
    runs.to_csv(OUT_DIR / "runs.csv", index=False)
    pd.DataFrame(grid).to_csv(OUT_DIR / "validation_grid.csv", index=False)
    pd.concat(preds).to_csv(OUT_DIR / "test_predictions.csv", index=False)
    pd.concat(rankings).to_csv(OUT_DIR / "q3_q4_feature_ranking_train.csv", index=False)
    info = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "stage_1": "CNN (cnn_small_<protocol>): high arousal if P(Q1) + P(Q2) >= P(Q3) + P(Q4); then Q1 or Q2 by the larger CNN probability",
        "stage_2": "low-arousal clips: weight_cnn * P_CNN(Q3 | Q3 or Q4) + (1 - weight_cnn) * P_specialist(Q3) >= 0.5 -> Q3, else Q4",
        "specialist": "binary classifier trained only on the Q3 and Q4 training songs, on the top-k tonal features ranked by Kruskal-Wallis epsilon squared between Q3 and Q4 on those songs",
        "grids": {k: {kk: [str(x) for x in vv] for kk, vv in v.items()} for k, v in GRIDS.items()},
        "top_k": [k or "all" for k in TOP_K], "weights": WEIGHTS,
        "selection": "on each split, the CNN alone or the two-stage candidate with the highest validation macro-F1; ties go to the CNN alone",
        "cnn_probabilities": "read from results/cnn/<run>/validate_predictions.csv and test_predictions.csv (6 crops)",
        "elapsed_minutes": round((time.time() - start) / 60, 1),
        "seed": SEED, "python": sys.version.split()[0], "platform": platform.platform(), "sklearn": sklearn.__version__, "numpy": np.__version__,
    }
    (OUT_DIR / "info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    b = runs[runs.protocol != "A"].assign(q34=lambda d: d.test_q3_as_q4 + d.test_q4_as_q3)
    summary = b.groupby("model").agg(macro_f1=("test_macro_f1", "mean"), sd=("test_macro_f1", "std"), f1_Q3=("test_f1_Q3", "mean"),
                                     f1_Q4=("test_f1_Q4", "mean"), q3_q4_confusions=("q34", "mean"))
    print("\nProtocol B (mean over 5 splits):")
    print(summary.loc[["cnn", "two_stage_cnn_only", "two_stage_specialist_only", "selected"]].round(4).to_string())
    print(f"\nDone in {info['elapsed_minutes']} min; results in {OUT_DIR}")


if __name__ == "__main__":
    main()
