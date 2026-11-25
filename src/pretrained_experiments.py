import itertools
import json
import platform
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from experiments_classic import load_split, splits
from train_classic import PROJECT_ROOT, QUADRANTS, SEED, load_features
from two_stage import LOW, WEIGHTS, cnn_proba, cnn_runs, describe, two_stage

EMB_FILE = PROJECT_ROOT / "data" / "embeddings" / "panns_cnn14.csv"
OUT_DIR = PROJECT_ROOT / "results" / "pretrained"
PROTOCOLS = ["A", "B42", "B43", "B44", "B45", "B46"]
GRIDS = {
    "logreg": {"C": [0.001, 0.01, 0.1, 1]},
    "svm": {"C": [0.1, 1, 10], "gamma": ["scale", 0.0001, 0.001]},
}


def build(name, params, calibrated=False):
    if name == "logreg":
        return Pipeline([("scaler", StandardScaler()), ("logreg", LogisticRegression(C=params["C"], max_iter=5000, random_state=SEED))])
    svm = Pipeline([("scaler", StandardScaler()), ("svm", SVC(kernel="rbf", C=params["C"], gamma=params["gamma"], random_state=SEED))])
    return CalibratedClassifierCV(svm, method="sigmoid", cv=5, ensemble=False) if calibrated else svm


def macro_f1(y, pred):
    return f1_score(y, pred, average="macro", labels=QUADRANTS)


def load_sets():
    emb = pd.read_csv(EMB_FILE, dtype={"Song": str}).set_index("Song")
    emb_cols = [c for c in emb.columns if c.startswith("panns_")]
    tonal, tonal_cols = load_features("tonal")
    joined = tonal.join(emb[emb_cols], how="inner")
    return joined, {"panns": emb_cols, "panns_tonal": tonal_cols + emb_cols}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    start = time.time()
    data, sets = load_sets()
    folders = cnn_runs()
    rows, grid, preds = [], [], []
    for protocol in PROTOCOLS:
        t0 = time.time()
        split_name, folder = splits()[protocol]
        ids = load_split(split_name, folder)
        y = {p: data.loc[ids[p], "Quadrant"].to_numpy() for p in ids}
        cnn = {p: cnn_proba(folders[protocol], p, ids[p]) for p in ["validate", "test"]}
        low = np.isin(y["train"], LOW)
        variants = {}
        for set_name, cols in sets.items():
            X = {p: data.loc[ids[p], cols].to_numpy() for p in ids}
            best = None
            for name, g in GRIDS.items():
                for values in itertools.product(*g.values()):
                    params = dict(zip(g, values))
                    pred = build(name, params).fit(X["train"], y["train"]).predict(X["validate"])
                    score = macro_f1(y["validate"], pred)
                    grid.append({"protocol": protocol, "stage": "four_class", "features": set_name, "model": name, "params": json.dumps({k: str(v) for k, v in params.items()}), "val_macro_f1": score})
                    if best is None or score > best[0]:
                        best = (score, name, params)
            score, name, params = best
            model = build(name, params, calibrated=True).fit(X["train"], y["train"])
            order = [list(model.classes_).index(q) for q in QUADRANTS]
            proba = {p: model.predict_proba(X[p])[:, order] for p in ["validate", "test"]}
            variants[f"{set_name}_{name}"] = (np.array(QUADRANTS)[proba["test"].argmax(1)],
                                              {"weight_cnn": 0.0, "params": json.dumps({k: str(v) for k, v in params.items()}), "val_macro_f1": macro_f1(y["validate"], np.array(QUADRANTS)[proba["validate"].argmax(1)])})
            by_w = {w: macro_f1(y["validate"], np.array(QUADRANTS)[(w * cnn["validate"] + (1 - w) * proba["validate"]).argmax(1)]) for w in WEIGHTS}
            w_best = max(WEIGHTS, key=lambda w: (round(by_w[w], 6), w))
            for w, s in by_w.items():
                grid.append({"protocol": protocol, "stage": "fusion_cnn", "features": set_name, "model": name, "weight_cnn": w, "val_macro_f1": s})
            variants[f"cnn_fusion_{set_name}"] = (np.array(QUADRANTS)[(w_best * cnn["test"] + (1 - w_best) * proba["test"]).argmax(1)],
                                                  {"weight_cnn": w_best, "params": json.dumps({k: str(v) for k, v in params.items()}), "val_macro_f1": by_w[w_best]})

            best2 = None
            for name2, g in GRIDS.items():
                for values in itertools.product(*g.values()):
                    params2 = dict(zip(g, values))
                    spec = build(name2, params2, calibrated=(name2 == "svm")).fit(X["train"][low], y["train"][low])
                    sad = list(spec.classes_).index("Q3")
                    p_sad = {p: spec.predict_proba(X[p])[:, sad] for p in ["validate", "test"]}
                    for w in WEIGHTS:
                        s = macro_f1(y["validate"], two_stage(cnn["validate"], p_sad["validate"], w))
                        grid.append({"protocol": protocol, "stage": "two_stage", "features": set_name, "model": name2, "params": json.dumps({k: str(v) for k, v in params2.items()}), "weight_cnn": w, "val_macro_f1": s})
                        if best2 is None or round(s, 6) > round(best2[0], 6):
                            best2 = (s, w, name2, params2, p_sad)
            s, w, name2, params2, p_sad = best2
            variants[f"two_stage_{set_name}"] = (two_stage(cnn["test"], p_sad["test"], w),
                                                 {"weight_cnn": w, "params": json.dumps({"specialist": name2, **{k: str(v) for k, v in params2.items()}}), "val_macro_f1": s})

        cnn_val = macro_f1(y["validate"], np.array(QUADRANTS)[cnn["validate"].argmax(1)])
        variants = {"cnn": (np.array(QUADRANTS)[cnn["test"].argmax(1)], {"weight_cnn": 1.0, "params": None, "val_macro_f1": cnn_val}), **variants}
        selected = max(variants, key=lambda k: (round(variants[k][1]["val_macro_f1"], 6), k == "cnn"))
        variants["selected"] = (variants[selected][0], {**variants[selected][1], "params": f"{selected}: {variants[selected][1]['params']}"})
        for name, (pred, extra) in variants.items():
            rows.append(describe(protocol, name, y["test"], pred, extra))
            preds.append(pd.DataFrame({"protocol": protocol, "model": name, "Song": ids["test"].values, "true": y["test"], "pred": pred}))
        r = {d["model"]: d for d in rows if d["protocol"] == protocol}
        print(f"{protocol:4s} selected on validation: {selected} | test macro-F1: CNN {r['cnn']['test_macro_f1']:.4f}, PANNs SVM/LR {r[[k for k in r if k.startswith('panns_') and not k.startswith('panns_tonal')][0]]['test_macro_f1']:.4f}, "
              f"selected {r['selected']['test_macro_f1']:.4f} | Q3<->Q4 {r['cnn']['test_q3_as_q4'] + r['cnn']['test_q4_as_q3']} -> {r['selected']['test_q3_as_q4'] + r['selected']['test_q4_as_q3']} | {time.time() - t0:.0f}s", flush=True)

    runs = pd.DataFrame(rows)
    runs.to_csv(OUT_DIR / "runs.csv", index=False)
    pd.DataFrame(grid).to_csv(OUT_DIR / "validation_grid.csv", index=False)
    pd.concat(preds).to_csv(OUT_DIR / "test_predictions.csv", index=False)
    info = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "features": {"panns": "2048-dimensional PANNs Cnn14 embedding (src/panns_embeddings.py)", "panns_tonal": "PANNs embedding + 155 tonal hand-crafted features"},
        "four_class": "logistic regression or RBF SVM on standardised features, hyperparameters chosen on validation; probabilities by sigmoid calibration (5-fold CV on training set)",
        "fusion": "weight_cnn * p_CNN + (1 - weight_cnn) * p_PANNs-model, weight chosen on validation",
        "two_stage": "stage 1 = CNN arousal decision; stage 2 = binary Q3/Q4 specialist trained on the training Q3/Q4 songs, combined with the CNN's Q3/Q4 ratio by a weight chosen on validation",
        "selection": "per split, the variant with the highest validation macro-F1; ties go to the CNN",
        "grids": {k: {kk: [str(x) for x in vv] for kk, vv in v.items()} for k, v in GRIDS.items()}, "weights": WEIGHTS,
        "elapsed_minutes": round((time.time() - start) / 60, 1),
        "seed": SEED, "python": sys.version.split()[0], "platform": platform.platform(), "sklearn": sklearn.__version__, "numpy": np.__version__,
    }
    (OUT_DIR / "info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    b = runs[runs.protocol != "A"].assign(q34=lambda d: d.test_q3_as_q4 + d.test_q4_as_q3, model=lambda d: d.model.str.replace(r"^(panns(_tonal)?)_(svm|logreg)$", r"\1_alone", regex=True))
    summary = b.groupby("model").agg(macro_f1=("test_macro_f1", "mean"), sd=("test_macro_f1", "std"), f1_Q3=("test_f1_Q3", "mean"), f1_Q4=("test_f1_Q4", "mean"),
                                     q3_q4_confusions=("q34", "mean"), sad_calm_accuracy=("test_sad_calm_accuracy_when_both_low", "mean"))
    print("\nProtocol B (mean over 5 splits):")
    print(summary.round(4).to_string())
    print(f"\nDone in {info['elapsed_minutes']} min; results in {OUT_DIR}")


if __name__ == "__main__":
    main()
