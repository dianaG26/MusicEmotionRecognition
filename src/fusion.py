import json
import platform
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score

from experiments_classic import best_params_for, load_split, make, splits
from train_classic import PROJECT_ROOT, QUADRANTS, SEED, load_features
from train_cnn import CNN, load_spectrograms

MODELS_DIR = PROJECT_ROOT / "models"
OUT_DIR = PROJECT_ROOT / "results" / "fusion"
PROTOCOLS = ["A", "B42", "B43", "B44", "B45", "B46"]
HOPS_SECONDS = [5.0, 2.5]
WEIGHTS = [round(w, 1) for w in np.arange(0, 1.01, 0.1)]
FEATURE_SET = "tonal"


def macro_f1(y_true, proba):
    return f1_score(y_true, np.array(QUADRANTS)[proba.argmax(1)], average="macro", labels=QUADRANTS)


def load_cnn(protocol):
    ckpt = torch.load(MODELS_DIR / f"cnn_small_{protocol}.pt", map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model = CNN(cfg["channels"], cfg["dropout"], cfg.get("convs_per_block", 2))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


@torch.no_grad()
def cnn_proba(model, X, crop, hop):
    starts = list(range(0, X.shape[2] - crop + 1, hop))
    if starts[-1] != X.shape[2] - crop:
        starts.append(X.shape[2] - crop)
    out = []
    for begin in range(0, len(X), 64):
        chunk = torch.from_numpy(X[begin:begin + 64])
        out.append(torch.stack([model(chunk[:, None, :, s:s + crop]).softmax(1) for s in starts]).mean(0).numpy())
    return np.concatenate(out), len(starts)


def svm_proba(protocol, ids, data, cols):
    params = best_params_for(pd.read_csv(PROJECT_ROOT / "results" / "experiments" / "runs.csv"), protocol, FEATURE_SET, "svm")
    X = {p: data.loc[ids[p], cols].to_numpy() for p in ids}
    y_train = data.loc[ids["train"], "Quadrant"].to_numpy()
    model = CalibratedClassifierCV(make("svm", params), method="sigmoid", cv=5, ensemble=False).fit(X["train"], y_train)
    order = [list(model.classes_).index(q) for q in QUADRANTS]
    return {p: model.predict_proba(X[p])[:, order] for p in ["validate", "test"]}, params


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    start = time.time()
    spec, index, info = load_spectrograms()
    data, cols = load_features(FEATURE_SET)
    rows, grid, preds = [], [], []
    for protocol in PROTOCOLS:
        t0 = time.time()
        split_name, folder = splits()[protocol]
        ids = load_split(split_name, folder)
        y = {p: data.loc[ids[p], "Quadrant"].to_numpy() for p in ["validate", "test"]}
        model, ckpt = load_cnn(protocol)
        mean, std = ckpt["norm_mean"].reshape(1, -1, 1), ckpt["norm_std"].reshape(1, -1, 1)
        X = {}
        for p in ["validate", "test"]:
            r = index.loc[ids[p], "row"].to_numpy()
            order = np.argsort(r)
            raw = np.asarray(spec[r[order]], dtype=np.float32)[np.argsort(order)]
            X[p] = ((raw - mean) / std).astype(np.float16).astype(np.float32)
        cnn = {}
        for hop_s in HOPS_SECONDS:
            hop = int(round(hop_s / info["seconds_per_frame"]))
            probs = {}
            for p in ["validate", "test"]:
                probs[p], n_crops = cnn_proba(model, X[p], ckpt["crop_frames"], hop)
            cnn[hop_s] = probs
            grid.append({"protocol": protocol, "kind": "cnn_hop", "hop_seconds": hop_s, "crops": n_crops, "weight_cnn": 1.0,
                         "val_macro_f1": macro_f1(y["validate"], probs["validate"])})
        best_hop = max(HOPS_SECONDS, key=lambda h: (round(macro_f1(y["validate"], cnn[h]["validate"]), 6), h))
        svm, svm_params = svm_proba(protocol, ids, data, cols)

        def fused(w, p):
            return w * cnn[best_hop][p] + (1 - w) * svm[p]

        for w in WEIGHTS:
            grid.append({"protocol": protocol, "kind": "fusion_weight", "hop_seconds": best_hop, "weight_cnn": w,
                         "val_macro_f1": macro_f1(y["validate"], fused(w, "validate"))})
        val_by_w = {w: round(macro_f1(y["validate"], fused(w, "validate")), 6) for w in WEIGHTS}
        best_w = max(WEIGHTS, key=lambda w: (val_by_w[w], w))

        candidates = {
            "cnn": cnn[5.0],
            "cnn_tta": cnn[best_hop],
            "svm_calibrated": svm,
            "fusion": {p: fused(best_w, p) for p in ["validate", "test"]},
            "fusion_equal": {p: 0.5 * cnn[best_hop][p] + 0.5 * svm[p] for p in ["validate", "test"]},
        }
        for name, probs in candidates.items():
            label = np.array(QUADRANTS)[probs["test"].argmax(1)]
            f1s = f1_score(y["test"], label, average=None, labels=QUADRANTS)
            confused = int(((y["test"] == "Q3") & (label == "Q4")).sum() + ((y["test"] == "Q4") & (label == "Q3")).sum())
            rows.append({"protocol": protocol, "model": name, "hop_seconds": best_hop if name != "cnn" else 5.0,
                         "weight_cnn": {"cnn": 1.0, "cnn_tta": 1.0, "svm_calibrated": 0.0, "fusion": best_w, "fusion_equal": 0.5}[name],
                         "val_macro_f1": macro_f1(y["validate"], probs["validate"]), "test_macro_f1": macro_f1(y["test"], probs["test"]),
                         "test_accuracy": float((label == y["test"]).mean()), **{f"test_f1_{q}": f for q, f in zip(QUADRANTS, f1s)},
                         "test_q3_q4": confused})
            preds.append(pd.DataFrame(probs["test"], columns=[f"prob_{q}" for q in QUADRANTS]).assign(
                protocol=protocol, model=name, Song=ids["test"].values, true=y["test"], pred=label))
        print(f"{protocol:4s} hop {best_hop}s, weight CNN {best_w} | test macro-F1: CNN {rows[-5]['test_macro_f1']:.4f}, CNN more crops {rows[-4]['test_macro_f1']:.4f}, "
              f"SVM {rows[-3]['test_macro_f1']:.4f}, fusion {rows[-2]['test_macro_f1']:.4f} | {time.time() - t0:.0f}s", flush=True)

    runs = pd.DataFrame(rows)
    runs.to_csv(OUT_DIR / "runs.csv", index=False)
    pd.DataFrame(grid).to_csv(OUT_DIR / "validation_grid.csv", index=False)
    pd.concat(preds).to_csv(OUT_DIR / "test_predictions.csv", index=False)
    info_out = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "evaluation": "number of crops and fusion weight chosen on each split's validation set; test evaluated once",
        "cnn": "cnn_small_<protocol>.pt from src/train_cnn.py, 10-second crops, probabilities averaged over crops",
        "hops_seconds_tried": HOPS_SECONDS,
        "svm": "SVM with the hyperparameters chosen in src/experiments_classic.py (tonal features), probabilities by sigmoid calibration with 5-fold cross-validation on the training set",
        "fusion": "weight_cnn * p_cnn + (1 - weight_cnn) * p_svm; ties broken towards the higher CNN weight",
        "weights_tried": WEIGHTS,
        "elapsed_minutes": round((time.time() - start) / 60, 1),
        "seed": SEED, "python": sys.version.split()[0], "platform": platform.platform(),
        "torch": torch.__version__, "sklearn": sklearn.__version__, "numpy": np.__version__,
    }
    (OUT_DIR / "info.json").write_text(json.dumps(info_out, indent=2), encoding="utf-8")
    b = runs[runs.protocol != "A"].groupby("model")["test_macro_f1"].agg(["mean", "std"])
    print("\nProtocol B, test macro-F1 (mean, sd over 5 splits):")
    print(b.loc[["cnn", "cnn_tta", "svm_calibrated", "fusion", "fusion_equal"]].round(4).to_string())
    print(f"\nDone in {info_out['elapsed_minutes']} min; results in {OUT_DIR}")


if __name__ == "__main__":
    main()
