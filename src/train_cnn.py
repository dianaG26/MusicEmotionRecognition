import argparse
import copy
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import f1_score

from experiments_classic import load_split, splits
from train_classic import PROJECT_ROOT, QUADRANTS, SEED, save_confusion

SPEC_DIR = PROJECT_ROOT / "data" / "spectrograms"
RESULTS_DIR = PROJECT_ROOT / "results" / "cnn"
MODELS_DIR = PROJECT_ROOT / "models"
PARTS = ["train", "validate", "test"]

CONFIGS = {
    "small": {"channels": [16, 32, 64, 128], "convs_per_block": 2, "dropout": 0.3, "crop_seconds": 10, "spec_augment": False, "lr": 1e-3, "weight_decay": 1e-4},
    "small_aug": {"channels": [16, 32, 64, 128], "convs_per_block": 2, "dropout": 0.3, "crop_seconds": 10, "spec_augment": True, "lr": 1e-3, "weight_decay": 1e-4},
    "fast": {"channels": [16, 32, 64, 128], "convs_per_block": 1, "dropout": 0.3, "crop_seconds": 10, "spec_augment": False, "lr": 1e-3, "weight_decay": 1e-4},
    "fast_aug": {"channels": [16, 32, 64, 128], "convs_per_block": 1, "dropout": 0.3, "crop_seconds": 10, "spec_augment": True, "lr": 1e-3, "weight_decay": 1e-4},
}
TRAINING = {"batch_size": 32, "max_epochs": 60, "patience": 12, "lr_patience": 4, "lr_factor": 0.5, "eval_crop_hop_seconds": 5,
            "freq_masks": 2, "freq_mask_max": 16, "time_masks": 2, "time_mask_max": 40, "gain_db": 6}


class CNN(nn.Module):
    def __init__(self, channels, dropout, convs_per_block):
        super().__init__()
        layers, c_in = [], 1
        for c in channels:
            for _ in range(convs_per_block):
                layers += [nn.Conv2d(c_in, c, 3, padding=1, bias=False), nn.BatchNorm2d(c), nn.ReLU(inplace=True)]
                c_in = c
            layers.append(nn.MaxPool2d(2))
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(c_in, len(QUADRANTS)))

    def forward(self, x):
        x = self.features(x)
        return self.head(x.mean(dim=(2, 3)))


def frames(seconds, info):
    return int(round(seconds / info["seconds_per_frame"]))


def load_spectrograms():
    info = json.loads((SPEC_DIR / "mel128.json").read_text(encoding="utf-8"))
    index = pd.read_csv(SPEC_DIR / "mel128_index.csv")
    index["Song"] = index["Song"].astype(str)
    data = np.load(SPEC_DIR / "mel128.npy", mmap_mode="r")
    return data, index.set_index("Song"), info


def augment(x, rng, inv_std):
    x = x + rng.uniform(-TRAINING["gain_db"], TRAINING["gain_db"]) * inv_std
    n_mels, n_frames = x.shape
    for _ in range(TRAINING["freq_masks"]):
        w = rng.integers(0, TRAINING["freq_mask_max"] + 1)
        f0 = rng.integers(0, n_mels - w + 1)
        x[f0:f0 + w, :] = 0.0
    for _ in range(TRAINING["time_masks"]):
        w = rng.integers(0, TRAINING["time_mask_max"] + 1)
        t0 = rng.integers(0, n_frames - w + 1)
        x[:, t0:t0 + w] = 0.0
    return x


def train_batches(X, y, crop, cfg, rng, inv_std):
    order = rng.permutation(len(X))
    for start in range(0, len(order), TRAINING["batch_size"]):
        idx = order[start:start + TRAINING["batch_size"]]
        batch = []
        for i in idx:
            t0 = rng.integers(0, X.shape[2] - crop + 1)
            x = X[i, :, t0:t0 + crop].astype(np.float32)
            if cfg["spec_augment"]:
                x = augment(x, rng, inv_std)
            batch.append(x)
        yield torch.from_numpy(np.stack(batch)[:, None]), torch.from_numpy(y[idx])


@torch.no_grad()
def predict_proba(model, X, crop, hop):
    model.eval()
    starts = list(range(0, X.shape[2] - crop + 1, hop))
    if starts[-1] != X.shape[2] - crop:
        starts.append(X.shape[2] - crop)
    out = []
    for begin in range(0, len(X), 64):
        chunk = torch.from_numpy(X[begin:begin + 64].astype(np.float32))
        probs = torch.stack([model(chunk[:, None, :, s:s + crop]).softmax(1) for s in starts]).mean(0)
        out.append(probs.numpy())
    return np.concatenate(out)


def macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average="macro", labels=range(len(QUADRANTS)))


def run(protocol, config_name, stamp, data, index, info, max_epochs, smoke):
    cfg = CONFIGS[config_name]
    split_name, folder = splits()[protocol]
    ids = load_split(split_name, folder)
    if smoke:
        ids = {p: s.groupby(index.loc[s, "Quadrant"].values).head(8).reset_index(drop=True) for p, s in ids.items()}
    rows = {p: index.loc[ids[p], "row"].to_numpy() for p in PARTS}
    y = {p: index.loc[ids[p], "Quadrant"].map({q: i for i, q in enumerate(QUADRANTS)}).to_numpy(dtype=np.int64, copy=True) for p in PARTS}

    t0 = time.time()
    raw = {p: np.asarray(data[np.sort(rows[p])]) for p in PARTS}
    order = {p: np.argsort(np.argsort(rows[p])) for p in PARTS}
    raw = {p: raw[p][order[p]] for p in PARTS}
    mean = raw["train"].astype(np.float32).mean(axis=(0, 2), keepdims=True)
    std = raw["train"].astype(np.float32).std(axis=(0, 2), keepdims=True) + 1e-6
    X = {p: ((raw[p].astype(np.float32) - mean) / std).astype(np.float16) for p in PARTS}
    inv_std = (1.0 / std[0]).astype(np.float32)
    del raw
    load_seconds = time.time() - t0

    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    crop = frames(cfg["crop_seconds"], info)
    hop = frames(TRAINING["eval_crop_hop_seconds"], info)
    model = CNN(cfg["channels"], cfg["dropout"], cfg["convs_per_block"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=TRAINING["lr_factor"], patience=TRAINING["lr_patience"])
    loss_fn = nn.CrossEntropyLoss()
    n_params = sum(p.numel() for p in model.parameters())
    tag = f"cnn_{config_name}_{protocol}"
    print(f"\n{tag}: {n_params:,} parameters | crop {crop} frames | train {len(y['train'])}, validate {len(y['validate'])}, test {len(y['test'])}", flush=True)

    history, best_state, best_f1, best_epoch, wait = [], None, -1.0, 0, 0
    start = time.time()
    for epoch in range(1, max_epochs + 1):
        model.train()
        e0, losses, correct = time.time(), [], 0
        for xb, yb in train_batches(X["train"], y["train"], crop, cfg, rng, inv_std):
            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()
            losses.append(loss.item() * len(yb))
            correct += int((logits.argmax(1) == yb).sum())
        val_proba = predict_proba(model, X["validate"], crop, hop)
        val_loss = float(nn.functional.nll_loss(torch.log(torch.from_numpy(val_proba) + 1e-9), torch.from_numpy(y["validate"])))
        val_f1 = macro_f1(y["validate"], val_proba.argmax(1))
        scheduler.step(val_f1)
        history.append({"epoch": epoch, "train_loss": sum(losses) / len(y["train"]), "train_crop_accuracy": correct / len(y["train"]),
                        "val_loss": val_loss, "val_macro_f1": val_f1, "lr": optimizer.param_groups[0]["lr"], "seconds": time.time() - e0})
        print(f"  epoch {epoch:3d} | train loss {history[-1]['train_loss']:.3f} | val loss {val_loss:.3f} | val macro-F1 {val_f1:.4f} | {history[-1]['seconds']:.0f}s", flush=True)
        if val_f1 > best_f1:
            best_f1, best_epoch, best_state, wait = val_f1, epoch, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= TRAINING["patience"]:
                break
    train_seconds = time.time() - start

    model.load_state_dict(best_state)
    proba = {p: predict_proba(model, X[p], crop, hop) for p in PARTS}
    pred = {p: proba[p].argmax(1) for p in PARTS}
    labels = {p: np.array(QUADRANTS)[pred[p]] for p in PARTS}
    truth = {p: np.array(QUADRANTS)[y[p]] for p in PARTS}

    out = (RESULTS_DIR.with_name("cnn_smoke") if smoke else RESULTS_DIR) / f"{stamp}_{tag}"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(out / "history.csv", index=False)
    cm = save_confusion(truth["test"], labels["test"], f"CNN {config_name} | protocol {protocol} | test", out / "confusion_matrix_test.png")
    pd.DataFrame(cm, index=[f"true_{q}" for q in QUADRANTS], columns=[f"pred_{q}" for q in QUADRANTS]).to_csv(out / "confusion_matrix_test.csv")
    for p in ["validate", "test"]:
        pd.DataFrame(proba[p], columns=[f"prob_{q}" for q in QUADRANTS]).assign(Song=ids[p].values, true=truth[p], pred=labels[p]).to_csv(out / f"{p}_predictions.csv", index=False)

    model_path = None
    if not smoke:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        model_path = MODELS_DIR / f"{tag}.pt"
        torch.save({"state_dict": best_state, "config": cfg, "training": TRAINING, "classes": QUADRANTS, "norm_mean": mean.squeeze(), "norm_std": std.squeeze(),
                    "spectrogram": info["config"], "crop_frames": crop, "eval_hop_frames": hop, "trained": stamp, "split": split_name}, model_path)

    f1_per = f1_score(y["test"], pred["test"], average=None, labels=range(len(QUADRANTS)))
    metrics = {
        "model": "cnn", "config": config_name, "protocol": protocol, "split": split_name,
        "evaluation": "epoch selected on validation macro-F1 (early stopping); test evaluated once with the selected epoch",
        "config_values": cfg, "training": TRAINING, "parameters": n_params,
        "n_samples": {p: int(len(y[p])) for p in PARTS},
        "best_epoch": best_epoch, "epochs_run": len(history),
        "train_macro_f1": round(macro_f1(y["train"], pred["train"]), 4),
        "validation_macro_f1": round(macro_f1(y["validate"], pred["validate"]), 4),
        "test_macro_f1": round(macro_f1(y["test"], pred["test"]), 4),
        "test_accuracy": round(float((pred["test"] == y["test"]).mean()), 4),
        "test_f1_per_class": {q: round(float(f), 4) for q, f in zip(QUADRANTS, f1_per)},
        "test_q3_q4_confusions": int(cm[2, 3] + cm[3, 2]),
        "saved_model": str(model_path.relative_to(PROJECT_ROOT)) if model_path else None,
        "model_mb": round(model_path.stat().st_size / 1e6, 2) if model_path else None,
        "seed": SEED, "load_seconds": round(load_seconds, 1), "train_seconds": round(train_seconds, 1),
        "seconds_per_epoch": round(train_seconds / len(history), 1), "threads": torch.get_num_threads(),
        "python": sys.version.split()[0], "platform": platform.platform(), "torch": torch.__version__, "numpy": np.__version__,
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"  best epoch {best_epoch} | macro-F1 train {metrics['train_macro_f1']:.4f} | validation {metrics['validation_macro_f1']:.4f} | test {metrics['test_macro_f1']:.4f} | Q3<->Q4 {metrics['test_q3_q4_confusions']}", flush=True)
    return {"config": config_name, "protocol": protocol, "parameters": n_params, "best_epoch": best_epoch,
            **{k: metrics[k] for k in ["train_macro_f1", "validation_macro_f1", "test_macro_f1", "test_accuracy", "test_q3_q4_confusions", "train_seconds"]},
            "results": out.name}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", nargs="+", choices=list(CONFIGS), default=["small"])
    parser.add_argument("--protocol", nargs="+", default=["A"])
    parser.add_argument("--epochs", type=int, default=TRAINING["max_epochs"])
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    protocols = list(splits()) if args.protocol == ["all"] else args.protocol
    unknown = [p for p in protocols if p not in splits()]
    if unknown:
        raise SystemExit(f"Unknown protocol {unknown}; choose from {list(splits())} or all")

    data, index, info = load_spectrograms()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rows = [run(p, c, stamp, data, index, info, args.epochs, args.smoke) for c in args.config for p in protocols]
    summary = pd.DataFrame(rows)
    out_dir = RESULTS_DIR.with_name("cnn_smoke") if args.smoke else RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / f"{stamp}_summary.csv", index=False)
    print("\n" + summary.drop(columns="results").to_string(index=False))


if __name__ == "__main__":
    main()
