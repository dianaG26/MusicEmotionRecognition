import argparse
import json
import os
import platform
import sys
import time
import urllib.request
import warnings
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET = PROJECT_ROOT / "data" / "raw" / "MERGE_Audio_Balanced"
PRETRAINED = PROJECT_ROOT / "models" / "pretrained"
PANNS_DATA = PRETRAINED / "panns_data"
OUT_DIR = PROJECT_ROOT / "data" / "embeddings"
CHECKPOINT = PANNS_DATA / "Cnn14_mAP=0.431.pth"
LABELS = PANNS_DATA / "class_labels_indices.csv"
CHECKPOINT_URL = "https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1"
LABELS_URL = "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv"
SAMPLE_RATE = 32000
DURATION = 30.0
MOOD_CLASSES = ["Happy music", "Funny music", "Sad music", "Tender music", "Exciting music", "Angry music", "Scary music"]


def download(url, path, min_bytes):
    if path.exists() and path.stat().st_size >= min_bytes:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {path.name} ...", flush=True)
    tmp = path.with_suffix(path.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(path)
    print(f"  saved {path} ({path.stat().st_size / 1e6:.0f} MB)", flush=True)


def load_model():
    download(LABELS_URL, LABELS, 1_000)
    download(CHECKPOINT_URL, CHECKPOINT, 300_000_000)
    os.environ["HOME"] = str(PRETRAINED)
    os.environ["USERPROFILE"] = str(PRETRAINED)
    from panns_inference.config import classes_num, labels
    from panns_inference.models import Cnn14
    model = Cnn14(sample_rate=SAMPLE_RATE, window_size=1024, hop_size=320, mel_bins=64, fmin=50, fmax=14000, classes_num=classes_num)
    model.load_state_dict(torch.load(CHECKPOINT, map_location="cpu", weights_only=False)["model"])
    model.eval()
    return model, labels


def load_audio(path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True, duration=DURATION)
    length = int(SAMPLE_RATE * DURATION)
    return np.pad(y, (0, max(0, length - len(y))))[:length].astype(np.float32), len(y) / SAMPLE_RATE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch", type=int, default=8)
    args = parser.parse_args()

    model, labels = load_model()
    meta = pd.read_csv(DATASET / "merge_audio_balanced_metadata.csv")[["Song", "Quadrant"]].sort_values("Song").reset_index(drop=True)
    if args.limit:
        meta = meta.groupby("Quadrant", group_keys=False).head(max(1, args.limit // 4)).sort_values("Song").reset_index(drop=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    mood_idx = [labels.index(c) for c in MOOD_CLASSES]

    start = time.time()
    embeddings, tags, seconds = [], [], []
    for begin in range(0, len(meta), args.batch):
        chunk = meta.iloc[begin:begin + args.batch]
        audio = [load_audio(DATASET / r.Quadrant / f"{r.Song}.mp3") for r in chunk.itertuples()]
        with torch.no_grad():
            out = model(torch.from_numpy(np.stack([a for a, _ in audio])), None)
        embeddings.append(out["embedding"].numpy())
        tags.append(out["clipwise_output"].numpy())
        seconds += [s for _, s in audio]
        done = begin + len(chunk)
        if done % 200 < args.batch or done == len(meta):
            elapsed = time.time() - start
            print(f"  {done}/{len(meta)} clips | {elapsed / 60:.1f} min | about {elapsed / done * (len(meta) - done) / 60:.0f} min left", flush=True)

    emb = np.concatenate(embeddings)
    tag = np.concatenate(tags)
    emb_df = pd.DataFrame(emb, columns=[f"panns_{i:04d}" for i in range(emb.shape[1])])
    emb_df.insert(0, "Quadrant", meta["Quadrant"])
    emb_df.insert(0, "Song", meta["Song"])
    emb_df.to_csv(OUT_DIR / "panns_cnn14.csv", index=False, float_format="%.6g")
    mood = pd.DataFrame(tag[:, mood_idx], columns=MOOD_CLASSES)
    mood.insert(0, "Quadrant", meta["Quadrant"])
    mood.insert(0, "Song", meta["Song"])
    mood.to_csv(OUT_DIR / "panns_cnn14_audioset_mood.csv", index=False, float_format="%.6g")
    info = {
        "model": "PANNs Cnn14 (Kong et al., 2020), pretrained on AudioSet, checkpoint Cnn14_mAP=0.431.pth",
        "checkpoint": str(CHECKPOINT.relative_to(PROJECT_ROOT)), "checkpoint_url": CHECKPOINT_URL,
        "input": f"{SAMPLE_RATE} Hz mono, first {DURATION:.0f} s, zero-padded at the end",
        "embedding": "2048-dimensional output of the penultimate layer (fc1), one vector per clip",
        "audioset_mood_classes": MOOD_CLASSES,
        "n_clips": len(meta), "n_padded": int(sum(s < DURATION - 0.05 for s in seconds)),
        "elapsed_minutes": round((time.time() - start) / 60, 1),
        "python": sys.version.split()[0], "platform": platform.platform(), "torch": torch.__version__, "librosa": librosa.__version__,
    }
    (OUT_DIR / "panns_cnn14.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"Done in {info['elapsed_minutes']} min: {emb.shape[0]} clips x {emb.shape[1]} values -> {OUT_DIR / 'panns_cnn14.csv'}")


if __name__ == "__main__":
    main()
