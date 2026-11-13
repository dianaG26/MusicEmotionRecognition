import argparse
import json
import platform
import sys
import time
import warnings
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

QUADRANTS = ["Q1", "Q2", "Q3", "Q4"]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = PROJECT_ROOT / "data" / "raw" / "MERGE_Audio_Balanced"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "features" / "features_librosa.csv"

CONFIG = {
    "sample_rate": 22050,
    "mono": True,
    "duration_seconds": 30.0,
    "n_fft": 2048,
    "hop_length": 512,
    "n_mfcc": 20,
    "n_contrast_bands": 6,
    "rolloff_percent": 0.85,
    "aggregation": ["mean", "std"],
}


def aggregate(prefix, matrix):
    matrix = np.atleast_2d(matrix)
    feats = {}
    for i, row in enumerate(matrix):
        name = prefix if matrix.shape[0] == 1 else f"{prefix}_{i + 1:02d}"
        feats[f"{name}_mean"] = float(np.mean(row))
        feats[f"{name}_std"] = float(np.std(row))
    return feats


def extract(path, cfg):
    y, sr = librosa.load(path, sr=cfg["sample_rate"], mono=cfg["mono"], duration=cfg["duration_seconds"])
    n_fft, hop = cfg["n_fft"], cfg["hop_length"]
    stft = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop))
    power = stft ** 2

    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(librosa.feature.melspectrogram(S=power, sr=sr)), sr=sr, n_mfcc=cfg["n_mfcc"])
    feats = {"duration_loaded": len(y) / sr}
    feats.update(aggregate("mfcc", mfcc))
    feats.update(aggregate("mfcc_delta", librosa.feature.delta(mfcc)))
    feats.update(aggregate("spectral_centroid", librosa.feature.spectral_centroid(S=stft, sr=sr)))
    feats.update(aggregate("spectral_bandwidth", librosa.feature.spectral_bandwidth(S=stft, sr=sr)))
    feats.update(aggregate("spectral_rolloff", librosa.feature.spectral_rolloff(S=stft, sr=sr, roll_percent=cfg["rolloff_percent"])))
    feats.update(aggregate("spectral_flatness", librosa.feature.spectral_flatness(S=stft)))
    feats.update(aggregate("spectral_contrast", librosa.feature.spectral_contrast(S=stft, sr=sr, n_bands=cfg["n_contrast_bands"])))
    feats.update(aggregate("zcr", librosa.feature.zero_crossing_rate(y, frame_length=n_fft, hop_length=hop)))
    feats.update(aggregate("rms", librosa.feature.rms(S=stft, frame_length=n_fft, hop_length=hop)))
    feats.update(aggregate("chroma", librosa.feature.chroma_stft(S=power, sr=sr)))

    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    tempo = librosa.feature.tempo(onset_envelope=onset_env, sr=sr, hop_length=hop)
    feats["tempo"] = float(np.atleast_1d(tempo)[0])
    feats.update(aggregate("onset_strength", onset_env))
    return feats


def process(song, quadrant, path, cfg):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            row = extract(path, cfg)
        return {"Song": song, "Quadrant": quadrant, **row}, None
    except Exception as e:
        return None, {"Song": song, "path": str(path), "error": repr(e)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    meta = pd.read_csv(args.dataset / "merge_audio_balanced_metadata.csv")[["Song", "Quadrant"]]
    meta = meta.sort_values("Song").reset_index(drop=True)
    if args.limit:
        meta = meta.groupby("Quadrant", group_keys=False).head(max(1, args.limit // 4))

    jobs = [(r.Song, r.Quadrant, args.dataset / r.Quadrant / f"{r.Song}.mp3") for r in meta.itertuples()]
    print(f"Extracting features for {len(jobs)} files with n_jobs={args.n_jobs}")
    start = time.time()
    results = Parallel(n_jobs=args.n_jobs, verbose=5)(delayed(process)(s, q, p, CONFIG) for s, q, p in jobs)
    elapsed = time.time() - start

    rows = [r for r, e in results if r is not None]
    errors = [e for r, e in results if e is not None]
    df = pd.DataFrame(rows).sort_values("Song").reset_index(drop=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    if errors:
        pd.DataFrame(errors).to_csv(args.output.with_name(args.output.stem + "_errors.csv"), index=False)

    info = {
        "config": CONFIG,
        "dataset": str(args.dataset),
        "n_files": len(jobs),
        "n_extracted": len(rows),
        "n_errors": len(errors),
        "n_features": int(df.shape[1] - 2),
        "elapsed_seconds": round(elapsed, 1),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "librosa": librosa.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    args.output.with_suffix(".json").write_text(json.dumps(info, indent=2), encoding="utf-8")

    print(f"Saved {len(rows)} rows x {df.shape[1] - 2} features to {args.output}")
    print(f"Errors: {len(errors)}; time: {elapsed / 60:.1f} min")
    if len(rows):
        short = df[df["duration_loaded"] < CONFIG["duration_seconds"] - 1]
        print(f"Clips shorter than {CONFIG['duration_seconds'] - 1:.0f} s: {len(short)}")


if __name__ == "__main__":
    main()
