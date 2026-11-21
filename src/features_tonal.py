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
from scipy.ndimage import gaussian_filter1d

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = PROJECT_ROOT / "data" / "raw" / "MERGE_Audio_Balanced"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "features" / "features_tonal.csv"

CONFIG = {
    "sample_rate": 22050,
    "mono": True,
    "duration_seconds": 30.0,
    "hop_length": 512,
    "chroma": "chroma_cqt on the harmonic component (HPSS)",
    "key_profiles": "Krumhansl-Kessler major and minor",
    "segment_seconds": 3.0,
    "hcdf_smoothing_sigma_frames": 8,
    "aggregation": ["mean", "std"],
}

MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def key_templates():
    rows = [np.roll(MAJOR_PROFILE, k) for k in range(12)] + [np.roll(MINOR_PROFILE, k) for k in range(12)]
    t = np.array(rows)
    t = t - t.mean(axis=1, keepdims=True)
    return t / np.linalg.norm(t, axis=1, keepdims=True)


TEMPLATES = key_templates()


def key_strengths(chroma_vector):
    c = chroma_vector - chroma_vector.mean()
    norm = np.linalg.norm(c)
    if norm == 0:
        return np.zeros(24)
    return TEMPLATES @ (c / norm)


def key_summary(chroma_vector):
    s = key_strengths(chroma_vector)
    best_major, best_minor = s[:12].max(), s[12:].max()
    return {
        "key_clarity": float(s.max()),
        "modality": float(best_major - best_minor),
        "key_strength_major": float(best_major),
        "key_strength_minor": float(best_minor),
    }


def extract(path, cfg):
    y, sr = librosa.load(path, sr=cfg["sample_rate"], mono=cfg["mono"], duration=cfg["duration_seconds"])
    hop = cfg["hop_length"]
    harmonic = librosa.effects.harmonic(y)
    chroma = librosa.feature.chroma_cqt(y=harmonic, sr=sr, hop_length=hop)

    feats = key_summary(chroma.mean(axis=1))

    seg = max(1, int(round(cfg["segment_seconds"] * sr / hop)))
    segments = [key_summary(chroma[:, i:i + seg].mean(axis=1)) for i in range(0, chroma.shape[1], seg) if chroma[:, i:i + seg].shape[1] >= seg // 2]
    for name in ["key_clarity", "modality"]:
        values = np.array([s[name] for s in segments])
        feats[f"{name}_segments_mean"] = float(values.mean())
        feats[f"{name}_segments_std"] = float(values.std())

    tonnetz = librosa.feature.tonnetz(chroma=chroma, sr=sr)
    for i, row in enumerate(tonnetz):
        feats[f"tonnetz_{i + 1:02d}_mean"] = float(row.mean())
        feats[f"tonnetz_{i + 1:02d}_std"] = float(row.std())

    smooth = gaussian_filter1d(tonnetz, sigma=cfg["hcdf_smoothing_sigma_frames"], axis=1)
    hcdf = np.linalg.norm(smooth[:, 2:] - smooth[:, :-2], axis=0) if smooth.shape[1] > 2 else np.zeros(1)
    feats["harmonic_change_mean"] = float(hcdf.mean())
    feats["harmonic_change_std"] = float(hcdf.std())
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
    print(f"Extracting tonal features for {len(jobs)} files with n_jobs={args.n_jobs}")
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


if __name__ == "__main__":
    main()
