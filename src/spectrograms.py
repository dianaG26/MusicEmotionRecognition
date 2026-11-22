import argparse
import json
import platform
import sys
import time
import warnings
from pathlib import Path

import librosa
import matplotlib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from numpy.lib.format import open_memmap

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = PROJECT_ROOT / "data" / "raw" / "MERGE_Audio_Balanced"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "spectrograms"

CONFIG = {
    "sample_rate": 22050,
    "mono": True,
    "duration_seconds": 30.0,
    "pad": "zeros at the end for clips shorter than 30 s",
    "n_fft": 2048,
    "hop_length": 512,
    "n_mels": 128,
    "fmin": 0,
    "fmax": 11025,
    "power": 2.0,
    "db_reference": 1.0,
    "db_amin": 1e-10,
    "top_db": None,
    "dtype": "float16",
}


def n_frames(cfg):
    return 1 + int(cfg["sample_rate"] * cfg["duration_seconds"]) // cfg["hop_length"]


def mel(path, cfg):
    y, sr = librosa.load(path, sr=cfg["sample_rate"], mono=cfg["mono"], duration=cfg["duration_seconds"])
    length = int(sr * cfg["duration_seconds"])
    loaded = len(y) / sr
    y = np.pad(y, (0, max(0, length - len(y))))[:length]
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=cfg["n_fft"], hop_length=cfg["hop_length"], n_mels=cfg["n_mels"],
                                       fmin=cfg["fmin"], fmax=cfg["fmax"], power=cfg["power"])
    S = librosa.power_to_db(S, ref=cfg["db_reference"], amin=cfg["db_amin"], top_db=cfg["top_db"])
    return S.astype(cfg["dtype"]), loaded


def process(i, song, quadrant, path, cfg, png_dir):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            S, loaded = mel(path, cfg)
            if png_dir:
                save_png(S, f"{song} ({quadrant})", png_dir / f"{song}.png", cfg)
        return i, song, S, loaded, None
    except Exception as e:
        return i, song, None, None, repr(e)


def save_png(S, title, path, cfg):
    import librosa.display
    fig, ax = plt.subplots(figsize=(8, 3))
    img = librosa.display.specshow(S.astype(np.float32), sr=cfg["sample_rate"], hop_length=cfg["hop_length"],
                                   x_axis="time", y_axis="mel", fmax=cfg["fmax"], ax=ax, cmap="magma")
    fig.colorbar(img, ax=ax, format="%+.0f dB")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--png", action="store_true")
    args = parser.parse_args()

    meta = pd.read_csv(args.dataset / "merge_audio_balanced_metadata.csv")[["Song", "Quadrant"]]
    meta = meta.sort_values("Song").reset_index(drop=True)
    if args.limit:
        meta = meta.groupby("Quadrant", group_keys=False).head(max(1, args.limit // 4)).sort_values("Song").reset_index(drop=True)

    name = f"mel{CONFIG['n_mels']}"
    args.output.mkdir(parents=True, exist_ok=True)
    shape = (len(meta), CONFIG["n_mels"], n_frames(CONFIG))
    store = open_memmap(args.output / f"{name}.npy", mode="w+", dtype=CONFIG["dtype"], shape=shape)
    png_dir = args.output / "png" if args.png else None
    if png_dir:
        png_dir.mkdir(exist_ok=True)

    jobs = [(i, r.Song, r.Quadrant, args.dataset / r.Quadrant / f"{r.Song}.mp3") for i, r in enumerate(meta.itertuples())]
    print(f"Computing {shape[1]}x{shape[2]} Mel-spectrograms for {len(jobs)} files with n_jobs={args.n_jobs}")
    start = time.time()
    loaded, errors = np.full(len(meta), np.nan), []
    for i, song, S, sec, err in Parallel(n_jobs=args.n_jobs, verbose=5, return_as="generator")(delayed(process)(i, s, q, p, CONFIG, png_dir) for i, s, q, p in jobs):
        if err:
            errors.append({"Song": song, "error": err})
            continue
        store[i] = S
        loaded[i] = sec
    store.flush()
    elapsed = time.time() - start

    index = meta.assign(row=range(len(meta)), duration_loaded=loaded, ok=~np.isnan(loaded))
    index.to_csv(args.output / f"{name}_index.csv", index=False)
    if errors:
        pd.DataFrame(errors).to_csv(args.output / f"{name}_errors.csv", index=False)

    values = np.asarray(store[index["ok"].to_numpy()], dtype=np.float32)
    info = {
        "config": CONFIG,
        "file": f"{name}.npy",
        "shape": list(shape),
        "axes": ["song (row in index csv)", "mel band (low to high frequency)", "time frame"],
        "seconds_per_frame": CONFIG["hop_length"] / CONFIG["sample_rate"],
        "dataset": str(args.dataset),
        "n_files": len(jobs),
        "n_computed": int(index["ok"].sum()),
        "n_errors": len(errors),
        "n_padded": int((index["duration_loaded"] < CONFIG["duration_seconds"] - 0.05).sum()),
        "db_min": float(values.min()),
        "db_max": float(values.max()),
        "db_mean": float(values.mean()),
        "png_saved": args.png,
        "size_mb": round((args.output / f"{name}.npy").stat().st_size / 1e6, 1),
        "elapsed_seconds": round(elapsed, 1),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "librosa": librosa.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    (args.output / f"{name}.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"Done in {elapsed / 60:.1f} min: {info['n_computed']} spectrograms, {len(errors)} errors, {info['size_mb']} MB")


if __name__ == "__main__":
    main()
