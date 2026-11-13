import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import StratifiedGroupKFold

QUADRANTS = ["Q1", "Q2", "Q3", "Q4"]
PARTS = ["train", "validate", "test"]
SEED = 42
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = PROJECT_ROOT / "data" / "raw" / "MERGE_Audio_Balanced"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "splits"
SPLIT_NAME = "artist_70_15_15"


def artist_groups(meta):
    artist = meta["Artist"].fillna("").str.strip().str.casefold()
    return np.where(artist == "", "unknown::" + meta["Song"], artist)


def pick_fold(frame, n_splits, seed):
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    _, held_out = next(cv.split(frame, frame["Quadrant"], frame["group"]))
    return frame.iloc[held_out].index


def load_official(dataset):
    split_dir = dataset / "tvt_dataframes" / "tvt_70_15_15"
    frames = [pd.read_csv(split_dir / f"tvt_70_15_15_{p}_audio_balanced.csv").assign(Split=p) for p in PARTS]
    return pd.concat(frames, ignore_index=True)[["Song", "Split"]]


def describe(frame, label):
    counts = pd.crosstab(frame["Split"], frame["Quadrant"]).reindex(index=PARTS, columns=QUADRANTS, fill_value=0)
    counts["total"] = counts.sum(axis=1)
    counts["share"] = (counts["total"] / counts["total"].sum()).round(3)
    groups = {p: set(frame.loc[frame["Split"] == p, "group"]) for p in PARTS}
    shared = {f"{a} & {b}": len(groups[a] & groups[b]) for i, a in enumerate(PARTS) for b in PARTS[i + 1:]}
    train_groups = groups["train"]
    seen = {
        p: round(float(frame.loc[frame["Split"] == p, "group"].isin(train_groups).mean()), 4)
        for p in ["validate", "test"]
    }
    dup = frame[frame.duplicated(["Artist", "Title"], keep=False)]
    split_pairs = int((dup.groupby(["Artist", "Title"])["Split"].nunique() > 1).sum())
    return {
        "label": label,
        "counts": counts,
        "artists_shared_between_sets": shared,
        "share_of_songs_with_artist_in_train": seen,
        "duplicate_pairs_split_across_sets": split_pairs,
    }


def to_markdown(stats):
    lines = [f"### {stats['label']}", "", "| set | " + " | ".join(QUADRANTS) + " | total | share |", "|---|" + "---|" * 6]
    for part, row in stats["counts"].iterrows():
        lines.append(f"| {part} | " + " | ".join(str(int(row[q])) for q in QUADRANTS) + f" | {int(row['total'])} | {row['share']:.1%} |")
    lines.append("")
    lines.append("- artists shared between sets: " + ", ".join(f"{k}: {v}" for k, v in stats["artists_shared_between_sets"].items()))
    lines.append("- songs whose artist also appears in train: " + ", ".join(f"{k}: {v:.1%}" for k, v in stats["share_of_songs_with_artist_in_train"].items()))
    lines.append(f"- duplicate recordings divided between sets: {stats['duplicate_pairs_split_across_sets']}")
    lines.append("")
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    meta = pd.read_csv(args.dataset / "merge_audio_balanced_metadata.csv")[["Song", "Quadrant", "Artist", "Title"]]
    meta["group"] = artist_groups(meta)

    test_idx = pick_fold(meta, n_splits=7, seed=args.seed)
    rest = meta.drop(index=test_idx)
    val_idx = pick_fold(rest, n_splits=6, seed=args.seed)
    meta["Split"] = "train"
    meta.loc[test_idx, "Split"] = "test"
    meta.loc[val_idx, "Split"] = "validate"

    artist_stats = describe(meta, "Artist-grouped split (protocol B)")
    if any(artist_stats["artists_shared_between_sets"].values()):
        raise SystemExit("Artist leakage detected in the generated split")

    official = meta.drop(columns="Split").merge(load_official(args.dataset), on="Song")
    official_stats = describe(official, "Official MERGE split tvt_70_15_15 (protocol A), for comparison")

    split_dir = args.output / SPLIT_NAME
    split_dir.mkdir(parents=True, exist_ok=True)
    for part in PARTS:
        meta.loc[meta["Split"] == part, ["Song", "Quadrant"]].sort_values("Song").to_csv(
            split_dir / f"{SPLIT_NAME}_{part}_audio_balanced.csv", index=False
        )

    info = {
        "split_name": SPLIT_NAME,
        "method": "StratifiedGroupKFold: test = first fold of 7, validation = first fold of 6 on the remainder; groups = artist (case-insensitive), songs without artist form their own group; stratified by quadrant",
        "seed": args.seed,
        "n_songs": int(len(meta)),
        "n_groups": int(meta["group"].nunique()),
        "artist_split": {
            "counts": artist_stats["counts"].to_dict(orient="index"),
            "artists_shared_between_sets": artist_stats["artists_shared_between_sets"],
            "share_of_songs_with_artist_in_train": artist_stats["share_of_songs_with_artist_in_train"],
            "duplicate_pairs_split_across_sets": artist_stats["duplicate_pairs_split_across_sets"],
        },
        "official_split": {
            "artists_shared_between_sets": official_stats["artists_shared_between_sets"],
            "share_of_songs_with_artist_in_train": official_stats["share_of_songs_with_artist_in_train"],
            "duplicate_pairs_split_across_sets": official_stats["duplicate_pairs_split_across_sets"],
        },
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "sklearn": sklearn.__version__,
        "pandas": pd.__version__,
    }
    (args.output / f"{SPLIT_NAME}_report.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

    report = [
        f"# Artist-grouped split `{SPLIT_NAME}`",
        "",
        "Generated by `src/splits.py`. Supports decision D06 in `docs/DECISIONS.md`.",
        "",
        f"Every artist appears in exactly one of train, validation and test. Songs without an artist form their own group. "
        f"The split is stratified by quadrant (StratifiedGroupKFold, seed {args.seed}).",
        "",
        f"Songs: {len(meta)}; artist groups: {meta['group'].nunique()}.",
        "",
    ]
    report += to_markdown(artist_stats) + to_markdown(official_stats)
    shown = split_dir.resolve().relative_to(PROJECT_ROOT) if split_dir.resolve().is_relative_to(PROJECT_ROOT) else split_dir
    report += [f"Split files: `{shown.as_posix()}`", ""]
    (args.output / f"{SPLIT_NAME}_report.md").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))


if __name__ == "__main__":
    main()
