import argparse
import hashlib
import sys
from pathlib import Path

import pandas as pd

QUADRANTS = ["Q1", "Q2", "Q3", "Q4"]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = PROJECT_ROOT / "data" / "raw" / "MERGE_Audio_Balanced"
DEFAULT_REPORT = PROJECT_ROOT / "docs" / "data_check_report.md"

lines = []
problems = []


def out(text=""):
    lines.append(text)
    print(text)


def fail(text):
    problems.append(text)
    out(f"- **PROBLEM:** {text}")


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def quadrant_from_av(arousal, valence, threshold=0.5):
    high_a = arousal >= threshold
    high_v = valence >= threshold
    if high_a and high_v:
        return "Q1"
    if high_a:
        return "Q2"
    if not high_v:
        return "Q3"
    return "Q4"


def check_metadata(root):
    out("## 1. Metadata and AV values")
    meta = pd.read_csv(root / "merge_audio_balanced_metadata.csv")
    av = pd.read_csv(root / "merge_audio_balanced_av_values.csv")
    out(f"- metadata rows: {len(meta)}, columns: {len(meta.columns)}")
    out(f"- av_values rows: {len(av)}, columns: {list(av.columns)}")

    counts = meta["Quadrant"].value_counts().reindex(QUADRANTS, fill_value=0)
    out(f"- samples per quadrant: {counts.to_dict()}")
    if counts.nunique() != 1:
        fail("quadrants are not balanced")

    for name, df in [("metadata", meta), ("av_values", av)]:
        dup = df["Song"].duplicated().sum()
        if dup:
            fail(f"{dup} duplicated Song IDs in {name}")
    if set(meta["Song"]) != set(av["Song"]):
        fail("Song IDs differ between metadata and av_values")
    else:
        out("- metadata and av_values contain the same Song IDs")

    merged = meta.merge(av, on="Song")
    derived = merged.apply(lambda r: quadrant_from_av(r["Arousal"], r["Valence"]), axis=1)
    agreement = (derived == merged["Quadrant"]).mean()
    out(f"- quadrant label vs. AV values thresholded at 0.5: {agreement:.2%} agreement")
    out()
    return meta


def check_audio(root, meta):
    out("## 2. Audio files")
    expected = dict(zip(meta["Song"], meta["Quadrant"]))
    found = {}
    for q in QUADRANTS:
        folder = root / q
        if not folder.exists():
            fail(f"folder {q}/ not found")
            continue
        files = sorted(folder.glob("*.mp3"))
        out(f"- {q}/: {len(files)} mp3 files")
        for f in files:
            found[f.stem] = q
    missing = sorted(set(expected) - set(found))
    extra = sorted(set(found) - set(expected))
    wrong = sorted(s for s in set(expected) & set(found) if expected[s] != found[s])
    out(f"- metadata IDs without audio file: {len(missing)} {missing[:10]}")
    out(f"- audio files without metadata row: {len(extra)} {extra[:10]}")
    out(f"- files stored in a different quadrant folder than their label: {len(wrong)} {wrong[:10]}")
    if missing or extra or wrong:
        fail("audio files do not match metadata")
    out()


def check_duplicates_and_artists(meta):
    out("## 3. Duplicates and artists")
    dup = meta[meta.duplicated(["Artist", "Title"], keep=False)].sort_values(["Artist", "Title"])
    out(f"- rows sharing the same Artist + Title: {len(dup)}")
    for _, r in dup.iterrows():
        out(f"  - {r['Song']} | {r['Quadrant']} | {r['Artist']} - {r['Title']}")
    known = meta.dropna(subset=["Artist"])
    per_artist = known["Artist"].value_counts()
    multi = known[known["Artist"].isin(per_artist[per_artist > 1].index)]
    majority = multi.groupby("Artist")["Quadrant"].agg(lambda s: s.value_counts().iloc[0]).sum()
    out(f"- distinct artists: {known['Artist'].nunique()} (missing artist: {meta['Artist'].isna().sum()})")
    out(f"- artists with more than one song: {len(per_artist[per_artist > 1])}, covering {len(multi)} songs")
    out(f"- share of those songs in their artist's majority quadrant: {majority / len(multi):.2%}")
    out()


def find_song_column(df):
    for col in df.columns:
        if str(col).strip().lower() in {"song", "song_id", "id", "songid"}:
            return col
    return None


def check_splits(root, meta):
    out("## 4. Train / validation / test splits")
    split_root = root / "tvt_dataframes"
    if not split_root.exists():
        fail("tvt_dataframes/ not found")
        return
    labels = dict(zip(meta["Song"], meta["Quadrant"]))
    artists = dict(zip(meta["Song"], meta["Artist"]))
    titles = dict(zip(meta["Song"], meta["Artist"].fillna("") + " - " + meta["Title"].fillna("")))
    for split_dir in sorted(p for p in split_root.iterdir() if p.is_dir()):
        out(f"### {split_dir.name}")
        sets = {}
        for f in sorted(split_dir.glob("*.csv")):
            df = pd.read_csv(f)
            col = find_song_column(df)
            out(f"- {f.name}: {len(df)} rows, columns {list(df.columns)[:8]}")
            if col is None:
                fail(f"{f.name}: no Song ID column recognized, inspect manually")
                continue
            ids = set(df[col].astype(str))
            sets[f.stem] = ids
            unknown = ids - set(labels)
            if unknown:
                fail(f"{f.name}: {len(unknown)} IDs not in metadata")
            dist = pd.Series([labels.get(i) for i in ids]).value_counts().reindex(QUADRANTS, fill_value=0)
            out(f"  - per quadrant: {dist.to_dict()}")
        names = list(sets)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                songs = sets[a] & sets[b]
                art = {artists[s] for s in sets[a] if s in artists} & {artists[s] for s in sets[b] if s in artists}
                art.discard(None)
                same_title = {titles[s] for s in sets[a] if s in titles} & {titles[s] for s in sets[b] if s in titles}
                out(f"- {a} vs {b}: shared IDs {len(songs)}, shared artists {len(art)}, same Artist+Title {len(same_title)}")
                if songs:
                    fail(f"{split_dir.name}: {a} and {b} share {len(songs)} Song IDs")
        out()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--zip", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    out("# MERGE Audio Balanced — data check")
    out(f"- dataset folder: {args.dataset}")
    out(f"- pandas {pd.__version__}, Python {sys.version.split()[0]}")
    if args.zip:
        out(f"- archive: {args.zip.name}, MD5 {md5(args.zip)}")
    out()

    meta = check_metadata(args.dataset)
    check_audio(args.dataset, meta)
    check_duplicates_and_artists(meta)
    check_splits(args.dataset, meta)

    out("## Summary")
    out(f"- problems found: {len(problems)}")
    for p in problems:
        out(f"  - {p}")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
