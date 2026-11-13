# Decision Log

Every methodological decision of the project is recorded here with the evidence that supports it. Evidence is always a file in this repository (code, notebook section, generated figure or table) or a cited publication, so that each decision can be traced back to the analysis that produced it.

Status values: **decided** — supported by evidence and applied; **pending** — awaiting an analysis or the coordinator's confirmation.

| ID | Decision | Status | Main evidence |
|---|---|---|---|
| D01 | Use MERGE Audio Balanced as the dataset | decided (to confirm with coordinator) | Literature review; `docs/data_check_report.md` |
| D02 | No class rebalancing | decided | `notebooks/01_eda.ipynb` §1 |
| D03 | Use the quadrant labels as provided | decided | `notebooks/01_eda.ipynb` §2 |
| D04 | Keep all clips, including the very short ones | decided | `notebooks/01_eda.ipynb` §3, §10 |
| D05 | Keep duplicate recordings in the official protocol | decided | `notebooks/01_eda.ipynb` §4 |
| D06 | Evaluate on the official split and on an artist-grouped split | decided (to confirm with coordinator) | `notebooks/01_eda.ipynb` §5, §6; `results/splits/artist_70_15_15_report.md` |
| D07 | Standardise features for SVM, fitted on training data only | decided | `notebooks/01_eda.ipynb` §7 |
| D08 | Keep all 133 features, without feature selection | decided | `notebooks/01_eda.ipynb` §8, §9, §11 |
| D09 | Hand-crafted feature set and extraction settings | decided | `src/features.py`; `data/features/features_librosa.json` |
| D10 | Model selection on validation, single evaluation on test, macro-F1 as main metric | decided | `src/train_classic.py` (planned) |
| D11 | Verify the downloaded archive against the Zenodo checksum | pending | `docs/data_check_report.md` |

---

## D01 — Dataset: MERGE Audio Balanced

**Decision.** All experiments use MERGE Audio Balanced (Zenodo record 13939205, version 1.1.0): 3,232 audio excerpts of 30 seconds, 808 per Russell quadrant.

**Reasons.**
- It is labelled directly with the four Russell quadrants, which is the task of this thesis. DEAM, the main alternative, provides continuous, time-varying annotations designed for dynamic emotion recognition, and would require an arbitrary threshold to obtain quadrants.
- It provides official train/validation/test splits and published audio-only baselines, so the results can be compared with the literature.
- It is balanced across quadrants and includes artist and genre metadata, which make the analyses in D06 possible.

**Evidence.** Louro et al., *MERGE: A Bimodal Audio-Lyrics Dataset for Static Music Emotion Recognition*, IEEE TAFFC, 2026, DOI 10.1109/TAFFC.2026.3668037; Aljanaki et al., *Developing a benchmark for emotional analysis of music*, PLOS ONE, 2017, DOI 10.1371/journal.pone.0173392; integrity of the local copy in `docs/data_check_report.md` (all 3,232 audio files present and correctly labelled).

## D02 — No class rebalancing

**Decision.** No oversampling, undersampling or class weighting is applied.

**Reason.** The dataset and the official split are perfectly balanced (808 per quadrant; 566/121/121 per quadrant in train/validation/test).

**Evidence.** `notebooks/01_eda.ipynb` §1; `results/eda/01_composition.csv`, `results/eda/01_official_split_counts.csv`.

## D03 — Quadrant labels used as provided

**Decision.** The `Quadrant` column of the metadata is the target. No relabelling, and no samples are removed for being close to a quadrant boundary.

**Reasons.** The label equals the arousal and valence values thresholded at 0.5 for 100% of samples, so it is consistent with the continuous annotations. About 25% of samples lie within 0.05 of a boundary, with the largest share in Q2. Removing them would make the task artificially easier and the results incomparable with the literature; it is instead reported as a limitation.

**Evidence.** `notebooks/01_eda.ipynb` §2; `results/eda/02_valence_arousal_scatter.png`, `results/eda/02_arousal_valence_histograms.png`; `results/eda/summary.json` (`label_av_agreement`, `share_near_boundary_0.05`).

## D04 — Keep all clips, including the very short ones

**Decision.** No clips are removed. The four clips shorter than 15 s are kept for the feature-based models and will be padded to a fixed length for the CNN.

**Reasons.** 72 clips are shorter than 29 s, four of them much shorter (4.8 s and 3 × 10.1 s). Their aggregated features are not outliers: no training clip has ten or more extreme feature values. Removing them would change the official split.

**Evidence.** `notebooks/01_eda.ipynb` §3, §10; `results/eda/03_clip_duration.png`, `results/eda/03_very_short_clips.csv`, `results/eda/10_outliers.png`, `results/eda/10_outlier_clips_train.csv`.

## D05 — Duplicate recordings kept in the official protocol

**Decision.** The six songs stored under two IDs are kept in the official protocol and reported. In the artist-grouped protocol they are handled automatically, because both copies share the artist.

**Reasons.** No duplicate pair is divided between training and test in the official split, so the test score is not affected. Three pairs are divided between training and validation and one between test and validation, which can only slightly favour hyperparameter selection. Keeping them preserves comparability with published results on the same split.

**Evidence.** `notebooks/01_eda.ipynb` §4; `results/eda/04_duplicates.csv`.

## D06 — Two evaluation protocols: official split and artist-grouped split

**Decision.** Every model is evaluated twice:
- **Protocol A** — official 70-15-15 split, for comparison with the literature.
- **Protocol B** — artist-grouped split with the same proportions and balanced quadrants, in which every artist appears in only one of training, validation and test, generated by `src/splits.py` (StratifiedGroupKFold grouped by artist, stratified by quadrant, seed 42).

The difference between the two scores measures how much of a model's performance depends on recognising artists rather than emotion.

**Reasons.** 85% of the songs of multi-song artists fall in the artist's most frequent quadrant, and genre is strongly associated with quadrant (Cramér's V ≈ 0.38). In the official split, 66% of test songs come from artists also present in training. A model can therefore score well by recognising an artist's or genre's sound, which overestimates performance on music by new artists.

**Evidence.** `notebooks/01_eda.ipynb` §5, §6; `results/eda/05_top_artists_by_quadrant.png`, `results/eda/05_artist_overlap_official_split.csv`, `results/eda/06_genre_by_quadrant.png`. The generated split and its verification (no artist shared between sets, quadrant balance, comparison with the official split) are in `results/splits/artist_70_15_15_report.md` and `results/splits/artist_70_15_15/`.

## D07 — Feature standardisation for SVM

**Decision.** For the SVM, features are standardised (zero mean, unit variance) by a `StandardScaler` placed inside a scikit-learn `Pipeline`, so that it is fitted on the training set only. Random Forest uses the raw features.

**Reasons.** The scales of the features differ by about five orders of magnitude; an RBF-kernel SVM depends on distances and would be dominated by the largest-scale features. Fitting the scaler on training data only prevents information from the validation and test sets from leaking into the model. Decision trees split on thresholds and are insensitive to feature scale.

**Evidence.** `notebooks/01_eda.ipynb` §7; `results/eda/07_feature_description_train.csv`.

## D08 — All 133 features kept, no feature selection

**Decision.** The models receive all 133 features.

**Reasons.** 120 of 133 features differ significantly across quadrants. Only ten pairs are strongly correlated (|ρ| > 0.9), which neither an RBF SVM nor a Random Forest is sensitive to. Univariate selection would remove features that are weak on their own but useful for the hardest distinction, Q3 versus Q4, where even the best single feature has a very small effect (ε² ≈ 0.03 compared with ≈ 0.51 for arousal). The PCA projection shows strongly overlapping classes, so non-linear models are used. Correlated features are taken into account when interpreting Random Forest importances.

**Evidence.** `notebooks/01_eda.ipynb` §8, §9, §11; `results/eda/08_feature_ranking_kruskal_train.csv`, `results/eda/08_arousal_vs_q3q4_separation.csv`, `results/eda/08_feature_boxplots.png`, `results/eda/09_strongly_correlated_pairs.csv`, `results/eda/11_pca.png`.

## D09 — Hand-crafted feature set

**Decision.** 133 features are extracted with librosa from the first 30 s of each clip (mono, 22,050 Hz, frames of 2,048 samples with hop 512) and aggregated over time by mean and standard deviation: 20 MFCCs and their deltas, spectral centroid, bandwidth, rolloff, flatness and contrast, zero-crossing rate, RMS energy, 12 chroma bins, onset strength and tempo.

**Reasons.** These are the standard emotionally relevant audio descriptors in the MER literature: timbre (MFCC, spectral shape), dynamics (RMS), harmony (chroma) and rhythm (onset strength, tempo). Mean and standard deviation give every clip a fixed-length vector, as required by SVM and Random Forest.

**Evidence.** `src/features.py`; settings and library versions in `data/features/features_librosa.json`; Panda, Malheiro and Paiva, *Audio Features for Music Emotion Recognition: A Survey*, IEEE TAFFC, 2023, DOI 10.1109/TAFFC.2020.3032373.

## D10 — Evaluation protocol and metric

**Decision.** Hyperparameters are selected by macro-F1 on the validation set; the final model is trained on the training set and evaluated once on the test set. Macro-F1 is the main metric, reported with accuracy, per-quadrant precision, recall and F1, and the confusion matrix. Random seeds are fixed.

**Reasons.** Using the test set for any choice would make the reported score optimistic. Macro-F1 weights every quadrant equally and is the metric used in the MERGE publications, which allows a direct comparison.

**Evidence.** `src/train_classic.py` (planned); Louro et al., 2026.

## D11 — Archive integrity check

**Decision.** The MD5 checksum of the local archive (`bbca042c520fd16bca74b572ac6c88e3`) is compared with the checksum published on Zenodo.

**Status.** Pending: the Zenodo checksum has not been compared yet.

**Evidence.** `docs/data_check_report.md`.
