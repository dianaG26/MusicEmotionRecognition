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
| D06 | Evaluate on the official split and on an artist-grouped split | decided (approved by coordinator, 13 Nov) | `notebooks/01_eda.ipynb` §5, §6; `results/splits/artist_70_15_15_report.md` |
| D07 | Standardise features for SVM, fitted on training data only | decided | `notebooks/01_eda.ipynb` §7 |
| D08 | Keep all 133 features, without feature selection | decided | `notebooks/01_eda.ipynb` §8, §9, §11 |
| D09 | Hand-crafted feature set and extraction settings | decided | `src/features.py`; `data/features/features_librosa.json` |
| D10 | Model selection on validation, single evaluation on test, macro-F1 as main metric | decided | `src/train_classic.py`, `src/experiments_classic.py` |
| D11 | Verify the downloaded archive against the Zenodo checksum | pending | `docs/data_check_report.md` |
| D12 | Repeat protocol B with five random seeds (42–46) | decided | `notebooks/02_split_analysis.ipynb` §3–§5; `notebooks/04_classic_experiments.ipynb` §2 |
| D13 | Add tonal and harmonic (valence-related) features and evaluate models with and without them | decided | `notebooks/03_valence_features.ipynb`; `notebooks/04_classic_experiments.ipynb` §2, §7 |
| D14 | Report chance level and a linear reference, and test differences with McNemar's test | decided | `notebooks/04_classic_experiments.ipynb` §1, §3 |
| D15 | No ensemble; SVM and Random Forest kept as equal classical references, SVM used in the interface | decided | `notebooks/04_classic_experiments.ipynb` §3, §8, §11 |

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

**Reasons.** The labels were assigned by the dataset authors from the majority of each song's AllMusic emotion tags and validated by human annotators; the arousal and valence values were computed separately from the same tags and were not manually validated (Louro et al., 2026). For 100% of samples the label agrees with the AV values when each axis is split at 0.5, the centre of the [0, 1] scale used in the released files (0 on the article's [-1, 1] scale), so labels and AV values are consistent. The 0.5 threshold is our choice for this check, not a parameter of the dataset. About 25% of samples lie within 0.05 of a boundary, with the largest share in Q2. Removing them would make the task artificially easier and the results incomparable with the literature; it is instead reported as a limitation.

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

**Evidence.** `src/train_classic.py`, `src/experiments_classic.py`; Louro et al., 2026.

## D11 — Archive integrity check

**Decision.** The MD5 checksum of the local archive (`bbca042c520fd16bca74b572ac6c88e3`) is compared with the checksum published on Zenodo.

**Status.** Pending: the Zenodo checksum has not been compared yet.

**Evidence.** `docs/data_check_report.md`.

## D12 — Repeat protocol B with several random seeds

**Decision.** Protocol B is generated with five random seeds (42, 43, 44, 45, 46) by `python src/splits.py --seed 42 43 44 45 46`. Seed 42 is the original split (`results/splits/artist_70_15_15/`); the others are saved as `results/splits/artist_70_15_15_seed<N>/`, each with its own verification report. Every model is evaluated on all five, and the result of protocol B is reported as the mean and standard deviation across seeds.

**Reasons.** The test sets of the two protocols are comparable in class balance, emotion labels (Kolmogorov–Smirnov p ≈ 0.93 for arousal, 0.56 for valence), share of samples near a quadrant boundary (24% and 22%) and the ten most discriminative features (all |SMD| < 0.1). They differ, as intended, in artist overlap (66% against 0% of test songs with an artist seen in training). They also differ moderately in genre mix (chi-square p ≈ 0.006, total variation distance ≈ 0.14; B's test set has more Pop/Rock), because grouping by artist moves genres in blocks. With a single artist-grouped split, part of the difference between A and B could come from this particular genre mix. Averaging over several seeds averages over different genre mixes and also shows how much the result of protocol B varies from one split to another.

**Outcome.** The standard deviation of test macro-F1 across the five seeds is between 0.005 and 0.014 for every classical model, so the ranking of the models does not depend on which artists are tested, and differences smaller than about 0.015 between two models cannot be interpreted from a single split.

**Evidence.** `notebooks/04_classic_experiments.ipynb` §2; `results/experiments/runs.csv`, `results/experiments/figures/02_seed_stability.png`; `notebooks/02_split_analysis.ipynb` §3–§5; `results/splits/02_genre_share_test_sets.png`, `results/splits/02_labels_test_sets.png`, `results/splits/02_feature_shift_test_sets.png`, `results/splits/02_split_comparison_summary.json`.

## D13 — Tonal and harmonic features for valence

**Decision.** 22 tonal and harmonic features are extracted by `src/features_tonal.py`: modality (major vs minor), key clarity, best major and minor key strength, modality and key clarity on 3-second segments (mean and standard deviation), the 6-dimensional tonal centroid (tonnetz, mean and standard deviation) and the harmonic change detection function (mean and standard deviation). Every classical model is trained twice, with the 133 librosa features and with the 133 + 22 features, under both protocols. The effect is judged on macro-F1 and on the number of Q3/Q4 confusions in the test confusion matrix. The 22 features are kept as one block; none is selected or removed on the basis of the results.

**Reasons.** The 133 librosa features barely separate Q3 (sad) from Q4 (calm): best ε² ≈ 0.025, against ≈ 0.51 for high vs low arousal (`01_eda.ipynb` §8). The survey by Panda, Malheiro and Paiva (IEEE TAFFC, 2023) links major mode and consonant harmony to positive valence and minor mode and dissonant harmony to negative valence, and lists modality, key strength, key clarity, tonal centroid and harmonic change among the computational harmony features. On the training set, the new features point in the predicted direction (Q4 has the most major modality, Q2 the only minor-leaning mean) and the modality features carry information the existing features do not (strongest |ρ| with any existing feature below 0.3). On their own they do not separate Q3 from Q4 better than the existing features (best ε² ≈ 0.020), so their usefulness can only be judged by training with and without them.

**Evidence.** `notebooks/03_valence_features.ipynb`; `results/valence/02_tonality_boxplots.png`, `results/valence/03_q3_vs_q4_top_features.png`, `results/valence/05_redundancy_with_existing.png`, `results/valence/summary.json`; extraction settings in `data/features/features_tonal.json`. Original sampling rate of the MERGE audio files checked on four files: 32,000 Hz, stereo, resampled to 22,050 Hz mono by both extraction scripts.

**Outcome.** With the same test songs, adding the 22 features improves the SVM on 5 of 5 artist splits (mean +0.017) and the Random Forest on 4 of 5 (mean +0.013), and both on the official split. The gain is consistent but too small to be significant on any single test set (McNemar's test), and the ablation shows that it depends on the model. All final classical models therefore use the 155 features. Evidence: `notebooks/04_classic_experiments.ipynb` §2, §3, §7; `results/experiments/figures/02_tonal_minus_base_by_seed.csv`, `results/experiments/figures/07_ablation.csv`.

## D14 — Reference models and statistical testing

**Decision.** Every comparison includes two reference models: a chance-level classifier (`DummyClassifier`, stratified) and multinomial logistic regression on standardised features. Two models evaluated on the same test songs are compared with the exact McNemar test, separately on each of the six test sets (protocol A and five protocol-B seeds). A difference is called significant only if it is significant on several test sets in the same direction.

**Reasons.** A score is only meaningful relative to what is trivially achievable (chance) and what a simple model achieves (linear boundary). Differences of 0.01–0.02 in macro-F1 between models are of the same size as the variation between splits (D12), so they must be tested rather than read off a table. McNemar's test uses the songs on which two models disagree, needs no assumption about score distributions and is the standard test for comparing two classifiers on one test set (Dietterich, *Approximate Statistical Tests for Comparing Supervised Classification Learning Algorithms*, Neural Computation, 1998).

**Outcome.** Chance level is ≈ 0.26 macro-F1 on protocol B; logistic regression reaches ≈ 0.65 and SVM, Random Forest and the vote ≈ 0.66. No difference between SVM and Random Forest is significant on any test set.

**Evidence.** `src/experiments_classic.py`; `notebooks/04_classic_experiments.ipynb` §1, §3; `results/experiments/figures/01_main_table.csv`, `results/experiments/figures/03_mcnemar.csv`, `results/experiments/figures/03_mcnemar_summary.csv`.

## D15 — Final classical models: no ensemble, SVM in the interface

**Decision.** A majority vote of SVM, Random Forest and logistic regression was evaluated and is not used. SVM and Random Forest are both reported as classical references for the CNN. The SVM is the classical model offered in the interface.

**Reasons.** On validation, where the choice must be made, the vote beats the best single model in only 6 of 12 configurations, and its test scores are similar to those of the best single model. SVM and Random Forest are statistically indistinguishable (D14), so neither can be called better; the SVM is chosen for the interface because its saved model is about ten times smaller (≈ 2 MB against ≈ 28 MB) and trains in under a second, with the same accuracy.

**Evidence.** `notebooks/04_classic_experiments.ipynb` §8, §11; `results/experiments/ensemble_validation.csv`, `results/experiments/figures/08_ensemble_validation.csv`, `results/experiments/figures/11_cost.csv`.
