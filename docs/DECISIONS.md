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
| D16 | Mel-spectrogram settings and storage for the CNN | decided (storage format to confirm with coordinator) | `src/spectrograms.py`; `data/spectrograms/mel128.json` |
| D17 | CNN architecture, training procedure and choice of variant on validation | decided | `src/train_cnn.py`; `results/cnn/`; `notebooks/06_cnn_vs_classic.ipynb` |
| D18 | Final deep model: CNN alone, 6 crops; fusion with the SVM and more crops tried and not adopted | decided | `src/fusion.py`; `notebooks/07_fusion.ipynb` |
| D19 | Sad vs calm: two-stage model and pretrained PANNs embeddings; final model = CNN fused with an SVM on PANNs + hand-crafted features | decided | `src/two_stage.py`; `src/panns_embeddings.py`; `src/pretrained_experiments.py`; `notebooks/08_sad_vs_calm.ipynb` |

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

## D16 — Mel-spectrograms for the CNN

**Decision.** `src/spectrograms.py` computes one log-Mel spectrogram per clip and stores all of them once, so that every CNN experiment reads the same inputs:
- audio loaded exactly as for the hand-crafted features (D09): 22,050 Hz, mono, first 30 s; clips shorter than 30 s are padded with silence at the end (D04);
- STFT with `n_fft` 2048 and hop 512 (as in D09), 128 Mel bands from 0 to 11,025 Hz, power spectrogram converted to decibels relative to a fixed reference (1.0), floor −100 dB, no per-clip normalisation;
- result: 128 bands × 1,292 frames per clip (≈ 23 ms per frame), stored as `float16` in one array `data/spectrograms/mel128.npy` (3,232 × 128 × 1,292, ≈ 1.07 GB), with `mel128_index.csv` mapping each row to its song and quadrant and `mel128.json` recording the settings, versions and value range;
- optionally (`--png`) every spectrogram is also saved as an image in `data/spectrograms/png/` for visual inspection.

**Reasons.** Using the same sampling rate and STFT as the hand-crafted features means the CNN and the classical models see the same audio, so the comparison between them is about the model and the representation, not about preprocessing. 128 Mel bands is the usual resolution for music CNNs on spectrograms. A fixed dB reference keeps loudness differences between songs, which relate to arousal; normalising each clip to its own maximum would remove them. Input standardisation is done later, in training, with statistics computed on the training set only (as in D07). `float16` halves the storage; its resolution at these magnitudes (about 0.03 dB) is far below any audible or learnable difference. Storing all spectrograms once avoids recomputing them for every epoch and every experiment, and the saved arrays (and images) can be shown to the coordinator, who asked for the inputs to be saved.

**Status.** Settings decided; whether the coordinator also wants the images for all clips or only examples is to be confirmed.

**Evidence.** `src/spectrograms.py`; `data/spectrograms/mel128.json`; tested on six clips, including the two shortest (4.8 s and 10.1 s), before the full run.

## D17 — CNN: architecture, training and choice of variant

**Decision.** The deep-learning model is a small convolutional network trained from scratch on the Mel-spectrograms of D16 (`src/train_cnn.py`):
- four blocks of 3×3 convolutions with batch normalisation and ReLU, each followed by 2×2 max-pooling (16, 32, 64, 128 channels), then global average pooling, dropout 0.3 and a linear layer to the four quadrants;
- input standardised per Mel band with the mean and standard deviation of the training set only;
- training on random 10-second crops of each clip (a new crop every epoch), cross-entropy loss, AdamW (learning rate 0.001, weight decay 0.0001), batch size 32;
- after each epoch, validation macro-F1 is computed on the whole clip (average of the class probabilities of five overlapping 10-second crops); the learning rate is halved after 4 epochs without improvement, training stops after 12 epochs without improvement or at 60 epochs, and the weights of the best validation epoch are kept;
- the test set is evaluated once, with those weights.

Three variants were trained on protocol A and compared **on validation**:

| variant | convolutions per block | augmentation | parameters | best validation macro-F1 | seconds per epoch |
|---|---|---|---|---|---|
| `small` | 2 | none | 294,228 | 0.723 | 135 |
| `fast` | 1 | none | 97,908 | 0.702 | 65 |
| `fast_aug` | 1 | frequency and time masking, ±6 dB gain | 97,908 | 0.688 | 66 |

`small` has the highest validation score and is the variant evaluated on the five artist-grouped splits (D12) and compared with the classical models.

**Reasons.** The audio clips are short (30 s) and the dataset small (2,264 training clips), so a compact network trained from scratch is appropriate and can be trained on the laptop's CPU. Random crops act as data augmentation and let the network see different parts of each song; averaging several crops at evaluation uses the whole clip, as the classical features do. Selecting the epoch and the variant on validation keeps the test set untouched, as in D10. Frequency/time masking and gain changes did not improve validation macro-F1 for the smaller network, so augmentation beyond random cropping is not used. Validation scores fluctuate by several hundredths between epochs, so differences of about 0.02 between variants are reported as indicative, not conclusive; their test scores (0.682, 0.686, 0.669) were not used for the choice.

**Outcome on unseen artists.** On the five artist-grouped splits the `small` CNN reaches a test macro-F1 of 0.694 ± 0.018, against 0.662 ± 0.013 for the SVM and 0.663 ± 0.014 for the Random Forest. It beats the SVM on all five splits (+0.032 on average) and the Random Forest on four; McNemar's test is significant on two of five splits against the SVM and, pooled over the five splits, p = 0.0045 against the SVM and p = 0.0055 against the Random Forest. The gain comes from the high-arousal quadrants; the sad/calm confusion is not reduced.

**Evidence.** `src/train_cnn.py`; `results/cnn/20261121_193756_cnn_small_A/`, `results/cnn/20261121_222524_cnn_fast_A/`, `results/cnn/20261121_222524_cnn_fast_aug_A/`, `results/cnn/20261122_122104_cnn_small_B42/` … `_B46/` (`history.csv`, `metrics.json`); `notebooks/06_cnn_vs_classic.ipynb`; `results/comparison/`.

## D18 — Final deep-learning model: CNN alone; fusion and more crops not adopted

**Decision.** The final deep-learning model is the `small` CNN of D17, evaluated by averaging its probabilities over six 10-second crops per clip. Two extensions were tested and are not used: evaluating the CNN on nine crops instead of six, and combining its probabilities with those of the SVM.

**What was tested.** `src/fusion.py` reuses the saved CNNs and the SVM hyperparameters of `experiments_classic.py` and trains nothing new. For each split (A and B42–B46):
- the CNN is evaluated with crops every 5 s (6 crops) and every 2.5 s (9 crops);
- the SVM's decision scores are turned into class probabilities by sigmoid calibration fitted with 5-fold cross-validation on the training set;
- the fusion is `w · p_CNN + (1 − w) · p_SVM`, with `w` from 0 to 1 in steps of 0.1;
- the number of crops and `w` are chosen on the validation set, ties going to fewer crops and to the CNN alone; the test set is evaluated once. A fixed 50/50 fusion is also reported.

**Results.** More crops were preferred only on B42 and changed the mean test macro-F1 on protocol B from 0.6939 to 0.6942. The weighted fusion was chosen on two of five artist splits (`w` = 0.8 and 0.9) and gave 0.699 ± 0.024 against 0.694 ± 0.018 for the CNN alone, a difference that is not significant (pooled McNemar p = 0.17). The 50/50 fusion gave 0.690 ± 0.008: slightly lower on average, but with less than half the variation between splits and fewer sad/calm confusions.

**Reasons.** Neither extension gives a gain that is significant or consistent across splits, so the simpler model is kept: it is easier to explain, and each prediction of a fused model would need both the spectrogram and the 155 hand-crafted features and two models instead of one. The complementarity of the two models (26 % of songs classified correctly by only one of them, D17 outcome) does not translate into a gain because, when only the CNN is right, the SVM is usually the more confident of the two, so a fusion that rescues the SVM's correct answers loses part of the CNN's. The stability of the 50/50 fusion is reported as a direction for future work.

**Evidence.** `src/fusion.py`; `results/fusion/` (`runs.csv`, `validation_grid.csv`, `test_predictions.csv`); `notebooks/07_fusion.ipynb`.

## D19 — Reducing the sad/calm confusion: two-stage model and pretrained embeddings

**Context.** Every model confuses Q3 (sad) and Q4 (calm): with the CNN, 37 % of sad songs are predicted as calm and 29 % of calm songs as sad, and the average spectra of the two quadrants differ by 0.4 dB (`notebooks/05_spectrograms.ipynb`). The coordinator asked for an attempt to reduce this confusion. Lyrics, which help most for these quadrants in the MERGE paper, are not part of MERGE Audio Balanced and would have to be collected from lyrics websites, so the attempt stays audio-only.

**Step 1 — two-stage model on our features (`src/two_stage.py`).** The CNN decides high vs low arousal (P(Q1)+P(Q2) against P(Q3)+P(Q4)); for low-arousal clips a binary specialist trained only on the Q3/Q4 training songs decides sad vs calm. Candidates: logistic regression, SVM or Random Forest on the top 20/40/80/155 features ranked by Kruskal–Wallis ε² between Q3 and Q4 on the training set, combined with the CNN's Q3/Q4 ratio by a weight from 0 to 1; the CNN alone or the best candidate is chosen on validation. **Result:** no change on unseen artists (macro-F1 0.694, 75.6 sad/calm confusions per split, as for the CNN). The validation gains of 0.01–0.03 did not carry over to the test sets.

**Step 2 — pretrained embeddings (`src/panns_embeddings.py`, `src/pretrained_experiments.py`).** PANNs Cnn14 (Kong et al., 2020, pretrained on AudioSet) gives a 2,048-dimensional embedding per clip (32 kHz mono, first 30 s), extracted once (48 minutes on the laptop) and stored in `data/embeddings/`. The model checkpoint is downloaded from Zenodo into `models/pretrained/`. On each split, with all choices on validation: a four-class SVM or logistic regression on the embedding alone or with the 155 features; its probabilities (sigmoid-calibrated) fused with the CNN's; and the two-stage model of step 1 with a specialist on the embedding.

**Results on the five artist splits (mean).**

| model | macro-F1 | F1 Q3 | F1 Q4 | sad/calm confusions per split |
|---|---|---|---|---|
| CNN (D17) | 0.694 ± 0.018 | 0.558 | 0.567 | 75.6 |
| two-stage, our features | 0.694 ± 0.018 | 0.563 | 0.574 | 75.6 |
| SVM on PANNs + 155 features | 0.711 ± 0.020 | 0.585 | 0.592 | 68.8 |
| **CNN + (SVM on PANNs + 155 features)** | **0.713 ± 0.023** | 0.592 | 0.582 | 69.4 |
| two-stage, PANNs specialist | 0.709 ± 0.024 | **0.607** | **0.597** | **68.0** |

**Decision.** The final model is the fusion of the CNN with the SVM on PANNs embedding + 155 features: it had the highest validation macro-F1 on all six splits, improves on the CNN on unseen artists (+0.019; pooled McNemar p = 0.027) and reaches 0.726 on the official split (published audio-only baseline: 68.52 %). The two-stage model with a PANNs specialist is reported as the model aimed at the sad/calm pair: it gives the largest and the only clearly significant sad/calm improvement (on the Q3/Q4 test songs, 133 songs right where the CNN is wrong against 75, p = 0.0001), but it was not the best variant on validation and is therefore not the selected model.

**Reasons.** The negative result of step 1 shows that the limitation lies in the information available to the models, not in their structure; a representation learned from about two million audio clips supplies cues that 2,264 training songs cannot. Selecting the final model by validation macro-F1, as in D10, keeps the test sets untouched; the sad/calm-specific result is reported separately and labelled as measured on test. The CNN (D17) and the SVM on hand-crafted features (D15) remain the reference models of the two families compared in the thesis; the pretrained model is reported as an extension.

**Evidence.** `src/two_stage.py`, `results/two_stage/`; `src/panns_embeddings.py`, `data/embeddings/panns_cnn14.json`; `src/pretrained_experiments.py`, `results/pretrained/`; `notebooks/08_sad_vs_calm.ipynb`, `results/pretrained/figures/`.
