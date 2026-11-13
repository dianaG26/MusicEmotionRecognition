# MERGE Audio Balanced — data check
- dataset folder: C:\Diana\licenta\MusicEmotionRecognition\data\raw\MERGE_Audio_Balanced
- pandas 3.1.1, Python 3.12.10
- archive: MERGE_Audio_Balanced.zip, MD5 bbca042c520fd16bca74b572ac6c88e3

## 1. Metadata and AV values
- metadata rows: 3232, columns: 26
- av_values rows: 3232, columns: ['Song', 'Arousal', 'Valence']
- samples per quadrant: {'Q1': 808, 'Q2': 808, 'Q3': 808, 'Q4': 808}
- metadata and av_values contain the same Song IDs
- quadrant label vs. AV values thresholded at 0.5: 100.00% agreement

## 2. Audio files
- Q1/: 808 mp3 files
- Q2/: 808 mp3 files
- Q3/: 808 mp3 files
- Q4/: 808 mp3 files
- metadata IDs without audio file: 0 []
- audio files without metadata row: 0 []
- files stored in a different quadrant folder than their label: 0 []

## 3. Duplicates and artists
- rows sharing the same Artist + Title: 12
  - A066-140 | Q4 | Enya - Journey of the Angels
  - MT0009795009 | Q4 | Enya - Journey of the Angels
  - MT0003965796 | Q1 | Gloria Estefan - Rhythm Is Gonna Get You
  - MT0007132421 | Q1 | Gloria Estefan - Rhythm Is Gonna Get You
  - A114-120 | Q1 | Luther Vandross - Never Too Much
  - MT0011933875 | Q1 | Luther Vandross - Never Too Much
  - A105-117 | Q1 | Prince - Kiss
  - MT0002343207 | Q1 | Prince - Kiss
  - A005 | Q1 | Prince - U Got the Look
  - MT0004444570 | Q1 | Prince - U Got the Look
  - A051 | Q2 | Slayer - Can't Stand You
  - MT0028810869 | Q2 | Slayer - Can't Stand You
- distinct artists: 1620 (missing artist: 3)
- artists with more than one song: 577, covering 2186 songs
- share of those songs in their artist's majority quadrant: 85.00%

## 4. Train / validation / test splits
### tvt_40_30_30
- tvt_40_30_30_test_audio_balanced.csv: 968 rows, columns ['Song', 'Quadrant']
  - per quadrant: {'Q1': 242, 'Q2': 242, 'Q3': 242, 'Q4': 242}
- tvt_40_30_30_train_audio_balanced.csv: 1296 rows, columns ['Song', 'Quadrant']
  - per quadrant: {'Q1': 324, 'Q2': 324, 'Q3': 324, 'Q4': 324}
- tvt_40_30_30_validate_audio_balanced.csv: 968 rows, columns ['Song', 'Quadrant']
  - per quadrant: {'Q1': 242, 'Q2': 242, 'Q3': 242, 'Q4': 242}
- tvt_40_30_30_test_audio_balanced vs tvt_40_30_30_train_audio_balanced: shared IDs 0, shared artists 219, same Artist+Title 0
- tvt_40_30_30_test_audio_balanced vs tvt_40_30_30_validate_audio_balanced: shared IDs 0, shared artists 197, same Artist+Title 4
- tvt_40_30_30_train_audio_balanced vs tvt_40_30_30_validate_audio_balanced: shared IDs 0, shared artists 257, same Artist+Title 0

### tvt_70_15_15
- tvt_70_15_15_test_audio_balanced.csv: 484 rows, columns ['Song', 'Quadrant']
  - per quadrant: {'Q1': 121, 'Q2': 121, 'Q3': 121, 'Q4': 121}
- tvt_70_15_15_train_audio_balanced.csv: 2264 rows, columns ['Song', 'Quadrant']
  - per quadrant: {'Q1': 566, 'Q2': 566, 'Q3': 566, 'Q4': 566}
- tvt_70_15_15_validate_audio_balanced.csv: 484 rows, columns ['Song', 'Quadrant']
  - per quadrant: {'Q1': 121, 'Q2': 121, 'Q3': 121, 'Q4': 121}
- tvt_70_15_15_test_audio_balanced vs tvt_70_15_15_train_audio_balanced: shared IDs 0, shared artists 215, same Artist+Title 0
- tvt_70_15_15_test_audio_balanced vs tvt_70_15_15_validate_audio_balanced: shared IDs 0, shared artists 101, same Artist+Title 1
- tvt_70_15_15_train_audio_balanced vs tvt_70_15_15_validate_audio_balanced: shared IDs 0, shared artists 223, same Artist+Title 3

## Summary
- problems found: 0