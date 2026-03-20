# Data

This directory is not tracked by git. Download the required datasets below.

## Gaddy & Klein (2020) — Closed-Vocabulary Benchmark

Pre-extracted hand-crafted (HC) features (112 features per timestep, 8 channels × 14 features):

- **Zenodo**: [https://zenodo.org/records/15557946](https://zenodo.org/records/15557946)
- Place the extracted files in `data/extracted_emg_features/`
- 500 utterances, 67 unique phrases

The original raw EMG dataset is available from:
- **Gaddy & Klein (2020)**: [https://doi.org/10.5281/zenodo.4064408](https://doi.org/10.5281/zenodo.4064408)

## Personal EMG Data

Our custom 4-channel recordings using MyoWare 2.0 sensors are not publicly hosted due to size.
To reproduce:
1. Record raw EMG CSVs using `firmware/host.py` with the Arduino setup
2. Convert to features using `data_preprocess/convert_my_emg.py`:
   ```bash
   python data_preprocess/convert_my_emg.py \
       --input_dir <raw_csv_dir> \
       --output_dir data/combined_emg_features_v2/ \
       --robust_norm --stft_prenorm --clip 5.0
   ```

## Train/Dev Split

The train/dev split used in all experiments is at `config/10_selected_samples.json`.
