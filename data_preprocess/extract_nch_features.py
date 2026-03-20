"""
=============================================================================
SilentSpeechLLM — Extract N-Channel Features from 8-Channel Gaddy Data
=============================================================================

Takes the 112-feature .npy files (8 channels x 14 features) and creates
N*14-feature .npy files by selecting the best N channels via Borda-count
ranking (variance + uniqueness + energy), or manually specified channels.

USAGE:
    # Auto-select best 3 channels
    python hpc/extract_nch_features.py --top_n 3

    # Auto-select best 4 channels
    python hpc/extract_nch_features.py --top_n 4

    # Manually specify channels
    python hpc/extract_nch_features.py --channels 0 2 4

    # Custom output directory
    python hpc/extract_nch_features.py --top_n 3 --output_dir data/my_3ch/
=============================================================================
"""

import os
import sys
import numpy as np
import json
import glob
import shutil
import argparse

FEATURES_PER_CHANNEL = 14
NUM_CHANNELS = 8


def rank_channels(input_dir):
    """Rank all 8 channels using Borda count (variance + uniqueness + energy).
    Returns list of channel indices sorted best-to-worst."""

    # Load all silent EMG features
    files = sorted(glob.glob(os.path.join(input_dir, "*_silent.npy")))
    if not files:
        print(f"ERROR: No *_silent.npy files found in {input_dir}")
        sys.exit(1)

    all_features = [np.load(f) for f in files]
    data = np.concatenate(all_features, axis=0)
    print(f"  Loaded {len(files)} files, {data.shape[0]} frames, {data.shape[1]} features")

    def get_ch(ch):
        return data[:, ch * FEATURES_PER_CHANNEL:(ch + 1) * FEATURES_PER_CHANNEL]

    # Method 1: Variance (higher = more signal)
    var_scores = []
    for ch in range(NUM_CHANNELS):
        total_var = np.var(get_ch(ch), axis=0).sum()
        var_scores.append((ch, total_var))
    var_rank = [ch for ch, _ in sorted(var_scores, key=lambda x: -x[1])]

    # Method 2: Uniqueness (lower avg correlation with others = more unique)
    corr_matrix = np.zeros((NUM_CHANNELS, NUM_CHANNELS))
    for i in range(NUM_CHANNELS):
        for j in range(NUM_CHANNELS):
            if i != j:
                fi, fj = get_ch(i).flatten(), get_ch(j).flatten()
                corr_matrix[i, j] = abs(np.corrcoef(fi, fj)[0, 1])
    uniq_scores = []
    for ch in range(NUM_CHANNELS):
        others = [corr_matrix[ch, j] for j in range(NUM_CHANNELS) if j != ch]
        uniq_scores.append((ch, 1 - np.mean(others)))
    corr_rank = [ch for ch, _ in sorted(uniq_scores, key=lambda x: -x[1])]

    # Method 3: Energy (higher L2 norm = stronger signal)
    energy_scores = []
    for ch in range(NUM_CHANNELS):
        total_energy = np.linalg.norm(get_ch(ch), axis=0).sum()
        energy_scores.append((ch, total_energy))
    energy_rank = [ch for ch, _ in sorted(energy_scores, key=lambda x: -x[1])]

    # Borda count
    borda = {}
    for ch in range(NUM_CHANNELS):
        borda[ch] = var_rank.index(ch) + corr_rank.index(ch) + energy_rank.index(ch)

    ranked = sorted(borda.items(), key=lambda x: x[1])

    print(f"  Channel ranking (Borda count):")
    print(f"  {'Rank':<6} {'Ch':<5} {'VarRk':>6} {'UniqRk':>7} {'EnRk':>6} {'Score':>6}")
    for rank, (ch, score) in enumerate(ranked):
        print(f"  {rank+1:<6} {ch:<5} {var_rank.index(ch)+1:>6} "
              f"{corr_rank.index(ch)+1:>7} {energy_rank.index(ch)+1:>6} {score:>6}")

    return [ch for ch, _ in ranked]


def extract_features(input_dir, output_dir, selected_channels):
    """Extract features for selected channels from all .npy files."""
    n_ch = len(selected_channels)
    feature_indices = []
    for ch in selected_channels:
        start = ch * FEATURES_PER_CHANNEL
        feature_indices.extend(range(start, start + FEATURES_PER_CHANNEL))

    num_output_features = len(feature_indices)
    print(f"\n  Extracting {n_ch} channels {selected_channels} → {num_output_features} features")

    os.makedirs(output_dir, exist_ok=True)

    # Process .npy files
    npy_files = sorted(glob.glob(os.path.join(input_dir, "*.npy")))
    print(f"  Processing {len(npy_files)} .npy files...")

    count = 0
    for npy_path in npy_files:
        fname = os.path.basename(npy_path)
        data = np.load(npy_path)
        assert data.shape[1] == 112, f"Expected 112 features, got {data.shape[1]} in {fname}"

        data_nch = data[:, feature_indices]
        assert data_nch.shape[1] == num_output_features

        np.save(os.path.join(output_dir, fname), data_nch)
        count += 1

    # Copy JSON metadata files
    json_files = sorted(glob.glob(os.path.join(input_dir, "*.json")))
    for json_path in json_files:
        fname = os.path.basename(json_path)
        out_path = os.path.join(output_dir, fname)
        if not os.path.exists(out_path):
            shutil.copy2(json_path, out_path)

    # Save extraction config
    config = {
        "source_dir": input_dir,
        "selected_channels": selected_channels,
        "feature_indices": feature_indices,
        "num_features": num_output_features,
        "features_per_channel": FEATURES_PER_CHANNEL,
        "original_num_channels": NUM_CHANNELS,
    }
    config_path = os.path.join(output_dir, f"_{n_ch}ch_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"  Done! {count} files → {output_dir}")
    print(f"  Config: {config_path}")
    return num_output_features


def main():
    parser = argparse.ArgumentParser(description='Extract N-channel features from 8ch Gaddy data')
    parser.add_argument('--input_dir',
                        default=os.path.expanduser('~/aml_lab/data/extracted_emg_features'),
                        help='Directory with 112-feature .npy files')
    parser.add_argument('--output_dir', default=None,
                        help='Output directory (default: auto-named {input}_Nch)')
    parser.add_argument('--channels', nargs='+', type=int, default=None,
                        help='Channel indices to select (e.g., --channels 0 2 4)')
    parser.add_argument('--top_n', type=int, default=None,
                        help='Auto-select best N channels via Borda-count ranking')

    args = parser.parse_args()

    if args.channels is None and args.top_n is None:
        parser.error("Must specify either --channels or --top_n")

    if args.channels and args.top_n:
        parser.error("Cannot specify both --channels and --top_n")

    if args.channels:
        selected = sorted(args.channels)
        n_ch = len(selected)
        print(f"Using manually specified channels: {selected}")
    else:
        n_ch = args.top_n
        print(f"Auto-selecting best {n_ch} channels...")
        ranked = rank_channels(args.input_dir)
        selected = sorted(ranked[:n_ch])
        print(f"  Selected: {selected}")

    if args.output_dir is None:
        args.output_dir = args.input_dir.rstrip('/') + f'_{n_ch}ch'

    extract_features(args.input_dir, args.output_dir, selected)


if __name__ == '__main__':
    main()
