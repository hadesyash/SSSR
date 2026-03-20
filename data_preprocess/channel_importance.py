"""
=============================================================================
SilentSpeechLLM — Channel Importance Analysis
=============================================================================

Determines which 4 out of 8 EMG channels are most important for the task.

Methods:
  1. VARIANCE ANALYSIS — channels with higher variance carry more signal
  2. CORRELATION ANALYSIS — drop highly correlated (redundant) channels
  3. GRADIENT-BASED — use the best trained adaptor (Gemma-2B) to measure
     which channel features the model relies on most (via input gradient)
  4. BRUTE-FORCE — train with each combination of 4 channels (C(8,4)=70)
     and pick the best (too expensive, so we do methods 1-3 instead)

We use methods 1 & 2 to rank channels, then pick the best 4.
The selected channels are used to create 4-channel (56-feature) .npy files.

Feature layout in the 112-dim vector:
  Features  0-13  = Channel 0 (5 stats + 9 STFT)
  Features 14-27  = Channel 1
  Features 28-41  = Channel 2
  Features 42-55  = Channel 3
  Features 56-69  = Channel 4
  Features 70-83  = Channel 5
  Features 84-97  = Channel 6
  Features 98-111 = Channel 7

USAGE:
    python ~/aml_lab/hpc/channel_importance.py

Output: ranked channels + recommendation for best 4
=============================================================================
"""

import os
import numpy as np
import json
import glob

DATA_DIR = os.path.expanduser("~/aml_lab/data/extracted_emg_features")
JSON_PATH = os.path.expanduser("~/aml_lab/data/10_selected_samples.json")

FEATURES_PER_CHANNEL = 14
NUM_CHANNELS = 8


def get_channel_features(data, ch):
    """Extract features for a specific channel from the 112-dim vector."""
    start = ch * FEATURES_PER_CHANNEL
    end = start + FEATURES_PER_CHANNEL
    return data[:, start:end]


def load_all_features():
    """Load all silent EMG feature files and concatenate."""
    all_features = []
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*_silent.npy")))
    print(f"Loading {len(files)} silent EMG files...")
    for f in files:
        features = np.load(f)
        all_features.append(features)
    combined = np.concatenate(all_features, axis=0)
    print(f"Total frames: {combined.shape[0]}, Features: {combined.shape[1]}")
    return combined


def variance_analysis(data):
    """Rank channels by total feature variance (higher = more signal)."""
    print("\n" + "=" * 70)
    print("METHOD 1: VARIANCE ANALYSIS")
    print("  Higher variance = more informative signal")
    print("=" * 70)

    channel_vars = []
    for ch in range(NUM_CHANNELS):
        ch_data = get_channel_features(data, ch)
        # Mean variance across all 14 features of this channel
        mean_var = np.var(ch_data, axis=0).mean()
        total_var = np.var(ch_data, axis=0).sum()
        channel_vars.append((ch, mean_var, total_var))

    channel_vars.sort(key=lambda x: -x[2])  # Sort by total variance, descending

    print(f"\n  {'Rank':<6} {'Channel':<10} {'Mean Var':>10} {'Total Var':>12} {'% of Total':>12}")
    print("-" * 55)
    total = sum(v[2] for v in channel_vars)
    for rank, (ch, mv, tv) in enumerate(channel_vars):
        pct = tv / total * 100
        print(f"  {rank+1:<6} Ch {ch:<7} {mv:>10.4f} {tv:>12.4f} {pct:>11.1f}%")

    return [ch for ch, _, _ in channel_vars]


def correlation_analysis(data):
    """Find redundant channels via inter-channel correlation."""
    print("\n" + "=" * 70)
    print("METHOD 2: INTER-CHANNEL CORRELATION")
    print("  High correlation = redundant channels")
    print("=" * 70)

    # Compute mean feature vector per channel, then correlate
    channel_means = []
    for ch in range(NUM_CHANNELS):
        ch_data = get_channel_features(data, ch)
        channel_means.append(ch_data)

    # Correlation between channel feature vectors (flatten time dimension)
    print(f"\n  Correlation matrix (mean absolute correlation between channels):")
    print(f"  {'':>6}", end="")
    for ch in range(NUM_CHANNELS):
        print(f" Ch{ch:>2}", end="")
    print()

    corr_matrix = np.zeros((NUM_CHANNELS, NUM_CHANNELS))
    for i in range(NUM_CHANNELS):
        print(f"  Ch{i:>2} ", end="")
        for j in range(NUM_CHANNELS):
            # Correlate the flattened feature vectors
            fi = channel_means[i].flatten()
            fj = channel_means[j].flatten()
            corr = np.abs(np.corrcoef(fi, fj)[0, 1])
            corr_matrix[i, j] = corr
            if i == j:
                print(f" 1.00", end="")
            else:
                print(f" {corr:.2f}", end="")
        print()

    # Per-channel average correlation with others (lower = more unique)
    print(f"\n  {'Channel':<10} {'Avg Corr with Others':>22} {'Uniqueness':>12}")
    print("-" * 50)
    uniqueness = []
    for ch in range(NUM_CHANNELS):
        others = [corr_matrix[ch, j] for j in range(NUM_CHANNELS) if j != ch]
        avg_corr = np.mean(others)
        uniqueness.append((ch, avg_corr, 1 - avg_corr))

    uniqueness.sort(key=lambda x: -x[2])  # Sort by uniqueness, descending
    for ch, corr, uniq in uniqueness:
        print(f"  Ch {ch:<7} {corr:>22.4f} {uniq:>12.4f}")

    return [ch for ch, _, _ in uniqueness]


def feature_energy_analysis(data):
    """Rank channels by signal energy (L2 norm)."""
    print("\n" + "=" * 70)
    print("METHOD 3: SIGNAL ENERGY (L2 NORM)")
    print("  Higher energy = stronger muscle activation signal")
    print("=" * 70)

    channel_energy = []
    for ch in range(NUM_CHANNELS):
        ch_data = get_channel_features(data, ch)
        energy = np.linalg.norm(ch_data, axis=0).mean()
        total_energy = np.linalg.norm(ch_data, axis=0).sum()
        channel_energy.append((ch, energy, total_energy))

    channel_energy.sort(key=lambda x: -x[2])

    print(f"\n  {'Rank':<6} {'Channel':<10} {'Mean Energy':>12} {'Total Energy':>14}")
    print("-" * 48)
    for rank, (ch, me, te) in enumerate(channel_energy):
        print(f"  {rank+1:<6} Ch {ch:<7} {me:>12.4f} {te:>14.4f}")

    return [ch for ch, _, _ in channel_energy]


def combined_ranking(var_rank, corr_rank, energy_rank):
    """Combine rankings using Borda count."""
    print("\n" + "=" * 70)
    print("COMBINED RANKING (Borda Count)")
    print("  Lower score = better (ranked higher across methods)")
    print("=" * 70)

    scores = {}
    for ch in range(NUM_CHANNELS):
        v_pos = var_rank.index(ch)
        c_pos = corr_rank.index(ch)
        e_pos = energy_rank.index(ch)
        total = v_pos + c_pos + e_pos
        scores[ch] = {
            "variance_rank": v_pos + 1,
            "uniqueness_rank": c_pos + 1,
            "energy_rank": e_pos + 1,
            "total_score": total,
        }

    sorted_channels = sorted(scores.items(), key=lambda x: x[1]["total_score"])

    print(f"\n  {'Rank':<6} {'Channel':<10} {'Var':>5} {'Uniq':>5} {'Energy':>7} {'Score':>7}")
    print("-" * 48)
    for rank, (ch, s) in enumerate(sorted_channels):
        marker = " <<<" if rank < 4 else ""
        print(f"  {rank+1:<6} Ch {ch:<7} {s['variance_rank']:>5} {s['uniqueness_rank']:>5} "
              f"{s['energy_rank']:>7} {s['total_score']:>7}{marker}")

    best_4 = [ch for ch, _ in sorted_channels[:4]]
    print(f"\n  RECOMMENDED 4 CHANNELS: {sorted(best_4)}")
    print(f"  Feature indices: {sorted([ch * 14 for ch in best_4])} (start of each channel's 14 features)")

    return sorted(best_4)


def main():
    data = load_all_features()

    var_rank = variance_analysis(data)
    corr_rank = correlation_analysis(data)
    energy_rank = feature_energy_analysis(data)
    best_4 = combined_ranking(var_rank, corr_rank, energy_rank)

    # Show what the 56-feature vector would look like
    print("\n" + "=" * 70)
    print("4-CHANNEL FEATURE LAYOUT (56 features)")
    print("=" * 70)
    feature_indices = []
    for i, ch in enumerate(best_4):
        start = ch * FEATURES_PER_CHANNEL
        end = start + FEATURES_PER_CHANNEL
        indices = list(range(start, end))
        feature_indices.extend(indices)
        print(f"  Channel {ch} → features [{start}:{end}] → new features [{i*14}:{(i+1)*14}]")
    print(f"  Total: {len(feature_indices)} features")

    # Save the selected channels for use by other scripts
    output = {
        "selected_channels": best_4,
        "feature_indices": feature_indices,
        "num_features": len(feature_indices),
        "features_per_channel": FEATURES_PER_CHANNEL,
    }
    out_path = os.path.expanduser("~/aml_lab/data/selected_4_channels.json")
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved channel selection to: {out_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
