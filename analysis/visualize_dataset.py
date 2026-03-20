"""
Visualise the SilentSpeechLLM EMG dataset.

Usage:
    conda activate emg_llm
    python visualize_dataset.py                  # dataset overview + random sample
    python visualize_dataset.py --sample 42      # visualise specific sample ID
    python visualize_dataset.py --compare 42     # silent vs voiced side-by-side
"""

import os
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend; saves PNGs without blocking
import matplotlib.pyplot as plt
from collections import Counter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "extracted_emg_features")
JSON_PATH = os.path.join(BASE_DIR, "SilentSpeechLLM", "train_id_json", "10_selected_samples.json")

# ─── Helpers ────────────────────────────────────────────────────────────

def load_all_metadata():
    """Load all JSON metadata files."""
    samples = {}
    for fname in os.listdir(DATA_DIR):
        if fname.endswith(".json"):
            sid = int(fname.replace(".json", ""))
            with open(os.path.join(DATA_DIR, fname)) as f:
                samples[sid] = json.load(f)
    return dict(sorted(samples.items()))

def load_sample(sid, silent=True):
    """Load EMG features and metadata for a sample."""
    suffix = "_silent" if silent else "_voiced"
    npy_path = os.path.join(DATA_DIR, f"{sid}{suffix}.npy")
    json_path = os.path.join(DATA_DIR, f"{sid}.json")
    features = np.load(npy_path)
    with open(json_path) as f:
        meta = json.load(f)
    return features, meta

# ─── 1. Dataset Overview ───────────────────────────────────────────────

def plot_dataset_overview(samples):
    """4-panel overview of the full dataset."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("SilentSpeechLLM Dataset Overview (Gaddy & Klein 2021)", fontsize=14, fontweight='bold')

    # --- Panel 1: Sequence length distribution ---
    lengths = []
    for sid in samples:
        npy = os.path.join(DATA_DIR, f"{sid}_silent.npy")
        if os.path.exists(npy):
            lengths.append(np.load(npy).shape[0])

    ax = axes[0, 0]
    ax.hist(lengths, bins=30, color='steelblue', edgecolor='white', alpha=0.8)
    ax.axvline(np.mean(lengths), color='red', linestyle='--', label=f'Mean={np.mean(lengths):.0f}')
    ax.set_xlabel("Timesteps (T)")
    ax.set_ylabel("Count")
    ax.set_title(f"Sequence Length Distribution (n={len(lengths)})")
    ax.legend()

    # --- Panel 2: Vocabulary (word frequency) ---
    all_words = []
    for meta in samples.values():
        all_words.extend(meta['text'].lower().split())

    word_freq = Counter(all_words)
    top_words = word_freq.most_common(20)
    words, counts = zip(*top_words)

    ax = axes[0, 1]
    bars = ax.barh(range(len(words)), counts, color='coral', edgecolor='white')
    ax.set_yticks(range(len(words)))
    ax.set_yticklabels(words, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Frequency")
    ax.set_title(f"Top 20 Words ({len(word_freq)} unique)")

    # --- Panel 3: Words per sentence ---
    word_counts = [len(meta['text'].split()) for meta in samples.values()]

    ax = axes[1, 0]
    ax.hist(word_counts, bins=range(1, max(word_counts) + 2), color='mediumseagreen',
            edgecolor='white', alpha=0.8, align='left')
    ax.set_xlabel("Words per sentence")
    ax.set_ylabel("Count")
    ax.set_title(f"Sentence Length Distribution (mean={np.mean(word_counts):.1f})")
    ax.set_xticks(range(1, max(word_counts) + 1))

    # --- Panel 4: Train/dev split ---
    with open(JSON_PATH) as f:
        splits = json.load(f)
    train_ids = set(splits['train_sentence_indices'])
    dev_ids = set(splits['dev_sentence_indices'])

    ax = axes[1, 1]
    categories = ['Train', 'Dev', 'Total']
    values = [len(train_ids), len(dev_ids), len(samples)]
    colors = ['#4CAF50', '#FF9800', '#2196F3']
    bars = ax.bar(categories, values, color=colors, edgecolor='white')
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3,
                str(val), ha='center', fontweight='bold')
    ax.set_ylabel("Samples")
    ax.set_title("Train / Dev Split")
    ax.set_ylim(0, max(values) * 1.15)

    plt.tight_layout()
    out_path = os.path.join(BASE_DIR, "dataset_overview.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close(fig)


# ─── 2. Single Sample Visualisation ────────────────────────────────────

# Feature group names based on Gaddy & Klein's feature extraction
# 112 features = 8 channels x 14 features per channel
# Per channel: [raw_mean, raw_std, raw_rms, raw_zcr, raw_ma1..4, stft_mag1..6]
FEATURE_GROUPS = {
    "Statistical (mean/std/rms/zcr)": list(range(0, 32)),     # 4 stats x 8 channels
    "Moving Averages": list(range(32, 64)),                     # 4 MAs x 8 channels
    "STFT Magnitudes": list(range(64, 112)),                    # 6 freq bins x 8 channels
}

def plot_single_sample(sid, silent=True):
    """Detailed visualisation of one EMG sample."""
    features, meta = load_sample(sid, silent=silent)
    T, F = features.shape
    mode = "Silent" if silent else "Voiced"

    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(f"Sample {sid} ({mode}) — \"{meta['text']}\"\n"
                 f"Shape: ({T} timesteps x {F} features)", fontsize=13, fontweight='bold')

    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)

    # --- Panel 1: Full heatmap ---
    ax = fig.add_subplot(gs[0, :])
    im = ax.imshow(features.T, aspect='auto', cmap='RdBu_r',
                   interpolation='nearest', vmin=-3, vmax=3)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Feature index")
    ax.set_title("All 112 Features (heatmap)")
    plt.colorbar(im, ax=ax, shrink=0.6, label="Value")

    # Annotate feature groups
    group_boundaries = [0, 32, 64, 112]
    group_labels = ["Stats", "MA", "STFT"]
    for i, (start, label) in enumerate(zip(group_boundaries[:-1], group_labels)):
        ax.axhline(y=start - 0.5, color='black', linewidth=0.5, alpha=0.5)
        ax.text(-T * 0.02, (start + group_boundaries[i + 1]) / 2, label,
                ha='right', va='center', fontsize=8, fontweight='bold')

    # --- Panel 2: Feature group means over time ---
    ax = fig.add_subplot(gs[1, 0])
    colors = ['#e74c3c', '#3498db', '#2ecc71']
    for (name, indices), color in zip(FEATURE_GROUPS.items(), colors):
        group_mean = features[:, indices].mean(axis=1)
        ax.plot(group_mean, label=name.split("(")[0].strip(), color=color, alpha=0.8)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Mean value")
    ax.set_title("Feature Group Means Over Time")
    ax.legend(fontsize=8)
    ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')

    # --- Panel 3: Per-channel energy (RMS across features) ---
    ax = fig.add_subplot(gs[1, 1])
    n_channels = 8
    features_per_ch = F // n_channels  # 14
    for ch in range(n_channels):
        ch_start = ch * features_per_ch
        ch_end = ch_start + features_per_ch
        ch_energy = np.sqrt(np.mean(features[:, ch_start:ch_end] ** 2, axis=1))
        ax.plot(ch_energy, label=f"Ch {ch}", alpha=0.7)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("RMS energy")
    ax.set_title("Per-Channel Energy (8 EMG channels)")
    ax.legend(fontsize=7, ncol=2)

    # --- Panel 4: Feature distribution ---
    ax = fig.add_subplot(gs[2, 0])
    flat = features.flatten()
    ax.hist(flat, bins=100, color='steelblue', edgecolor='none', alpha=0.8, density=True)
    ax.set_xlabel("Feature value")
    ax.set_ylabel("Density")
    ax.set_title(f"Feature Value Distribution (mean={flat.mean():.2f}, std={flat.std():.2f})")
    ax.axvline(0, color='red', linewidth=0.5, linestyle='--')

    # --- Panel 5: Feature correlation ---
    ax = fig.add_subplot(gs[2, 1])
    # Subsample features for readability
    step = max(1, F // 32)
    sub_features = features[:, ::step]
    corr = np.corrcoef(sub_features.T)
    im = ax.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
    ax.set_title(f"Feature Correlation (every {step}th)")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Feature")
    plt.colorbar(im, ax=ax, shrink=0.6)

    out_path = os.path.join(BASE_DIR, f"sample_{sid}_{mode.lower()}.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close(fig)


# ─── 3. Silent vs Voiced Comparison ────────────────────────────────────

def plot_silent_vs_voiced(sid):
    """Side-by-side comparison of silent and voiced EMG for the same sentence."""
    silent, meta = load_sample(sid, silent=True)
    voiced_path = os.path.join(DATA_DIR, f"{sid}_voiced.npy")
    if not os.path.exists(voiced_path):
        print(f"No voiced sample for ID {sid}")
        return
    voiced = np.load(voiced_path)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f"Silent vs Voiced — Sample {sid}: \"{meta['text']}\"",
                 fontsize=13, fontweight='bold')

    # Heatmaps
    for i, (data, label) in enumerate([(silent, "Silent"), (voiced, "Voiced")]):
        ax = axes[0, i]
        im = ax.imshow(data.T, aspect='auto', cmap='RdBu_r', vmin=-3, vmax=3,
                       interpolation='nearest')
        ax.set_title(f"{label} EMG ({data.shape[0]} timesteps)")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Feature")
        plt.colorbar(im, ax=ax, shrink=0.6)

    # Energy profiles
    ax = axes[1, 0]
    silent_energy = np.sqrt(np.mean(silent ** 2, axis=1))
    voiced_energy = np.sqrt(np.mean(voiced ** 2, axis=1))
    ax.plot(silent_energy, label="Silent", color='#e74c3c', alpha=0.8)
    ax.plot(voiced_energy, label="Voiced", color='#3498db', alpha=0.8)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("RMS energy")
    ax.set_title("Overall Energy Comparison")
    ax.legend()

    # Feature distributions
    ax = axes[1, 1]
    ax.hist(silent.flatten(), bins=80, alpha=0.6, label=f"Silent (std={silent.std():.2f})",
            color='#e74c3c', density=True, edgecolor='none')
    ax.hist(voiced.flatten(), bins=80, alpha=0.6, label=f"Voiced (std={voiced.std():.2f})",
            color='#3498db', density=True, edgecolor='none')
    ax.set_xlabel("Feature value")
    ax.set_ylabel("Density")
    ax.set_title("Feature Value Distributions")
    ax.legend()

    out_path = os.path.join(BASE_DIR, f"compare_{sid}.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close(fig)


# ─── Main ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualise SilentSpeechLLM dataset")
    parser.add_argument("--sample", type=int, default=None,
                        help="Visualise a specific sample ID")
    parser.add_argument("--compare", type=int, default=None,
                        help="Compare silent vs voiced for a sample ID")
    parser.add_argument("--no-overview", action="store_true",
                        help="Skip dataset overview")
    args = parser.parse_args()

    samples = load_all_metadata()
    print(f"Dataset: {len(samples)} samples")
    print(f"Vocabulary: {len(set(w for m in samples.values() for w in m['text'].lower().split()))} unique words")
    print(f"Example texts:")
    import random
    random.seed(42)
    for sid in random.sample(list(samples.keys()), 5):
        print(f"  [{sid}] \"{samples[sid]['text']}\"")

    if not args.no_overview:
        print("\n--- Dataset Overview ---")
        plot_dataset_overview(samples)

    if args.sample is not None:
        sid = args.sample
        if sid not in samples:
            print(f"Sample {sid} not found!")
        else:
            print(f"\n--- Sample {sid} Detail ---")
            plot_single_sample(sid, silent=True)

    elif args.compare is not None:
        sid = args.compare
        if sid not in samples:
            print(f"Sample {sid} not found!")
        else:
            print(f"\n--- Silent vs Voiced: Sample {sid} ---")
            plot_silent_vs_voiced(sid)

    else:
        # Default: also show a random sample
        sid = random.choice(list(samples.keys()))
        print(f"\n--- Random Sample {sid}: \"{samples[sid]['text']}\" ---")
        plot_single_sample(sid, silent=True)
