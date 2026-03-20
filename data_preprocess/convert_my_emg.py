"""
=============================================================================
SilentSpeechLLM — Convert Personal EMG Data for Pipeline
=============================================================================

Full pipeline to convert MyoWare 2.0 CSV recordings into the format
expected by our training/evaluation scripts.

Steps:
  1. Read CSV files (4ch raw ADC) + metadata (session JSON or d_data.txt)
  2. Clean Ch3 spikes (interpolate single-sample ADC glitches)
  3. Resample to fixed 516.79 Hz (matching Gaddy pipeline)
  4. Apply 50 Hz notch filter + harmonics (UK mains, not 60 Hz)
  5. Apply 2 Hz highpass (drift removal)
  6. Extract 56 HC features (4 channels × 14 features each)
  7. (Optional) Per-feature z-score normalization
  8. Save as {prefix}_{idx}_silent.npy + {prefix}_{idx}.json
  9. Create train/dev split JSON

Supports two metadata formats:
  - Session JSON (i_unvoiced_1 style): emg_session_*.json with completed_log
  - d_data.txt (d_unvoiced_1 style): JSON array of {id, spoken, timestamp}

USAGE:
  # Single dataset (backwards compatible):
  python convert_my_emg.py --input_dir ~/aml_lab/i_unvoiced_1

  # Multiple datasets with normalization:
  python convert_my_emg.py \
      --input_dirs ~/aml_lab/i_unvoiced_1 ~/aml_lab/d_unvoiced_1 \
      --dataset_prefixes i d \
      --normalize \
      --output_dir ~/aml_lab/data/combined_emg_features
=============================================================================
"""

import os
import numpy as np
import json
import csv
import glob
import argparse
import random

import scipy.signal
import librosa


# ==========================================================================
# HC Feature Extraction (inlined from data_utils.py)
# ==========================================================================
def double_average(x):
    assert len(x.shape) == 1
    f = np.ones(9) / 9.0
    v = np.convolve(x, f, mode='same')
    w = np.convolve(v, f, mode='same')
    return w

def get_emg_features(emg_data):
    """Extract 14 HC features per channel from multi-channel EMG.
    Per channel: 5 time-domain + 9 frequency-domain = 14 features.
    Returns: (num_frames, num_channels * 14) float32 array.
    """
    xs = emg_data - emg_data.mean(axis=0, keepdims=True)
    frame_features = []
    for i in range(emg_data.shape[1]):
        x = xs[:, i]
        w = double_average(x)
        p = x - w
        r = np.abs(p)

        w_h = librosa.util.frame(w, frame_length=16, hop_length=6).mean(axis=0)
        p_w = librosa.feature.rms(y=w, frame_length=16, hop_length=6, center=False)
        p_w = np.squeeze(p_w, 0)
        p_r = librosa.feature.rms(y=r, frame_length=16, hop_length=6, center=False)
        p_r = np.squeeze(p_r, 0)
        z_p = librosa.feature.zero_crossing_rate(p, frame_length=16, hop_length=6, center=False)
        z_p = np.squeeze(z_p, 0)
        r_h = librosa.util.frame(r, frame_length=16, hop_length=6).mean(axis=0)

        s = abs(librosa.stft(np.ascontiguousarray(x), n_fft=16, hop_length=6, center=False))

        frame_features.append(np.stack([w_h, p_w, p_r, z_p, r_h], axis=1))
        frame_features.append(s.T)

    frame_features = np.concatenate(frame_features, axis=1)
    return frame_features.astype(np.float32)


# ==========================================================================
# Signal processing
# ==========================================================================
def remove_drift(signal, fs):
    """2 Hz highpass filter to remove DC drift."""
    b, a = scipy.signal.butter(3, 2, 'highpass', fs=fs)
    return scipy.signal.filtfilt(b, a, signal)

def notch(signal, freq, sample_frequency):
    """Single notch filter."""
    b, a = scipy.signal.iirnotch(freq, 30, sample_frequency)
    return scipy.signal.filtfilt(b, a, signal)

def notch_harmonics(signal, freq, sample_frequency):
    """Notch filter at fundamental + harmonics."""
    for harmonic in range(1, 8):
        if freq * harmonic < sample_frequency / 2:  # Don't exceed Nyquist
            signal = notch(signal, freq * harmonic, sample_frequency)
    return signal

def subsample(signal, new_freq, old_freq):
    """Resample using linear interpolation."""
    times = np.arange(len(signal)) / old_freq
    sample_times = np.arange(0, times[-1], 1 / new_freq)
    result = np.interp(sample_times, times, signal)
    return result

def apply_to_all(function, signal_array, *args, **kwargs):
    """Apply function to each channel."""
    results = []
    for i in range(signal_array.shape[1]):
        results.append(function(signal_array[:, i], *args, **kwargs))
    return np.stack(results, 1)


# ==========================================================================
# Spike cleaning
# ==========================================================================
def clean_spikes(emg, threshold_std=10):
    """
    Clean single-sample ADC spikes by interpolation.
    Detects outliers per channel (> threshold_std * std from mean)
    and replaces with linear interpolation of neighbors.
    """
    cleaned = emg.copy()
    total_cleaned = 0
    for ch in range(emg.shape[1]):
        signal = cleaned[:, ch]
        mean = np.median(signal)  # Use median (robust to spikes)
        std = np.std(signal[np.abs(signal - mean) < 5 * np.std(signal)])  # Std without extreme outliers

        # Find spikes
        spike_mask = np.abs(signal - mean) > threshold_std * std
        spike_indices = np.where(spike_mask)[0]

        if len(spike_indices) == 0:
            continue

        # Interpolate each spike from neighbors
        for si in spike_indices:
            left = signal[si - 1] if si > 0 else signal[si + 1]
            right = signal[si + 1] if si < len(signal) - 1 else signal[si - 1]
            cleaned[si, ch] = (left + right) / 2.0

        total_cleaned += len(spike_indices)

    return cleaned, total_cleaned


# ==========================================================================
# Metadata loading — supports both session JSON and d_data.txt formats
# ==========================================================================
def load_metadata(input_dir, mouth_duration_override=0, allow_missing=False):
    """
    Load recording metadata from either format.

    Format 1 (i_unvoiced_1): emg_session_*.json with completed_log
    Format 2 (d_unvoiced_1): d_data.txt (JSON array of {id, spoken, timestamp})

    Args:
        input_dir: directory containing CSV + metadata
        mouth_duration_override: fallback duration for d_data format
        allow_missing: if False, fail fast on missing CSV references

    Returns:
        entries: list of dicts with {csv_filename, output_id, text, mouth_duration}
        mouth_duration: float
    """
    entries = []

    # Try session JSON first (i_unvoiced_1 / d_unvoices_2 format)
    json_files = glob.glob(os.path.join(input_dir, 'emg_*session_*.json'))
    if json_files:
        session_path = json_files[0]
        print(f"  Metadata: {os.path.basename(session_path)} (session JSON format)")
        with open(session_path) as f:
            session = json.load(f)

        completed = session['completed_log']
        config = session['session']['config']
        mouth_duration = config.get('mouthDuration', config.get('mouth', 3.0))

        # Build set of available numeric CSV ids.
        csv_files = glob.glob(os.path.join(input_dir, '*.csv'))
        csv_nums = set()
        for p in csv_files:
            stem = os.path.basename(p).replace('.csv', '')
            if stem.isdigit():
                csv_nums.add(int(stem))

        # New quick_record sessions store explicit CSV mapping per entry.
        has_explicit_csv = any(('csv_num' in e) or ('csv_filename' in e) for e in completed)

        if has_explicit_csv:
            print("  Using explicit csv_num/csv_filename metadata from session log")

        # Fallback detection for legacy sessions without explicit CSV mapping.
        csv_offset = 0
        if completed and not has_explicit_csv:
            candidates = [0, 1, -1, 2, -2]
            scored = []
            for off in candidates:
                matched = sum(1 for e in completed if (e['index'] + off) in csv_nums)
                scored.append((matched, -abs(off), off))
            scored.sort(reverse=True)
            best_matched, _, best_offset = scored[0]
            csv_offset = best_offset
            print(f"  Detected CSV offset={csv_offset:+d} "
                  f"({best_matched}/{len(completed)} entries matched)")

        missing = 0
        for entry in completed:
            idx = int(entry['index'])

            # Prefer explicit mapping when present (new sessions).
            csv_num = None
            if 'csv_num' in entry:
                try:
                    csv_num = int(entry['csv_num'])
                except Exception:
                    csv_num = None
            if csv_num is None and 'csv_filename' in entry:
                stem = str(entry['csv_filename']).replace('.csv', '')
                if stem.isdigit():
                    csv_num = int(stem)
            if csv_num is None:
                csv_num = idx + csv_offset

            csv_name = f'{csv_num}.csv'
            csv_path = os.path.join(input_dir, csv_name)
            if os.path.exists(csv_path):
                entries.append({
                    'csv_filename': csv_name,
                    'output_id': idx,
                    'text': entry['text'],
                    'mouth_duration': mouth_duration,
                })
            else:
                print(f"    WARNING: Missing {csv_name} (index={idx})")
                missing += 1

        if missing > 0 and not allow_missing:
            raise FileNotFoundError(
                f"{missing} CSV files referenced by session metadata are missing in {input_dir}. "
                "Rerun quick_record resume or pass --allow_missing_csv to continue anyway."
            )

        return entries, mouth_duration

    # Try d_data.txt format (d_unvoiced_1 format)
    ddata_path = os.path.join(input_dir, 'd_data.txt')
    if os.path.exists(ddata_path):
        print(f"  Metadata: d_data.txt (flat JSON array format)")
        with open(ddata_path) as f:
            data = json.load(f)

        # Default mouth_duration: 3s (timestamps ~5s apart = 1s countdown + 3s mouth + 1s rest)
        mouth_duration = mouth_duration_override if mouth_duration_override > 0 else 3.0

        # Build id -> entry mapping
        id_map = {e['id']: e for e in data}

        # CSV n.csv -> id (n-1)
        csv_files = sorted(glob.glob(os.path.join(input_dir, '*.csv')),
                           key=lambda x: int(os.path.basename(x).replace('.csv', '')))
        missing = 0
        for csv_path in csv_files:
            csv_num = int(os.path.basename(csv_path).replace('.csv', ''))
            data_id = csv_num - 1  # CSV n -> id n-1
            if data_id in id_map:
                entries.append({
                    'csv_filename': f'{csv_num}.csv',
                    'output_id': data_id,
                    'text': id_map[data_id]['spoken'],
                    'mouth_duration': mouth_duration,
                })
            else:
                print(f"    WARNING: CSV {csv_num} has no matching id {data_id} in d_data.txt")
                missing += 1

        if missing > 0 and not allow_missing:
            raise ValueError(
                f"{missing} CSV files in {input_dir} do not map to d_data.txt ids. "
                "Fix metadata or pass --allow_missing_csv to continue."
            )

        return entries, mouth_duration

    raise FileNotFoundError(
        f"No session JSON (emg_session_*.json) or d_data.txt found in {input_dir}"
    )


# ==========================================================================
# Normalization
# ==========================================================================
def compute_normalization_stats(all_features):
    """Compute per-feature mean and std across all samples.

    Args:
        all_features: list of numpy arrays, each (T, num_features)

    Returns:
        mean: (num_features,) array
        std: (num_features,) array
    """
    all_frames = np.concatenate(all_features, axis=0)
    mean = all_frames.mean(axis=0)
    std = all_frames.std(axis=0)
    # Prevent division by zero for constant/zero-padded columns
    std[std < 1e-8] = 1.0
    return mean, std


def compute_robust_normalization_stats(all_features):
    """Compute per-feature median and IQR-based scale across all samples.

    Robust to outliers: uses median instead of mean, and IQR/1.3489 instead
    of std (IQR/1.3489 ≈ std for Gaussian data).

    Args:
        all_features: list of numpy arrays, each (T, num_features)

    Returns:
        center: (num_features,) array  — median per feature
        scale: (num_features,) array   — IQR/1.3489 per feature
    """
    all_frames = np.concatenate(all_features, axis=0)
    center = np.median(all_frames, axis=0)
    q75 = np.percentile(all_frames, 75, axis=0)
    q25 = np.percentile(all_frames, 25, axis=0)
    iqr = q75 - q25
    # IQR/1.3489 ≈ std for Gaussian data
    scale = iqr / 1.3489
    scale[scale < 1e-8] = 1.0
    return center, scale


def apply_normalization(features, mean, std):
    """Per-feature z-score normalization."""
    return ((features - mean) / std).astype(np.float32)


def per_sample_stft_prenorm(features, n_channels):
    """Per-sample normalization of STFT features to equalize amplitude.

    For each channel, divides STFT features (indices 5-13 within the 14
    per-channel block) by their RMS. This prevents high-amplitude recordings
    from dominating the global normalization statistics.

    Args:
        features: (T, num_features) array
        n_channels: number of EMG channels

    Returns:
        features: (T, num_features) array with STFT features normalized
    """
    features = features.copy()
    for ch in range(n_channels):
        stft_start = ch * 14 + 5   # s0 through s8
        stft_end = ch * 14 + 14
        stft_block = features[:, stft_start:stft_end]
        rms = np.sqrt(np.mean(stft_block ** 2))
        if rms > 1e-8:
            features[:, stft_start:stft_end] = stft_block / rms
    return features


# ==========================================================================
# Single-recording processing
# ==========================================================================
def process_single_recording(csv_path, mouth_duration, target_sr, mains_freq, pad_to,
                              drop_channels=None, drop_spike_rows_threshold=0):
    """
    Process a single CSV recording into HC features.

    Returns:
        features: numpy array (T, num_features), or None if skipped
        est_sr: estimated sample rate
        n_cleaned: number of spikes cleaned
        n_channels_used: number of channels after dropping
        n_rows_dropped: number of rows dropped due to spike threshold
    """
    with open(csv_path) as f:
        reader = csv.reader(f)
        header = next(reader)
        raw = np.array([[float(x) for x in r] for r in reader])

    n_samples = raw.shape[0]
    est_sr = n_samples / mouth_duration

    # Skip very short recordings (< 0.5s worth)
    if n_samples < est_sr * 0.5:
        return None, est_sr, 0, raw.shape[1], 0

    # Step 0.5: Drop rows where any channel exceeds spike threshold
    n_rows_dropped = 0
    if drop_spike_rows_threshold > 0:
        row_max = np.max(np.abs(raw), axis=1)
        keep_mask = row_max < drop_spike_rows_threshold
        n_rows_dropped = np.sum(~keep_mask)
        raw = raw[keep_mask]
        if raw.shape[0] < est_sr * 0.5:
            return None, est_sr, 0, raw.shape[1], n_rows_dropped

    # Step 1: Clean spikes (on all channels, before dropping)
    raw_clean, n_cleaned = clean_spikes(raw, threshold_std=10)

    # Step 1.5: Drop specified channels
    if drop_channels:
        keep = [c for c in range(raw_clean.shape[1]) if c not in drop_channels]
        raw_clean = raw_clean[:, keep]

    n_channels = raw_clean.shape[1]

    # Step 2: Resample to fixed rate
    resampled = apply_to_all(subsample, raw_clean, target_sr, est_sr)

    # Step 3: Notch filter (mains + harmonics)
    filtered = apply_to_all(notch_harmonics, resampled, mains_freq, target_sr)

    # Step 4: Highpass filter (2 Hz, remove drift)
    filtered = apply_to_all(remove_drift, filtered, target_sr)

    # Step 5: Extract HC features
    features = get_emg_features(filtered)

    expected_feats = n_channels * 14
    assert features.shape[1] == expected_feats, \
        f"Expected {expected_feats} features, got {features.shape[1]}"

    # Step 6: Optional padding to match checkpoint
    if pad_to > 0 and features.shape[1] < pad_to:
        pad_width = pad_to - features.shape[1]
        features = np.pad(features, ((0, 0), (0, pad_width)), mode='constant', constant_values=0)
    elif pad_to > 0 and features.shape[1] > pad_to:
        features = features[:, :pad_to]

    return features, est_sr, n_cleaned, n_channels, n_rows_dropped


# ==========================================================================
# Main
# ==========================================================================
parser = argparse.ArgumentParser(description='Convert personal EMG data for pipeline')
# Single-dataset mode (backwards compatible)
parser.add_argument('--input_dir',
                    default=os.path.expanduser('~/aml_lab/i_unvoiced_1'),
                    type=str, help='Directory with CSV files + metadata (single-dataset mode)')
# Multi-dataset mode
parser.add_argument('--input_dirs', nargs='+', default=None,
                    help='Multiple input directories (overrides --input_dir)')
parser.add_argument('--dataset_prefixes', nargs='+', default=None,
                    help='Prefix for each input dir to avoid ID collisions (e.g., i d)')
# Processing options
parser.add_argument('--output_dir',
                    default=os.path.expanduser('~/aml_lab/data/my_emg_features'),
                    type=str, help='Output directory for extracted features')
parser.add_argument('--mains_freq', default=50, type=int,
                    help='Mains frequency for notch filter (50=UK, 60=US)')
parser.add_argument('--target_sr', default=516.79, type=float,
                    help='Target sample rate after resampling')
parser.add_argument('--pad_to', default=0, type=int,
                    help='Pad features to this width (0=no pad, auto-computed from channels)')
parser.add_argument('--drop_channels', nargs='+', type=int, default=None,
                    help='Channel indices to drop (e.g., --drop_channels 3 to drop Ch3)')
parser.add_argument('--drop_spike_rows', default=0, type=float,
                    help='Drop rows where any channel value exceeds this threshold (e.g., 30000). 0=disabled.')
parser.add_argument('--normalize', action='store_true', default=False,
                    help='Apply per-feature z-score normalization across all samples')
parser.add_argument('--robust_norm', action='store_true', default=False,
                    help='Use robust normalization (median/IQR) instead of mean/std')
parser.add_argument('--stft_prenorm', action='store_true', default=False,
                    help='Per-sample STFT pre-normalization (equalize amplitude before global norm)')
parser.add_argument('--clip', default=0, type=float,
                    help='Clip normalized features to [-clip, clip] (0=no clip, recommended: 5.0)')
parser.add_argument('--norm_stats_file', default=None, type=str,
                    help='Load pre-computed norm stats from this file (for test-time normalization)')
parser.add_argument('--mouth_duration_override', default=0, type=float,
                    help='Override mouth duration (seconds) when not available in metadata')
parser.add_argument('--allow_missing_csv', action='store_true', default=False,
                    help='Allow missing/unmapped CSVs instead of failing fast')
parser.add_argument('--dev_ratio', default=0.1, type=float,
                    help='Dev set ratio')
parser.add_argument('--test_mode', action='store_true', default=False,
                    help='Put all samples in both train and dev (for eval-only runs)')
parser.add_argument('--seed', default=42, type=int)
args = parser.parse_args()

# ---- Determine input directories and prefixes ----
if args.input_dirs:
    input_dirs = args.input_dirs
    if args.dataset_prefixes:
        assert len(args.dataset_prefixes) == len(args.input_dirs), \
            f"Got {len(args.dataset_prefixes)} prefixes for {len(args.input_dirs)} dirs"
        prefixes = args.dataset_prefixes
    else:
        prefixes = [f'd{i}' for i in range(len(input_dirs))]
else:
    input_dirs = [args.input_dir]
    prefixes = ['']  # No prefix for single-dataset mode

os.makedirs(args.output_dir, exist_ok=True)
random.seed(args.seed)

print("=" * 60)
print("SilentSpeechLLM — EMG Feature Conversion")
print("=" * 60)
print(f"  Datasets: {len(input_dirs)}")
for d, p in zip(input_dirs, prefixes):
    print(f"    [{p or 'none'}] {d}")
print(f"  Output: {args.output_dir}")
print(f"  Mains freq: {args.mains_freq} Hz")
print(f"  Target SR: {args.target_sr} Hz")
print(f"  Pad to: {args.pad_to}")
print(f"  Normalize: {args.normalize}")
print(f"  Robust norm: {args.robust_norm}")
print(f"  STFT pre-norm: {args.stft_prenorm}")
print(f"  Clip: {args.clip}")
print(f"  Drop channels: {args.drop_channels}")
print(f"  Dev ratio: {args.dev_ratio}")
print(f"  Seed: {args.seed}")

# ---- Pass 1: Process all recordings from all datasets ----
all_records = []  # List of {key, features, text, session, original_id, est_sr, n_channels}
total_spikes = 0
total_skipped = 0

for input_dir, prefix in zip(input_dirs, prefixes):
    print(f"\n{'='*60}")
    print(f"Processing: {input_dir}")
    print(f"  Prefix: '{prefix}'")
    print(f"{'='*60}")

    entries, mouth_duration = load_metadata(
        input_dir,
        args.mouth_duration_override,
        allow_missing=args.allow_missing_csv,
    )
    print(f"  Found {len(entries)} recordings, mouth_duration={mouth_duration}s")

    extracted = 0
    skipped = 0
    sample_rates = []

    for entry in entries:
        csv_path = os.path.join(input_dir, entry['csv_filename'])
        features, est_sr, n_cleaned, n_ch_used, n_rows_dropped = process_single_recording(
            csv_path, entry['mouth_duration'], args.target_sr, args.mains_freq, args.pad_to,
            drop_channels=args.drop_channels,
            drop_spike_rows_threshold=args.drop_spike_rows
        )
        sample_rates.append(est_sr)
        total_spikes += n_cleaned
        if n_rows_dropped > 0:
            print(f"    {entry['csv_filename']}: dropped {n_rows_dropped} spike rows")

        if features is None:
            print(f"    WARNING: {entry['csv_filename']} too short, skipping")
            skipped += 1
            total_skipped += 1
            continue

        # Build output key
        oid = entry['output_id']
        if prefix:
            output_key = f"{prefix}_{oid}"
        else:
            output_key = str(oid)

        all_records.append({
            'key': output_key,
            'features': features,
            'text': entry['text'],
            'session': os.path.basename(input_dir),
            'original_id': oid,
            'source_csv': entry['csv_filename'],
            'est_sr': est_sr,
            'n_channels': n_ch_used,
        })
        extracted += 1

        if extracted % 50 == 0:
            print(f"    Processed {extracted} recordings...")

    sr_arr = np.array(sample_rates)
    print(f"  Extracted: {extracted}, Skipped: {skipped}")
    print(f"  Sample rate: {sr_arr.mean():.0f} Hz avg (range {sr_arr.min():.0f}-{sr_arr.max():.0f})")

print(f"\n{'='*60}")
print(f"TOTAL: {len(all_records)} records from {len(input_dirs)} datasets")
print(f"  Spikes cleaned: {total_spikes}")
print(f"  Skipped: {total_skipped}")

all_sr = np.array([r['est_sr'] for r in all_records], dtype=np.float64)
print(f"  Source SR: {all_sr.mean():.2f} Hz avg (range {all_sr.min():.2f}-{all_sr.max():.2f})")

# ---- Pass 1.5 (optional): Per-sample STFT pre-normalization ----
if args.stft_prenorm:
    print(f"\n{'='*60}")
    print(f"STFT PRE-NORMALIZATION: Equalizing per-sample STFT amplitude")
    print(f"{'='*60}")
    n_channels = all_records[0]['n_channels']
    for r in all_records:
        r['features'] = per_sample_stft_prenorm(r['features'], n_channels)
    sample = all_records[0]['features']
    print(f"  Post-prenorm sample [{all_records[0]['key']}]: "
          f"mean={sample.mean():.4f}, std={sample.std():.4f}, "
          f"range=[{sample.min():.4f}, {sample.max():.4f}]")

# ---- Pass 2 (optional): Normalization ----
if args.normalize:
    print(f"\n{'='*60}")

    if args.norm_stats_file and os.path.exists(args.norm_stats_file):
        # Load pre-computed stats (for test-time normalization)
        norm_type = "pre-computed"
        print(f"NORMALIZATION: Loading pre-computed stats from {args.norm_stats_file}")
        print(f"{'='*60}")
        with open(args.norm_stats_file) as f:
            loaded_stats = json.load(f)
        mean = np.array(loaded_stats['mean'])
        std = np.array(loaded_stats['std'])
        print(f"  Source: {args.norm_stats_file}")
        print(f"  Trained on {loaded_stats.get('num_samples', '?')} samples")
        print(f"  Mode: {loaded_stats.get('norm_type', 'standard')}")
        # Warn if settings don't match
        if loaded_stats.get('stft_prenorm', False) != args.stft_prenorm:
            print(f"  WARNING: stft_prenorm mismatch! Stats used {loaded_stats.get('stft_prenorm')}, current={args.stft_prenorm}")
        if loaded_stats.get('clip', 0) != args.clip:
            print(f"  NOTE: Stats computed with clip={loaded_stats.get('clip', 0)}, current clip={args.clip}")
    elif args.robust_norm:
        # Robust normalization (median/IQR)
        norm_type = "robust"
        print(f"NORMALIZATION: Computing ROBUST per-feature statistics (median/IQR)")
        print(f"{'='*60}")
        all_feats = [r['features'] for r in all_records]
        mean, std = compute_robust_normalization_stats(all_feats)
    else:
        # Standard z-score normalization
        norm_type = "standard"
        print(f"NORMALIZATION: Computing per-feature z-score statistics")
        print(f"{'='*60}")
        all_feats = [r['features'] for r in all_records]
        mean, std = compute_normalization_stats(all_feats)

    print(f"  Norm type: {norm_type}")
    print(f"  Num features: {len(mean)}")
    print(f"  Center range: [{mean.min():.4f}, {mean.max():.4f}]")
    print(f"  Scale range:  [{std.min():.4f}, {std.max():.4f}]")

    # Apply normalization
    for r in all_records:
        r['features'] = apply_normalization(r['features'], mean, std)

    # Save normalization stats for reproducibility / inference
    norm_stats = {
        'mean': mean.tolist(),
        'std': std.tolist(),
        'num_samples': len(all_records),
        'num_features': int(len(mean)),
        'norm_type': norm_type,
        'stft_prenorm': args.stft_prenorm,
        'clip': args.clip,
        'mains_freq': args.mains_freq,
        'target_sr': args.target_sr,
        'drop_channels': args.drop_channels,
        'source_sr_mean': float(all_sr.mean()),
        'source_sr_min': float(all_sr.min()),
        'source_sr_max': float(all_sr.max()),
    }
    norm_path = os.path.join(args.output_dir, 'norm_stats.json')
    with open(norm_path, 'w') as f:
        json.dump(norm_stats, f, indent=2)
    print(f"  Saved: {norm_path}")

    # Verify post-norm stats
    sample = all_records[0]['features']
    print(f"  Post-norm sample [{all_records[0]['key']}]: "
          f"mean={sample.mean():.4f}, std={sample.std():.4f}, "
          f"range=[{sample.min():.4f}, {sample.max():.4f}]")

# ---- Pass 2.5 (optional): Clipping ----
if args.clip > 0:
    print(f"\n{'='*60}")
    print(f"CLIPPING: Clipping features to [-{args.clip}, {args.clip}]")
    print(f"{'='*60}")
    total_clipped = 0
    total_elements = 0
    for r in all_records:
        f = r['features']
        n_clipped = np.sum(np.abs(f) > args.clip)
        total_clipped += n_clipped
        total_elements += f.size
        r['features'] = np.clip(f, -args.clip, args.clip).astype(np.float32)
    pct = total_clipped / total_elements * 100 if total_elements > 0 else 0
    print(f"  Clipped {total_clipped} values ({pct:.3f}% of all elements)")
    sample = all_records[0]['features']
    print(f"  Post-clip sample [{all_records[0]['key']}]: "
          f"range=[{sample.min():.4f}, {sample.max():.4f}]")

# ---- Save all files ----
print(f"\n{'='*60}")
print(f"SAVING {len(all_records)} files to {args.output_dir}")
print(f"{'='*60}")

all_keys = []
for r in all_records:
    key = r['key']

    npy_path = os.path.join(args.output_dir, f'{key}_silent.npy')
    np.save(npy_path, r['features'])

    meta = {
        "sentence_index": key,
        "text": r['text'],
        "book": "personal_recording",
        "session": r['session'],
        "original_index": r['original_id'],
        "source_csv": r.get('source_csv'),
        "num_frames": int(r['features'].shape[0]),
        "num_features": int(r['features'].shape[1]),
        "est_sample_rate": round(r['est_sr'], 1),
        "channels_used": r['n_channels'],
    }
    json_path = os.path.join(args.output_dir, f'{key}.json')
    with open(json_path, 'w') as f:
        json.dump(meta, f, indent=2)

    all_keys.append(key)

# ---- Create train/dev split ----
if args.test_mode:
    # Test mode: all samples in both train and dev (for eval-only runs)
    train_keys = sorted(all_keys, key=str)
    dev_keys = sorted(all_keys, key=str)
else:
    random.shuffle(all_keys)
    n_dev = max(1, int(len(all_keys) * args.dev_ratio))
    dev_keys = sorted(all_keys[:n_dev], key=str)
    train_keys = sorted(all_keys[n_dev:], key=str)

split = {
    "train_sentence_indices": train_keys,
    "dev_sentence_indices": dev_keys,
    "total_samples": len(all_keys),
    "train_samples": len(train_keys),
    "dev_samples": len(dev_keys),
}
split_path = os.path.join(args.output_dir, 'split.json')
with open(split_path, 'w') as f:
    json.dump(split, f, indent=2)

# ---- Summary ----
print(f"\n{'='*60}")
print(f"CONVERSION COMPLETE")
print(f"{'='*60}")
print(f"  Total records:  {len(all_records)}")
print(f"  Train:          {len(train_keys)}")
print(f"  Dev:            {len(dev_keys)}")
print(f"  Normalized:     {args.normalize}")
print(f"  Output dir:     {args.output_dir}")
print(f"  Split file:     {split_path}")

# Verify a sample
sample_files = sorted(glob.glob(os.path.join(args.output_dir, '*_silent.npy')))
if sample_files:
    s = np.load(sample_files[0])
    print(f"\n  Sample check: {os.path.basename(sample_files[0])}")
    print(f"    Shape: {s.shape}")
    print(f"    Range: [{s.min():.4f}, {s.max():.4f}]")
    print(f"    Mean: {s.mean():.4f}, Std: {s.std():.4f}")

# Show dev set texts
print(f"\n  Dev set samples:")
for dk in dev_keys[:10]:
    jp = os.path.join(args.output_dir, f'{dk}.json')
    with open(jp) as f:
        m = json.load(f)
    print(f"    [{dk}] \"{m['text']}\"")
if len(dev_keys) > 10:
    print(f"    ... and {len(dev_keys) - 10} more")
print(f"{'='*60}")
