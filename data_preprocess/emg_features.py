"""
=============================================================================
SilentSpeechLLM — EMG Feature Extraction Module
=============================================================================

Reusable functions for extracting HC features from raw EMG data.
Used by both convert_my_emg.py (batch) and realtime_inference.py (single sample).

Pipeline: raw ADC → spike clean → resample → notch → highpass → HC features
          → STFT prenorm → robust norm (with saved stats) → clip
=============================================================================
"""

import numpy as np
import scipy.signal
import librosa


# ==========================================================================
# HC Feature Extraction
# ==========================================================================
def double_average(x):
    """Double 9-point moving average (smoothing filter)."""
    assert len(x.shape) == 1
    f = np.ones(9) / 9.0
    v = np.convolve(x, f, mode='same')
    w = np.convolve(v, f, mode='same')
    return w


def get_emg_features(emg_data, frame_length=16, hop_length=6):
    """Extract 14 HC features per channel from multi-channel EMG.

    Per channel: 5 time-domain + 9 frequency-domain (STFT) = 14 features.

    Args:
        emg_data: (num_samples, num_channels) array
        frame_length: STFT frame length (default 16)
        hop_length: STFT hop length (default 6)

    Returns:
        (num_frames, num_channels * 14) float32 array
    """
    xs = emg_data - emg_data.mean(axis=0, keepdims=True)
    frame_features = []
    for i in range(emg_data.shape[1]):
        x = xs[:, i]
        w = double_average(x)
        p = x - w
        r = np.abs(p)

        w_h = librosa.util.frame(w, frame_length=frame_length, hop_length=hop_length).mean(axis=0)
        p_w = librosa.feature.rms(y=w, frame_length=frame_length, hop_length=hop_length, center=False)
        p_w = np.squeeze(p_w, 0)
        p_r = librosa.feature.rms(y=r, frame_length=frame_length, hop_length=hop_length, center=False)
        p_r = np.squeeze(p_r, 0)
        z_p = librosa.feature.zero_crossing_rate(p, frame_length=frame_length, hop_length=hop_length, center=False)
        z_p = np.squeeze(z_p, 0)
        r_h = librosa.util.frame(r, frame_length=frame_length, hop_length=hop_length).mean(axis=0)

        s = abs(librosa.stft(np.ascontiguousarray(x), n_fft=frame_length, hop_length=hop_length, center=False))

        frame_features.append(np.stack([w_h, p_w, p_r, z_p, r_h], axis=1))
        frame_features.append(s.T)

    frame_features = np.concatenate(frame_features, axis=1)
    return frame_features.astype(np.float32)


# ==========================================================================
# Signal Processing
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
    """Notch filter at fundamental + harmonics (up to 7th)."""
    for harmonic in range(1, 8):
        if freq * harmonic < sample_frequency / 2:
            signal = notch(signal, freq * harmonic, sample_frequency)
    return signal


def subsample(signal, new_freq, old_freq):
    """Resample using linear interpolation."""
    times = np.arange(len(signal)) / old_freq
    sample_times = np.arange(0, times[-1], 1 / new_freq)
    result = np.interp(sample_times, times, signal)
    return result


def apply_to_all(function, signal_array, *args, **kwargs):
    """Apply function to each channel independently."""
    results = []
    for i in range(signal_array.shape[1]):
        results.append(function(signal_array[:, i], *args, **kwargs))
    return np.stack(results, 1)


# ==========================================================================
# Spike Cleaning
# ==========================================================================
def clean_spikes(emg, threshold_std=10):
    """Clean single-sample ADC spikes by interpolation.

    Detects outliers per channel (> threshold_std * std from median)
    and replaces with linear interpolation of neighbors.

    Args:
        emg: (num_samples, num_channels) array
        threshold_std: outlier threshold in standard deviations

    Returns:
        cleaned: (num_samples, num_channels) array
        total_cleaned: number of spikes cleaned
    """
    cleaned = emg.copy()
    total_cleaned = 0
    for ch in range(emg.shape[1]):
        signal = cleaned[:, ch]
        mean = np.median(signal)
        std = np.std(signal[np.abs(signal - mean) < 5 * np.std(signal)])

        spike_mask = np.abs(signal - mean) > threshold_std * std
        spike_indices = np.where(spike_mask)[0]

        if len(spike_indices) == 0:
            continue

        for si in spike_indices:
            left = signal[si - 1] if si > 0 else signal[si + 1]
            right = signal[si + 1] if si < len(signal) - 1 else signal[si - 1]
            cleaned[si, ch] = (left + right) / 2.0

        total_cleaned += len(spike_indices)

    return cleaned, total_cleaned


# ==========================================================================
# Normalization (for inference with pre-computed stats)
# ==========================================================================
def per_sample_stft_prenorm(features, n_channels):
    """Per-sample normalization of STFT features to equalize amplitude.

    For each channel, divides STFT features (indices 5-13 within the 14
    per-channel block) by their RMS.

    Args:
        features: (T, num_features) array
        n_channels: number of EMG channels

    Returns:
        features: (T, num_features) array with STFT features normalized
    """
    features = features.copy()
    for ch in range(n_channels):
        stft_start = ch * 14 + 5
        stft_end = ch * 14 + 14
        stft_block = features[:, stft_start:stft_end]
        rms = np.sqrt(np.mean(stft_block ** 2))
        if rms > 1e-8:
            features[:, stft_start:stft_end] = stft_block / rms
    return features


def normalize_with_stats(features, centers, scales, clip_value=5.0):
    """Apply pre-computed robust normalization + clipping.

    Args:
        features: (T, num_features) array
        centers: (num_features,) array — median per feature
        scales: (num_features,) array — IQR/1.3489 per feature
        clip_value: clip to [-clip_value, clip_value]

    Returns:
        (T, num_features) float32 array, clipped to [-clip_value, clip_value]
    """
    normed = ((features - centers) / scales).astype(np.float32)
    if clip_value > 0:
        normed = np.clip(normed, -clip_value, clip_value)
    return normed


# ==========================================================================
# Full Pipeline: raw CSV data → normalized features
# ==========================================================================
def process_raw_emg(raw_data, mouth_duration, drop_channels=None,
                    target_sr=516.79, mains_freq=50, clip_value=5.0,
                    norm_centers=None, norm_scales=None, stft_prenorm=True):
    """Process raw 4-channel EMG data into normalized HC features.

    Full pipeline: spike clean → drop channels → resample → filter → features
                   → STFT prenorm → robust norm → clip

    Args:
        raw_data: (num_samples, 4) array of raw ADC values
        mouth_duration: recording duration in seconds
        drop_channels: list of channel indices to drop (e.g., [3])
        target_sr: target sample rate (default 516.79 Hz, matches Gaddy)
        mains_freq: mains frequency for notch filter (50 Hz UK)
        clip_value: clip range (default 5.0)
        norm_centers: pre-computed normalization centers (from norm_stats.json)
        norm_scales: pre-computed normalization scales (from norm_stats.json)
        stft_prenorm: whether to apply per-sample STFT prenormalization

    Returns:
        features: (T, num_features) float32 array, normalized and clipped
        info: dict with processing metadata
    """
    n_samples = raw_data.shape[0]
    est_sr = n_samples / mouth_duration

    info = {
        'n_raw_samples': n_samples,
        'est_sr': est_sr,
    }

    # Step 1: Clean spikes
    raw_clean, n_cleaned = clean_spikes(raw_data, threshold_std=10)
    info['n_spikes_cleaned'] = n_cleaned

    # Step 2: Drop channels
    if drop_channels:
        keep = [c for c in range(raw_clean.shape[1]) if c not in drop_channels]
        raw_clean = raw_clean[:, keep]
    n_channels = raw_clean.shape[1]
    info['n_channels'] = n_channels

    # Step 3: Resample to fixed rate
    resampled = apply_to_all(subsample, raw_clean, target_sr, est_sr)

    # Step 4: Notch filter (mains + harmonics)
    filtered = apply_to_all(notch_harmonics, resampled, mains_freq, target_sr)

    # Step 5: Highpass filter (2 Hz, remove drift)
    filtered = apply_to_all(remove_drift, filtered, target_sr)

    # Step 6: Extract HC features
    features = get_emg_features(filtered)
    info['feature_shape'] = features.shape
    info['num_features'] = features.shape[1]

    # Step 7: Optional STFT pre-normalization (per-sample)
    if stft_prenorm:
        features = per_sample_stft_prenorm(features, n_channels)

    # Step 8: Robust normalization with saved stats
    if norm_centers is not None and norm_scales is not None:
        features = normalize_with_stats(features, norm_centers, norm_scales, clip_value)
        info['normalized'] = True
    else:
        info['normalized'] = False

    return features, info


def load_norm_stats(norm_stats_path):
    """Load pre-computed normalization statistics from norm_stats.json.

    The file uses 'mean'/'std' keys regardless of norm type.
    For robust normalization, 'mean' = median and 'std' = IQR/1.3489.

    Args:
        norm_stats_path: path to norm_stats.json

    Returns:
        centers: (num_features,) array
        scales: (num_features,) array
        metadata: dict with norm_type, num_features, etc.
    """
    import json
    with open(norm_stats_path) as f:
        stats = json.load(f)

    centers = np.array(stats['mean'])
    scales = np.array(stats['std'])
    metadata = {
        'norm_type': stats.get('norm_type', 'standard'),
        'stft_prenorm': stats.get('stft_prenorm', False),
        'clip': stats.get('clip', 0),
        'num_features': stats.get('num_features', len(centers)),
    }
    return centers, scales, metadata
