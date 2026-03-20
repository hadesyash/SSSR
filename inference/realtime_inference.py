#!/usr/bin/env python3
"""
=============================================================================
SilentSpeechLLM — Real-Time EMG Inference
=============================================================================

Runs real-time EMG-to-text inference for a closed vocabulary (e.g., 10 fruit words).
Loads the LLM + adaptor once at startup, then loops: record → features → predict.

Two modes:
  --live     : Record from Arduino via serial (default)
  --offline  : Test with saved .npy or .csv files (no hardware needed)

USAGE (live):
  python hpc/realtime_inference.py \
      --checkpoint checkpoints/fruit/ivan2_3ch_ms384/best_transNet.pth \
      --norm_stats data/ivan2_3ch/norm_stats.json \
      --serial_port /dev/tty.usbmodemF412FA6987F82

USAGE (offline, test with saved .npy):
  python hpc/realtime_inference.py \
      --checkpoint checkpoints/fruit/ivan2_3ch_ms384/best_transNet.pth \
      --norm_stats data/ivan2_3ch/norm_stats.json \
      --offline --test_file data/ivan2_3ch/0_silent.npy

USAGE (offline, test with saved .csv):
  python hpc/realtime_inference.py \
      --checkpoint checkpoints/fruit/ivan2_3ch_ms384/best_transNet.pth \
      --norm_stats data/ivan2_3ch/norm_stats.json \
      --offline --test_csv path/to/recording.csv --mouth_duration 2.0
=============================================================================
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModelForCausalLM, AutoTokenizer
from emg_features import (
    process_raw_emg,
    load_norm_stats,
)


# ==========================================================================
# Model definitions (copied from train_hc_llama3b_improved.py)
# ==========================================================================
class LstmBlock(nn.Module):
    def __init__(self, num_ins, lstm_embed_out=128, device=None):
        super().__init__()
        self.num_layers = 1
        self.lstm_embed_out = lstm_embed_out
        self._device = device
        self.lstm = nn.LSTM(num_ins, lstm_embed_out, batch_first=True,
                            bidirectional=True, num_layers=self.num_layers)

    def forward(self, x):
        h0 = torch.zeros(2 * self.num_layers, x.size(0), self.lstm_embed_out).to(self._device)
        c0 = torch.zeros(2 * self.num_layers, x.size(0), self.lstm_embed_out).to(self._device)
        x, _ = self.lstm(x, (h0, c0))
        return x


class ResBlock(nn.Module):
    def __init__(self, num_ins, num_outs, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(num_ins, num_outs, kernel_size=3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm1d(num_outs)
        self.conv2 = nn.Conv1d(num_outs, num_outs, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(num_outs)
        if stride != 1 or num_ins != num_outs:
            self.residual_path = nn.Conv1d(num_ins, num_outs, kernel_size=1, stride=stride)
            self.res_norm = nn.BatchNorm1d(num_outs)
        else:
            self.residual_path = None

    def forward(self, x):
        input_value = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        if self.residual_path is not None:
            res = self.res_norm(self.residual_path(input_value))
        else:
            res = input_value
        return F.relu(x + res)


class TransductionModelImproved(nn.Module):
    def __init__(self, model_size=768, num_features=112, llm_hidden=3072, dropout=0.15):
        super().__init__()
        self.conv1D_decim_1 = nn.Sequential(
            nn.Conv1d(num_features, num_features, kernel_size=6, padding=0, stride=6),
            nn.GELU()
        )
        self.conv_blocks = nn.Sequential(
            ResBlock(num_features, model_size, stride=1),
            ResBlock(model_size, model_size // 2, stride=2),
            ResBlock(model_size // 2, model_size // 4, stride=2),
        )
        self.conv1D_decim = nn.Sequential(
            nn.Conv1d(model_size // 2, model_size // 2, kernel_size=2, padding=0, stride=2),
            nn.GELU()
        )
        self.w_raw_in = nn.Linear(model_size // 4, model_size // 4)
        self.lstm = LstmBlock(model_size // 4, model_size // 4, None)  # device set later
        self.w_out = nn.Linear(model_size // 2, llm_hidden)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.output_norm = nn.LayerNorm(llm_hidden)

    def forward(self, x_feat, x_raw, session_ids):
        x_raw = x_raw.transpose(1, 2)
        x_raw = self.conv1D_decim_1(x_raw)
        x_raw = self.conv_blocks(x_raw)
        x_raw = self.drop1(x_raw)
        x_raw = self.lstm(x_raw.permute(0, 2, 1)).permute(0, 2, 1)
        x_raw = self.conv1D_decim(x_raw)
        x_raw = self.drop2(x_raw)
        x_raw = x_raw.transpose(1, 2)
        out = self.w_out(x_raw)
        out = self.output_norm(out)
        return out


# ==========================================================================
# Serial recording (from host.py protocol)
# ==========================================================================
MAGIC_HEADER = b'\xde\xad'
PAYLOAD_SIZE = 256  # bytes per packet: 32 samples x 4ch x 2 bytes


def find_header(ser):
    """Find the 0xDEAD magic header in serial stream."""
    while True:
        b = ser.read(1)
        if b == b'\xde':
            b2 = ser.read(1)
            if b2 == b'\xad':
                return True


def record_emg(ser, duration):
    """Record EMG from Arduino for a fixed duration.

    Args:
        ser: serial.Serial object
        duration: recording duration in seconds

    Returns:
        raw_data: (num_samples, 4) numpy array of raw ADC values
    """
    import serial as pyserial

    # Flush and start
    ser.reset_input_buffer()
    ser.write(b'S')

    raw_bytes = bytearray()
    start_time = time.time()

    while time.time() - start_time < duration:
        n = ser.in_waiting
        if n > 0:
            raw_bytes.extend(ser.read(n))
        else:
            time.sleep(0.001)

    # Stop recording
    ser.write(b'E')
    time.sleep(0.05)
    # Drain trailing bytes still in USB buffer
    n_tail = ser.in_waiting
    if n_tail > 0:
        raw_bytes.extend(ser.read(n_tail))
    ser.reset_input_buffer()

    if not raw_bytes:
        return np.zeros((0, 4), dtype=np.float64)

    def parse_with_header(buffer_bytes, h0, h1):
        packets = []
        i = 0
        n = len(buffer_bytes)
        # Sliding parse: tolerate misalignment and garbage bytes.
        while i + 2 + PAYLOAD_SIZE <= n:
            if buffer_bytes[i] == h0 and buffer_bytes[i + 1] == h1:
                payload = buffer_bytes[i + 2:i + 2 + PAYLOAD_SIZE]
                if len(payload) == PAYLOAD_SIZE:
                    packets.append(np.frombuffer(payload, dtype='<u2').reshape(-1, 4))
                i += 2 + PAYLOAD_SIZE
            else:
                i += 1
        return packets

    # Preferred format: 0xDEAD header + 256-byte payload
    packets = parse_with_header(raw_bytes, 0xDE, 0xAD)
    # Fallback for devices that emit reversed header order
    if not packets:
        packets = parse_with_header(raw_bytes, 0xAD, 0xDE)
    if packets:
        return np.vstack(packets).astype(np.float64)

    # Final fallback: header-less stream of uint16[4] samples
    sample_bytes = 4 * 2
    usable = (len(raw_bytes) // sample_bytes) * sample_bytes
    if usable <= 0:
        return np.zeros((0, 4), dtype=np.float64)
    raw = np.frombuffer(raw_bytes[:usable], dtype='<u2').reshape(-1, 4)
    return raw.astype(np.float64)


# ==========================================================================
# Constrained decoding (forced scoring with KV-cache)
# ==========================================================================
@torch.no_grad()
def score_words(emg_embed, word_token_ids, llm, embed_fn, tokenizer,
                prompt_before, prompt_after, device):
    """Score each candidate word given EMG embeddings.

    Uses KV-cache: one main forward pass for shared prefix,
    then tiny cached passes for each word's tokens.

    Args:
        emg_embed: (1, T', llm_hidden) adaptor output
        word_token_ids: dict of {word: [token_id, ...]}
        llm: the LLM model
        embed_fn: embedding function (e.g., llm.model.embed_tokens)
        tokenizer: the tokenizer
        prompt_before: (1, N1, hidden) prompt embedding before EMG
        prompt_after: (1, N2, hidden) prompt embedding after EMG
        device: torch device

    Returns:
        scores: dict of {word: log_probability}
        confidences: dict of {word: softmax_probability}
    """
    # Build shared prefix: [BOS] + [prompt1] + [emg_embed] + [prompt2]
    bos = torch.ones([1, 1], dtype=torch.long, device=device) * tokenizer.bos_token_id
    bos_embeds = embed_fn(bos)

    prefix = torch.cat([bos_embeds, prompt_before, emg_embed, prompt_after], dim=1)
    attn_mask = torch.ones(prefix.shape[:2], dtype=torch.long, device=device)

    # Main forward pass with KV-cache
    outputs = llm(inputs_embeds=prefix, attention_mask=attn_mask, use_cache=True)
    past_kv = outputs.past_key_values
    first_logits = outputs.logits[0, -1, :]  # logits predicting first output token
    first_log_probs = F.log_softmax(first_logits, dim=-1)

    scores = {}
    for word, token_ids in word_token_ids.items():
        log_prob = first_log_probs[token_ids[0]].item()

        if len(token_ids) > 1:
            # Score remaining tokens using KV-cache
            cached_kv = past_kv
            cached_attn = attn_mask
            for i in range(1, len(token_ids)):
                prev_token = torch.tensor([[token_ids[i - 1]]], device=device)
                prev_embed = embed_fn(prev_token)
                cached_attn = torch.cat([cached_attn,
                                         torch.ones([1, 1], dtype=torch.long, device=device)], dim=1)
                out = llm(inputs_embeds=prev_embed,
                          attention_mask=cached_attn,
                          past_key_values=cached_kv,
                          use_cache=True)
                cached_kv = out.past_key_values
                token_logits = out.logits[0, -1, :]
                token_log_probs = F.log_softmax(token_logits, dim=-1)
                log_prob += token_log_probs[token_ids[i]].item()

        scores[word] = log_prob

    # Softmax over scores for confidence
    words = list(scores.keys())
    score_tensor = torch.tensor([scores[w] for w in words])
    probs = F.softmax(score_tensor, dim=0)
    confidences = {w: probs[i].item() for i, w in enumerate(words)}

    return scores, confidences


# ==========================================================================
# Text-to-Speech
# ==========================================================================
def speak(text, voice='Fred', blocking=False):
    """Speak text using macOS 'say' command.

    Args:
        text: text to speak
        voice: macOS voice name (e.g. Albert, Fred, Daniel)
        blocking: if True, wait for speech to finish before returning
    """
    if platform.system() != 'Darwin':
        return  # Only works on macOS
    try:
        fn = subprocess.run if blocking else subprocess.Popen
        fn(['say', '-v', voice, text],
           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass  # Silently ignore TTS failures


# ==========================================================================
# Main
# ==========================================================================
def main():
    parser = argparse.ArgumentParser(description='Real-time EMG-to-text inference')

    # Model args
    parser.add_argument('--checkpoint', required=True,
                        help='Path to best_transNet.pth checkpoint')
    parser.add_argument('--norm_stats', required=True,
                        help='Path to norm_stats.json from training conversion')
    parser.add_argument('--llm_path', default='google/gemma-2-2b-it',
                        help='LLM model path (default: google/gemma-2-2b-it)')
    parser.add_argument('--num_features', type=int, default=42,
                        help='Number of input features (default: 42 for 3ch)')
    parser.add_argument('--model_size', type=int, default=384,
                        help='Adaptor model size (default: 384)')
    parser.add_argument('--dropout', type=float, default=0.15,
                        help='Dropout (default: 0.15, not used at inference)')

    # Recording args
    parser.add_argument('--drop_channels', nargs='+', type=int, default=[3],
                        help='Channel indices to drop (default: [3])')
    parser.add_argument('--serial_port', default='/dev/tty.usbmodemF412FA6987F82',
                        help='Serial port for Arduino')
    parser.add_argument('--baud_rate', type=int, default=115200,
                        help='Serial baud rate (default: 115200)')
    parser.add_argument('--mouth_duration', type=float, default=2.0,
                        help='Recording duration in seconds (default: 2.0)')
    parser.add_argument('--mains_freq', type=float, default=50.0,
                        help='Mains frequency for notch filter (default: 50 Hz)')
    # Vocabulary
    parser.add_argument('--vocab', nargs='+',
                        default=['artichoke', 'asparagus', 'cauliflower', 'cranberry',
                                 'dragonfruit', 'guava', 'mangosteen', 'pomegranate',
                                 'sprouts', 'zucchini'],
                        help='Vocabulary words for constrained decoding')

    # Mode
    parser.add_argument('--offline', action='store_true',
                        help='Offline mode (no Arduino needed)')
    parser.add_argument('--test_file', default=None,
                        help='Path to .npy file for offline testing (pre-extracted features)')
    parser.add_argument('--test_csv', default=None,
                        help='Path to .csv file for offline testing (raw EMG)')
    parser.add_argument('--test_dir', default=None,
                        help='Path to feature directory for batch offline testing')

    # TTS
    parser.add_argument('--voice', default='Fred',
                        help='macOS say voice (default: Fred)')
    parser.add_argument('--no_speak', action='store_true',
                        help='Disable text-to-speech')

    # Device
    parser.add_argument('--device', default='auto',
                        help='Device: auto, cpu, cuda, mps')

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Resolve device
    # ------------------------------------------------------------------
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)

    print(f"\n{'='*60}")
    print(f"  SilentSpeechLLM — Real-Time Inference")
    print(f"{'='*60}")
    print(f"  Device:        {device}")
    print(f"  LLM:           {args.llm_path}")
    print(f"  Checkpoint:    {args.checkpoint}")
    print(f"  Norm stats:    {args.norm_stats}")
    print(f"  Num features:  {args.num_features}")
    print(f"  Model size:    {args.model_size}")
    print(f"  Drop channels: {args.drop_channels}")
    print(f"  Vocabulary:    {len(args.vocab)} words")
    print(f"  Mode:          {'offline' if args.offline else 'live'}")
    print(f"  TTS:           {'off' if args.no_speak else f'{args.voice} voice'}")
    print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # Phase 1: Load everything
    # ------------------------------------------------------------------
    t_start = time.time()

    # 1a. Load LLM
    print("[1/5] Loading LLM...", end=' ', flush=True)
    t0 = time.time()

    # Use float16 everywhere (Apple Silicon MPS supports float16 fine)
    llm_dtype = torch.float16

    tokenizer = AutoTokenizer.from_pretrained(args.llm_path, use_fast=True)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.padding_side = "right"

    if device.type == 'mps':
        # MPS: load to CPU first, then move to device (avoids allocator issues)
        llm = AutoModelForCausalLM.from_pretrained(
            args.llm_path,
            torch_dtype=llm_dtype,
            low_cpu_mem_usage=True,
        )
        llm = llm.to(device)
    else:
        llm = AutoModelForCausalLM.from_pretrained(
            args.llm_path,
            torch_dtype=llm_dtype,
            device_map={"": device},
        )
    llm.eval()
    for p in llm.parameters():
        p.requires_grad = False

    # Get embedding function
    if hasattr(llm, 'model') and hasattr(llm.model, 'embed_tokens'):
        embed_fn = llm.model.embed_tokens
    elif hasattr(llm, 'transformer') and hasattr(llm.transformer, 'wte'):
        embed_fn = llm.transformer.wte
    else:
        raise ValueError(f"Cannot find embed_tokens for {type(llm)}")

    llm_hidden = llm.config.hidden_size
    print(f"done ({time.time()-t0:.1f}s, hidden={llm_hidden})")

    # 1b. Load adaptor
    print("[2/5] Loading adaptor...", end=' ', flush=True)
    t0 = time.time()

    adaptor = TransductionModelImproved(
        model_size=args.model_size,
        num_features=args.num_features,
        llm_hidden=llm_hidden,
        dropout=args.dropout,
    )

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if isinstance(ckpt, dict) and 'transNet_state_dict' in ckpt:
        adaptor.load_state_dict(ckpt['transNet_state_dict'])
    else:
        adaptor.load_state_dict(ckpt)

    adaptor.to(device).eval()
    # Update LSTM device
    adaptor.lstm._device = device
    n_params = sum(p.numel() for p in adaptor.parameters())
    print(f"done ({time.time()-t0:.1f}s, {n_params:,} params)")

    # 1c. Load normalization stats
    print("[3/5] Loading norm stats...", end=' ', flush=True)
    norm_centers, norm_scales, norm_meta = load_norm_stats(args.norm_stats)
    print(f"done ({norm_meta['num_features']} features, {norm_meta['norm_type']} norm)")

    # 1d. Pre-compute prompt embeddings
    print("[4/5] Pre-computing prompts...", end=' ', flush=True)
    t0 = time.time()
    with torch.no_grad():
        emg_prompt = 'Unvoiced EMG :'
        prompt_text = 'Prompt : Convert unvoiced EMG to text'

        prompt_before_ids = tokenizer(
            emg_prompt, return_tensors="pt", padding="longest",
            truncation=True, max_length=128, add_special_tokens=False
        ).to(device)
        prompt_after_ids = tokenizer(
            prompt_text, return_tensors="pt",
            truncation=True, add_special_tokens=False
        ).to(device)

        prompt_before = embed_fn(prompt_before_ids.input_ids)
        prompt_after = embed_fn(prompt_after_ids.input_ids)
    print(f"done ({time.time()-t0:.1f}s)")

    # 1e. Pre-tokenize vocabulary
    print("[5/5] Tokenizing vocabulary...", end=' ', flush=True)
    word_token_ids = {}
    for word in args.vocab:
        ids = tokenizer.encode(word, add_special_tokens=False)
        word_token_ids[word] = ids
        print(f"\n        {word:15s} -> {len(ids)} tokens: {ids}", end='')
    print(f"\n       done")

    total_startup = time.time() - t_start
    print(f"\n  Startup complete in {total_startup:.1f}s")
    print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # Phase 2: Inference
    # ------------------------------------------------------------------

    def run_inference(features_np):
        """Run inference on pre-extracted features.

        Args:
            features_np: (T, num_features) numpy array, normalized

        Returns:
            predicted_word, confidence, all_confidences, elapsed
        """
        t0 = time.time()

        # To tensor
        features_t = torch.tensor(features_np, dtype=torch.float32).unsqueeze(0).to(device)

        # Adaptor forward
        with torch.no_grad():
            emg_embed = adaptor(None, features_t, None)
            # Cast to LLM dtype (float16) to avoid dtype mismatch in cat
            emg_embed = emg_embed.to(prompt_before.dtype)

        # Constrained scoring
        scores, confidences = score_words(
            emg_embed, word_token_ids, llm, embed_fn, tokenizer,
            prompt_before, prompt_after, device
        )

        # Best prediction
        best_word = max(confidences, key=confidences.get)
        best_conf = confidences[best_word]

        elapsed = time.time() - t0
        return best_word, best_conf, confidences, elapsed

    # ---- Offline mode ----
    if args.offline:
        if args.test_file:
            # Single .npy file
            print(f"  Loading: {args.test_file}")
            features = np.load(args.test_file)
            print(f"  Shape: {features.shape}, range: [{features.min():.2f}, {features.max():.2f}]")

            word, conf, all_confs, elapsed = run_inference(features)

            if not args.no_speak:
                speak(word, voice=args.voice)

            print(f"\n  {'='*50}")
            print(f"  Predicted: {word} ({conf*100:.1f}%)")
            print(f"  Inference: {elapsed*1000:.0f}ms")
            print(f"  {'='*50}")
            print(f"\n  All probabilities:")
            for w, c in sorted(all_confs.items(), key=lambda x: -x[1]):
                bar = '#' * int(c * 40)
                print(f"    {w:15s} {c*100:5.1f}% {bar}")

        elif args.test_csv:
            # Single .csv file — process from raw
            print(f"  Loading CSV: {args.test_csv}")
            import csv
            with open(args.test_csv) as f:
                reader = csv.reader(f)
                header = next(reader)
                raw = np.array([[float(x) for x in r] for r in reader])
            print(f"  Raw shape: {raw.shape}")

            t0 = time.time()
            features, info = process_raw_emg(
                raw, args.mouth_duration,
                drop_channels=args.drop_channels,
                norm_centers=norm_centers, norm_scales=norm_scales,
            )
            feat_time = time.time() - t0
            print(f"  Feature extraction: {feat_time*1000:.0f}ms")
            print(f"  Features shape: {features.shape}, range: [{features.min():.2f}, {features.max():.2f}]")

            word, conf, all_confs, elapsed = run_inference(features)

            if not args.no_speak:
                speak(word, voice=args.voice)

            print(f"\n  {'='*50}")
            print(f"  Predicted: {word} ({conf*100:.1f}%)")
            print(f"  Feature extraction: {feat_time*1000:.0f}ms")
            print(f"  Model inference:    {elapsed*1000:.0f}ms")
            print(f"  Total:              {(feat_time+elapsed)*1000:.0f}ms")
            print(f"  {'='*50}")
            print(f"\n  All probabilities:")
            for w, c in sorted(all_confs.items(), key=lambda x: -x[1]):
                bar = '#' * int(c * 40)
                print(f"    {w:15s} {c*100:5.1f}% {bar}")

        elif args.test_dir:
            # Batch test on a feature directory
            split_path = os.path.join(args.test_dir, 'split.json')
            with open(split_path) as f:
                split = json.load(f)
            dev_keys = split['dev_sentence_indices']

            print(f"  Testing {len(dev_keys)} dev samples from {args.test_dir}\n")

            correct = 0
            total = 0
            for key in dev_keys:
                npy_path = os.path.join(args.test_dir, f'{key}_silent.npy')
                json_path = os.path.join(args.test_dir, f'{key}.json')

                features = np.load(npy_path)
                with open(json_path) as f:
                    meta = json.load(f)
                ground_truth = meta['text']

                word, conf, all_confs, elapsed = run_inference(features)
                is_correct = word == ground_truth
                correct += int(is_correct)
                total += 1

                if not args.no_speak:
                    speak(word, voice=args.voice, blocking=True)

                status = "✓" if is_correct else "✗"
                print(f"  [{status}] {key:>4s}: pred={word:15s} ({conf*100:5.1f}%)  "
                      f"truth={ground_truth:15s}  {elapsed*1000:.0f}ms")

            accuracy = correct / total if total > 0 else 0
            print(f"\n  {'='*50}")
            print(f"  Accuracy: {correct}/{total} = {accuracy*100:.1f}%")
            print(f"  {'='*50}")

        else:
            print("  Error: --offline requires --test_file, --test_csv, or --test_dir")
            sys.exit(1)

        return

    # ---- Live mode ----
    try:
        import serial as pyserial
    except ImportError:
        print("  Error: pyserial not installed. Run: pip install pyserial")
        sys.exit(1)

    print(f"  Opening serial port: {args.serial_port}")
    try:
        ser = pyserial.Serial(args.serial_port, args.baud_rate, timeout=0.1)
    except Exception as e:
        print(f"  Error opening serial port: {e}")
        print(f"  Check that Arduino is connected and port is correct.")
        sys.exit(1)

    time.sleep(1)  # Wait for Arduino to reset
    ser.reset_input_buffer()
    print(f"  Serial connected. Ready for inference.\n")

    inference_count = 0
    try:
        while True:
            print(f"  Press ENTER to record ({args.mouth_duration}s), or 'q' + ENTER to quit: ",
                  end='', flush=True)
            user_input = input().strip().lower()
            if user_input == 'q':
                break

            # Record
            print(f"  Recording for {args.mouth_duration}s...", end=' ', flush=True)
            raw_data = record_emg(ser, args.mouth_duration)
            print(f"got {raw_data.shape[0]} samples ({raw_data.shape[0]/args.mouth_duration:.0f} Hz)")

            if raw_data.shape[0] < 100:
                print(f"  Warning: too few samples ({raw_data.shape[0]}). Skipping.\n")
                continue

            # Feature extraction
            t0 = time.time()
            features, info = process_raw_emg(
                raw_data, args.mouth_duration,
                drop_channels=args.drop_channels,
                norm_centers=norm_centers, norm_scales=norm_scales,
            )
            feat_time = time.time() - t0

            print(f"  Features: {features.shape}, "
                  f"range [{features.min():.2f}, {features.max():.2f}], "
                  f"{feat_time*1000:.0f}ms")

            # Inference
            word, conf, all_confs, model_time = run_inference(features)

            inference_count += 1
            total_time = feat_time + model_time

            if not args.no_speak:
                speak(word, voice=args.voice)

            print(f"\n  {'='*50}")
            print(f"  #{inference_count} Predicted: {word} ({conf*100:.1f}%)")
            print(f"  Feature extraction: {feat_time*1000:.0f}ms")
            print(f"  Model inference:    {model_time*1000:.0f}ms")
            print(f"  Total:              {total_time*1000:.0f}ms")
            print(f"  {'='*50}")

            # Show top 3
            sorted_confs = sorted(all_confs.items(), key=lambda x: -x[1])
            print(f"  Top 3:")
            for w, c in sorted_confs[:3]:
                bar = '#' * int(c * 40)
                print(f"    {w:15s} {c*100:5.1f}% {bar}")
            print()

    except KeyboardInterrupt:
        print("\n\n  Interrupted.")
    finally:
        ser.close()
        print(f"  Serial port closed. {inference_count} inferences performed.")


if __name__ == '__main__':
    main()
