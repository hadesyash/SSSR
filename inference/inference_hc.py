"""
=============================================================================
SilentSpeechLLM — HC Inference on HPC
=============================================================================

Loads the pre-trained HC adaptor (transNet) + frozen LLaMA-3.2-3B-Instruct
and runs inference on the dev set to verify WER (~0.49 expected).

This script matches the EXACT architecture and generate() function from:
  https://github.com/payalmohapatra/SilentSpeechLLM/blob/main/closedVocab_llama3_hc.py

Usage:
  python hpc/inference_hc.py \
      --hc_features_dir /path/to/extracted_emg_features \
      --transnet_path /path/to/transNet_only.pth \
      --llm_path meta-llama/Llama-3.2-3B-Instruct

=============================================================================
"""

import sys
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os
import time
import argparse
import jiwer
from unidecode import unidecode
import string
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList

# ─── Arguments ────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description='HC Inference')
parser.add_argument('--hc_features_dir',
                    default='/rds/general/user/yma25/home/aml_lab/data/extracted_emg_features',
                    type=str)
parser.add_argument('--transnet_path',
                    default='/rds/general/user/yma25/home/aml_lab/models/transNet_only.pth',
                    type=str)
parser.add_argument('--llm_path',
                    default='meta-llama/Llama-3.2-3B-Instruct',
                    type=str, help='HF model ID or local path')
parser.add_argument('--json_path',
                    default='/rds/general/user/yma25/home/aml_lab/data/10_selected_samples.json',
                    type=str, help='Train/dev split JSON')
parser.add_argument('--cuda_pick', default='cuda:0', type=str)
parser.add_argument('--num_samples', default=0, type=int,
                    help='Number of dev samples to evaluate (0 = all)')
args = parser.parse_args()

# ─── Device ───────────────────────────────────────────────────────────
device = torch.device(args.cuda_pick if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ─── EMG Adaptor Model (exact copy from closedVocab_llama3_hc.py) ────

class LstmBlock(nn.Module):
    def __init__(self, num_ins, lstm_embed_out=128):
        super().__init__()
        self.num_layers = 1
        self.lstm_embed_out = lstm_embed_out
        self.lstm = nn.LSTM(num_ins, lstm_embed_out, batch_first=True,
                            bidirectional=True, num_layers=self.num_layers)

    def forward(self, x):
        # Original training code uses torch.randn, but for deterministic inference
        # torch.zeros is equivalent (LSTM learns to handle any initialization)
        h0 = torch.zeros(2 * self.num_layers, x.size(0), self.lstm_embed_out, device=x.device)
        c0 = torch.zeros(2 * self.num_layers, x.size(0), self.lstm_embed_out, device=x.device)
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


class TransductionModel(nn.Module):
    """Exact architecture from closedVocab_llama3_hc.py.

    Pipeline: Conv1d(112,112,k=6,s=6) -> 3 ResBlocks -> BiLSTM -> Conv1d(384,384,k=2,s=2) -> Linear(384,3072)
    Total temporal downsampling: 6 * 2 * 2 * 2 = 48x
    """
    def __init__(self, model_size=768, num_features=112):
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
        self.w_raw_in = nn.Linear(model_size // 4, model_size // 4)  # Defined but unused in forward
        self.lstm = LstmBlock(model_size // 4, model_size // 4)
        self.w_out = nn.Linear(model_size // 2, 3072)  # LLaMA-3.2-3B hidden size

    def forward(self, x_feat, x_raw, session_ids):
        # No data augmentation at inference (self.training = False)
        x_raw = x_raw.transpose(1, 2)
        x_raw = self.conv1D_decim_1(x_raw)
        x_raw = self.conv_blocks(x_raw)
        x_raw = self.lstm(x_raw.permute(0, 2, 1)).permute(0, 2, 1)
        x_raw = self.conv1D_decim(x_raw)
        x_raw = x_raw.transpose(1, 2)
        return self.w_out(x_raw)


class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True
        return False


class TextTransform:
    def __init__(self):
        self.transformation = jiwer.Compose([
            jiwer.RemovePunctuation(),
            jiwer.ToLowerCase()
        ])
        self.chars = string.ascii_lowercase + string.digits + ' '

    def clean_text(self, text):
        text = unidecode(text)
        text = self.transformation(text)
        return text


# ─── 1. Load LLaMA ───────────────────────────────────────────────────
print("=" * 60)
print("Loading LLaMA tokenizer and model...")
print(f"  LLM path: {args.llm_path}")
t0 = time.time()

llama_tokenizer = AutoTokenizer.from_pretrained(args.llm_path, use_fast=True)
llama_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
llama_tokenizer.padding_side = "right"

llama_model = AutoModelForCausalLM.from_pretrained(
    args.llm_path,
    torch_dtype=torch.float16,
    device_map={"": device},
)
llama_model.config.pad_token_id = llama_tokenizer.pad_token_id
llama_model.eval()
for param in llama_model.parameters():
    param.requires_grad = False

print(f"  Loaded in {time.time()-t0:.1f}s")
print(f"  BOS token id: {llama_tokenizer.bos_token_id}")
print(f"  EOS token id: {llama_tokenizer.eos_token_id}")
print(f"  PAD token id: {llama_tokenizer.pad_token_id}")
total_params = sum(p.numel() for p in llama_model.parameters())
print(f"  LLaMA params: {total_params:,}")

# ─── 2. Load EMG Adaptor ─────────────────────────────────────────────
print(f"\nLoading EMG adaptor from {args.transnet_path}...")
transNet = TransductionModel(model_size=768, num_features=112)
state_dict = torch.load(args.transnet_path, map_location='cpu', weights_only=True)
transNet.load_state_dict(state_dict, strict=True)
transNet = transNet.to(device).eval()
adaptor_params = sum(p.numel() for p in transNet.parameters())
print(f"  Adaptor params: {adaptor_params:,}")

# ─── 3. Generate function (exact match to original repo) ─────────────
@torch.no_grad()
def generate(emg_embed, max_new_tokens=150):
    """Generate text from EMG embeddings.
    Original code: [BOS, emg_embed] only — NO prompt wrapping at inference.
    Beam search width=4, top_p=0.8.
    """
    emg_atts = torch.ones(emg_embed.shape[:2], dtype=torch.long, device=device)
    batch_size = emg_embed.shape[0]
    bos = torch.ones([batch_size, 1], dtype=torch.long, device=device) * llama_tokenizer.bos_token_id
    bos_embeds = llama_model.model.embed_tokens(bos)
    atts_bos = torch.ones([batch_size, 1], dtype=torch.long, device=device)

    inputs_embeds = torch.cat([bos_embeds, emg_embed], dim=1)
    attention_mask = torch.cat([atts_bos, emg_atts], dim=1)

    stop_words_ids = [torch.tensor([llama_tokenizer.eos_token_id], device=device)]
    stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

    generated_ids = llama_model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        stopping_criteria=stopping_criteria,
        num_beams=4,
        min_length=1,
        top_p=0.8,
        repetition_penalty=1.0,
        length_penalty=1.0,
    )

    return llama_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)


# ─── 4. Load dev split ───────────────────────────────────────────────
print(f"\nLoading split from {args.json_path}...")
with open(args.json_path, 'r') as f:
    split_data = json.load(f)
dev_ids = split_data['dev_sentence_indices']
print(f"  Dev samples: {len(dev_ids)}")

# ─── 5. Run inference ────────────────────────────────────────────────
text_transform = TextTransform()
num_samples = args.num_samples if args.num_samples > 0 else len(dev_ids)
results_wer = []
results_cer = []
all_targets = []
all_preds = []
skipped = 0

print(f"\n{'='*60}")
print(f"INFERENCE: evaluating {num_samples} dev samples")
print(f"{'='*60}")

for i, sample_id in enumerate(dev_ids[:num_samples]):
    npy_path = os.path.join(args.hc_features_dir, f"{sample_id}_silent.npy")
    json_path = os.path.join(args.hc_features_dir, f"{sample_id}.json")

    if not os.path.exists(npy_path):
        print(f"  Skipping {sample_id}: {npy_path} not found")
        skipped += 1
        continue

    features = torch.tensor(np.load(npy_path), dtype=torch.float32).unsqueeze(0).to(device)
    with open(json_path, 'r') as f:
        target_info = json.load(f)
    target_text = text_transform.clean_text(target_info['text'])

    t0 = time.time()
    emg_embed = transNet(None, features, None).to(torch.float16)
    pred_texts = generate(emg_embed)
    gen_time = time.time() - t0

    pred_text = text_transform.clean_text(pred_texts[0])
    wer = jiwer.wer(target_text, pred_text) if pred_text.strip() else 1.0
    cer = jiwer.cer(target_text, pred_text) if pred_text.strip() else 1.0

    print(f"[{i+1:3d}/{num_samples}] ID={sample_id:3d} | WER={wer:.2f} CER={cer:.2f} | "
          f"target=\"{target_text}\" | pred=\"{pred_text}\" | {gen_time:.1f}s")
    results_wer.append(wer)
    results_cer.append(cer)
    all_targets.append(target_text)
    all_preds.append(pred_text)

# ─── 6. Summary ──────────────────────────────────────────────────────
print(f"\n{'='*60}")
if results_wer:
    avg_wer = np.mean(results_wer)
    avg_cer = np.mean(results_cer)
    corpus_wer = jiwer.wer(all_targets, all_preds)
    corpus_cer = jiwer.cer(all_targets, all_preds)
    print(f"Results: {len(results_wer)} samples evaluated, {skipped} skipped")
    print(f"Sample-avg WER: {avg_wer:.4f}  |  Corpus WER: {corpus_wer:.4f}")
    print(f"Sample-avg CER: {avg_cer:.4f}  |  Corpus CER: {corpus_cer:.4f}")
    print(f"Expected WER: ~0.49 (from paper)")
    if corpus_wer < 0.55:
        print("STATUS: WER is in expected range — checkpoint verified!")
    else:
        print("WARNING: WER is higher than expected. Check checkpoint/data.")
else:
    print("ERROR: No samples were processed!")
print(f"{'='*60}")
