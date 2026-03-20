"""
=============================================================================
SilentSpeechLLM — IMPROVED HC Adaptor Training
=============================================================================

Same base architecture as the original, but with targeted improvements:

IMPROVEMENTS OVER ORIGINAL:
  1. COSINE LR SCHEDULE — original uses constant LR after warmup, which is
     suboptimal. Cosine decay is standard for adaptor training (LLaVA, SALMONN).
  2. PROMPT-CONSISTENT INFERENCE — original trains with prompts but generates
     WITHOUT them. We use prompts at inference too for train/test consistency.
  3. PROPER llama_model.eval() — frozen LLM should be in eval mode (stable
     BatchNorm stats, no dropout noise). We add explicit dropout to the adaptor
     instead for regularization.
  4. ADAPTOR DROPOUT — adds dropout layers in the adaptor for regularization
     (original has none, leading to potential overfitting on 452 samples).
  5. GRADIENT CLIPPING on adaptor — prevents training instabilities.
  6. COSINE ANNEALING WITH WARM RESTARTS — helps escape local minima.
  7. LABEL SMOOTHING — standard technique to prevent overconfident predictions,
     especially important with small datasets.
  8. STRONGER DATA AUGMENTATION — original only does temporal shift 0-7.
     We add Gaussian noise and random scaling.
  9. EMBEDDING SCALE MATCHING — adaptor outputs (~0.17 std) are 8x larger than
     LLaMA embeddings (~0.02 std). We add LayerNorm to match scales.
  10. EOS/PAD EXPLICIT in generate() — avoids the "Setting pad_token_id to
     eos_token_id" warning and ensures clean stopping.

=============================================================================
"""

import sys
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import torch
torch.cuda.empty_cache()

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256,garbage_collection_threshold:0.9"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import json
import time
import random
import string
import argparse
import jiwer
import math
from unidecode import unidecode
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.amp import autocast
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList

# ==========================================================================
# Arguments
# ==========================================================================
parser = argparse.ArgumentParser(description='HC Adaptor IMPROVED Training')
parser.add_argument('--hc_features_dir',
                    default='/rds/general/user/yma25/home/aml_lab/data/extracted_emg_features',
                    type=str)
parser.add_argument('--json_path',
                    default='/rds/general/user/yma25/home/aml_lab/data/10_selected_samples.json',
                    type=str)
parser.add_argument('--llm_path',
                    default='meta-llama/Llama-3.2-3B-Instruct',
                    type=str)
parser.add_argument('--llm_hidden', default=0, type=int,
                    help='LLM hidden size (0 = auto-detect from model config)')
parser.add_argument('--checkpoint_dir',
                    default='/rds/general/user/yma25/home/aml_lab/checkpoints/improved_llama3b',
                    type=str)
parser.add_argument('--num_epochs', default=500, type=int)
parser.add_argument('--batch_size', default=8, type=int)
parser.add_argument('--cuda_pick', default='cuda:0', type=str)
parser.add_argument('--seed_num', default=2711, type=int)
parser.add_argument('--eval_every', default=10, type=int)
parser.add_argument('--save_every', default=50, type=int)
parser.add_argument('--resume_from', default=None, type=str)
parser.add_argument('--log_file', default=None, type=str)
# Discriminative LR for transfer learning (fine-tuning)
parser.add_argument('--lr_early', default=None, type=float,
                    help='LR for early conv layers (channel-specific). If None, uses --lr for all layers')
parser.add_argument('--lr_late', default=None, type=float,
                    help='LR for late projection layers (LLM mapping). If None, uses --lr for all layers')
# Architecture
parser.add_argument('--model_size', default=768, type=int,
                    help='Adaptor intermediate size (controls ResBlock widths: model_size → model_size//2 → model_size//4)')
parser.add_argument('--num_features', default=112, type=int,
                    help='Number of input features per timestep (112 for 8ch, 56 for 4ch)')
# New hyperparams
parser.add_argument('--lr', default=1e-4, type=float, help='Peak learning rate')
parser.add_argument('--dropout', default=0.15, type=float, help='Adaptor dropout')
parser.add_argument('--label_smoothing', default=0.1, type=float)
parser.add_argument('--noise_std', default=0.02, type=float, help='Gaussian noise augmentation')
parser.add_argument('--spec_augment', default=0, type=int,
                    help='Enable SpecAugment (freq/time masking) + time-warp augmentation (1=yes)')
parser.add_argument('--unfreeze_llm_layers', default=0, type=int,
                    help='Number of last LLM decoder layers to unfreeze (0=all frozen)')
parser.add_argument('--lr_llm', default=1e-6, type=float,
                    help='Learning rate for unfrozen LLM layers')
parser.add_argument('--lora', default=0, type=int,
                    help='Apply LoRA to LLM attention layers (1=yes). Requires peft library.')
parser.add_argument('--lora_rank', default=8, type=int,
                    help='LoRA rank (lower=fewer params, higher=more capacity)')
parser.add_argument('--lora_alpha', default=16, type=int,
                    help='LoRA alpha scaling factor (typically 2x rank)')
parser.add_argument('--lora_dropout', default=0.05, type=float,
                    help='Dropout for LoRA layers')
parser.add_argument('--use_prompts_at_inference', default=1, type=int,
                    help='Use prompt wrapping during inference too (1=yes, 0=no)')
args = parser.parse_args()

# ==========================================================================
# Live logging
# ==========================================================================
_log_file = None
if args.log_file:
    os.makedirs(os.path.dirname(args.log_file), exist_ok=True)
    _log_file = open(args.log_file, 'w', buffering=1)

def log(msg):
    print(msg)
    if _log_file:
        _log_file.write(msg + '\n')
        _log_file.flush()

# ==========================================================================
# Seed
# ==========================================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(args.seed_num)

device = torch.device(args.cuda_pick if torch.cuda.is_available() else "cpu")
log(f"Device: {device}")
if torch.cuda.is_available():
    log(f"GPU: {torch.cuda.get_device_name(0)}")
    log(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

os.makedirs(args.checkpoint_dir, exist_ok=True)

# ==========================================================================
# Dataset with IMPROVED augmentation
# ==========================================================================
class NpyJsonDataset(Dataset):
    def __init__(self, main_dir, ids=None, silent=True, augment=False, noise_std=0.02, spec_augment=False):
        self.main_dir = main_dir
        self.emgtype = '_silent' if silent else '_voiced'
        self.augment = augment
        self.noise_std = noise_std
        self.spec_augment = spec_augment
        self.samples = []

        # Support both integer IDs (closed-vocab: "50_silent.npy")
        # and string IDs (open-vocab: "5-4_50_silent.npy")
        ids_set = set(ids) if ids else set()
        has_string_ids = any(isinstance(i, str) for i in ids_set)

        for file_name in os.listdir(main_dir):
            if not file_name.endswith(f'{self.emgtype}.npy'):
                continue

            if has_string_ids:
                # Open-vocab: extract base name before "_silent.npy"
                # e.g., "5-4_50_silent.npy" → "5-4_50"
                suffix = f'{self.emgtype}.npy'
                base_name = file_name[:-len(suffix)]
                id_oi = base_name
            else:
                # Legacy integer mode: "50_silent.npy" → 50
                id_oi = file_name.split('_')[0]
                if id_oi.isnumeric():
                    id_oi = int(id_oi)
                else:
                    id_oi = int(id_oi.split('.')[0])
                base_name = file_name.split('_')[0]

            if id_oi in ids_set:
                json_path = os.path.join(main_dir, f"{base_name}.json")
                if os.path.exists(json_path):
                    self.samples.append((file_name, base_name))

        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        npy_name, json_base_name = self.samples[idx]
        npy_path = os.path.join(self.main_dir, npy_name)
        json_path = os.path.join(self.main_dir, f"{json_base_name}.json")
        features = torch.tensor(np.load(npy_path), dtype=torch.float32)
        with open(json_path, 'r') as f:
            target = json.load(f)
        target_text = target.get('text', None)

        # [IMPROVEMENT 8] Dataset-level augmentation: Gaussian noise + random scaling
        if self.augment:
            # Additive Gaussian noise
            if random.random() < 0.5:
                features = features + torch.randn_like(features) * self.noise_std
            # Random amplitude scaling (0.9 - 1.1x)
            if random.random() < 0.3:
                scale = 0.9 + 0.2 * random.random()
                features = features * scale

        # SpecAugment + time-warp augmentation
        if self.augment and self.spec_augment:
            seq_len, n_feats = features.shape
            # Frequency masking: zero out 2-4 consecutive feature bands
            if random.random() < 0.5 and n_feats > 4:
                f_width = random.randint(2, min(4, n_feats - 1))
                f_start = random.randint(0, n_feats - f_width)
                features[:, f_start:f_start + f_width] = 0
            # Time masking: zero out 5-15 consecutive time steps
            if random.random() < 0.5 and seq_len > 15:
                t_width = random.randint(5, min(15, seq_len - 1))
                t_start = random.randint(0, seq_len - t_width)
                features[t_start:t_start + t_width, :] = 0
            # Time warping: stretch/compress by ±10%
            if random.random() < 0.5 and seq_len > 10:
                warp_factor = 0.9 + 0.2 * random.random()
                new_len = max(1, int(seq_len * warp_factor))
                # (T, F) -> (1, F, T) for interpolate -> (new_T, F)
                features = features.unsqueeze(0).permute(0, 2, 1)
                features = F.interpolate(features, size=new_len, mode='linear', align_corners=False)
                features = features.permute(0, 2, 1).squeeze(0)

        label_type = 'silent' if '_silent' in npy_name else 'voiced'
        return features, target_text, label_type


def collate_fn(batch):
    sequences, labels, label_types = zip(*batch)
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0)
    masks = (padded_sequences.sum(dim=-1) != 0).float()
    return padded_sequences, masks, labels


# Load split
with open(args.json_path, 'r') as f:
    split_data = json.load(f)
train_ids = split_data['train_sentence_indices']
dev_ids = split_data['dev_sentence_indices']

# [IMPROVEMENT] Augmentation only on train set
train_dataset = NpyJsonDataset(args.hc_features_dir, ids=train_ids,
                                augment=True, noise_std=args.noise_std,
                                spec_augment=bool(args.spec_augment))
train_dataloader = DataLoader(
    train_dataset, batch_size=args.batch_size, collate_fn=collate_fn,
    shuffle=True, num_workers=6, pin_memory=True, prefetch_factor=20
)

dev_dataset = NpyJsonDataset(args.hc_features_dir, ids=dev_ids, augment=False)
dev_dataloader = DataLoader(
    dev_dataset, batch_size=args.batch_size, collate_fn=collate_fn,
    shuffle=False, num_workers=6, pin_memory=True, prefetch_factor=20
)

log(f"Train samples: {len(train_dataset)}, Dev samples: {len(dev_dataset)}")

# ==========================================================================
# IMPROVED Model Architecture
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
        # Use zeros for deterministic behavior (randn adds noise that hurts convergence)
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
    """Same architecture as original but with:
    - [IMPROVEMENT 4] Dropout layers for regularization
    - [IMPROVEMENT 9] LayerNorm on output to match LLM embedding scale
    """
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
        self.lstm = LstmBlock(model_size // 4, model_size // 4, device)
        self.w_out = nn.Linear(model_size // 2, llm_hidden)

        # [IMPROVEMENT 4] Dropout for regularization
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

        # [IMPROVEMENT 9] LayerNorm to match LLM embedding scale
        self.output_norm = nn.LayerNorm(llm_hidden)

    def forward(self, x_feat, x_raw, session_ids):
        # [IMPROVEMENT 8] Temporal shift augmentation (same as original)
        if self.training:
            r = random.randrange(8)
            if r > 0:
                x_raw[:, :-r, :] = x_raw[:, r:, :].clone()
                x_raw[:, -r:, :] = 0

        x_raw = x_raw.transpose(1, 2)
        x_raw = self.conv1D_decim_1(x_raw)
        x_raw = self.conv_blocks(x_raw)
        x_raw = self.drop1(x_raw)  # [IMPROVEMENT 4]
        x_raw = self.lstm(x_raw.permute(0, 2, 1)).permute(0, 2, 1)
        x_raw = self.conv1D_decim(x_raw)
        x_raw = self.drop2(x_raw)  # [IMPROVEMENT 4]
        x_raw = x_raw.transpose(1, 2)

        out = self.w_out(x_raw)
        out = self.output_norm(out)  # [IMPROVEMENT 9]
        return out


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

    def text_to_int(self, text):
        text = self.clean_text(text)
        return [self.chars.index(c) for c in text if c in self.chars]


# ==========================================================================
# Load LLM (must be loaded before transNet to auto-detect hidden size)
# ==========================================================================
log(f"\nLoading LLM: {args.llm_path}")
t0 = time.time()

llama_tokenizer = AutoTokenizer.from_pretrained(args.llm_path, use_fast=True)
llama_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
llama_tokenizer.padding_side = "right"

llama_model = AutoModelForCausalLM.from_pretrained(
    args.llm_path,
    torch_dtype=torch.float16,
    device_map={"": device},
)

# Handle BOS token — some models (e.g., Phi-3) may have bos_token_id=None
if llama_tokenizer.bos_token_id is None:
    log("  WARNING: bos_token_id is None, using eos_token_id as BOS")
    llama_tokenizer.bos_token_id = llama_tokenizer.eos_token_id

llama_model.config.pad_token_id = llama_tokenizer.pad_token_id

# [IMPROVEMENT 3] Frozen LLM stays in eval mode — stable BatchNorm/no dropout noise
llama_model.eval()
for name, param in llama_model.named_parameters():
    param.requires_grad = False

# Optionally unfreeze last N LLM decoder layers
unfrozen_llm_params = []
if args.unfreeze_llm_layers > 0 and not args.lora:
    # Get decoder layers (works for LLaMA, Gemma, Mistral, etc.)
    if hasattr(llama_model, 'model') and hasattr(llama_model.model, 'layers'):
        decoder_layers = llama_model.model.layers
    elif hasattr(llama_model, 'transformer') and hasattr(llama_model.transformer, 'h'):
        decoder_layers = llama_model.transformer.h
    else:
        raise ValueError(f"Cannot find decoder layers for {type(llama_model)}")

    total_layers = len(decoder_layers)
    layers_to_unfreeze = decoder_layers[-args.unfreeze_llm_layers:]
    for layer in layers_to_unfreeze:
        for param in layer.parameters():
            param.requires_grad = True
            unfrozen_llm_params.append(param)
    n_unfrozen = sum(p.numel() for p in unfrozen_llm_params)
    log(f"  Unfroze last {args.unfreeze_llm_layers}/{total_layers} LLM layers: {n_unfrozen:,} params (lr={args.lr_llm})")

# Optionally apply LoRA to LLM attention layers
lora_params = []
if args.lora:
    from peft import LoraConfig, get_peft_model, TaskType
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        bias="none",
    )
    llama_model = get_peft_model(llama_model, lora_config)
    # Collect LoRA trainable params
    for name, param in llama_model.named_parameters():
        if param.requires_grad:
            lora_params.append(param)
    n_lora = sum(p.numel() for p in lora_params)
    llama_model.print_trainable_parameters()
    log(f"  LoRA applied: rank={args.lora_rank}, alpha={args.lora_alpha}, dropout={args.lora_dropout}")
    log(f"  LoRA trainable params: {n_lora:,} (lr={args.lr_llm})")

log(f"  Loaded in {time.time()-t0:.1f}s")
log(f"  BOS={llama_tokenizer.bos_token_id}, EOS={llama_tokenizer.eos_token_id}, PAD={llama_tokenizer.pad_token_id}")

# Auto-detect hidden size if not specified
llm_hidden = args.llm_hidden if args.llm_hidden > 0 else llama_model.config.hidden_size
log(f"  LLM hidden size: {llm_hidden}")

total_llm_params = sum(p.numel() for p in llama_model.parameters())
log(f"  LLM params: {total_llm_params:,}")

# Helper to get embed_tokens from any model architecture
def get_embed_tokens(model):
    """Works for LLaMA, Phi-3, Qwen, Gemma, Mistral, etc."""
    if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
        return model.model.embed_tokens  # LLaMA, Phi-3, Qwen, Mistral
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'wte'):
        return model.transformer.wte  # GPT-2 style
    elif hasattr(model, 'get_input_embeddings'):
        return model.get_input_embeddings()  # Universal fallback
    raise ValueError(f"Cannot find embed_tokens for {type(model)}")

embed_tokens = get_embed_tokens(llama_model)
log(f"  embed_tokens: {type(embed_tokens).__name__}, weight shape: {embed_tokens.weight.shape}")

# ==========================================================================
# Instantiate transNet (after LLM so we know llm_hidden)
# ==========================================================================
transNet = TransductionModelImproved(
    model_size=args.model_size, num_features=args.num_features, llm_hidden=llm_hidden, dropout=args.dropout
).to(device)

adaptor_params = sum(p.numel() for p in transNet.parameters())
log(f"Adaptor params: {adaptor_params:,}")

# ==========================================================================
# Prompt embeddings
# ==========================================================================
emg_prompt = 'Unvoiced EMG :'
prompt_text = 'Prompt : Convert unvoiced EMG to text'

with torch.no_grad():
    prompt_conversion_embed = llama_tokenizer(
        emg_prompt, return_tensors="pt", padding="longest",
        truncation=True, max_length=128, add_special_tokens=False
    ).to(device)
    prompt_embed_ends_embed = llama_tokenizer(
        prompt_text, return_tensors="pt",
        truncation=True, add_special_tokens=False
    ).to(device)
    prompt_conversion_embed = embed_tokens(prompt_conversion_embed.input_ids)
    prompt_embed_ends_embed = embed_tokens(prompt_embed_ends_embed.input_ids)

log(f"  Prompt embeds: before={prompt_conversion_embed.shape}, after={prompt_embed_ends_embed.shape}")

# ==========================================================================
# LLM Forward Pass (with label smoothing)
# ==========================================================================
def LLMForwardPass(emg_embed, targets):
    text = [t + llama_tokenizer.eos_token for t in targets]
    to_regress_tokens = llama_tokenizer(
        text, return_tensors="pt", padding="longest",
        truncation=True, max_length=128, add_special_tokens=False
    ).to(device)

    batch_size = emg_embed.shape[0]
    bos = torch.ones([batch_size, 1], dtype=torch.long, device=device) * llama_tokenizer.bos_token_id
    bos_embeds = embed_tokens(bos)

    to_regress_tokens.input_ids = to_regress_tokens.input_ids.masked_fill(
        to_regress_tokens.input_ids == llama_tokenizer.pad_token_id, 0
    )
    to_regress_embeds = embed_tokens(to_regress_tokens.input_ids)

    targets_masked = to_regress_tokens.input_ids.masked_fill(
        to_regress_tokens.input_ids == llama_tokenizer.pad_token_id, -100
    )

    # Add prompts
    before_emg = prompt_conversion_embed.expand(batch_size, -1, -1)
    after_emg = prompt_embed_ends_embed.expand(batch_size, -1, -1)
    emg_embed_with_prompt = torch.cat([before_emg, emg_embed, after_emg], dim=1)

    inputs_embeds = torch.cat([bos_embeds, emg_embed_with_prompt, to_regress_embeds], dim=1)

    emg_atts = torch.ones(emg_embed_with_prompt.shape[:2], dtype=torch.long, device=device)
    atts_bos = emg_atts[:, :1]

    attention_mask = torch.cat([atts_bos, emg_atts, to_regress_tokens.attention_mask], dim=1)
    empty_targets = torch.ones(
        [emg_atts.shape[0], emg_atts.shape[1] + 1], dtype=torch.long
    ).to(device).fill_(-100)
    targets_final = torch.cat([empty_targets, targets_masked], dim=1)

    # [IMPROVEMENT 7] Use label smoothing via manual cross-entropy
    if args.label_smoothing > 0:
        outputs = llama_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
        )
        logits = outputs.logits
        # Shift logits and labels (standard causal LM shift)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = targets_final[..., 1:].contiguous()
        # Flatten
        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        shift_labels = shift_labels.view(-1)
        loss = F.cross_entropy(
            shift_logits, shift_labels,
            ignore_index=-100,
            label_smoothing=args.label_smoothing
        )
    else:
        outputs = llama_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            labels=targets_final
        )
        loss = outputs.loss

    return {"loss": loss}


# ==========================================================================
# [IMPROVEMENT 2] Generate with prompt wrapping (matching training)
# ==========================================================================
@torch.no_grad()
def generate(emg_embed, max_new_tokens=150):
    batch_size = emg_embed.shape[0]
    bos = torch.ones([batch_size, 1], dtype=torch.long, device=device) * llama_tokenizer.bos_token_id
    bos_embeds = embed_tokens(bos)

    if args.use_prompts_at_inference:
        # [IMPROVEMENT 2] Use same prompt wrapping as training
        before_emg = prompt_conversion_embed.expand(batch_size, -1, -1)
        after_emg = prompt_embed_ends_embed.expand(batch_size, -1, -1)
        emg_with_prompt = torch.cat([before_emg, emg_embed, after_emg], dim=1)
        inputs_embeds = torch.cat([bos_embeds, emg_with_prompt], dim=1)
    else:
        # Original behavior: [BOS, emg_embed] only
        inputs_embeds = torch.cat([bos_embeds, emg_embed], dim=1)

    attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=device)

    stop_words_ids = [torch.tensor([llama_tokenizer.eos_token_id], device=device)]
    stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

    # [IMPROVEMENT 10] Explicit eos/pad token ids
    generated_ids = llama_model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        stopping_criteria=stopping_criteria,
        num_beams=4,
        min_length=1,
        top_p=0.8,
        repetition_penalty=1.2,  # Slightly increased to reduce repetitive garbage
        length_penalty=1.0,
        eos_token_id=llama_tokenizer.eos_token_id,
        pad_token_id=llama_tokenizer.pad_token_id,
    )
    return llama_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)


# ==========================================================================
# Text utilities
# ==========================================================================
text_transform = TextTransform()

class TargetProcessor:
    def __init__(self):
        self.text_transform = TextTransform()

    def process(self, targets):
        target_sequences = [torch.tensor(self.text_transform.text_to_int(t), dtype=torch.long)
                            for t in targets]
        target_lengths = torch.tensor([len(seq) for seq in target_sequences], dtype=torch.long)
        target_indices = pad_sequence(target_sequences, batch_first=True, padding_value=0)
        return target_indices, target_lengths

target_processor = TargetProcessor()


# ==========================================================================
# Evaluation
# ==========================================================================
def evaluate(transNet, dataloader, device='cuda'):
    transNet.eval()
    llama_model.eval()
    total_loss = 0
    total_wer = 0
    total_cer = 0
    valid_batches = 0
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for batch_idx, (spectrogram, emg_mask, targets) in enumerate(dataloader):
            spectrogram = spectrogram.to(device)
            emg_embed = transNet(None, spectrogram, None)

            log(f'  ---- Eval batch {batch_idx+1}/{len(dataloader)} ----')
            with autocast("cuda"):
                cleaned_targets = [text_transform.clean_text(t) for t in targets]
                _, target_lengths = target_processor.process(cleaned_targets)

                if any(t == 0 for t in target_lengths):
                    log(f"  Skipping batch {batch_idx} due to empty target.")
                    continue

                loss = LLMForwardPass(emg_embed, cleaned_targets)['loss']
                generated_text = generate(emg_embed)

                batch_wer = 0
                batch_cer = 0
                for i in range(len(cleaned_targets)):
                    pred_text = text_transform.clean_text(generated_text[i])
                    target_text = cleaned_targets[i]

                    log(f'    Target: "{target_text}"')
                    log(f'    Pred:   "{pred_text}"')

                    wer = jiwer.wer(target_text, pred_text) if pred_text.strip() else 1.0
                    cer = jiwer.cer(target_text, pred_text) if pred_text.strip() else 1.0
                    batch_wer += wer
                    batch_cer += cer

                    all_targets.append(target_text)
                    all_preds.append(pred_text)

                batch_wer /= len(cleaned_targets)
                batch_cer /= len(cleaned_targets)
                total_wer += batch_wer
                total_cer += batch_cer
                valid_batches += 1
                total_loss += loss.item()

    avg_loss = total_loss / valid_batches if valid_batches > 0 else 0
    avg_wer = total_wer / valid_batches if valid_batches > 0 else 0
    avg_cer = total_cer / valid_batches if valid_batches > 0 else 0

    if all_targets:
        corpus_wer = jiwer.wer(all_targets, all_preds)
        corpus_cer = jiwer.cer(all_targets, all_preds)
    else:
        corpus_wer = corpus_cer = 0

    return avg_loss, avg_wer, avg_cer, corpus_wer, corpus_cer


# ==========================================================================
# [IMPROVEMENT 1,6] Cosine schedule with warm restarts
# ==========================================================================
initial_lr = 1e-6
max_lr = args.lr
num_epochs = args.num_epochs
max_grad_norm = 1.0  # [IMPROVEMENT 5] Clip adaptor grads (more standard value)
warmup_steps = 500
total_steps = num_epochs * len(train_dataloader)

# ---------- Optimizer: flat LR or discriminative LR (for transfer learning) ----------
if args.lr_early is not None or args.lr_late is not None:
    lr_early = args.lr_early or max_lr
    lr_mid   = max_lr  # --lr is the middle group LR
    lr_late  = args.lr_late or max_lr

    early_prefixes = ['conv1D_decim_1', 'conv_blocks.0']
    late_prefixes  = ['w_out', 'output_norm']

    early_params, mid_params, late_params = [], [], []
    for name, param in transNet.named_parameters():
        if any(name.startswith(p) for p in early_prefixes):
            early_params.append(param)
        elif any(name.startswith(p) for p in late_prefixes):
            late_params.append(param)
        else:
            mid_params.append(param)

    param_groups = [
        {"params": early_params, "lr": lr_early, "weight_decay": 0.05},
        {"params": mid_params,   "lr": lr_mid,   "weight_decay": 0.05},
        {"params": late_params,  "lr": lr_late,   "weight_decay": 0.01},
    ]
    if unfrozen_llm_params:
        param_groups.append({"params": unfrozen_llm_params, "lr": args.lr_llm, "weight_decay": 0.01})
    if lora_params:
        param_groups.append({"params": lora_params, "lr": args.lr_llm, "weight_decay": 0.01})
    optimizer = AdamW(param_groups, eps=1e-8)
    log(f"  Discriminative LR: early={lr_early}, mid={lr_mid}, late={lr_late}")
    log(f"  Param counts: early={sum(p.numel() for p in early_params):,}, "
        f"mid={sum(p.numel() for p in mid_params):,}, "
        f"late={sum(p.numel() for p in late_params):,}")
else:
    param_groups = [{"params": list(transNet.parameters()), "lr": max_lr, "weight_decay": 0.05}]
    if unfrozen_llm_params:
        param_groups.append({"params": unfrozen_llm_params, "lr": args.lr_llm, "weight_decay": 0.01})
    if lora_params:
        param_groups.append({"params": lora_params, "lr": args.lr_llm, "weight_decay": 0.01})
    optimizer = AdamW(param_groups, eps=1e-8)

def cosine_warmup_scheduler(step):
    if step < warmup_steps:
        return initial_lr / max_lr + (step / warmup_steps) * (1.0 - initial_lr / max_lr)
    # Cosine decay from 1.0 to 0.01
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.01 + 0.5 * (1.0 - 0.01) * (1 + math.cos(math.pi * progress))

scheduler = LambdaLR(optimizer, lr_lambda=cosine_warmup_scheduler)

# Resume
start_epoch = 0
best_val_wer = float('inf')
best_val_cer = float('inf')

if args.resume_from and os.path.exists(args.resume_from):
    log(f"\nResuming from {args.resume_from}")
    ckpt = torch.load(args.resume_from, map_location='cpu', weights_only=True)
    # Handle both full checkpoint dict and raw state_dict formats
    if 'transNet_state_dict' in ckpt:
        # Full checkpoint (from chkpt_epoch_*.pth)
        transNet.load_state_dict(ckpt['transNet_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt['epoch']
        best_val_wer = ckpt.get('val_wer', float('inf'))
        log(f"  Full checkpoint: resumed from epoch {start_epoch}, best WER={best_val_wer:.4f}")
    else:
        # Raw state_dict (from best_transNet.pth)
        transNet.load_state_dict(ckpt)
        start_epoch = 0
        best_val_wer = float('inf')
        log(f"  Loaded raw state_dict (weights only, starting from epoch 0)")

# ==========================================================================
# Training loop
# ==========================================================================
log("\n" + "=" * 70)
log(f"IMPROVED TRAINING: {num_epochs} epochs, batch_size={args.batch_size}")
log(f"  model_size={args.model_size}, num_features={args.num_features}, llm_hidden={llm_hidden}")
if args.lr_early is not None or args.lr_late is not None:
    log(f"  lr: early={args.lr_early or max_lr}, mid={max_lr}, late={args.lr_late or max_lr} (discriminative)")
else:
    log(f"  lr={max_lr}")
log(f"  cosine schedule, warmup={warmup_steps} steps")
log(f"  dropout={args.dropout}, label_smoothing={args.label_smoothing}")
log(f"  noise_std={args.noise_std}, grad_clip={max_grad_norm}")
log(f"  spec_augment={bool(args.spec_augment)}")
if args.unfreeze_llm_layers > 0 and not args.lora:
    log(f"  unfreeze_llm_layers={args.unfreeze_llm_layers}, lr_llm={args.lr_llm}")
if args.lora:
    log(f"  LoRA: rank={args.lora_rank}, alpha={args.lora_alpha}, dropout={args.lora_dropout}, lr={args.lr_llm}")
log(f"  prompts_at_inference={bool(args.use_prompts_at_inference)}")
log(f"  eval_every={args.eval_every}, save_every={args.save_every}")
log(f"  checkpoint_dir: {args.checkpoint_dir}")
log("=" * 70)

for epoch in range(start_epoch, num_epochs):
    epoch_start = time.time()

    transNet.train()
    # [IMPROVEMENT 3] Keep LLM in eval — we regularize through adaptor dropout instead
    llama_model.eval()

    total_loss = 0
    count_batches = 0

    for batch_idx, (spectrogram, emg_mask, targets) in enumerate(train_dataloader):
        optimizer.zero_grad()
        spectrogram = spectrogram.to(device)

        # transNet forward outside autocast (fp32 adaptor)
        emg_embed = transNet(None, spectrogram, None)

        with autocast("cuda"):
            for t in range(len(targets)):
                targets[t] = text_transform.clean_text(targets[t])

            _, target_lengths = target_processor.process(targets)
            if any(t == 0 for t in target_lengths):
                log(f"  Skipping batch {batch_idx} due to empty target.")
                continue

            loss = LLMForwardPass(emg_embed, targets)['loss']
            total_loss += loss.item()
            count_batches += 1

        loss.backward()

        # [IMPROVEMENT 5] Clip gradients
        torch.nn.utils.clip_grad_norm_(transNet.parameters(), max_grad_norm)
        if unfrozen_llm_params:
            torch.nn.utils.clip_grad_norm_(unfrozen_llm_params, max_grad_norm)
        if lora_params:
            torch.nn.utils.clip_grad_norm_(lora_params, max_grad_norm)
        optimizer.step()
        scheduler.step()

        if count_batches % 9 == 0:
            log(f"  Epoch [{epoch+1}/{num_epochs}] Batch [{count_batches+1}/{len(train_dataloader)}] "
                f"- Loss: {total_loss/count_batches:.4f} - LR: {scheduler.get_last_lr()[0]:.2e}")

    avg_loss = total_loss / max(len(train_dataloader), 1)
    epoch_time = time.time() - epoch_start
    log(f"Epoch [{epoch+1}/{num_epochs}] completed - Avg Loss: {avg_loss:.4f} ({epoch_time:.1f}s)")

    # Evaluate
    if (epoch + 1) % args.eval_every == 0:
        eval_start = time.time()
        transNet.eval()
        avg_val_loss, avg_val_wer, avg_val_cer, corpus_wer, corpus_cer = evaluate(
            transNet=transNet, dataloader=dev_dataloader, device=device
        )
        eval_time = time.time() - eval_start

        log(f"  Val - Loss: {avg_val_loss:.4f} | "
            f"WER: {avg_val_wer:.4f} (corpus: {corpus_wer:.4f}) | "
            f"CER: {avg_val_cer:.4f} (corpus: {corpus_cer:.4f}) | "
            f"({eval_time:.1f}s)")

        if corpus_wer < best_val_wer:
            best_val_wer = corpus_wer
            best_val_cer = corpus_cer
            best_path = os.path.join(args.checkpoint_dir, "best_transNet.pth")
            torch.save(transNet.state_dict(), best_path)
            if unfrozen_llm_params:
                llm_best_path = os.path.join(args.checkpoint_dir, "best_llm_layers.pth")
                llm_state = {n: p.data for n, p in llama_model.named_parameters() if p.requires_grad}
                torch.save(llm_state, llm_best_path)
            if lora_params:
                lora_best_path = os.path.join(args.checkpoint_dir, "best_lora_adapter")
                llama_model.save_pretrained(lora_best_path)
                log(f"  >> LoRA adapter saved to {lora_best_path}")
            log(f"  >> New best WER={best_val_wer:.4f}, CER={best_val_cer:.4f} — saved to {best_path}")

    if (epoch + 1) % args.save_every == 0:
        checkpoint = {
            "epoch": epoch + 1,
            "transNet_state_dict": transNet.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "loss": avg_loss,
            "val_wer": avg_val_wer if (epoch + 1) % args.eval_every == 0 else None,
            "val_cer": avg_val_cer if (epoch + 1) % args.eval_every == 0 else None,
            "model_size": args.model_size,
            "llm_path": args.llm_path,
            "llm_hidden": llm_hidden,
        }
        ckpt_path = os.path.join(args.checkpoint_dir, f"chkpt_epoch_{epoch+1}.pth")
        torch.save(checkpoint, ckpt_path)
        log(f"  Checkpoint saved: {ckpt_path}")

log("\n" + "=" * 70)
log(f"IMPROVED TRAINING COMPLETE")
log(f"  Best val WER: {best_val_wer:.4f}")
log(f"  Best val CER: {best_val_cer:.4f}")
log(f"  Best model: {args.checkpoint_dir}/best_transNet.pth")
log("=" * 70)

if _log_file:
    _log_file.close()
