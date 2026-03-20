"""
=============================================================================
SilentSpeechLLM — Fruit Vocab Training (Colab Pro)
=============================================================================

Self-contained training script for personal EMG data on Google Colab Pro.
Adapted from train_hc_llama3b_improved.py with speed optimizations.

SETUP (run in Colab cells BEFORE this script):
  # Cell 1: Mount Drive
  from google.colab import drive
  drive.mount('/content/drive')

  # Cell 2: Install deps
  !pip install transformers jiwer unidecode

  # Cell 3: HF login (for gated models like Gemma)
  from huggingface_hub import login
  login(token="YOUR_HF_TOKEN")

  # Cell 4: Train!
  !python /content/drive/MyDrive/aml_lab/train_fruit_colab.py

Or with custom args:
  !python /content/drive/MyDrive/aml_lab/train_fruit_colab.py \
      --data_dir /content/drive/MyDrive/aml_lab/data/yash2_3ch \
      --num_epochs 300 --eval_every 25

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
import math
import argparse
import jiwer
from unidecode import unidecode
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.amp import autocast
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList


# ==========================================================================
# Arguments (defaults tuned for personal fruit vocab on Colab)
# ==========================================================================
parser = argparse.ArgumentParser(description='Fruit Vocab Training (Colab Pro)')

# Paths — default to Google Drive locations
parser.add_argument('--data_dir',
                    default='/content/drive/MyDrive/aml_lab/data/yash2_3ch',
                    type=str, help='Directory with .npy/.json features + split.json')
parser.add_argument('--llm_path',
                    default='google/gemma-2-2b-it',
                    type=str, help='HuggingFace LLM path')
parser.add_argument('--checkpoint_dir',
                    default='/content/drive/MyDrive/aml_lab/checkpoints/fruit/yash2_3ch_ms384',
                    type=str, help='Where to save checkpoints (on Drive for persistence)')

# Architecture
parser.add_argument('--num_features', default=42, type=int,
                    help='Input features (42 for 3ch, 56 for 4ch)')
parser.add_argument('--model_size', default=384, type=int,
                    help='Adaptor intermediate size')

# Training
parser.add_argument('--num_epochs', default=300, type=int)
parser.add_argument('--batch_size', default=8, type=int)
parser.add_argument('--lr', default=5e-5, type=float, help='Peak learning rate')
parser.add_argument('--dropout', default=0.20, type=float)
parser.add_argument('--noise_std', default=0.05, type=float)
parser.add_argument('--label_smoothing', default=0.1, type=float)

# Evaluation
parser.add_argument('--eval_every', default=25, type=int)
parser.add_argument('--save_every', default=50, type=int)
parser.add_argument('--max_new_tokens', default=10, type=int,
                    help='Max tokens for generation during eval (fruit words are short)')

# System
parser.add_argument('--seed', default=2711, type=int)
parser.add_argument('--num_workers', default=2, type=int,
                    help='DataLoader workers (Colab has limited CPUs)')
parser.add_argument('--resume_from', default=None, type=str,
                    help='Path to checkpoint to resume from')

args = parser.parse_args()


# ==========================================================================
# Logging
# ==========================================================================
log_path = os.path.join(args.checkpoint_dir, 'training.log')

def log(msg):
    print(msg)
    try:
        with open(log_path, 'a') as f:
            f.write(msg + '\n')
    except Exception:
        pass


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

set_seed(args.seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
log(f"Device: {device}")
if torch.cuda.is_available():
    log(f"GPU: {torch.cuda.get_device_name(0)}")
    log(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

os.makedirs(args.checkpoint_dir, exist_ok=True)


# ==========================================================================
# Dataset (from train_hc_llama3b_improved.py)
# ==========================================================================
class NpyJsonDataset(Dataset):
    def __init__(self, main_dir, ids=None, silent=True, augment=False, noise_std=0.02):
        self.main_dir = main_dir
        self.emgtype = '_silent' if silent else '_voiced'
        self.augment = augment
        self.noise_std = noise_std
        self.samples = []

        ids_set = set(ids) if ids else set()
        has_string_ids = any(isinstance(i, str) for i in ids_set)

        for file_name in os.listdir(main_dir):
            if not file_name.endswith(f'{self.emgtype}.npy'):
                continue

            if has_string_ids:
                suffix = f'{self.emgtype}.npy'
                base_name = file_name[:-len(suffix)]
                id_oi = base_name
            else:
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

        # Augmentation: Gaussian noise + random scaling
        if self.augment:
            if random.random() < 0.5:
                features = features + torch.randn_like(features) * self.noise_std
            if random.random() < 0.3:
                scale = 0.9 + 0.2 * random.random()
                features = features * scale

        label_type = 'silent' if '_silent' in npy_name else 'voiced'
        return features, target_text, label_type


def collate_fn(batch):
    sequences, labels, label_types = zip(*batch)
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0)
    masks = (padded_sequences.sum(dim=-1) != 0).float()
    return padded_sequences, masks, labels


# ==========================================================================
# Load split
# ==========================================================================
split_path = os.path.join(args.data_dir, 'split.json')
if not os.path.exists(split_path):
    # Fallback: try 10_selected_samples.json format
    alt_path = os.path.join(args.data_dir, '10_selected_samples.json')
    if os.path.exists(alt_path):
        split_path = alt_path
    else:
        raise FileNotFoundError(
            f"No split.json or 10_selected_samples.json found in {args.data_dir}")

with open(split_path, 'r') as f:
    split_data = json.load(f)

# Support both naming conventions
train_ids = split_data.get('train_sentence_indices', split_data.get('train_ids', []))
dev_ids = split_data.get('dev_sentence_indices', split_data.get('dev_ids', []))

train_dataset = NpyJsonDataset(args.data_dir, ids=train_ids,
                                augment=True, noise_std=args.noise_std)
train_dataloader = DataLoader(
    train_dataset, batch_size=args.batch_size, collate_fn=collate_fn,
    shuffle=True, num_workers=args.num_workers, pin_memory=True
)

dev_dataset = NpyJsonDataset(args.data_dir, ids=dev_ids, augment=False)
dev_dataloader = DataLoader(
    dev_dataset, batch_size=args.batch_size, collate_fn=collate_fn,
    shuffle=False, num_workers=args.num_workers, pin_memory=True
)

log(f"Train samples: {len(train_dataset)}, Dev samples: {len(dev_dataset)}")
log(f"Train batches: {len(train_dataloader)}, Dev batches: {len(dev_dataloader)}")


# ==========================================================================
# Model Architecture (from train_hc_llama3b_improved.py)
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
        self.lstm = LstmBlock(model_size // 4, model_size // 4, device)
        self.w_out = nn.Linear(model_size // 2, llm_hidden)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.output_norm = nn.LayerNorm(llm_hidden)

    def forward(self, x_feat, x_raw, session_ids):
        # Temporal shift augmentation
        if self.training:
            r = random.randrange(8)
            if r > 0:
                x_raw[:, :-r, :] = x_raw[:, r:, :].clone()
                x_raw[:, -r:, :] = 0

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
# Load LLM
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

if llama_tokenizer.bos_token_id is None:
    log("  WARNING: bos_token_id is None, using eos_token_id as BOS")
    llama_tokenizer.bos_token_id = llama_tokenizer.eos_token_id

llama_model.config.pad_token_id = llama_tokenizer.pad_token_id
llama_model.eval()
for name, param in llama_model.named_parameters():
    param.requires_grad = False

log(f"  Loaded in {time.time()-t0:.1f}s")
log(f"  BOS={llama_tokenizer.bos_token_id}, EOS={llama_tokenizer.eos_token_id}, PAD={llama_tokenizer.pad_token_id}")

llm_hidden = llama_model.config.hidden_size
log(f"  LLM hidden size: {llm_hidden}")

# Get embed_tokens
def get_embed_tokens(model):
    if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
        return model.model.embed_tokens
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'wte'):
        return model.transformer.wte
    elif hasattr(model, 'get_input_embeddings'):
        return model.get_input_embeddings()
    raise ValueError(f"Cannot find embed_tokens for {type(model)}")

embed_tokens = get_embed_tokens(llama_model)
log(f"  embed_tokens: {type(embed_tokens).__name__}, weight shape: {embed_tokens.weight.shape}")


# ==========================================================================
# Instantiate transNet
# ==========================================================================
transNet = TransductionModelImproved(
    model_size=args.model_size,
    num_features=args.num_features,
    llm_hidden=llm_hidden,
    dropout=args.dropout
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

    # Label smoothing via manual cross-entropy
    if args.label_smoothing > 0:
        outputs = llama_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
        )
        logits = outputs.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = targets_final[..., 1:].contiguous()
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
# Generate (with prompts, beam search)
# ==========================================================================
@torch.no_grad()
def generate(emg_embed, max_new_tokens=None):
    if max_new_tokens is None:
        max_new_tokens = args.max_new_tokens

    batch_size = emg_embed.shape[0]
    bos = torch.ones([batch_size, 1], dtype=torch.long, device=device) * llama_tokenizer.bos_token_id
    bos_embeds = embed_tokens(bos)

    before_emg = prompt_conversion_embed.expand(batch_size, -1, -1)
    after_emg = prompt_embed_ends_embed.expand(batch_size, -1, -1)
    emg_with_prompt = torch.cat([before_emg, emg_embed, after_emg], dim=1)
    inputs_embeds = torch.cat([bos_embeds, emg_with_prompt], dim=1)

    attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=device)

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
        repetition_penalty=1.2,
        length_penalty=1.0,
        eos_token_id=llama_tokenizer.eos_token_id,
        pad_token_id=llama_tokenizer.pad_token_id,
    )
    return llama_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)


# ==========================================================================
# Evaluation
# ==========================================================================
def evaluate(transNet, dataloader):
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
# Cosine schedule with warmup
# ==========================================================================
initial_lr = 1e-6
max_lr = args.lr
num_epochs = args.num_epochs
max_grad_norm = 1.0
warmup_steps = 500
total_steps = num_epochs * len(train_dataloader)

# Adjust warmup if total steps is small
if warmup_steps > total_steps // 4:
    warmup_steps = max(total_steps // 10, 50)
    log(f"  Adjusted warmup_steps to {warmup_steps} (total_steps={total_steps})")

optimizer = AdamW(transNet.parameters(), lr=max_lr, weight_decay=0.05, eps=1e-8)

def cosine_warmup_scheduler(step):
    if step < warmup_steps:
        return initial_lr / max_lr + (step / warmup_steps) * (1.0 - initial_lr / max_lr)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.01 + 0.5 * (1.0 - 0.01) * (1 + math.cos(math.pi * progress))

scheduler = LambdaLR(optimizer, lr_lambda=cosine_warmup_scheduler)


# ==========================================================================
# Resume from checkpoint
# ==========================================================================
start_epoch = 0
best_val_wer = float('inf')
best_val_cer = float('inf')

if args.resume_from and os.path.exists(args.resume_from):
    log(f"\nResuming from {args.resume_from}")
    ckpt = torch.load(args.resume_from, map_location='cpu', weights_only=True)
    if 'transNet_state_dict' in ckpt:
        transNet.load_state_dict(ckpt['transNet_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt['epoch']
        best_val_wer = ckpt.get('val_wer', float('inf'))
        log(f"  Full checkpoint: resumed from epoch {start_epoch}, best WER={best_val_wer:.4f}")
    else:
        transNet.load_state_dict(ckpt)
        log(f"  Loaded raw state_dict (weights only, starting from epoch 0)")
else:
    # Auto-detect if there's a checkpoint in checkpoint_dir to resume from
    auto_resume = os.path.join(args.checkpoint_dir, 'latest_checkpoint.pth')
    if os.path.exists(auto_resume):
        log(f"\nAuto-resuming from {auto_resume}")
        ckpt = torch.load(auto_resume, map_location='cpu', weights_only=True)
        if 'transNet_state_dict' in ckpt:
            transNet.load_state_dict(ckpt['transNet_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            start_epoch = ckpt['epoch']
            best_val_wer = ckpt.get('val_wer', float('inf'))
            log(f"  Resumed from epoch {start_epoch}, best WER={best_val_wer:.4f}")


# ==========================================================================
# Training loop
# ==========================================================================
log("\n" + "=" * 70)
log(f"FRUIT VOCAB TRAINING — {num_epochs} epochs, batch_size={args.batch_size}")
log(f"  model_size={args.model_size}, num_features={args.num_features}, llm_hidden={llm_hidden}")
log(f"  lr={max_lr}, cosine schedule, warmup={warmup_steps} steps")
log(f"  dropout={args.dropout}, label_smoothing={args.label_smoothing}")
log(f"  noise_std={args.noise_std}, grad_clip={max_grad_norm}")
log(f"  eval_every={args.eval_every}, save_every={args.save_every}")
log(f"  max_new_tokens={args.max_new_tokens} (for eval generation)")
log(f"  checkpoint_dir: {args.checkpoint_dir}")
log("=" * 70)

training_start = time.time()

for epoch in range(start_epoch, num_epochs):
    epoch_start = time.time()

    transNet.train()
    llama_model.eval()

    total_loss = 0
    count_batches = 0

    for batch_idx, (spectrogram, emg_mask, targets) in enumerate(train_dataloader):
        optimizer.zero_grad()
        spectrogram = spectrogram.to(device)

        # Adaptor forward in fp32
        emg_embed = transNet(None, spectrogram, None)

        with autocast("cuda"):
            targets = [text_transform.clean_text(t) for t in targets]

            _, target_lengths = target_processor.process(targets)
            if any(t == 0 for t in target_lengths):
                log(f"  Skipping batch {batch_idx} due to empty target.")
                continue

            loss = LLMForwardPass(emg_embed, targets)['loss']
            total_loss += loss.item()
            count_batches += 1

        loss.backward()
        torch.nn.utils.clip_grad_norm_(transNet.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()

        if count_batches % 9 == 0:
            log(f"  Epoch [{epoch+1}/{num_epochs}] Batch [{count_batches}/{len(train_dataloader)}] "
                f"- Loss: {total_loss/count_batches:.4f} - LR: {scheduler.get_last_lr()[0]:.2e}")

    avg_loss = total_loss / max(count_batches, 1)
    epoch_time = time.time() - epoch_start
    elapsed_total = time.time() - training_start
    eta = (elapsed_total / max(epoch - start_epoch + 1, 1)) * (num_epochs - epoch - 1)
    log(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {avg_loss:.4f} ({epoch_time:.1f}s) "
        f"[elapsed: {elapsed_total/60:.0f}m, ETA: {eta/60:.0f}m]")

    # Evaluate
    if (epoch + 1) % args.eval_every == 0:
        eval_start = time.time()
        transNet.eval()
        avg_val_loss, avg_val_wer, avg_val_cer, corpus_wer, corpus_cer = evaluate(
            transNet=transNet, dataloader=dev_dataloader
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
            log(f"  >> New best WER={best_val_wer:.4f}, CER={best_val_cer:.4f} — saved!")

    # Save periodic checkpoint (for resume on Colab disconnect)
    if (epoch + 1) % args.save_every == 0:
        checkpoint = {
            "epoch": epoch + 1,
            "transNet_state_dict": transNet.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "loss": avg_loss,
            "val_wer": best_val_wer,
            "val_cer": best_val_cer,
            "model_size": args.model_size,
            "llm_path": args.llm_path,
            "llm_hidden": llm_hidden,
        }
        # Save as latest (for auto-resume) and epoch-specific
        latest_path = os.path.join(args.checkpoint_dir, "latest_checkpoint.pth")
        torch.save(checkpoint, latest_path)
        ckpt_path = os.path.join(args.checkpoint_dir, f"chkpt_epoch_{epoch+1}.pth")
        torch.save(checkpoint, ckpt_path)
        log(f"  Checkpoint saved: {ckpt_path}")

total_time = time.time() - training_start
log("\n" + "=" * 70)
log(f"TRAINING COMPLETE in {total_time/60:.1f} minutes")
log(f"  Best val WER: {best_val_wer:.4f}")
log(f"  Best val CER: {best_val_cer:.4f}")
log(f"  Best model: {args.checkpoint_dir}/best_transNet.pth")
log("=" * 70)
log(f"\nDownload best_transNet.pth to your Mac and run:")
log(f"  python hpc/realtime_gui.py \\")
log(f"      --checkpoint checkpoints/fruit/yash2_3ch_ms384/best_transNet.pth \\")
log(f"      --norm_stats data/yash2_3ch/norm_stats.json")
