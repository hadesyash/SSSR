"""
=============================================================================
SilentSpeechLLM — Ablation Study Results Summarizer
=============================================================================

USAGE:
    python ~/aml_lab/hpc/summarize_ablation.py
=============================================================================
"""

import os
import re
import glob

LOG_DIR = os.path.expanduser("~/aml_lab/hpc/logs")

STAGE_LABELS = {
    0: "Baseline (all OFF)",
    1: "+ LLM eval mode",
    2: "+ Prompt-consistent inference",
    3: "+ Cosine LR schedule",
    4: "+ Gradient clipping",
    5: "+ Adaptor dropout (0.15)",
    6: "+ Output LayerNorm",
    7: "+ Label smoothing (0.1)",
    8: "+ Data augmentation",
    9: "+ Weight decay (0.05)",
}

STAGE_SHORT = {
    0: "Baseline",
    1: "+Eval",
    2: "+Prompt",
    3: "+CosLR",
    4: "+GClip",
    5: "+Drop",
    6: "+LNorm",
    7: "+LSmooth",
    8: "+Aug",
    9: "+WD",
}


def parse_log(log_path):
    result = {
        "log_file": os.path.basename(log_path),
        "llm_path": None,
        "llm_hidden": None,
        "model_size": None,
        "adaptor_params": None,
        "stage": None,
        "best_wer": None,
        "best_cer": None,
        "best_epoch": None,
        "final_epoch": 0,
        "status": "unknown",
        # Ablation flags
        "llm_eval": None,
        "prompts_inf": None,
        "cosine_lr": None,
        "grad_clip": None,
        "dropout": None,
        "layernorm": None,
        "label_smooth": None,
        "noise_aug": None,
        "weight_decay": None,
    }

    try:
        with open(log_path, 'r') as f:
            lines = f.readlines()
    except:
        result["status"] = "error"
        return result

    if not lines:
        result["status"] = "empty"
        return result

    best_wer = float('inf')
    best_cer = float('inf')
    best_epoch = None

    for line in lines:
        m = re.search(r'Loading LLM:\s*(.+)', line)
        if m: result["llm_path"] = m.group(1).strip()

        m = re.search(r'LLM hidden size:\s*(\d+)', line)
        if m: result["llm_hidden"] = int(m.group(1))

        m = re.search(r'model_size[:=]\s*(\d+)', line)
        if m: result["model_size"] = int(m.group(1))

        m = re.search(r'Adaptor params:\s*([\d,]+)', line)
        if m: result["adaptor_params"] = int(m.group(1).replace(',', ''))

        # Parse ablation config
        m = re.search(r'S1 llm_eval:\s*(True|False)', line)
        if m: result["llm_eval"] = m.group(1) == "True"

        m = re.search(r'S2 prompts_inf:\s*(True|False)', line)
        if m: result["prompts_inf"] = m.group(1) == "True"

        m = re.search(r'S3 cosine_lr:\s*(True|False)', line)
        if m: result["cosine_lr"] = m.group(1) == "True"

        m = re.search(r'S4 grad_clip:\s*(True|False)', line)
        if m: result["grad_clip"] = m.group(1) == "True"

        m = re.search(r'S5 dropout:\s*([\d.]+)', line)
        if m: result["dropout"] = float(m.group(1))

        m = re.search(r'S6 layernorm:\s*(True|False)', line)
        if m: result["layernorm"] = m.group(1) == "True"

        m = re.search(r'S7 label_smooth:\s*([\d.]+)', line)
        if m: result["label_smooth"] = float(m.group(1))

        m = re.search(r'S8 noise_aug:\s*([\d.]+)', line)
        if m: result["noise_aug"] = float(m.group(1))

        m = re.search(r'S9 weight_decay:\s*([\d.]+)', line)
        if m: result["weight_decay"] = float(m.group(1))

        m = re.search(r'Epoch \[(\d+)/\d+\] completed', line)
        if m: result["final_epoch"] = int(m.group(1))

        m = re.search(r'New best WER=([\d.]+),\s*CER=([\d.]+)', line)
        if m:
            wer = float(m.group(1))
            cer = float(m.group(2))
            if wer < best_wer:
                best_wer = wer
                best_cer = cer
                best_epoch = result["final_epoch"]

        if 'TRAINING COMPLETE' in line:
            result["status"] = "completed"

    if best_wer < float('inf'):
        result["best_wer"] = best_wer
        result["best_cer"] = best_cer
        result["best_epoch"] = best_epoch

    if result["status"] == "unknown":
        if result["final_epoch"] > 0:
            result["status"] = f"running (ep {result['final_epoch']})"
        else:
            result["status"] = "starting"

    # Infer stage from filename
    fname = result["log_file"]
    for s in range(10):
        if f"_S{s}_" in fname:
            result["stage"] = s
            break

    return result


def fmt(val, fmt_str="{}", default="-"):
    if val is None:
        return default
    return fmt_str.format(val)


def main():
    log_files = sorted(glob.glob(os.path.join(LOG_DIR, "abl_*.log")))

    if not log_files:
        print("No ablation logs found in", LOG_DIR)
        print("Run: bash ~/aml_lab/hpc/run_ablation.pbs")
        return

    results = []
    for lf in log_files:
        results.append(parse_log(lf))

    # Group by LLM
    llm_groups = {}
    for r in results:
        llm = r["llm_path"]
        if llm:
            llm_short = llm.split("/")[-1]
        else:
            # Try to get from filename
            fname = r["log_file"]
            if "gemma" in fname: llm_short = "gemma-2-2b-it"
            elif "phi3" in fname: llm_short = "Phi-3-mini-4k-instruct"
            elif "llama3b" in fname: llm_short = "Llama-3.2-3B-Instruct"
            elif "qwen" in fname: llm_short = "Qwen2.5-3B-Instruct"
            else: llm_short = "unknown"
        if llm_short not in llm_groups:
            llm_groups[llm_short] = []
        llm_groups[llm_short].append(r)

    # ======================================================================
    # Architecture details (print once)
    # ======================================================================
    print()
    print("=" * 90)
    print("ADAPTOR ARCHITECTURE (fixed across all ablation runs)")
    print("=" * 90)
    print("  Input:      112 HC features per timestep")
    print("  Conv1d:     112→112, k=6, s=6 (temporal downsampling)")
    print("  ResBlocks:  112→768 → 768→384 → 384→192")
    print("  BiLSTM:     192→384 (bidirectional)")
    print("  Conv1d:     384→384, k=2, s=2 (further downsampling)")
    print("  Projection: 384→llm_hidden (varies per LLM)")
    print("  model_size: 768 (fixed for all runs)")
    print()

    # Print LLM projection sizes
    seen = set()
    for r in results:
        llm = r["llm_path"]
        lh = r["llm_hidden"]
        if llm and lh and llm not in seen:
            seen.add(llm)
            print(f"  {llm.split('/')[-1]:<30} hidden={lh}  proj=384→{lh}")
    print("=" * 90)

    # ======================================================================
    # Per-LLM ablation tables
    # ======================================================================
    for llm_short, runs in sorted(llm_groups.items()):
        # Sort by stage
        runs.sort(key=lambda r: r["stage"] if r["stage"] is not None else 99)

        print()
        print()
        print("=" * 105)
        print(f"  {llm_short}")
        hidden = runs[0]["llm_hidden"] if runs[0]["llm_hidden"] else "?"
        params = fmt(runs[0]["adaptor_params"], "{:,}")
        print(f"  hidden={hidden}, adaptor_params={params}")
        print("=" * 105)
        print(f"  {'Stage':<6} {'Improvement Added':<30} {'WER':>7} {'CER':>7} "
              f"{'Δ WER':>7} {'BstEp':>5} {'Ep':>4} {'Status':<16}")
        print("-" * 105)

        prev_wer = None
        for r in runs:
            stage = r["stage"]
            if stage is not None and stage in STAGE_LABELS:
                label = STAGE_LABELS[stage]
            else:
                label = "?"

            wer = r["best_wer"]
            cer = r["best_cer"]
            wer_str = fmt(wer, "{:.4f}")
            cer_str = fmt(cer, "{:.4f}")

            # Delta WER from previous stage
            if wer is not None and prev_wer is not None:
                delta = wer - prev_wer
                if delta < 0:
                    delta_str = f"\033[32m{delta:+.4f}\033[0m"  # green for improvement
                elif delta > 0:
                    delta_str = f"\033[31m{delta:+.4f}\033[0m"  # red for regression
                else:
                    delta_str = f"{delta:+.4f}"
            elif wer is not None and prev_wer is None:
                delta_str = "  base"
            else:
                delta_str = "     -"

            best_ep = fmt(r["best_epoch"], "{}")
            final_ep = str(r["final_epoch"])

            stage_str = f"S{stage}" if stage is not None else "?"
            print(f"  {stage_str:<6} {label:<30} {wer_str:>7} {cer_str:>7} "
                  f"{delta_str:>16} {best_ep:>5} {final_ep:>4} {r['status']:<16}")

            if wer is not None:
                prev_wer = wer

        # Summary for this LLM
        completed = [r for r in runs if r["best_wer"] is not None]
        if completed:
            best = min(completed, key=lambda r: r["best_wer"])
            worst = max(completed, key=lambda r: r["best_wer"])
            print("-" * 105)
            best_stage = best["stage"]
            print(f"  Best:  S{best_stage} ({STAGE_SHORT.get(best_stage, '?')}) "
                  f"WER={best['best_wer']:.4f}  |  "
                  f"Baseline S0 WER={worst['best_wer'] if runs[0]['best_wer'] is not None else '?'}  |  "
                  f"Total improvement: {(runs[0]['best_wer'] or 0) - best['best_wer']:+.4f}")

    # ======================================================================
    # Cross-LLM comparison: best stage for each LLM
    # ======================================================================
    print()
    print()
    print("=" * 90)
    print("CROSS-LLM COMPARISON: Best result per LLM")
    print("=" * 90)
    print(f"  {'LLM':<30} {'Best Stage':<12} {'WER':>7} {'CER':>7} {'vs Paper 0.49':>14}")
    print("-" * 90)

    all_best = []
    for llm_short, runs in sorted(llm_groups.items()):
        completed = [r for r in runs if r["best_wer"] is not None]
        if completed:
            best = min(completed, key=lambda r: r["best_wer"])
            stage = best["stage"]
            stage_str = f"S{stage}" if stage is not None else "?"
            improvement = (0.49 - best["best_wer"]) / 0.49 * 100
            sign = "+" if improvement > 0 else ""
            print(f"  {llm_short:<30} {stage_str:<12} {best['best_wer']:>7.4f} "
                  f"{best['best_cer']:>7.4f} {sign}{improvement:>12.1f}%")
            all_best.append(best)

    if all_best:
        overall = min(all_best, key=lambda r: r["best_wer"])
        llm_short = overall["llm_path"].split("/")[-1] if overall["llm_path"] else "?"
        print("-" * 90)
        print(f"  OVERALL BEST: {llm_short}, S{overall['stage']}, "
              f"WER={overall['best_wer']:.4f}, CER={overall['best_cer']:.4f}")
    print("=" * 90)

    # ======================================================================
    # Which improvement helped most? (average delta across LLMs)
    # ======================================================================
    print()
    print("=" * 90)
    print("IMPROVEMENT IMPACT: Average WER change per stage (across LLMs)")
    print("=" * 90)
    print(f"  {'Stage':<6} {'Improvement':<30} {'Avg ΔWER':>10} {'# LLMs':>8} {'Helped/Hurt':>12}")
    print("-" * 90)

    for stage in range(1, 10):
        deltas = []
        for llm_short, runs in llm_groups.items():
            runs_sorted = sorted(runs, key=lambda r: r["stage"] if r["stage"] is not None else 99)
            prev = next((r for r in runs_sorted if r["stage"] == stage - 1 and r["best_wer"] is not None), None)
            curr = next((r for r in runs_sorted if r["stage"] == stage and r["best_wer"] is not None), None)
            if prev and curr:
                deltas.append(curr["best_wer"] - prev["best_wer"])

        if deltas:
            avg_delta = sum(deltas) / len(deltas)
            helped = sum(1 for d in deltas if d < -0.001)
            hurt = sum(1 for d in deltas if d > 0.001)
            neutral = len(deltas) - helped - hurt

            if avg_delta < -0.001:
                color_start, color_end = "\033[32m", "\033[0m"
            elif avg_delta > 0.001:
                color_start, color_end = "\033[31m", "\033[0m"
            else:
                color_start, color_end = "", ""

            impact = f"{helped}↑ {hurt}↓ {neutral}→"
            print(f"  S{stage:<5} {STAGE_LABELS[stage]:<30} "
                  f"{color_start}{avg_delta:>+10.4f}{color_end} {len(deltas):>8} {impact:>12}")
        else:
            print(f"  S{stage:<5} {STAGE_LABELS[stage]:<30} {'?':>10} {'0':>8} {'?':>12}")

    print("=" * 90)
    print()
    print("Legend: Δ WER < 0 = improvement (green), > 0 = regression (red)")
    print("        ↑ = helped, ↓ = hurt, → = neutral (within ±0.001)")


if __name__ == "__main__":
    main()
