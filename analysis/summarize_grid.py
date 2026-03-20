"""
=============================================================================
SilentSpeechLLM — Grid Search Results Summarizer
=============================================================================
Parses live logs from grid search runs and creates a results table
with full adaptor architecture details.

USAGE:
    python ~/aml_lab/hpc/summarize_grid.py

Output: sorted table with adaptor arch breakdown + architecture comparison
=============================================================================
"""

import os
import re
import glob

LOG_DIR = os.path.expanduser("~/aml_lab/hpc/logs")
CKPT_DIR = os.path.expanduser("~/aml_lab/checkpoints")


def compute_arch_details(model_size, llm_hidden):
    """Compute the adaptor layer dimensions and param counts for given config.

    Architecture:
        Conv1d(112, 112, k=6, s=6)                       — FIXED (input features)
        ResBlock(112, MS)  → ResBlock(MS, MS//2) → ResBlock(MS//2, MS//4) — varies with model_size (MS)
        BiLSTM(MS//4, MS//4)  → output = MS//2            — varies with model_size
        Conv1d(MS//2, MS//2, k=2, s=2)                    — varies with model_size
        Linear(MS//2, llm_hidden) + LayerNorm(llm_hidden) — varies with BOTH
    """
    if model_size is None or llm_hidden is None:
        return None

    ms = model_size
    lh = llm_hidden

    # ResBlock widths
    res1 = f"112→{ms}"
    res2 = f"{ms}→{ms//2}"
    res3 = f"{ms//2}→{ms//4}"

    # BiLSTM
    lstm_in = ms // 4
    lstm_out = ms // 2  # bidirectional doubles

    # Final projection
    proj = f"{ms//2}→{lh}"

    arch_str = f"Res[{res1},{res2},{res3}] BiLSTM({lstm_in}→{lstm_out}) Proj({proj})"

    # What actually changes
    changes = []
    changes.append(f"ResBlocks: {res1} | {res2} | {res3}")
    changes.append(f"BiLSTM: in={lstm_in}, out={lstm_out}")
    changes.append(f"Projection: {ms//2} → {lh}")
    changes.append(f"LayerNorm: {lh}")

    return {
        "arch_short": f"Res({ms}) LSTM({lstm_in}) Proj({ms//2}→{lh})",
        "res_widths": f"{ms}/{ms//2}/{ms//4}",
        "lstm_dim": f"{lstm_in}→{lstm_out}",
        "proj_dim": f"{ms//2}→{lh}",
        "changes": changes,
    }


def parse_log(log_path):
    """Extract key metrics from a training log file."""
    result = {
        "log_file": os.path.basename(log_path),
        "llm_path": None,
        "llm_hidden": None,
        "model_size": None,
        "adaptor_params": None,
        "best_wer": None,
        "best_cer": None,
        "best_epoch": None,
        "final_epoch": None,
        "status": "unknown",
    }

    try:
        with open(log_path, 'r') as f:
            lines = f.readlines()
    except Exception as e:
        result["status"] = f"error: {e}"
        return result

    if not lines:
        result["status"] = "empty"
        return result

    best_wer = float('inf')
    best_cer = float('inf')
    best_epoch = None
    last_epoch = 0

    for line in lines:
        # LLM path
        m = re.search(r'Loading LLM:\s*(.+)', line)
        if m:
            result["llm_path"] = m.group(1).strip()

        # LLM hidden size
        m = re.search(r'LLM hidden size:\s*(\d+)', line)
        if m:
            result["llm_hidden"] = int(m.group(1))

        # Model size
        m = re.search(r'model_size=(\d+)', line)
        if m:
            result["model_size"] = int(m.group(1))

        # Adaptor params
        m = re.search(r'Adaptor params:\s*([\d,]+)', line)
        if m:
            result["adaptor_params"] = int(m.group(1).replace(',', ''))

        # Epoch progress
        m = re.search(r'Epoch \[(\d+)/\d+\] completed', line)
        if m:
            last_epoch = int(m.group(1))

        # Best WER update
        m = re.search(r'New best WER=([\d.]+),\s*CER=([\d.]+)', line)
        if m:
            wer = float(m.group(1))
            cer = float(m.group(2))
            if wer < best_wer:
                best_wer = wer
                best_cer = cer
                best_epoch = last_epoch

        # Training complete
        if 'TRAINING COMPLETE' in line:
            result["status"] = "completed"

    if best_wer < float('inf'):
        result["best_wer"] = best_wer
        result["best_cer"] = best_cer
        result["best_epoch"] = best_epoch

    result["final_epoch"] = last_epoch

    if result["status"] == "unknown":
        if last_epoch > 0:
            result["status"] = f"running (ep {last_epoch})"
        else:
            result["status"] = "starting"

    return result


def fmt(val, fmt_str="{}", default="-"):
    """Format a value, returning default if None."""
    if val is None:
        return default
    return fmt_str.format(val)


def main():
    # Find all grid search logs
    log_files = sorted(glob.glob(os.path.join(LOG_DIR, "grid_*.log")))

    # Also include the original improved runs and new model experiments
    for extra in ["improved_live.log", "phi3_live.log", "deepseek_r1_live.log",
                   "liquidai_lfm2_live.log", "smollm2_live.log", "stablelm2_live.log"]:
        path = os.path.join(LOG_DIR, extra)
        if os.path.exists(path):
            log_files.append(path)

    if not log_files:
        print("No grid search logs found in", LOG_DIR)
        print("Run: bash ~/aml_lab/hpc/run_grid_search.pbs")
        return

    results = []
    for lf in log_files:
        r = parse_log(lf)
        # Skip failed LLaMA-1B runs (no results)
        if "llama1b" in r["log_file"] and r["best_wer"] is None:
            continue
        results.append(r)

    # Sort by best WER (completed first, then by WER)
    def sort_key(r):
        if r["best_wer"] is not None:
            return (0, r["best_wer"])
        return (1, 999)

    results.sort(key=sort_key)

    # ======================================================================
    # TABLE 1: Main Results
    # ======================================================================
    print()
    print("=" * 140)
    print("TABLE 1: GRID SEARCH RESULTS (sorted by WER)")
    print("=" * 140)
    print(f"{'#':<3} {'LLM':<28} {'MS':>4} {'Hidden':>6} {'Params':>10} "
          f"{'WER':>7} {'CER':>7} {'BestEp':>6} {'Ep':>4} "
          f"{'ResBlock Widths':<16} {'LSTM':<10} {'Projection':<14} {'Status':<12}")
    print("-" * 140)

    for i, r in enumerate(results):
        llm_short = r["llm_path"].split("/")[-1] if r["llm_path"] else "-"
        # Truncate long LLM names
        if len(llm_short) > 26:
            llm_short = llm_short[:24] + ".."

        arch = compute_arch_details(r["model_size"], r["llm_hidden"])

        ms = fmt(r["model_size"], "{}")
        hidden = fmt(r["llm_hidden"], "{}")
        params = fmt(r["adaptor_params"], "{:,}")
        wer = fmt(r["best_wer"], "{:.4f}")
        cer = fmt(r["best_cer"], "{:.4f}")
        best_ep = fmt(r["best_epoch"], "{}")
        final_ep = fmt(r["final_epoch"], "{}", "0")
        res_w = arch["res_widths"] if arch else "-"
        lstm_d = arch["lstm_dim"] if arch else "-"
        proj_d = arch["proj_dim"] if arch else "-"

        print(f"{i+1:<3} {llm_short:<28} {ms:>4} {hidden:>6} {params:>10} "
              f"{wer:>7} {cer:>7} {best_ep:>6} {final_ep:>4} "
              f"{res_w:<16} {lstm_d:<10} {proj_d:<14} {r['status']:<12}")

    print("=" * 140)

    # ======================================================================
    # TABLE 2: Architecture Comparison — what's same vs different
    # ======================================================================
    print()
    print("=" * 100)
    print("TABLE 2: ADAPTOR ARCHITECTURE COMPARISON")
    print("=" * 100)
    print()
    print("FIXED across ALL configs (never changes):")
    print("  Input Conv:   Conv1d(112, 112, kernel=6, stride=6) + GELU")
    print("  Dropout:      0.15 (after ResBlocks + after LSTM)")
    print("  Decimation:   Conv1d(MS//2, MS//2, kernel=2, stride=2) + GELU")
    print("  Activations:  GELU (conv) + ReLU (resblocks)")
    print()
    print("VARIES with model_size (MS):")
    print("-" * 100)
    print(f"  {'MS':>4}  {'ResBlock 1':<14} {'ResBlock 2':<14} {'ResBlock 3':<14} "
          f"{'BiLSTM in→out':<16} {'Decim Conv':<16}")
    print("-" * 100)

    seen_ms = set()
    for r in results:
        ms = r["model_size"]
        if ms is None or ms in seen_ms:
            continue
        seen_ms.add(ms)
        print(f"  {ms:>4}  {'112→'+str(ms):<14} {str(ms)+'→'+str(ms//2):<14} "
              f"{str(ms//2)+'→'+str(ms//4):<14} "
              f"{str(ms//4)+'→'+str(ms//2):<16} "
              f"{str(ms//2)+'→'+str(ms//2):<16}")

    print()
    print("VARIES with LLM (llm_hidden):")
    print("-" * 100)
    print(f"  {'LLM':<28} {'Hidden':>6}  {'Projection (Linear)':<22} {'LayerNorm':<14}")
    print("-" * 100)

    seen_llm = set()
    for r in results:
        llm = r["llm_path"]
        if llm is None or llm in seen_llm:
            continue
        seen_llm.add(llm)
        llm_short = llm.split("/")[-1]
        lh = r["llm_hidden"]
        ms = r["model_size"]
        if lh and ms:
            print(f"  {llm_short:<28} {lh:>6}  {str(ms//2)+'→'+str(lh):<22} {str(lh):<14}")

    print()
    print("KEY INSIGHT:")
    print("  - Adaptors with the SAME model_size but DIFFERENT LLMs share identical")
    print("    internal processing (ResBlocks, LSTM) but differ ONLY in the final")
    print("    Linear projection layer (MS//2 → llm_hidden) and LayerNorm.")
    print("  - Adaptors with DIFFERENT model_size have entirely different internal")
    print("    dimensions throughout the network (ResBlocks, LSTM, Conv).")
    print("  - The projection layer is the bridge between adaptor and LLM — its size")
    print("    is determined by the LLM's embedding dimension, NOT by model_size.")
    print()

    # ======================================================================
    # TABLE 3: Summary + best result
    # ======================================================================
    completed = [r for r in results if r["best_wer"] is not None]
    if completed:
        best = min(completed, key=lambda r: r["best_wer"])
        llm_short = best["llm_path"].split("/")[-1] if best["llm_path"] else "-"

        print("=" * 100)
        print("BEST RESULT")
        print("=" * 100)
        print(f"  LLM:         {llm_short}")
        print(f"  model_size:  {best['model_size']}")
        print(f"  llm_hidden:  {best['llm_hidden']}")
        print(f"  Params:      {best['adaptor_params']:,}")
        print(f"  WER:         {best['best_wer']:.4f}")
        print(f"  CER:         {best['best_cer']:.4f}")
        print(f"  Best epoch:  {best['best_epoch']}")

        arch = compute_arch_details(best["model_size"], best["llm_hidden"])
        if arch:
            print(f"  Architecture: {arch['arch_short']}")

        print()
        print(f"  vs Paper best: 0.49 WER → {(0.49 - best['best_wer'])/0.49*100:.1f}% relative improvement")
        print("=" * 100)


if __name__ == "__main__":
    main()
