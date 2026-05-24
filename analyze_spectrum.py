import json
import os
import glob

import numpy as np
import matplotlib.pyplot as plt


def compute_fft(signal):
    fft = np.fft.fft(signal)
    freq = np.fft.fftfreq(len(signal))
    return freq, np.abs(fft)


def plot_single_spectrum(filepath, save_path=None):
    data = np.load(filepath)
    freq, amp = compute_fft(data)
    n = len(freq) // 2
    plt.figure()
    plt.plot(freq[:n], amp[:n])
    plt.xlim(0, 0.5)
    plt.xlabel("Frequency")
    plt.ylabel("Amplitude")
    plt.title(f"Drift Spectrum: {os.path.basename(filepath)}")
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def batch_plot_spectra(data_dir="spectral_data", output_path="drift_spectrum_batch.png"):
    files = sorted(glob.glob(os.path.join(data_dir, "drift_*.npy")))
    if not files:
        print(f"No drift data found in {data_dir}/")
        return

    plt.figure(figsize=(12, 8))
    omega_groups = {}
    tag_map = {"det": "det", "rand-noctx": "rand_noctx", "rand-ctx": "rand_ctx"}
    for f in files:
        basename = os.path.basename(f).replace(".npy", "")
        parts = basename.split("_")
        k_val = [p for p in parts if p.startswith("k")][0][1:]
        w_val = [p for p in parts if p.startswith("w")][0][1:]
        tag_raw = [p for p in parts if p in ("det", "rand-noctx", "rand-ctx")][0]
        tag = tag_map.get(tag_raw, tag_raw)
        s_val = [p for p in parts if p.startswith("s")][0][1:]
        key = f"kappa={k_val} omega={w_val} {tag}"
        if key not in omega_groups:
            omega_groups[key] = []
        data = np.load(f)
        freq, amp = compute_fft(data)
        n = len(freq) // 2
        omega_groups[key].append((freq[:n], amp[:n]))

    for label, traces in omega_groups.items():
        avg_amp = np.mean([t[1] for t in traces], axis=0)
        freq = traces[0][0]
        plt.plot(freq, avg_amp, label=label, alpha=0.8)

    plt.xlim(0, 0.5)
    plt.xlabel("Frequency")
    plt.ylabel("Amplitude")
    plt.title("Drift Spectra (averaged over seeds)")
    plt.legend(fontsize=7, loc="upper right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Batch spectrum plot saved to {output_path}")


def compute_model_transfer(perf_file="spectral_performance.json",
                           data_dir="spectral_data",
                           output_file="model_transfer.json"):
    with open(perf_file, "r") as f:
        perf_data = json.load(f)

    results = []
    for entry in perf_data:
        kappa = entry["kappa"]
        omega = entry["omega"]

        drift_pattern = os.path.join(data_dir, f"drift_k{kappa}_w{omega}_det_*.npy")
        error_pattern = os.path.join(data_dir, f"error_k{kappa}_w{omega}_det_*.npy")
        drift_files = sorted(glob.glob(drift_pattern))
        error_files = sorted(glob.glob(error_pattern))

        if not drift_files or not error_files:
            continue

        err_at_omega = []
        drift_at_omega = []

        for df, ef in zip(drift_files, error_files):
            drift_signal = np.load(df)
            error_signal = np.load(ef)
            min_len = min(len(drift_signal), len(error_signal))
            drift_signal = drift_signal[:min_len]
            error_signal = error_signal[:min_len]

            freq, fft_err = compute_fft(error_signal)
            _, fft_drift = compute_fft(drift_signal)
            n = len(freq) // 2

            target_freq = omega / (2 * np.pi)
            idx = np.argmin(np.abs(freq[:n] - target_freq))
            err_at_omega.append(float(fft_err[idx]))
            drift_at_omega.append(float(fft_drift[idx]))

        transfer = float(np.mean(err_at_omega)) / (float(np.mean(drift_at_omega)) + 1e-8)
        results.append({
            "kappa": kappa,
            "omega": omega,
            "model_transfer": transfer,
            "err_at_omega": float(np.mean(err_at_omega)),
            "drift_at_omega": float(np.mean(drift_at_omega)),
            "mse_gap_mean": entry["mse_gap_mean"],
            "verdict": entry.get("verdict", ""),
        })

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nModel transfer data saved to {output_file}")

    print("\n" + "=" * 60)
    print("MODEL TRANSFER FUNCTION")
    print("=" * 60)
    print(f"{'kappa':>6} {'omega':>8} {'transfer':>10} {'mse_gap':>10} {'verdict':<48}")
    print("-" * 90)
    for r in results:
        print(f"{r['kappa']:>6.1f} {r['omega']:>8.4f} "
              f"{r['model_transfer']:>10.4f} {r['mse_gap_mean']:>10.4f} "
              f"{r['verdict']:<48}")

    return results


def align_performance_with_peak(perf_file="spectral_performance.json",
                                 data_dir="spectral_data",
                                 output_file="spectral_alignment.json"):
    with open(perf_file, "r") as f:
        perf_data = json.load(f)

    results = []
    for entry in perf_data:
        kappa = entry["kappa"]
        omega = entry["omega"]

        pattern = os.path.join(data_dir, f"drift_k{kappa}_w{omega}_det_*.npy")
        drift_files = sorted(glob.glob(pattern))
        if not drift_files:
            continue

        peak_freqs = []
        for f in drift_files:
            data = np.load(f)
            freq, amp = compute_fft(data)
            n = len(freq) // 2
            peak_idx = np.argmax(amp[:n])
            peak_freqs.append(float(freq[peak_idx]))

        results.append({
            "kappa": kappa,
            "omega": omega,
            "mse_gap_mean": entry["mse_gap_mean"],
            "test_large_mean": entry["test_large_mean"],
            "oracle_mse_mean": entry["oracle_mse_mean"],
            "routing_gap_mean": entry["routing_gap_mean"],
            "verdict": entry.get("verdict", ""),
            "peak_freq": float(np.mean(peak_freqs)),
            "peak_freq_std": float(np.std(peak_freqs)) if len(peak_freqs) > 1 else 0.0,
        })

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nAlignment data saved to {output_file}")

    print("\n" + "=" * 60)
    print("SPECTRUM-PERFORMANCE ALIGNMENT")
    print("=" * 60)
    print(f"{'kappa':>6} {'omega':>8} {'peak_freq':>10} {'mse_gap':>10} {'verdict':<48}")
    print("-" * 90)
    for r in results:
        print(f"{r['kappa']:>6.1f} {r['omega']:>8.4f} "
              f"{r['peak_freq']:>10.4f} {r['mse_gap_mean']:>10.4f} "
              f"{r['verdict']:<48}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Drift spectrum analysis")
    parser.add_argument("--batch", action="store_true", help="Batch plot all spectra")
    parser.add_argument("--align", action="store_true", help="Align performance with peak frequency")
    parser.add_argument("--transfer", action="store_true", help="Compute model transfer function")
    parser.add_argument("--data-dir", default="spectral_data", help="Directory containing drift data")
    parser.add_argument("--perf-file", default="spectral_performance.json", help="Performance data file")

    args = parser.parse_args()

    if args.batch or (not args.batch and not args.align and not args.transfer):
        batch_plot_spectra(data_dir=args.data_dir)

    if args.align or (not args.batch and not args.align and not args.transfer):
        if os.path.exists(args.perf_file):
            align_performance_with_peak(perf_file=args.perf_file, data_dir=args.data_dir)
        else:
            print(f"Performance file {args.perf_file} not found; run main_stage10_spectral.py first")

    if args.transfer:
        if os.path.exists(args.perf_file):
            compute_model_transfer(perf_file=args.perf_file, data_dir=args.data_dir)
        else:
            print(f"Performance file {args.perf_file} not found; run main_stage10_spectral.py first")


def compute_memory_transfer_curve(data_dir="memory_data", output_file="memory_transfer.json"):
    import re
    files = sorted(glob.glob(os.path.join(data_dir, "*_drift.npy")))
    if not files:
        print(f"No memory data found in {data_dir}/")
        return

    mem_groups = {}
    for f in files:
        basename = os.path.basename(f).replace("_drift.npy", "")
        match = re.match(r"mem([\d.]+)_k([\d.]+)_w([\d.]+)_det_s(\d+)", basename)
        if not match:
            continue
        mem_val = float(match.group(1))
        kappa_val = float(match.group(2))
        omega_val = float(match.group(3))

        drift_file = f
        error_file = f.replace("_drift.npy", "_error.npy")
        if not os.path.exists(error_file):
            continue

        drift_signal = np.load(drift_file)
        error_signal = np.load(error_file)
        min_len = min(len(drift_signal), len(error_signal))
        drift_signal = drift_signal[:min_len]
        error_signal = error_signal[:min_len]

        freq, fft_err = compute_fft(error_signal)
        _, fft_drift = compute_fft(drift_signal)
        n = len(freq) // 2

        target_freq = omega_val / (2 * np.pi)
        idx = np.argmin(np.abs(freq[:n] - target_freq))
        transfer = float(fft_err[idx]) / (float(fft_drift[idx]) + 1e-8)

        key = (mem_val, kappa_val, omega_val)
        if key not in mem_groups:
            mem_groups[key] = []
        mem_groups[key].append(transfer)

    results = {}
    for (mem_val, kappa_val, omega_val), transfers in mem_groups.items():
        if mem_val not in results:
            results[mem_val] = []
        results[mem_val].append({
            "kappa": kappa_val,
            "omega": omega_val,
            "model_transfer": float(np.mean(transfers)),
        })

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nMemory transfer data saved to {output_file}")

    print("\n" + "=" * 60)
    print("MODEL TRANSFER vs MEMORY")
    print("=" * 60)
    for mem_val in sorted(results.keys()):
        entries = sorted(results[mem_val], key=lambda x: x["omega"])
        print(f"\n--- memory={mem_val} ---")
        print(f"  {'omega':>8} {'transfer':>10}")
        print(f"  {'-'*20}")
        for e in entries:
            print(f"  {e['omega']:>8.4f} {e['model_transfer']:>10.4f}")

    return results


def align_transfer_performance(transfer_file="memory_transfer.json",
                                perf_file="memory_performance.json",
                                output_plot="transfer_vs_performance.png"):
    import re

    with open(transfer_file) as f:
        transfer_data = json.load(f)
    with open(perf_file) as f:
        perf_entries = json.load(f)

    perf_map = {}
    for entry in perf_entries:
        key = entry["key"]
        m = re.match(r"k([\d.]+)_w([\d.]+)_k(\d+)", key)
        if m:
            kappa = float(m.group(1))
            omega = float(m.group(2))
            mem = float(m.group(3))
            perf_map[(mem, omega)] = entry["mse_gap_mean"]

    mem_vals = sorted(set(float(k) for k in transfer_data.keys()))
    n = len(mem_vals)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    results = {}

    for ax, mem_val in zip(axes, mem_vals):
        key_str = str(mem_val)
        entries = sorted(transfer_data[key_str], key=lambda x: x["omega"])
        omegas = [e["omega"] for e in entries]
        transfers = [e["model_transfer"] for e in entries]
        perfs = [perf_map.get((mem_val, o), 0.0) for o in omegas]

        ax2 = ax.twinx()
        line1, = ax.plot(omegas, transfers, 'b-o', linewidth=2, label='Model_transfer')
        line2, = ax2.plot(omegas, perfs, 'r-s', linewidth=2, label='Performance (mse_gap)')
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

        ax.set_xlabel('omega')
        ax.set_ylabel('Model_transfer', color='b')
        ax2.set_ylabel('mse_gap (performance)', color='r')
        ax.set_title(f'memory_k = {int(mem_val)}')
        ax.grid(True, alpha=0.3)
        lines = [line1, line2]
        ax.legend(lines, [l.get_label() for l in lines], loc='upper right', fontsize=8)

        corr = float(np.corrcoef(transfers, perfs)[0, 1]) if len(omegas) >= 3 else 0.0
        peak_t = omegas[np.argmax(transfers)]
        peak_p = omegas[np.argmax(perfs)]

        category = "unknown"
        t_has_structure = max(transfers) - min(transfers) > 0.2
        p_has_structure = max(perfs) - min(perfs) > 0.1
        if abs(corr) > 0.5 and t_has_structure and p_has_structure:
            category = "A: performance ∝ transfer"
        elif t_has_structure and not p_has_structure:
            category = "B: transfer有结构, performance没结构"
        elif not t_has_structure:
            category = "transfer无结构"
        else:
            category = "不确定"

        results[str(mem_val)] = {
            "correlation": round(corr, 4),
            "peak_transfer_omega": round(peak_t, 4),
            "peak_perf_omega": round(peak_p, 4),
            "transfer_range": round(max(transfers) - min(transfers), 4),
            "perf_range": round(max(perfs) - min(perfs), 4),
            "category": category,
        }

        ax.set_title(f'memory_k = {int(mem_val)}  [{category[:20]}]', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_plot, dpi=150)
    plt.close()
    print(f"\nTransfer vs Performance plot saved to {output_plot}")

    print("\n" + "=" * 60)
    print("CATEGORY ANALYSIS")
    print("=" * 60)
    for mem_val in sorted(results.keys()):
        r = results[mem_val]
        print(f"\nmemory_k={mem_val}:  r={r['correlation']:.4f}  "
              f"transfer_range={r['transfer_range']:.4f}  perf_range={r['perf_range']:.4f}")
        print(f"  peak_transfer_ω={r['peak_transfer_omega']:.4f}  peak_perf_ω={r['peak_perf_omega']:.4f}")
        print(f"  → {r['category']}")

    return results
