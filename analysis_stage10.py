import json
import numpy as np

from metrics import compute_stability


def has_structure(cond):
    return (cond["mse_gap_pass"]
            and cond["switch_rate_pass"]
            and cond["separation_majority"])


def judge(triplet):
    A = has_structure(triplet["det"])
    B = has_structure(triplet["rand_noctx"])
    C = has_structure(triplet["rand_ctx"])

    if not A:
        return "— 基线未触发（det无结构）"

    if A and not B and not C:
        return "❌ 时间伪结构（依赖可预测时序）"

    if A and not B and C:
        return "✔  需显式上下文（状态可驱动路由）"

    if A and B:
        return "✅ 真实状态结构（对时间扰动鲁棒）"

    return "— 未触发/不确定"


def compute_stability_curve(all_results):
    sample_key = next(iter(all_results.keys()))
    sep = "_d" if "_d" in sample_key else "_w"
    drifts = sorted(set(
        float(key.split(sep)[1])
        for key in all_results.keys()
    ))
    drift_S = {d: [] for d in drifts}
    drift_Sadv = {d: [] for d in drifts}

    for key, triplet in all_results.items():
        drift = float(key.split(sep)[1])
        det = triplet["det"]
        v = det["variance_mean"]
        c = det["consistency_mean"]
        d = det["dwell_time_mean"]
        S_ensemble = compute_stability(v, c, d)
        S_single = det["S_single_mean"]
        S_adv = S_ensemble / (S_single + 1e-8)
        drift_S[drift].append(S_ensemble)
        drift_Sadv[drift].append(S_adv)
        triplet["S_det"] = S_ensemble
        triplet["S_adv"] = S_adv

    S_values = [float(np.mean(drift_S[d])) for d in drifts]
    S_adv_values = [float(np.mean(drift_Sadv[d])) for d in drifts]
    return drifts, S_values, S_adv_values


def export_stability_curve(drifts, S_values):
    data = {"drift": drifts, "S": S_values}
    with open("stability_curve.json", "w") as f:
        json.dump(data, f)


def check_unimodal(S_values):
    if len(S_values) < 3:
        return False
    peak_idx = int(np.argmax(S_values))
    n = len(S_values)
    if peak_idx == 0 or peak_idx == n - 1:
        return False
    left_monotonic = all(
        S_values[i] <= S_values[i + 1]
        for i in range(peak_idx)
    )
    right_monotonic = all(
        S_values[i] >= S_values[i + 1]
        for i in range(peak_idx, n - 1)
    )
    return left_monotonic and right_monotonic


def plot_stability_curve(drifts, S_values):
    import matplotlib.pyplot as plt
    plt.plot(drifts, S_values, marker='o')
    plt.xlabel("Drift")
    plt.ylabel("Stability S")
    plt.title("Stability vs Drift")
    plt.grid(True)
    plt.savefig("stability_curve.png")
    print("\nStability curve saved to stability_curve.png")


def stability_analysis(all_results):
    drifts, S_values, S_adv_values = compute_stability_curve(all_results)

    sep = "_d" if "_d" in next(iter(all_results.keys())) else "_w"
    drift_Ssingle = {d: [] for d in drifts}
    for key, triplet in all_results.items():
        drift = float(key.split(sep)[1])
        det = triplet["det"]
        drift_Ssingle[drift].append(det["S_single_mean"])

    print()
    print("=" * 60)
    print("STABILITY CURVE ANALYSIS")
    print("=" * 60)
    for d, s, sa in zip(drifts, S_values, S_adv_values):
        ss = float(np.mean(drift_Ssingle[d]))
        print(f"  drift={d:.3f}  S_ensemble={s:.4f}  S_single={ss:.4f}  S_adv={sa:.4f}")

    export_stability_curve(drifts, S_adv_values)
    print("\nExported stability_curve.json (S_adv)")

    plot_stability_curve(drifts, S_adv_values)

    is_unimodal = check_unimodal(S_adv_values)
    print(f"\nUnimodal (中间高，两边低): {'YES' if is_unimodal else 'NO'}")
    return is_unimodal


def diagnose_triplet(triplet):
    diag = {}
    for tag in ["det", "rand_noctx", "rand_ctx"]:
        t = triplet[tag]
        large_mse = t["test_large_mean"]
        oracle_mse = t["oracle_mse_mean"]
        routing_gap = t["routing_gap_mean"]
        multi_mse = float(np.mean([s["test_mse"] for s in t["seeds"]]))

        decomposable = (oracle_mse < 0.5 * large_mse)
        routing_ok = (routing_gap < 0.5 * oracle_mse)
        unlearnable = (oracle_mse > max(1.0, large_mse))

        if large_mse < 0.1 or (not decomposable and not unlearnable):
            category = "trivial"
            reason = "环境太简单（C1失败，单专家Large已收敛）"
        elif unlearnable:
            category = "unlearnable"
            reason = "环境变化超出模型表达能力（oracle也高）"
        elif decomposable and not routing_ok:
            category = "capacity_ok_routing_fail"
            reason = "结构存在，但routing跟不上（时间尺度上限）"
        elif decomposable and routing_ok:
            category = "working"
            reason = "结构可分解且routing有效"
        else:
            category = "other"
            reason = f"large={large_mse:.3f} oracle={oracle_mse:.3f} gap={routing_gap:+.3f}"

        diag[tag] = {
            "category": category,
            "reason": reason,
            "large_mse": large_mse,
            "oracle_mse": oracle_mse,
            "routing_gap": routing_gap,
            "multi_mse": multi_mse,
        }
    return diag


def print_matrix(all_results):
    sample_key = next(iter(all_results.keys()))
    param_label = "drift" if "_d" in sample_key else "omega"
    header = f"{'kappa':>6} {param_label:>7} | {'det':>6} {'rand_noctx':>12} {'rand_ctx':>10} | {'判定':<48}"
    print()
    print("=" * 110)
    print(header)
    print("-" * 110)

    counts = {"pseudo": 0, "context": 0, "robust": 0, "none": 0}

    for key in sorted(all_results.keys()):
        t = all_results[key]
        if "_d" in key:
            k, d = key.replace("k", "").split("_d")
            param_label = "drift"
        else:
            k, d = key.replace("k", "").split("_w")
            param_label = "omega"
        kappa = float(k)
        param_val = float(d)

        det_s = "STR" if has_structure(t["det"]) else "---"
        rnc_s = "STR" if has_structure(t["rand_noctx"]) else "---"
        rc_s = "STR" if has_structure(t["rand_ctx"]) else "---"
        verdict = judge(t)

        if verdict.startswith("❌"):
            counts["pseudo"] += 1
        elif verdict.startswith("✔"):
            counts["context"] += 1
        elif verdict.startswith("✅"):
            counts["robust"] += 1
        else:
            counts["none"] += 1

        print(
            f"{kappa:>6.1f} {param_val:>7.3f} | {det_s:>6} {rnc_s:>12} {rc_s:>10} | {verdict:<48}"
        )

    print("-" * 110)
    print(f"\nSummary: ❌伪结构={counts['pseudo']}  ✔需上下文={counts['context']}  "
          f"✅鲁棒={counts['robust']}  —未触发={counts['none']}")
    print("=" * 110)


def print_diagnosis(all_results):
    print()
    print("=" * 110)
    print("DIAGNOSIS (oracle / routing_gap / classification)")
    print("=" * 110)

    for key in sorted(all_results.keys()):
        t = all_results[key]
        diag = diagnose_triplet(t)
        sep_key = "_d" if "_d" in key else "_w"
        k, d = key.replace("k", "").split(sep_key)
        param_label_key = "drift" if sep_key == "_d" else "omega"
        print(f"\n--- κ={k}, {param_label_key}={d} ---")
        for tag in ["det", "rand_noctx", "rand_ctx"]:
            dg = diag[tag]
            print(f"  {tag:15s}: large={dg['large_mse']:.4f}  oracle={dg['oracle_mse']:.4f}  "
                  f"r_gap={dg['routing_gap']:+.4f}  multi={dg['multi_mse']:.4f}")
            print(f"  {'':15s}  → [{dg['category']}] {dg['reason']}")

    print("\n" + "=" * 110)
    print("Legend:")
    print("  trivial:                single_large already solves the task")
    print("  unlearnable:            even oracle (best expert) cannot fit")
    print("  capacity_ok_routing_fail: experts CAN fit but gating selects wrong one")
    print("  working:                both decomposable and routing effective")
    print("=" * 110)


def extract_R_data(all_results):
    sep = "_d" if "_d" in next(iter(all_results.keys())) else "_w"
    drift_response = {}
    for key, triplet in all_results.items():
        drift = float(key.split(sep)[1])
        det = triplet["det"]
        tau_r = det["response_time_mean"]
        if drift not in drift_response:
            drift_response[drift] = []
        drift_response[drift].append(tau_r)

    drifts = sorted(drift_response.keys())
    tau_response_avg = [float(np.mean(drift_response[d])) for d in drifts]

    S_adv = {}
    for key, triplet in all_results.items():
        drift = float(key.split(sep)[1])
        if drift not in S_adv:
            S_adv[drift] = []
        S_adv[drift].append(triplet.get("S_adv", 1.0))
    S_adv_avg = [float(np.mean(S_adv[d])) for d in drifts]

    tau_drift = [1.0 / d for d in drifts]
    R_values = [tau_response_avg[i] / tau_drift[i] for i in range(len(drifts))]

    return drifts, tau_response_avg, tau_drift, R_values, S_adv_avg


def fit_threshold(R_values, S_adv_values, drifts):
    high_r_candidates = []
    low_r_candidates = []
    for i, sa in enumerate(S_adv_values):
        if sa > 1.0:
            high_r_candidates.append(R_values[i])
        else:
            low_r_candidates.append(R_values[i])

    if high_r_candidates and low_r_candidates:
        tau_star = (max(low_r_candidates) + min(high_r_candidates)) / 2
    elif high_r_candidates:
        tau_star = max(high_r_candidates) * 1.5
    else:
        tau_star = min(low_r_candidates) * 0.5

    direction = ""
    return tau_star, direction


def interpolate_tau_response(drifts_known, tau_known, drift_target):
    return float(np.interp(drift_target, drifts_known, tau_known))


def run_R_analysis(all_results):
    drifts, tau_resp, tau_drift, R_values, S_adv_values = extract_R_data(all_results)

    print()
    print("=" * 60)
    print("R ANALYSIS (τ_response / τ_drift)")
    print("=" * 60)
    print(f"{'drift':>8} {'τ_resp':>8} {'τ_drift':>8} {'R':>8} {'S_adv':>8}")
    print("-" * 48)
    for i in range(len(drifts)):
        print(f"{drifts[i]:>8.3f} {tau_resp[i]:>8.1f} {tau_drift[i]:>8.1f} "
              f"{R_values[i]:>8.4f} {S_adv_values[i]:>8.4f}")

    tau_star, direction = fit_threshold(R_values, S_adv_values, drifts)
    print(f"\nτ* = {tau_star:.4f}")

    return drifts, tau_resp, tau_drift, R_values, S_adv_values, tau_star
