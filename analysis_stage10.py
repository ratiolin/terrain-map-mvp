import numpy as np


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
    header = f"{'kappa':>6} {'drift':>7} | {'det':>6} {'rand_noctx':>12} {'rand_ctx':>10} | {'判定':<48}"
    print()
    print("=" * 110)
    print(header)
    print("-" * 110)

    counts = {"pseudo": 0, "context": 0, "robust": 0, "none": 0}

    for key in sorted(all_results.keys()):
        t = all_results[key]
        k, d = key.replace("k", "").split("_d")
        kappa = float(k)
        drift = float(d)

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
            f"{kappa:>6.1f} {drift:>7.3f} | {det_s:>6} {rnc_s:>12} {rc_s:>10} | {verdict:<48}"
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
        k, d = key.replace("k", "").split("_d")
        print(f"\n--- κ={k}, drift={d} ---")
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
