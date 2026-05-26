import numpy as np
from sklearn.linear_model import LinearRegression


def compute_k80(S):
    S = np.asarray(S, dtype=np.float64)
    if np.sum(S ** 2) == 0:
        return 1
    cumsum = np.cumsum(S ** 2) / np.sum(S ** 2)
    idx = np.argmax(cumsum >= 0.80)
    return int(idx) + 1


def effective_rank(S):
    S = np.asarray(S, dtype=np.float64)
    sum_sq = np.sum(S ** 2)
    if sum_sq == 0:
        return 1.0
    p = S ** 2 / sum_sq
    p = p[p > 0]
    if len(p) == 0:
        return 1.0
    return float(np.exp(-np.sum(p * np.log(p))))


def alignment(V1, V2, k=None):
    V1 = np.asarray(V1, dtype=np.float64)
    V2 = np.asarray(V2, dtype=np.float64)
    if k is None:
        k = min(V1.shape[1], V2.shape[1])
    M = V1[:, :k].T @ V2[:, :k]
    return float(np.linalg.norm(M, ord='fro') / np.sqrt(float(k)))


def R2_probe(h, C):
    h = np.asarray(h, dtype=np.float64)
    C = np.asarray(C, dtype=np.float64)
    if h.ndim == 1:
        h = h.reshape(1, -1)
    if C.ndim == 1:
        C = C.reshape(-1, 1)
    model = LinearRegression().fit(h, C)
    return float(model.score(h, C))


def rank_J(S, tol_scale=1e-3):
    S = np.asarray(S, dtype=np.float64)
    if S.max() == 0:
        return 0
    return int(np.sum(S > tol_scale * S.max()))


def compute_controllability(env, model, s, a_zero=None):
    if a_zero is None:
        a_zero = np.zeros(env.action_dim)
    state_np = s if isinstance(s, np.ndarray) else np.array(s, dtype=np.float32)
    from .models import compute_jacobian
    try:
        a_actual = model.act_numpy(state_np)
        s_next_actual = env.forward_static(state_np, a_actual)
        s_next_zero = env.forward_static(state_np, a_zero)
        delta = s_next_actual[:env.k] - s_next_zero[:env.k]
        return float(np.linalg.norm(delta))
    except AttributeError:
        return 0.0


def von_neumann_entropy(S):
    S = np.asarray(S, dtype=np.float64)
    s_sum = np.sum(S)
    if s_sum == 0:
        return 0.0
    p = S / s_sum
    p = p[p > 0]
    if len(p) == 0:
        return 0.0
    return float(-np.sum(p * np.log(p)))


def spectral_entropy(S):
    return von_neumann_entropy(S)


def cross_seed_consistency(models_list, states, k=None):
    if len(models_list) < 2:
        return 1.0
    all_V = []
    for model in models_list:
        Js = []
        for s in states:
            from .models import compute_jacobian
            J = compute_jacobian(model, s)
            Js.append(J)
        J_mean = np.mean(Js, axis=0)
        _, S, Vt = np.linalg.svd(J_mean, full_matrices=False)
        V = Vt.T
        if k is None:
            k_use = compute_k80(S)
        else:
            k_use = min(k, S.shape[0])
        all_V.append(V[:, :k_use])

    alignments = []
    for i in range(len(all_V)):
        for j in range(i + 1, len(all_V)):
            alignments.append(alignment(all_V[i], all_V[j], k=k_use))
    return float(np.mean(alignments)) if alignments else 1.0


def alignment_time(model, env, states, delta=100):
    Js_time = []
    for t, s in enumerate(states):
        from .models import compute_jacobian
        J = compute_jacobian(model, s)
        Js_time.append(J)

    if len(Js_time) <= delta:
        return []

    alignments = []
    for t in range(delta, len(Js_time)):
        J1 = Js_time[t]
        J2 = Js_time[t - delta]
        _, S1, Vt1 = np.linalg.svd(J1, full_matrices=False)
        _, S2, Vt2 = np.linalg.svd(J2, full_matrices=False)
        k = min(compute_k80(S1), compute_k80(S2), S1.shape[0], S2.shape[0])
        V1 = Vt1.T[:, :k]
        V2 = Vt2.T[:, :k]
        alignments.append(alignment(V1, V2, k=k))

    return alignments


def analyze_jacobian(model, states):
    results = []
    for s in states:
        from .models import compute_jacobian
        J = compute_jacobian(model, s)
        U, S, Vt = np.linalg.svd(J, full_matrices=False)
        V = Vt.T

        results.append({
            "S": S.copy(),
            "V": V.copy(),
            "k80": compute_k80(S),
            "effective_rank_val": effective_rank(S),
            "rank_J_val": rank_J(S),
            "spectral_entropy_val": spectral_entropy(S),
        })
    return results


def failure_criteria(k80, d, R2, alignment_val):
    R2_fail = R2 < 0.2
    non_lowrank = (k80 / d) > 0.5 if d > 0 else True
    unstable = alignment_val < 0.3
    return {
        "R2_fail": bool(R2_fail),
        "non_lowrank": bool(non_lowrank),
        "unstable": bool(unstable),
        "any_fail": bool(R2_fail or non_lowrank or unstable),
    }


def compute_silhouette_scores(hidden_states, labels):
    from sklearn.metrics import silhouette_score as sk_silhouette
    try:
        X = np.array(hidden_states)
        lbl = np.array(labels)
        if X.shape[0] < 3 or len(np.unique(lbl)) < 2:
            return 0.0
        return float(sk_silhouette(X, lbl))
    except Exception:
        return 0.0
