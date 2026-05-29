import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.mixture import GaussianMixture
from scipy.stats import wasserstein_distance, spearmanr
from scipy.optimize import curve_fit


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


def rank_J(S, tol_scale=1e-3):
    S = np.asarray(S, dtype=np.float64)
    if S.max() == 0:
        return 0
    return int(np.sum(S > tol_scale * S.max()))


def spectral_entropy(S):
    S = np.asarray(S, dtype=np.float64)
    s_sum = np.sum(S)
    if s_sum == 0:
        return 0.0
    p = S / s_sum
    p = p[p > 0]
    if len(p) == 0:
        return 0.0
    return float(-np.sum(p * np.log(p)))


def compute_wasserstein(samples1, samples2):
    """Compute 1D/1D Wasserstein distance per dimension, then average."""
    s1 = np.asarray(samples1)
    s2 = np.asarray(samples2)
    if s1.ndim == 1:
        return float(wasserstein_distance(s1, s2))
    assert s1.shape[1] == s2.shape[1]
    dists = []
    for dim in range(s1.shape[1]):
        dists.append(wasserstein_distance(s1[:, dim], s2[:, dim]))
    return float(np.mean(dists))


def compute_w2_gaussian(samples1, samples2):
    """Compute 2-Wasserstein distance under Gaussian approximation.

    W2^2 = ||mu1 - mu2||^2 + tr(S1 + S2 - 2*(S1^{1/2} S2 S1^{1/2})^{1/2})

    This gives a proper multivariate distance metric. Falls back to
    per-dimension W1 average if the matrix square root fails.
    """
    s1 = np.asarray(samples1, dtype=np.float64)
    s2 = np.asarray(samples2, dtype=np.float64)

    if s1.ndim == 1:
        s1 = s1.reshape(-1, 1)
    if s2.ndim == 1:
        s2 = s2.reshape(-1, 1)

    mu1 = np.mean(s1, axis=0)
    mu2 = np.mean(s2, axis=0)
    mean_diff_sq = np.sum((mu1 - mu2) ** 2)

    try:
        S1 = np.cov(s1, rowvar=False)
        S2 = np.cov(s2, rowvar=False)
        S1 += np.eye(S1.shape[0]) * 1e-8
        S2 += np.eye(S2.shape[0]) * 1e-8

        sqrt_S1 = _matrix_sqrt(S1)
        inner = sqrt_S1 @ S2 @ sqrt_S1
        sqrt_inner = _matrix_sqrt(inner)
        trace_term = np.trace(S1 + S2 - 2.0 * sqrt_inner)

        w2_sq = mean_diff_sq + max(trace_term, 0.0)
        return float(np.sqrt(w2_sq))
    except Exception:
        return compute_wasserstein(s1, s2)


def _matrix_sqrt(A):
    """Compute matrix square root via eigendecomposition."""
    eigvals, eigvecs = np.linalg.eigh(A)
    eigvals = np.maximum(eigvals, 0.0)
    return eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T


def compute_cka(X, Y):
    """Centered Kernel Alignment between two matrices. Uses linear kernel."""
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)

    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)

    K_X = X @ X.T
    K_Y = Y @ Y.T

    K_X = K_X - K_X.mean(axis=0, keepdims=True)
    K_Y = K_Y - K_Y.mean(axis=0, keepdims=True)

    hsic = np.sum(K_X * K_Y)
    hsic_X = np.sqrt(np.sum(K_X * K_X))
    hsic_Y = np.sqrt(np.sum(K_Y * K_Y))

    if hsic_X == 0 or hsic_Y == 0:
        return 0.0

    return float(hsic / (hsic_X * hsic_Y))


def fit_gmm_and_select(data, max_components=2):
    """Fit GMM with 1 and 2 components, select via BIC."""
    data = np.asarray(data, dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    if len(data) < 10:
        return 1, None

    best_components = 1
    best_bic = float('inf')
    models = {}

    for n_comp in range(1, max_components + 1):
        try:
            gmm = GaussianMixture(n_components=n_comp, random_state=0,
                                  covariance_type='full', n_init=3)
            gmm.fit(data)
            bic = gmm.bic(data)
            models[n_comp] = (gmm, bic)
            if bic < best_bic:
                best_bic = bic
                best_components = n_comp
        except Exception:
            return 1, None

    return best_components, models


def sigmoid_func(x, a, b, c, d):
    return a + (b - a) / (1.0 + np.exp(-c * (x - d)))


def fit_sigmoid_proportion(x, y):
    """Fit sigmoid to proportion data. Returns slope (c) and inflection point (d)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    try:
        p0 = [0.0, 1.0, 1.0, np.median(x)]
        popt, _ = curve_fit(sigmoid_func, x, y, p0=p0, maxfev=5000)
        return {
            "a": float(popt[0]),
            "b": float(popt[1]),
            "slope_c": float(popt[2]),
            "inflection_d": float(popt[3]),
            "r2": float(1.0 - np.sum((y - sigmoid_func(x, *popt)) ** 2) / np.sum((y - np.mean(y)) ** 2)),
        }
    except Exception:
        return {"slope_c": 0.0, "inflection_d": 0.0, "r2": 0.0}


def compute_spearman_correlation(x, y):
    """Spearman rank correlation."""
    import warnings
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = ~(np.isnan(x) | np.isnan(y))
    if np.sum(mask) < 3:
        return float('nan'), 1.0
    if np.std(x[mask]) == 0 or np.std(y[mask]) == 0:
        return float('nan'), 1.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rho, pval = spearmanr(x[mask], y[mask])
    return float(rho) if not np.isnan(rho) else float('nan'), float(pval) if not np.isnan(pval) else 1.0


def analyze_jacobian_spectrum(Js_batch):
    """Analyze Jacobian spectra across a batch of states."""
    results = []
    for J in Js_batch:
        U, S, Vt = np.linalg.svd(J, full_matrices=False)
        V = Vt.T
        results.append({
            "S": S.tolist(),
            "k80": compute_k80(S),
            "effective_rank": effective_rank(S),
            "rank": rank_J(S),
            "spectral_entropy": spectral_entropy(S),
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


def compute_controllability(env, model, s):
    try:
        a_actual = model.act_numpy(s)
        s_next_actual = env.forward_static(s, a_actual)
        s_next_zero = env.forward_static(s, np.zeros(env.action_dim))
        delta = s_next_actual[:env.k] - s_next_zero[:env.k]
        return float(np.linalg.norm(delta))
    except Exception:
        return 0.0
