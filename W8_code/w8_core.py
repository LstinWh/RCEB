"""
W8 — Spectral-credibility-weighted robust unmixing with explicit regularization
     control and trust-region dual-channel consensus.

SELF-CONTAINED: this module depends on NumPy ONLY.  It does not import anything
from S2_code/.  Copy this single file into a paper's code release and it runs.

=============================================================================
1. WHAT THE METHOD OPTIMISES  (paper notation)
=============================================================================
Let Y = [y_1,...,y_N] in R^{L x N} be the observed (possibly corrupted) cube,
E in R^{L x P} the endmembers, A = [a_1,...,a_N] in R^{P x N} the abundances,
o_n in {0,1} the observability indicator and D the set of dead pixels.

(a) Nominal weighted robust-fitting objective (S2's design intent):

    min_{E>=0, A>=0}  J(E,A) = sum_n o_n * w_n * || y_n - E a_n ||_2^2
    s.t.  1^T a_n = 1 (ASC),   a_n = a_n^{(0)}  for n in D.

    w_n = sigmoid( gamma * (LFE_n - tau) ),
    LFE_n = sum_{k in Omega_low} |F_n(k)|^2 / sum_k |F_n(k)|^2,
    Omega_low = the lowest `keep` fraction of |fft_freq| bins,
    F = 1D-FFT along the wavelength axis,  tau = median(LFE over observed pixels).
    Dead pixels receive w_n = 0.

(b) ACTUAL solver operator T_K(Y, w) -- an honest caveat that MUST go into the
    paper.  The implementation uses the multiplicative updates

        E <- E * (Y_o A^T) / ( E (A*w) A^T + eps )
        A <- Pi_Delta[ A * (E^T Y_o) / ( (E^T E)(A*w) + eps ) ]

    where w enters the DENOMINATOR ONLY.  This is NOT the legitimate MU rule of
    J above (the correct rule weights the numerator as well).  It is a biased,
    non-convergent iteration: the abundance RMSE decreases for a few steps and
    then INCREASES monotonically.  Hence the iteration budget K is the effective
    regularisation strength, and `K = 10` (fixed for all scenes/conditions) is
    an EXPLICIT EARLY STOP.  A "properly derived" weighted MU converges to a
    WORSE solution.  Do not claim convergence to J in the paper; describe the
    method as "credibility-weighted fitting with explicit regularisation control".

(c) W8's final estimate is the closed-form minimiser of a trust-region
    consensus objective over two complementary candidate solutions A_3, A_7:

    min_{A >= 0, 1^T a_n = 1}
        Phi(A) = (1 - eta) ||A - A_c||_F^2
                 + (eta/2) ( ||A - A_3||_F^2 + ||A - A_7||_F^2 ),
        eta = 0.5 (fixed a priori, never tuned per dataset).

    Unconstrained solution:  A* = (1 - eta) A_c + eta (A_3 + A_7)/2,
    followed by the ASC projection and the dead-pixel reset.  eta = 0.5 bounds
    the worst-case regression to one half of the correction's error.

=============================================================================
2. WHY TWO CHANNELS
=============================================================================
  Channel 3 (SIGNAL):  w_3 = 0.7 w_0 + 0.3 w_SAM, where the spectral angle
      between y_n and the core reconstruction E_c a_n is SCALE-INVARIANT and
      therefore orthogonal to the energy-based LFE.  A_3 = T_K(Y, w_3).
  Channel 7 (REGULARISATION TRAJECTORY):  A_7 = (A_c + T_{K/2}(Y, w_0))/2,
      i.e. bagging over two stopping budgets.  It acts on the iteration count,
      not on the credibility signal.
The two corrections have weakly correlated errors; averaging them is a variance
reduction that is empirically strictly non-negative on all 16 evaluated cells.

=============================================================================
3. REPORTED NUMBERS (4 datasets x 4 degradation conditions, RMSE)
=============================================================================
  vs Baseline (plain masked NMF, 50 iters) : +28.61% mean gain
  vs S2       (LFE gate, 50 iters)         : +18.28% mean gain
  vs S2-E     (LFE gate, K=10, no W8)      :  +1.55% mean gain, 16/16 cells win
  IMPORTANT: ~17.19 of the 18.28 points vs S2 come from the explicit early
  stopping; only 1.55 points come from the W8 recalibration.  Report all three
  references in the paper.
"""
import numpy as np

__all__ = [
    "fcls", "vca", "rmse", "lfe_of", "lfe_gate", "neighbor_interp",
    "weighted_masked_nmf", "spectral_angle_credibility",
    "trust_region_consensus", "w8_unmix",
    "baseline_unmix", "s2_unmix", "s2e_core_unmix",
    "K_DEFAULT", "ETA", "GAMMA", "KEEP",
]

# ---------------------------------------------------------------------------
# Hyper-parameters (ALL fixed a priori; identical for every dataset/condition)
# ---------------------------------------------------------------------------
K_DEFAULT = 10      # explicit early-stop budget of the weighted MU
ETA       = 0.5     # trust-region shrinkage
GAMMA     = 8.0     # sigmoid sharpness of the LFE credibility gate
KEEP      = 0.15    # fraction of FFT bins counted as "low frequency"
SAM_SCALE = 4.0     # exp(-4 * SAM) angular credibility
SAM_FLOOR = 0.2     # SAM credibility floor (cannot zero out the LFE prior)
W3_MIX    = 0.7     # convex mix: 0.7 * w_LFE + 0.3 * w_SAM
EPS       = 1e-9


# ---------------------------------------------------------------------------
# Basic unmixing utilities
# ---------------------------------------------------------------------------
def fcls(E, Y, it=30):
    """Heuristic clip-and-renormalise routine (paper notation FCLS_h).

    NOTE: despite the historical name this is NOT an exact ASC-constrained
    least-squares solver.  It starts from pinv(E) @ Y and applies `it`
    rounds of nonnegativity clipping plus column-wise sum-to-one
    renormalisation.  The exact KKT support-enumeration solution is
    implemented separately in theory_code/round11_exactfcls.py and is
    denoted FCLS* in the paper.  Houston's pseudo-reference is generated
    by this routine, which is why it reproduces it with RMSE exactly 0.
    """
    A = np.linalg.pinv(E) @ Y
    for _ in range(it):
        A = np.clip(A, 0, None)
        s = A.sum(0, keepdims=True)
        s[s == 0] = 1
        A = A / s
    return A


def vca(Y, R, seed=0):
    """Vertex Component Analysis (used only to build the Houston reference)."""
    rng = np.random.default_rng(seed)
    L, N = Y.shape
    ym = Y.mean(1, keepdims=True)
    Yo = Y - ym
    Ud = np.linalg.svd(Yo @ Yo.T / N)[0][:, :R]
    x = Ud.T @ Y
    snr = 10 * np.log10(np.linalg.norm(x) ** 2 /
                        (np.linalg.norm(Yo) ** 2 - np.linalg.norm(x) ** 2 + 1e-12))
    y = (np.vstack([x, np.ones((1, N))]) if snr >= 15
         else np.vstack([x, np.amax(np.sum(x ** 2, 0)) ** 0.5 * np.ones((1, N))]))
    D = R + 1
    I = np.eye(D)
    Ab = np.zeros((D, 0))
    Me = np.zeros((L, R))
    for i in range(R):
        w = rng.standard_normal((D, 1))
        w /= np.linalg.norm(w)
        for _ in range(100):
            p = w.T @ y
            w = y @ p.T / (p @ p.T + 1e-12)
            w /= np.linalg.norm(w)
        j = int(np.argmax(np.abs(w.T @ y)))
        Me[:, i] = Y[:, j]
        Ab = np.hstack([Ab, y[:, j][:, None]])
        y = (I - Ab @ np.linalg.pinv(Ab)) @ y
    return Me


def rmse(A, B):
    """Root mean squared error between two abundance maps (P x N)."""
    return float(np.sqrt(((A - B) ** 2).mean()))


# ---------------------------------------------------------------------------
# Fourier / spectral credibility
# ---------------------------------------------------------------------------
def lfe_of(Y, keep=KEEP):
    """Low-Frequency-Energy fraction of each spectrum (L x N) -> (N,)."""
    Fn = np.fft.fft(Y, axis=0) / np.sqrt(Y.shape[0])
    pw = np.abs(Fn) ** 2
    L = Y.shape[0]
    order = np.argsort(np.abs(np.fft.fftfreq(L)))
    low = order[:max(1, int(round(keep * L)))]
    return pw[low].sum(0) / (pw.sum(0) + 1e-12)


def lfe_gate(lfe, obs, gain=GAMMA, tau=None):
    """Sigmoid credibility gate; dead/unobserved pixels get w_n = 0."""
    if tau is None:
        idx = np.where(obs > 0.5)[0]
        tau = float(np.median(lfe[idx])) if idx.size else 0.5
    w = 1.0 / (1.0 + np.exp(-gain * (lfe - tau)))
    w[obs < 0.5] = 0.0
    return w


def neighbor_interp(Y, dead, L, H, W):
    """Fill whole-pixel gaps by averaging observed 4-neighbours (init only)."""
    grid = Y.reshape(L, H, W).copy()
    out = Y.copy()
    for n in np.where(dead)[0]:
        r, c = divmod(n, W)
        nb = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            rr, cc = r + dr, c + dc
            if 0 <= rr < H and 0 <= cc < W and not dead[rr * W + cc]:
                nb.append(grid[:, rr, cc])
        if nb:
            out[:, n] = np.mean(nb, 0)
    return out


# ---------------------------------------------------------------------------
# Core solver: T_K(Y, w)
# ---------------------------------------------------------------------------
def weighted_masked_nmf(Yc, obs, E0, A0, dead, iters=K_DEFAULT,
                        dw=None, fix_E=False):
    """
    K-step credibility-weighted masked NMF, i.e. the operator T_K(Y, w).

    See module docstring (b): `dw` weights the DENOMINATOR only.  This is a
    deliberately early-stopped biased iteration, NOT a convergent MU of J.
    Dead pixels are held at their interpolated initial abundance A0[:, dead].

    Parameters
    ----------
    Yc    : (L, N) corrupted cube
    obs   : (N,) 1 = observed, 0 = dead
    E0    : (L, P) initial endmembers (given; learned when fix_E=False)
    A0    : (P, N) initial abundances (already interpolated for dead pixels)
    dead  : (N,) bool mask
    iters : iteration budget K (the effective regularisation strength)
    dw    : (N,) per-pixel spectral credibility, or None for the plain baseline
    fix_E : set False in every experiment of this study

    Returns (E, A).
    """
    E = E0.copy()
    A = A0.copy()
    Ainit_dead = A0[:, dead].copy()
    for _ in range(iters):
        Yd = Yc * obs[None, :]
        wgt = (obs if dw is None else obs * dw)[None, :]
        if not fix_E:
            numE = Yd @ A.T
            denE = E @ (A * wgt) @ A.T + EPS
            E = np.clip(E * numE / denE, 1e-4, None)
        numA = E.T @ Yd
        denA = (E.T @ E) @ (A * wgt) + EPS
        A = A * numA / denA
        A = np.clip(A, 1e-4, None)
        A = A / A.sum(0, keepdims=True)          # ASC projection
        A[:, dead] = Ainit_dead                  # missing spectra stay at init
    return E, A


# ---------------------------------------------------------------------------
# W8 components
# ---------------------------------------------------------------------------
def spectral_angle_credibility(Yn, obs, Yr,
                               scale=SAM_SCALE, floor=SAM_FLOOR):
    """
    Scale-invariant angular credibility of the core reconstruction:

        SAM_n = arccos( <y_n, yhat_n> / (||y_n|| ||yhat_n||) ),
        w_SAM,n = clip( exp(-scale * SAM_n), floor, 1 ),

    where Yr = E_c A_c.  Dead pixels get 0.  LFE is an ENERGY (2nd-order)
    statistic; SAM is invariant to a common spectral scale, so the two cues
    are complementary rather than redundant.
    """
    num = (Yn * Yr * obs[None, :]).sum(0)
    den = np.sqrt(((Yn * obs[None, :]) ** 2).sum(0)
                  * ((Yr * obs[None, :]) ** 2).sum(0) + 1e-12)
    sam = np.arccos(np.clip(num / den, -1.0, 1.0))
    w_sam = np.clip(np.exp(-scale * sam), floor, 1.0)
    w_sam[obs < 0.5] = 0.0
    return w_sam


def trust_region_consensus(A_core, A3, A7, dead, A0, eta=ETA):
    """
    Closed-form solution of Phi(A) (module docstring (c)), followed by the
    feasible projections: non-negativity + ASC + dead-pixel reset.

    The `eta` shrinkage is a trust region: with eta=0.5 the worst-case damage
    from a misleading correction is at most half of that correction's error.
    """
    A_alt = 0.5 * (A3 + A7)
    A_alt = np.clip(A_alt, 1e-4, None)
    A_alt = A_alt / A_alt.sum(0, keepdims=True)
    A_alt[:, dead] = A0[:, dead]
    A = (1.0 - eta) * A_core + eta * A_alt
    A = np.clip(A, 1e-4, None)
    A = A / A.sum(0, keepdims=True)
    A[:, dead] = A0[:, dead]
    return A


def w8_unmix(Yn, obs, E0, A0, dead, K=K_DEFAULT, eta=ETA,
             core_EA=None, return_diag=False):
    """
    W8 — full method in one call.

    Parameters
    ----------
    Yn   : (L, N) corrupted cube (dead columns are zeros)
    obs  : (N,) observability, 1 = observed / 0 = dead
    E0   : (L, P) given endmembers
    A0   : (P, N) abundance init from FCLS on the neighbour-interpolated cube
    dead : (N,) bool mask of dead pixels
    K    : early-stop budget (10 in the paper; same for all datasets)
    eta  : trust-region shrinkage (0.5 in the paper)
    core_EA : optional precomputed (E_c, A_c) from `s2e_core_unmix`, to avoid
              recomputing the core when several W variants are compared.

    Returns
    -------
    (E, A)                if return_diag=False
    (E, A, diag)          otherwise; diag contains w0, w3, w_sam, A_core,
                          A3, A7, A_alt for figures / ablations.
    """
    # --- Step 1: spectral credibility prior ---------------------------------
    w0 = lfe_gate(lfe_of(Yn), obs)

    # --- Step 2: core (explicitly early-stopped S2) -------------------------
    if core_EA is None:
        Ec, Ac = weighted_masked_nmf(Yn, obs, E0, A0, dead, K,
                                     dw=w0, fix_E=False)
    else:
        Ec, Ac = core_EA
    Yr = Ec @ Ac

    # --- Step 3: channel 3 — angular-credibility recalibration --------------
    w_sam = spectral_angle_credibility(Yn, obs, Yr)
    w3 = W3_MIX * w0 + (1.0 - W3_MIX) * w_sam
    w3[obs < 0.5] = 0.0
    _, A3 = weighted_masked_nmf(Yn, obs, E0, A0, dead, K,
                                dw=w3, fix_E=False)

    # --- Step 4: channel 7 — regularisation-trajectory bagging --------------
    K_half = max(3, K // 2)
    _, A_half = weighted_masked_nmf(Yn, obs, E0, A0, dead, K_half,
                                    dw=w0, fix_E=False)
    A7 = 0.5 * (Ac + A_half)

    # --- Step 5: trust-region dual-channel consensus ------------------------
    A = trust_region_consensus(Ac, A3, A7, dead, A0, eta)

    if return_diag:
        diag = {"w0": w0, "w_sam": w_sam, "w3": w3,
                "A_core": Ac, "A3": A3, "A7": A7,
                "A_alt": 0.5 * (A3 + A7), "E_core": Ec}
        return Ec, A, diag
    return Ec, A


# ---------------------------------------------------------------------------
# References (all three MUST be reported in the paper)
# ---------------------------------------------------------------------------
def baseline_unmix(Yn, obs, E0, A0, dead, iters=50):
    """Plain masked NMF, no credibility weighting, 50 iterations."""
    return weighted_masked_nmf(Yn, obs, E0, A0, dead, iters,
                               dw=None, fix_E=False)


def s2_unmix(Yn, obs, E0, A0, dead, iters=50):
    """Original S2: LFE gate + 50 iterations (the user's reference baseline)."""
    w = lfe_gate(lfe_of(Yn), obs)
    return weighted_masked_nmf(Yn, obs, E0, A0, dead, iters,
                               dw=w, fix_E=False)


def s2e_core_unmix(Yn, obs, E0, A0, dead, K=K_DEFAULT):
    """S2-E: identical LFE gate, but explicitly early-stopped at K iterations.
    This isolates the early-stopping contribution from W8's recalibration."""
    w = lfe_gate(lfe_of(Yn), obs)
    return weighted_masked_nmf(Yn, obs, E0, A0, dead, K,
                               dw=w, fix_E=False)
