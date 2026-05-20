"""
hmm_regime.py — Baum-Welch Hidden Markov Model for Oil Market Regime Detection
=================================================================================
Implements the complete Baum-Welch EM algorithm as derived in:
  Hasegawa-Johnson, "Lecture 15: Baum-Welch," ECE 417: Multimedia Signal Processing

Applied to WTI crude oil market regime detection:
  Hidden states  q_t  ∈ {BULL, BEAR, VOLATILE, SIDEWAYS}  (N = 4)
  Observations   x_t  = [daily_return%, rolling_vol%, rsi_norm, log_volume_norm]  (D = 4)
  Emission model:  b_i(x) = N(x; μ_i, Σ_i)   ← Gaussian pdf, Lecture §4

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Forward algorithm  [Lecture p.6]:
  α_1(i)   = π_i b_i(x_1)
  α_t(j)   = [Σ_i α_{t-1}(i) a_ij] b_j(x_t)
  p(X|Λ)  = Σ_i α_T(i)

Backward algorithm  [Lecture p.7]:
  β_T(i)   = 1
  β_t(i)   = Σ_j a_ij b_j(x_{t+1}) β_{t+1}(j)

E-step posteriors  [Lecture p.8]:
  γ_t(i)   = α_t(i) β_t(i) / Σ_k α_t(k) β_t(k)
  ξ_t(i,j) = α_t(i) a_ij b_j(x_{t+1}) β_{t+1}(j) / p(X|Λ)

Baum-Welch M-step  [Lecture p.24-27, p.32, p.35]:
  π'_i     = γ_1(i)
  a'_ij    = Σ_t ξ_t(i,j) / Σ_j Σ_t ξ_t(i,j)
  μ'_i     = Σ_t γ_t(i) x_t / Σ_t γ_t(i)
  Σ'_i     = Σ_t γ_t(i)(x_t − μ_i)(x_t − μ_i)^T / Σ_t γ_t(i)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Economic rationale for 4 states in oil markets:
  BULL     — Supply tightening, backwardation, positive returns, moderate vol
  BEAR     — Oversupply, contango, negative returns, moderate vol
  VOLATILE — Crisis regime: OPEC shock / geopolitical event, very high vol
  SIDEWAYS — Range-bound, demand/supply balanced, low vol, low volume

Why HMM beats simple MA crossover for regime detection:
  1. Regime persistence modelled by a_ij — markets don't jump instantly
  2. Soft assignment γ_t(i) gives probability weights for position sizing
  3. Multivariate Gaussian captures correlated feature structure
  4. Parameters optimised from data via EM — no hand-tuned thresholds
  5. Mathematically proven to improve p(X|Λ) at each iteration (EM guarantee)

Sources:
  Hasegawa-Johnson, ECE 417, Lecture 15: Baum-Welch (2021)
  Rabiner, "A Tutorial on Hidden Markov Models" (1989)
  Ang & Timmermann, "Regime Changes and Financial Markets" (2012)
  Hamilton, "A New Approach to the Economic Analysis of Time Series" (1989)
  Gupta & Dhingra, "Stock Market Prediction Using Hidden Markov Models" (IEEE 2012)
    MAP next-bar prediction: Ô_{d+1} = argmax_O P(O₁,...,O_d,O|λ)
    D=3 OHLC fractional features: fracChange=(C-O)/O, fracHigh=(H-O)/O, fracLow=(O-L)/O
    5000-point vectorised grid search: fracChange∈[-0.05,0.05]×50, fracHigh/Low∈[0,0.04]×10
"""

from __future__ import annotations

import datetime
import logging
import math
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

try:
    from scipy.special import logsumexp as _logsumexp
    from scipy.stats import multivariate_normal as _mvn
    _SCIPY = True
except ImportError:
    _SCIPY = False

try:
    import pandas as pd
    _PD = True
except ImportError:
    _PD = False

try:
    import yfinance as yf
    _YF = True
except ImportError:
    _YF = False


# ============================================================================
# 1. ENUMERATIONS & CONTAINERS
# ============================================================================

class OilRegime(str, Enum):
    BULL      = "BULL"       # positive trend, moderate vol — buy dips
    BEAR      = "BEAR"       # negative trend, moderate vol — sell rallies
    VOLATILE  = "VOLATILE"   # crisis / OPEC shock — reduce size, widen stops
    SIDEWAYS  = "SIDEWAYS"   # range-bound — mean reversion trades

N_STATES   = 4
N_FEATURES = 4   # [daily_return%, rolling_vol%, rsi_norm, log_vol_norm]

REGIME_ORDER = [OilRegime.BULL, OilRegime.BEAR, OilRegime.VOLATILE, OilRegime.SIDEWAYS]


@dataclass
class HMMParams:
    """Parameters Λ = {π, A, μ, Σ} for the Gaussian HMM."""
    pi:    np.ndarray   # (N,)   initial state distribution
    A:     np.ndarray   # (N, N) transition matrix — rows sum to 1
    mu:    np.ndarray   # (N, D) Gaussian emission means
    sigma: np.ndarray   # (N, D, D) Gaussian emission covariance matrices


@dataclass
class RegimeResult:
    """Output of get_hmm_regime()."""
    regime:          OilRegime
    probabilities:   Dict[str, float]   # soft posteriors {regime_name: prob}
    log_likelihood:  float
    n_iter:          int
    trained_on_bars: int
    explanation:     str
    # MAP next-bar prediction (Gupta & Dhingra, IEEE 2012)
    map_direction:   str   = "FLAT"
    map_frac_change: float = 0.0
    map_explanation: str   = "MAP prediction unavailable"


# ============================================================================
# 2. BAUM-WELCH GAUSSIAN HMM
# ============================================================================

class OilMarketHMM:
    """
    Gaussian Hidden Markov Model trained via the Baum-Welch (EM) algorithm.

    Observation model: b_i(x) = N(x; μ_i, Σ_i)   [Lecture §4, slide p.29]
    All forward/backward passes run in log-space to prevent numeric underflow.
    """

    def __init__(self,
                 n_states:   int   = N_STATES,
                 n_features: int   = N_FEATURES,
                 max_iter:   int   = 100,
                 tol:        float = 1e-4,
                 reg:        float = 1e-3):
        self.n_states   = n_states
        self.n_features = n_features
        self.max_iter   = max_iter
        self.tol        = tol
        self.reg        = reg   # diagonal covariance regularisation
        self.params: Optional[HMMParams] = None
        self.is_fitted  = False
        self._last_ll   = -np.inf
        self._last_iter = 0
        self._n_obs     = 0
        self._init_params()

    # ── Initialisation ───────────────────────────────────────────────────────

    def _init_params(self) -> None:
        """
        Oil-market-informed parameter initialisation.
        Features: [daily_return%, rolling_vol%, rsi_norm[-1,1], log_vol_norm]
        """
        pi = np.array([0.30, 0.30, 0.20, 0.20])

        # Regime persistence: markets tend to stay in regime for days/weeks
        A = np.array([
            [0.90, 0.05, 0.03, 0.02],  # BULL → BULL, BEAR, VOLATILE, SIDEWAYS
            [0.05, 0.88, 0.05, 0.02],  # BEAR
            [0.05, 0.05, 0.85, 0.05],  # VOLATILE
            [0.03, 0.03, 0.04, 0.90],  # SIDEWAYS
        ])

        # Emission means per state
        # Features: [return%, ann_vol%, rsi_norm, log_volume_norm]
        mu = np.array([
            [ 0.80, 18.0,  0.25,  0.20],   # BULL
            [-0.80, 18.0, -0.25,  0.10],   # BEAR
            [ 0.00, 48.0,  0.00,  0.80],   # VOLATILE (very high vol)
            [ 0.00,  9.0,  0.00, -0.50],   # SIDEWAYS (low vol, quiet)
        ])

        # Diagonal covariance initialisation
        scales = np.array([
            [1.5**2, 7**2, 0.30**2, 0.40**2],   # BULL
            [1.5**2, 7**2, 0.30**2, 0.40**2],   # BEAR
            [4.5**2, 18**2, 0.40**2, 0.50**2],  # VOLATILE
            [0.8**2, 4**2,  0.20**2, 0.35**2],  # SIDEWAYS
        ])
        sigma = np.zeros((self.n_states, self.n_features, self.n_features))
        for i in range(self.n_states):
            sigma[i] = np.diag(scales[i])

        self.params = HMMParams(pi=pi, A=A, mu=mu, sigma=sigma)

    # ── Emission pdf ─────────────────────────────────────────────────────────

    def _log_emission(self, X: np.ndarray) -> np.ndarray:
        """
        log b_i(x_t) for all states i and frames t.
        Returns (T, N) log-probability matrix.
        Gaussian pdf: b_i(x) = N(x; μ_i, Σ_i)   [Lecture p.30]
        """
        T = X.shape[0]
        log_B = np.full((T, self.n_states), -1e10)
        if not _SCIPY:
            # Diagonal Gaussian fallback (no scipy)
            for i in range(self.n_states):
                var = np.diag(self.params.sigma[i]) + self.reg
                diff = X - self.params.mu[i]
                log_B[:, i] = -0.5 * (
                    np.sum(diff**2 / var, axis=1)
                    + np.sum(np.log(var))
                    + self.n_features * math.log(2 * math.pi)
                )
            return log_B
        for i in range(self.n_states):
            cov = self.params.sigma[i] + self.reg * np.eye(self.n_features)
            try:
                rv = _mvn(mean=self.params.mu[i], cov=cov, allow_singular=True)
                log_B[:, i] = rv.logpdf(X)
            except Exception:
                pass
        return log_B

    # ── Forward algorithm (log-space) ────────────────────────────────────────

    def _forward(self, X: np.ndarray,
                 log_B: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Forward pass in log-space.  [Lecture p.6]

        α_1(i) = π_i b_i(x_1)
        α_t(j) = [Σ_i α_{t-1}(i) a_ij] b_j(x_t)   for t ≥ 2
        p(X|Λ) = Σ_i α_T(i)
        """
        T = X.shape[0]
        log_A = np.log(np.clip(self.params.A, 1e-300, 1))
        log_alpha = np.empty((T, self.n_states))

        # Initialise
        log_alpha[0] = np.log(np.clip(self.params.pi, 1e-300, 1)) + log_B[0]

        # Iterate
        for t in range(1, T):
            for j in range(self.n_states):
                log_alpha[t, j] = (
                    _lse(log_alpha[t - 1] + log_A[:, j]) + log_B[t, j]
                )

        log_likelihood = _lse(log_alpha[-1])
        return log_alpha, float(log_likelihood)

    # ── Backward algorithm (log-space) ───────────────────────────────────────

    def _backward(self, X: np.ndarray, log_B: np.ndarray) -> np.ndarray:
        """
        Backward pass in log-space.  [Lecture p.7]

        β_T(i) = 1   → log β_T(i) = 0
        β_t(i) = Σ_j a_ij b_j(x_{t+1}) β_{t+1}(j)
        """
        T = X.shape[0]
        log_A = np.log(np.clip(self.params.A, 1e-300, 1))
        log_beta = np.zeros((T, self.n_states))

        for t in range(T - 2, -1, -1):
            for i in range(self.n_states):
                log_beta[t, i] = _lse(
                    log_A[i, :] + log_B[t + 1, :] + log_beta[t + 1, :]
                )

        return log_beta

    # ── E-step: γ and ξ ──────────────────────────────────────────────────────

    def _e_step(self,
                log_alpha: np.ndarray,
                log_beta:  np.ndarray,
                log_B:     np.ndarray,
                log_ll:    float) -> Tuple[np.ndarray, np.ndarray]:
        """
        E-step: compute state posterior γ and segment posterior ξ.  [Lecture p.8]

        γ_t(i)   = α_t(i) β_t(i) / Σ_k α_t(k) β_t(k)
        ξ_t(i,j) = α_t(i) a_ij b_j(x_{t+1}) β_{t+1}(j) / p(X|Λ)
        """
        T = log_alpha.shape[0]
        log_A = np.log(np.clip(self.params.A, 1e-300, 1))

        # State posterior γ_t(i)
        log_gamma = log_alpha + log_beta
        log_gamma -= _lse(log_gamma, axis=1, keepdims=True)
        gamma = np.exp(log_gamma)

        # Segment posterior ξ_t(i,j)  for t = 0..T-2
        xi = np.zeros((T - 1, self.n_states, self.n_states))
        for t in range(T - 1):
            for i in range(self.n_states):
                for j in range(self.n_states):
                    xi[t, i, j] = math.exp(
                        log_alpha[t, i] + log_A[i, j]
                        + log_B[t + 1, j] + log_beta[t + 1, j]
                        - log_ll
                    )
            # Normalise row to avoid drift
            row_sum = xi[t].sum()
            if row_sum > 0:
                xi[t] /= row_sum

        return gamma, xi

    # ── M-step: update Λ ─────────────────────────────────────────────────────

    def _m_step(self, X: np.ndarray,
                gamma: np.ndarray, xi: np.ndarray) -> None:
        """
        M-step: re-estimate π, A, μ, Σ.  [Lecture p.24-27, p.32, p.35]

        π'_i     = γ_1(i)
        a'_ij    = Σ_t ξ_t(i,j) / Σ_j Σ_t ξ_t(i,j)
        μ'_i     = Σ_t γ_t(i) x_t / Σ_t γ_t(i)
        Σ'_i     = Σ_t γ_t(i)(x_t − μ_i)(x_t − μ_i)^T / Σ_t γ_t(i)
        """
        D = X.shape[1]

        # π
        pi_new = gamma[0] + 1e-10
        pi_new /= pi_new.sum()

        # A — transition matrix
        A_num = xi.sum(axis=0) + 1e-10            # (N, N)
        A_new = A_num / A_num.sum(axis=1, keepdims=True)

        # μ and Σ — Gaussian emission parameters
        mu_new    = np.zeros((self.n_states, D))
        sigma_new = np.zeros((self.n_states, D, D))

        for i in range(self.n_states):
            g_i = gamma[:, i]                      # (T,)
            g_sum = g_i.sum() + 1e-10

            # μ'_i  [Lecture p.32]
            mu_new[i] = (g_i[:, None] * X).sum(axis=0) / g_sum

            # Σ'_i  [Lecture p.35]
            diff = X - mu_new[i]                   # (T, D)
            sigma_new[i] = (g_i[:, None] * diff).T @ diff / g_sum
            sigma_new[i] += self.reg * np.eye(D)  # regularisation

        self.params = HMMParams(pi=pi_new, A=A_new,
                                mu=mu_new, sigma=sigma_new)

    # ── Training ─────────────────────────────────────────────────────────────

    def fit(self, X: np.ndarray) -> "OilMarketHMM":
        """
        Train HMM via Baum-Welch EM until convergence.

        Guaranteed to non-decrease p(X|Λ) at every iteration (EM property).
        Stops when |ΔlogL| < tol or max_iter reached.
        """
        if X.ndim == 1:
            X = X[:, None]
        if X.shape[1] != self.n_features:
            raise ValueError(
                f"Expected {self.n_features} features, got {X.shape[1]}"
            )

        prev_ll = -np.inf
        for it in range(self.max_iter):
            log_B     = self._log_emission(X)
            log_alpha, log_ll = self._forward(X, log_B)
            log_beta  = self._backward(X, log_B)
            gamma, xi = self._e_step(log_alpha, log_beta, log_B, log_ll)
            self._m_step(X, gamma, xi)

            if abs(log_ll - prev_ll) < self.tol:
                logger.debug("Baum-Welch converged at iter %d  logL=%.3f",
                             it + 1, log_ll)
                break
            prev_ll = log_ll

        self.is_fitted   = True
        self._last_ll    = log_ll
        self._last_iter  = it + 1
        self._n_obs      = len(X)
        return self

    # ── Inference ────────────────────────────────────────────────────────────

    def state_posteriors(self, X: np.ndarray) -> np.ndarray:
        """
        Compute γ_t(i) = p(q_t=i | X, Λ) for every frame.
        Returns (T, N) array.  Used for soft position sizing.
        """
        log_B     = self._log_emission(X)
        log_alpha, log_ll = self._forward(X, log_B)
        log_beta  = self._backward(X, log_B)
        gamma, _  = self._e_step(log_alpha, log_beta, log_B, log_ll)
        return gamma

    def current_regime(self, X: np.ndarray) -> Tuple[OilRegime, np.ndarray]:
        """
        Identify the most likely regime at the LAST observation.
        Returns (regime_label, state_probability_vector).
        """
        gamma = self.state_posteriors(X)
        g_last = gamma[-1]                         # γ_T(i) for all states
        best   = int(np.argmax(g_last))
        label  = self._label_state(best)
        return label, g_last

    def _label_state(self, state_idx: int) -> OilRegime:
        """
        Map a learned state index → economic regime label.
        Heuristic: highest-vol state = VOLATILE; then rank by mean return.
        """
        mu  = self.params.mu                       # (N, D)
        vol = mu[:, 1]                             # annualised vol feature

        vol_rank   = np.argsort(-vol)              # descending
        vol_thresh = np.percentile(vol, 75)

        labels: List[OilRegime] = [OilRegime.SIDEWAYS] * self.n_states
        for i in range(self.n_states):
            if vol[i] >= vol_thresh:
                labels[i] = OilRegime.VOLATILE
            elif mu[i, 0] > 0.20:
                labels[i] = OilRegime.BULL
            elif mu[i, 0] < -0.20:
                labels[i] = OilRegime.BEAR
            else:
                labels[i] = OilRegime.SIDEWAYS

        return labels[state_idx]


# ============================================================================
# 3. FEATURE ENGINEERING (observation vector builder)
# ============================================================================

def build_hmm_features(close: "pd.Series",
                       volume: Optional["pd.Series"] = None) -> np.ndarray:
    """
    Build the D=4 observation matrix X from a price series.

    Features:
      x[0] = daily log-return   × 100  (%)
      x[1] = 10-day rolling annualised vol (%)
      x[2] = RSI(14) normalised to [−1, +1]
      x[3] = log(volume) z-scored    (or 0.0 if volume unavailable)

    Rows < 20 are dropped (warm-up period).
    """
    if not _PD:
        raise ImportError("pandas required for build_hmm_features")

    close = pd.Series(close).astype(float).dropna()

    # Feature 0: daily log-return (%)
    ret = np.log(close / close.shift(1)) * 100

    # Feature 1: 10-day rolling annualised vol (%)
    rv = ret.rolling(10).std() * math.sqrt(252)

    # Feature 2: RSI(14) → [−1, +1]
    delta = close.diff()
    up    = delta.clip(lower=0).rolling(14).mean()
    dn    = (-delta.clip(upper=0)).rolling(14).mean()
    rsi   = 100 - 100 / (1 + up / (dn + 1e-9))
    rsi_n = (rsi - 50) / 50                       # normalised

    # Feature 3: log-volume z-score
    if volume is not None and len(volume) == len(close):
        lv = np.log(volume.clip(lower=1).astype(float))
        lv_n = (lv - lv.mean()) / (lv.std() + 1e-9)
    else:
        lv_n = pd.Series(0.0, index=close.index)

    df = pd.DataFrame({
        "ret":  ret,
        "rv":   rv,
        "rsi_n": rsi_n,
        "lv_n": lv_n,
    }).dropna()

    return df.values.astype(float)


def build_ohlc_features(open_s:  "pd.Series",
                        high_s:  "pd.Series",
                        low_s:   "pd.Series",
                        close_s: "pd.Series") -> np.ndarray:
    """
    Build the D=3 OHLC fractional observation matrix for MAP-HMM.

    Gupta & Dhingra (IEEE 2012) equation (3):
      fracChange = (Close − Open) / Open
      fracHigh   = (High  − Open) / Open
      fracLow    = (Open  − Low)  / Open

    All three features are non-negative by construction for fracHigh/fracLow
    and centred near zero for fracChange, making them well-suited to a
    diagonal Gaussian emission model.
    """
    if not _PD:
        raise ImportError("pandas required for build_ohlc_features")

    denom = pd.Series(open_s).astype(float).clip(lower=1e-9)
    frac_change = (pd.Series(close_s).astype(float) - denom) / denom
    frac_high   = (pd.Series(high_s).astype(float)  - denom) / denom
    frac_low    = (denom - pd.Series(low_s).astype(float))   / denom

    df = pd.DataFrame({
        "frac_change": frac_change,
        "frac_high":   frac_high,
        "frac_low":    frac_low,
    }).dropna()

    return df.values.astype(float)


# ============================================================================
# 4. MODULE-LEVEL MODEL CACHE  (avoid retraining every call)
# ============================================================================

_hmm_model:      Optional[OilMarketHMM] = None
_hmm_bar_count:  int = 0
_RETRAIN_EVERY   = 63   # retrain every quarter (~63 trading days)

# OHLC MAP-HMM cache (Gupta & Dhingra 2012) — separate D=3 model
_ohlc_hmm_model:     Optional[OilMarketHMM] = None
_ohlc_hmm_bar_count: int = 0


def get_hmm_regime(ticker:   str = "CL=F",
                   close:    Optional["pd.Series"] = None,
                   volume:   Optional["pd.Series"] = None,
                   retrain:  bool = False) -> RegimeResult:
    """
    High-level API: return current WTI market regime via Baum-Welch HMM
    plus MAP next-bar direction prediction (Gupta & Dhingra, IEEE 2012).

    Caches the trained model and retrains only when the observation count
    has grown by _RETRAIN_EVERY bars since last training.

    Args:
        ticker:  yfinance ticker (default CL=F = WTI continuous front-month)
        close:   pre-fetched close price Series (skips yfinance if provided)
        volume:  pre-fetched volume Series (optional)
        retrain: force full retraining even if cache is warm

    Returns:
        RegimeResult with regime label, soft probabilities, log-likelihood,
        MAP next-bar direction, and plain-English explanation strings.
    """
    global _hmm_model, _hmm_bar_count

    # 1. Fetch price data
    if close is None:
        close, volume = _fetch_wti_data(ticker)

    if not _PD or close is None or len(close) < 30:
        return _fallback_regime()

    close = pd.Series(close).astype(float).dropna()

    # 2. Build observation matrix
    try:
        X = build_hmm_features(close, volume)
    except Exception as e:
        logger.warning("[HMM] Feature build failed: %s — using fallback", e)
        return _fallback_regime()

    if len(X) < 20:
        return _fallback_regime()

    # 3. Train or reuse cached regime HMM
    needs_train = (
        _hmm_model is None
        or retrain
        or (len(X) - _hmm_bar_count) >= _RETRAIN_EVERY
    )

    if needs_train:
        logger.info("[HMM] Training Baum-Welch HMM on %d observations ...", len(X))
        model = OilMarketHMM()
        try:
            model.fit(X)
            _hmm_model     = model
            _hmm_bar_count = len(X)
        except Exception as e:
            logger.warning("[HMM] Training failed: %s — using fallback", e)
            return _fallback_regime()
    else:
        model = _hmm_model

    # 4. Infer current regime
    try:
        regime, gamma_last = model.current_regime(X)
    except Exception as e:
        logger.warning("[HMM] Inference failed: %s", e)
        return _fallback_regime()

    # 5. Build soft probability dict
    probs: Dict[str, float] = {}
    for idx, reg in enumerate(REGIME_ORDER):
        state_label = model._label_state(idx)
        probs[state_label.value] = probs.get(state_label.value, 0.0) + float(gamma_last[idx])
    for reg in OilRegime:
        probs.setdefault(reg.value, 0.0)

    # 6. Build regime explanation
    dominant_prob = probs[regime.value]
    regime_descs = {
        OilRegime.BULL:     "supply tightening, backwardation — buy pullbacks",
        OilRegime.BEAR:     "oversupply, contango — sell rallies",
        OilRegime.VOLATILE: "crisis / OPEC shock — reduce size, widen stops",
        OilRegime.SIDEWAYS: "range-bound — mean reversion, tight stops",
    }
    explanation = (
        f"[HMM] Regime: {regime.value} ({dominant_prob*100:.1f}% confidence) — "
        f"{regime_descs.get(regime, '')}\n"
        f"  Soft posteriors: "
        + " | ".join(f"{k}={v*100:.1f}%" for k, v in sorted(probs.items()))
        + f"\n  Trained on {model._n_obs} bars, {model._last_iter} EM iters, "
        f"logL={model._last_ll:.2f}"
    )

    # 7. MAP next-bar prediction (Gupta & Dhingra, IEEE 2012)
    map_direction, map_frac_change, map_expl = _run_map_prediction(ticker, retrain)

    return RegimeResult(
        regime          = regime,
        probabilities   = probs,
        log_likelihood  = model._last_ll,
        n_iter          = model._last_iter,
        trained_on_bars = model._n_obs,
        explanation     = explanation,
        map_direction   = map_direction,
        map_frac_change = map_frac_change,
        map_explanation = map_expl,
    )


def _run_map_prediction(ticker: str, retrain: bool) -> Tuple[str, float, str]:
    """
    Run the OHLC MAP-HMM pipeline.  Returns (direction, frac_change, explanation).
    Isolated so get_hmm_regime() stays readable.
    """
    global _ohlc_hmm_model, _ohlc_hmm_bar_count
    try:
        ohlc_df = _fetch_wti_ohlc(ticker)
        if ohlc_df is None or len(ohlc_df) < 30:
            return "FLAT", 0.0, "[MAP] Insufficient OHLC data."
        X_ohlc = build_ohlc_features(
            ohlc_df["open"], ohlc_df["high"], ohlc_df["low"], ohlc_df["close"]
        )
        if len(X_ohlc) < 20:
            return "FLAT", 0.0, "[MAP] Too few OHLC observations."
        needs_ohlc_train = (
            _ohlc_hmm_model is None
            or retrain
            or (len(X_ohlc) - _ohlc_hmm_bar_count) >= _RETRAIN_EVERY
        )
        if needs_ohlc_train:
            logger.info("[MAP-HMM] Training OHLC HMM on %d bars ...", len(X_ohlc))
            ohlc_model = _make_ohlc_hmm()
            ohlc_model.fit(X_ohlc)
            _ohlc_hmm_model     = ohlc_model
            _ohlc_hmm_bar_count = len(X_ohlc)
        else:
            ohlc_model = _ohlc_hmm_model
        best_fc, direction, log_prob = map_predict_next_bar(ohlc_model, X_ohlc)
        expl = (
            f"[MAP-HMM] Next-bar prediction: {direction} "
            f"(fracChange={best_fc:+.4f}, logP={log_prob:.2f}) "
            f"— Gupta & Dhingra (IEEE 2012)"
        )
        return direction, best_fc, expl
    except Exception as e:
        logger.warning("[MAP-HMM] Prediction failed: %s", e)
        return "FLAT", 0.0, f"[MAP] Error: {e}"


# ============================================================================
# 5. HELPERS
# ============================================================================

def _fetch_wti_data(ticker: str) -> Tuple[Optional["pd.Series"],
                                          Optional["pd.Series"]]:
    """Fetch close and volume from yfinance."""
    if not _YF or not _PD:
        return None, None
    try:
        df = yf.download(ticker, period="3y", interval="1d",
                         progress=False, auto_adjust=True)
        if df is None or len(df) < 30:
            return None, None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0].lower() for c in df.columns]
        else:
            df.columns = [c.lower() for c in df.columns]
        close  = df["close"].dropna()
        volume = df.get("volume")
        if volume is not None:
            volume = volume.dropna()
        return close, volume
    except Exception as e:
        logger.warning("[HMM] yfinance fetch failed: %s", e)
        return None, None


def _lse(a: np.ndarray, axis=None, keepdims=False) -> np.ndarray:
    """log-sum-exp with scipy or pure-numpy fallback."""
    if _SCIPY:
        return _logsumexp(a, axis=axis, keepdims=keepdims)
    # Pure-numpy fallback
    if axis is None:
        c = a.max()
        return c + math.log(np.sum(np.exp(a - c)))
    c = a.max(axis=axis, keepdims=True)
    result = c + np.log(np.sum(np.exp(a - c), axis=axis, keepdims=keepdims))
    if not keepdims and axis is not None:
        result = np.squeeze(result, axis=axis)
    return result


def _make_ohlc_hmm() -> "OilMarketHMM":
    """
    Factory for the D=3 OHLC MAP-HMM (Gupta & Dhingra, IEEE 2012).
    Prior means calibrated to WTI 1-day OHLC fractional move distributions.
    """
    hmm = OilMarketHMM(n_states=4, n_features=3)
    # Features: [fracChange, fracHigh, fracLow]
    hmm.params.mu = np.array([
        [ 0.008, 0.015, 0.008],   # BULL: positive close, moderate high, small low
        [-0.008, 0.007, 0.016],   # BEAR: negative close, small high, extended low
        [ 0.001, 0.025, 0.025],   # VOLATILE: flat close, large high+low excursions
        [ 0.001, 0.006, 0.006],   # SIDEWAYS: flat close, tight range
    ])
    scales = np.array([
        [0.006**2, 0.012**2, 0.010**2],
        [0.006**2, 0.010**2, 0.012**2],
        [0.015**2, 0.020**2, 0.020**2],
        [0.003**2, 0.005**2, 0.005**2],
    ])
    sigma = np.zeros((4, 3, 3))
    for i in range(4):
        sigma[i] = np.diag(scales[i])
    hmm.params.sigma = sigma
    return hmm


def _fetch_wti_ohlc(ticker: str) -> Optional["pd.DataFrame"]:
    """Fetch 3-year daily OHLC from yfinance for the MAP-HMM."""
    if not _YF or not _PD:
        return None
    try:
        df = yf.download(ticker, period="3y", interval="1d",
                         progress=False, auto_adjust=True)
        if df is None or len(df) < 30:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0].lower() for c in df.columns]
        else:
            df.columns = [c.lower() for c in df.columns]
        needed = {"open", "high", "low", "close"}
        if not needed.issubset(set(df.columns)):
            return None
        return df[["open", "high", "low", "close"]].dropna()
    except Exception as e:
        logger.warning("[MAP-HMM] OHLC fetch failed: %s", e)
        return None


def map_predict_next_bar(model: "OilMarketHMM",
                         X_ohlc: np.ndarray,
                         latency: int = 10) -> Tuple[float, str, float]:
    """
    MAP next-bar OHLC prediction (Gupta & Dhingra, IEEE 2012).

    Ô_{d+1} = argmax_O  P(O₁, O₂, ..., O_d, O | λ)

    Computed as:
      log P(O₁...O_d, O_{d+1}|λ) = lse_j[ log_trans[j] + log b_j(O_{d+1}) ]
    where:
      log_trans[j] = lse_i[ log α_d(i) + log A(i,j) ]   (precomputed once)

    Grid (Table II, paper): 50 × 10 × 10 = 5 000 candidates
      fracChange ∈ [−0.05, +0.05]  (50 steps)
      fracHigh   ∈ [  0.0,  0.04]  (10 steps)
      fracLow    ∈ [  0.0,  0.04]  (10 steps)

    Direction threshold: |fracChange| > 0.002 → UP / DOWN, else FLAT.

    Returns:
        (best_frac_change, direction_str, best_log_prob)
    """
    _DIRECTION_THRESHOLD = 0.002

    window = X_ohlc[-latency:] if len(X_ohlc) >= latency else X_ohlc

    log_A    = np.log(np.clip(model.params.A, 1e-300, 1))
    log_B_w  = model._log_emission(window)
    log_alpha, _ = model._forward(window, log_B_w)
    log_alpha_d  = log_alpha[-1]   # (N,) — forward variable at last observed bar

    # Precompute log P(next state = j | X_1..d) for each state j
    N = model.n_states
    log_trans = np.array([_lse(log_alpha_d + log_A[:, j]) for j in range(N)])

    # 5 000-point candidate grid (vectorised — no Python loops over candidates)
    fc_grid = np.linspace(-0.05,  0.05, 50)
    fh_grid = np.linspace( 0.0,   0.04, 10)
    fl_grid = np.linspace( 0.0,   0.04, 10)
    fc_all, fh_all, fl_all = np.meshgrid(fc_grid, fh_grid, fl_grid, indexing="ij")
    candidates = np.column_stack([
        fc_all.ravel(), fh_all.ravel(), fl_all.ravel()
    ])   # (5000, 3)

    log_B_cand  = model._log_emission(candidates)          # (5000, N)
    log_combined = log_trans[None, :] + log_B_cand         # (5000, N)
    if _SCIPY:
        log_probs = _logsumexp(log_combined, axis=1)       # (5000,)
    else:
        c = log_combined.max(axis=1, keepdims=True)
        log_probs = (c[:, 0]
                     + np.log(np.exp(log_combined - c).sum(axis=1)))

    best_idx        = int(np.argmax(log_probs))
    best_frac_change = float(candidates[best_idx, 0])
    best_log_prob   = float(log_probs[best_idx])

    if best_frac_change > _DIRECTION_THRESHOLD:
        direction = "UP"
    elif best_frac_change < -_DIRECTION_THRESHOLD:
        direction = "DOWN"
    else:
        direction = "FLAT"

    return best_frac_change, direction, best_log_prob


def _fallback_regime() -> RegimeResult:
    """Return a neutral fallback when HMM cannot be computed."""
    probs = {r.value: 0.25 for r in OilRegime}
    return RegimeResult(
        regime          = OilRegime.SIDEWAYS,
        probabilities   = probs,
        log_likelihood  = 0.0,
        n_iter          = 0,
        trained_on_bars = 0,
        explanation     = "[HMM] Insufficient data — defaulting to SIDEWAYS regime.",
    )


# ============================================================================
# 6. POSITION SIZING HELPER (Baum-Welch soft posteriors → size multiplier)
# ============================================================================

def regime_size_multiplier(result: RegimeResult) -> float:
    """
    Convert HMM soft posteriors γ_t(i) into a position-size multiplier (0–1).

    Multiplier table (CLAUDE.md / Hull Ch.16):
      BULL / BEAR  → 0.50 + γ_{regime} (confidence bonus), capped at 1.0
      VOLATILE     → 0.25  (75% size reduction — OPEC shock / crisis)
      SIDEWAYS     → 0.50  (half size — no directional edge)

    Returns:
        float in [0.25, 1.0] — multiply raw position size by this value.
    """
    regime = result.regime
    p      = result.probabilities.get(regime.value, 0.25)

    if regime == OilRegime.VOLATILE:
        mult = 0.25   # 75% reduction per CLAUDE.md
    elif regime in (OilRegime.BULL, OilRegime.BEAR):
        mult = min(1.0, 0.50 + p)   # full Kelly at high confidence
    else:
        mult = 0.50   # SIDEWAYS = half size

    return mult


# ============================================================================
# 7. STANDALONE DEMO
# ============================================================================

if __name__ == "__main__":
    import sys, io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

    print("=" * 70)
    print("  OIL MARKET HMM — Baum-Welch Regime Detector + MAP Predictor")
    print("  ECE 417 / Gupta & Dhingra (IEEE 2012) → WTI Application")
    print("=" * 70)

    result = get_hmm_regime("CL=F")
    print(result.explanation)
    print()
    print(result.map_explanation)
    print()

    mult = regime_size_multiplier(result)
    p    = result.probabilities.get(result.regime.value, 0)
    print(
        f"Regime:        {result.regime.value}  ({p*100:.1f}% confidence)\n"
        f"MAP direction: {result.map_direction}  "
        f"(fracChange={result.map_frac_change:+.4f})\n"
        f"Size mult:     {mult:.2f}x"
    )
