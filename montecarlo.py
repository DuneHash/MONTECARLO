import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import warnings

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# Pure-NumPy Gaussian HMM  (Baum-Welch EM + Viterbi)
# Χωρίς hmmlearn — τρέχει παντού χωρίς compiler
# ══════════════════════════════════════════════════════════════════════════════

class GaussianHMM:
    """
    Hidden Markov Model με Gaussian emissions.
    Εκπαίδευση: Baum-Welch (EM).
    Αποκωδικοποίηση: Viterbi.
    """

    def __init__(self, n_states: int = 3, n_iter: int = 100, tol: float = 1e-4,
                 random_state: int = 0):
        self.n_states = n_states
        self.n_iter   = n_iter
        self.tol      = tol
        self.rng      = np.random.default_rng(random_state)

    def _init_params(self, X: np.ndarray):
        N, D = X.shape
        K    = self.n_states
        self.pi_ = np.ones(K) / K
        self.A_  = np.eye(K) * 0.6 + np.ones((K, K)) * 0.4 / K
        self.A_ /= self.A_.sum(axis=1, keepdims=True)
        idx          = self.rng.choice(N, K, replace=False)
        self.means_  = X[idx].copy()
        self.covs_   = np.array([np.eye(D) * X.var(axis=0).mean() for _ in range(K)])

    def _log_emission(self, X: np.ndarray) -> np.ndarray:
        T, D  = X.shape
        K     = self.n_states
        log_b = np.zeros((T, K))
        for k in range(K):
            diff           = X - self.means_[k]
            cov            = self.covs_[k] + np.eye(D) * 1e-6
            sign, logdet   = np.linalg.slogdet(cov)
            if sign <= 0:
                log_b[:, k] = -1e10
                continue
            inv_cov        = np.linalg.inv(cov)
            mahal          = np.einsum("td,dd,td->t", diff, inv_cov, diff)
            log_b[:, k]   = -0.5 * (D * np.log(2 * np.pi) + logdet + mahal)
        return log_b

    def _forward(self, log_b: np.ndarray):
        T, K  = log_b.shape
        log_A = np.log(self.A_ + 1e-300)
        alpha = np.full((T, K), -np.inf)
        alpha[0] = np.log(self.pi_ + 1e-300) + log_b[0]
        for t in range(1, T):
            for k in range(K):
                alpha[t, k] = np.logaddexp.reduce(alpha[t-1] + log_A[:, k]) + log_b[t, k]
        return alpha

    def _backward(self, log_b: np.ndarray):
        T, K  = log_b.shape
        log_A = np.log(self.A_ + 1e-300)
        beta  = np.full((T, K), -np.inf)
        beta[-1] = 0.0
        for t in range(T - 2, -1, -1):
            for k in range(K):
                beta[t, k] = np.logaddexp.reduce(log_A[k] + log_b[t+1] + beta[t+1])
        return beta

    def fit(self, X: np.ndarray):
        self._init_params(X)
        T, D  = X.shape
        K     = self.n_states
        prev_ll = -np.inf

        for _ in range(self.n_iter):
            log_b = self._log_emission(X)
            alpha = self._forward(log_b)
            beta  = self._backward(log_b)
            ll    = np.logaddexp.reduce(alpha[-1])
            if abs(ll - prev_ll) < self.tol:
                break
            prev_ll = ll

            log_gamma = alpha + beta
            log_gamma -= np.logaddexp.reduce(log_gamma, axis=1, keepdims=True)
            gamma = np.exp(log_gamma)

            log_A = np.log(self.A_ + 1e-300)
            xi    = np.zeros((T - 1, K, K))
            for t in range(T - 1):
                for i in range(K):
                    for j in range(K):
                        xi[t, i, j] = (alpha[t, i] + log_A[i, j]
                                       + log_b[t+1, j] + beta[t+1, j])
                xi[t] -= np.logaddexp.reduce(xi[t].ravel())
                xi[t]  = np.exp(xi[t])

            self.pi_ = gamma[0] + 1e-10
            self.pi_ /= self.pi_.sum()
            self.A_  = xi.sum(axis=0) + 1e-10
            self.A_ /= self.A_.sum(axis=1, keepdims=True)

            for k in range(K):
                g_k            = gamma[:, k]
                g_sum          = g_k.sum() + 1e-10
                self.means_[k] = (g_k[:, None] * X).sum(axis=0) / g_sum
                diff           = X - self.means_[k]
                self.covs_[k]  = (g_k[:, None, None]
                                  * np.einsum("td,te->tde", diff, diff)).sum(axis=0) / g_sum

        self.log_likelihood_ = prev_ll
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        T, K  = X.shape[0], self.n_states
        log_b = self._log_emission(X)
        log_A = np.log(self.A_ + 1e-300)
        delta = np.full((T, K), -np.inf)
        psi   = np.zeros((T, K), dtype=int)
        delta[0] = np.log(self.pi_ + 1e-300) + log_b[0]
        for t in range(1, T):
            for k in range(K):
                scores      = delta[t-1] + log_A[:, k]
                psi[t, k]   = np.argmax(scores)
                delta[t, k] = scores[psi[t, k]] + log_b[t, k]
        states     = np.zeros(T, dtype=int)
        states[-1] = np.argmax(delta[-1])
        for t in range(T - 2, -1, -1):
            states[t] = psi[t + 1, states[t + 1]]
        return states

    def score(self, X: np.ndarray) -> float:
        log_b = self._log_emission(X)
        alpha = self._forward(log_b)
        return float(np.logaddexp.reduce(alpha[-1]))


# ══════════════════════════════════════════════════════════════════════════════
# Streamlit App
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(page_title="Quant Simulator", page_icon="📊", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Rajdhani', sans-serif;
    background-color: #080812;
    color: #e0e0ff;
}
.stApp { background-color: #080812; }
h1, h2, h3 { font-family: 'Share Tech Mono', monospace; }

section[data-testid="stSidebar"] {
    background-color: #0d0d1f;
    border-right: 1px solid #1e1e44;
}

.stTabs [data-baseweb="tab-list"] {
    background-color: #0d0d1f;
    border-bottom: 1px solid #1e1e44;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.9rem;
    color: #6688bb;
    background-color: transparent;
    border: 1px solid #1e1e44;
    border-radius: 6px 6px 0 0;
    padding: 8px 20px;
    letter-spacing: 0.05em;
}
.stTabs [aria-selected="true"] {
    color: #00d4ff !important;
    border-color: #00d4ff !important;
    background-color: #0d0d2b !important;
}

[data-testid="stMetric"] {
    background: linear-gradient(135deg, #0d0d2b, #111130);
    border: 1px solid #1e1e55;
    border-radius: 10px;
    padding: 12px 16px;
}
[data-testid="stMetricLabel"] { color: #6688bb !important; font-size: 0.75rem; }
[data-testid="stMetricValue"] {
    font-family: 'Share Tech Mono', monospace !important;
    color: #00d4ff !important;
    font-size: 1.3rem !important;
}
[data-testid="stMetricDelta"] { font-size: 0.85rem !important; }

.stSlider > div > div > div > div { background: #00d4ff !important; }

.stButton > button {
    background: linear-gradient(90deg, #0044ff, #00d4ff);
    color: #fff;
    font-family: 'Share Tech Mono', monospace;
    font-size: 1rem;
    border: none;
    border-radius: 8px;
    padding: 0.6rem 2rem;
    letter-spacing: 0.08em;
    transition: opacity 0.2s;
    width: 100%;
}
.stButton > button:hover { opacity: 0.85; }
hr { border-color: #1e1e44; }
</style>
""", unsafe_allow_html=True)

st.markdown("# 📊 Quant Simulator")
st.markdown("*Monte Carlo GBM  ·  Hidden Markov Model — pure NumPy, no compiler needed*")
st.markdown("—")

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙️ Παράμετροι")

    ticker = st.text_input(
        "Ticker Symbol", value="AAPL",
        help="π.χ. MSFT, GOOGL, ^GSPC, BTC-USD"
    ).upper().strip()

    period = st.selectbox(
        "Ιστορικά δεδομένα", ["3mo", "6mo", "1y", "2y", "5y"], index=3
    )

    st.markdown("---")
    st.markdown("#### 🎲 Monte Carlo")
    days_ahead  = st.slider("Ημέρες πρόβλεψης", 30, 504, 252, step=21,
                             help="252 ≈ 1 χρόνος συναλλαγών")
    simulations = st.select_slider("Αριθμός προσομοιώσεων",
                                   options=[500, 1000, 2000, 5000, 10000], value=1000)

    st.markdown("---")
    st.markdown("#### 🔍 HMM")
    n_states          = st.slider("Αριθμός κρυφών καταστάσεων", 2, 4, 3,
                                   help="2=Bull/Bear  3=Bull/Bear/Sideways  4=λεπτομερές")
    hmm_forecast_days = st.slider("Ημέρες πρόβλεψης HMM", 10, 120, 30, step=5)

    st.markdown("---")
    run = st.button("▶  ΕΚΤΕΛΕΣΗ")

# ── Helpers ───────────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_data(ticker: str, period: str):
    try:
        df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    except Exception as e:
        return None, str(e)
    if df.empty:
        return None, f"Δεν βρέθηκαν δεδομένα για '{ticker}'."
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if "Close" not in df.columns:
        return None, f"Δεν βρέθηκε στήλη 'Close'. Διαθέσιμες: {list(df.columns)}"
    close = df["Close"].dropna()
    if len(close) < 40:
        return None, f"Ανεπαρκή δεδομένα ({len(close)} ημέρες). Επιλέξτε μεγαλύτερο διάστημα."
    return close, None


def label_regimes(means: np.ndarray, n_states: int):
    order = np.argsort(means)
    labels, colors = {}, {}
    if n_states == 2:
        names = ["🐻 Bear", "🐂 Bull"]
        clrs  = ["#ff4e4e", "#4eff91"]
    elif n_states == 3:
        names = ["🐻 Bear", "➡️ Sideways", "🐂 Bull"]
        clrs  = ["#ff4e4e", "#ffcc00", "#4eff91"]
    else:
        names = [f"State {i+1}" for i in range(n_states)]
        clrs  = [plt.cm.RdYlGn(i / (n_states - 1)) for i in range(n_states)]
    for rank, state_idx in enumerate(order):
        labels[state_idx] = names[rank]
        colors[state_idx] = clrs[rank]
    return labels, colors


def standardize(X: np.ndarray):
    mu  = X.mean(axis=0)
    std = X.std(axis=0) + 1e-10
    return (X - mu) / std


# ── TABS ──────────────────────────────────────────────────────────────────────

tab_mc, tab_hmm = st.tabs(["🎲  Monte Carlo  GBM", "🔍  Hidden Markov Model"])

# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — Monte Carlo
# ════════════════════════════════════════════════════════════════════════════════

with tab_mc:
    if not run:
        st.info("👈 Ορίστε τις παραμέτρους στα αριστερά και πατήστε **ΕΚΤΕΛΕΣΗ**.")
    else:
        close, err = load_data(ticker, period)
        if err:
            st.error(f"❌ {err}")
            st.stop()

        S0      = float(close.iloc[-1])
        log_ret = np.log(close / close.shift(1)).dropna()

        if log_ret.std() == 0:
            st.error("❌ Zero volatility — αδύνατος υπολογισμός.")
            st.stop()

        mu    = float(log_ret.mean())
        sigma = float(log_ret.std())

        rng    = np.random.default_rng(42)
        shocks = rng.normal((mu - 0.5 * sigma**2), sigma, size=(days_ahead, simulations))
        paths  = S0 * np.exp(np.cumsum(shocks, axis=0))
        paths  = np.vstack([np.full(simulations, S0), paths])

        final = paths[-1]
        p5, p25, p50, p75, p95 = np.percentile(final, [5, 25, 50, 75, 95])
        avg         = final.mean()
        prob_profit = (final > S0).mean() * 100
        exp_ret     = (avg / S0 - 1) * 100
        annual_vol  = sigma * np.sqrt(252) * 100

        st.markdown(f"### {ticker} — Monte Carlo ({simulations:,} προσομοιώσεις, {days_ahead} ημέρες)")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Τρέχουσα τιμή",    f"${S0:.2f}")
        c2.metric("Διάμεσος 50%",     f"${p50:.2f}", f"{(p50/S0-1)*100:+.1f}%")
        c3.metric("Μέση τελ. τιμή",   f"${avg:.2f}", f"{exp_ret:+.1f}%")
        c4.metric("Πιθ. κέρδους",     f"{prob_profit:.1f}%")
        c5.metric("Ετήσια Volatility", f"{annual_vol:.1f}%")
        st.markdown("---")

        with st.expander("📋 Πίνακας Percentiles"):
            tbl = pd.DataFrame({
                "Percentile": ["5% (Απαισιόδοξο)", "25%", "50% (Διάμεσος)", "75%", "95% (Αισιόδοξο)"],
                "Τιμή ($)":  [f"${p:.2f}" for p in [p5, p25, p50, p75, p95]],
                "Μεταβολή":  [f"{(p/S0-1)*100:+.1f}%" for p in [p5, p25, p50, p75, p95]],
            })
            st.dataframe(tbl, use_container_width=True, hide_index=True)

        BG  = "#080812"
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), facecolor=BG)
        x = np.arange(days_ahead + 1)

        ax1.set_facecolor(BG)
        sample = rng.choice(simulations, size=min(300, simulations), replace=False)
        for i in sample:
            ax1.plot(x, paths[:, i], alpha=0.03, lw=0.5, color="#00d4ff")

        pct5_t  = np.percentile(paths, 5,  axis=1)
        pct25_t = np.percentile(paths, 25, axis=1)
        pct50_t = np.percentile(paths, 50, axis=1)
        pct75_t = np.percentile(paths, 75, axis=1)
        pct95_t = np.percentile(paths, 95, axis=1)

        ax1.fill_between(x, pct5_t, pct95_t, alpha=0.12, color="#00d4ff")
        ax1.fill_between(x, pct25_t, pct75_t, alpha=0.22, color="#00d4ff")
        ax1.plot(x, pct50_t, color="#ffffff", lw=2,   label="Διάμεσος")
        ax1.plot(x, pct5_t,  color="#ff4e4e", lw=1.2, ls="--", label=f"5th  ${p5:.0f}")
        ax1.plot(x, pct95_t, color="#4eff91", lw=1.2, ls="--", label=f"95th ${p95:.0f}")
        ax1.axhline(S0, color="#ffcc00", lw=1.5, ls=":", label=f"Είσοδος ${S0:.2f}")
        ax1.set_title(f"Monte Carlo — {ticker}  |  {simulations:,} paths  |  {days_ahead} ημέρες",
                      color="white", fontsize=12, pad=10)
        ax1.set_xlabel("Ημέρες", color="#888")
        ax1.set_ylabel("Τιμή ($)", color="#888")
        ax1.tick_params(colors="#666")
        ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.0f"))
        for sp in ax1.spines.values(): sp.set_edgecolor("#1e1e44")
        ax1.legend(fontsize=8, framealpha=0.3, labelcolor="white",
                   facecolor="#0d0d2b", loc="upper left")

        ax2.set_facecolor(BG)
        n_hist, bins, patches = ax2.hist(final, bins=80, edgecolor="none")
        for patch, left in zip(patches, bins[:-1]):
            patch.set_facecolor("#4eff91" if left >= S0 else "#ff4e4e")
            patch.set_alpha(0.65)
        ax2.axvline(S0,  color="#ffcc00", lw=2,   ls=":",  label=f"Είσοδος ${S0:.2f}")
        ax2.axvline(p50, color="#ffffff", lw=2,   ls="-",  label=f"Διάμεσος ${p50:.2f}")
        ax2.axvline(p5,  color="#ff4e4e", lw=1.4, ls="--", label=f"5th ${p5:.2f}")
        ax2.axvline(p95, color="#4eff91", lw=1.4, ls="--", label=f"95th ${p95:.2f}")
        ax2.set_title(f"Κατανομή τελικής τιμής μετά από {days_ahead} ημέρες",
                      color="white", fontsize=12, pad=10)
        ax2.set_xlabel("Τιμή ($)", color="#888")
        ax2.set_ylabel("Συχνότητα", color="#888")
        ax2.tick_params(colors="#666")
        ax2.xaxis.set_major_formatter(mticker.FormatStrFormatter("$%.0f"))
        for sp in ax2.spines.values(): sp.set_edgecolor("#1e1e44")
        ax2.legend(fontsize=8, framealpha=0.3, labelcolor="white",
                   facecolor="#0d0d2b", loc="upper right")
        ax2.text(0.02, 0.92, f"Πιθανότητα κέρδους: {prob_profit:.1f}%",
                 transform=ax2.transAxes, color="#ffcc00", fontsize=10, fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="#0d0d2b", alpha=0.8))

        plt.tight_layout(pad=2.5)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — HMM
# ════════════════════════════════════════════════════════════════════════════════

with tab_hmm:
    if not run:
        st.info("👈 Ορίστε τις παραμέτρους στα αριστερά και πατήστε **ΕΚΤΕΛΕΣΗ**.")
    else:
        close, err = load_data(ticker, period)
        if err:
            st.error(f"❌ {err}")
            st.stop()

        log_ret  = np.log(close / close.shift(1)).dropna()
        roll_vol = log_ret.rolling(5).std().dropna()
        idx      = log_ret.index.intersection(roll_vol.index)

        ret_vals     = log_ret.loc[idx].values
        features_raw = np.column_stack([ret_vals, roll_vol.loc[idx].values])
        features     = standardize(features_raw)

        with st.spinner("Εκπαίδευση HMM (pure NumPy — Baum-Welch)…"):
            best_model, best_score = None, -np.inf
            for seed in range(8):
                try:
                    m = GaussianHMM(n_states=n_states, n_iter=80, tol=1e-4,
                                    random_state=seed)
                    m.fit(features)
                    s = m.score(features)
                    if s > best_score:
                        best_score, best_model = s, m
                except Exception:
                    continue

        if best_model is None:
            st.error("❌ Αποτυχία εκπαίδευσης HMM. Δοκιμάστε διαφορετικές παραμέτρους.")
            st.stop()

        hidden_states = best_model.predict(features)
        close_aligned = close.loc[idx]
        dates         = close_aligned.index
        trans_mat     = best_model.A_

        state_means_ret = np.array([
            ret_vals[hidden_states == s].mean() if (hidden_states == s).sum() > 0 else 0.0
            for s in range(n_states)
        ])
        regime_labels, regime_colors = label_regimes(state_means_ret, n_states)
        current_state = int(hidden_states[-1])
        current_label = regime_labels[current_state]

        # KPI
        st.markdown(f"### {ticker} — Hidden Markov Model  ({n_states} καταστάσεις)")
        cols = st.columns(n_states + 2)
        cols[0].metric("Τρέχον Regime", current_label)
        cols[1].metric("Log-Likelihood", f"{best_score:.1f}")
        for s in range(n_states):
            pct       = (hidden_states == s).mean() * 100
            stay_prob = trans_mat[s, s] * 100
            cols[2 + s].metric(regime_labels[s], f"{pct:.1f}% χρόνου",
                               f"Stay prob {stay_prob:.1f}%")
        st.markdown("---")

        BG  = "#080812"
        fig = plt.figure(figsize=(13, 14), facecolor=BG)
        gs  = fig.add_gridspec(3, 2, hspace=0.45, wspace=0.3)
        ax_price = fig.add_subplot(gs[0, :])
        ax_ret   = fig.add_subplot(gs[1, 0])
        ax_trans = fig.add_subplot(gs[1, 1])
        ax_fore  = fig.add_subplot(gs[2, :])

        # 1. Τιμή + regime shading
        ax_price.set_facecolor(BG)
        ax_price.plot(dates, close_aligned.values, color="#e0e0ff", lw=1.2, zorder=3)
        prev_s, start_i = hidden_states[0], 0
        for i in range(1, len(hidden_states) + 1):
            cur = hidden_states[i] if i < len(hidden_states) else -1
            if cur != prev_s:
                ax_price.axvspan(dates[start_i], dates[i - 1],
                                 alpha=0.18, color=regime_colors[prev_s], zorder=1)
                start_i, prev_s = i, cur
        patches_leg = [mpatches.Patch(color=regime_colors[s], alpha=0.6,
                                      label=regime_labels[s]) for s in range(n_states)]
        ax_price.legend(handles=patches_leg, fontsize=8, framealpha=0.3,
                        labelcolor="white", facecolor="#0d0d2b", loc="upper left")
        ax_price.set_title(f"Τιμή {ticker} — Ανίχνευση Regime (Viterbi)",
                           color="white", fontsize=11, pad=8)
        ax_price.set_ylabel("Τιμή ($)", color="#888")
        ax_price.tick_params(colors="#666")
        ax_price.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.0f"))
        for sp in ax_price.spines.values(): sp.set_edgecolor("#1e1e44")

        # 2. Return distributions ανά regime
        ax_ret.set_facecolor(BG)
        for s in range(n_states):
            mask = hidden_states == s
            if mask.sum() > 1:
                ax_ret.hist(ret_vals[mask] * 100, bins=40, alpha=0.55,
                            color=regime_colors[s], label=regime_labels[s],
                            edgecolor="none", density=True)
        ax_ret.axvline(0, color="#ffffff", lw=1, ls="--")
        ax_ret.set_title("Κατανομή Αποδόσεων ανά Regime", color="white", fontsize=10, pad=8)
        ax_ret.set_xlabel("Log-Return (%)", color="#888")
        ax_ret.set_ylabel("Πυκνότητα", color="#888")
        ax_ret.tick_params(colors="#666")
        ax_ret.legend(fontsize=7, framealpha=0.3, labelcolor="white", facecolor="#0d0d2b")
        for sp in ax_ret.spines.values(): sp.set_edgecolor("#1e1e44")

        # 3. Transition matrix heatmap
        ax_trans.set_facecolor(BG)
        im = ax_trans.imshow(trans_mat, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        for i in range(n_states):
            for j in range(n_states):
                ax_trans.text(j, i, f"{trans_mat[i,j]:.2f}", ha="center", va="center",
                              color="white" if trans_mat[i,j] < 0.6 else "#080812",
                              fontsize=9, fontweight="bold")
        tick_lbls = [regime_labels[s] for s in range(n_states)]
        ax_trans.set_xticks(range(n_states))
        ax_trans.set_yticks(range(n_states))
        ax_trans.set_xticklabels(tick_lbls, color="#aaa", fontsize=8, rotation=15)
        ax_trans.set_yticklabels(tick_lbls, color="#aaa", fontsize=8)
        ax_trans.set_title("Πίνακας Μεταβάσεων", color="white", fontsize=10, pad=8)
        plt.colorbar(im, ax=ax_trans, fraction=0.046, pad=0.04).ax.tick_params(colors="#aaa")
        for sp in ax_trans.spines.values(): sp.set_edgecolor("#1e1e44")

        # 4. HMM Forecast
        ax_fore.set_facecolor(BG)
        S0_hmm   = float(close_aligned.iloc[-1])
        rng_fore = np.random.default_rng(99)
        n_sims   = 500

        state_mu  = np.array([ret_vals[hidden_states == s].mean()
                               if (hidden_states == s).sum() > 1 else 0.0
                               for s in range(n_states)])
        state_sig = np.array([ret_vals[hidden_states == s].std()
                               if (hidden_states == s).sum() > 1 else 1e-4
                               for s in range(n_states)])

        all_paths    = np.zeros((hmm_forecast_days + 1, n_sims))
        all_paths[0] = S0_hmm
        for sim in range(n_sims):
            state = current_state
            price = S0_hmm
            for d in range(hmm_forecast_days):
                state               = rng_fore.choice(n_states, p=trans_mat[state])
                price               = price * np.exp(rng_fore.normal(state_mu[state],
                                                                      state_sig[state]))
                all_paths[d+1, sim] = price

        x_fore = np.arange(hmm_forecast_days + 1)
        fp5, fp25, fp50, fp75, fp95 = [np.percentile(all_paths, p, axis=1)
                                        for p in [5, 25, 50, 75, 95]]

        ax_fore.fill_between(x_fore, fp5,  fp95,  alpha=0.12, color="#00d4ff")
        ax_fore.fill_between(x_fore, fp25, fp75,  alpha=0.22, color="#00d4ff")
        ax_fore.plot(x_fore, fp50, color="#ffffff", lw=2,
                     label=f"Διάμεσος ${fp50[-1]:.2f}")
        ax_fore.plot(x_fore, fp5,  color="#ff4e4e", lw=1.2, ls="--",
                     label=f"5th ${fp5[-1]:.2f}")
        ax_fore.plot(x_fore, fp95, color="#4eff91", lw=1.2, ls="--",
                     label=f"95th ${fp95[-1]:.2f}")
        ax_fore.axhline(S0_hmm, color="#ffcc00", lw=1.5, ls=":",
                        label=f"Σήμερα ${S0_hmm:.2f}")
        prob_up = (all_paths[-1] > S0_hmm).mean() * 100
        ax_fore.set_title(
            f"HMM Forecast — {hmm_forecast_days} ημέρες  |  {n_states} regimes  |  "
            f"Πιθανότητα ανόδου: {prob_up:.1f}%",
            color="white", fontsize=11, pad=8)
        ax_fore.set_xlabel("Ημέρες", color="#888")
        ax_fore.set_ylabel("Τιμή ($)", color="#888")
        ax_fore.tick_params(colors="#666")
        ax_fore.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.0f"))
        for sp in ax_fore.spines.values(): sp.set_edgecolor("#1e1e44")
        ax_fore.legend(fontsize=8, framealpha=0.3, labelcolor="white",
                       facecolor="#0d0d2b", loc="upper left")

        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.markdown("---")
        with st.expander("📋 Στατιστικά ανά Regime"):
            rows = []
            for s in range(n_states):
                mask = hidden_states == s
                r    = ret_vals[mask] * 100
                rows.append({
                    "Regime":           regime_labels[s],
                    "% Χρόνου":         f"{mask.mean()*100:.1f}%",
                    "Μέση Ημ. Απόδοση": f"{r.mean():.3f}%",
                    "Volatility (ημ.)": f"{r.std():.3f}%",
                    "Sharpe (ημ.)":     f"{r.mean()/r.std():.3f}" if r.std() > 0 else "—",
                    "Stay Probability": f"{trans_mat[s,s]*100:.1f}%",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
