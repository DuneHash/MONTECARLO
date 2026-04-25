import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Monte Carlo Simulator",
    page_icon="📈",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────

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
.stSelectbox select, .stTextInput input {
    background: #0d0d2b !important;
    color: #e0e0ff !important;
    border: 1px solid #1e1e55 !important;
}

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

# ── Header ────────────────────────────────────────────────────────────────────

st.markdown("# 📈 Monte Carlo Simulator")
st.markdown("*Geometric Brownian Motion — powered by yfinance*")
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

    days_ahead = st.slider(
        "Ημέρες πρόβλεψης", 30, 504, 252, step=21,
        help="252 ≈ 1 χρόνος συναλλαγών"
    )

    simulations = st.select_slider(
        "Αριθμός προσομοιώσεων",
        options=[500, 1000, 2000, 5000, 10000],
        value=1000
    )

    st.markdown("---")
    run = st.button("▶  ΕΚΤΕΛΕΣΗ ΠΡΟΣΟΜΟΙΩΣΗΣ")

# ── Main ──────────────────────────────────────────────────────────────────────

if run:
    # 1. Λήψη δεδομένων
    with st.spinner(f"Λήψη δεδομένων για **{ticker}** …"):
        try:
            df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
        except Exception as e:
            st.error(f"❌ Σφάλμα κατά τη λήψη δεδομένων: {e}")
            st.stop()

    if df.empty:
        st.error(f"❌ Δεν βρέθηκαν δεδομένα για '{ticker}'. Ελέγξτε το σύμβολο.")
        st.stop()

    # FIX: Νέες εκδόσεις yfinance επιστρέφουν multi-level columns
    # Χρειάζεται flatten για να πάρουμε τη στήλη "Close" σωστά
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if "Close" not in df.columns:
        st.error(f"❌ Δεν βρέθηκε στήλη 'Close' στα δεδομένα. Διαθέσιμες στήλες: {list(df.columns)}")
        st.stop()

    close = df["Close"].dropna()

    if len(close) < 20:
        st.error(f"❌ Ανεπαρκή ιστορικά δεδομένα ({len(close)} ημέρες). Επιλέξτε μεγαλύτερο διάστημα.")
        st.stop()

    S0 = float(close.iloc[-1])

    # 2. Παράμετροι GBM
    log_ret = np.log(close / close.shift(1)).dropna()

    if log_ret.empty or log_ret.std() == 0:
        st.error("❌ Δεν ήταν δυνατός ο υπολογισμός αποδόσεων. Δοκιμάστε διαφορετικό ticker ή περίοδο.")
        st.stop()

    mu    = float(log_ret.mean())
    sigma = float(log_ret.std())

    # 3. Monte Carlo — χρήση numpy Generator αντί deprecated np.random.seed
    rng    = np.random.default_rng(42)
    shocks = rng.normal(
        (mu - 0.5 * sigma ** 2),
        sigma,
        size=(days_ahead, simulations)
    )
    paths = S0 * np.exp(np.cumsum(shocks, axis=0))
    paths = np.vstack([np.full(simulations, S0), paths])

    final               = paths[-1]
    p5, p25, p50, p75, p95 = np.percentile(final, [5, 25, 50, 75, 95])
    avg         = final.mean()
    prob_profit = (final > S0).mean() * 100
    exp_ret     = (avg / S0 - 1) * 100
    annual_vol  = sigma * np.sqrt(252) * 100

    # ── KPI cards ─────────────────────────────────────────────────────────────
    st.markdown(
        f"### {ticker} — Αποτελέσματα ({simulations:,} προσομοιώσεις, {days_ahead} ημέρες)"
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Τρέχουσα τιμή",     f"${S0:.2f}")
    c2.metric("Διάμεσος 50%",      f"${p50:.2f}", f"{(p50/S0-1)*100:+.1f}%")
    c3.metric("Μέση τελ. τιμή",    f"${avg:.2f}", f"{exp_ret:+.1f}%")
    c4.metric("Πιθ. κέρδους",      f"{prob_profit:.1f}%")
    c5.metric("Ετήσια Volatility",  f"{annual_vol:.1f}%")

    st.markdown("---")

    # ── Percentile table ──────────────────────────────────────────────────────
    with st.expander("📋 Πίνακας Percentiles"):
        tbl = pd.DataFrame({
            "Percentile": [
                "5% (Απαισιόδοξο)", "25%", "50% (Διάμεσος)", "75%", "95% (Αισιόδοξο)"
            ],
            "Τιμή ($)": [f"${p:.2f}" for p in [p5, p25, p50, p75, p95]],
            "Μεταβολή": [f"{(p/S0-1)*100:+.1f}%" for p in [p5, p25, p50, p75, p95]],
        })
        st.dataframe(tbl, use_container_width=True, hide_index=True)

    # ── Charts ────────────────────────────────────────────────────────────────
    BG  = "#080812"
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), facecolor=BG)

    x = np.arange(days_ahead + 1)

    # --- Paths chart ---
    ax1.set_facecolor(BG)
    sample_size = min(300, simulations)
    sample      = rng.choice(simulations, size=sample_size, replace=False)
    for i in sample:
        ax1.plot(x, paths[:, i], alpha=0.03, lw=0.5, color="#00d4ff")

    pct5_t  = np.percentile(paths, 5,  axis=1)
    pct25_t = np.percentile(paths, 25, axis=1)
    pct50_t = np.percentile(paths, 50, axis=1)
    pct75_t = np.percentile(paths, 75, axis=1)
    pct95_t = np.percentile(paths, 95, axis=1)

    ax1.fill_between(x, pct5_t,  pct95_t, alpha=0.12, color="#00d4ff")
    ax1.fill_between(x, pct25_t, pct75_t, alpha=0.22, color="#00d4ff")
    ax1.plot(x, pct50_t, color="#ffffff",  lw=2,   label="Διάμεσος")
    ax1.plot(x, pct5_t,  color="#ff4e4e",  lw=1.2, ls="--", label=f"5th pct  ${p5:.0f}")
    ax1.plot(x, pct95_t, color="#4eff91",  lw=1.2, ls="--", label=f"95th pct ${p95:.0f}")
    ax1.axhline(S0, color="#ffcc00", lw=1.5, ls=":", label=f"Είσοδος ${S0:.2f}")

    ax1.set_title(
        f"Monte Carlo — {ticker}  |  {simulations:,} paths  |  {days_ahead} ημέρες",
        color="white", fontsize=12, pad=10
    )
    ax1.set_xlabel("Ημέρες", color="#888")
    ax1.set_ylabel("Τιμή ($)", color="#888")
    ax1.tick_params(colors="#666")
    ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.0f"))
    for sp in ax1.spines.values():
        sp.set_edgecolor("#1e1e44")
    ax1.legend(
        fontsize=8, framealpha=0.3, labelcolor="white",
        facecolor="#0d0d2b", loc="upper left"
    )

    # --- Distribution chart ---
    ax2.set_facecolor(BG)
    n, bins, patches = ax2.hist(final, bins=80, edgecolor="none")
    for patch, left in zip(patches, bins[:-1]):
        patch.set_facecolor("#4eff91" if left >= S0 else "#ff4e4e")
        patch.set_alpha(0.65)

    ax2.axvline(S0,  color="#ffcc00", lw=2,   ls=":",  label=f"Είσοδος ${S0:.2f}")
    ax2.axvline(p50, color="#ffffff", lw=2,   ls="-",  label=f"Διάμεσος ${p50:.2f}")
    ax2.axvline(p5,  color="#ff4e4e", lw=1.4, ls="--", label=f"5th ${p5:.2f}")
    ax2.axvline(p95, color="#4eff91", lw=1.4, ls="--", label=f"95th ${p95:.2f}")

    ax2.set_title(
        f"Κατανομή τελικής τιμής μετά από {days_ahead} ημέρες",
        color="white", fontsize=12, pad=10
    )
    ax2.set_xlabel("Τιμή ($)", color="#888")
    ax2.set_ylabel("Συχνότητα", color="#888")
    ax2.tick_params(colors="#666")
    ax2.xaxis.set_major_formatter(mticker.FormatStrFormatter("$%.0f"))
    for sp in ax2.spines.values():
        sp.set_edgecolor("#1e1e44")
    ax2.legend(
        fontsize=8, framealpha=0.3, labelcolor="white",
        facecolor="#0d0d2b", loc="upper right"
    )
    ax2.text(
        0.02, 0.92, f"Πιθανότητα κέρδους: {prob_profit:.1f}%",
        transform=ax2.transAxes, color="#ffcc00", fontsize=10, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#0d0d2b", alpha=0.8)
    )

    plt.tight_layout(pad=2.5)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)  # FIX: αποφυγή memory leak από ανοιχτά figures

else:
    st.info("👈 Ορίστε τις παραμέτρους στα αριστερά και πατήστε **ΕΚΤΕΛΕΣΗ ΠΡΟΣΟΜΟΙΩΣΗΣ**.")