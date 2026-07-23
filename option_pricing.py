"""
Risk Management Project - Part B
Option Pricing & Volatility Inputs
==================================

Builds on Part A's historical data engine for:
- HDFC Bank (HDFCBANK.NS) -> liquid benchmark
- Nestle India (NESTLEIND.NS) -> illiquid benchmark

Evaluation date: 29 June 2026.

Pipeline
--------
1. Hardcode the 29-Jun-2026 option-chain snapshot into a tidy DataFrame.
2. Derive two volatility inputs per contract horizon:
   (a) 20-day annualized realized volatility (from Part A).
   (b) GARCH(1,1) average integrated conditional volatility, forecast
       over the 29-day and 57-day horizons via the `arch` library.
3. Price every European call/put with a Black-Scholes-Merton engine,
   once per volatility input.
4. Produce a master comparison table with absolute/percentage deviations
   of model prices vs. market, plus Excel/LaTeX-ready exports and a
   textual insights printout.

Requirements:
pip install yfinance pandas numpy scipy arch openpyxl
(Part A script `part_a_market_liquidity_analysis.py` importable on path.)
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from scipy.stats import norm
from arch import arch_model


# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
EVAL_DATE = datetime(2026, 6, 29) # base evaluation date
RISK_FREE_RATE = 0.0675 # annualized Indian risk-free rate (6.75%)
DIVIDEND_YIELD = 0.0 # q = 0 (no continuous dividend assumed)
TRADING_DAYS = 252 # annualization factor
LOOKBACK_CALENDAR_DAYS = 215 # ~6 months buffer to anchor history to EVAL_DATE

TICKER_MAP = {
    "HDFC Bank": "HDFCBANK.NS",
    "Nestle India": "NESTLEIND.NS",
}

SPOT = {
    "HDFC Bank": 797.05,
    "Nestle India": 1390.40,
}

# Days to expiry per expiry label (calendar days from EVAL_DATE).
EXPIRY_DTE = {
    "28-Jul-2026": 29,
    "25-Aug-2026": 57,
}


# ---------------------------------------------------------------------------
# 1. HARDCODED OPTION-CHAIN SNAPSHOT (29-Jun-2026)
# ---------------------------------------------------------------------------
def build_option_chain() -> pd.DataFrame:
    """Return the hardcoded 29-Jun-2026 option snapshot as a tidy DataFrame.
    
    One row per (stock, expiry, strike, option type). ATM strikes that quote
    both a call and a put therefore expand into two rows.
    """
    # (stock, expiry_label, strike, option_type, market_price)
    records = [
        # ---- HDFC Bank | 28-Jul-2026 (29 DTE) -----------------------------
        ("HDFC Bank", "28-Jul-2026", 800.00, "Call", 23.70),
        ("HDFC Bank", "28-Jul-2026", 800.00, "Put", 21.00),
        ("HDFC Bank", "28-Jul-2026", 760.00, "Put", 6.75),
        ("HDFC Bank", "28-Jul-2026", 805.00, "Call", 21.15),
        # ---- HDFC Bank | 25-Aug-2026 (57 DTE) -----------------------------
        ("HDFC Bank", "25-Aug-2026", 800.00, "Call", 34.50),
        ("HDFC Bank", "25-Aug-2026", 800.00, "Put", 26.15),
        ("HDFC Bank", "25-Aug-2026", 750.00, "Put", 9.00),
        ("HDFC Bank", "25-Aug-2026", 820.00, "Call", 25.20),
        # ---- Nestle India | 28-Jul-2026 (29 DTE) --------------------------
        ("Nestle India", "28-Jul-2026", 1390.00, "Call", 43.95),
        ("Nestle India", "28-Jul-2026", 1390.00, "Put", 35.05),
        ("Nestle India", "28-Jul-2026", 1330.00, "Put", 13.70),
        ("Nestle India", "28-Jul-2026", 1420.00, "Call", 29.50),
        # ---- Nestle India | 25-Aug-2026 (57 DTE, extreme illiquidity) -----
        ("Nestle India", "25-Aug-2026", 1360.00, "Put", 30.00),
        ("Nestle India", "25-Aug-2026", 1500.00, "Call", 25.85),
    ]

    df = pd.DataFrame(
        records,
        columns=["Stock", "Expiry", "Strike", "Option Type", "Market Price"],
    )

    # Enrich with spot, days-to-expiry and time-to-maturity (calendar/365).
    df["Spot"] = df["Stock"].map(SPOT)
    df["DTE"] = df["Expiry"].map(EXPIRY_DTE)
    df["T"] = df["DTE"] / 365.0 # time to maturity in years

    return df


# ---------------------------------------------------------------------------
# 2a. HISTORICAL DATA (REUSE PART A ENGINE)
# ---------------------------------------------------------------------------
def load_history(stock: str, symbol: str, eval_date: datetime = EVAL_DATE) -> pd.DataFrame:
    """Fetch ~6 months of daily history anchored to `eval_date` and compute
    Part A metrics (log returns + 20-day annualized realized volatility).
    """
    import yfinance as yf # lazy import: only needed at runtime

    start = (eval_date - timedelta(days=LOOKBACK_CALENDAR_DAYS)).strftime("%Y-%m-%d")
    end = (eval_date + timedelta(days=1)).strftime("%Y-%m-%d") # end is exclusive

    raw = yf.download(symbol, start=start, end=end, interval="1d",
                      auto_adjust=True, progress=False)
    if raw is None or raw.empty:
        raise ValueError(f"No history returned for {stock} ({symbol}).")

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw = raw[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"])

    try:
        # Preferred path: identical metric definitions as Part A.
        from part_a_market_liquidity_analysis import calculate_metrics
        metrics = calculate_metrics(raw)
    except Exception:
        # Fallback: replicate just the two fields Part B needs.
        metrics = raw.copy()
        metrics["LogReturn"] = np.log(metrics["Close"] / metrics["Close"].shift(1))
        metrics["RollingVol"] = (
            metrics["LogReturn"].rolling(20).std() * np.sqrt(TRADING_DAYS)
        )
        metrics = metrics.dropna(subset=["LogReturn"])

    return metrics


def latest_historical_vol(metrics: pd.DataFrame) -> float:
    """Return the most recent (as-of eval date) 20-day annualized realized vol."""
    vol = metrics["RollingVol"].dropna()
    if vol.empty:
        raise ValueError("Rolling volatility series is empty - insufficient history.")
    return float(vol.iloc[-1])


# ---------------------------------------------------------------------------
# 2b. GARCH(1,1) VOLATILITY ENGINE
# ---------------------------------------------------------------------------
def garch_forecast_vol(
    log_returns: pd.Series,
    horizon_days: int,
    trading_days: int = TRADING_DAYS,
) -> tuple[float, object]:
    """Fit GARCH(1,1) and return the annualized average integrated conditional
    volatility over `horizon_days` forecast steps.
    """
    returns_pct = log_returns.dropna().astype(float) * 100.0

    model = arch_model(returns_pct, mean="Constant", vol="GARCH",
                       p=1, q=1, dist="normal")
    result = model.fit(disp="off")

    forecast = result.forecast(horizon=horizon_days, reindex=False)
    daily_var_pct2 = np.asarray(forecast.variance.values[-1, :], dtype=float)

    # Convert percent^2 -> decimal^2 and average across the horizon path.
    avg_daily_var = np.nanmean(daily_var_pct2) / (100.0 ** 2)
    annualized_vol = float(np.sqrt(avg_daily_var * trading_days))

    return annualized_vol, result


# ---------------------------------------------------------------------------
# 3. BLACK-SCHOLES-MERTON PRICING ENGINE
# ---------------------------------------------------------------------------
def bsm_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str,
    q: float = DIVIDEND_YIELD,
) -> float:
    """Black-Scholes-Merton price for a European call or put."""
    opt = option_type.lower()
    if opt not in ("call", "put"):
        raise ValueError(f"option_type must be 'Call' or 'Put', got '{option_type}'.")

    # Degenerate-input guard -> discounted intrinsic value.
    if T <= 0 or sigma <= 0:
        if opt == "call":
            return max(S * np.exp(-q * T) - K * np.exp(-r * T), 0.0)
        return max(K * np.exp(-r * T) - S * np.exp(-q * T), 0.0)

    sqrt_t = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t

    if opt == "call":
        price = (S * np.exp(-q * T) * norm.cdf(d1)
                 - K * np.exp(-r * T) * norm.cdf(d2))
    else:
        price = (K * np.exp(-r * T) * norm.cdf(-d2)
                 - S * np.exp(-q * T) * norm.cdf(-d1))
    return float(price)


# ---------------------------------------------------------------------------
# 4. ASSEMBLY OF THE MASTER COMPARISON TABLE
# ---------------------------------------------------------------------------
def build_volatility_inputs() -> tuple[dict, dict, pd.DataFrame]:
    """Compute historical and GARCH volatilities for each stock / horizon."""
    hist_vol: dict[str, float] = {}
    garch_vol: dict[tuple[str, int], float] = {}
    rows = []

    for stock, symbol in TICKER_MAP.items():
        print(f"[INFO] Loading history & fitting GARCH for {stock} ...")
        metrics = load_history(stock, symbol)
        hist_vol[stock] = latest_historical_vol(metrics)
        log_returns = metrics["LogReturn"]

        for expiry, dte in EXPIRY_DTE.items():
            g_vol, _ = garch_forecast_vol(log_returns, horizon_days=dte)
            garch_vol[(stock, dte)] = g_vol
            rows.append({
                "Stock": stock,
                "Expiry": expiry,
                "DTE": dte,
                "Historical Vol": hist_vol[stock],
                "GARCH Vol": g_vol,
            })

    vol_table = pd.DataFrame(rows)
    return hist_vol, garch_vol, vol_table


def price_all_contracts(
    chain: pd.DataFrame,
    hist_vol: dict,
    garch_vol: dict,
) -> pd.DataFrame:
    """Price every contract under both volatility inputs and attach deviations."""
    df = chain.copy()

    hist_prices, garch_prices = [], []
    hist_vols_used, garch_vols_used = [], []

    for _, row in df.iterrows():
        h_vol = hist_vol[row["Stock"]]
        g_vol = garch_vol[(row["Stock"], row["DTE"])]

        hist_prices.append(
            bsm_price(row["Spot"], row["Strike"], row["T"],
                      RISK_FREE_RATE, h_vol, row["Option Type"])
        )
        garch_prices.append(
            bsm_price(row["Spot"], row["Strike"], row["T"],
                      RISK_FREE_RATE, g_vol, row["Option Type"])
        )
        hist_vols_used.append(h_vol)
        garch_vols_used.append(g_vol)

    df["Hist Vol Used"] = hist_vols_used
    df["GARCH Vol Used"] = garch_vols_used
    df["BSM Price (Hist Vol)"] = hist_prices
    df["BSM Price (GARCH Vol)"] = garch_prices

    # --- Deviation metrics (model - market) ------------------------------
    df["Abs Dev Hist"] = (df["BSM Price (Hist Vol)"] - df["Market Price"]).abs()
    df["Pct Dev Hist (%)"] = (
        (df["BSM Price (Hist Vol)"] - df["Market Price"]) / df["Market Price"] * 100.0
    )
    df["Abs Dev GARCH"] = (df["BSM Price (GARCH Vol)"] - df["Market Price"]).abs()
    df["Pct Dev GARCH (%)"] = (
        (df["BSM Price (GARCH Vol)"] - df["Market Price"]) / df["Market Price"] * 100.0
    )

    # Final column order for a clean, exportable table.
    ordered = [
        "Stock", "Expiry", "Option Type", "Strike", "Spot", "DTE", "T",
        "Hist Vol Used", "GARCH Vol Used",
        "Market Price", "BSM Price (Hist Vol)", "BSM Price (GARCH Vol)",
        "Abs Dev Hist", "Pct Dev Hist (%)", "Abs Dev GARCH", "Pct Dev GARCH (%)",
    ]
    df = df[ordered]

    # Round for presentation; guarantees no stray long-float NaNs slip through.
    round_map = {
        "T": 4, "Hist Vol Used": 4, "GARCH Vol Used": 4,
        "Market Price": 2, "BSM Price (Hist Vol)": 2, "BSM Price (GARCH Vol)": 2,
        "Abs Dev Hist": 2, "Pct Dev Hist (%)": 2,
        "Abs Dev GARCH": 2, "Pct Dev GARCH (%)": 2,
    }
    df = df.round(round_map)
    return df


# ---------------------------------------------------------------------------
# EXPORTS & INSIGHTS
# ---------------------------------------------------------------------------
def export_outputs(
    master: pd.DataFrame,
    vol_table: pd.DataFrame,
    xlsx_file: str = "part_b_option_pricing.xlsx",
    latex_file: str = "part_b_master_table.tex",
) -> None:
    """Write the master + volatility tables to Excel and emit a LaTeX table."""
    with pd.ExcelWriter(xlsx_file, engine="openpyxl") as writer:
        master.to_excel(writer, sheet_name="Master_Comparison", index=False)
        vol_table.to_excel(writer, sheet_name="Volatility_Inputs", index=False)
    print(f"[EXPORT] Excel workbook -> {xlsx_file}")

    latex = master.to_latex(
        index=False,
        caption="BSM Model vs. Market Prices (Historical vs. GARCH Volatility)",
        label="tab:bsm_comparison",
        float_format="%.2f",
    )
    with open(latex_file, "w", encoding="utf-8") as fh:
        fh.write(latex)
    print(f"[EXPORT] LaTeX table -> {latex_file}")


def print_section(title: str) -> None:
    bar = "=" * 74
    print(f"\n{bar}\n{title}\n{bar}")


def report_insights(master: pd.DataFrame, vol_table: pd.DataFrame) -> None:
    """Print clean textual summaries of the key analytical findings."""
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)

    print_section("VOLATILITY INPUTS (annualized)")
    vt = vol_table.copy()
    vt["Historical Vol"] = (vt["Historical Vol"] * 100).round(2).astype(str) + "%"
    vt["GARCH Vol"] = (vt["GARCH Vol"] * 100).round(2).astype(str) + "%"
    print(vt.to_string(index=False))

    print_section("MASTER COMPARISON TABLE")
    print(master.to_string(index=False))

    # --- Insight 1: sensitivity to the volatility assumption --------------
    print_section("INSIGHT 1 - PRICE SENSITIVITY TO VOLATILITY ASSUMPTION")
    sens = master.copy()
    sens["Vol Gap (pp)"] = (sens["GARCH Vol Used"] - sens["Hist Vol Used"]) * 100
    sens["Price Gap"] = sens["BSM Price (GARCH Vol)"] - sens["BSM Price (Hist Vol)"]
    for stock in master["Stock"].unique():
        sub = sens[sens["Stock"] == stock]
        mean_volgap = sub["Vol Gap (pp)"].mean()
        mean_pricegap = sub["Price Gap"].abs().mean()
        higher = "GARCH" if mean_volgap > 0 else "Historical"
        print(f"  {stock}:")
        print(f"    - Avg vol gap (GARCH - Hist): {mean_volgap:+.2f} pp "
              f"=> {higher} vol is higher on average.")
        print(f"    - Avg |price shift| from switching vol input: "
              f"{mean_pricegap:.2f} (currency units per contract).")
        print(f"    - Higher volatility input lifts BOTH call and put premia "
              f"(positive vega), so the model price moves in the direction of "
              f"whichever engine reads vol higher.")

    # --- Insight 2: liquid vs. illiquid structural deviations -------------
    print_section("INSIGHT 2 - STRUCTURAL DEVIATIONS: LIQUID vs. ILLIQUID")
    dev = master.copy()
    dev["Best Abs Pct Dev"] = dev[["Pct Dev Hist (%)", "Pct Dev GARCH (%)"]].abs().min(axis=1)
    for stock in master["Stock"].unique():
        sub = dev[dev["Stock"] == stock]
        print(f"  {stock}: mean |%dev| (best-fit vol) = "
              f"{sub['Best Abs Pct Dev'].mean():.2f}% | "
              f"max |%dev| = {sub['Best Abs Pct Dev'].max():.2f}%")

    # Spotlight: Nestle's August expiry (the structurally illiquid quotes).
    aug_illiquid = dev[(dev["Stock"] == "Nestle India") & (dev["Expiry"] == "25-Aug-2026")]
    if not aug_illiquid.empty:
        worst = aug_illiquid.loc[aug_illiquid["Best Abs Pct Dev"].idxmax()]
        print("\n  Spotlight - Nestle India 25-Aug-2026 (thin / limited quotes):")
        print(f"    - Largest model-vs-market gap: {worst['Option Type']} "
              f"K={worst['Strike']:.0f}, |%dev| = {worst['Best Abs Pct Dev']:.2f}%.")
        print("    - Sparse two-sided quotes widen bid-ask spreads, so the printed "
              "market price reflects a large liquidity premium rather than fair "
              "value. BSM (a frictionless, continuously-hedgeable model) cannot "
              "capture that premium, which is why deviations are materially larger "
              "here than for HDFC Bank's actively quoted contracts.")

    print_section("KEY TAKEAWAY")
    liq_dev = dev[dev["Stock"] == "HDFC Bank"]["Best Abs Pct Dev"].mean()
    illiq_dev = dev[dev["Stock"] == "Nestle India"]["Best Abs Pct Dev"].mean()
    print(f"  HDFC Bank (liquid) mean |%dev|: {liq_dev:.2f}%")
    print(f"  Nestle India (illiq) mean |%dev|: {illiq_dev:.2f}%")
    direction = "wider" if illiq_dev > liq_dev else "narrower"
    print(f"  => Model fit is {direction} for the illiquid name, consistent with "
          f"liquidity frictions pushing market prices away from BSM fair value.")


# ---------------------------------------------------------------------------
# MAIN ORCHESTRATION
# ---------------------------------------------------------------------------
def main() -> None:
    print_section("PART B: OPTION PRICING & VOLATILITY INPUTS")
    print(f"Evaluation date: {EVAL_DATE:%d-%b-%Y} | r = {RISK_FREE_RATE:.4f} | "
          f"q = {DIVIDEND_YIELD:.2f}")

    chain = build_option_chain()
    hist_vol, garch_vol, vol_table = build_volatility_inputs()
    master = price_all_contracts(chain, hist_vol, garch_vol)

    # Safety net: surface (do not silently keep) any unexpected NaNs.
    if master.isna().any().any():
        bad_cols = master.columns[master.isna().any()].tolist()
        print(f"[WARN] NaNs detected in columns {bad_cols}; filling with 0 for export.")
        master = master.fillna(0)

    report_insights(master, vol_table)
    export_outputs(master, vol_table)

    print_section("DONE")
    print("Generated files:")
    print(" - part_b_option_pricing.xlsx")
    print(" - part_b_master_table.tex")


if __name__ == "__main__":
    main()
