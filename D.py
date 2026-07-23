"""
Risk Management Project - Part D
Risk Measurement & Stress Analysis
==================================

Builds on Part A (6-month daily log returns + 20-day rolling volatility) for:
- HDFC Bank (HDFCBANK.NS) -> liquid benchmark
- Nestle India (NESTLEIND.NS) -> illiquid benchmark

Evaluation date: 29-Jun-2026. Standardized portfolio value: Rs 1,000,000 per stock.

Methods
-------
1. Standard parametric (variance-covariance) 1-day VaR @ 95% / 99%.
2. Regime-split parametric VaR (Normal 75% vs. High-Vol top 25% by rolling vol).
3. GARCH(1,1) 1-day conditional-vol forecast -> Monte Carlo VaR (>=10k sims).
4. Consolidated master VaR table + strategic insights.

Dependencies: NumPy, SciPy, Pandas, arch.

INPUT HAND-OFF
--------------
Historical data + the Part A metric definitions (LogReturn, RollingVol,
TurnoverValue, Amihud) are reused via the Part A module when importable;
otherwise a compact local fallback reproduces them.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from scipy.stats import norm


# ===========================================================================
# CONFIGURATION
# ===========================================================================
EVAL_DATE = datetime(2026, 6, 29)
PORTFOLIO_VALUE = 1_000_000.0    # Rs per stock (standardized)
CONFIDENCE_LEVELS = (0.95, 0.99) # VaR confidence levels
HIGH_VOL_PERCENTILE = 0.75       # top 25% rolling-vol days = high-vol regime
ROLLING_WINDOW = 20
TRADING_DAYS = 252
LOOKBACK_CALENDAR_DAYS = 215     # ~6 months buffer to anchor history to EVAL_DATE

N_SIMULATIONS = 50_000           # Monte Carlo paths (>= 10,000)
MC_DISTRIBUTION = "t"            # GARCH innovation dist: "t" (fat tails) or "normal"
RANDOM_SEED = 42                 # reproducible Monte Carlo

TICKER_MAP = {
    "HDFC Bank": "HDFCBANK.NS",
    "Nestle India": "NESTLEIND.NS",
}


# ===========================================================================
# 0. HISTORICAL DATA (REUSE PART A ENGINE)
# ===========================================================================
def load_history(stock: str, symbol: str, eval_date: datetime = EVAL_DATE) -> pd.DataFrame:
    """Fetch ~6 months of daily history anchored to `eval_date` and compute the
    Part A metrics (LogReturn, RollingVol, TurnoverValue, Amihud).

    Reuses Part A's `calculate_metrics` when available; otherwise falls back to
    a local replica so Part D runs standalone. Imports are local so importing
    this module triggers no network access.
    """
    import yfinance as yf # lazy import

    start = (eval_date - timedelta(days=LOOKBACK_CALENDAR_DAYS)).strftime("%Y-%m-%d")
    end = (eval_date + timedelta(days=1)).strftime("%Y-%m-%d") # end exclusive

    raw = yf.download(symbol, start=start, end=end, interval="1d",
                      auto_adjust=True, progress=False)
    
    if raw is None or raw.empty:
        raise ValueError(f"No history returned for {stock} ({symbol}).")
    
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    
    raw = raw[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"])

    try:
        from part_a_market_liquidity_analysis import calculate_metrics
        metrics = calculate_metrics(raw)
    except Exception:
        metrics = raw.copy()
        metrics["LogReturn"] = np.log(metrics["Close"] / metrics["Close"].shift(1))
        metrics["RollingVol"] = (
            metrics["LogReturn"].rolling(ROLLING_WINDOW).std() * np.sqrt(TRADING_DAYS)
        )
        metrics["TurnoverValue"] = metrics["Volume"] * metrics["Close"]
        safe_turnover = metrics["TurnoverValue"].where(metrics["TurnoverValue"] > 0)
        metrics["Amihud"] = (metrics["LogReturn"].abs() / (safe_turnover + 1e-12)) * 1e9
        metrics = metrics.dropna(subset=["LogReturn"])

    return metrics


# ===========================================================================
# 1. STANDARD PARAMETRIC (VARIANCE-COVARIANCE) VaR
# ===========================================================================
def parametric_var(
    returns: pd.Series,
    portfolio_value: float = PORTFOLIO_VALUE,
    confidence_levels: tuple = CONFIDENCE_LEVELS,
) -> dict[float, float]:
    """1-day parametric VaR for each confidence level.

    Formula (loss reported as a positive number):
        VaR = PortfolioValue * ( Z * sigma_daily - mu_daily )
    where Z = Phi^{-1}(confidence) is the standard-normal quantile (norm.ppf),
    sigma_daily = std of daily log returns, mu_daily = mean daily log return.
    The -mu term gives a (small) drift credit against the loss.
    """
    clean = returns.dropna()
    mu = clean.mean()
    sigma = clean.std(ddof=1) # sample std

    out = {}
    for cl in confidence_levels:
        z = norm.ppf(cl) # 1.645 @95%, 2.326 @99%
        out[cl] = portfolio_value * (z * sigma - mu)
    return out


# ===========================================================================
# 2. REGIME-BASED VaR (NORMAL vs. HIGH-VOLATILITY)
# ===========================================================================
def split_by_regime(
    metrics: pd.DataFrame,
    percentile: float = HIGH_VOL_PERCENTILE,
) -> tuple[pd.Series, pd.Series, float]:
    """Classify days into Normal / High-Vol regimes by 20-day rolling vol.

    High-Vol = days whose RollingVol is at/above the `percentile` cutoff
    (top 25% by default); Normal = the remaining days. Returns the LogReturn
    series for each regime plus the rolling-vol threshold used.
    """
    # Align: only days that have BOTH a valid rolling vol and a valid return.
    df = metrics.dropna(subset=["RollingVol", "LogReturn"]).copy()
    threshold = df["RollingVol"].quantile(percentile)

    high_mask = df["RollingVol"] >= threshold
    high_vol_returns = df.loc[high_mask, "LogReturn"]
    normal_returns = df.loc[~high_mask, "LogReturn"]
    return normal_returns, high_vol_returns, threshold


# ===========================================================================
# 3. GARCH(1,1) + MONTE CARLO VaR
# ===========================================================================
def garch_monte_carlo_var(
    returns: pd.Series,
    portfolio_value: float = PORTFOLIO_VALUE,
    confidence_levels: tuple = CONFIDENCE_LEVELS,
    n_sims: int = N_SIMULATIONS,
    distribution: str = MC_DISTRIBUTION,
    seed: int = RANDOM_SEED,
) -> tuple[dict[float, float], float]:
    """Fit GARCH(1,1), forecast 1-day conditional vol, and Monte Carlo the VaR.

    Steps
    -----
    1. Scale returns to percent (*100) for arch numerical stability.
    2. Fit constant-mean GARCH(1,1) with the chosen innovation distribution.
    3. 1-step-ahead conditional variance forecast -> daily sigma (decimal).
    4. Draw `n_sims` standardized innovations (unit variance) from the fitted
       distribution, build simulated returns r = mu + sigma * z, and convert to
       PnL = PortfolioValue * r.
    5. VaR = -percentile(PnL, 100*(1-cl)) (5th pct -> 95% VaR, 1st pct -> 99%).

    Returns (var_dict, sigma_1d_decimal).
    """
    from arch import arch_model # local import keeps module import light

    scaled = returns.dropna().astype(float) * 100.0
    model = arch_model(scaled, mean="Constant", vol="GARCH",
                       p=1, q=1, dist=distribution)
    res = model.fit(disp="off")

    # --- 1-day-ahead conditional volatility forecast ----------------------
    fc = res.forecast(horizon=1, reindex=False)
    sigma_1d = float(np.sqrt(fc.variance.values[-1, 0]) / 100.0) # back to decimal
    mu_daily = float(res.params.get("mu", 0.0)) / 100.0          # constant mean

    # --- Monte Carlo innovations (standardized to unit variance) ----------
    rng = np.random.default_rng(seed)
    if distribution == "t" and "nu" in res.params:
        nu = float(res.params["nu"])
        # Student-t has Var = nu/(nu-2); rescale to unit variance.
        raw = rng.standard_t(nu, size=n_sims)
        z = raw * np.sqrt((nu - 2.0) / nu) if nu > 2 else raw
    else:
        z = rng.standard_normal(n_sims)

    simulated_returns = mu_daily + sigma_1d * z
    simulated_pnl = portfolio_value * simulated_returns

    out = {}
    for cl in confidence_levels:
        pct = 100.0 * (1.0 - cl) # 5.0 for 95%, 1.0 for 99%
        out[cl] = float(-np.percentile(simulated_pnl, pct))
    return out, sigma_1d


# ===========================================================================
# 4. MASTER TABLE & FORMATTING
# ===========================================================================
def build_master_table(results: dict[str, dict]) -> pd.DataFrame:
    """Assemble the numeric master VaR table from per-stock result dicts.

    `results[stock][method] = {0.95: var95, 0.99: var99}`
    """
    method_order = [
        "Standard Parametric",
        "Normal Regime",
        "High-Vol Regime",
        "GARCH-MC",
    ]
    rows = []
    for stock, methods in results.items():
        for method in method_order:
            var = methods[method]
            rows.append({
                "Stock": stock,
                "VaR Method": method,
                "95% VaR (Rs)": var[0.95],
                "99% VaR (Rs)": var[0.99],
            })
    return pd.DataFrame(rows)


def format_master_table(master: pd.DataFrame) -> pd.DataFrame:
    """Return an export-ready copy with monetary values as Rs-comma strings."""
    fmt = master.copy()
    for col in ("95% VaR (Rs)", "99% VaR (Rs)"):
        fmt[col] = fmt[col].map(lambda v: f"Rs {v:,.2f}")
    return fmt


# ===========================================================================
# 5. STRATEGIC INSIGHTS
# ===========================================================================
def print_section(title: str) -> None:
    bar = "=" * 78
    print(f"\n{bar}\n{title}\n{bar}")


def strategic_insights(
    master: pd.DataFrame,
    liquidity: dict[str, dict],
    sigma_forecasts: dict[str, float],
) -> None:
    """Print structured commentary on stability, liquidity, and regime stress."""
    stocks = master["Stock"].unique()

    # ---- Insight 1: stability of estimates across methods ----------------
    print_section("INSIGHT 1 - STABILITY OF VaR ACROSS METHODOLOGIES")
    for stock in stocks:
        sub = master[master["Stock"] == stock]
        v99 = sub["99% VaR (Rs)"]
        spread = v99.max() - v99.min()
        rel = spread / v99.mean() * 100
        print(f" {stock}: 99% VaR ranges Rs {v99.min():,.0f} - Rs {v99.max():,.0f} "
              f"across methods (spread = {rel:.1f}% of mean).")
        
        gmc = sub.loc[sub['VaR Method'] == 'GARCH-MC', '99% VaR (Rs)'].iloc[0]
        par = sub.loc[sub['VaR Method'] == 'Standard Parametric', '99% VaR (Rs)'].iloc[0]
        rel_gp = (gmc - par) / par * 100
        higher = "above" if gmc > par else "below"
        print(f"   GARCH-MC sits {abs(rel_gp):.1f}% {higher} the static parametric figure "
              f"(1-day GARCH sigma = {sigma_forecasts[stock]:.4f}), reflecting the "
              f"current conditional-volatility state vs. the full-sample average.")

    # ---- Insight 2: liquid vs. illiquid (referencing Part A proxies) -----
    print_section("INSIGHT 2 - LIQUID (HDFC) vs. ILLIQUID (NESTLE) RISK")
    for stock in stocks:
        liq = liquidity[stock]
        print(f" {stock}: avg turnover = Rs {liq['turnover']:,.0f}/day | "
              f"avg Amihud (illiquidity) = {liq['amihud']:.4f}")
        
    if len(stocks) == 2:
        a, b = stocks
        turn_ratio = liquidity[a]["turnover"] / max(liquidity[b]["turnover"], 1e-9)
        amih_ratio = liquidity[b]["amihud"] / max(liquidity[a]["amihud"], 1e-9)
        print(f"\n {a} trades ~{turn_ratio:.1f}x the daily value of {b}, and {b}'s "
              f"Amihud illiquidity is ~{amih_ratio:.1f}x higher.")
        
    print(" CAVEAT: return-based VaR measures only price risk. Two books can show")
    print(" similar VaR yet face very different LIQUIDATION risk - unwinding the")
    print(" illiquid name moves the market (high Amihud, Part A) and incurs the")
    print(" slippage seen in Part C. True risk for Nestle is understated by VaR alone;")
    print(" a liquidity-adjusted VaR (add an Amihud-scaled cost term) is the honest view.")

    # ---- Insight 3: regime deterioration ---------------------------------
    print_section("INSIGHT 3 - HIGH-VOL REGIME vs. NORMAL REGIME")
    for stock in stocks:
        sub = master[master["Stock"] == stock].set_index("VaR Method")
        for cl_col in ("95% VaR (Rs)", "99% VaR (Rs)"):
            normal = sub.loc["Normal Regime", cl_col]
            high = sub.loc["High-Vol Regime", cl_col]
            mult = high / normal if normal else np.nan
            print(f" {stock} {cl_col}: Normal Rs {normal:,.0f} -> "
                  f"High-Vol Rs {high:,.0f} ({mult:.2f}x).")
            
    print("\n Risk is strongly state-dependent: conditioning on the top-quartile")
    print(" volatility days inflates VaR well above the blended full-sample estimate,")
    print(" which is exactly the tail a static parametric model under-prices.")


# ===========================================================================
# EXPORTS
# ===========================================================================
def export_outputs(master: pd.DataFrame, formatted: pd.DataFrame) -> None:
    """Emit numeric CSV + formatted LaTeX (Excel attempted opportunistically)."""
    master.to_csv("part_d_var_master.csv", index=False)
    print("[EXPORT] CSV -> part_d_var_master.csv")

    with open("part_d_var_master.tex", "w", encoding="utf-8") as fh:
        fh.write(formatted.to_latex(index=False,
                                    caption="Consolidated 1-Day VaR by Method",
                                    label="tab:var_master"))
    print("[EXPORT] LaTeX -> part_d_var_master.tex")

    try:
        with pd.ExcelWriter("part_d_var_master.xlsx") as writer:
            master.to_excel(writer, sheet_name="VaR_Numeric", index=False)
            formatted.to_excel(writer, sheet_name="VaR_Formatted", index=False)
        print("[EXPORT] Excel -> part_d_var_master.xlsx")
    except Exception as exc:
        print(f"[EXPORT] Excel skipped ({type(exc).__name__}); CSV/LaTeX cover it.")


# ===========================================================================
# MAIN ORCHESTRATION
# ===========================================================================
def main() -> None:
    print_section("PART D: RISK MEASUREMENT & STRESS ANALYSIS")
    print(f"Eval date {EVAL_DATE:%d-%b-%Y} | PV = Rs {PORTFOLIO_VALUE:,.0f}/stock | "
          f"MC sims = {N_SIMULATIONS:,} | dist = {MC_DISTRIBUTION}")

    results: dict[str, dict] = {}
    liquidity: dict[str, dict] = {}
    sigma_forecasts: dict[str, float] = {}

    for stock, symbol in TICKER_MAP.items():
        print(f"\n[INFO] Processing {stock} ({symbol}) ...")
        metrics = load_history(stock, symbol)
        returns = metrics["LogReturn"].dropna()

        # 1. Standard parametric VaR (full 6-month sample).
        std_var = parametric_var(returns)

        # 2. Regime-based VaR.
        normal_ret, high_ret, thr = split_by_regime(metrics)
        normal_var = parametric_var(normal_ret)
        high_var = parametric_var(high_ret)
        print(f"   Regime split @ rolling-vol threshold {thr:.4f}: "
              f"{len(normal_ret)} normal / {len(high_ret)} high-vol days.")

        # 3. GARCH(1,1) + Monte Carlo VaR.
        mc_var, sigma_1d = garch_monte_carlo_var(returns)
        sigma_forecasts[stock] = sigma_1d

        results[stock] = {
            "Standard Parametric": std_var,
            "Normal Regime": normal_var,
            "High-Vol Regime": high_var,
            "GARCH-MC": mc_var,
        }
        liquidity[stock] = {
            "turnover": float(metrics["TurnoverValue"].mean()),
            "amihud": float(metrics["Amihud"].mean()),
        }

    # 4. Master table + formatting.
    master = build_master_table(results)
    formatted = format_master_table(master)

    print_section("MASTER VaR TABLE (1-day, per Rs 1,000,000 portfolio)")
    print(formatted.to_string(index=False))

    # 5. Strategic insights.
    strategic_insights(master, liquidity, sigma_forecasts)

    export_outputs(master, formatted)
    print_section("DONE")


if __name__ == "__main__":
    main()