"""
Risk Management Project - Part C
Greeks, Portfolio Construction & Hedging
========================================

Consumes structural inputs from Parts A & B (as of 29-Jun-2026) for:
- HDFC Bank (HDFCBANK.NS) -> liquid benchmark
- Nestle India (NESTLEIND.NS) -> illiquid benchmark

Sections
--------
1. Analytical BSM Greeks engine (Delta, Gamma, Vega).
2. Identical 3-leg options portfolio per stock + baseline delta hedge.
3. Liquidity-adjusted hedging (haircut + slippage from Part A proxies).
4. Spot x Volatility stress grid: unhedged / delta-hedged / liq-adjusted PnL.
5. Report tables + strategic printout.

Dependencies: NumPy, SciPy, Pandas only.
(Excel export attempted opportunistically; CSV + LaTeX always emitted.)

NOTE ON INPUTS
--------------
BASE_VOL and ILLIQUIDITY_INDEX below are the hand-off points from Parts A & B.
Replace BASE_VOL with the Part B 29-day Historical or GARCH volatility, and
ILLIQUIDITY_INDEX with the relative Amihud (illiquid / liquid) ratio from Part A.
Defaults are representative illustrative values.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.stats import norm


# ===========================================================================
# CONFIGURATION (hand-off from Parts A & B)
# ===========================================================================
EVAL_DATE = "2026-06-29"
RISK_FREE_RATE = 0.0675  # r, annualized
DIVIDEND_YIELD = 0.0     # q
CONTRACT_MULTIPLIER = 100 # share-equivalents per contract (set NSE lot size if desired)
T_29 = 29 / 365.0        # 29-day expiry in years (calendar/365)

SPOT = {
    "HDFC Bank": 797.05,
    "Nestle India": 1390.40,
}

# --- From Part B: 29-day annualized volatility input (Historical or GARCH) --
BASE_VOL = {
    "HDFC Bank": 0.16,    # <- replace with Part B vol
    "Nestle India": 0.15, # <- replace with Part B vol
}

# --- From Part A: relative illiquidity index (mean Amihud_stock / Amihud_liquid)
# 1.0 == most liquid reference; larger == more illiquid.
ILLIQUIDITY_INDEX = {
    "HDFC Bank": 1.0,     # liquid reference
    "Nestle India": 8.0,  # ~8x more illiquid (illustrative, from Part A Amihud)
}

# --- Liquidity-haircut & slippage model parameters -------------------------
HAIRCUT_BASE = 0.005      # minimum execution shortfall even for liquid names
HAIRCUT_SENS = 0.04       # extra haircut per unit of illiquidity index
HAIRCUT_CAP = 0.30        # maximum fraction of the hedge left unexecuted
SLIPPAGE_BPS_BASE = 2.0   # bps of executed notional at illiquidity index = 1

# --- 29-day contracts (Strike, Type, Market Price) from Part B snapshot -----
CONTRACTS = {
    "HDFC Bank": {
        "ATM_Call": (800.00, "Call", 23.70),
        "OTM_Call": (805.00, "Call", 21.15),
        "OTM_Put": (760.00, "Put", 6.75),
    },
    "Nestle India": {
        "ATM_Call": (1390.00, "Call", 43.95),
        "OTM_Call": (1420.00, "Call", 29.50),
        "OTM_Put": (1330.00, "Put", 13.70),
    },
}

# Identical portfolio structure for both stocks (quantity in contracts; +long/-short)
PORTFOLIO = {
    "ATM_Call": +10, # long 10 ATM calls
    "OTM_Call": -5,  # short 5 OTM calls
    "OTM_Put": +5,   # long 5 OTM puts
}

# --- Stress grid -----------------------------------------------------------
SPOT_SHOCKS = [-0.02, -0.01, 0.0, 0.01, 0.02] # relative spot moves
VOL_SHOCKS = [-0.20, 0.0, 0.20]               # relative vol moves
# (spec asks for -20% and +20%; 0% added as a base-vol reference column)


# ===========================================================================
# 1. ANALYTICAL BSM PRICING & GREEKS ENGINE
# ===========================================================================
def _d1_d2(S, K, T, r, sigma, q):
    """Return BSM d1, d2."""
    sqrt_t = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    return d1, d2


def bsm_price(S, K, T, r, sigma, option_type, q=DIVIDEND_YIELD):
    """European BSM price (call/put)."""
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    disc_r, disc_q = np.exp(-r * T), np.exp(-q * T)
    if option_type.lower() == "call":
        return S * disc_q * norm.cdf(d1) - K * disc_r * norm.cdf(d2)
    return K * disc_r * norm.cdf(-d2) - S * disc_q * norm.cdf(-d1)


def bsm_greeks(S, K, T, r, sigma, option_type, q=DIVIDEND_YIELD):
    """Compute per-share BSM Greeks for a European option.

    Delta = dV/dS Call: e^{-qT} N(d1) Put: e^{-qT}(N(d1)-1)
    Gamma = d^2V/dS^2 = e^{-qT} phi(d1) / (S sigma sqrt(T)) [same for C/P]
    Vega = dV/dsigma = S e^{-qT} phi(d1) sqrt(T) [per 1.00 vol]

    Returns a dict; Vega is also reported per 1% vol move for reporting.
    """
    d1, d2 = _d1_d2(S, K, T, r, sigma, q)
    pdf_d1 = norm.pdf(d1)
    disc_q = np.exp(-q * T)
    sqrt_t = np.sqrt(T)

    if option_type.lower() == "call":
        delta = disc_q * norm.cdf(d1)
    else:
        delta = disc_q * (norm.cdf(d1) - 1.0)

    gamma = disc_q * pdf_d1 / (S * sigma * sqrt_t)
    vega = S * disc_q * pdf_d1 * sqrt_t # per 1.00 change in sigma

    return {
        "price": bsm_price(S, K, T, r, sigma, option_type, q),
        "delta": delta,
        "gamma": gamma,
        "vega": vega,
        "vega_1pct": vega / 100.0, # per 1% vol move (report-friendly)
    }


# ===========================================================================
# 2. PORTFOLIO CONSTRUCTION & BASELINE DELTA HEDGE
# ===========================================================================
@dataclass
class Leg:
    """A single option position in the portfolio."""
    name: str
    strike: float
    option_type: str
    market_price: float
    quantity: int # contracts (+long / -short)


def build_portfolio(stock: str) -> list[Leg]:
    """Assemble the identical 3-leg portfolio for a given stock."""
    legs = []
    for name, qty in PORTFOLIO.items():
        strike, opt_type, mkt = CONTRACTS[stock][name]
        legs.append(Leg(name, strike, opt_type, mkt, qty))
    return legs


def portfolio_greeks_table(stock: str, legs: list[Leg]) -> tuple[pd.DataFrame, dict]:
    """Build the per-leg Greeks table and aggregate (position-weighted) net Greeks.

    Position-weighted Greek = quantity * CONTRACT_MULTIPLIER * per_share_Greek.
    """
    S, sigma = SPOT[stock], BASE_VOL[stock]
    rows = []
    net = {"delta": 0.0, "gamma": 0.0, "vega": 0.0}

    for leg in legs:
        g = bsm_greeks(S, leg.strike, T_29, RISK_FREE_RATE, sigma, leg.option_type)
        weight = leg.quantity * CONTRACT_MULTIPLIER
        pos_delta = weight * g["delta"]
        pos_gamma = weight * g["gamma"]
        pos_vega = weight * g["vega_1pct"]

        net["delta"] += pos_delta
        net["gamma"] += pos_gamma
        net["vega"] += pos_vega

        rows.append({
            "Position": leg.name,
            "Type": leg.option_type,
            "Strike": leg.strike,
            "Qty (contracts)": leg.quantity,
            "Mult": CONTRACT_MULTIPLIER,
            "Mkt Price": leg.market_price,
            "BSM Price": round(g["price"], 2),
            "Delta": round(g["delta"], 4),
            "Gamma": round(g["gamma"], 6),
            "Vega (1%)": round(g["vega_1pct"], 4),
            "Pos Delta": round(pos_delta, 2),
            "Pos Gamma": round(pos_gamma, 4),
            "Pos Vega (1%)": round(pos_vega, 2),
        })

    table = pd.DataFrame(rows)
    # Aggregate net-Greeks summary row.
    agg_row = {
        "Position": "NET PORTFOLIO", "Type": "", "Strike": np.nan,
        "Qty (contracts)": np.nan, "Mult": np.nan, "Mkt Price": np.nan,
        "BSM Price": np.nan, "Delta": np.nan, "Gamma": np.nan, "Vega (1%)": np.nan,
        "Pos Delta": round(net["delta"], 2),
        "Pos Gamma": round(net["gamma"], 4),
        "Pos Vega (1%)": round(net["vega"], 2),
    }
    table = pd.concat([table, pd.DataFrame([agg_row])], ignore_index=True)
    return table, net


def baseline_delta_hedge(net_delta_shares: float) -> float:
    """Shares of the underlying needed to neutralize aggregate delta.

    To make the book delta-neutral we hold the opposite of the portfolio delta:
    hedge_shares = -net_delta_shares
    (positive => buy stock; negative => short stock).
    """
    return -net_delta_shares


# ===========================================================================
# 3. LIQUIDITY-ADJUSTED HEDGING CONSTRAINTS
# ===========================================================================
@dataclass
class HedgePlan:
    stock: str
    net_delta: float         # share-equivalents (pre-hedge)
    baseline_shares: float   # theoretical, frictionless
    haircut: float           # fraction of hedge left unexecuted
    executed_shares: float   # actually traded after liquidity constraint
    residual_delta: float    # delta still exposed after liq-adjusted hedge
    slippage_bps: float
    slippage_cost: float     # one-off currency cost of executing the hedge


def liquidity_haircut(illiquidity_index: float) -> float:
    """Fraction of the theoretical hedge that cannot be executed cleanly.

    Linear in the Part A illiquidity index, floored at HAIRCUT_BASE and capped
    at HAIRCUT_CAP. Liquid names ~ HAIRCUT_BASE; illiquid names approach the cap.
    """
    raw = HAIRCUT_BASE + HAIRCUT_SENS * (illiquidity_index - 1.0)
    return float(np.clip(raw, 0.0, HAIRCUT_CAP))


def build_hedge_plan(stock: str, net_delta_shares: float) -> HedgePlan:
    """Construct baseline + liquidity-adjusted hedge execution details."""
    S = SPOT[stock]
    illiq = ILLIQUIDITY_INDEX[stock]

    baseline = baseline_delta_hedge(net_delta_shares)  # frictionless hedge
    haircut = liquidity_haircut(illiq)
    executed = baseline * (1.0 - haircut)  # under-execution on illiquid names
    residual = net_delta_shares + executed # leftover delta (= net_delta * haircut)

    # Slippage scales with illiquidity and executed notional.
    slippage_bps = SLIPPAGE_BPS_BASE * illiq
    executed_notional = abs(executed) * S
    slippage_cost = (slippage_bps / 1e4) * executed_notional

    return HedgePlan(
        stock=stock,
        net_delta=net_delta_shares,
        baseline_shares=baseline,
        haircut=haircut,
        executed_shares=executed,
        residual_delta=residual,
        slippage_bps=slippage_bps,
        slippage_cost=slippage_cost,
    )


# ===========================================================================
# 4. SHOCK SIMULATION & STRESS-TESTING MATRIX
# ===========================================================================
def _portfolio_value(stock: str, legs: list[Leg], S: float, sigma: float) -> float:
    """Mark-to-market value of the options book (per CONTRACT_MULTIPLIER)."""
    total = 0.0
    for leg in legs:
        price = bsm_price(S, leg.strike, T_29, RISK_FREE_RATE, sigma, leg.option_type)
        total += leg.quantity * CONTRACT_MULTIPLIER * price
    return total


def run_stress_matrix(stock: str, legs: list[Leg], hedge: HedgePlan) -> dict:
    """Full-revaluation PnL across the Spot x Vol grid for three hedge regimes.

    PnL definitions (instantaneous shock, time held fixed):
    dV_options = V(S', sigma') - V(S0, sigma0)
    Unhedged PnL = dV_options
    Delta-hedged PnL = dV_options + baseline_shares * (S' - S0)
    Liq-adj PnL = dV_options + executed_shares * (S' - S0) - slippage_cost
    """
    S0, sigma0 = SPOT[stock], BASE_VOL[stock]
    base_value = _portfolio_value(stock, legs, S0, sigma0)

    grids = {"Unhedged": {}, "Delta-Hedged": {}, "Liquidity-Adjusted": {}}

    for ds in SPOT_SHOCKS:
        S_new = S0 * (1.0 + ds)
        for dv in VOL_SHOCKS:
            sigma_new = sigma0 * (1.0 + dv)
            d_options = _portfolio_value(stock, legs, S_new, sigma_new) - base_value
            d_spot = S_new - S0

            unhedged = d_options
            delta_hedged = d_options + hedge.baseline_shares * d_spot
            liq_hedged = d_options + hedge.executed_shares * d_spot - hedge.slippage_cost

            grids["Unhedged"][(ds, dv)] = unhedged
            grids["Delta-Hedged"][(ds, dv)] = delta_hedged
            grids["Liquidity-Adjusted"][(ds, dv)] = liq_hedged

    # Convert each regime's dict into a spot(rows) x vol(cols) pivot table.
    pivots = {}
    for regime, data in grids.items():
        df = pd.Series(data).unstack()  # index=spot, cols=vol
        df.index = [f"{s:+.0%}" for s in df.index]
        df.columns = [f"{v:+.0%}" for v in df.columns]
        df.index.name = "Spot Shock"
        df.columns.name = "Vol Shock"
        pivots[regime] = df.round(2)
    return pivots


def tidy_stress_frame(stock: str, pivots: dict) -> pd.DataFrame:
    """Flatten the three pivots into one tidy long DataFrame for export."""
    frames = []
    for regime, df in pivots.items():
        long = df.stack().rename("PnL").reset_index()
        long.insert(0, "Stock", stock)
        long.insert(1, "Hedge Regime", regime)
        frames.append(long)
    return pd.concat(frames, ignore_index=True)


# ===========================================================================
# REPORTING HELPERS
# ===========================================================================
def print_section(title: str) -> None:
    bar = "=" * 78
    print(f"\n{bar}\n{title}\n{bar}")


def hedge_details_table(plans: dict[str, HedgePlan], nets: dict[str, dict]) -> pd.DataFrame:
    """Baseline vs. liquidity-adjusted execution details for both stocks."""
    rows = []
    for stock, plan in plans.items():
        rows.append({
            "Stock": stock,
            "Net Delta (shares)": round(plan.net_delta, 2),
            "Net Gamma": round(nets[stock]["gamma"], 4),
            "Net Vega (1%)": round(nets[stock]["vega"], 2),
            "Baseline Hedge (shares)": round(plan.baseline_shares, 2),
            "Liquidity Haircut": f"{plan.haircut:.1%}",
            "Executed Hedge (shares)": round(plan.executed_shares, 2),
            "Residual Delta (shares)": round(plan.residual_delta, 2),
            "Slippage (bps)": round(plan.slippage_bps, 1),
            "Slippage Cost": round(plan.slippage_cost, 2),
        })
    return pd.DataFrame(rows)


def liquidity_narrative(plans: dict[str, HedgePlan]) -> None:
    """Explain how liquidity frictions alter practical delta-hedge execution."""
    print_section("LIQUIDITY-ADJUSTED HEDGING - NARRATIVE")
    for stock, plan in plans.items():
        print(f" {stock}:")
        print(f"   - Theoretical hedge: {plan.baseline_shares:+.1f} shares to reach delta-neutral.")
        print(f"   - Liquidity haircut of {plan.haircut:.1%} means only "
              f"{plan.executed_shares:+.1f} shares are realistically executed,")
        print(f"     leaving a RESIDUAL delta of {plan.residual_delta:+.1f} shares unhedged.")
        print(f"   - Executing incurs ~{plan.slippage_bps:.1f} bps slippage "
              f"= {plan.slippage_cost:,.2f} in cost.")
    print("\n Interpretation: a frictionless model assumes the entire delta can be")
    print(" offset instantly at the mid price. In practice, the illiquid name's thin")
    print(" turnover (high Amihud) forces partial execution to avoid moving the market,")
    print(" so the book stays directionally exposed AND pays a wider slippage toll.")
    print(" The liquid name (HDFC) executes near-fully at minimal cost; the illiquid")
    print(" name (Nestle) cannot, which is precisely where hedge error accumulates.")


def strategic_printout(
    nets: dict[str, dict],
    plans: dict[str, HedgePlan],
    all_pivots: dict[str, dict],
) -> None:
    """Contrast hedging effectiveness between the liquid and illiquid names."""
    print_section("STRATEGIC INSIGHTS - HEDGE EFFECTIVENESS: HDFC vs. NESTLE")

    for stock in SPOT.keys():
        piv = all_pivots[stock]
        base_vol_col = "+0%"  # the unshocked-vol reference column

        # Gamma signature: delta-hedged PnL at base vol for small vs. large spot moves.
        dh = piv["Delta-Hedged"][base_vol_col]
        small_up = dh.get("+1%", np.nan)
        large_up = dh.get("+2%", np.nan)
        small_dn = dh.get("-1%", np.nan)
        large_dn = dh.get("-2%", np.nan)

        # Dispersion (max-min) of PnL across the whole grid, per regime.
        disp = {r: (piv[r].values.max() - piv[r].values.min()) for r in piv}

        print(f"\n --- {stock} (net gamma = {nets[stock]['gamma']:+.4f}, "
              f"net vega 1% = {nets[stock]['vega']:+.2f}) ---")
        print(f" Gamma signature (delta-hedged PnL @ base vol):")
        print(f"   small move +/-1%: {small_dn:+.2f} / {small_up:+.2f}  "
              f"large move +/-2%: {large_dn:+.2f} / {large_up:+.2f}")
        ratio_up = (large_up / small_up) if small_up not in (0, np.nan) else np.nan
        print(f"   -> residual scales ~quadratically with the move "
              f"(2% PnL / 1% PnL ~= {ratio_up:.1f}x), the hallmark of gamma convexity.")
        print(f" PnL dispersion across grid (max-min):")
        print(f"   Unhedged: {disp['Unhedged']:,.2f} | "
              f"Delta-Hedged: {disp['Delta-Hedged']:,.2f} | "
              f"Liq-Adjusted: {disp['Liquidity-Adjusted']:,.2f}")

    # Cross-asset contrast.
    hdfc, nestle = "HDFC Bank", "Nestle India"
    hdfc_resid = abs(plans[hdfc].residual_delta)
    nestle_resid = abs(plans[nestle].residual_delta)
    print_section("KEY TAKEAWAYS")
    print(f" 1. GAMMA RISK (small vs. large jumps):")
    print(f"    A delta hedge only cancels first-order risk. Both books are NET LONG")
    print(f"    gamma, so the delta-hedged PnL is convex - tiny at +/-1% but materially")
    print(f"    larger at +/-2%. Gamma protects against small moves cheaply yet leaves")
    print(f"    meaningful re-hedging needs when jumps are large.")
    print(f" 2. LIQUIDITY DEGRADATION:")
    print(f"    Residual delta after the liquidity-adjusted hedge is "
          f"{hdfc_resid:.1f} shares for HDFC vs. {nestle_resid:.1f} for Nestle "
          f"({nestle_resid / max(hdfc_resid, 1e-9):.1f}x worse).")
    print(f"    Under severe vol shocks (+/-20%) combined with large spot moves, Nestle's")
    print(f"    un-executed delta and higher slippage make its liquidity-adjusted PnL")
    print(f"    diverge sharply from the clean delta-hedged result - the hedge that looks")
    print(f"    fine on paper underperforms most exactly when stress is greatest.")


# ===========================================================================
# EXPORTS
# ===========================================================================
def export_outputs(
    summary_tables: dict[str, pd.DataFrame],
    hedge_table: pd.DataFrame,
    tidy_stress: pd.DataFrame,
) -> None:
    """Emit CSV + LaTeX always; attempt Excel if a writer engine is available."""
    tidy_stress.to_csv("part_c_stress_matrix.csv", index=False)
    hedge_table.to_csv("part_c_hedge_details.csv", index=False)
    print("[EXPORT] CSV -> part_c_stress_matrix.csv, part_c_hedge_details.csv")

    with open("part_c_hedge_details.tex", "w", encoding="utf-8") as fh:
        fh.write(hedge_table.to_latex(index=False, float_format="%.2f",
                                      caption="Hedge Execution: Baseline vs. Liquidity-Adjusted",
                                      label="tab:hedge_exec"))
    print("[EXPORT] LaTeX -> part_c_hedge_details.tex")

    try:
        with pd.ExcelWriter("part_c_hedging.xlsx") as writer:
            for stock, tbl in summary_tables.items():
                tbl.to_excel(writer, sheet_name=f"Greeks_{stock[:20]}", index=False)
            hedge_table.to_excel(writer, sheet_name="Hedge_Details", index=False)
            tidy_stress.to_excel(writer, sheet_name="Stress_Matrix", index=False)
        print("[EXPORT] Excel -> part_c_hedging.xlsx")
    except Exception as exc:  # openpyxl not guaranteed (stdlib-only constraint)
        print(f"[EXPORT] Excel skipped ({type(exc).__name__}); CSV/LaTeX cover the deliverables.")


# ===========================================================================
# MAIN ORCHESTRATION
# ===========================================================================
def main() -> None:
    print_section("PART C: GREEKS, PORTFOLIO CONSTRUCTION & HEDGING")
    print(f"Eval date {EVAL_DATE} | r={RISK_FREE_RATE:.4f} q={DIVIDEND_YIELD:.2f} "
          f"| T={T_29*365:.0f}d | multiplier={CONTRACT_MULTIPLIER}")

    summary_tables: dict[str, pd.DataFrame] = {}
    nets: dict[str, dict] = {}
    plans: dict[str, HedgePlan] = {}
    all_pivots: dict[str, dict] = {}
    tidy_frames = []

    # --- Per-stock build: Greeks -> hedge plan -> stress grid ----------------
    for stock in SPOT.keys():
        legs = build_portfolio(stock)
        table, net = portfolio_greeks_table(stock, legs)
        plan = build_hedge_plan(stock, net["delta"])
        pivots = run_stress_matrix(stock, legs, plan)

        summary_tables[stock] = table
        nets[stock] = net
        plans[stock] = plan
        all_pivots[stock] = pivots
        tidy_frames.append(tidy_stress_frame(stock, pivots))

    # --- Deliverable 1: Portfolio summary tables -----------------------------
    print_section("PORTFOLIO SUMMARY TABLES (per-leg + net Greeks, pre-hedge)")
    for stock, table in summary_tables.items():
        print(f"\n--- {stock} | Spot={SPOT[stock]:.2f} | Base Vol={BASE_VOL[stock]:.2%} ---")
        print(table.to_string(index=False))

    # --- Deliverable 2: Hedge execution details ------------------------------
    print_section("HEDGE EXECUTION DETAILS (baseline vs. liquidity-adjusted)")
    hedge_table = hedge_details_table(plans, nets)
    print(hedge_table.to_string(index=False))

    # --- Deliverable 3: Liquidity narrative ----------------------------------
    liquidity_narrative(plans)

    # --- Deliverable 4: Scenario matrices ------------------------------------
    print_section("SCENARIO MATRICES - NET PnL ACROSS STRESS GRID")
    for stock in SPOT.keys():
        print(f"\n##### {stock} #####")
        for regime, piv in all_pivots[stock].items():
            print(f"\n [{regime}] (rows = spot shock, cols = vol shock)")
            print(piv.to_string())

    # --- Deliverable 5: Strategic printout -----------------------------------
    strategic_printout(nets, plans, all_pivots)

    # --- Exports -------------------------------------------------------------
    tidy_stress = pd.concat(tidy_frames, ignore_index=True)
    export_outputs(summary_tables, hedge_table, tidy_stress)

    print_section("DONE")


if __name__ == "__main__":
    main()