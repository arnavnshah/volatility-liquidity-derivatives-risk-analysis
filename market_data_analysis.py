from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns


# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
TICKERS: dict[str, str] = {
    "HDFC Bank": "HDFCBANK.NS",       # liquid / heavily traded
    "Nestle India": "NESTLEIND.NS",   # relatively illiquid
}

PERIOD = "6mo"             # lookback window (yfinance period string)
INTERVAL = "1d"            # daily bars
ROLLING_WINDOW = 20        # trading days for rolling volatility
TRADING_DAYS = 252         # annualization factor (sqrt scaling)
EPSILON = 1e-12            # guards against division-by-zero in Amihud
AMIHUD_SCALE = 1e9         # scaling so raw Amihud values are human-readable
PLOT_DPI = 300             # high-resolution export for executive reports

# Metric columns used throughout for stats / correlation / export
METRIC_COLUMNS = ["LogReturn", "RollingVol", "TurnoverValue", "Amihud"]

sns.set_theme(style="whitegrid", context="talk")


# ---------------------------------------------------------------------------
# 1. DATA SOURCING & SETUP
# ---------------------------------------------------------------------------
def fetch_data(ticker: str, period: str = PERIOD, interval: str = INTERVAL) -> pd.DataFrame:
    """Fetch and clean daily OHLCV data for a single ticker via yfinance.
    
    Parameters
    ----------
    ticker : str
        Yahoo Finance ticker symbol (e.g. "HDFCBANK.NS").
    period : str
        Lookback period accepted by yfinance (default "6mo").
    interval : str
        Bar interval (default "1d").

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame indexed by date with columns
        [Open, High, Low, Close, Volume]. auto_adjust=True is used so that
        'Close' is corporate-action adjusted (correct basis for returns).
    """
    df = yf.download(
        ticker,
        period=period,
        interval=interval,
        auto_adjust=True,  # split/dividend-adjusted Close -> clean returns
        progress=False,
    )

    if df is None or df.empty:
        raise ValueError(f"No data returned for ticker '{ticker}'. Check symbol/connectivity.")

    # yfinance may return MultiIndex columns (('Close', 'TICKER')); flatten them.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Keep the standard OHLCV set and drop rows with missing prices.
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df = df.dropna(subset=["Close"])  # cannot compute returns without price
    df = df[~df.index.duplicated(keep="last")]  # guard against duplicate timestamps
    df = df.sort_index()

    return df


# ---------------------------------------------------------------------------
# 2. QUANTITATIVE CALCULATIONS
# ---------------------------------------------------------------------------
def calculate_metrics(
    df: pd.DataFrame,
    window: int = ROLLING_WINDOW,
    trading_days: int = TRADING_DAYS,
    epsilon: float = EPSILON,
    amihud_scale: float = AMIHUD_SCALE,
) -> pd.DataFrame:
    """Compute return, volatility and liquidity metrics for one stock.

    Formulas
    --------
    Daily log return: r_t = ln(P_t / P_{t-1})
    Rolling realized volatility: sigma_t = std(r over `window`) * sqrt(252)
    Turnover value (Proxy 1): TV_t = Volume_t * Close_t
    Amihud illiquidity (Proxy 2): A_t = |r_t| / TV_t (then scaled for display)

    Returns
    -------
    pd.DataFrame
        Original OHLCV plus LogReturn, RollingVol, TurnoverValue, Amihud.
    """
    out = df.copy()

    # --- Daily log returns: ln(P_t / P_{t-1}) -----------------------------
    out["LogReturn"] = np.log(out["Close"] / out["Close"].shift(1))

    # --- 20-day rolling realized volatility, annualized -------------------
    # Annualization scales the daily std by sqrt(252) trading days.
    out["RollingVol"] = (
        out["LogReturn"].rolling(window=window).std() * np.sqrt(trading_days)
    )

    # --- Liquidity Proxy 1: Daily Turnover Value (Volume * Close) ----------
    out["TurnoverValue"] = out["Volume"] * out["Close"]

    # --- Liquidity Proxy 2: Amihud Illiquidity ----------------------------
    # A_t = |r_t| / (Volume_t * Close_t). High Amihud => price moves a lot
    # per unit of traded value => illiquid. Days with zero turnover are set
    # to NaN to avoid a meaningless / undefined illiquidity reading.
    safe_turnover = out["TurnoverValue"].where(out["TurnoverValue"] > 0)
    out["Amihud"] = (
        out["LogReturn"].abs() / (safe_turnover + epsilon)
    ) * amihud_scale

    # Drop the first row (NaN return) so downstream stats are clean.
    out = out.dropna(subset=["LogReturn"])

    return out


def build_metrics_for_all(tickers: dict[str, str]) -> dict[str, pd.DataFrame]:
    """Fetch + compute metrics for every ticker, returning a name->DataFrame map."""
    metrics: dict[str, pd.DataFrame] = {}
    for name, symbol in tickers.items():
        print(f"[INFO] Fetching {name} ({symbol}) ...")
        raw = fetch_data(symbol)
        metrics[name] = calculate_metrics(raw)
        print(f" -> {len(metrics[name])} trading days processed.")
    return metrics


# ---------------------------------------------------------------------------
# 3. DATA VISUALIZATION
# ---------------------------------------------------------------------------
def plot_vol_vs_amihud(
    metrics: dict[str, pd.DataFrame],
    outfile: str = "plot1_volatility_vs_amihud.png",
) -> None:
    """Dual-axis time series: Rolling Volatility (left) vs. Amihud (right).

    One stacked panel per stock so each can be read on its own scale.
    """
    n = len(metrics)
    fig, axes = plt.subplots(n, 1, figsize=(14, 5.5 * n), dpi=PLOT_DPI)
    if n == 1:
        axes = [axes]

    vol_color, amihud_color = "#1f3b73", "#c0392b"

    for ax, (name, m) in zip(axes, metrics.items()):
        ax2 = ax.twinx()

        line_vol = ax.plot(
            m.index, m["RollingVol"],
            color=vol_color, linewidth=2.0,
            label="20D Rolling Volatility (annualized)",
        )
        line_amh = ax2.plot(
            m.index, m["Amihud"],
            color=amihud_color, linewidth=1.6, alpha=0.85,
            label="Amihud Illiquidity (scaled)",
        )

        ax.set_title(f"{name}: Rolling Volatility vs. Amihud Illiquidity",
                     fontsize=15, fontweight="bold")
        ax.set_ylabel("Annualized Volatility", color=vol_color)
        ax2.set_ylabel("Amihud Illiquidity", color=amihud_color)
        ax.tick_params(axis="y", labelcolor=vol_color)
        ax2.tick_params(axis="y", labelcolor=amihud_color)
        ax2.grid(False)  # avoid double grid clutter on the secondary axis

        # Combined legend across both axes.
        lines = line_vol + line_amh
        ax.legend(lines, [ln.get_label() for ln in lines],
                  loc="upper left", fontsize=11, framealpha=0.9)

        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b"))

    fig.tight_layout()
    fig.savefig(outfile, dpi=PLOT_DPI, bbox_inches="tight")
    print(f"[PLOT] Saved dual-axis volatility/Amihud chart -> {outfile}")
    plt.close(fig)


def plot_volatility_clustering(
    metrics: dict[str, pd.DataFrame],
    outfile: str = "plot2_volatility_clustering.png",
) -> None:
    """Compare volatility clustering via absolute log returns for both stocks.

    Periods of large |returns| tend to cluster together (volatility
    clustering). Plotting them side by side highlights regime differences
    between the liquid and illiquid names.
    """
    n = len(metrics)
    fig, axes = plt.subplots(n, 1, figsize=(14, 4.5 * n), dpi=PLOT_DPI, sharex=True)
    if n == 1:
        axes = [axes]

    palette = sns.color_palette("rocket", n_colors=n)

    for ax, color, (name, m) in zip(axes, palette, metrics.items()):
        abs_ret = m["LogReturn"].abs()
        ax.fill_between(m.index, abs_ret, color=color, alpha=0.35)
        ax.plot(m.index, abs_ret, color=color, linewidth=1.2)
        ax.set_title(f"{name}: Absolute Daily Log Returns (Volatility Clustering)",
                     fontsize=14, fontweight="bold")
        ax.set_ylabel("|Log Return|")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b"))

    axes[-1].set_xlabel("Date")
    fig.tight_layout()
    fig.savefig(outfile, dpi=PLOT_DPI, bbox_inches="tight")
    print(f"[PLOT] Saved volatility-clustering comparison -> {outfile}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4. STATISTICAL DELIVERABLES
# ---------------------------------------------------------------------------
def compute_summary_stats(
    metrics: pd.DataFrame,
    columns: list[str] = METRIC_COLUMNS,
) -> pd.DataFrame:
    """Build a tidy summary-statistics table for the chosen metric columns.

    Rows = metrics (LogReturn, RollingVol, TurnoverValue, Amihud)
    Columns = Mean, Median, Std Dev, Skewness, Kurtosis, Min, Max
    """
    rows = {}
    for col in columns:
        series = metrics[col].dropna()
        rows[col] = {
            "Mean": series.mean(),
            "Median": series.median(),
            "Std Dev": series.std(),
            "Skewness": series.skew(),
            "Kurtosis": series.kurt(),  # excess kurtosis (Fisher)
            "Min": series.min(),
            "Max": series.max(),
        }
    table = pd.DataFrame(rows).T
    return table[["Mean", "Median", "Std Dev", "Skewness", "Kurtosis", "Min", "Max"]]


def compute_correlations(metrics: pd.DataFrame) -> pd.DataFrame:
    """Pearson correlation matrix among Volatility and the two liquidity proxies."""
    sub = metrics[["RollingVol", "TurnoverValue", "Amihud"]].dropna()
    return sub.corr(method="pearson")


def export_to_excel(
    summaries: dict[str, pd.DataFrame],
    correlations: dict[str, pd.DataFrame],
    outfile: str = "part_a_results.xlsx",
) -> None:
    """Write summary-stats and correlation tables to a multi-sheet Excel file."""
    with pd.ExcelWriter(outfile, engine="openpyxl") as writer:
        for name, table in summaries.items():
            sheet = f"Summary_{name[:24]}"
            table.to_excel(writer, sheet_name=sheet)
        for name, corr in correlations.items():
            sheet = f"Corr_{name[:27]}"
            corr.to_excel(writer, sheet_name=sheet)
    print(f"[EXPORT] Wrote summary + correlation tables -> {outfile}")


# ---------------------------------------------------------------------------
# CONSOLE REPORTING
# ---------------------------------------------------------------------------
def print_section(title: str) -> None:
    bar = "=" * 70
    print(f"\n{bar}\n{title}\n{bar}")


def report_findings(
    summaries: dict[str, pd.DataFrame],
    correlations: dict[str, pd.DataFrame],
    metrics: dict[str, pd.DataFrame],
) -> None:
    """Print clean text summaries and interpret the key correlation results."""
    print_section("SUMMARY STATISTICS")
    for name, table in summaries.items():
        print(f"\n--- {name} ---")
        # Friendly formatting; turnover/Amihud can be large/small respectively.
        print(table.to_string(float_format=lambda x: f"{x:,.6g}"))

    print_section("PEARSON CORRELATION: Volatility vs. Liquidity Proxies")
    for name, corr in correlations.items():
        print(f"\n--- {name} ---")
        print(corr.to_string(float_format=lambda x: f"{x:+.3f}"))

        vol_turn = corr.loc["RollingVol", "TurnoverValue"]
        vol_amih = corr.loc["RollingVol", "Amihud"]
        print(f"  > Corr(Volatility, Turnover Value) = {vol_turn:+.3f}")
        print(f"  > Corr(Volatility, Amihud Illiq.)  = {vol_amih:+.3f}")
        amih_dir = "rises" if vol_amih > 0 else "falls"
        print(f"    Interpretation: as volatility increases, illiquidity {amih_dir} "
              f"(Amihud relationship is {'positive' if vol_amih > 0 else 'negative'}).")

    print_section("LIQUIDITY COMPARISON (period averages)")
    comp = pd.DataFrame({
        name: {
            "Avg Turnover Value": m["TurnoverValue"].mean(),
            "Avg Amihud Illiquidity": m["Amihud"].mean(),
            "Avg Annualized Vol": m["RollingVol"].mean(),
        }
        for name, m in metrics.items()
    }).T
    print(comp.to_string(float_format=lambda x: f"{x:,.6g}"))

    # Identify the more liquid name (higher turnover, lower Amihud).
    most_liquid = comp["Avg Turnover Value"].idxmax()
    least_liquid = comp["Avg Amihud Illiquidity"].idxmax()
    print(f"\n  > Highest average turnover (most liquid): {most_liquid}")
    print(f"  > Highest average Amihud (least liquid):  {least_liquid}")


# ---------------------------------------------------------------------------
# MAIN ORCHESTRATION
# ---------------------------------------------------------------------------
def main() -> None:
    print_section("PART A: MARKET DATA, RETURNS & LIQUIDITY ANALYSIS")

    # 1-2. Fetch data and compute all metrics.
    metrics = build_metrics_for_all(TICKERS)

    # 3. Visualizations.
    plot_vol_vs_amihud(metrics)
    plot_volatility_clustering(metrics)

    # 4. Statistical deliverables.
    summaries = {name: compute_summary_stats(m) for name, m in metrics.items()}
    correlations = {name: compute_correlations(m) for name, m in metrics.items()}

    # Console reporting + Excel export.
    report_findings(summaries, correlations, metrics)
    export_to_excel(summaries, correlations)

    print_section("DONE")
    print("Generated files:")
    print("  - plot1_volatility_vs_amihud.png")
    print("  - plot2_volatility_clustering.png")
    print("  - part_a_results.xlsx")


if __name__ == "__main__":
    main()
