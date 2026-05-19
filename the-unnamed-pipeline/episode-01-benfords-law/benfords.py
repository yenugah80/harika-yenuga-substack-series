"""
Benford's Law — First-Digit Frequency Test
The Unnamed Pipeline, Episode 1

Production-grade implementation for fraud detection,
audit support, and data quality validation.

Author: Hary (Harika Y)
"""

import pandas as pd
import numpy as np
from scipy.stats import chisquare
from typing import Tuple, Optional


# Benford's expected frequencies for digits 1-9
BENFORD_EXPECTED = {
    d: np.log10(1 + 1 / d) for d in range(1, 10)
}


def extract_leading_digit(value: float) -> Optional[int]:
    """Extract the leading (first) non-zero digit from a number."""
    if value == 0 or pd.isna(value):
        return None
    return int(str(f"{abs(value):.10e}")[0])


def benfords_test(
    data: pd.DataFrame,
    column_name: str,
    significance_level: float = 0.05,
) -> Tuple[pd.DataFrame, float, float, bool]:
    """
    Run Benford's Law first-digit test on a numeric column.

    Parameters
    ----------
    data : pd.DataFrame
        Input dataframe containing the numeric column.
    column_name : str
        Name of the column to test.
    significance_level : float
        P-value threshold for flagging anomalies. Default 0.05.

    Returns
    -------
    results : pd.DataFrame
        Per-digit observed vs expected frequencies and deviations.
    chi2 : float
        Chi-square test statistic.
    p_value : float
        P-value from chi-square goodness-of-fit test.
    is_anomalous : bool
        True if p_value < significance_level.
    """
    values = data[column_name].dropna()
    values = values[values != 0].abs()

    if len(values) < 500:
        print(
            f"WARNING: Only {len(values)} observations. "
            f"Chi-square test unreliable below ~500. "
            f"Results should be interpreted with caution."
        )

    leading_digits = values.apply(extract_leading_digit).dropna().astype(int)
    n = len(leading_digits)

    observed_counts = leading_digits.value_counts().sort_index()
    observed_counts = observed_counts.reindex(range(1, 10), fill_value=0)

    expected_freq = [BENFORD_EXPECTED[d] for d in range(1, 10)]
    expected_counts = [freq * n for freq in expected_freq]

    chi2, p_value = chisquare(observed_counts.values, f_exp=expected_counts)

    results = pd.DataFrame({
        "digit": range(1, 10),
        "observed_count": observed_counts.values,
        "observed_pct": (observed_counts.values / n) * 100,
        "expected_pct": [f * 100 for f in expected_freq],
        "deviation_pct": (
            (observed_counts.values / n * 100)
            - np.array([f * 100 for f in expected_freq])
        ),
    })

    is_anomalous = bool(p_value < significance_level)

    return results, chi2, p_value, is_anomalous


def flag_suspicious_digits(
    results: pd.DataFrame,
    deviation_threshold: float = 3.0,
) -> pd.DataFrame:
    """
    Return digits where observed frequency deviates from expected
    by more than the threshold (in percentage points).

    Parameters
    ----------
    results : pd.DataFrame
        Output from benfords_test().
    deviation_threshold : float
        Minimum absolute deviation (percentage points) to flag.

    Returns
    -------
    pd.DataFrame
        Subset of results where |deviation_pct| > threshold.
    """
    return results[
        results["deviation_pct"].abs() > deviation_threshold
    ].copy()


def validate_preconditions(
    data: pd.DataFrame,
    column_name: str,
) -> dict:
    """
    Check whether a dataset meets Benford's Law preconditions.

    Returns a dict with:
      - n_observations: int
      - min_value, max_value: float
      - orders_of_magnitude: float
      - sufficient_observations: bool (>= 500)
      - sufficient_range: bool (>= 2 orders of magnitude)
      - is_valid: bool (both conditions met)
    """
    values = data[column_name].dropna()
    values = values[values != 0].abs()

    n = len(values)
    min_val = values.min()
    max_val = values.max()

    if min_val > 0 and max_val > 0:
        orders = np.log10(max_val / min_val)
    else:
        orders = 0.0

    sufficient_obs = bool(n >= 500)
    sufficient_range = bool(orders >= 2.0)

    return {
        "n_observations": n,
        "min_value": float(min_val),
        "max_value": float(max_val),
        "orders_of_magnitude": round(float(orders), 2),
        "sufficient_observations": sufficient_obs,
        "sufficient_range": sufficient_range,
        "is_valid": bool(sufficient_obs and sufficient_range),
    }


def print_report(
    results: pd.DataFrame,
    chi2: float,
    p_value: float,
    is_anomalous: bool,
    preconditions: Optional[dict] = None,
) -> None:
    """Print a formatted Benford's Law analysis report."""
    print("=" * 60)
    print("BENFORD'S LAW ANALYSIS REPORT")
    print("=" * 60)

    if preconditions:
        print(f"\nPrecondition check:")
        print(f"  Observations:        {preconditions['n_observations']}")
        print(f"  Value range:         {preconditions['min_value']:.2f} — {preconditions['max_value']:.2f}")
        print(f"  Orders of magnitude: {preconditions['orders_of_magnitude']}")
        print(f"  Valid for Benford's: {'YES' if preconditions['is_valid'] else 'NO'}")

        if not preconditions["sufficient_observations"]:
            print("  ⚠ Fewer than 500 observations. Results unreliable.")
        if not preconditions["sufficient_range"]:
            print("  ⚠ Data spans fewer than 2 orders of magnitude.")

    print(f"\nDigit distribution:")
    print(f"{'Digit':>7} {'Observed':>10} {'Expected':>10} {'Deviation':>10}")
    print("-" * 40)
    for _, row in results.iterrows():
        marker = " *" if abs(row["deviation_pct"]) > 3.0 else ""
        print(
            f"{int(row['digit']):>7} "
            f"{row['observed_pct']:>9.1f}% "
            f"{row['expected_pct']:>9.1f}% "
            f"{row['deviation_pct']:>+9.1f}%{marker}"
        )

    print(f"\nChi-square statistic: {chi2:.4f}")
    print(f"P-value:              {p_value:.6f}")
    print(f"Result:               {'ANOMALOUS — investigate further' if is_anomalous else 'CONSISTENT with Benford distribution'}")
    print("=" * 60)
