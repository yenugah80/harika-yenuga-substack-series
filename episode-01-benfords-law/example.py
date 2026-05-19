"""
Example usage — Benford's Law implementation
The Unnamed Pipeline, Episode 1

Three test cases:
  1. Synthetic vendor invoices (clean — should pass)
  2. Expense claims with $49 threshold gaming (should flag)
  3. Uniform random integers (should fail — negative control)
"""

import numpy as np
import pandas as pd
from benfords import benfords_test, validate_preconditions, print_report, flag_suspicious_digits
from visualize import plot_benfords, plot_deviation


def generate_benford_data(n: int, seed: int = 42) -> pd.Series:
    """
    Generate synthetic data that follows Benford's distribution.
    Uses log-uniform sampling across multiple orders of magnitude.
    """
    rng = np.random.default_rng(seed)
    log_min, log_max = 1.0, 6.0  # $10 to $1,000,000
    log_values = rng.uniform(log_min, log_max, size=n)
    return pd.Series(10 ** log_values, name="amount")


def generate_expense_gaming_data(n: int, seed: int = 42) -> pd.Series:
    """
    Generate expense data with artificial clustering below $49.
    Simulates employees gaming an approval threshold.
    """
    rng = np.random.default_rng(seed)

    # 85% normal expenses (Benford-distributed)
    n_normal = int(n * 0.85)
    normal = 10 ** rng.uniform(0.5, 3.5, size=n_normal)

    # 15% gamed expenses clustered at $40-$48.99
    n_gamed = n - n_normal
    gamed = rng.uniform(40.0, 48.99, size=n_gamed)

    combined = np.concatenate([normal, gamed])
    rng.shuffle(combined)
    return pd.Series(combined, name="amount")


def main():
    print("\n" + "=" * 60)
    print("TEST CASE 1: Vendor invoices (clean dataset)")
    print("=" * 60)

    invoices = generate_benford_data(14000)
    df1 = pd.DataFrame({"amount": invoices})

    pre1 = validate_preconditions(df1, "amount")
    results1, chi2_1, p1, anomalous1 = benfords_test(df1, "amount")
    print_report(results1, chi2_1, p1, anomalous1, pre1)
    plot_benfords(results1, title="Vendor invoices (n=14,000)", save_path="output/invoices_benfords.png")

    print("\n" + "=" * 60)
    print("TEST CASE 2: Expense claims ($49 threshold gaming)")
    print("=" * 60)

    expenses = generate_expense_gaming_data(8200)
    df2 = pd.DataFrame({"amount": expenses})

    pre2 = validate_preconditions(df2, "amount")
    results2, chi2_2, p2, anomalous2 = benfords_test(df2, "amount")
    print_report(results2, chi2_2, p2, anomalous2, pre2)

    suspicious = flag_suspicious_digits(results2)
    if not suspicious.empty:
        print("\nSuspicious digits (deviation > 3pp):")
        print(suspicious.to_string(index=False))

    plot_benfords(results2, title="Expense claims — $49 threshold (n=8,200)", save_path="output/expenses_benfords.png")
    plot_deviation(results2, title="Expense claims — deviation from expected", save_path="output/expenses_deviation.png")

    print("\n" + "=" * 60)
    print("TEST CASE 3: Uniform random integers (negative control)")
    print("=" * 60)

    rng = np.random.default_rng(42)
    uniform = pd.Series(rng.integers(100, 999, size=5000), name="amount")
    df3 = pd.DataFrame({"amount": uniform})

    pre3 = validate_preconditions(df3, "amount")
    results3, chi2_3, p3, anomalous3 = benfords_test(df3, "amount")
    print_report(results3, chi2_3, p3, anomalous3, pre3)
    plot_benfords(results3, title="Uniform random integers (negative control)", save_path="output/uniform_benfords.png")


if __name__ == "__main__":
    main()
