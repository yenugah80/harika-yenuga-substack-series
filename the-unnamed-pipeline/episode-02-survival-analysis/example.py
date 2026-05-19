"""
Example usage — Survival Analysis
The Unnamed Pipeline, Episode 2

Three test cases:
  1. Kaplan-Meier overall survival curve
  2. Monthly vs annual plan group comparison (log-rank test)
  3. Cox Proportional Hazards model + urgent customer flagging
"""

import numpy as np
import pandas as pd
from pathlib import Path
from survival import (
    fit_kaplan_meier, compare_groups, fit_cox_model,
    predict_individual_survival, flag_urgent_customers,
    median_survival_time, print_report,
)
from visualize import plot_survival_curve, plot_group_comparison, plot_hazard_ratios

Path("output").mkdir(exist_ok=True)


def generate_saas_data(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic SaaS subscription data with covariate-driven churn.

    Churn mechanics (hazard model):
      - Monthly plan:          HR ≈ 2.1  (much higher baseline risk)
      - days_since_login:      HR ≈ 1.03 per day (compound inactivity risk)
      - support_tickets:       HR ≈ 1.15 per ticket (friction signal)
      - features_activated:    HR ≈ 0.88 per feature (engagement protection)
      - team_size > 3:         HR ≈ 0.65 (team accounts more sticky)
    """
    rng = np.random.default_rng(seed)

    plan_type = rng.choice(["monthly", "annual"], size=n, p=[0.6, 0.4])
    days_since_login = rng.integers(0, 61, size=n)
    support_tickets = rng.integers(0, 11, size=n)
    features_activated = rng.integers(1, 13, size=n)
    team_size = rng.integers(1, 21, size=n)

    # Compute individual log-hazard from covariates
    log_hazard = (
        -0.2
        + 0.75 * (plan_type == "monthly").astype(float)
        + 0.03 * days_since_login
        + 0.14 * support_tickets
        - 0.13 * features_activated
        - 0.43 * (team_size > 3).astype(float)
    )
    hazard = np.exp(log_hazard)

    # Exponential survival times from individual hazards
    # T ~ Exponential(hazard)
    survival_times = rng.exponential(scale=1.0 / hazard)
    survival_times = np.clip(survival_times * 365, 1, 730).astype(int)

    # Administrative censoring at 730 days
    # Additional random censoring: ~30% of customers still active
    censor_time = rng.integers(180, 731, size=n)
    churned = (survival_times <= censor_time).astype(int)
    observed_duration = np.minimum(survival_times, censor_time)

    return pd.DataFrame({
        "customer_id": range(1, n + 1),
        "tenure_days": observed_duration,
        "churned": churned,
        "plan_type": plan_type,
        "plan_monthly": (plan_type == "monthly").astype(int),
        "days_since_login": days_since_login,
        "support_tickets": support_tickets,
        "features_activated": features_activated,
        "team_size": team_size,
        "large_team": (team_size > 3).astype(int),
    })


def main():
    print("Generating synthetic SaaS subscription data (n=5,000)...")
    df = generate_saas_data(5000)
    churn_rate = df["churned"].mean()
    print(f"Churn rate in dataset: {churn_rate:.1%}")
    print(f"Median observed tenure: {df['tenure_days'].median():.0f} days\n")

    # ----------------------------------------------------------------
    print("=" * 60)
    print("TEST CASE 1: Kaplan-Meier — overall survival curve")
    print("=" * 60)

    kmf_all = fit_kaplan_meier(df, "tenure_days", "churned", "All customers")
    print_report(kmf_all)
    plot_survival_curve(
        kmf_all,
        title="Kaplan-Meier — SaaS customer survival (n=5,000)",
        save_path="output/km_overall.png",
    )

    # ----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("TEST CASE 2: Monthly vs Annual plan comparison")
    print("=" * 60)

    comparison = compare_groups(df, "tenure_days", "churned", "plan_type")
    p_val = comparison.get("p_value", None)
    if p_val is not None:
        print(f"Log-rank test p-value: {p_val:.6f}")
        print(f"Groups significantly different: {p_val < 0.05}")

    for group, kmf in comparison["fitters"].items():
        med = median_survival_time(kmf)
        print(f"  {group}: median survival = {med:.1f} days")

    plot_group_comparison(
        comparison["fitters"],
        title="Survival by plan type — monthly vs annual",
        p_value=p_val,
        save_path="output/km_plan_comparison.png",
    )

    # ----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("TEST CASE 3: Cox Proportional Hazards model")
    print("=" * 60)

    covariates = [
        "plan_monthly", "days_since_login",
        "support_tickets", "features_activated", "large_team",
    ]

    cph = fit_cox_model(df, "tenure_days", "churned", covariates)
    print_report(kmf_all, cph)

    plot_hazard_ratios(
        cph,
        title="Cox model — hazard ratios by feature",
        save_path="output/cox_hazard_ratios.png",
    )

    # Individual predictions for 5 sample customers
    print("\nIndividual survival predictions (5 sample customers):")
    sample = df[covariates].head(5)
    sf_individual = predict_individual_survival(cph, sample, times=[7, 14, 30, 90])
    print(sf_individual.T.rename(
        columns={7: "7d", 14: "14d", 30: "30d", 90: "90d"}
    ).to_string())

    # Flag urgent customers
    urgent = flag_urgent_customers(
        cph, df[covariates], horizon_days=14, threshold=0.60
    )
    print(f"\nUrgent customers (14-day survival < 60%): {len(urgent)}")
    print(f"Top 5 most urgent:")
    print(urgent.head(5)[["survival_prob_14d", "predicted_median_survival",
                           "plan_monthly", "days_since_login"]].to_string())

    print("\nAll charts saved to output/")


if __name__ == "__main__":
    main()
