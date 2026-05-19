"""
Survival Analysis — Customer Churn Timing
The Unnamed Pipeline, Episode 2

Production-grade implementation for modeling time-to-cancellation
in subscription businesses, HR attrition, and insurance lapse.

Author: Hary (Harika Y)
"""

import pandas as pd
import numpy as np
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
from typing import Optional, Tuple


def fit_kaplan_meier(
    data: pd.DataFrame,
    duration_col: str,
    event_col: str,
    label: str = "All customers",
) -> KaplanMeierFitter:
    """
    Fit a Kaplan-Meier survival curve.

    Parameters
    ----------
    data : pd.DataFrame
    duration_col : str
        Time-to-event column (days, weeks, or months).
    event_col : str
        Event indicator: 1 = churned, 0 = censored (still active).
    label : str
        Curve label for plots.

    Returns
    -------
    KaplanMeierFitter
        Fitted estimator. Key attributes:
        - .survival_function_: DataFrame of S(t) at each event time
        - .median_survival_time_: float
        - .confidence_interval_: DataFrame of 95% CI
    """
    kmf = KaplanMeierFitter()
    kmf.fit(
        durations=data[duration_col],
        event_observed=data[event_col],
        label=label,
    )
    return kmf


def compare_groups(
    data: pd.DataFrame,
    duration_col: str,
    event_col: str,
    group_col: str,
) -> dict:
    """
    Fit separate KM curves per group and run a log-rank test
    (when exactly two groups are present).

    Returns
    -------
    dict with keys:
        "fitters"        : {group_value: KaplanMeierFitter}
        "test_statistic" : float  (2-group only)
        "p_value"        : float  (2-group only)
    """
    groups = sorted(data[group_col].unique())
    fitters = {}
    for group in groups:
        mask = data[group_col] == group
        fitters[group] = fit_kaplan_meier(
            data[mask], duration_col, event_col, label=str(group)
        )

    result = {"fitters": fitters}

    if len(groups) == 2:
        g1, g2 = groups
        m1 = data[group_col] == g1
        m2 = data[group_col] == g2
        lr = logrank_test(
            data[m1][duration_col], data[m2][duration_col],
            event_observed_A=data[m1][event_col],
            event_observed_B=data[m2][event_col],
        )
        result["test_statistic"] = float(lr.test_statistic)
        result["p_value"] = float(lr.p_value)

    return result


def fit_cox_model(
    data: pd.DataFrame,
    duration_col: str,
    event_col: str,
    covariate_cols: list,
    penalizer: float = 0.01,
) -> CoxPHFitter:
    """
    Fit a Cox Proportional Hazards model.

    Parameters
    ----------
    data : pd.DataFrame
    duration_col : str
    event_col : str
    covariate_cols : list
        Feature columns to include in the model.
    penalizer : float
        L2 regularization strength. Default 0.01.

    Returns
    -------
    CoxPHFitter
        Fitted model. Key attributes:
        - .summary: DataFrame of coefficients, HRs, CIs, p-values
        - .hazard_ratios_: Series of exp(coef) per feature
        - .concordance_index_: float (analog of AUC for survival)
    """
    model_data = data[[duration_col, event_col] + covariate_cols].dropna()
    cph = CoxPHFitter(penalizer=penalizer)
    cph.fit(model_data, duration_col=duration_col, event_col=event_col)
    return cph


def predict_individual_survival(
    cph: CoxPHFitter,
    customer_data: pd.DataFrame,
    times: Optional[list] = None,
) -> pd.DataFrame:
    """
    Predict personalized survival curves for individual customers.

    Parameters
    ----------
    cph : CoxPHFitter
        Fitted Cox model.
    customer_data : pd.DataFrame
        Rows = customers, columns = covariates used in model.
    times : list, optional
        Time points to evaluate. Default: 1 to 365 days.

    Returns
    -------
    pd.DataFrame
        Rows = time points, columns = customer index.
        Values = survival probability S(t|X) for each customer.
    """
    if times is None:
        times = list(range(1, 366))
    return cph.predict_survival_function(customer_data, times=times)


def flag_urgent_customers(
    cph: CoxPHFitter,
    customer_data: pd.DataFrame,
    horizon_days: int = 14,
    threshold: float = 0.60,
) -> pd.DataFrame:
    """
    Flag customers whose survival probability at horizon_days
    falls below threshold.

    Parameters
    ----------
    cph : CoxPHFitter
        Fitted Cox model.
    customer_data : pd.DataFrame
        Rows = customers, columns = covariates.
    horizon_days : int
        Look-ahead window in days. Default 14.
    threshold : float
        Survival probability cutoff. Customers below this are flagged.

    Returns
    -------
    pd.DataFrame
        Flagged customers sorted by urgency (most urgent first).
        Columns added:
        - f"survival_prob_{horizon_days}d": survival probability at horizon
        - "predicted_median_survival": median survival in days
    """
    sf = predict_individual_survival(cph, customer_data, times=[horizon_days])
    probs = sf.loc[horizon_days]

    result = customer_data.copy()
    result[f"survival_prob_{horizon_days}d"] = probs.values

    median_sf = predict_individual_survival(cph, customer_data)
    medians = []
    for col in median_sf.columns:
        curve = median_sf[col]
        below = curve[curve <= 0.5]
        medians.append(float(below.index[0]) if not below.empty else float("inf"))
    result["predicted_median_survival"] = medians

    flagged = result[result[f"survival_prob_{horizon_days}d"] < threshold].copy()
    return flagged.sort_values(f"survival_prob_{horizon_days}d")


def median_survival_time(kmf: KaplanMeierFitter) -> float:
    """Return the median survival time from a fitted KM estimator."""
    return float(kmf.median_survival_time_)


def print_report(
    kmf: KaplanMeierFitter,
    cph: Optional[CoxPHFitter] = None,
) -> None:
    """Print a formatted survival analysis report to stdout."""
    print("=" * 60)
    print("SURVIVAL ANALYSIS REPORT")
    print("=" * 60)

    med = kmf.median_survival_time_
    print(f"\nMedian survival time: {med:.1f} days")

    sf = kmf.survival_function_
    print("\nSurvival probabilities:")
    for t in [30, 90, 180, 365]:
        idx = sf.index[sf.index <= t]
        if len(idx) > 0:
            prob = float(sf.loc[idx[-1]].iloc[0])
            print(f"  At day {t:>3}: {prob:.3f}")
        else:
            print(f"  At day {t:>3}: 1.000 (no events yet)")

    if cph is not None:
        print(f"\nCox model concordance index: {cph.concordance_index_:.4f}")
        print("\nHazard ratios:")
        hr = cph.hazard_ratios_
        ci = cph.confidence_intervals_
        print(f"{'Feature':<30} {'HR':>8} {'95% CI'}")
        print("-" * 55)
        for feat in hr.index:
            h = hr[feat]
            lo = np.exp(ci.loc[feat, "95% lower-bound"])
            hi = np.exp(ci.loc[feat, "95% upper-bound"])
            direction = "↑ risk" if h > 1 else "↓ risk"
            print(f"{feat:<30} {h:>8.3f}  ({lo:.3f}–{hi:.3f})  {direction}")

    print("=" * 60)
