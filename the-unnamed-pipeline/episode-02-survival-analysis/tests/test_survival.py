"""
Tests — Survival Analysis implementation
The Unnamed Pipeline, Episode 2
"""

import numpy as np
import pandas as pd
import pytest
from survival import (
    fit_kaplan_meier, compare_groups, fit_cox_model,
    predict_individual_survival, flag_urgent_customers,
    median_survival_time,
)
from lifelines import KaplanMeierFitter, CoxPHFitter


@pytest.fixture(scope="module")
def saas_data():
    rng = np.random.default_rng(42)
    n = 800
    plan_monthly = rng.choice([0, 1], size=n, p=[0.4, 0.6])
    days_since_login = rng.integers(0, 61, size=n)
    support_tickets = rng.integers(0, 11, size=n)
    features_activated = rng.integers(1, 13, size=n)
    large_team = rng.choice([0, 1], size=n, p=[0.5, 0.5])

    log_hazard = (
        -0.2                           # baseline: avg survival ~190 days
        + 0.75 * plan_monthly
        + 0.03 * days_since_login
        + 0.14 * support_tickets
        - 0.13 * features_activated
        - 0.43 * large_team
    )
    hazard = np.exp(log_hazard)
    survival_times = np.clip(
        rng.exponential(scale=1.0 / hazard) * 365, 1, 730
    ).astype(int)
    censor_time = rng.integers(180, 731, size=n)
    churned = (survival_times <= censor_time).astype(int)
    observed = np.minimum(survival_times, censor_time)

    return pd.DataFrame({
        "tenure_days": observed,
        "churned": churned,
        "plan_type": np.where(plan_monthly == 1, "monthly", "annual"),
        "plan_monthly": plan_monthly,
        "days_since_login": days_since_login,
        "support_tickets": support_tickets,
        "features_activated": features_activated,
        "large_team": large_team,
    })


@pytest.fixture(scope="module")
def kmf_fitted(saas_data):
    return fit_kaplan_meier(saas_data, "tenure_days", "churned")


@pytest.fixture(scope="module")
def cph_fitted(saas_data):
    covariates = ["plan_monthly", "days_since_login",
                  "support_tickets", "features_activated", "large_team"]
    return fit_cox_model(saas_data, "tenure_days", "churned", covariates)


@pytest.fixture(scope="module")
def covariates_df(saas_data):
    return saas_data[["plan_monthly", "days_since_login",
                       "support_tickets", "features_activated", "large_team"]]


class TestFitKaplanMeier:
    def test_returns_kmf_instance(self, kmf_fitted):
        assert isinstance(kmf_fitted, KaplanMeierFitter)

    def test_survival_starts_at_one(self, kmf_fitted):
        first_val = float(kmf_fitted.survival_function_.iloc[0])
        assert abs(first_val - 1.0) < 0.01

    def test_survival_is_monotonically_non_increasing(self, kmf_fitted):
        sf = kmf_fitted.survival_function_.iloc[:, 0].values
        assert all(sf[i] >= sf[i + 1] for i in range(len(sf) - 1))

    def test_median_is_positive_finite(self, kmf_fitted):
        med = kmf_fitted.median_survival_time_
        assert np.isfinite(med)
        assert med > 0


class TestCompareGroups:
    def test_returns_fitters_key(self, saas_data):
        result = compare_groups(saas_data, "tenure_days", "churned", "plan_type")
        assert "fitters" in result

    def test_two_groups_returns_p_value(self, saas_data):
        result = compare_groups(saas_data, "tenure_days", "churned", "plan_type")
        assert "p_value" in result
        assert "test_statistic" in result

    def test_monthly_vs_annual_significant(self, saas_data):
        result = compare_groups(saas_data, "tenure_days", "churned", "plan_type")
        assert result["p_value"] < 0.05


class TestFitCoxModel:
    def test_returns_cph_instance(self, cph_fitted):
        assert isinstance(cph_fitted, CoxPHFitter)

    def test_concordance_between_05_and_1(self, cph_fitted):
        ci = cph_fitted.concordance_index_
        assert 0.5 < ci < 1.0

    def test_summary_has_correct_rows(self, cph_fitted):
        covariates = ["plan_monthly", "days_since_login",
                      "support_tickets", "features_activated", "large_team"]
        assert len(cph_fitted.summary) == len(covariates)

    def test_hazard_ratios_contains_all_covariates(self, cph_fitted):
        expected = {"plan_monthly", "days_since_login",
                    "support_tickets", "features_activated", "large_team"}
        assert expected == set(cph_fitted.hazard_ratios_.index)


class TestPredictIndividualSurvival:
    def test_returns_dataframe(self, cph_fitted, covariates_df):
        result = predict_individual_survival(cph_fitted, covariates_df.head(5))
        assert isinstance(result, pd.DataFrame)

    def test_values_between_0_and_1(self, cph_fitted, covariates_df):
        result = predict_individual_survival(cph_fitted, covariates_df.head(5))
        assert result.values.min() >= 0.0
        assert result.values.max() <= 1.0

    def test_survival_non_increasing_per_customer(self, cph_fitted, covariates_df):
        result = predict_individual_survival(
            cph_fitted, covariates_df.head(3), times=list(range(1, 100))
        )
        for col in result.columns:
            vals = result[col].values
            assert all(vals[i] >= vals[i + 1] for i in range(len(vals) - 1))


class TestFlagUrgentCustomers:
    def test_returns_dataframe(self, cph_fitted, covariates_df):
        result = flag_urgent_customers(cph_fitted, covariates_df.head(50))
        assert isinstance(result, pd.DataFrame)

    def test_all_flagged_below_threshold(self, cph_fitted, covariates_df):
        threshold = 0.60
        result = flag_urgent_customers(
            cph_fitted, covariates_df.head(100),
            horizon_days=14, threshold=threshold,
        )
        if len(result) > 0:
            assert (result["survival_prob_14d"] < threshold).all()


class TestMedianSurvivalTime:
    def test_returns_float(self, kmf_fitted):
        result = median_survival_time(kmf_fitted)
        assert isinstance(result, float)

    def test_value_is_positive(self, kmf_fitted):
        result = median_survival_time(kmf_fitted)
        assert result > 0
