"""
Tests — Benford's Law implementation
The Unnamed Pipeline, Episode 1
"""

import numpy as np
import pandas as pd
import pytest
from benfords import (
    extract_leading_digit,
    benfords_test,
    validate_preconditions,
    flag_suspicious_digits,
    BENFORD_EXPECTED,
)


class TestExtractLeadingDigit:
    def test_simple_integers(self):
        assert extract_leading_digit(123) == 1
        assert extract_leading_digit(456) == 4
        assert extract_leading_digit(789) == 7
        assert extract_leading_digit(9) == 9

    def test_floats(self):
        assert extract_leading_digit(3.14) == 3
        assert extract_leading_digit(0.0056) == 5
        assert extract_leading_digit(0.91) == 9

    def test_large_numbers(self):
        assert extract_leading_digit(1_000_000) == 1
        assert extract_leading_digit(5_432_100) == 5

    def test_negative_values(self):
        assert extract_leading_digit(-250) == 2
        assert extract_leading_digit(-0.007) == 7

    def test_zero_returns_none(self):
        assert extract_leading_digit(0) is None

    def test_nan_returns_none(self):
        assert extract_leading_digit(float("nan")) is None


class TestBenfordsTest:
    @pytest.fixture
    def benford_data(self):
        """Generate data that follows Benford's distribution."""
        rng = np.random.default_rng(42)
        values = 10 ** rng.uniform(1.0, 6.0, size=10000)
        return pd.DataFrame({"amount": values})

    @pytest.fixture
    def uniform_data(self):
        """Generate uniform data that violates Benford's."""
        rng = np.random.default_rng(42)
        values = rng.integers(100, 999, size=5000)
        return pd.DataFrame({"amount": values})

    def test_benford_data_passes(self, benford_data):
        results, chi2, p_value, is_anomalous = benfords_test(
            benford_data, "amount"
        )
        assert not is_anomalous
        assert p_value > 0.05

    def test_uniform_data_fails(self, uniform_data):
        results, chi2, p_value, is_anomalous = benfords_test(
            uniform_data, "amount"
        )
        assert is_anomalous
        assert p_value < 0.05

    def test_results_shape(self, benford_data):
        results, _, _, _ = benfords_test(benford_data, "amount")
        assert len(results) == 9
        assert list(results["digit"]) == list(range(1, 10))

    def test_observed_sums_to_100(self, benford_data):
        results, _, _, _ = benfords_test(benford_data, "amount")
        assert abs(results["observed_pct"].sum() - 100.0) < 0.1

    def test_custom_significance_level(self, benford_data):
        _, _, p_value, is_anomalous_strict = benfords_test(
            benford_data, "amount", significance_level=0.99
        )
        # At 0.99 threshold, even good data might flag
        # Just verify it runs without error
        assert isinstance(is_anomalous_strict, bool)


class TestValidatePreconditions:
    def test_valid_dataset(self):
        rng = np.random.default_rng(42)
        values = 10 ** rng.uniform(1.0, 5.0, size=1000)
        df = pd.DataFrame({"amount": values})
        result = validate_preconditions(df, "amount")
        assert result["is_valid"] is True
        assert result["sufficient_observations"] is True
        assert result["sufficient_range"] is True

    def test_too_few_observations(self):
        df = pd.DataFrame({"amount": [100, 200, 300]})
        result = validate_preconditions(df, "amount")
        assert result["sufficient_observations"] is False
        assert result["is_valid"] is False

    def test_narrow_range(self):
        rng = np.random.default_rng(42)
        values = rng.uniform(50, 99, size=1000)
        df = pd.DataFrame({"amount": values})
        result = validate_preconditions(df, "amount")
        assert result["sufficient_range"] is False
        assert result["is_valid"] is False


class TestFlagSuspiciousDigits:
    def test_flags_large_deviations(self):
        results = pd.DataFrame({
            "digit": range(1, 10),
            "observed_pct": [30, 17, 12, 16, 8, 7, 5, 3, 2],
            "expected_pct": [30.1, 17.6, 12.5, 9.7, 7.9, 6.7, 5.8, 5.1, 4.6],
            "deviation_pct": [-0.1, -0.6, -0.5, 6.3, 0.1, 0.3, -0.8, -2.1, -2.6],
        })
        flagged = flag_suspicious_digits(results, deviation_threshold=3.0)
        assert len(flagged) == 1
        assert flagged.iloc[0]["digit"] == 4

    def test_no_flags_on_clean_data(self):
        results = pd.DataFrame({
            "digit": range(1, 10),
            "observed_pct": [30.1, 17.6, 12.5, 9.7, 7.9, 6.7, 5.8, 5.1, 4.6],
            "expected_pct": [30.1, 17.6, 12.5, 9.7, 7.9, 6.7, 5.8, 5.1, 4.6],
            "deviation_pct": [0] * 9,
        })
        flagged = flag_suspicious_digits(results)
        assert len(flagged) == 0


class TestBenfordExpected:
    def test_expected_sums_to_one(self):
        total = sum(BENFORD_EXPECTED.values())
        assert abs(total - 1.0) < 1e-10

    def test_digit_1_is_highest(self):
        assert BENFORD_EXPECTED[1] > BENFORD_EXPECTED[2]
        assert BENFORD_EXPECTED[1] > 0.30

    def test_digit_9_is_lowest(self):
        assert BENFORD_EXPECTED[9] < BENFORD_EXPECTED[8]
        assert BENFORD_EXPECTED[9] < 0.05
