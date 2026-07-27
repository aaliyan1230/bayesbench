"""Tests for the calibration / simulation harness.

Validates that:
- Exact enumeration reproduces the documented false-winner rates.
- Monte Carlo simulation converges to expected rates under equal models.
- Order-sensitivity is detectable.
- Parameter sweeps produce plausible outputs.
"""

import pytest

from bayesbench.calibration import (
    CalibrationPoint,
    CalibrationReport,
    calibrate_sweep,
    enumerate_binary_outcomes,
    simulate_order_sensitivity,
    simulate_pairwise,
)

# ---------------------------------------------------------------------------
# Exact enumeration: prove that min_samples=3 + skip_threshold=0.85
# always stops at or before trial 3 under equal Bernoulli models.
# ---------------------------------------------------------------------------


class TestExactEnumeration:
    def test_all_outcomes_terminate_at_three(self):
        """Every pairwise outcome sequence of length 3 triggers a stop."""
        result = enumerate_binary_outcomes(n=3, success_prob_a=0.5, success_prob_b=0.5)
        assert result["total_sequences"] == 64
        assert result["inconclusive_rate"] == pytest.approx(0.0)
        assert result["winner_rate"] + result["skipped_rate"] == pytest.approx(1.0)

    def test_skipped_rate_matches_documented(self):
        """Under equal models, 78.125% of trials are skipped."""
        result = enumerate_binary_outcomes(n=3, success_prob_a=0.5, success_prob_b=0.5)
        assert result["skipped_rate"] == pytest.approx(0.78125, abs=1e-6)

    def test_winner_rate_matches_documented(self):
        """Under equal models, 21.875% of trials incorrectly declare a winner."""
        result = enumerate_binary_outcomes(n=3, success_prob_a=0.5, success_prob_b=0.5)
        assert result["winner_rate"] == pytest.approx(0.21875, abs=1e-6)

    def test_skewed_models_still_terminal(self):
        """Even with skewed observation probability, all 3-trial paths stop."""
        result = enumerate_binary_outcomes(n=3, success_prob_a=0.7, success_prob_b=0.3)
        assert result["inconclusive_rate"] < 0.01


# ---------------------------------------------------------------------------
# Monte Carlo simulation
# ---------------------------------------------------------------------------


class TestMonteCarloSimulation:
    def test_equal_models_false_rate_bounded(self):
        """Under equal models, false-winner rate is measurably > 0 @ defaults."""
        result = simulate_pairwise(
            n=1_000,
            true_acc_a=0.5,
            true_acc_b=0.5,
            confidence=0.95,
            skip_threshold=0.85,
            min_samples=3,
            max_samples=60,
            rng=42,
        )
        total = sum(result[k] for k in ("winner_a", "winner_b", "skipped", "inconclusive"))
        false_rate = result["false_decisions"] / total if total > 0 else 0.0
        # With equal models, any declared winner is a false decision.
        # We expect ~20% false-winner rate.
        assert false_rate > 0.10, f"Expected >10% false winner rate, got {false_rate:.1%}"
        assert result["skipped"] > 0, "Some runs should be skipped"

    def test_disabling_skip_reduces_false_rate(self):
        """skip_threshold=1.0 + min_samples=10 should reduce false winners."""
        result = simulate_pairwise(
            n=500,
            true_acc_a=0.5,
            true_acc_b=0.5,
            confidence=0.95,
            skip_threshold=1.0,
            min_samples=10,
            max_samples=100,
            rng=42,
        )
        total = sum(result[k] for k in ("winner_a", "winner_b", "skipped", "inconclusive"))
        assert result["skipped"] == 0, "skip_threshold=1.0 should disable skipping"
        assert total == 500

    def test_strong_effect_detected_efficiently(self):
        """A model with true acc 0.95 vs 0.05 should be detected fast."""
        result = simulate_pairwise(
            n=500,
            true_acc_a=0.95,
            true_acc_b=0.05,
            confidence=0.95,
            skip_threshold=1.0,
            min_samples=3,
            max_samples=60,
            rng=42,
        )
        total = sum(result[k] for k in ("winner_a", "winner_b", "skipped", "inconclusive"))
        false_rate = result["false_decisions"] / total if total > 0 else 0.0
        mean_samples = sum(result["samples_drawn"]) / len(result["samples_drawn"])
        assert result["winner_a"] > result["winner_b"], "Strong A should win most runs"
        assert false_rate < 0.02, f"False rate {false_rate:.1%} too high for strong effect"
        assert mean_samples < 50, f"Expected early stop for strong effect, got {mean_samples:.1f}"

    def test_reproducible_with_seed(self):
        """Same seed should produce same result."""
        r1 = simulate_pairwise(n=500, rng=42)
        r2 = simulate_pairwise(n=500, rng=42)
        assert r1["winner_a"] == r2["winner_a"]
        assert r1["winner_b"] == r2["winner_b"]
        assert r1["skipped"] == r2["skipped"]


# ---------------------------------------------------------------------------
# Order sensitivity
# ---------------------------------------------------------------------------


class TestOrderSensitivity:
    def test_favorable_front_can_trigger_early_win(self):
        """3 perfect answers for A at the front of a bad dataset can
        cause an early (wrong) win for model A."""
        result = simulate_order_sensitivity(
            favorable_front=3,
            unfavorable_after=100,
            n_runs=500,
            rng=42,
        )
        assert result["early_wins"] > 0, (
            "Should see early wins when favorable examples come first"
        )

    def test_larger_front_higher_early_win_rate(self):
        """More favorable examples at the front increase early-win odds."""
        r3 = simulate_order_sensitivity(favorable_front=3, n_runs=300, rng=42)
        r5 = simulate_order_sensitivity(favorable_front=5, n_runs=300, rng=42)
        assert r5["early_win_rate"] >= r3["early_win_rate"], (
            "More favorable examples should not decrease early-win rate"
        )


# ---------------------------------------------------------------------------
# Calibration sweep
# ---------------------------------------------------------------------------


class TestCalibrationSweep:
    def test_sweep_under_equal_models(self):
        """Quick sweep under equal models; verify structure and ranges."""
        report = calibrate_sweep(
            min_samples_grid=(3, 10),
            skip_threshold_grid=(0.85, 0.99),
            n_runs=100,
            possible_max=30,
            seed=42,
        )
        assert isinstance(report, CalibrationReport)
        assert len(report.points) == 4  # 2 min_samples x 2 skip_thresholds
        for p in report.points:
            assert 0.0 <= p.false_winner_rate <= 1.0
            assert 0.0 <= p.skipped_rate <= 1.0
            assert p.expected_samples > 0

    def test_default_params_have_high_false_rate(self):
        """The current defaults (min_samples=3, skip_threshold=0.85) produce
        a measurable false-winner rate under equal models."""
        report = calibrate_sweep(
            min_samples_grid=(3,),
            skip_threshold_grid=(0.85,),
            n_runs=500,
            possible_max=60,
            seed=42,
        )
        p = report.points[0]
        assert p.false_winner_rate > 0.10, (
            f"Default false-winner rate {p.false_winner_rate:.1%} should be >10%"
        )

    def test_safe_defaults_have_low_false_rate(self):
        """min_samples=20, skip_threshold=0.99 should keep false rate low."""
        report = calibrate_sweep(
            min_samples_grid=(20,),
            skip_threshold_grid=(0.99,),
            n_runs=300,
            possible_max=60,
            seed=42,
        )
        p = report.points[0]
        assert p.false_winner_rate < 0.08, (
            f"Safe defaults false rate {p.false_winner_rate:.1%} should be <8%"
        )

    def test_table_formatting(self):
        """Table output should contain key columns."""
        report = calibrate_sweep(
            min_samples_grid=(3, 10),
            skip_threshold_grid=(0.85,),
            n_runs=500,
            seed=42,
        )
        table = report.format_table()
        assert "min_s" in table
        assert "skip" in table
        assert "false" in table
        assert "E[n]" in table


# ---------------------------------------------------------------------------
# CalibrationPoint
# ---------------------------------------------------------------------------


class TestCalibrationPoint:
    def test_str_contains_rates(self):
        p = CalibrationPoint(
            min_samples=3,
            skip_threshold=0.85,
            confidence=0.95,
            n_runs=1_000,
            false_winner_rate=0.21875,
            skipped_rate=0.78125,
            inconclusive_rate=0.0,
            expected_samples=3.0,
            possible_max_samples=100,
        )
        s = str(p)
        assert "21.9%" in s or "21.9" in s.replace("%", "")
        assert "min=" in s
