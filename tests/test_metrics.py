"""Unit tests for the exact reviewer-facing metric definitions."""

from __future__ import annotations

import math
import unittest

import numpy as np

from ob1_repro.attention_profile import profile_metrics


class MetricTests(unittest.TestCase):
    """Exercise the frozen Spearman and base-2 squared-JS definitions."""

    def test_identical_profiles_have_perfect_similarity(self) -> None:
        """Require perfect rank agreement and zero base-2 squared JS."""
        values = np.array([0.1, 0.2, 0.7])
        result = profile_metrics(values, values, np.array([-1, 0, 1]))
        self.assertTrue(math.isclose(result["profile_spearman"], 1.0))
        self.assertTrue(math.isclose(result["js_divergence"], 0.0))

    def test_disjoint_profiles_have_unit_base2_squared_js(self) -> None:
        """Require the known maximum for disjoint normalized profiles."""
        result = profile_metrics(
            np.array([1.0, 0.0]),
            np.array([0.0, 1.0]),
            np.array([0, 1]),
        )
        self.assertTrue(math.isclose(result["profile_spearman"], -1.0))
        self.assertTrue(math.isclose(result["js_divergence"], 1.0))

    def test_spearman_excludes_shared_zero_offsets(self) -> None:
        """Require the stated union-of-nonzero-offset Spearman support."""
        result = profile_metrics(
            np.array([0.0, 0.2, 0.8, 0.0]),
            np.array([0.0, 0.4, 0.6, 0.0]),
            np.array([-2, -1, 0, 1]),
        )
        self.assertTrue(math.isclose(result["profile_spearman"], 1.0))


if __name__ == "__main__":
    unittest.main()
