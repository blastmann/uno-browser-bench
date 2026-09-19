import unittest

from scripts.calibrate_decider_and_curve import ece, fit_temperature, materialize, summarize


class CalibrationTests(unittest.TestCase):
    def test_temperature_reduces_overconfidence(self):
        rows = [
            {"probs": [0.99, 0.01], "gold_index": 0, "gold": "a", "candidates": ["a", "b"]},
            {"probs": [0.99, 0.01], "gold_index": 1, "gold": "b", "candidates": ["a", "b"]},
            {"probs": [0.99, 0.01], "gold_index": 1, "gold": "b", "candidates": ["a", "b"]},
            {"probs": [0.99, 0.01], "gold_index": 0, "gold": "a", "candidates": ["a", "b"]},
        ]
        temperature = fit_temperature(rows)
        self.assertGreater(temperature, 1.0)
        calibrated = materialize(rows, temperature)
        self.assertLess(summarize(calibrated)["mean_confidence"], 0.99)

    def test_ece_bins_do_not_double_count(self):
        rows = [{"confidence": 0.4, "correct": 1}, {"confidence": 0.9, "correct": 1}]
        self.assertLessEqual(ece(rows), 1.0)


if __name__ == "__main__":
    unittest.main()
