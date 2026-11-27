import sys
from pathlib import Path
import unittest
import tempfile
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from espn_adjusted_research import (
    load_and_validate, design_matrix, fit_correction, predict, metrics, run_study,
    paired_bootstrap_brier, CANDIDATES
)

class ESPNAdjustedResearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = ROOT / 'reference' / 'espn_core_2026_500.csv'
        cls.data = load_and_validate(cls.path)

    def test_required_archive_values(self):
        self.assertEqual(len(self.data), 500)
        self.assertEqual(self.data.game_id.nunique(), 500)

    def test_chronological_split(self):
        r = run_study(self.data)
        self.assertEqual(r['split_counts'], {'train': 293, 'validation': 97, 'evaluation': 110})

    def test_baseline_is_exact_espn(self):
        b = predict(self.data, CANDIDATES[0], np.array([]))
        np.testing.assert_allclose(b, self.data.espn_home_probability.to_numpy(), rtol=1e-12, atol=1e-12)

    def test_correction_coefficients(self):
        train = self.data[self.data.month.isin(['2025-11','2025-12','2026-01'])]
        for c in CANDIDATES[1:]:
            b = fit_correction(train, c)
            self.assertTrue(np.isfinite(b).all())
            p = predict(train, c, b)
            self.assertTrue(((p > 0) & (p < 1)).all())

    def test_design_matrix_sizes(self):
        self.assertEqual(design_matrix(self.data, CANDIDATES[0]).shape, (500, 0))
        self.assertEqual(design_matrix(self.data, CANDIDATES[1]).shape, (500, 2))
        self.assertEqual(design_matrix(self.data, CANDIDATES[2]).shape, (500, 4))

    def test_duplicate_game_id_rejected(self):
        bad = self.data.copy(); bad.loc[1, 'game_id'] = bad.loc[0, 'game_id']
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / 'dupe.csv'
            bad.to_csv(p, index=False)
            with self.assertRaisesRegex(ValueError, 'duplicated'):
                load_and_validate(p)

    def test_metrics_known_values(self):
        m = metrics([1,0],[0.8,0.2]);self.assertEqual(m['correct'],2)
        self.assertAlmostEqual(m['brier'],0.04)

    def test_bootstrap_zero_difference(self):
        ci = paired_bootstrap_brier([0,1],[0.2,0.8],[0.2,0.8],100)
        self.assertEqual(ci, [0.0,0.0])

    def test_evaluation_labels_do_not_affect_selection_or_fit(self):
        original = run_study(self.data)
        changed = self.data.copy()
        mask = changed.month.isin(['2026-03', '2026-04'])
        changed.loc[mask, 'home_win'] = 1 - changed.loc[mask, 'home_win']
        new = run_study(changed)
        self.assertEqual(original['selected'], new['selected'])
        self.assertEqual(original['coefficients'], new['coefficients'])
        pd.testing.assert_frame_equal(original['validation'], new['validation'])

    def test_validation_and_evaluation_disjoint(self):
        r = run_study(self.data)
        evaluation = r['predictions']
        self.assertEqual(set(evaluation.month.unique()), {'2026-03','2026-04'})
        self.assertFalse(set(evaluation.game_id) & set(self.data[self.data.month=='2026-02'].game_id))

    def test_no_future_games_in_archive(self):
        self.assertTrue((self.data.retrieved_at_utc > self.data.tipoff_utc).all())

if __name__ == '__main__': unittest.main()
