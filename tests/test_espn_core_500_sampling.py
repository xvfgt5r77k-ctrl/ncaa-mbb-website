"""Offline tests for the 500-game retrospective month-balanced sampler."""
import unittest
import pandas as pd
from espn_core_historical_audit import sample_across_months


def mock_history():
    rows = []
    for m, count in [('2025-11', 220), ('2025-12', 500),
                     ('2026-01', 700), ('2026-02', 650),
                     ('2026-03', 350), ('2026-04', 13)]:
        for j in range(count):
            rows.append({'game_id':f'{m}-{j}', 'game_date':f'{m}-01',
                         'competitive':j % 2 == 0})
    return pd.DataFrame(rows)


class SamplingTests(unittest.TestCase):
    def test_sample_500_and_all_months(self):
        x = sample_across_months(mock_history(), 500)
        self.assertEqual(len(x), 500)
        self.assertEqual(x.game_id.nunique(), 500)
        counts = x.groupby(x.game_date.str[:7]).size()
        self.assertEqual(len(counts), 6)
        self.assertEqual(counts['2026-04'], 13)
        self.assertLessEqual(counts.max() - counts.min(), 100)
        self.assertLessEqual(abs(int(x.competitive.sum()) - 250), 5)

    def test_deterministic_sample(self):
        base = mock_history()
        a = sample_across_months(base, 120)
        b = sample_across_months(base.sample(frac=1,random_state=91), 120)
        self.assertEqual(a.game_id.tolist(), b.game_id.tolist())

    def test_all_records_if_small(self):
        base = mock_history().iloc[:30]
        sample = sample_across_months(base, 500)
        self.assertEqual(len(sample), 30)

    def test_reject_out_of_range(self):
        with self.assertRaises(ValueError):
            sample_across_months(mock_history(), 501)

    def test_reject_duplicate_ids(self):
        base = mock_history()
        base.loc[1, 'game_id'] = base.loc[0, 'game_id']
        with self.assertRaises(ValueError):
            sample_across_months(base, 120)

    def test_reject_bad_dates(self):
        base = mock_history()
        base.loc[0, 'game_date'] = 'not-date'
        with self.assertRaises(ValueError):
            sample_across_months(base, 120)


if __name__=='__main__':
    unittest.main()
