import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from acc_historical_archive_pilot import is_historical_archive_frame, candidate_from_cells


class HistoricalArchiveTests(unittest.TestCase):
    def test_archive_iframe_allowed(self):
        self.assertTrue(is_historical_archive_frame(
            "https://app.hdintelligence.com/?conf=ACC&source=ACC&sport=Men%27s+Basketball&type=archive"))

    def test_current_feed_rejected(self):
        self.assertFalse(is_historical_archive_frame(
            'https://app.hdintelligence.com/?conf=ACC&type=current'))

    def test_wrong_host_rejected(self):
        self.assertFalse(is_historical_archive_frame(
            'https://example.org/?type=archive'))

    def test_http_rejected(self):
        self.assertFalse(is_historical_archive_frame(
            'http://app.hdintelligence.com/?type=archive'))

    def test_known_status(self):
        row = candidate_from_cells(['Example Student', 'Sample University', 'Out'])
        self.assertEqual(row['candidate_status'], 'Out')
        self.assertIn('Example Student', row['row_text'])

    def test_game_time_decision(self):
        self.assertEqual(candidate_from_cells(['Example Student', 'Game-Time Decision'])['candidate_status'],
                         'Game Time Decision')

    def test_no_status_not_inferred(self):
        self.assertIsNone(candidate_from_cells(['Example Student', 'DNP - Coach Decision']))

    def test_ambiguous_status_rejected(self):
        self.assertIsNone(candidate_from_cells(['Example Student', 'Available', 'Out']))

    def test_legend_header_rejected(self):
        self.assertIsNone(candidate_from_cells(['Player', 'Out']))

    def test_single_cell_rejected(self):
        self.assertIsNone(candidate_from_cells(['Out']))


if __name__ == '__main__':
    unittest.main()
