import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit_historical_report_coverage import analyze, official_url_matches, load_sources

SOURCES = {'acc': {'source_id': 'acc', 'source_type': 'official_conference',
                   'official_entry_url': 'https://theacc.com/sports/archive'},
           'secondary': {'source_id': 'secondary', 'source_type': 'secondary',
                         'official_entry_url': ''}}
GAMES = [{'season': 2026, 'game_id': '1', 'home_team_id': 'A', 'away_team_id': 'B'},
         {'season': 2026, 'game_id': '2', 'home_team_id': 'C', 'away_team_id': 'D'}]
def evidence(team='A', source='acc', verified='yes', url='https://theacc.com/sports/doc'):
    return {'season': '2026', 'game_id': '1', 'team_id': team,
            'source_id': source, 'source_url': url, 'verified': verified}

class AuditTests(unittest.TestCase):
    def test_no_evidence_is_unknown(self):
        summary, games = analyze(GAMES, [], SOURCES)
        self.assertEqual(summary['no_official_report_evidence'], 2)
        self.assertEqual(games[0]['coverage'], 'unknown')
    def test_one_team_not_full(self):
        summary, games = analyze(GAMES, [evidence()], SOURCES)
        self.assertEqual(summary['one_team_documented'], 1)
        self.assertEqual(summary['both_teams_documented'], 0)
    def test_both_teams_full(self):
        summary, games = analyze(GAMES, [evidence(), evidence(team='B')], SOURCES)
        self.assertEqual(summary['both_teams_documented'], 1)
    def test_secondary_excluded(self):
        summary, _ = analyze(GAMES, [evidence(source='secondary')], SOURCES)
        self.assertEqual(summary['no_official_report_evidence'], 2)
    def test_unverified_excluded(self):
        summary, _ = analyze(GAMES, [evidence(verified='no')], SOURCES)
        self.assertEqual(summary['no_official_report_evidence'], 2)
    def test_untrusted_url_excluded(self):
        summary, _ = analyze(GAMES, [evidence(url='https://evil.example.com')], SOURCES)
        self.assertEqual(summary['no_official_report_evidence'], 2)
    def test_team_not_in_game_excluded(self):
        summary, _ = analyze(GAMES, [evidence(team='X')], SOURCES)
        self.assertEqual(summary['no_official_report_evidence'], 2)
    def test_known_url(self):
        self.assertTrue(official_url_matches('https://theacc.com/sports/doc', SOURCES['acc']))
        self.assertFalse(official_url_matches('https://not-theacc.com/doc', SOURCES['acc']))
    def test_source_registry(self):
        path = Path(__file__).resolve().parents[1] / 'reference' / 'source_registry.csv'
        self.assertGreaterEqual(len(load_sources(path)), 10)

if __name__ == '__main__':
    unittest.main()
