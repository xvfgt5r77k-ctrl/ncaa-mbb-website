"""Offline tests for historical opponent-adjusted ratings and SOS snapshots."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from historical_analytics import (
    adjusted_ratings, build_analytics, RATING_COLUMNS, SNAPSHOT_COLUMNS, SCHEMA,
)


def game(n, date, home, away, hs, aws, home_d1=1, away_d1=1):
    return dict(game_id=str(n), season=2027, game_date=date,
                home_team_id=home, away_team_id=away,
                home_score=hs, away_score=aws, home_points=hs, away_points=aws,
                home_possessions=70., away_possessions=70., neutral_site=0,
                home_is_d1=home_d1, away_is_d1=away_d1)


class HistoricalAnalyticsTests(unittest.TestCase):
    def setUp(self):
        self.games = [
            game(1, '2026-11-01', 'A', 'B', 80, 70),
            game(2, '2026-11-02', 'C', 'D', 65, 72),
            game(3, '2026-11-03', 'A', 'C', 71, 68),
            game(4, '2026-11-04', 'B', 'D', 70, 75),
            game(5, '2026-11-05', 'A', 'D', 68, 80),
            game(6, '2026-11-06', 'B', 'C', 79, 63),
            game(7, '2026-11-07', 'C', 'A', 76, 70),
            game(8, '2026-11-08', 'D', 'B', 68, 65),
        ]

    def test_snapshots_do_not_change_when_future_game_is_added(self):
        _, earlier = build_analytics(self.games, generated_at='test')
        _, later = build_analytics(self.games + [
            game(100, '2026-11-30', 'A', 'D', 120, 40)], generated_at='test')
        self.assertEqual(earlier, later[:len(earlier)])

    def test_final_team_ratings_and_sos(self):
        rows, history = build_analytics(self.games)
        self.assertEqual(len(rows), 4)
        self.assertEqual(len(history), 16)
        ratings = {
            dict(zip(RATING_COLUMNS, row))['team_id']: dict(zip(RATING_COLUMNS, row))
            for row in rows
        }
        self.assertEqual(ratings['A']['games_played'], 4)
        self.assertLess(abs(ratings['A']['opponent_sos']), 100)
        self.assertAlmostEqual(ratings['A']['adjusted_net'],
                               ratings['A']['adjusted_offense'] -
                               ratings['A']['adjusted_defense'])
        self.assertTrue(all(-100 < r['sos'] < 100
                            for r in adjusted_ratings(self.games).values()))

    def test_snapshot_is_end_of_day_and_only_team_play_dates(self):
        _, history = build_analytics(self.games)
        entries = [dict(zip(SNAPSHOT_COLUMNS, row)) for row in history]
        first_a = next(x for x in entries if x['team_id'] == 'A')
        self.assertEqual(first_a['snapshot_date'], '2026-11-01')
        self.assertEqual(first_a['games_played'], 1)
        second_a = next(x for x in entries if x['team_id'] == 'A' and
                        x['snapshot_date'] == '2026-11-03')
        self.assertEqual(second_a['games_played'], 2)
        self.assertEqual(len([x for x in entries if x['team_id'] == 'A']), 4)

    def test_d1_snapshot_only_but_non_d1_opponent_counts(self):
        games = [game(1, '2026-11-01', 'A', 'X', 70, 60, away_d1=0),
                 game(2, '2026-11-02', 'A', 'B', 72, 69)]
        ratings, history = build_analytics(games)
        self.assertNotIn('X', {r[1] for r in ratings})
        self.assertNotIn('X', {r[2] for r in history})
        self.assertEqual(next(r[2] for r in ratings if r[1] == 'A'), 2)

    def test_incomplete_box_ignored(self):
        rows = self.games + [dict(game(11, '2026-11-11', 'B', 'C', 90, 80),
                                  home_possessions=0)]
        ratings, history = build_analytics(rows)
        self.assertEqual(len(ratings), 4)
        self.assertEqual(len(history), 16)

    def test_schema_protects_derived_tables(self):
        self.assertIn('ALTER TABLE team_adjusted_ratings ENABLE ROW LEVEL SECURITY', SCHEMA)
        self.assertIn('ALTER TABLE historical_team_snapshots ENABLE ROW LEVEL SECURITY', SCHEMA)
        self.assertNotIn('DELETE FROM games', SCHEMA)


if __name__ == '__main__':
    unittest.main()
