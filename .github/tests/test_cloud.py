import unittest
from argparse import Namespace
from datetime import date
from cloud_store import upsert_sql, POSTGRES_SCHEMA
from cloud_sync import select_range


class TestCloudConversion(unittest.TestCase):
    def test_upsert_unique_game(self):
        query = upsert_sql('games', ['game_id', 'status', 'home_score'])
        self.assertIn('ON CONFLICT (game_id) DO UPDATE SET', query)
        self.assertIn('status=EXCLUDED.status', query)
        self.assertEqual(query.count('%s'), 3)

    def test_upsert_preserves_d1_and_non_null_team_metadata(self):
        query = upsert_sql('teams', ['team_id','name','conference','is_d1'])
        self.assertIn('GREATEST(teams.is_d1,EXCLUDED.is_d1)', query)
        self.assertIn('COALESCE(EXCLUDED.conference,teams.conference)', query)

    def test_invalid_table_rejected(self):
        with self.assertRaises(ValueError):
            upsert_sql('weird_table', ['id','name'])

    def test_schedule_range(self):
        args=Namespace(days_back=4, days_forward=14, start_date=None, end_date=None)
        start,end=select_range(args, today=date(2026,11,23))
        self.assertEqual(start, date(2026,11,19))
        self.assertEqual(end, date(2026,12,7))

    def test_backfill_range(self):
        args=Namespace(days_back=4, days_forward=14, start_date=date(2026,11,1), end_date=date(2026,11,20))
        self.assertEqual(select_range(args), (date(2026,11,1),date(2026,11,20)))

    def test_schema_protects_postgres_rest_api(self):
        for table in ('teams','games','team_game_stats','players','player_game_stats','injury_reports','sync_runs'):
            self.assertIn(f'ALTER TABLE {table} ENABLE ROW LEVEL SECURITY', POSTGRES_SCHEMA)

if __name__ == '__main__':
    unittest.main()
