"""Offline self-test for postgame-only audit."""
import importlib.util
import sys
import types
import unittest
if importlib.util.find_spec('psycopg2') is None:
    sys.modules['psycopg2']=types.ModuleType('psycopg2')
from mbb_postgame_snapshot_audit import evaluate, QUERY

class PostgameTest(unittest.TestCase):
    def test_two_games(self):
        rows=[('1',71,83,False,11.27,16.88),('2',76,67,False,.01,2.93)]
        games,summary=evaluate(rows)
        self.assertEqual(summary['count'],2)
        self.assertEqual(summary['groups']['all']['correct'],1)
        self.assertEqual(summary['groups']['all']['accuracy_pct'],50.0)
        self.assertEqual(games[0]['rating_result'],'correct')
    def test_empty(self):
        games,summary=evaluate([])
        self.assertEqual(games,[])
        self.assertEqual(summary['groups'],{})
    def test_only_finished_and_early_snapshots(self):
        self.assertIn('g.completed = 1',QUERY)
        self.assertIn('h.recorded_at < h.tipoff_utc',QUERY)
        self.assertIn('a.recorded_at < a.tipoff_utc',QUERY)
        self.assertNotIn('INSERT',QUERY)
    def test_competitive_partition(self):
        games,summary=evaluate([('12',65,60,True,10,12),('13',75,70,True,10,20)])
        self.assertEqual(summary['groups']['competitive']['games'],1)
        self.assertEqual(summary['groups']['other']['games'],1)
        self.assertEqual(summary['groups']['neutral']['games'],2)

if __name__=='__main__': unittest.main()
