from __future__ import annotations

import json
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import compare_completed_2027 as comp


class TestCompletedGameResearch(unittest.TestCase):
    def test_archived_selection_is_exact(self):
        d = json.loads(comp.SPECS.read_text())
        actual = comp.check_specifications(d)
        self.assertEqual(actual, d['v8']['hybrid']['features'])
        self.assertEqual(d['v12']['winning_model'], 'V12 component v5_shooting')

    def test_cutoff_requires_original_source(self):
        self.assertIn('BETWEEN 2022 AND 2027', comp.extend_sql('where x BETWEEN 2022 AND 2026'))
        with self.assertRaises(RuntimeError):
            comp.extend_sql('where x BETWEEN 2022 AND 2027')

    def test_same_day_results_not_used_by_strict_elo(self):
        dates = ["2026-11-22", "2026-11-22", "2026-11-23"]
        games = pd.DataFrame({
            'season':[2027]*3, 'game_id':['a','b','c'], 'game_date':dates,
            'home_team_id':['A','A','A'], 'away_team_id':['B','C','D'],
            'neutral_site':[0]*3, 'home_score':[70,70,70], 'away_score':[60,61,62]})
        base = comp.v4.chronological_features(games)
        edited = games.copy()
        edited.loc[0,'home_score'] = 120
        updated = comp.v4.chronological_features(edited)
        for same_day_game in ['a','b']:
            row1 = base[base.game_id==same_day_game].iloc[0]
            row2 = updated[updated.game_id==same_day_game].iloc[0]
            self.assertAlmostEqual(float(row1.elo_gap),float(row2.elo_gap))
        self.assertNotAlmostEqual(float(base.loc[2,'elo_gap']),float(updated.loc[2,'elo_gap']))

    def one_snapshot(self, stage='initial', late=False):
        tipoff = '2026-11-21T21:00:00+00:00'
        return dict(game_id='game123', season=2027, model_version='V1',
                    snapshot_stage=stage,
                    recorded_at=('2026-11-21T22:00:00+00:00' if late
                                 else '2026-11-21T19:00:00+00:00'),
                    tipoff_utc=tipoff, game_tipoff_utc=tipoff,
                    matched_game_id='game123',
                    snap_home_id='A',game_home_id='A',
                    snap_away_id='B',game_away_id='B',
                    snap_neutral_site=0,game_neutral_site=0,
                    game_completed='1',home_score=81,away_score=72,
                    projected_home_points=78, projected_away_points=70,
                    projected_margin=8,projected_total=148)

    def test_original_v1_only_and_late_rejection(self):
        sample = pd.DataFrame([dict(game_id='game123', season=2027,
              game_date='2026-11-21',home_score=81,away_score=72,
              home_actual_win=1,actual_total=153,elo_close=True,
              v8_winner_correct=1,v12_winner_correct=1,
              v8_home_win_probability=.67,v12_home_win_probability=.64,
              v12_projected_total=150,v12_total_absolute_error=3)])
        out, statuses, count = comp.attach_original_v1(sample,
            [self.one_snapshot(), self.one_snapshot(stage='within_24h')],
            datetime(2026,11,25,tzinfo=timezone.utc))
        self.assertEqual(count,1)
        self.assertEqual(out.v1_winner_correct.iloc[0], 1)
        self.assertEqual(len(statuses),1)
        out, statuses, count = comp.attach_original_v1(sample,
            [self.one_snapshot(late=True)],datetime(2026,11,25,tzinfo=timezone.utc))
        self.assertEqual(count,0)
        self.assertTrue(pd.isna(out.v1_winner_correct.iloc[0]))
        self.assertTrue(any('snapshot_not_pre_tipoff' in k for k in statuses))

    def test_summary_same_game_denominator(self):
        sample = pd.DataFrame([
          dict(game_id='a',elo_close=True,home_actual_win=1,
               v1_winner_correct=1.,v1_total_absolute_error=3,
               v8_winner_correct=1,v12_winner_correct=1,
               v12_total_absolute_error=4),
          dict(game_id='b',elo_close=False,home_actual_win=0,
               v1_winner_correct=np.nan,v1_total_absolute_error=np.nan,
               v8_winner_correct=0,v12_winner_correct=1,
               v12_total_absolute_error=7),
        ])
        s = comp.summary_frame(sample)
        sub = s[s.scope=='matched_three_models']
        self.assertEqual(sub.games.tolist(),[1,1,1])
        self.assertEqual(sub.winner_accuracy_pct.tolist(),[100.,100.,100.])
        self.assertEqual(len(s[s.scope=='all_2027_replay_eligible']),2)

    def test_completed_only_synthetic_model_fit(self):
        # Fixtures are synthetic, not real NCAA outcomes. Proves pipeline can
        # train the archived model families without a database connection.
        fields = comp.check_specifications(json.loads(comp.SPECS.read_text()))
        columns = list(dict.fromkeys(fields + comp.v12.WIN_COMPONENTS['v5_shooting'] +
                                   comp.v11.SCORE_FIELDS + ['v9_scoring_baseline_total']))
        rng=np.random.default_rng(17)
        n_train,n_review=630,24
        d={col:rng.normal(size=n_train+n_review) for col in columns}
        d['v9_scoring_baseline_total']=rng.normal(142,8,size=n_train+n_review)
        d['neutral_site']=rng.integers(0,2,size=n_train+n_review)
        diff = 8*d['elo_gap'] + rng.normal(0,8,size=n_train+n_review)
        scores_h=np.round((140+diff)/2).astype(int)
        scores_a=np.round((140-diff)/2).astype(int)
        ties=scores_h==scores_a
        scores_h[ties] += 1
        f=pd.DataFrame(d)
        f['season']=([2022]*210+[2023]*210+[2024]*210+[2027]*n_review)
        f['game_id']=[f'fake-{i}' for i in range(len(f))]
        f['game_date']=['2026-11-20']*len(f)
        f['home_score']=scores_h
        f['away_score']=scores_a
        f['home_win']=(scores_h>scores_a).astype(int)
        out=comp.evaluate_models(f,fields)
        self.assertEqual(len(out),n_review)
        self.assertTrue(out.v8_home_win_probability.between(0,1).all())
        self.assertTrue(out.v12_home_win_probability.between(0,1).all())
        self.assertTrue(np.isfinite(out.v12_projected_total).all())

    def test_dashboard_page_present(self):
        dash = (ROOT/'cloud_dashboard.py').read_text()
        self.assertIn('V1 / V8 / V12 completed-game comparison',dash)
        self.assertIn('Live results',dash)
        self.assertNotIn('V7 Historical Predictions',dash)


if __name__=='__main__':
    unittest.main()
