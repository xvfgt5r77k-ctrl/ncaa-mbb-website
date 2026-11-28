"""Offline unit tests; no DB or API required."""
import importlib.util
import pathlib
import sys
import unittest
import types
from unittest.mock import patch
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd

BASE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE))
from winner_engine import TEAM_FIELDS, FULL, make_features, valid_history, valid_future, describe_predictions, model_search, matchup_sql, fresh_future_inputs
from grade_winners import evaluate


def sample_rows():
    now=pd.Timestamp.now(tz='UTC')
    ds=[]
    for season,complete in [(2026,True),(2027,False)]:
        tip=(pd.Timestamp('2025-11-20T20:00:00Z') if complete else now+pd.Timedelta(days=1))
        row={'game_id':str(100+season),'season':season,'game_date':tip.date().isoformat(),
             'start_time_utc':tip.isoformat(),'home_team_id':'11','away_team_id':'12',
             'neutral_site':0,'home_score':70 if complete else None,'away_score':68 if complete else None,
             'completed':1 if complete else 0}
        for side,scale in [('home',1),('away',-1)]:
            for k in TEAM_FIELDS:
                row[side+'_'+k]=None
            row[side+'_games_before']=9
            row[side+'_elo_rating']=1500+scale*40
            row[side+'_adjusted_net_per100']=scale*6
            row[side+'_adjusted_offense_per100']=103+scale*2
            row[side+'_adjusted_defense_per100']=103-scale*4
            row[side+'_opponent_elo_mean']=1500
            row[side+'_average_points_for']=75+scale*5
            row[side+'_average_points_against']=69-scale*3
            row[side+'_estimated_possessions']=69
            row[side+'_effective_fg_pct']=.53+scale*.02
            row[side+'_three_point_rate']=.36
            row[side+'_turnover_per_possession']=.14
            row[side+'_free_throw_rate']=.25
            row[side+'_rest_days']=2
            row[side+'_recorded_at']=tip-pd.Timedelta(days=1,hours=1)
            row[side+'_tipoff_utc']=tip
            row[side+'_game_day']=tip.date()
            row[side+'_quality_flags']=['reconstructed_not_saved_pretip'] if complete else []
        ds.append(row)
    return pd.DataFrame(ds)

class WinnerTests(unittest.TestCase):
    def test_columns_do_not_contain_results(self):
        forbidden=['score','actual','result','correct','win','label']
        self.assertFalse([name for name in FULL if any(t in name.lower() for t in forbidden)])
    def test_source_uses_separate_prospective_snapshots(self):
        sql=matchup_sql()
        self.assertIn("h.snapshot_kind='historical_replay'",sql)
        self.assertIn('h.team_id=g.home_team_id::text',sql)
        self.assertIn('a.team_id=g.away_team_id::text',sql)
    def test_features_and_label(self):
        d=make_features(sample_rows())
        self.assertEqual(d.net_gap.tolist(),[12,12])
        self.assertEqual(d.elo_gap.tolist(),[80,80])
        self.assertEqual(d.home_win.dropna().tolist(),[1.0])
    def test_invalid_duplicates_rejected(self):
        raw=sample_rows()
        with self.assertRaisesRegex(ValueError,'Duplicate'):
            make_features(pd.concat([raw,raw.iloc[:1]],ignore_index=True))
    def test_history_does_not_contain_future(self):
        h=valid_history(make_features(sample_rows()))
        self.assertEqual(len(h),1)
        self.assertEqual(h.season.iloc[0],2026)
    def test_future_requires_strict_time_cutoff(self):
        now=pd.Timestamp.now(tz='UTC')
        d=make_features(sample_rows())
        future=valid_future(d,2027,now)
        self.assertEqual(len(future),1)
        d.loc[d.season.eq(2027),'home_recorded_at']=d.loc[d.season.eq(2027),'tipoff']
        self.assertEqual(len(valid_future(d,2027,now)),0)
    def test_expired_future_never_scored(self):
        d=make_features(sample_rows())
        later=pd.Timestamp.now(tz='UTC')+pd.Timedelta(days=4)
        self.assertEqual(len(valid_future(d,2027,later)),0)
    def test_all_picks_counted(self):
        scored=pd.DataFrame({'home_win':[1,0,1,0]})
        report=describe_predictions(scored,np.array([.6,.7,.3,.4]),'test')
        self.assertEqual(report['games'],4)
        self.assertEqual(report['correct'],2)
        self.assertEqual(report['accuracy_pct'],50)
    def test_no_sneaky_retraining_if_not_enough_history(self):
        with self.assertRaisesRegex(ValueError,'Insufficient history'):
            model_search(valid_history(make_features(sample_rows())))
    def test_live_refresh_uses_newly_generated_features(self):
        raw=sample_rows().iloc[1].to_dict()
        now=pd.Timestamp.now(tz='UTC')
        # Two fresh snapshots, not the database's old frozen prospective rows.
        tip=now+pd.Timedelta(hours=2)
        raw['start_time_utc']=tip.isoformat()
        raw['game_date']=tip.date().isoformat()
        for side in ('home','away'):
            raw[side+'_tipoff_utc']=tip
            raw[side+'_game_day']=tip.date()
        current_game={k:raw[k] for k in ('game_id','season','game_date','start_time_utc',
                     'home_team_id','away_team_id','neutral_site','home_score',
                     'away_score','completed')}
        teams=[]
        for side in ('home','away'):
            t={'game_id':raw['game_id'],'team_id':raw[side+'_team_id'],
               'snapshot_kind':'prospective'}
            for field in TEAM_FIELDS:
                if field!='recorded_at':t[field]=raw[side+'_'+field]
            teams.append(t)
        fake=types.ModuleType('db_upgrade_v2')
        fake.fetch_source=lambda *args:([current_game], {}, {}, {})
        fake.generate=lambda *args,**kwargs:(teams, [], {'source_games':1})
        with patch.dict(sys.modules,{'db_upgrade_v2':fake}):
            live, audit=fresh_future_inputs('dummy_url',2027,now.to_pydatetime())
        self.assertEqual(len(live),1)
        self.assertEqual(audit['fresh_feature_pairs_generated'],1)
        self.assertEqual(live['home_recorded_at'].iloc[0], now.to_pydatetime())

    def test_grade_real_pregame_only(self):
        d=pd.DataFrame([{'game_id':'1','created_at':'2026-11-01T10:00:00Z',
                         'tipoff_utc':'2026-11-01T12:00:00Z','home_team_id':'H','away_team_id':'A',
                         'predicted_winner_team_id':'H','home_score':71,'away_score':69,'completed':1},
                        {'game_id':'2','created_at':'2026-11-02T10:00:00Z',
                         'tipoff_utc':'2026-11-02T12:00:00Z','home_team_id':'H','away_team_id':'A',
                         'predicted_winner_team_id':'A','home_score':None,'away_score':None,'completed':0}])
        _, report=evaluate(d)
        self.assertEqual(report['winner_accuracy_pct'],100.0)
        self.assertEqual(report['graded_games'],1)
        self.assertEqual(report['ungraded_games'],1)
    def test_grade_refuses_late_forecast(self):
        d=pd.DataFrame([{'game_id':'x','created_at':'2026-11-03T13:00:00Z',
                         'tipoff_utc':'2026-11-03T12:00:00Z','home_team_id':'H','away_team_id':'A',
                         'predicted_winner_team_id':'H','home_score':74,'away_score':69,'completed':1}])
        with self.assertRaisesRegex(ValueError,'not before tipoff'):
            evaluate(d)

if __name__=='__main__':unittest.main()
