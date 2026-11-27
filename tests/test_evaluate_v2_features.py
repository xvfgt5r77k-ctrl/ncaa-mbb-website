"""Deterministic offline tests; no Supabase or real-game forecasts."""
import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from evaluate_v2_features import ATTRS, prepare_games, evaluate, archive_comparison


def fixture(n_per_year=125, seed=17):
    rng=np.random.default_rng(seed)
    result=[]
    for year in range(2022,2027):
        for i in range(n_per_year):
            date = datetime(year-1,11,1,tzinfo=timezone.utc) + timedelta(days=i//3)
            row={'game_id':f'{year}{i:06d}','season':year,'game_date':date.date().isoformat(),
                 'start_time_utc':(date+timedelta(hours=21)).isoformat(),
                 'home_team_id':str(i*2+10),'away_team_id':str(i*2+11),
                 'neutral_site':int(i%5==0)}
            strength=rng.normal(0,12)
            home_adv=0 if row['neutral_site'] else 2.5
            p=1/(1+np.exp(-(strength+home_adv)/15))
            home_win=int(rng.random()<p)
            margin=int(2+rng.poisson(7))*(1 if home_win else -1)
            row['home_score']=75+max(margin,0)
            row['away_score']=75+max(-margin,0)
            for side,sgn in [('home',1),('away',-1)]:
                net=sgn*strength/2+rng.normal(0,3)
                row[f'{side}_game_day']=row['game_date']
                row[f'{side}_tipoff_utc']=row['start_time_utc']
                row[f'{side}_games_before']=int(rng.integers(5,30))
                row[f'{side}_elo_rating']=1500+sgn*strength*2 +rng.normal(0,10)
                row[f'{side}_adjusted_offense_per100']=100+net/2
                row[f'{side}_adjusted_defense_per100']=100-net/2
                row[f'{side}_adjusted_net_per100']=net
                row[f'{side}_estimated_possessions']=70+rng.normal(0,3)
                row[f'{side}_effective_fg_pct']=.5 + sgn*strength/700+rng.normal(0,.02)
                row[f'{side}_three_point_rate']=.35 + rng.normal(0,.04)
                row[f'{side}_turnover_per_possession']=.15+rng.normal(0,.02)
                row[f'{side}_free_throw_rate']=.24+rng.normal(0,.03)
                row[f'{side}_rest_days']=int(rng.integers(0,7))
                good=rng.random()>.10
                row[f'{side}_player_coverage']='sufficient' if good else 'partial'
                row[f'{side}_expected_rotation_points_per40']=13+sgn*strength/30+rng.normal(0,1)
                row[f'{side}_rotation_top3_minutes_share']=.4+rng.normal(0,.03)
                row[f'{side}_rotation_continuity']=.7+rng.normal(0,.07)
                row[f'{side}_feature_version']='v2.0'
                row[f'{side}_quality_flags']=['reconstructed_not_saved_pretipoff']
            result.append(row)
    return pd.DataFrame(result)


class EvaluationTests(unittest.TestCase):
    def test_valid_fixture(self):
        prepared,audit=prepare_games(fixture())
        self.assertEqual(len(prepared),625)
        self.assertEqual(audit['excluded_total_unique_games'],0)
        self.assertGreater(prepared.player_complete_for_comparison.mean(),.7)

    def test_bad_tipoff_removed(self):
        d=fixture(20); d.loc[2,'home_tipoff_utc']='2025-01-01T15:00:00+00:00'
        clean,audit=prepare_games(d)
        self.assertEqual(len(clean),99)
        self.assertEqual(audit['missing_or_changed_tipoff'],1)

    def test_missing_replay_flag_removed(self):
        d=fixture(20);d.at[0,'away_quality_flags']=[]
        clean,audit=prepare_games(d)
        self.assertEqual(len(clean),99)
        self.assertEqual(audit['missing_replay_provenance_flag'],1)

    def test_inconsistent_net_removed(self):
        d=fixture(20);d.at[0,'home_adjusted_net_per100']=100
        clean,audit=prepare_games(d)
        self.assertEqual(len(clean),99)
        self.assertEqual(audit['inconsistent_or_missing_adjusted_net'],1)

    def test_selection_not_based_on_2026(self):
        clean,_=prepare_games(fixture(125))
        first,*_=evaluate(clean)
        changed=clean.copy()
        flip=changed.season.eq(2026)
        changed.loc[flip,'home_win']=1-changed.loc[flip,'home_win']
        second,*_=evaluate(changed)
        self.assertEqual(first['selection'],second['selection'])
        self.assertEqual(first['validation_selected'],second['validation_selected'])

    def test_archived_label_mismatch_not_compared(self):
        data,_=prepare_games(fixture(5))
        test=data.loc[data.season.eq(2026)].copy()
        archive=pd.DataFrame({'game_id':test.game_id,'season':2026,
                 'home_win':test.home_win.to_numpy(),
                 'v8_hybrid_home_win_prob':.6,'v8_strict_home_win_prob':.6})
        archive.loc[archive.index[0],'home_win'] = 1-archive.loc[archive.index[0],'home_win']
        paired,audit=archive_comparison(test,np.full(len(test),.5),archive)
        self.assertEqual(len(paired),len(test)-1)
        self.assertEqual(audit['archive_outcome_mismatches_excluded'],1)

    def test_full_report_export(self):
        import tempfile
        from evaluate_v2_features import write_report
        from pathlib import Path
        clean, audit = prepare_games(fixture(125))
        result, val, ret, subset, final, paired, subgroup = evaluate(clean)
        with tempfile.TemporaryDirectory() as folder:
            write_report(Path(folder), result, audit, val, ret, subset, paired, subgroup)
            content=(Path(folder)/'report.html').read_text()
            self.assertIn('2022–2026', content)
            self.assertIn('Historical reconstruction',content)

    def test_no_player_history_does_not_invent_availability(self):
        d=fixture(10); d['home_player_coverage']='unavailable'
        clean,_=prepare_games(d)
        self.assertEqual(int(clean.both_player_sufficient.sum()),0)
        self.assertTrue(clean.rotation_points_gap.isna().all())


if __name__=='__main__':unittest.main()
