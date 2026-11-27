import sys
import unittest
from pathlib import Path
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import espn_probability_pilot as pilot

class TestESPNResearch(unittest.TestCase):
    def test_projection_percentages(self):
        self.assertAlmostEqual(pilot.as_probability('62.5','37.5'),.625)
        self.assertAlmostEqual(pilot.as_probability('0.42','0.58'),.42)
        self.assertIsNone(pilot.as_probability('80','60'))
        self.assertIsNone(pilot.as_probability(None,40))

    def test_parser_safely_rejects_no_predictor(self):
        self.assertEqual(pilot.parse_espn_summary({'winprobability':[{'homeWinPercentage':.99}]},'4','9')[1], 'no_predictor')

    def test_parser_checks_ids(self):
        data={'predictor':{'homeTeam':{'id':'4','gameProjection':'59.1'},
                           'awayTeam':{'id':'9','gameProjection':'40.9'}}}
        self.assertEqual(pilot.parse_espn_summary(data,'4','9'),(.591,'found'))
        self.assertEqual(pilot.parse_espn_summary(data,'9','4')[1],'team_id_mismatch')
        self.assertEqual(pilot.parse_espn_summary(data,'4','8')[1],'team_id_mismatch')

    def test_balanced_sample(self):
        frame=pd.DataFrame({'game_id':[str(i) for i in range(40)],
                            'competitive':[True]*20+[False]*20})
        a=pilot.sample_balanced(frame,10)
        b=pilot.sample_balanced(frame,10)
        self.assertEqual(len(a),10)
        self.assertEqual(int(a.competitive.sum()),5)
        self.assertEqual(a.game_id.tolist(),b.game_id.tolist())

    def test_metrics_and_calibration(self):
        frame=pd.DataFrame({'home_win':[1,0,1,0], 'competitive':[True,False,True,False],
             'neutral_site':[0,0,1,1], 'espn_home_probability':[.8,.2,.7,.3],
             'selected_home_win_probability':[.9,.1,.6,.4],
             'v8_hybrid_home_win_prob':[.3,.7,.8,.2],
             'v8_strict_home_win_prob':[.5,.5,.6,.4]})
        scores,cals=pilot.evaluate(frame)
        self.assertEqual(len(scores.loc[scores.group.eq('all_pilot_matches')]),4)
        self.assertEqual(scores.loc[(scores.group=='all_pilot_matches') &
                       (scores.model=='ESPN retrieved predictor'),'accuracy_pct'].iloc[0],100.)
        self.assertFalse(cals.empty)

    def test_empty_espn_data_is_not_imputed(self):
        scores,cals=pilot.evaluate(pd.DataFrame())
        self.assertTrue(scores.empty)
        self.assertTrue(cals.empty)

    def test_historical_provenance_flag(self):
        # Structural guarantee: retrieved historical forecasts are always marked
        # timing-unverified, never labeled as saved pretipoff.
        import inspect
        src=inspect.getsource(pilot.fetch_forecast)
        self.assertIn("'historical_timing_verified':False",src)

if __name__=='__main__':
    unittest.main()

# End-to-end report smoke test: no live ESPN request, no Supabase write.
class TestReportExport(unittest.TestCase):
    def test_report_with_and_without_matching_predictors(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            chosen = pd.DataFrame({
                'game_id':['1','2'], 'season':[2026,2026],
                'competitive':[True,False], 'neutral_site':[1,0], 'home_win':[1,0],
                'selected_home_win_probability':[.6,.3],
                'v8_hybrid_home_win_prob':[.7,.4], 'v8_strict_home_win_prob':[.8,.2],
                'home_team_id':['4','8'], 'away_team_id':['9','3'],
            })
            retrieved = pd.DataFrame({
                'game_id':['1','2'], 'retrieved_at_utc':['2026-11-27T00:00:00+00:00']*2,
                'source_url':['https://example.test/1','https://example.test/2'],
                'source_field':['predictor.homeTeam.gameProjection']*2,
                'historical_timing_verified':[False]*2,
                'status':['found','no_predictor'],
                'espn_home_probability':[.66,None],
            })
            merged=chosen.merge(retrieved[['game_id','status','espn_home_probability']],on='game_id',how='inner')
            score,cal=pilot.evaluate(merged.loc[merged.status.eq('found')])
            output=Path(tmp)/'report'
            audit=pilot.write_report(output,chosen,retrieved,score,cal,chosen)
            self.assertEqual(audit['espn_forecasts_found'],1)
            self.assertEqual(audit['retrieval_statuses']['no_predictor'],1)
            self.assertEqual(len(pd.read_csv(output/'paired_model_metrics.csv').loc[lambda x:x.group.eq('all_pilot_matches')]),4)
            retrieved['status']=['no_predictor','no_predictor']
            retrieved['espn_home_probability']=[None,None]
            score,cal=pilot.evaluate(pd.DataFrame())
            output=Path(tmp)/'none'
            audit=pilot.write_report(output,chosen,retrieved,score,cal,chosen)
            self.assertEqual(audit['espn_forecasts_found'],0)
            self.assertIn('unavailable for all', (output/'README_RESULTS.md').read_text())
