import sys
from datetime import datetime,timedelta,timezone
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import research_opponent_adjustment_v3 as v3

NOW=datetime(2026,11,27,21,tzinfo=timezone.utc)

def game(gid,offset,h,a,hs=75,aw=70,completed=True,neutral=False):
    dt=datetime(2026,11,12,18,tzinfo=timezone.utc)+timedelta(days=offset)
    return {'game_id':gid,'game_date':dt.date().isoformat(),
        'start_time_utc':dt.isoformat(),'season':2027,'home_team_id':h,'away_team_id':a,
        'home_score':hs if completed else None,'away_score':aw if completed else None,
        'completed':int(completed),'neutral_site':int(neutral)}

def fixtures():
    games=[game('g1',0,'A','B'),game('g2',1,'B','C',60,73),
           game('g3',2,'A','C',87,78),game('g4',4,'C','A',70,68),
           game('same1',6,'B','A',68,67),game('same2',6,'C','A',72,73),
           game('next',16,'B','C',completed=False)]
    boxes={}
    for g in games:
        if g['completed']:
            for tid,score in ((g['home_team_id'],g['home_score']),(g['away_team_id'],g['away_score'])):
                boxes[(g['game_id'],tid)]={'possessions':70,'points':score}
    return games, boxes

class TestV3(unittest.TestCase):
    def test_same_day_equal(self):
        g,b=fixtures()
        rows,_,_=v3.generate(g,b,now=NOW)
        for tid in ('A',):
            one=next(r for r in rows if r['game_id']=='same1' and r['team_id']==tid)
            two=next(r for r in rows if r['game_id']=='same2' and r['team_id']==tid)
            for key in ('adjusted_net_per100','adjusted_offense_per100','games_before','history_games_in_fit'):
                self.assertEqual(one[key],two[key])

    def test_own_score_does_not_change_pregame(self):
        g,b=fixtures()
        before,_,_=v3.generate(g,b,now=NOW)
        g[2]['home_score']=120
        b[('g3','A')]['points']=120
        after,_,_=v3.generate(g,b,now=NOW)
        selected=lambda rows:next(r for r in rows if r['game_id']=='g3' and r['team_id']=='A')
        self.assertEqual(selected(before),selected(after))

    def test_older_opponent_game_changes_later_ratings(self):
        g,b=fixtures()
        before,_,_=v3.generate(g,b,now=NOW)
        g[1]['away_score']=95
        b[('g2','C')]['points']=95
        after,_,_=v3.generate(g,b,now=NOW)
        selected=lambda rows:next(r for r in rows if r['game_id']=='same1' and r['team_id']=='A')
        self.assertNotEqual(selected(before)['adjusted_net_per100'],selected(after)['adjusted_net_per100'])

    def test_missing_or_mismatched_possessions_rejected(self):
        g,b=fixtures()
        b[('g1','A')]['possessions']=81
        self.assertIsNone(v3.observe(v3.v2.d1_games_sorted(g)[0],b))
        rows,obs,stat=v3.generate(g,b,now=NOW)
        self.assertNotIn('g1',obs)
        self.assertEqual(stat['completed_without_usable_pair_boxes'],1)

    def test_future_prospective_no_future_information(self):
        g,b=fixtures()
        # Make next fixture future relative to the evaluation clock.
        for e in g:
            if e['game_id']=='next':
                e['game_date']='2026-12-01'
                e['start_time_utc']='2026-12-01T18:00:00+00:00'
        rows,_,_=v3.generate(g,b,now=NOW)
        future=[r for r in rows if r['game_id']=='next']
        self.assertEqual(len(future),2)
        self.assertTrue(all(r['snapshot_kind']=='prospective' for r in future))
        self.assertTrue(all(r['games_before']>0 for r in future))

    def test_fitted_inputs_must_be_before_day(self):
        g,b=fixtures()
        one=v3.v2.d1_games_sorted(g)[0]
        obs=v3.observe(one,b)
        with self.assertRaises(ValueError):
            v3.fit_before_day([obs],['A','B'],obs.game_day)

    def test_venue_shift_does_not_change_normalized_strength_direction(self):
        o=v3.Observation('g1',datetime(2026,11,10).date(),'A','B',115,85,0,68)
        neutral,_=v3.fit_before_day([o],['A','B'],datetime(2026,11,20).date())
        with_venue,_=v3.fit_before_day([v3.Observation('g1',o.game_day,'A','B',115,85,1,68)],
                                        ['A','B'],datetime(2026,11,20).date())
        self.assertGreater(neutral['A']['net'],neutral['B']['net'])
        self.assertGreater(with_venue['A']['net'],with_venue['B']['net'])
        self.assertLess(with_venue['A']['net'],neutral['A']['net'])

    def test_duplicate_or_bad_score_excluded(self):
        g,b=fixtures()
        g[0]['home_score'],g[0]['away_score']=0,2
        self.assertIsNone(v3.observe(v3.v2.d1_games_sorted(g)[0],b))

    def test_comparison_exact_matching(self):
        g,b=fixtures()
        rows,obs,_=v3.generate(g,b,now=NOW)
        v2rows={(x['game_id'],x['team_id']):{'offense':100.0,'defense':100.0}
                for x in rows if x['snapshot_kind']=='historical_replay'}
        paired,summary=v3.compare(rows,obs,v2rows)
        self.assertEqual(len(paired),6)
        self.assertEqual(sum(v['matched_games'] for v in summary.values()),6)

if __name__=='__main__':unittest.main()
