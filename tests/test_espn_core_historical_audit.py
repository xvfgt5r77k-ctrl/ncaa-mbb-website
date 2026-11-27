"""Offline fixture tests, no internet access or credentials required."""
import unittest
from unittest.mock import Mock

from espn_core_historical_audit import (
    _team_id, _projection, parse_core_predictor, fetch_core_predictor,
)


def teams(h=62.5,a=37.5):
    return {
        'homeTeam':{'team':{'$ref':'https://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/teams/12'},
                    'statistics':[{'name':'gameProjection','value':h}]},
        'awayTeam':{'team':{'$ref':'https://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/teams/13'},
                    'statistics':[{'name':'gameProjection','value':a}]},
    }

class CoreTests(unittest.TestCase):
    def test_core_schema_percent(self):
        p,s=parse_core_predictor(teams(),'12','13')
        self.assertEqual(s,'found')
        self.assertAlmostEqual(p,.625)

    def test_core_schema_fraction(self):
        p,s=parse_core_predictor(teams(.625,.375),'12','13')
        self.assertEqual(s,'found')
        self.assertAlmostEqual(p,.625)

    def test_explicit_stat_names_only(self):
        x=teams();x['homeTeam']['statistics'][0]['name']='predictedScore'
        p,s=parse_core_predictor(x,'12','13')
        self.assertIsNone(p)
        self.assertEqual(s,'missing_or_invalid_projection')

    def test_identity_mismatch(self):
        p,s=parse_core_predictor(teams(),'13','12')
        self.assertIsNone(p)
        self.assertEqual(s,'team_identity_mismatch')

    def test_missing_id_rejected(self):
        x=teams();x['awayTeam'].pop('team')
        p,s=parse_core_predictor(x,'12','13')
        self.assertIsNone(p)
        self.assertEqual(s,'team_identity_unverifiable')

    def test_inconsistent_probabilities_rejected(self):
        p,s=parse_core_predictor(teams(90,85),'12','13')
        self.assertIsNone(p)
        self.assertEqual(s,'missing_or_invalid_projection')

    def test_object_statistic_form(self):
        x=teams();x['homeTeam']['statistics']={'gameProjection':{'value':'62.5%'}}
        x['awayTeam']['statistics']={'gameProjection':{'value':'37.5%'}}
        p,s=parse_core_predictor(x,'12','13')
        self.assertEqual(s,'found')
        self.assertAlmostEqual(p,.625)

    def test_no_endpoint_data(self):
        client=Mock();client.get.return_value.status_code=404
        row=fetch_core_predictor(client,'401234','12','13',pause=0)
        self.assertEqual(row['status'],'http_404')
        self.assertIsNone(row['espn_home_probability'])
        self.assertIn('401234/competitions/401234/predictor',row['source_url'])

    def test_invalid_json(self):
        resp=Mock();resp.status_code=200;resp.json.side_effect=ValueError('bad')
        client=Mock();client.get.return_value=resp
        row=fetch_core_predictor(client,'401234','12','13',pause=0)
        self.assertEqual(row['status'],'invalid_json')

    def test_current_game_indicators_not_misused(self):
        x=teams();x['homeTeam'].pop('statistics');x['awayTeam'].pop('statistics')
        x['winprobability']={'homeWinPercentage':.7}
        p,s=parse_core_predictor(x,'12','13')
        self.assertIsNone(p)
        self.assertEqual(s,'missing_or_invalid_projection')

if __name__=='__main__': unittest.main()
