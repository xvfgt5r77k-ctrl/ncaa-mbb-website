import unittest
from datetime import datetime, timedelta, timezone

from prospective_monitor import assess_snapshot, make_report, metrics

NOW = datetime(2026, 11, 25, 12, tzinfo=timezone.utc)
TIP = NOW - timedelta(days=1)


def row(**changes):
    r = dict(game_id='100', season=2027, model_version='V1',
             snapshot_stage='initial', recorded_at=TIP - timedelta(days=3),
             tipoff_utc=TIP, snap_home_id='1', snap_away_id='2',
             snap_neutral_site=0, home_games=6, away_games=6,
             projected_home_points=75, projected_away_points=70,
             projected_margin=5, projected_total=145, matched_game_id='100',
             game_tipoff_utc=TIP, game_home_id='1', game_away_id='2',
             game_neutral_site=0, game_completed='1', home_score=80, away_score=74)
    r.update(changes)
    return r


class ProspectiveTests(unittest.TestCase):
    def test_valid_game_only_after_final(self):
        a = assess_snapshot(row(), NOW)
        self.assertEqual(a['status'], 'evaluated')
        self.assertEqual(a['winner_correct'], 1)
        self.assertEqual(a['total_absolute_error'], 9)

    def test_early_after_tipoff_is_not_a_legal_snapshot(self):
        a = assess_snapshot(row(recorded_at=TIP + timedelta(minutes=1)), NOW)
        self.assertEqual(a['reason'], 'snapshot_not_pre_tipoff')

    def test_reschedule_and_changed_teams_excluded(self):
        a = assess_snapshot(row(game_tipoff_utc=TIP - timedelta(hours=2)), NOW)
        self.assertEqual(a['reason'], 'rescheduled_since_snapshot')
        b = assess_snapshot(row(game_away_id='another'), NOW)
        self.assertEqual(b['reason'], 'opponent_ids_changed')

    def test_no_score_counts_before_completion(self):
        a = assess_snapshot(row(game_completed='0', home_score=None, away_score=None), NOW)
        self.assertEqual(a['status'], 'awaiting_result')
        self.assertNotIn('winner_correct', a)

    def test_no_upcoming_predictions_in_result_export(self):
        a = assess_snapshot(row(game_completed='0', home_score=None, away_score=None,
                                tipoff_utc=NOW+timedelta(days=3),
                                game_tipoff_utc=NOW+timedelta(days=3)), NOW)
        self.assertEqual(a['status'], 'upcoming')
        self.assertNotIn('total_absolute_error', a)

    def test_admin_result_rejected(self):
        a = assess_snapshot(row(home_score=0, away_score=2), NOW)
        self.assertEqual(a['reason'], 'administrative_score')

    def test_inconsistent_total_rejected(self):
        a = assess_snapshot(row(projected_total=200), NOW)
        self.assertEqual(a['reason'], 'inconsistent_forecast')

    def test_duplicates_excluded(self):
        a = assess_snapshot(row(), NOW, duplicates=True)
        self.assertEqual(a['reason'], 'duplicate_model_stage_game')

    def test_stages_do_not_inflate_initial_numerator(self):
        a = row()
        b = row(snapshot_stage='within_24h', recorded_at=TIP-timedelta(hours=4))
        report, completed = make_report([a,b], [dict(game_id='100',game_tipoff_utc=TIP, game_completed='1')], 2027, NOW)
        self.assertEqual(report['headline_initial_v1']['evaluated_games'], 1)
        self.assertEqual(len(completed), 2)
        self.assertEqual(len(report['per_model_and_stage']), 2)

    def test_forecast_tie_denominator(self):
        tied = assess_snapshot(row(projected_home_points=72.5, projected_away_points=72.5,
                                   projected_margin=0, projected_total=145), NOW)
        stat = metrics([tied])
        self.assertEqual(stat['evaluated_games'], 1)
        self.assertEqual(stat['winner_evaluable'], 0)
        self.assertIsNone(stat['winner_accuracy_pct'])


if __name__ == '__main__':
    unittest.main()
