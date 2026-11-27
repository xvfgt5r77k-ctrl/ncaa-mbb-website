import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import db_upgrade_v2 as v2

NOW = datetime(2026, 11, 26, 15, 0, tzinfo=timezone.utc)


def game(gid, when, home="A", away="B", completed=True, hs=75, aw=70):
    return {"game_id": gid, "season": 2027, "game_date": when.date().isoformat(),
            "start_time_utc": when.isoformat(), "home_team_id": home,
            "away_team_id": away, "neutral_site": 0, "completed": 1 if completed else 0,
            "home_score": hs if completed else None, "away_score": aw if completed else None}


def sample():
    d0 = datetime(2026,11,20,19,tzinfo=timezone.utc)
    games = [game("older", d0, "A", "B"),
             game("same_1", d0+timedelta(days=2), "A", "B", True, 63, 62),
             game("same_2", d0+timedelta(days=2, hours=2), "A", "C", True, 100, 95),
             game("future", NOW+timedelta(days=3), "A", "B", False),
             game("past_unfinished", d0+timedelta(days=1), "B", "C", False)]
    boxes = {}
    for g in games:
        if not g["completed"]:
            continue
        for team in [g["home_team_id"],g["away_team_id"]]:
            boxes[(g["game_id"],team)] = {"possessions":"68", "fg_m":"25",
                "fg_a":"55", "three_m":"7", "three_a":"22",
                "turnovers":"12", "off_rebounds":"9", "ft_a":"19"}
    players = {}
    for key in boxes:
        players[key] = [{"player_id":f"{key[1]}{i}","minutes":"40","points":"10"}
                        for i in range(5)]
    return games,boxes,players


class TestFeatureStore(unittest.TestCase):
    def test_future_uses_prior_games_only(self):
        g,b,p=sample()
        teams,player_rows,stats=v2.generate(g,b,p,now=NOW)
        f=next(x for x in teams if x["game_id"]=="future" and x["team_id"]=="A")
        self.assertEqual(f["snapshot_kind"], "prospective")
        self.assertEqual(f["games_before"],3)
        self.assertEqual(f["prior_player_box_games"],3)
        self.assertEqual(f["player_coverage"],"sufficient")
        self.assertEqual(len([x for x in player_rows if x["game_id"]=="future" and x["team_id"]=="A"]),5)
        self.assertTrue(stats["historical_replay_games"]>=3)

    def test_same_day_cannot_see_previous_same_day_game(self):
        g,b,p=sample()
        rows,_,_=v2.generate(g,b,p,now=NOW)
        a1=next(x for x in rows if x["game_id"]=="same_1" and x["team_id"]=="A")
        a2=next(x for x in rows if x["game_id"]=="same_2" and x["team_id"]=="A")
        for col in ("games_before","elo_rating","adjusted_net_per100",
                    "average_points_for","prior_player_box_games"):
            self.assertEqual(a1[col],a2[col],col)

    def test_a_games_own_final_score_does_not_affect_own_features(self):
        g,b,p=sample()
        first,_,_=v2.generate(g,b,p,now=NOW)
        g[1]["home_score"],g[1]["away_score"] = 110,35
        second,_,_=v2.generate(g,b,p,now=NOW)
        x=next(r for r in first if r["game_id"]=="same_1" and r["team_id"]=="A")
        y=next(r for r in second if r["game_id"]=="same_1" and r["team_id"]=="A")
        self.assertEqual(x,y)

    def test_no_fake_injury_from_absent_player_boxes(self):
        g,b,p=sample()
        rows, players, _=v2.generate(g,b,{},now=NOW)
        future=[r for r in rows if r["game_id"]=="future"]
        self.assertTrue(all(r["player_coverage"]=="unavailable" for r in future))
        self.assertTrue(all(r["expected_rotation_points_per40"] is None for r in future))
        self.assertFalse(players)

    def test_past_unfinished_game_not_a_snapshot(self):
        g,b,p=sample()
        rows, _,_=v2.generate(g,b,p,now=NOW)
        self.assertNotIn("past_unfinished",{r["game_id"] for r in rows})

    def test_missing_shooting_stats_not_zero(self):
        g,_,_=sample()
        rows,_,_=v2.generate(g,{}, {},now=NOW)
        future=next(x for x in rows if x["game_id"]=="future" and x["team_id"]=="A")
        self.assertIsNone(future["three_point_rate"])
        self.assertIsNone(future["free_throw_rate"])
        self.assertIsNone(future["turnover_per_possession"])

    def test_prospective_requires_trustworthy_future_tipoff(self):
        g,b,p=sample()
        g[-2]["start_time_utc"] = "2026-11-29 15:00:00"  # no tz
        rows,_,_=v2.generate(g,b,p,now=NOW)
        self.assertNotIn("future", {r["game_id"] for r in rows})

    def test_no_duplicate_pregame_keys(self):
        g,b,p=sample()
        rows, players,_=v2.generate(g,b,p,now=NOW)
        tkeys=[(r["game_id"],r["team_id"],r["snapshot_kind"]) for r in rows]
        pkeys=[(r["game_id"],r["team_id"],r["player_id"],r["snapshot_kind"]) for r in players]
        self.assertEqual(len(tkeys),len(set(tkeys)))
        self.assertEqual(len(pkeys),len(set(pkeys)))

    def test_player_history_only_prior_day(self):
        g,b,p=sample()
        rows, players,_=v2.generate(g,b,p,now=NOW)
        athlete=next(x for x in players if x["game_id"]=="future" and x["player_id"]=="A0")
        self.assertEqual(athlete["prior_games_seen"],3)
        self.assertEqual(athlete["prior_minutes"],120)
        self.assertTrue(athlete["prior_box_score_only"])

    def test_invalid_admin_score_never_updates_rating(self):
        g,b,p=sample()
        g[0]["home_score"],g[0]["away_score"] = 0,2
        rows,_,_=v2.generate(g,b,p,now=NOW)
        current=next(x for x in rows if x["game_id"]=="same_1" and x["team_id"]=="A")
        self.assertEqual(current["games_before"],0)


if __name__ == "__main__":
    unittest.main()
