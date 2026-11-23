import os
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from db import TEAM_RANKINGS, connect
from espn import extract_team_box, game_from_event
from pipeline import sync_range

EVENT = {
    "id":"401900010", "date":"2026-11-23T01:00:00Z", "season":{"year":2027},
    "status":{"type":{"name":"STATUS_FINAL","completed":True,"description":"Final"}},
    "competitions":[{"neutralSite":False,"venue":{"fullName":"Example Arena"},
      "competitors":[
         {"homeAway":"home","team":{"id":"1","displayName":"Alpha University","abbreviation":"ALP"},"score":"80"},
         {"homeAway":"away","team":{"id":"2","displayName":"Beta University","abbreviation":"BET"},"score":"70"}
      ]}]
}


def team_stats(team_id, fg, three, ft, orebs, dreb, to, pts):
    return {"team":{"id":team_id},"statistics":[
       {"name":"fieldGoalsMade-fieldGoalsAttempted","displayValue":fg},
       {"name":"threePointFieldGoalsMade-threePointFieldGoalsAttempted","displayValue":three},
       {"name":"freeThrowsMade-freeThrowsAttempted","displayValue":ft},
       {"name":"offensiveRebounds","displayValue":str(orebs)},
       {"name":"defensiveRebounds","displayValue":str(dreb)},
       {"name":"turnovers","displayValue":str(to)},
       {"name":"points","displayValue":str(pts)}
    ]}

SUMMARY = {"boxscore":{
  "teams":[team_stats("1","28-60","8-20","16-20",10,25,12,80),
           team_stats("2","25-58","7-19","13-16",8,27,10,70)],
  "players":[
   {"team":{"id":"1"},"statistics":[{"labels":["MIN","FG","3PT","FT","REB","AST","TO","STL","BLK","PTS"],
     "athletes":[{"athlete":{"id":"111","displayName":"A Player","position":{"abbreviation":"G"}},"starter":True,"stats":["30","5-10","2-4","4-4","5","3","2","1","0","16"]}]}]},
   {"team":{"id":"2"},"statistics":[{"labels":["MIN","FG","3PT","FT","REB","AST","TO","STL","BLK","PTS"],
     "athletes":[{"athlete":{"id":"222","displayName":"B Player"},"starter":False,"stats":["20","4-9","1-3","2-2","7","2","1","0","0","11"]}]}]}
  ]
}}

class FakeESPN:
    def __init__(self):
        self.summary_calls=0
    def teams(self):
        return {"sports":[{"leagues":[{"teams":[{"team":{"id":"1","displayName":"Alpha University"}},
                                                     {"team":{"id":"2","displayName":"Beta University"}}]}]}]}
    def scoreboard(self, day):
        return {"events":[EVENT]} if day==date(2026,11,22) else {"events":[]}
    def summary(self, game_id):
        self.summary_calls+=1
        return SUMMARY

class TestDatabase(unittest.TestCase):
    def test_game_date_timezone(self):
        game=game_from_event(EVENT)
        self.assertEqual(game["game_date"],"2026-11-22")
        self.assertEqual(game["completed"],1)

    def test_efficiency_calculation(self):
        game=game_from_event(EVENT)
        rows=extract_team_box(SUMMARY,game)
        self.assertEqual(len(rows),2)
        # Alpha poss: 60 - 10 + 12 + .475*20 = 71.5
        # Beta poss:  58 - 8 + 10 + .475*16 = 67.6
        poss=(71.5+67.6)/2
        self.assertAlmostEqual(rows[0]["possessions"],poss)
        self.assertAlmostEqual(rows[0]["offensive_efficiency"],80/poss*100)
        self.assertAlmostEqual(rows[1]["defensive_efficiency"],80/poss*100)

    def test_import_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"test.sqlite3"
            fake=FakeESPN()
            first=sync_range(date(2026,11,22),date(2026,11,22),db_path=path,client=fake)
            self.assertEqual(first["status"],"ok")
            self.assertEqual(first["boxscores"],1)
            second=sync_range(date(2026,11,22),date(2026,11,22),db_path=path,client=fake)
            self.assertEqual(second["boxscores"],0)
            self.assertEqual(fake.summary_calls,1)
            conn=connect(path)
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM games").fetchone()[0],1)
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM team_game_stats").fetchone()[0],2)
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM player_game_stats").fetchone()[0],2)
            ranking=conn.execute(TEAM_RANKINGS).fetchall()
            self.assertEqual(ranking[0]["name"],"Alpha University")
            self.assertEqual(ranking[0]["wins"],1)
            self.assertAlmostEqual(ranking[0]["net_eff"],14.4,places=1)
            conn.close()

if __name__ == "__main__":
    unittest.main()
