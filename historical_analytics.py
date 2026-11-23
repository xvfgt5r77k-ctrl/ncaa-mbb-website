"""Opponent-adjusted NCAA MBB strength-of-schedule and historical trends.

Run on GitHub after a successful ESPN import, or manually with:
    python historical_analytics.py

Requires existing DATABASE_URL. This produces descriptive statistics and a
historical, date-by-date view of team ratings, not a model or a game forecast.
No ESPN source tables are modified.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from itertools import groupby
from math import isfinite
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS team_adjusted_ratings (
  season INTEGER NOT NULL,
  team_id TEXT NOT NULL REFERENCES teams(team_id),
  games_played INTEGER NOT NULL,
  adjusted_offense DOUBLE PRECISION,
  adjusted_defense DOUBLE PRECISION,
  adjusted_net DOUBLE PRECISION,
  raw_net DOUBLE PRECISION,
  opponent_sos DOUBLE PRECISION,
  updated_at TEXT NOT NULL,
  PRIMARY KEY(season,team_id)
);
CREATE INDEX IF NOT EXISTS idx_adjusted_ratings_season
  ON team_adjusted_ratings(season, adjusted_net DESC);

CREATE TABLE IF NOT EXISTS historical_team_snapshots (
  season INTEGER NOT NULL,
  snapshot_date TEXT NOT NULL,
  team_id TEXT NOT NULL REFERENCES teams(team_id),
  games_played INTEGER NOT NULL,
  adjusted_offense DOUBLE PRECISION,
  adjusted_defense DOUBLE PRECISION,
  adjusted_net DOUBLE PRECISION,
  raw_net DOUBLE PRECISION,
  opponent_sos DOUBLE PRECISION,
  updated_at TEXT NOT NULL,
  PRIMARY KEY(season,snapshot_date,team_id)
);
CREATE INDEX IF NOT EXISTS idx_historical_snapshots_team
  ON historical_team_snapshots(season,team_id,snapshot_date);

ALTER TABLE team_adjusted_ratings ENABLE ROW LEVEL SECURITY;
ALTER TABLE historical_team_snapshots ENABLE ROW LEVEL SECURITY;
"""

SOURCE_SQL = """
SELECT g.game_id,g.season,g.game_date,g.home_team_id,g.away_team_id,
       g.home_score,g.away_score,g.neutral_site,
       th.is_d1 AS home_is_d1, ta.is_d1 AS away_is_d1,
       hs.points AS home_points,hs.possessions AS home_possessions,
       ax.points AS away_points,ax.possessions AS away_possessions
FROM games g
JOIN teams th ON th.team_id=g.home_team_id
JOIN teams ta ON ta.team_id=g.away_team_id
JOIN team_game_stats hs ON hs.game_id=g.game_id AND hs.team_id=g.home_team_id
JOIN team_game_stats ax ON ax.game_id=g.game_id AND ax.team_id=g.away_team_id
WHERE g.completed=1 AND g.season IS NOT NULL
  AND g.home_score IS NOT NULL AND g.away_score IS NOT NULL
  AND hs.possessions>0 AND ax.possessions>0
  AND hs.points IS NOT NULL AND ax.points IS NOT NULL
ORDER BY g.season,g.game_date,g.game_id
"""

RATING_COLUMNS = ("season", "team_id", "games_played", "adjusted_offense",
                  "adjusted_defense", "adjusted_net", "raw_net", "opponent_sos",
                  "updated_at")
SNAPSHOT_COLUMNS = ("season", "snapshot_date", "team_id", "games_played",
                    "adjusted_offense", "adjusted_defense", "adjusted_net",
                    "raw_net", "opponent_sos", "updated_at")


def _clean_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Reject incomplete or non-finite game boxes, without modifying raw data."""
    out = []
    for row in rows:
        try:
            if not row["game_id"] or not row["game_date"] or not row["season"]:
                continue
            if row["home_team_id"] == row["away_team_id"]:
                continue
            if any(row[key] is None for key in ("home_score", "away_score",
                                                      "home_points", "away_points")):
                continue
            if not all(isfinite(float(row[key])) and float(row[key]) > 0
                       for key in ("home_possessions", "away_possessions")):
                continue
            if float(row["home_points"]) < 0 or float(row["away_points"]) < 0:
                continue
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
        out.append(row)
    return sorted(out, key=lambda r: (int(r["season"]), str(r["game_date"]), str(r["game_id"])))


def _accumulate(games: list[dict[str, Any]]) -> dict[str, dict]:
    teams = defaultdict(lambda: {"for": 0., "against": 0., "poss": 0.,
                                 "n": 0, "opp": [], "tempo_sum": 0.})
    for g in games:
        home, away = str(g["home_team_id"]), str(g["away_team_id"])
        hp, ap = float(g["home_points"]), float(g["away_points"])
        hpos, apos = float(g["home_possessions"]), float(g["away_possessions"])
        for team, opponent, scored, conceded, poss in (
            (home, away, hp, ap, hpos), (away, home, ap, hp, apos)
        ):
            t = teams[team]
            t["for"] += scored
            t["against"] += conceded
            t["poss"] += poss
            t["tempo_sum"] += poss
            t["n"] += 1
            t["opp"].append(opponent)
    return dict(teams)


def adjusted_ratings(games: list[dict[str, Any]], *, prior_games: int = 5,
                     iterations: int = 18) -> dict[str, dict[str, float]]:
    """Iterated opponent correction with small-sample shrinkage.

    Both sides of each matchup are in the input. The function works on one
    season and only on games already completed by the selected cutoff date.
    """
    agg = _accumulate(games)
    if not agg:
        return {}
    total_points = sum(t["for"] for t in agg.values())
    total_poss = sum(t["poss"] for t in agg.values())
    baseline = 100.0 * total_points / total_poss
    raw = {tid: {"off": 100.0*t["for"]/t["poss"],
                 "def": 100.0*t["against"]/t["poss"]}
           for tid, t in agg.items()}
    offense = {tid: baseline for tid in agg}
    defense = {tid: baseline for tid in agg}
    for _ in range(iterations):
        new_off, new_def = {}, {}
        for tid, t in agg.items():
            opponents = t["opp"]
            strength = t["n"] / (t["n"] + prior_games)
            opp_def = sum(defense[opp] - baseline for opp in opponents)/len(opponents)
            opp_off = sum(offense[opp] - baseline for opp in opponents)/len(opponents)
            target_off = baseline + strength * (raw[tid]["off"] - baseline - opp_def)
            target_def = baseline + strength * (raw[tid]["def"] - baseline - opp_off)
            new_off[tid] = 0.65*target_off + 0.35*offense[tid]
            new_def[tid] = 0.65*target_def + 0.35*defense[tid]
        offense, defense = new_off, new_def
    results = {}
    for tid, t in agg.items():
        opponents = t["opp"]
        results[tid] = {
            "games": t["n"],
            "off": offense[tid], "def": defense[tid],
            "net": offense[tid] - defense[tid],
            "raw_net": raw[tid]["off"] - raw[tid]["def"],
            "sos": sum(offense[o] - defense[o] for o in opponents) / len(opponents),
            "tempo": t["tempo_sum"] / t["n"],
        }
    return results


def build_analytics(rows: list[dict[str, Any]], generated_at: str | None = None):
    """Return (current team-season ratings, historical end-of-day snapshots).

    The snapshots include only dates on which the team played. Each snapshot
    uses games through and including its own date, never later games.
    """
    data = _clean_rows(rows)
    generated_at = generated_at or datetime.now(timezone.utc).isoformat(timespec="seconds")
    rating_rows, snapshots = [], []
    for season, season_iter in groupby(data, key=lambda g: int(g["season"])):
        season_games = list(season_iter)
        previous: list[dict[str, Any]] = []
        for game_date, day_iter in groupby(season_games,
                                            key=lambda g: str(g["game_date"])):
            todays_games = list(day_iter)
            previous.extend(todays_games)
            asof = adjusted_ratings(previous)
            participants = set()
            for game in todays_games:
                if game["home_is_d1"]:
                    participants.add(str(game["home_team_id"]))
                if game["away_is_d1"]:
                    participants.add(str(game["away_team_id"]))
            for team_id in sorted(participants):
                row = asof.get(team_id)
                if row is None:
                    continue
                fields = {
                    "season": season, "snapshot_date": game_date,
                    "team_id": team_id, "games_played": row["games"],
                    "adjusted_offense": row["off"],
                    "adjusted_defense": row["def"],
                    "adjusted_net": row["net"],
                    "raw_net": row["raw_net"],
                    "opponent_sos": row["sos"],
                    "updated_at": generated_at,
                }
                snapshots.append(tuple(fields[col] for col in SNAPSHOT_COLUMNS))
        final = adjusted_ratings(previous)
        d1_team_ids = {
            str(g["home_team_id"]) for g in season_games if g["home_is_d1"]
        } | {
            str(g["away_team_id"]) for g in season_games if g["away_is_d1"]
        }
        for team_id in sorted(d1_team_ids):
            row = final.get(team_id)
            if row is None:
                continue
            fields = {
                "season": season, "team_id": team_id,
                "games_played": row["games"],
                "adjusted_offense": row["off"],
                "adjusted_defense": row["def"],
                "adjusted_net": row["net"],
                "raw_net": row["raw_net"],
                "opponent_sos": row["sos"],
                "updated_at": generated_at,
            }
            rating_rows.append(tuple(fields[col] for col in RATING_COLUMNS))
    return rating_rows, snapshots


def _insert_rows(cur, table: str, columns: tuple[str, ...], rows: list[tuple]):
    if not rows:
        return
    statement = (f"INSERT INTO {table} ({','.join(columns)}) VALUES "
                 f"({','.join('%s' for _ in columns)})")
    for idx in range(0, len(rows), 500):
        cur.executemany(statement, rows[idx:idx+500])


def publish():
    from psycopg.rows import dict_row
    from cloud_store import open_cloud_connection

    # Fetch and compute before modifying analytical tables. Schema creation,
    # deletion and insertion are committed as a single transaction.
    with open_cloud_connection() as connection:
        with connection.cursor(row_factory=dict_row) as cur:
            cur.execute(SOURCE_SQL)
            source_rows = cur.fetchall()
            rating_rows, snapshots = build_analytics(source_rows)
            cur.execute(SCHEMA)
            cur.execute("DELETE FROM historical_team_snapshots")
            cur.execute("DELETE FROM team_adjusted_ratings")
            _insert_rows(cur, "team_adjusted_ratings", RATING_COLUMNS, rating_rows)
            _insert_rows(cur, "historical_team_snapshots", SNAPSHOT_COLUMNS, snapshots)
    print(f"Historical analytics refreshed: {len(rating_rows)} D1 team-season ratings, "
          f"{len(snapshots)} date-by-date team snapshots.")
    if not snapshots:
        print("No completed games with both team box scores yet; "
              "raw ESPN data is untouched.")


if __name__ == "__main__":
    publish()
