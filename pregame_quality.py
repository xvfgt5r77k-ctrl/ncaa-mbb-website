"""Coverage, leak-safe pregame team history and out-of-season descriptive audit.

Does not change ESPN/source tables and does not generate wagers, game picks or
win probabilities. A walk-forward mean-score baseline tests historical scoring
stability, not a trained game outcome prediction model.

Run: python pregame_quality.py --start-season 2022 --end-season 2026
Needs the existing DATABASE_URL secret and cloud_store.py from the project.
"""
from __future__ import annotations

import argparse
from bisect import bisect_left
from collections import defaultdict, deque
from datetime import date, datetime, timezone
from itertools import groupby
from math import isfinite, sqrt
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS historical_coverage_audit (
  season INTEGER PRIMARY KEY,
  all_game_records INTEGER NOT NULL,
  completed_games INTEGER NOT NULL,
  not_completed_games INTEGER NOT NULL,
  d1_matchups INTEGER NOT NULL,
  complete_scores INTEGER NOT NULL,
  both_box_scores INTEGER NOT NULL,
  usable_stat_games INTEGER NOT NULL,
  score_mismatches INTEGER NOT NULL,
  incomplete_possessions INTEGER NOT NULL,
  missing_start_time INTEGER NOT NULL,
  generated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS pregame_team_features (
  season INTEGER NOT NULL,
  game_id TEXT NOT NULL REFERENCES games(game_id),
  game_date TEXT NOT NULL,
  team_id TEXT NOT NULL REFERENCES teams(team_id),
  opponent_id TEXT NOT NULL REFERENCES teams(team_id),
  is_home INTEGER NOT NULL,
  neutral_site INTEGER NOT NULL,
  games_before INTEGER NOT NULL,
  avg_points_for DOUBLE PRECISION,
  avg_points_against DOUBLE PRECISION,
  avg_possessions DOUBLE PRECISION,
  raw_offense DOUBLE PRECISION,
  raw_defense DOUBLE PRECISION,
  raw_net DOUBLE PRECISION,
  recent_five_margin DOUBLE PRECISION,
  rest_days INTEGER,
  adjusted_net_before DOUBLE PRECISION,
  opponent_sos_before DOUBLE PRECISION,
  rating_snapshot_date TEXT,
  generated_at TEXT NOT NULL,
  PRIMARY KEY (game_id,team_id),
  CONSTRAINT pregame_prior_snapshot_only CHECK (
    rating_snapshot_date IS NULL OR rating_snapshot_date < game_date
  )
);
CREATE INDEX IF NOT EXISTS idx_pregame_team_season_date
  ON pregame_team_features(season,team_id,game_date);
CREATE TABLE IF NOT EXISTS historical_score_baseline_audit (
  season INTEGER PRIMARY KEY,
  split_name TEXT NOT NULL,
  eligible_d1_games INTEGER NOT NULL,
  evaluated_games INTEGER NOT NULL,
  evaluated_team_scores INTEGER NOT NULL,
  points_mae DOUBLE PRECISION,
  points_rmse DOUBLE PRECISION,
  points_bias DOUBLE PRECISION,
  generated_at TEXT NOT NULL
);
ALTER TABLE historical_coverage_audit ENABLE ROW LEVEL SECURITY;
ALTER TABLE pregame_team_features ENABLE ROW LEVEL SECURITY;
ALTER TABLE historical_score_baseline_audit ENABLE ROW LEVEL SECURITY;
"""

# LEFT JOINs allow coverage auditing even when an ESPN box score is absent.
GAME_SQL = """
SELECT g.game_id, g.season, g.game_date, g.start_time_utc, g.completed,
       g.home_team_id, g.away_team_id, g.home_score, g.away_score,
       g.neutral_site, h.is_d1 AS home_is_d1, a.is_d1 AS away_is_d1,
       hs.points AS home_points, hs.possessions AS home_possessions,
       ax.points AS away_points, ax.possessions AS away_possessions
FROM games g
JOIN teams h ON h.team_id = g.home_team_id
JOIN teams a ON a.team_id = g.away_team_id
LEFT JOIN team_game_stats hs
  ON hs.game_id = g.game_id AND hs.team_id = g.home_team_id
LEFT JOIN team_game_stats ax
  ON ax.game_id = g.game_id AND ax.team_id = g.away_team_id
WHERE g.season BETWEEN %s AND %s
ORDER BY g.season, g.game_date, g.game_id
"""

SNAPSHOT_SQL = """
SELECT season, team_id, snapshot_date, adjusted_net, opponent_sos
FROM historical_team_snapshots
WHERE season BETWEEN %s AND %s
ORDER BY season, team_id, snapshot_date
"""

COVERAGE_COLS = (
    "season", "all_game_records", "completed_games", "not_completed_games",
    "d1_matchups", "complete_scores", "both_box_scores", "usable_stat_games",
    "score_mismatches", "incomplete_possessions", "missing_start_time", "generated_at"
)
FEATURE_COLS = (
    "season", "game_id", "game_date", "team_id", "opponent_id", "is_home",
    "neutral_site", "games_before", "avg_points_for", "avg_points_against",
    "avg_possessions", "raw_offense", "raw_defense", "raw_net",
    "recent_five_margin", "rest_days", "adjusted_net_before",
    "opponent_sos_before", "rating_snapshot_date", "generated_at"
)
BASELINE_COLS = (
    "season", "split_name", "eligible_d1_games", "evaluated_games",
    "evaluated_team_scores", "points_mae", "points_rmse", "points_bias",
    "generated_at"
)


def _day(s: Any) -> str | None:
    try:
        value = str(s)[:10]
        return date.fromisoformat(value).isoformat()
    except (ValueError, TypeError):
        return None


def _number(value: Any, *, positive: bool = False) -> float | None:
    try:
        n = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not isfinite(n) or (positive and n <= 0):
        return None
    return n


def _box_complete(game: dict) -> bool:
    return game.get("home_points") is not None and game.get("away_points") is not None


def _usable(game: dict) -> bool:
    return (bool(game.get("completed")) and _day(game.get("game_date")) is not None
            and game.get("home_team_id") != game.get("away_team_id")
            and all(_number(game.get(col)) is not None and _number(game.get(col)) >= 0
                    for col in ("home_score", "away_score", "home_points", "away_points"))
            and all(_number(game.get(col), positive=True) is not None
                    for col in ("home_possessions", "away_possessions"))
            and _number(game.get("home_points")) == _number(game.get("home_score"))
            and _number(game.get("away_points")) == _number(game.get("away_score")))


def _coverage(game_rows: list[dict], stamp: str):
    results = []
    for season, it in groupby(sorted(game_rows,
                                      key=lambda g: (int(g["season"]), str(g["game_date"]), str(g["game_id"]))),
                              key=lambda g: int(g["season"])):
        rows = list(it)
        completed = [g for g in rows if bool(g.get("completed"))]
        games_with_scores = [g for g in completed if g.get("home_score") is not None
                             and g.get("away_score") is not None]
        box_scores = [g for g in completed if _box_complete(g)]
        mismatch = sum(1 for g in games_with_scores if _box_complete(g)
                       and (_number(g["home_points"]) != _number(g["home_score"])
                            or _number(g["away_points"]) != _number(g["away_score"])))
        bad_pos = sum(1 for g in completed if _box_complete(g)
                      and (_number(g.get("home_possessions"), positive=True) is None
                           or _number(g.get("away_possessions"), positive=True) is None))
        record = dict(season=season, all_game_records=len(rows), completed_games=len(completed),
                      not_completed_games=len(rows)-len(completed),
                      d1_matchups=sum(bool(g.get("home_is_d1")) and bool(g.get("away_is_d1"))
                                      for g in completed),
                      complete_scores=len(games_with_scores), both_box_scores=len(box_scores),
                      usable_stat_games=sum(_usable(g) for g in completed),
                      score_mismatches=mismatch, incomplete_possessions=bad_pos,
                      missing_start_time=sum(not g.get("start_time_utc") for g in completed),
                      generated_at=stamp)
        results.append(tuple(record[c] for c in COVERAGE_COLS))
    return results


def _snapshot_index(snapshot_rows: list[dict]):
    """Date-sorted snapshots grouped by season/team for strict-prior retrieval."""
    idx = defaultdict(lambda: ([], []))
    for r in sorted(snapshot_rows, key=lambda r: (int(r["season"]),
                                                  str(r["team_id"]), str(r["snapshot_date"]))):
        d = _day(r.get("snapshot_date"))
        if d is None:
            continue
        key = (int(r["season"]), str(r["team_id"]))
        dates, values = idx[key]
        if dates and d <= dates[-1]:
            raise ValueError("Duplicate or non-increasing snapshot date for team")
        dates.append(d)
        values.append(r)
    return idx


def _lookup_prior(idx: dict, season: int, team: str, game_day: str):
    dates, values = idx.get((season, team), ([], []))
    pos = bisect_left(dates, game_day) - 1
    if pos < 0:
        return None
    found = values[pos]
    if not _day(found["snapshot_date"]) < game_day:
        raise AssertionError("A same-day or future snapshot reached pregame features")
    return found


def _state():
    return {"games": 0, "points_for": 0.0, "points_against": 0.0,
            "possessions": 0.0, "last5": deque(maxlen=5), "last_date": None}


def _feature(season, game, team, opponent, home, st, idx, stamp):
    when = _day(game["game_date"])
    n = st["games"]
    poss = st["possessions"]
    prior = _lookup_prior(idx, season, team, when)
    rest = (date.fromisoformat(when) - date.fromisoformat(st["last_date"])).days - 1 \
        if st["last_date"] else None
    if rest is not None and rest < 0:
        raise AssertionError("Encountered game before team's previous finished date")
    cols = dict(
        season=season, game_id=str(game["game_id"]), game_date=when,
        team_id=team, opponent_id=opponent, is_home=int(home),
        neutral_site=int(game.get("neutral_site") or 0), games_before=n,
        avg_points_for=st["points_for"]/n if n else None,
        avg_points_against=st["points_against"]/n if n else None,
        avg_possessions=poss/n if n else None,
        raw_offense=100*st["points_for"]/poss if poss > 0 else None,
        raw_defense=100*st["points_against"]/poss if poss > 0 else None,
        raw_net=100*(st["points_for"]-st["points_against"])/poss if poss > 0 else None,
        recent_five_margin=(sum(st["last5"])/len(st["last5"])) if st["last5"] else None,
        rest_days=rest,
        adjusted_net_before=(_number(prior.get("adjusted_net")) if prior else None),
        opponent_sos_before=(_number(prior.get("opponent_sos")) if prior else None),
        rating_snapshot_date=_day(prior["snapshot_date"]) if prior else None,
        generated_at=stamp
    )
    return tuple(cols[c] for c in FEATURE_COLS)


def _advance(st: dict, scored: float, conceded: float, possessions: float, game_day: str):
    st["games"] += 1
    st["points_for"] += scored
    st["points_against"] += conceded
    st["possessions"] += possessions
    st["last5"].append(scored - conceded)
    st["last_date"] = game_day


def _split(season: int, all_seasons: list[int]) -> str:
    """Explicitly label most recent completed season test and preceding validation."""
    last = all_seasons[-1]
    if season == last:
        return "test"
    if len(all_seasons) >= 2 and season == all_seasons[-2]:
        return "validation"
    return "development"


def build_dataset(games: list[dict], snapshots: list[dict], generated_at: str | None = None):
    """Build rows and walk-forward historical *scoring* baseline diagnostics.

    Features for ALL games on a date are created BEFORE updating ANY game on
    that date, preventing even double-header or unknown-tipoff same-day leakage.
    Prior adjusted ratings come only from snapshots with date < game date.
    Baseline: per-team prior mean points, minimum five earlier usable games each.
    Its MAE/RMSE do not measure trained matchup or winner prediction accuracy.
    """
    stamp = generated_at or datetime.now(timezone.utc).isoformat(timespec="seconds")
    rows = [dict(g) for g in games if g.get("season") is not None]
    rows.sort(key=lambda g: (int(g["season"]), str(g.get("game_date")), str(g["game_id"])))
    index = _snapshot_index(snapshots)
    coverage = _coverage(rows, stamp)
    all_seasons = sorted({int(g["season"]) for g in rows})
    features = []
    accuracy = []
    for season, it in groupby(rows, key=lambda g: int(g["season"])):
        state = defaultdict(_state)
        game_errors = []
        eligible_d1 = 0
        evaluated_games = 0
        for when, day_it in groupby(it, key=lambda g: str(g.get("game_date"))):
            date_games = list(day_it)
            valid_day = _day(when)
            if not valid_day:
                continue
            # Prevent same-day completed games from modifying each other's features.
            for g in date_games:
                if not bool(g.get("completed")) or g.get("home_score") is None or g.get("away_score") is None:
                    continue
                home = str(g["home_team_id"])
                away = str(g["away_team_id"])
                if home == away:
                    continue
                hs, ass = state[home], state[away]
                features.append(_feature(season, g, home, away, True, hs, index, stamp))
                features.append(_feature(season, g, away, home, False, ass, index, stamp))
                # Reporting-only out-of-season historical scoring baseline.
                # This is a descriptive calibration test, not a game winner model.
                if bool(g.get("home_is_d1")) and bool(g.get("away_is_d1")):
                    eligible_d1 += 1
                    if hs["games"] >= 5 and ass["games"] >= 5 and _usable(g):
                        hp = _number(g["home_score"])
                        ap = _number(g["away_score"])
                        game_errors.extend((hs["points_for"]/hs["games"] - hp,
                                            ass["points_for"]/ass["games"] - ap))
                        evaluated_games += 1
            # Only now incorporate box scores from this entire calendar day.
            for g in date_games:
                if not _usable(g):
                    continue
                home, away = str(g["home_team_id"]), str(g["away_team_id"])
                hp, ap = float(g["home_points"]), float(g["away_points"])
                _advance(state[home], hp, ap, float(g["home_possessions"]), valid_day)
                _advance(state[away], ap, hp, float(g["away_possessions"]), valid_day)
        count = len(game_errors)
        accuracy_row = dict(
            season=season, split_name=_split(season, all_seasons),
            eligible_d1_games=eligible_d1, evaluated_games=evaluated_games,
            evaluated_team_scores=count,
            points_mae=sum(abs(x) for x in game_errors)/count if count else None,
            points_rmse=sqrt(sum(x*x for x in game_errors)/count) if count else None,
            points_bias=sum(game_errors)/count if count else None,
            generated_at=stamp
        )
        accuracy.append(tuple(accuracy_row[c] for c in BASELINE_COLS))
    return coverage, features, accuracy


def _replace(cur, table: str, columns: tuple, rows: list[tuple], start_season: int, end_season: int):
    # Replacing only the chosen season range avoids deleting unrelated years.
    cur.execute(f"DELETE FROM {table} WHERE season BETWEEN %s AND %s", (start_season, end_season))
    if not rows:
        return
    placeholders = ",".join("%s" for _ in columns)
    sql = f"INSERT INTO {table} ({','.join(columns)}) VALUES ({placeholders})"
    for offset in range(0, len(rows), 300):
        cur.executemany(sql, rows[offset:offset+300])


def publish(start_season: int = 2022, end_season: int = 2026):
    if start_season > end_season:
        raise ValueError("--start-season must not exceed --end-season")
    from psycopg.rows import dict_row
    from cloud_store import open_cloud_connection
    with open_cloud_connection() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(GAME_SQL, (start_season, end_season))
            games = cur.fetchall()
            cur.execute("SELECT to_regclass('public.historical_team_snapshots') IS NOT NULL AS exists")
            has_snapshots = bool(cur.fetchone()["exists"])
            if not has_snapshots:
                raise RuntimeError("historical_team_snapshots is missing. Run the adjusted SOS workflow first.")
            cur.execute(SNAPSHOT_SQL, (start_season, end_season))
            snapshots = cur.fetchall()
            if not games:
                raise RuntimeError("No historical game records in the requested seasons. Nothing was changed.")
            coverage, features, accuracy = build_dataset(games, snapshots)
            cur.execute(SCHEMA)
            _replace(cur, "pregame_team_features", FEATURE_COLS, features, start_season, end_season)
            _replace(cur, "historical_coverage_audit", COVERAGE_COLS, coverage, start_season, end_season)
            _replace(cur, "historical_score_baseline_audit", BASELINE_COLS, accuracy, start_season, end_season)
    print(f"Historical QA updated: {len(coverage)} coverage seasons, "
          f"{len(features)} strictly pregame team feature rows, "
          f"{len(accuracy)} scoring-baseline audit seasons")
    for item in accuracy:
        line = dict(zip(BASELINE_COLS, item))
        mae = f"{line['points_mae']:.2f}" if line['points_mae'] is not None else "n/a"
        print(f"  {line['season']} {line['split_name']}: "
              f"{line['evaluated_games']} qualifying D1 games, "
              f"prior-points scoring MAE={mae}")


def main():
    parser = argparse.ArgumentParser(description="Build historical data-quality and pregame team-stat audits")
    parser.add_argument("--start-season", type=int, default=2022)
    parser.add_argument("--end-season", type=int, default=2026)
    args = parser.parse_args()
    publish(args.start_season, args.end_season)


if __name__ == "__main__":
    main()
