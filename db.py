"""SQLite storage and read-only reporting for NCAA Division I men's basketball."""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_DB = ROOT / "data" / "ncaa_mbb.sqlite3"

SCHEMA = """
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS teams (
    team_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    abbreviation TEXT,
    location TEXT,
    conference TEXT,
    logo_url TEXT,
    is_d1 INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS games (
    game_id TEXT PRIMARY KEY,
    season INTEGER,
    game_date TEXT NOT NULL,
    start_time_utc TEXT,
    home_team_id TEXT NOT NULL,
    away_team_id TEXT NOT NULL,
    status TEXT NOT NULL,
    status_detail TEXT,
    home_score INTEGER,
    away_score INTEGER,
    venue TEXT,
    neutral_site INTEGER NOT NULL DEFAULT 0,
    completed INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (home_team_id) REFERENCES teams(team_id),
    FOREIGN KEY (away_team_id) REFERENCES teams(team_id)
);
CREATE INDEX IF NOT EXISTS idx_games_date ON games(game_date);
CREATE INDEX IF NOT EXISTS idx_games_teams ON games(home_team_id, away_team_id);
CREATE TABLE IF NOT EXISTS team_game_stats (
    game_id TEXT NOT NULL,
    team_id TEXT NOT NULL,
    home_away TEXT,
    points INTEGER,
    fg_m INTEGER, fg_a INTEGER,
    three_m INTEGER, three_a INTEGER,
    ft_m INTEGER, ft_a INTEGER,
    offensive_rebounds INTEGER,
    defensive_rebounds INTEGER,
    rebounds INTEGER,
    assists INTEGER,
    steals INTEGER,
    blocks INTEGER,
    turnovers INTEGER,
    fouls INTEGER,
    possessions REAL,
    offensive_efficiency REAL,
    defensive_efficiency REAL,
    net_efficiency REAL,
    raw_json TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (game_id, team_id),
    FOREIGN KEY (game_id) REFERENCES games(game_id),
    FOREIGN KEY (team_id) REFERENCES teams(team_id)
);
CREATE INDEX IF NOT EXISTS idx_team_stats_team ON team_game_stats(team_id);
CREATE TABLE IF NOT EXISTS players (
    player_id TEXT PRIMARY KEY,
    team_id TEXT,
    name TEXT NOT NULL,
    position TEXT,
    jersey TEXT,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (team_id) REFERENCES teams(team_id)
);
CREATE TABLE IF NOT EXISTS player_game_stats (
    game_id TEXT NOT NULL,
    player_id TEXT NOT NULL,
    team_id TEXT NOT NULL,
    starter INTEGER,
    minutes TEXT,
    points INTEGER,
    rebounds INTEGER,
    assists INTEGER,
    steals INTEGER,
    blocks INTEGER,
    turnovers INTEGER,
    fg_m INTEGER, fg_a INTEGER,
    three_m INTEGER, three_a INTEGER,
    ft_m INTEGER, ft_a INTEGER,
    raw_json TEXT NOT NULL,
    PRIMARY KEY(game_id, player_id),
    FOREIGN KEY (game_id) REFERENCES games(game_id),
    FOREIGN KEY (player_id) REFERENCES players(player_id),
    FOREIGN KEY (team_id) REFERENCES teams(team_id)
);
CREATE INDEX IF NOT EXISTS idx_player_stats_game ON player_game_stats(game_id);
CREATE TABLE IF NOT EXISTS injury_reports (
    report_id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    source_player_id TEXT,
    player_name TEXT NOT NULL,
    team_name TEXT,
    status TEXT,
    injury_detail TEXT,
    expected_return TEXT,
    reported_at TEXT,
    retrieved_at TEXT NOT NULL,
    raw_json TEXT
);
CREATE TABLE IF NOT EXISTS sync_runs (
    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    status TEXT NOT NULL,
    dates_checked INTEGER NOT NULL DEFAULT 0,
    games_seen INTEGER NOT NULL DEFAULT 0,
    summaries_saved INTEGER NOT NULL DEFAULT 0,
    errors TEXT
);
"""


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect(path: str | Path = DEFAULT_DB) -> sqlite3.Connection:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 15000")
    return conn


def init(conn: sqlite3.Connection):
    conn.executescript(SCHEMA)
    conn.commit()


# Query returns only Division I teams with at least one completed, fully parsed game.
# Efficiency uses ratio of season total points to season estimated possessions.
TEAM_RANKINGS = """
WITH base AS (
 SELECT
    t.team_id, t.name, t.abbreviation, t.conference,
    COUNT(*) AS games,
    SUM(CASE WHEN s.points > o.points THEN 1 ELSE 0 END) AS wins,
    SUM(CASE WHEN s.points < o.points THEN 1 ELSE 0 END) AS losses,
    SUM(s.points) AS points_for,
    SUM(o.points) AS points_against,
    SUM(s.possessions) AS poss,
    SUM(s.fg_m) AS fg_m,
    SUM(s.fg_a) AS fg_a,
    SUM(s.three_m) AS three_m,
    SUM(s.three_a) AS three_a,
    SUM(s.ft_m) AS ft_m,
    SUM(s.ft_a) AS ft_a,
    SUM(s.turnovers) AS tov
 FROM team_game_stats s
 JOIN team_game_stats o ON o.game_id=s.game_id AND o.team_id<>s.team_id
 JOIN games g ON g.game_id=s.game_id AND g.completed=1
 JOIN teams t ON t.team_id=s.team_id
 WHERE t.is_d1=1 AND s.possessions>0 AND o.possessions>0
   AND g.season = (SELECT MAX(season) FROM games WHERE completed=1 AND season IS NOT NULL)
 GROUP BY t.team_id
)
SELECT *,
 ROUND(100.0 * points_for / NULLIF(poss,0),1) AS off_eff,
 ROUND(100.0 * points_against / NULLIF(poss,0),1) AS def_eff,
 ROUND(100.0 * (points_for-points_against) / NULLIF(poss,0),1) AS net_eff,
 ROUND(100.0 * (fg_m + 0.5*three_m) / NULLIF(fg_a,0),1) AS efg_pct,
 ROUND(100.0 * tov / NULLIF(poss,0),1) AS tov_pct
FROM base
ORDER BY net_eff DESC
"""
