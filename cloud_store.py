"""Publish local ESPN imports to PostgreSQL for the hosted read-only dashboard.

No Supabase account is created by this module. A DATABASE_URL secret is required.
"""
from __future__ import annotations

import os
from pathlib import Path
import sqlite3

# SQLite's local ingestion schema is in db.py; Postgres is the persistent copy.
# Integer values, UTC timestamps and local game dates are retained as originally
# parsed to keep the existing importer and report semantics stable.
POSTGRES_SCHEMA = """
CREATE TABLE IF NOT EXISTS teams (
    team_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    abbreviation TEXT, location TEXT, conference TEXT, logo_url TEXT,
    is_d1 INTEGER NOT NULL DEFAULT 0, updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS games (
    game_id TEXT PRIMARY KEY, season INTEGER, game_date TEXT NOT NULL,
    start_time_utc TEXT, home_team_id TEXT NOT NULL REFERENCES teams(team_id),
    away_team_id TEXT NOT NULL REFERENCES teams(team_id), status TEXT NOT NULL,
    status_detail TEXT, home_score INTEGER, away_score INTEGER, venue TEXT,
    neutral_site INTEGER NOT NULL DEFAULT 0, completed INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_games_date ON games(game_date);
CREATE INDEX IF NOT EXISTS idx_games_teams ON games(home_team_id, away_team_id);
CREATE TABLE IF NOT EXISTS team_game_stats (
    game_id TEXT NOT NULL REFERENCES games(game_id),
    team_id TEXT NOT NULL REFERENCES teams(team_id),
    home_away TEXT, points INTEGER,
    fg_m INTEGER, fg_a INTEGER, three_m INTEGER, three_a INTEGER,
    ft_m INTEGER, ft_a INTEGER, offensive_rebounds INTEGER,
    defensive_rebounds INTEGER, rebounds INTEGER, assists INTEGER,
    steals INTEGER, blocks INTEGER, turnovers INTEGER, fouls INTEGER,
    possessions DOUBLE PRECISION, offensive_efficiency DOUBLE PRECISION,
    defensive_efficiency DOUBLE PRECISION, net_efficiency DOUBLE PRECISION,
    raw_json TEXT NOT NULL, updated_at TEXT NOT NULL,
    PRIMARY KEY(game_id,team_id)
);
CREATE INDEX IF NOT EXISTS idx_team_stats_team ON team_game_stats(team_id);
CREATE TABLE IF NOT EXISTS players (
    player_id TEXT PRIMARY KEY,
    team_id TEXT REFERENCES teams(team_id), name TEXT NOT NULL,
    position TEXT, jersey TEXT, updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS player_game_stats (
    game_id TEXT NOT NULL REFERENCES games(game_id),
    player_id TEXT NOT NULL REFERENCES players(player_id),
    team_id TEXT NOT NULL REFERENCES teams(team_id),
    starter INTEGER, minutes TEXT, points INTEGER, rebounds INTEGER,
    assists INTEGER, steals INTEGER, blocks INTEGER, turnovers INTEGER,
    fg_m INTEGER, fg_a INTEGER, three_m INTEGER, three_a INTEGER,
    ft_m INTEGER, ft_a INTEGER, raw_json TEXT NOT NULL,
    PRIMARY KEY(game_id,player_id)
);
CREATE INDEX IF NOT EXISTS idx_player_stats_game ON player_game_stats(game_id);
CREATE TABLE IF NOT EXISTS injury_reports (
    report_id TEXT PRIMARY KEY, source TEXT NOT NULL,
    source_player_id TEXT, player_name TEXT NOT NULL,
    team_name TEXT, status TEXT, injury_detail TEXT,
    expected_return TEXT, reported_at TEXT,
    retrieved_at TEXT NOT NULL, raw_json TEXT
);
CREATE TABLE IF NOT EXISTS sync_runs (
    run_id BIGSERIAL PRIMARY KEY,
    started_at TEXT NOT NULL, finished_at TEXT, status TEXT NOT NULL,
    dates_checked INTEGER NOT NULL DEFAULT 0,
    games_seen INTEGER NOT NULL DEFAULT 0,
    summaries_saved INTEGER NOT NULL DEFAULT 0, errors TEXT
);
-- Prevent anonymous access through Supabase's automatic Data API.
-- Our Streamlit backend uses a server-side Postgres connection instead.
ALTER TABLE teams ENABLE ROW LEVEL SECURITY;
ALTER TABLE games ENABLE ROW LEVEL SECURITY;
ALTER TABLE team_game_stats ENABLE ROW LEVEL SECURITY;
ALTER TABLE players ENABLE ROW LEVEL SECURITY;
ALTER TABLE player_game_stats ENABLE ROW LEVEL SECURITY;
ALTER TABLE injury_reports ENABLE ROW LEVEL SECURITY;
ALTER TABLE sync_runs ENABLE ROW LEVEL SECURITY;
"""

TABLE_PKS = {
    "teams": ("team_id",),
    "games": ("game_id",),
    "team_game_stats": ("game_id", "team_id"),
    "players": ("player_id",),
    "player_game_stats": ("game_id", "player_id"),
}


def database_url() -> str:
    url = os.environ.get("DATABASE_URL", "").strip()
    if not url:
        raise RuntimeError("DATABASE_URL missing. Add it as a GitHub Actions secret or Streamlit secret.")
    if not url.startswith(("postgresql://", "postgres://")):
        raise ValueError("DATABASE_URL must be a PostgreSQL connection URL")
    return url


def open_cloud_connection():
    import psycopg
    # Supabase's session pooler (port 5432) works over IPv4.
    # prepare_threshold=None also works with transaction poolers.
    return psycopg.connect(database_url(), sslmode="require", connect_timeout=20, prepare_threshold=None)


def upsert_sql(table: str, columns: list[str]) -> str:
    """Generate SQL only for internally named tables/columns, never user input."""
    if table not in TABLE_PKS:
        raise ValueError(f"Unexpected table: {table}")
    if not columns or any(not col.isidentifier() for col in columns):
        raise ValueError("Invalid column names")
    pks = TABLE_PKS[table]
    if not set(pks).issubset(columns):
        raise ValueError("Primary keys missing")
    update = []
    for col in columns:
        if col in pks:
            continue
        if table == "teams" and col == "is_d1":
            update.append("is_d1=GREATEST(teams.is_d1,EXCLUDED.is_d1)")
        elif table == "teams" and col in ("abbreviation", "location", "conference", "logo_url"):
            update.append(f"{col}=COALESCE(EXCLUDED.{col},teams.{col})")
        else:
            update.append(f"{col}=EXCLUDED.{col}")
    if not update:
        raise ValueError("No columns to update")
    return (
        f"INSERT INTO {table} ({','.join(columns)}) VALUES "
        f"({','.join('%s' for _ in columns)}) "
        f"ON CONFLICT ({','.join(pks)}) DO UPDATE SET {','.join(update)}"
    )


def push_sqlite(source_path: Path, *, result: dict) -> dict[str, int]:
    """Read the temporary SQLite sync, upsert it in dependency order, and log the run.

    Entire cloud publish transaction is all-or-nothing; the existing PostgreSQL
    dataset is left untouched when anything fails.
    """
    source = sqlite3.connect(str(source_path))
    source.row_factory = sqlite3.Row
    counts: dict[str, int] = {}
    try:
        with open_cloud_connection() as cloud:
            with cloud.cursor() as cur:
                cur.execute(POSTGRES_SCHEMA)
                for table in TABLE_PKS:
                    rows = source.execute(f"SELECT * FROM {table}").fetchall()
                    if not rows:
                        counts[table] = 0
                        continue
                    cols = list(rows[0].keys())
                    sql = upsert_sql(table, cols)
                    # Modest batches to avoid sending a giant command all at once.
                    for offset in range(0, len(rows), 500):
                        cur.executemany(sql, [tuple(row) for row in rows[offset:offset+500]])
                    counts[table] = len(rows)
                last_run = source.execute("SELECT * FROM sync_runs ORDER BY run_id DESC LIMIT 1").fetchone()
                if last_run:
                    cur.execute("""INSERT INTO sync_runs
                        (started_at, finished_at, status, dates_checked, games_seen, summaries_saved, errors)
                        VALUES (%s,%s,%s,%s,%s,%s,%s)""",
                        (last_run["started_at"], last_run["finished_at"],
                         last_run["status"], last_run["dates_checked"],
                         last_run["games_seen"], last_run["summaries_saved"],
                         last_run["errors"]))
            # psycopg context manager commits on success and rolls back on errors.
    finally:
        source.close()
    return counts
