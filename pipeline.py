"""Idempotent ESPN ingestion, with failure logging and SQLite storage."""
from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path

from db import connect, init, utc_now
from espn import ESPN, extract_player_box, extract_team_box, game_from_event, teams_from_game, teams_from_list

LOG = logging.getLogger(__name__)


def _logo(team):
    logos = team.get("logos") or []
    return logos[0].get("href") if logos else None


def save_team(conn, team, is_d1=False):
    team_id = team.get("id")
    if team_id is None:
        return
    team_id = str(team_id)
    name = team.get("displayName") or team.get("name") or team.get("shortDisplayName") or team_id
    group = team.get("groups") or {}
    conference = group.get("name") if isinstance(group, dict) else None
    conn.execute("""
        INSERT INTO teams(team_id,name,abbreviation,location,conference,logo_url,is_d1,updated_at)
        VALUES (?,?,?,?,?,?,?,?)
        ON CONFLICT(team_id) DO UPDATE SET
           name=excluded.name,
           abbreviation=COALESCE(excluded.abbreviation,teams.abbreviation),
           location=COALESCE(excluded.location,teams.location),
           conference=COALESCE(excluded.conference,teams.conference),
           logo_url=COALESCE(excluded.logo_url,teams.logo_url),
           is_d1=MAX(teams.is_d1,excluded.is_d1),
           updated_at=excluded.updated_at
    """, (team_id, name, team.get("abbreviation"), team.get("location"),
          conference, _logo(team), int(is_d1), utc_now()))


def save_game(conn, game):
    columns = ["game_id", "season", "game_date", "start_time_utc", "home_team_id",
               "away_team_id", "status", "status_detail", "home_score", "away_score",
               "venue", "neutral_site", "completed"]
    cols = ",".join(columns + ["updated_at"])
    marks = ",".join(["?"] * (len(columns) + 1))
    sets = ",".join(f"{name}=excluded.{name}" for name in columns if name != "game_id")
    conn.execute(f"INSERT INTO games({cols}) VALUES ({marks}) ON CONFLICT(game_id) DO UPDATE SET {sets},updated_at=excluded.updated_at",
                 [game.get(x) for x in columns] + [utc_now()])


def save_summary(conn, summary, game):
    teams = extract_team_box(summary, game)
    if len(teams) != 2:
        LOG.warning("Game %s: no complete team box score; will retry on next sync", game["game_id"])
        return False
    cols = ["game_id", "team_id", "home_away", "points", "fg_m", "fg_a", "three_m",
            "three_a", "ft_m", "ft_a", "offensive_rebounds", "defensive_rebounds", "rebounds",
            "assists", "steals", "blocks", "turnovers", "fouls", "possessions",
            "offensive_efficiency", "defensive_efficiency", "net_efficiency", "raw_json", "updated_at"]
    for row in teams:
        conn.execute(f"""INSERT INTO team_game_stats({','.join(cols)})
                 VALUES ({','.join('?' for _ in cols)})
                 ON CONFLICT(game_id,team_id) DO UPDATE SET
                 {','.join(f'{c}=excluded.{c}' for c in cols if c not in ('game_id','team_id'))}
                 """, [row.get(c) if c != "updated_at" else utc_now() for c in cols])
    for row in extract_player_box(summary):
        team_exists = conn.execute("SELECT 1 FROM teams WHERE team_id=?", (row["team_id"],)).fetchone()
        if not team_exists:
            continue
        conn.execute("""INSERT INTO players(player_id,team_id,name,position,jersey,updated_at)
                   VALUES(?,?,?,?,?,?) ON CONFLICT(player_id) DO UPDATE SET
                   team_id=excluded.team_id,name=excluded.name,position=excluded.position,
                   jersey=excluded.jersey,updated_at=excluded.updated_at""",
                   (row["player_id"], row["team_id"], row["name"], row["position"], row["jersey"], utc_now()))
        player_cols = ["game_id", "player_id", "team_id", "starter", "minutes", "points",
                       "rebounds", "assists", "steals", "blocks", "turnovers", "fg_m", "fg_a",
                       "three_m", "three_a", "ft_m", "ft_a", "raw_json"]
        row["game_id"] = game["game_id"]
        conn.execute(f"""INSERT INTO player_game_stats({','.join(player_cols)})
                    VALUES({','.join('?' for _ in player_cols)})
                    ON CONFLICT(game_id,player_id) DO UPDATE SET
                    {','.join(f'{c}=excluded.{c}' for c in player_cols if c not in ('game_id','player_id'))}""",
                    [row.get(c) for c in player_cols])
    return True


def sync_range(start: date, end: date, db_path=None, client=None, force_recent=False):
    if end < start:
        raise ValueError("End must be on or after start")
    espn = client or ESPN()
    conn = connect(db_path) if db_path else connect()
    init(conn)
    run_id = conn.execute("INSERT INTO sync_runs(started_at,status) VALUES(?,?)", (utc_now(), "running")).lastrowid
    conn.commit()
    dates_checked = games_seen = summaries_saved = 0
    errors = []
    try:
        try:
            teams_payload = espn.teams()
            team_count = 0
            for t in teams_from_list(teams_payload):
                save_team(conn, t, is_d1=True)
                team_count += 1
            if team_count == 0:
                errors.append("D1 team list was empty; existing team classifications kept")
            conn.commit()
            LOG.info("D1 team list: %s teams", team_count)
        except Exception as exc:
            errors.append(f"Team list failed: {exc}")
            LOG.warning(errors[-1])
            conn.rollback()

        day = start
        while day <= end:
            dates_checked += 1
            try:
                payload = espn.scoreboard(day)
                events = payload.get("events", [])
                LOG.info("%s: %s events", day, len(events))
                if len(events) >= 500:
                    errors.append(f"{day}: ESPN reached scoreboard limit (500); check completeness")
            except Exception as exc:
                errors.append(f"Scoreboard {day}: {exc}")
                LOG.warning(errors[-1])
                day += timedelta(days=1)
                continue
            for event in events:
                try:
                    game = game_from_event(event)
                    if not game:
                        continue
                    games_seen += 1
                    for team in teams_from_game(event):
                        save_team(conn, team)
                    save_game(conn, game)
                    conn.commit()
                    if not game["completed"]:
                        continue
                    already = conn.execute("SELECT COUNT(*) FROM team_game_stats WHERE game_id=?",
                                           (game["game_id"],)).fetchone()[0]
                    # Most games are immutable once recorded. Re-fetch the last 2 days
                    # in a nightly sync to catch ESPN's late stat corrections.
                    refresh = force_recent and game["game_date"] >= (date.today() - timedelta(days=2)).isoformat()
                    if already >= 2 and not refresh:
                        continue
                    summary = espn.summary(game["game_id"])
                    if save_summary(conn, summary, game):
                        summaries_saved += 1
                    conn.commit()
                except Exception as exc:
                    conn.rollback()
                    errors.append(f"Game {event.get('id', '?')}: {exc}")
                    LOG.warning(errors[-1])
            day += timedelta(days=1)
    except Exception as exc:
        conn.rollback()
        errors.append(str(exc))
        LOG.exception("Unexpected synchronization failure")
    finally:
        status = "ok" if not errors else "partial" if games_seen else "failed"
        conn.execute("""UPDATE sync_runs SET finished_at=?,status=?,dates_checked=?,games_seen=?,
                     summaries_saved=?,errors=? WHERE run_id=?""",
                     (utc_now(), status, dates_checked, games_seen, summaries_saved,
                      "\n".join(errors[:100]), run_id))
        conn.commit()
        conn.close()
    LOG.info("Sync %s: days=%s, games=%s, summaries=%s, problems=%s", status, dates_checked, games_seen, summaries_saved, len(errors))
    return {"status": status, "days": dates_checked, "games": games_seen,
            "boxscores": summaries_saved, "errors": errors}
