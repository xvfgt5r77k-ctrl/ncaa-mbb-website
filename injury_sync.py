"""Refresh ESPN's *reported* men's college basketball injuries in Supabase.

ESPN college injuries are often absent or incomplete. An empty successful feed means
"no ESPN reports returned", NOT "all players are healthy". Only entries with ESPN
athlete IDs are saved, so the dashboard can match them to player statistics safely.

This module only changes injury_reports rows whose source is 'ESPN'. Other sources,
teams, games, and player statistics are untouched.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

import requests

from cloud_store import open_cloud_connection

LOG = logging.getLogger(__name__)
ESPN_INJURIES_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/basketball/"
    "mens-college-basketball/injuries"
)


def _plain(value) -> str:
    """Handle ESPN fields that may be either a string or a nested object."""
    if isinstance(value, dict):
        for key in ("displayName", "shortName", "name", "description", "text", "abbreviation"):
            if value.get(key):
                return str(value[key]).strip()
        return ""
    return str(value).strip() if value is not None else ""


def parse_espn_injuries(payload: dict, *, retrieved_at: str | None = None) -> list[dict]:
    """Normalize the site-v2 injuries array; require an ESPN athlete ID."""
    if not isinstance(payload, dict) or not isinstance(payload.get("injuries"), list):
        raise ValueError("ESPN injuries response missing the expected injuries list")
    retrieved_at = retrieved_at or datetime.now(timezone.utc).isoformat(timespec="seconds")
    by_athlete: dict[str, dict] = {}
    for item in payload["injuries"]:
        if not isinstance(item, dict):
            continue
        athlete = item.get("athlete") or {}
        team = item.get("team") or {}
        injury = item.get("injury") or {}
        if not isinstance(athlete, dict) or not isinstance(team, dict):
            continue
        if not isinstance(injury, dict):
            injury = {"type": injury}
        athlete_id = str(athlete.get("id") or "").strip()
        name = _plain(athlete.get("displayName") or athlete.get("fullName") or athlete.get("name"))
        if not athlete_id or not name:
            continue
        status = _plain(item.get("status"))
        if status.casefold() in ("healthy", "active", "available"):
            continue
        detail = _plain(injury.get("type")) or _plain(injury.get("description"))
        location = _plain(injury.get("location"))
        if location and location.casefold() not in detail.casefold() and detail:
            detail = f"{location} {detail}"
        if not detail:
            detail = _plain(item.get("shortComment"))
        if not detail:
            detail = _plain(item.get("longComment"))
        # Keep the annotation short and avoid untrusted markup in table cells.
        detail = " ".join(detail.split())[:160]
        if not detail and not status:
            continue
        row = {
            "report_id": f"espn:{athlete_id}",
            "source": "ESPN",
            "source_player_id": athlete_id,
            "player_name": name,
            "team_name": _plain(team.get("displayName") or team.get("name")) or None,
            "status": status or None,
            "injury_detail": detail or None,
            "expected_return": _plain(item.get("returnDate")) or None,
            "reported_at": _plain(item.get("date")) or None,
            "retrieved_at": retrieved_at,
            "raw_json": json.dumps(item, separators=(",", ":"), ensure_ascii=False),
        }
        # If the feed contains multiple reports for an athlete, prefer the
        # latest dated entry rather than writing conflicting current statuses.
        old = by_athlete.get(athlete_id)
        if old is None or (row["reported_at"] or "") >= (old["reported_at"] or ""):
            by_athlete[athlete_id] = row
    if payload["injuries"] and not by_athlete:
        raise ValueError("ESPN returned entries, but none had usable athlete IDs/status")
    return list(by_athlete.values())


def fetch_espn_injuries(session=None) -> list[dict]:
    session = session or requests.Session()
    response = session.get(ESPN_INJURIES_URL, timeout=25,
                           headers={"Accept": "application/json",
                                    "User-Agent": "NCAA-MBB-Research-Database/1.0"})
    response.raise_for_status()
    return parse_espn_injuries(response.json())


def replace_espn_injury_snapshot(rows: list[dict]) -> int:
    """Atomically replace only ESPN reports after a successful fetch/parse.

    If fetching/parsing fails, callers never invoke this method, preserving the
    last good data. If ESPN responds with a valid empty list, clear only ESPN
    reports rather than falsely displaying stale injuries.
    """
    columns = ("report_id", "source", "source_player_id", "player_name",
               "team_name", "status", "injury_detail", "expected_return",
               "reported_at", "retrieved_at", "raw_json")
    with open_cloud_connection() as cloud:
        with cloud.cursor() as cursor:
            cursor.execute("DELETE FROM injury_reports WHERE source = %s", ("ESPN",))
            if rows:
                sql = (f"INSERT INTO injury_reports ({','.join(columns)}) "
                       f"VALUES ({','.join('%s' for _ in columns)})")
                cursor.executemany(sql, [tuple(row[column] for column in columns) for row in rows])
    return len(rows)


def sync_espn_injuries() -> int:
    rows = fetch_espn_injuries()
    count = replace_espn_injury_snapshot(rows)
    LOG.info("ESPN injury feed: %d reported athletes saved (limited college coverage)", count)
    return count
