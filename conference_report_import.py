"""Import manually verified, official NCAA men's basketball conference availability reports.

This importer DOES NOT scrape conference pages, third-party embedded viewers, or
unpublished endpoints. Its CSV contains small, independently transcribed facts,
a direct URL to the official conference report, and the game to which the
status applies. An absent report is never interpreted as player availability.

It replaces only source='Official conference report' rows in injury_reports.
Other sources, player/game data, and rankings remain untouched.
"""
from __future__ import annotations

import argparse
import csv
from datetime import date, datetime, time, timedelta, timezone
import hashlib
import json
import logging
from pathlib import Path
import unicodedata
from urllib.parse import urlsplit

from cloud_store import open_cloud_connection

SOURCE = "Official conference report"
LOG = logging.getLogger(__name__)
REQUIRED = {"conference", "team_name", "player_name", "status",
            "game_date", "reported_at", "source_url"}
STATUSES = {
    "out": "Out",
    "questionable": "Questionable",
    "doubtful": "Doubtful",
    "probable": "Probable",
    "game time decision": "Game Time Decision",
    "game-time decision": "Game Time Decision",
    "gtd": "Game Time Decision",
}
# URLs must point to the official conference publication, not a news aggregator.
# A small allowlist prevents accidentally attaching an unrelated or misleading URL.
OFFICIAL_HOSTS = {
    "secsports.com", "theacc.com", "big12sports.com", "bigeast.com",
    "bigten.org", "themw.com", "mac-sports.com", "conferenceusa.com",
    "pac-12.com", "atlantica10.com", "wccsports.com", "mvc-sports.com",
}


def normalized_name(value: str) -> str:
    text = unicodedata.normalize("NFKD", (value or "").casefold())
    return " ".join("".join(c for c in text if not unicodedata.combining(c)).split())


def report_timestamp(value: str) -> str:
    value = value.strip()
    if not value:
        raise ValueError("reported_at is required")
    if len(value) == 10:
        instant = datetime.combine(date.fromisoformat(value), time.min, tzinfo=timezone.utc)
    else:
        instant = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if instant.tzinfo is None:
            raise ValueError("reported_at time needs a timezone, for example +00:00")
        instant = instant.astimezone(timezone.utc)
    if instant > datetime.now(timezone.utc) + timedelta(minutes=5):
        raise ValueError("reported_at cannot be in the future")
    return instant.isoformat(timespec="seconds")


def official_url(value: str) -> str:
    url = value.strip()
    parsed = urlsplit(url)
    host = (parsed.hostname or "").casefold()
    if parsed.scheme != "https" or parsed.username or parsed.password or not any(
        host == domain or host.endswith("." + domain) for domain in OFFICIAL_HOSTS
    ):
        raise ValueError("source_url must be a direct HTTPS official conference report URL")
    if len(url) > 1200:
        raise ValueError("source_url too long")
    return url


def read_reports(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not REQUIRED.issubset(reader.fieldnames or []):
            raise ValueError("CSV missing required columns: " + ", ".join(sorted(REQUIRED)))
        output = []
        for line_no, raw in enumerate(reader, 2):
            if not any(str(value or "").strip() for value in raw.values()):
                continue
            try:
                name = (raw.get("player_name") or "").strip()
                team = (raw.get("team_name") or "").strip()
                conf = (raw.get("conference") or "").strip()
                if not name or not team or not conf:
                    raise ValueError("conference, team_name and player_name are required")
                status_key = normalized_name(raw.get("status") or "")
                if status_key not in STATUSES:
                    raise ValueError("status must be Out, Questionable, Doubtful, Probable, or Game Time Decision")
                game_date = date.fromisoformat((raw.get("game_date") or "").strip()).isoformat()
                published = report_timestamp(raw.get("reported_at") or "")
                url = official_url(raw.get("source_url") or "")
                player_id = (raw.get("espn_player_id") or "").strip()
                if player_id and not player_id.isdigit():
                    raise ValueError("espn_player_id must be numeric if provided")
                output.append({"conference": conf[:70], "team_name": team[:160],
                               "player_name": name[:160], "status": STATUSES[status_key],
                               "game_date": game_date, "reported_at": published,
                               "source_url": url, "espn_player_id": player_id or None})
            except (ValueError, TypeError) as exc:
                raise ValueError(f"Invalid CSV row {line_no}: {exc}") from exc
    return output


def match_players(reports: list[dict], roster: list[dict]) -> list[dict]:
    """Only exact normalized player+team matches (or verified IDs); never guess."""
    index: dict[tuple[str, str], set[str]] = {}
    by_id = {}
    for player in roster:
        pid = str(player["player_id"])
        by_id[pid] = player
        key = (normalized_name(player["player_name"]), normalized_name(player["team_name"]))
        index.setdefault(key, set()).add(pid)
    matched = []
    for report in reports:
        r = dict(report)
        supplied_id = r.pop("espn_player_id")
        if supplied_id:
            player = by_id.get(supplied_id)
            if player is None:
                raise ValueError(f"ESPN player ID {supplied_id} does not exist in the database")
            if normalized_name(player["player_name"]) != normalized_name(r["player_name"]):
                raise ValueError(f"ESPN player ID/name conflict for {r['player_name']}")
            if normalized_name(player["team_name"]) != normalized_name(r["team_name"]):
                raise ValueError(f"ESPN player ID/team conflict for {r['player_name']}")
            pid = supplied_id
        else:
            candidates = index.get((normalized_name(r["player_name"]),
                                    normalized_name(r["team_name"])), set())
            pid = next(iter(candidates)) if len(candidates) == 1 else None
        r["source_player_id"] = pid
        if not pid:
            LOG.warning("Unmatched player %s (%s) — shown on Injuries page only",
                        r["player_name"], r["team_name"])
        matched.append(r)
    return matched


def publish_csv(path: Path) -> tuple[int, int]:
    reports = read_reports(path)  # Validate before accessing or changing the DB.
    retrieved = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with open_cloud_connection() as connection:
        with connection.cursor() as cur:
            cur.execute("""SELECT p.player_id,p.name,t.name
                           FROM players p JOIN teams t ON t.team_id=p.team_id""")
            roster = [dict(zip(("player_id", "player_name", "team_name"), row))
                      for row in cur.fetchall()]
            matched = match_players(reports, roster)
            prepared = {}
            for r in matched:
                identity = "|".join((r["game_date"], normalized_name(r["team_name"]),
                                     normalized_name(r["player_name"])))
                key = "conference:" + hashlib.sha256(identity.encode()).hexdigest()[:32]
                metadata = {"conference": r["conference"], "game_date": r["game_date"],
                            "source_url": r["source_url"]}
                row = (key, SOURCE, r["source_player_id"], r["player_name"],
                       r["team_name"], r["status"], None, None,
                       r["reported_at"], retrieved, json.dumps(metadata))
                if key not in prepared or row[8] >= prepared[key][8]:
                    prepared[key] = row
            # A transaction ensures invalid rows cannot partially clear reports.
            cur.execute("DELETE FROM injury_reports WHERE source=%s", (SOURCE,))
            if prepared:
                cur.executemany("""INSERT INTO injury_reports
                    (report_id,source,source_player_id,player_name,team_name,status,
                     injury_detail,expected_return,reported_at,retrieved_at,raw_json)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""", list(prepared.values()))
    return len(prepared), sum(bool(r["source_player_id"]) for r in matched)


def main():
    parser = argparse.ArgumentParser(description="Import official conference availability facts from verified CSV")
    parser.add_argument("--csv", type=Path, default=Path("data/conference_reports.csv"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    count, matched = publish_csv(args.csv)
    print(f"Official conference report rows saved: {count}; ESPN player ID matches: {matched}")
    print("Reports are game-specific; no entry does not imply a healthy/available player.")


if __name__ == "__main__":
    main()

