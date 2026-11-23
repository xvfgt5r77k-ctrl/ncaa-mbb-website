"""ESPN site endpoint adapter. These are undocumented, subject to change."""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

API_ROOT = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball"
CENTRAL = ZoneInfo("America/Chicago")


def integer(value):
    if value is None or value == "" or value == "--":
        return None
    try:
        return int(float(str(value).strip().replace(",", "")))
    except (ValueError, TypeError):
        return None


def shot_pair(value):
    if value is None:
        return None, None
    match = re.match(r"^\s*(\d+)\s*[-/]\s*(\d+)\s*$", str(value))
    return (int(match[1]), int(match[2])) if match else (None, None)


def iso_date(value):
    """Convert ESPN's UTC timestamps to the user's Central calendar date."""
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(CENTRAL).date().isoformat()
    except ValueError:
        return str(value)[:10]


class ESPN:
    def __init__(self, session=None):
        if session is not None:
            self.session = session
            return
        self.session = requests.Session()
        retry = Retry(
            total=3,
            connect=3,
            read=3,
            backoff_factor=0.9,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=frozenset(["GET"]),
        )
        self.session.mount("https://", HTTPAdapter(max_retries=retry))
        self.session.headers.update({"User-Agent": "NCAA-MBB-Research-Database/1.0 (personal statistics project)", "Accept": "application/json"})

    def get(self, path, params=None):
        response = self.session.get(f"{API_ROOT}/{path}", params=params, timeout=25)
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            raise ValueError(f"ESPN returned unexpected data for {path}")
        return data

    def teams(self):
        return self.get("teams", {"group": 50, "limit": 1000})

    def scoreboard(self, day):
        return self.get("scoreboard", {"dates": day.strftime("%Y%m%d"), "groups": 50, "limit": 500})

    def summary(self, game_id):
        return self.get("summary", {"event": str(game_id)})


def teams_from_list(payload):
    for sport in payload.get("sports", []):
        for league in sport.get("leagues", []):
            for item in league.get("teams", []):
                team = item.get("team", item)
                if team.get("id"):
                    yield team


def teams_from_game(event):
    competitions = event.get("competitions", [])
    if not competitions:
        return []
    return [c.get("team", {}) for c in competitions[0].get("competitors", [])]


def game_from_event(event):
    competitions = event.get("competitions", [])
    if not competitions:
        return None
    competition = competitions[0]
    by_side = {c.get("homeAway"): c for c in competition.get("competitors", [])}
    if "home" not in by_side or "away" not in by_side:
        return None
    home, away = by_side["home"], by_side["away"]
    if not home.get("team", {}).get("id") or not away.get("team", {}).get("id"):
        return None
    state = event.get("status", {}).get("type", {})
    detail = state.get("description") or state.get("detail") or ""
    completed = bool(state.get("completed")) or state.get("name") == "STATUS_FINAL"
    kickoff = event.get("date") or competition.get("date")
    return {
        "game_id": str(event["id"]),
        "season": integer((event.get("season") or {}).get("year")),
        "game_date": iso_date(kickoff) or "1970-01-01",
        "start_time_utc": kickoff,
        "home_team_id": str(home["team"]["id"]),
        "away_team_id": str(away["team"]["id"]),
        "status": state.get("name") or "UNKNOWN",
        "status_detail": detail,
        "home_score": integer(home.get("score")),
        "away_score": integer(away.get("score")),
        "venue": (competition.get("venue") or {}).get("fullName"),
        "neutral_site": int(bool(competition.get("neutralSite"))),
        "completed": int(completed),
    }


def _stat_map(stats):
    """ESPN often stores a value under name, abbreviation or label."""
    output = {}
    for stat in stats or []:
        v = stat.get("displayValue", stat.get("value"))
        for k in (stat.get("name"), stat.get("abbreviation"), stat.get("label")):
            if k:
                output[str(k).strip().lower()] = v
    return output


def _get(stats, *keys):
    for key in keys:
        value = stats.get(key.lower())
        if value is not None:
            return value
    return None


def _parse_team_stats(entry):
    stats = _stat_map(entry.get("statistics"))
    fg_m, fg_a = shot_pair(_get(stats, "fieldGoalsMade-fieldGoalsAttempted", "FG"))
    three_m, three_a = shot_pair(_get(stats, "threePointFieldGoalsMade-threePointFieldGoalsAttempted", "3PT", "3FG"))
    ft_m, ft_a = shot_pair(_get(stats, "freeThrowsMade-freeThrowsAttempted", "FT"))

    def prefer(number, name):
        return number if number is not None else integer(_get(stats, name))

    return {
        "points": integer(_get(stats, "points", "PTS")),
        "fg_m": prefer(fg_m, "fieldGoalsMade"),
        "fg_a": prefer(fg_a, "fieldGoalsAttempted"),
        "three_m": prefer(three_m, "threePointFieldGoalsMade"),
        "three_a": prefer(three_a, "threePointFieldGoalsAttempted"),
        "ft_m": prefer(ft_m, "freeThrowsMade"),
        "ft_a": prefer(ft_a, "freeThrowsAttempted"),
        "offensive_rebounds": integer(_get(stats, "offensiveRebounds", "OREB", "OR")),
        "defensive_rebounds": integer(_get(stats, "defensiveRebounds", "DREB", "DR")),
        "rebounds": integer(_get(stats, "totalRebounds", "rebounds", "REB")),
        "assists": integer(_get(stats, "assists", "AST")),
        "steals": integer(_get(stats, "steals", "STL")),
        "blocks": integer(_get(stats, "blocks", "BLK")),
        "turnovers": integer(_get(stats, "turnovers", "TO", "TOV")),
        "fouls": integer(_get(stats, "fouls", "totalFouls", "PF")),
    }


def extract_team_box(summary, game):
    """Extract both team totals and unadjusted opponent-shared possession efficiency."""
    box = summary.get("boxscore") or {}
    candidates = {}
    for entry in box.get("teams", []):
        team_id = (entry.get("team") or {}).get("id")
        if team_id:
            candidates[str(team_id)] = entry
    home_id, away_id = game["home_team_id"], game["away_team_id"]
    if home_id not in candidates or away_id not in candidates:
        return []
    rows = []
    for team_id, side in ((home_id, "home"), (away_id, "away")):
        entry = candidates[team_id]
        row = _parse_team_stats(entry)
        row.update({"game_id": game["game_id"], "team_id": team_id, "home_away": side, "raw_json": json.dumps(entry, separators=(",", ":"))})
        if row["points"] is None:
            row["points"] = game["home_score"] if side == "home" else game["away_score"]
        if all(row[k] is not None for k in ("fg_a", "offensive_rebounds", "turnovers", "ft_a")):
            row["team_possessions"] = row["fg_a"] - row["offensive_rebounds"] + row["turnovers"] + 0.475 * row["ft_a"]
        else:
            row["team_possessions"] = None
        rows.append(row)
    # Both teams in one game must use the same estimated possessions to prevent
    # inconsistent possession accounting between offense and defense.
    if all(r["team_possessions"] is not None and r["team_possessions"] > 0 for r in rows):
        possessions = sum(r["team_possessions"] for r in rows) / 2
    else:
        possessions = None
    for row in rows:
        other = next(r for r in rows if r["team_id"] != row["team_id"])
        row["possessions"] = possessions
        if possessions and row["points"] is not None and other["points"] is not None:
            row["offensive_efficiency"] = 100 * row["points"] / possessions
            row["defensive_efficiency"] = 100 * other["points"] / possessions
            row["net_efficiency"] = row["offensive_efficiency"] - row["defensive_efficiency"]
        else:
            row["offensive_efficiency"] = row["defensive_efficiency"] = row["net_efficiency"] = None
        del row["team_possessions"]
    return rows


def extract_player_box(summary):
    for group in (summary.get("boxscore") or {}).get("players", []):
        team_id = str((group.get("team") or {}).get("id") or "")
        if not team_id:
            continue
        for table in group.get("statistics", []):
            labels = [str(label).upper().strip() for label in (table.get("labels") or table.get("names") or [])]
            for appearance in table.get("athletes", []):
                athlete = appearance.get("athlete") or {}
                player_id = athlete.get("id")
                if not player_id:
                    continue
                stats = appearance.get("stats") or []
                parsed = dict(zip(labels, stats))
                fg_m, fg_a = shot_pair(parsed.get("FG"))
                three_m, three_a = shot_pair(parsed.get("3PT", parsed.get("3FG")))
                ft_m, ft_a = shot_pair(parsed.get("FT"))
                yield {
                    "player_id": str(player_id), "team_id": team_id,
                    "name": athlete.get("displayName") or athlete.get("fullName") or str(player_id),
                    "position": (athlete.get("position") or {}).get("abbreviation"),
                    "jersey": athlete.get("jersey"),
                    "starter": int(bool(appearance.get("starter"))),
                    "minutes": parsed.get("MIN"),
                    "points": integer(parsed.get("PTS")),
                    "rebounds": integer(parsed.get("REB")),
                    "assists": integer(parsed.get("AST")),
                    "steals": integer(parsed.get("STL")),
                    "blocks": integer(parsed.get("BLK")),
                    "turnovers": integer(parsed.get("TO", parsed.get("TOV"))),
                    "fg_m": fg_m, "fg_a": fg_a,
                    "three_m": three_m, "three_a": three_a,
                    "ft_m": ft_m, "ft_a": ft_a,
                    "raw_json": json.dumps(appearance, separators=(",", ":")),
                }
