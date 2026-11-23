"""Browser/mobile dashboard for the hosted NCAA men's D1 basketball database.

Streamlit executes this code on its server. DATABASE_URL is never sent to a
browser; data is read from PostgreSQL using parameterized, read-only queries.
"""
from __future__ import annotations

import os
import logging
import json
import re
from pathlib import Path
from html import escape
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="NCAA MBB | Game Schedules & Stats", page_icon="🏀", layout="wide")

CENTRAL = ZoneInfo("America/Chicago")



# Consistent white page, black text/borders and soft-gray control/card surfaces.
# This is applied from the Python entrypoint, so no second GitHub file is needed.
st.markdown("""
<style>
  :root { color-scheme: light !important; }
  html, body, .stApp, [data-testid="stAppViewContainer"],
  [data-testid="stHeader"], [data-testid="stMain"],
  [data-testid="stMainBlockContainer"] { background: #FFFFFF !important; color: #111111 !important; }
  .stApp, .stApp p, .stApp label, .stApp h1, .stApp h2, .stApp h3,
  .stApp h4, .stApp h5, .stApp h6,
  .stApp [data-testid="stMarkdownContainer"] { color: #111111 !important; }
  [data-testid="stSidebar"], [data-testid="stSidebarContent"] {
    background: #F2F3F5 !important; color: #111111 !important;
  }
  [data-testid="stMetric"], [data-testid="stVerticalBlockBorderWrapper"] > div,
  [data-testid="stAlert"] { background-color: #F6F6F6; color: #111111; border-radius: 8px; }
  [data-baseweb="select"] > div, [data-baseweb="input"] > div,
  [data-testid="stNumberInput"] input,
  .stApp input, .stApp textarea, .stApp button {
    background: #F4F4F4 !important; color: #111111 !important;
    border-color: #4A4A4A !important;
  }
  .stApp button:hover { background: #E7E7E7 !important; }
  [data-testid="stDataFrame"] { border: 1px solid #333333; border-radius: 8px; }
  .stApp a { color: #111111 !important; text-decoration: underline; }
  .mbb-team-card { display: inline-block; background: #F3F3F3;
                   border: 1px solid #333333; padding: 12px 16px;
                   border-radius: 9px; margin: 7px 0 12px; }
  .mbb-record { background: #E5E7EB; border: 1px solid #555;
                border-radius: 5px; padding: 3px 6px; font-size: .8rem;
                font-weight: 650; color: #111; margin-left: 5px; }
  .mbb-result { font-weight: 800; color: #111; margin-right: 6px; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def get_database_url() -> str:
    url = os.getenv("DATABASE_URL", "")
    if not url:
        try:
            url = st.secrets.get("DATABASE_URL", "")
        except (FileNotFoundError, KeyError):
            url = ""
    if not url:
        st.error("This website is not connected to the database yet. Add DATABASE_URL to Streamlit Cloud secrets.")
        st.stop()
    return url


@st.cache_data(ttl=300, show_spinner=False)
def query(sql: str, params: tuple = ()) -> pd.DataFrame:
    import psycopg
    from psycopg.rows import dict_row

    # No queries are built from user input; placeholders are parameterized.
    with psycopg.connect(get_database_url(), sslmode="require", connect_timeout=15,
                         prepare_threshold=None, row_factory=dict_row) as connection:
        with connection.cursor() as cursor:
            cursor.execute(sql, params)
            return pd.DataFrame(cursor.fetchall())


RANKINGS_QUERY = """
WITH base AS (
    SELECT t.team_id,t.name,t.abbreviation,t.conference,t.logo_url,
       COUNT(*) AS games,
       SUM(CASE WHEN s.points > o.points THEN 1 ELSE 0 END) AS wins,
       SUM(CASE WHEN s.points < o.points THEN 1 ELSE 0 END) AS losses,
       SUM(s.points) AS points_for,
       SUM(o.points) AS points_against,
       SUM(s.possessions) AS poss,
       SUM(s.fg_m) AS fg_m,
       SUM(s.fg_a) AS fg_a,
       SUM(s.three_m) AS three_m,
       SUM(s.turnovers) AS tov
    FROM team_game_stats s
    JOIN team_game_stats o ON o.game_id=s.game_id AND o.team_id<>s.team_id
    JOIN games g ON g.game_id=s.game_id AND g.completed=1
    JOIN teams t ON t.team_id=s.team_id
    WHERE t.is_d1=1 AND s.possessions>0 AND o.possessions>0 AND g.season=%s
    GROUP BY t.team_id,t.name,t.abbreviation,t.conference,t.logo_url
)
SELECT *,
  ROUND((100.0*points_for/NULLIF(poss,0))::numeric,1)::double precision AS off_eff,
  ROUND((100.0*points_against/NULLIF(poss,0))::numeric,1)::double precision AS def_eff,
  ROUND((100.0*(points_for-points_against)/NULLIF(poss,0))::numeric,1)::double precision AS net_eff,
  ROUND((100.0*(fg_m+0.5*three_m)/NULLIF(fg_a,0))::numeric,1)::double precision AS efg_pct,
  ROUND((100.0*turnovers/NULLIF(poss,0))::numeric,1)::double precision AS tov_pct
FROM (SELECT base.*,tov AS turnovers FROM base) r
ORDER BY net_eff DESC, name ASC
"""


# Season averages use the completed games already imported by GitHub Actions.
# Numeric playing time excludes ESPN DNP/DND rows from games played and averages.
# Shooting percentages use total makes / total attempts, not an average of game percentages.
PLAYER_SEASON_AVERAGES_QUERY = """
WITH appearances AS (
    SELECT t.team_id, t.name AS team, p.player_id, p.name AS player,
           p.position, p.jersey,
           NULLIF(COALESCE(
               s.raw_json::jsonb #>> '{athlete,displayHeight}',
               s.raw_json::jsonb #>> '{athlete,height}'
           ), '') AS height_raw,
           NULLIF(COALESCE(
               s.raw_json::jsonb #>> '{athlete,displayWeight}',
               s.raw_json::jsonb #>> '{athlete,weight}'
           ), '') AS weight_raw,
           CASE
             WHEN BTRIM(s.minutes) ~ '^[0-9]+:[0-9]{1,2}$'
               THEN SPLIT_PART(BTRIM(s.minutes), ':', 1)::numeric
                    + SPLIT_PART(BTRIM(s.minutes), ':', 2)::numeric / 60.0
             WHEN BTRIM(s.minutes) ~ '^[0-9]+([.][0-9]+)?$'
               THEN BTRIM(s.minutes)::numeric
             ELSE NULL
           END AS minutes_played,
           s.points, s.rebounds, s.assists, s.steals, s.blocks, s.turnovers,
           s.fg_m, s.fg_a, s.three_m, s.three_a, s.ft_m, s.ft_a
    FROM player_game_stats s
    JOIN games g ON g.game_id = s.game_id
    JOIN players p ON p.player_id = s.player_id
    JOIN teams t ON t.team_id = s.team_id
    WHERE g.season = %s AND g.completed = 1 AND t.is_d1 = 1
      AND (BTRIM(s.minutes) ~ '^[0-9]+:[0-9]{1,2}$'
           OR BTRIM(s.minutes) ~ '^[0-9]+([.][0-9]+)?$')
)
SELECT team, team_id, player_id, player, position, jersey,
       MAX(height_raw) AS height_raw, MAX(weight_raw) AS weight_raw,
       COUNT(*) AS gp,
       ROUND(AVG(minutes_played), 1)::double precision AS mpg,
       ROUND(AVG(COALESCE(points, 0)), 1)::double precision AS ppg,
       ROUND(AVG(COALESCE(rebounds, 0)), 1)::double precision AS rpg,
       ROUND(AVG(COALESCE(assists, 0)), 1)::double precision AS apg,
       ROUND(AVG(COALESCE(steals, 0)), 1)::double precision AS spg,
       ROUND(AVG(COALESCE(blocks, 0)), 1)::double precision AS bpg,
       ROUND(AVG(COALESCE(turnovers, 0)), 1)::double precision AS topg,
       ROUND(100.0 * COALESCE(SUM(fg_m), 0) / NULLIF(SUM(fg_a), 0), 1)::double precision AS fg_pct,
       ROUND(100.0 * COALESCE(SUM(three_m), 0) / NULLIF(SUM(three_a), 0), 1)::double precision AS three_pct,
       ROUND(100.0 * COALESCE(SUM(ft_m), 0) / NULLIF(SUM(ft_a), 0), 1)::double precision AS ft_pct
FROM appearances
GROUP BY team, team_id, player_id, player, position, jersey
ORDER BY LOWER(team), ppg DESC, LOWER(player)
"""


# Team Explorer: season-to-date per-game averages, using completed games for
# which ESPN supplied an actual team box score. Shooting percentages are based
# on combined season makes/attempts, not the mean of game percentages.
TEAM_SEASON_AVERAGES_QUERY = """
SELECT COUNT(*)::integer AS gp,
       ROUND(AVG(s.points)::numeric, 1)::double precision AS ppg,
       ROUND(AVG(o.points)::numeric, 1)::double precision AS opp_ppg,
       ROUND(AVG(s.rebounds)::numeric, 1)::double precision AS rpg,
       ROUND(AVG(s.assists)::numeric, 1)::double precision AS apg,
       ROUND(AVG(s.steals)::numeric, 1)::double precision AS spg,
       ROUND(AVG(s.blocks)::numeric, 1)::double precision AS bpg,
       ROUND(AVG(s.turnovers)::numeric, 1)::double precision AS topg,
       ROUND(100.0 * SUM(s.fg_m) / NULLIF(SUM(s.fg_a), 0), 1)::double precision AS fg_pct,
       ROUND(100.0 * SUM(s.three_m) / NULLIF(SUM(s.three_a), 0), 1)::double precision AS three_pct,
       ROUND(100.0 * SUM(s.ft_m) / NULLIF(SUM(s.ft_a), 0), 1)::double precision AS ft_pct
FROM team_game_stats s
JOIN games g ON g.game_id = s.game_id
LEFT JOIN team_game_stats o ON o.game_id = s.game_id AND o.team_id <> s.team_id
WHERE s.team_id = %s AND g.season = %s AND g.completed = 1
  AND s.points IS NOT NULL
"""


def show_table(frame: pd.DataFrame, columns: list[str] | None = None):
    st.dataframe(frame[columns] if columns else frame, use_container_width=True,
                 hide_index=True)


def _visible_jersey(value) -> str:
    """Only append a jersey number when the importer actually has one."""
    if value is None or pd.isna(value):
        return ""
    number = str(value).strip()
    return number if number and number.lower() not in ("nan", "none", "null") else ""


def _display_height(value) -> str:
    """ESPN provides displayHeight (e.g. 6' 5\") or numeric inches."""
    if value is None or pd.isna(value):
        return "—"
    text = str(value).strip()
    if not text:
        return "—"
    if text.replace(".", "", 1).isdigit():
        inches = float(text)
        if 48 <= inches <= 96:
            return f"{int(inches) // 12}'{int(inches) % 12}\""
        return "—"
    return text


def _display_weight(value) -> str:
    if value is None or pd.isna(value):
        return "—"
    text = str(value).strip()
    if not text:
        return "—"
    if text.replace(".", "", 1).isdigit():
        pounds = float(text)
        return f"{int(pounds)} lbs" if 80 <= pounds <= 500 else "—"
    return text


def _extract_roster_athletes(payload: dict) -> list[dict]:
    """Support ESPN's team detail and roster endpoints without guessing IDs."""
    team = payload.get("team") or {}
    athletes = team.get("athletes") or payload.get("athletes") or []
    if isinstance(athletes, dict):
        athletes = athletes.get("items") or athletes.get("athletes") or []
    found = []
    if not isinstance(athletes, list):
        return found
    for item in athletes:
        if not isinstance(item, dict):
            continue
        # Some rosters group athletes inside categories (e.g. guards/forwards).
        entries = item.get("items") or item.get("athletes")
        if isinstance(entries, list):
            found.extend(e for e in entries if isinstance(e, dict))
        elif item.get("id") is not None:
            found.append(item)
    return found


@st.cache_data(ttl=21600, show_spinner=False)
def current_roster_profiles(team_id: str) -> dict[str, dict]:
    """Fetch one selected team's CURRENT ESPN roster; don't scan every D1 team."""
    if not str(team_id).isdigit():
        return {}
    url = ("https://site.api.espn.com/apis/site/v2/sports/basketball/"
           f"mens-college-basketball/teams/{team_id}")
    try:
        response = requests.get(url, params={"enable": "roster"}, timeout=10,
                                headers={"Accept": "application/json",
                                         "User-Agent": "NCAA-MBB-Dashboard/1.0"})
        response.raise_for_status()
        athletes = _extract_roster_athletes(response.json())
    except (requests.RequestException, ValueError, TypeError, KeyError):
        return {}
    return {str(player["id"]): player for player in athletes if player.get("id") is not None}


def player_averages_display_rows(frame: pd.DataFrame, profiles: dict[str, dict],
                                 injury_by_player: dict[str, dict] | None = None) -> pd.DataFrame:
    """Combine player name/jersey and official game availability, plus roster details."""
    shown = frame.copy()
    injury_by_player = injury_by_player or {}
    visible_names = []
    heights = []
    weights = []
    for _, row in shown.iterrows():
        profile = profiles.get(str(row["player_id"]), {})
        number = _visible_jersey(row.get("jersey")) or _visible_jersey(profile.get("jersey"))
        visible_name = str(row["player"]) + (f" #{number}" if number else "")
        report = injury_by_player.get(str(row["player_id"]))
        if report:
            injury_type = str(report.get("injury_detail") or "").strip()
            injury_status = str(report.get("status") or "").strip()
            description = (f"{injury_type} ({injury_status})" if injury_type and injury_status
                           else injury_type or injury_status)
            if description:
                visible_name += f" ⛑️ {description}"
        visible_names.append(visible_name)
        height = profile.get("displayHeight") or profile.get("height") or row.get("height_raw")
        weight = profile.get("displayWeight") or profile.get("weight") or row.get("weight_raw")
        heights.append(_display_height(height))
        weights.append(_display_weight(weight))
    shown["player"] = visible_names
    shown["height"] = heights
    shown["weight"] = weights
    return shown


SCHEDULE_QUERY = """
    SELECT g.game_id, g.game_date, g.start_time_utc, g.status_detail, g.status,
           g.completed, g.away_score, g.home_score,
           a.team_id AS away_id, a.name AS away, a.logo_url AS away_logo,
           a.conference AS away_conference,
           h.team_id AS home_id, h.name AS home, h.logo_url AS home_logo,
           h.conference AS home_conference, g.venue
    FROM games g
    JOIN teams a ON a.team_id = g.away_team_id
    JOIN teams h ON h.team_id = g.home_team_id
    WHERE g.season = %s AND g.game_date >= %s AND g.game_date <= %s
    ORDER BY g.game_date, g.start_time_utc NULLS LAST, g.game_id
"""


def central_tipoff(timestamp: str | None) -> str:
    """Format ESPN's stored UTC tip-off time as local Central time."""
    if not timestamp or pd.isna(timestamp):
        return "TBD"
    try:
        value = datetime.fromisoformat(str(timestamp).replace("Z", "+00:00"))
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(CENTRAL).strftime("%I:%M %p").lstrip("0")
    except ValueError:
        return "TBD"


def parse_ap_poll(payload: dict) -> tuple[dict[str, int], str]:
    """Select the actual AP poll, never silently substitute the Coaches poll."""
    for poll in payload.get("rankings", []):
        poll_name = str(poll.get("shortName") or poll.get("name") or "")
        if "ap" not in poll_name.lower().split() and not poll_name.lower().startswith("ap "):
            continue
        ranks = {}
        for entry in poll.get("ranks", []):
            team_id = (entry.get("team") or {}).get("id")
            position = entry.get("current")
            if team_id is None or position is None:
                continue
            try:
                number = int(position)
            except (ValueError, TypeError):
                continue
            if 1 <= number <= 25:
                ranks[str(team_id)] = number
        if ranks:
            return ranks, poll_name
    return {}, ""


@st.cache_data(ttl=3600, show_spinner=False)
def current_ap_poll() -> tuple[dict[str, int], str]:
    """Fetch ESPN's most recent AP Top 25, keeping the site usable if unavailable."""
    endpoint = ("https://site.api.espn.com/apis/site/v2/sports/"
                "basketball/mens-college-basketball/rankings")
    try:
        response = requests.get(endpoint, timeout=10,
                                headers={"Accept": "application/json",
                                         "User-Agent": "NCAA-MBB-Dashboard/1.0"})
        response.raise_for_status()
        return parse_ap_poll(response.json())
    except (requests.RequestException, ValueError, TypeError, KeyError):
        return {}, ""


def parse_weekly_ap_poll(payload: dict) -> dict[str, int]:
    """Parse ESPN Core's historical AP Top 25, where teams are $ref links.

    Ranking source ID 1 is the AP poll. Do not substitute coaches rankings.
    """
    if not isinstance(payload, dict):
        return {}
    name = str(payload.get("name") or payload.get("shortName") or "").lower()
    if "coach" in name or (name and "ap" not in name.split()
                          and "associated press" not in name):
        return {}
    ranks = {}
    for row in payload.get("ranks", []):
        if not isinstance(row, dict):
            continue
        team = row.get("team") or {}
        if not isinstance(team, dict):
            continue
        team_id = team.get("id")
        if team_id is None:
            reference = str(team.get("$ref") or "")
            match = re.search(r"/teams/([0-9]+)(?:[/?#]|$)", reference)
            if match:
                team_id = match.group(1)
        try:
            position = int(row.get("current"))
        except (ValueError, TypeError):
            continue
        if team_id is not None and 1 <= position <= 25:
            ranks[str(team_id)] = position
    return ranks


def core_rankings_weeks(payload: dict) -> list[int]:
    """Find actual regular-season weeks; never guess a historical poll week."""
    if not isinstance(payload, dict):
        return []
    weeks = set()
    for item in payload.get("items", []):
        if not isinstance(item, dict):
            continue
        ref = str(item.get("$ref") or "")
        match = re.search(r"/weeks/([0-9]+)(?:[/?#]|$)", ref)
        if match:
            weeks.add(int(match.group(1)))
    return sorted(weeks, reverse=True)


@st.cache_data(ttl=21600, show_spinner=False)
def historical_ap_poll(espn_end_year: int) -> tuple[dict[str, int], str]:
    """Most recent *available regular-season* AP poll from the selected year.

    ESPN Core exposes archived polls by season and week. When unavailable,
    return no ranking rather than showing an unrelated current-year AP poll.
    """
    base = ("https://sports.core.api.espn.com/v2/sports/basketball/"
            "leagues/mens-college-basketball/seasons/"
            f"{int(espn_end_year)}/types/2")
    headers = {"Accept": "application/json", "User-Agent": "NCAA-MBB-Dashboard/1.0"}
    try:
        response = requests.get(f"{base}/weeks?limit=200", timeout=7,
                                headers=headers)
        response.raise_for_status()
        weeks = core_rankings_weeks(response.json())
    except (requests.RequestException, ValueError, TypeError):
        return {}, ""
    # Later weeks can lack AP polls; try up to eight of the most recent weeks.
    for week in weeks[:8]:
        try:
            response = requests.get(
                f"{base}/weeks/{week}/rankings/1", timeout=5, headers=headers)
            response.raise_for_status()
            rankings = parse_weekly_ap_poll(response.json())
            if rankings:
                return rankings, f"ESPN AP Top 25, regular-season week {week}"
        except (requests.RequestException, ValueError, TypeError):
            continue
    return {}, ""


def comparison_ap_poll(season: int, latest_season: int) -> tuple[dict[str, int], str]:
    """Current AP for current year, season-specific archived AP for older years."""
    if int(season) == int(latest_season):
        ranks, name = current_ap_poll()
        return ranks, f"Current {name or 'AP Top 25'}" if ranks else ""
    return historical_ap_poll(int(season))


def comparison_team_name(name: str, conference: str, team_id,
                         ap_ranks: dict[str, int], poll_found: bool) -> str:
    """Streamlit selectboxes support text; logos appear in linked team cells."""
    rank = ap_ranks.get(str(team_id))
    prefix = (f"AP #{rank} " if rank is not None else "NR " if poll_found else "")
    return f"{prefix}{name} · {conference}"


def rank_maps(season: int, latest_season: int) -> tuple[dict[str, int], dict[str, int], pd.DataFrame]:
    """Efficiency ranks use imported box scores; AP ranks use the current ESPN poll."""
    rankings = query(RANKINGS_QUERY, (season,))
    efficiency = {str(team_id): i for i, team_id in enumerate(rankings.get("team_id", []), start=1)}
    ap, _ = current_ap_poll() if season == latest_season else ({}, "")
    return efficiency, ap, rankings



# Overall W-L records are based on all imported completed games with final scores,
# not just games with complete team-box-score statistics.
SEASON_RECORDS_QUERY = """
SELECT team_id, SUM(wins)::integer AS wins, SUM(losses)::integer AS losses
FROM (
    SELECT home_team_id AS team_id,
           CASE WHEN home_score > away_score THEN 1 ELSE 0 END AS wins,
           CASE WHEN home_score < away_score THEN 1 ELSE 0 END AS losses
    FROM games WHERE season=%s AND completed=1
      AND home_score IS NOT NULL AND away_score IS NOT NULL
    UNION ALL
    SELECT away_team_id AS team_id,
           CASE WHEN away_score > home_score THEN 1 ELSE 0 END AS wins,
           CASE WHEN away_score < home_score THEN 1 ELSE 0 END AS losses
    FROM games WHERE season=%s AND completed=1
      AND home_score IS NOT NULL AND away_score IS NOT NULL
) completed_games
GROUP BY team_id
"""


def season_records(season: int) -> dict[str, tuple[int, int]]:
    frame = query(SEASON_RECORDS_QUERY, (season, season))
    return {str(row['team_id']): (int(row['wins']), int(row['losses']))
            for _, row in frame.iterrows()}


def record_label(team_id, records: dict[str, tuple[int, int]]) -> str:
    record = records.get(str(team_id))
    return f"{record[0]}–{record[1]}" if record else "—"


def parse_conference_standings(payload: dict) -> dict[str, int]:
    """Read ESPN's conference-group standings (not an overall league ranking).

    The ESPN level=3 response nests conference groups under `children`, each
    with ordered `standings.entries`. No team gets a fabricated official rank.
    """
    result: dict[str, int] = {}

    def visit(node, depth=0):
        if not isinstance(node, dict) or depth > 6:
            return
        children = node.get('children') or []
        if not isinstance(children, list):
            children = []
        standings = node.get('standings') or {}
        entries = standings.get('entries') if isinstance(standings, dict) else None
        # Only assign positions at conference level, NOT the root all-D1 group.
        group_name = str(node.get('name') or '').strip().lower()
        group_id = str(node.get('id') or node.get('groupId') or '')
        is_league_root = (group_id == '50' or
                          group_name in ('ncaa division i', 'division i',
                                         'mens college basketball', "men's college basketball"))
        if isinstance(entries, list) and entries and not is_league_root and not children:
            rank = 0
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                team = entry.get('team') or {}
                ident = team.get('id') if isinstance(team, dict) else None
                if ident is None:
                    continue
                rank += 1
                result[str(ident)] = rank
        for child in children:
            visit(child, depth + 1)

    visit(payload)
    return result


@st.cache_data(ttl=3600, show_spinner=False)
def espn_conference_positions(season: int) -> dict[str, int]:
    """Conference standings positions from ESPN, one public request per hour."""
    endpoint = ("https://site.web.api.espn.com/apis/v2/sports/"
                "basketball/mens-college-basketball/standings")
    try:
        response = requests.get(endpoint, params={
            'region': 'us', 'lang': 'en', 'contentorigin': 'espn',
            'season': season, 'group': 50, 'type': 0, 'level': 3,
            'sort': ('leaguewinpercent:desc,vsconf_winpercent:desc,'
                     'vsconf_gamesbehind:asc,vsconf_playoffseed:asc,'
                     'wins:desc,losses:asc,alpha:asc'),
        }, timeout=12, headers={'Accept': 'application/json',
                                'User-Agent': 'NCAA-MBB-Dashboard/1.0'})
        response.raise_for_status()
        return parse_conference_standings(response.json())
    except (requests.RequestException, ValueError, TypeError, KeyError):
        return {}


def sort_conference_teams(teams: pd.DataFrame, positions: dict[str, int],
                          records: dict[str, tuple[int, int]]) -> pd.DataFrame:
    """Group by conference; ESPN standings order, then clearly marked fallback."""
    shown = teams.copy()
    shown['sort_conference'] = shown['Conference'].str.casefold().map(
        lambda c: 'zzzz' if c == '—' else c)
    shown['official_rank'] = shown['team_id'].map(lambda x: positions.get(str(x)))
    shown['sort_missing'] = shown['official_rank'].isna().astype(int)
    shown['win_pct'] = shown['team_id'].map(
        lambda ident: (records.get(str(ident), (0, 0))[0] /
                       max(1, sum(records.get(str(ident), (0, 0))))))
    shown['wins_for_sort'] = shown['team_id'].map(
        lambda ident: records.get(str(ident), (0, 0))[0])
    # When ESPN is unavailable, order teams in each conference by overall W-L,
    # but do not label that an official conference standing.
    return shown.sort_values(
        ['sort_conference', 'sort_missing', 'official_rank', 'win_pct',
         'wins_for_sort', 'name'],
        ascending=[True, True, True, False, False, True],
        na_position='last', kind='stable')


def matchup_result(game: pd.Series, selected_id: str) -> str:
    """W/L and score from the chosen team's viewpoint, not home-team first."""
    is_home = str(game['home_team_id']) == str(selected_id)
    ours = game['home_score'] if is_home else game['away_score']
    theirs = game['away_score'] if is_home else game['home_score']
    if pd.isna(ours) or pd.isna(theirs):
        return '—'
    ours, theirs = int(ours), int(theirs)
    result = 'W' if ours > theirs else ('L' if ours < theirs else 'T')
    return (f'<span class="mbb-result">{result}</span>'
            f'{safe_cell(ours)}–{safe_cell(theirs)}')


def logo_url(value) -> str:
    """Only display valid remote HTTPS logo links already stored by the importer."""
    return value if isinstance(value, str) and value.startswith("https://") else ""


def rank_label(team_id, ranks: dict[str, int]) -> str:
    value = ranks.get(str(team_id))
    return f"#{value}" if value is not None else "—"


def ranked_team_name(name, team_id, ap_ranks: dict[str, int]) -> str:
    """Place the current AP poll rank inside the team-name column."""
    position = ap_ranks.get(str(team_id))
    return f"#{position} {name}" if position is not None else str(name)


@st.cache_data(ttl=21600, show_spinner=False)
def current_conference_logos() -> dict[str, str]:
    """Map ESPN conference names to their published logos; never guess artwork URLs.

    Conference display names were imported from this same ESPN endpoint, so
    a normalized name match avoids changes to Supabase or the nightly importer.
    """
    endpoint = ("https://site.api.espn.com/apis/site/v2/sports/"
                "basketball/mens-college-basketball/scoreboard/conferences")
    try:
        response = requests.get(endpoint, params={"seasontype": 2}, timeout=12,
                                headers={"Accept": "application/json",
                                         "User-Agent": "NCAA-MBB-Dashboard/1.0"})
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            return {}
        output = {}
        for conference in payload.get("conferences") or []:
            if not isinstance(conference, dict):
                continue
            # ESPN's conference listing publishes 'logo' (a direct URL).
            raw_logo = conference.get("logo")
            if isinstance(raw_logo, dict):
                raw_logo = raw_logo.get("href") or raw_logo.get("url")
            if not raw_logo and isinstance(conference.get("logos"), list):
                first = next((item for item in conference["logos"]
                              if isinstance(item, dict) and
                              (item.get("href") or item.get("url"))), None)
                raw_logo = (first.get("href") or first.get("url")) if first else None
            url = logo_url(raw_logo)
            if not url:
                continue
            for name_key in ("name", "shortName", "abbreviation"):
                name = conference.get(name_key)
                if name and str(name).strip():
                    output[str(name).strip().casefold()] = url
        return output
    except (requests.RequestException, ValueError, TypeError, KeyError):
        # If ESPN is unavailable, show readable conference names instead.
        return {}


def conference_logo(name, conference_logos: dict[str, str]) -> str:
    if not isinstance(name, str) or not name.strip():
        return ""
    return conference_logos.get(name.strip().casefold(), "")


def safe_cell(value) -> str:
    """Escape database values before they are added to the presentation HTML."""
    if value is None or (not isinstance(value, (list, tuple, dict)) and pd.isna(value)):
        return "—"
    return escape(str(value))


def team_badge(name, team_id, image_url, ap_ranks: dict[str, int],
               records: dict[str, tuple[int, int]] | None = None) -> str:
    """A single team cell in the order AP rank → team logo → team name."""
    position = ap_ranks.get(str(team_id))
    ap = (f'<span class="mbb-ap">#{int(position)}</span>' if position is not None
          else '<span class="mbb-ap" aria-label="Unranked"></span>')
    url = logo_url(image_url)
    picture = (f'<img class="mbb-team-logo" src="{escape(url, quote=True)}" '
               'width="26" height="26" loading="lazy" alt="">' if url
               else '<span class="mbb-logo-placeholder" aria-hidden="true"></span>')
    record_html = (f'<span class="mbb-record">{record_label(team_id, records)}</span>'
                   if records is not None else '')
    return (f'<span class="mbb-team">{ap}{picture}'
            f'<span class="mbb-team-name">{safe_cell(name)}</span>'
            f'{record_html}</span>')


def conference_badge(name, conference_logos: dict[str, str]) -> str:
    """Show ESPN conference artwork when possible, otherwise show readable text."""
    url = conference_logo(name, conference_logos)
    if not url:
        return safe_cell(name)
    safe_name = safe_cell(name)
    return (f'<img class="mbb-conference-logo" src="{escape(url, quote=True)}" '
            f'alt="{safe_name} conference" title="{safe_name}" '
            'width="32" height="28" loading="lazy">')


def show_visual_table(headers: list[str], rows: list[list[str]]) -> None:
    """HTML table allows a logo and AP rank in the SAME team-name column.

    Streamlit's standard dataframe does not support inline images and text in
    one cell. All plain text is escaped, while image URLs are HTTPS-validated.
    """
    css = """
    <style>
      .mbb-table-scroll {width:100%; overflow-x:auto; border:1px solid #333333;
                         border-radius:9px; margin:.5rem 0 1.1rem;}
      table.mbb-table {border-collapse:collapse; width:100%; min-width:790px;
                       font-size:.88rem; background:#F4F4F4; color:#111111;}
      .mbb-table th {padding:10px 12px; text-align:left; white-space:nowrap;
                     background:#E6E6E6; color:#111111; font-weight:650;
                     border-bottom:1px solid #333333;}
      .mbb-table td {padding:9px 12px; white-space:nowrap; vertical-align:middle;
                     border-bottom:1px solid #B8B8B8; color:#111111;}
      .mbb-table tr:last-child td {border-bottom:0;}
      .mbb-table tbody tr:hover {background:#E7E7E7;}
      .mbb-team {display:inline-flex; align-items:center; gap:7px; min-width:155px;}
      .mbb-ap {display:inline-block; min-width:28px; font-weight:700; font-size:.84rem;}
      .mbb-team-logo {width:26px; height:26px; object-fit:contain; flex:0 0 26px;}
      .mbb-logo-placeholder {display:inline-block; width:26px; flex:0 0 26px;}
      .mbb-team-name {font-weight:550;}
      .mbb-conference-logo {height:28px; width:32px; object-fit:contain;}
    </style>
    """
    header_html = "".join(f"<th scope=\"col\">{safe_cell(h)}</th>" for h in headers)
    body_html = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>"
                        for row in rows)
    html = (css + '<div class="mbb-table-scroll"><table class="mbb-table">'
            f'<thead><tr>{header_html}</tr></thead><tbody>{body_html}</tbody>'
            '</table></div>')
    st.markdown(html, unsafe_allow_html=True)


def show_schedule(games: pd.DataFrame, efficiency_ranks: dict[str, int],
                  ap_ranks: dict[str, int],
                  conference_logos: dict[str, str] | None = None):
    conference_logos = conference_logos or {}
    rows = []
    for _, game in games.iterrows():
        rows.append([
            safe_cell(central_tipoff(game["start_time_utc"])),
            team_badge(game["away"], game["away_id"], game["away_logo"], ap_ranks),
            conference_badge(game["away_conference"], conference_logos),
            safe_cell(rank_label(game["away_id"], efficiency_ranks)),
            team_badge(game["home"], game["home_id"], game["home_logo"], ap_ranks),
            conference_badge(game["home_conference"], conference_logos),
            safe_cell(rank_label(game["home_id"], efficiency_ranks)),
            safe_cell(game["venue"]),
        ])
    show_visual_table(["Time (CT)", "Away team", "Away conf.", "Away Eff.",
                       "Home team", "Home conf.", "Home Eff.", "Venue"], rows)


def load_schedule(season: int, start_date: str, end_date: str) -> pd.DataFrame:
    return query(SCHEDULE_QUERY, (season, start_date, end_date))


def select_next_five_games(games: pd.DataFrame, now_utc: datetime) -> pd.DataFrame:
    """Pick the next five not-yet-started games from ESPN's imported schedule.

    The `completed=0` SQL filter alone also includes games in progress, so
    exclude any match whose known UTC tipoff has already passed. Missing
    tipoff times are kept as TBD unless the game has already started.
    """
    if games.empty:
        return games.copy()
    upcoming = games.copy()
    tipoffs = pd.to_datetime(upcoming["start_time_utc"], errors="coerce", utc=True)
    current_time = pd.Timestamp(now_utc)
    if current_time.tzinfo is None:
        current_time = current_time.tz_localize("UTC")
    else:
        current_time = current_time.tz_convert("UTC")
    upcoming = upcoming.loc[tipoffs.isna() | (tipoffs > current_time)].copy()
    if upcoming.empty:
        return upcoming
    upcoming["_tipoff_sort"] = pd.to_datetime(upcoming["start_time_utc"],
                                               errors="coerce", utc=True)
    upcoming = upcoming.sort_values(["game_date", "_tipoff_sort", "game_id"],
                                    kind="stable", na_position="last")
    return upcoming.head(5).drop(columns=["_tipoff_sort"])




def historical_season_label(espn_end_year: int) -> str:
    """ESPN IDs refer to the year in which the basketball season ends."""
    year = int(espn_end_year)
    return f"{year - 1}–{year % 100:02d}"


def historical_rankings(rows: pd.DataFrame) -> pd.DataFrame:
    """Assign national ranks BEFORE any Top-N display filter is applied."""
    if rows.empty:
        return rows.copy()
    valid = rows.dropna(subset=["adj_net"]).copy()
    valid["adj_net"] = pd.to_numeric(valid["adj_net"], errors="coerce")
    valid = valid.dropna(subset=["adj_net"])
    valid = valid.sort_values(
        ["adj_net", "team", "team_id"],
        ascending=[False, True, True], kind="stable"
    ).reset_index(drop=True)
    valid.insert(0, "national_rank", range(1, len(valid) + 1))
    return valid


def historical_trend_chart(snapshots: pd.DataFrame, metric: str,
                           labels: dict[str, str]) -> pd.DataFrame:
    """Align team snapshots on real dates without inventing missing game dates."""
    if snapshots.empty:
        return pd.DataFrame()
    points = snapshots[["snapshot_date", "team_id", metric]].copy()
    points["snapshot_date"] = pd.to_datetime(points["snapshot_date"], format="%Y-%m-%d", errors="coerce")
    points[metric] = pd.to_numeric(points[metric], errors="coerce")
    points["Team"] = points["team_id"].astype(str).map(labels)
    points = points.dropna(subset=["snapshot_date", metric, "Team"])
    if points.empty:
        return pd.DataFrame()
    return points.pivot_table(index="snapshot_date", columns="Team", values=metric,
                              aggfunc="last").sort_index()


def historical_number(value, signed: bool = False) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value):+.1f}" if signed else f"{float(value):.1f}"




def scoring_pattern_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    """Retrospective descriptive scoring variation only; no future fixtures or picks.

    Rows must be from completed games with usable box scores. A null prior
    season average is not silently replaced by a current-game statistic.
    """
    cols = ["game_date", "team_id", "team", "location", "games_before", "prior_ppg", "points"]
    if frame.empty:
        return pd.DataFrame(columns=cols + ["score_difference", "absolute_difference", "history_group", "month"])
    df = frame[cols].copy()
    for col in ("games_before", "prior_ppg", "points"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    df = df.dropna(subset=["game_date", "prior_ppg", "points", "games_before"])
    df = df[(df["games_before"] >= 5) & (df["points"] >= 0)]
    df["score_difference"] = df["points"] - df["prior_ppg"]
    df["absolute_difference"] = df["score_difference"].abs()
    df["history_group"] = pd.cut(df["games_before"],
                                  bins=[4, 9, 19, float("inf")],
                                  labels=["5–9 prior games", "10–19 prior games", "20+ prior games"])
    df["month"] = df["game_date"].dt.strftime("%Y-%m")
    return df.sort_values(["game_date", "team"], kind="stable")


def scoring_pattern_group_summary(frame: pd.DataFrame, by: str) -> pd.DataFrame:
    """Aggregated historical observations, never forward game estimates."""
    if frame.empty:
        return pd.DataFrame(columns=[by, "Team box scores", "Scored PPG", "Typical variation", "Score SD"])
    groups = frame.groupby(by, observed=True, dropna=False).agg(
        games=("points", "size"),
        ppg=("points", "mean"),
        deviation=("absolute_difference", "mean"),
        standard_deviation=("points", "std")
    ).reset_index()
    return groups.rename(columns={"games": "Team box scores", "ppg": "Scored PPG",
                                  "deviation": "Typical variation", "standard_deviation": "Score SD"}).round(2)



# Retrospective feature associations: the values below are from completed games,
# with feature columns supplied by the STRICTLY PRE-GAME dataset. No model is
# trained and no results for upcoming games or betting decisions are generated.
FEATURE_ASSOCIATIONS = {
    "Prior adjusted net efficiency": ("prior_adj_net", "margin", "Actual scoring margin"),
    "Prior strength of schedule": ("prior_sos", "margin", "Actual scoring margin"),
    "Prior offensive efficiency": ("prior_offense", "points", "Actual team points"),
    "Prior defensive efficiency": ("prior_defense", "opp_points", "Actual points allowed"),
    "Prior raw net efficiency": ("prior_raw_net", "margin", "Actual scoring margin"),
    "Prior pace (possessions/game)": ("prior_pace", "total_points", "Actual combined points"),
    "Recent five scoring margin": ("recent_margin", "margin", "Actual scoring margin"),
    "Rest days": ("rest_days", "margin", "Actual scoring margin"),
    "Prior points per game": ("prior_ppg", "points", "Actual team points"),
}

FEATURE_ANALYSIS_SQL = """
SELECT f.season, f.game_id, f.game_date, f.team_id, t.name AS team,
       f.games_before, f.rating_snapshot_date,
       f.adjusted_net_before AS prior_adj_net,
       f.opponent_sos_before AS prior_sos,
       f.raw_offense AS prior_offense,
       f.raw_defense AS prior_defense,
       f.raw_net AS prior_raw_net,
       f.avg_points_for AS prior_ppg,
       f.avg_possessions AS prior_pace,
       f.recent_five_margin AS recent_margin,
       f.rest_days,
       CASE WHEN f.neutral_site=1 THEN 'Neutral'
            WHEN f.is_home=1 THEN 'Home' ELSE 'Away' END AS venue,
       s.points AS points, o.points AS opp_points
FROM pregame_team_features f
JOIN games g ON g.game_id=f.game_id AND g.season=f.season AND g.completed=1
JOIN teams t ON t.team_id=f.team_id AND t.is_d1=1
JOIN teams ot ON ot.team_id=f.opponent_id AND ot.is_d1=1
JOIN team_game_stats s ON s.game_id=f.game_id AND s.team_id=f.team_id
JOIN team_game_stats o ON o.game_id=f.game_id AND o.team_id=f.opponent_id
WHERE f.season BETWEEN %s AND %s
  AND s.points IS NOT NULL AND o.points IS NOT NULL
  AND s.possessions > 0 AND o.possessions > 0
  AND ((f.is_home=1 AND s.points=g.home_score AND o.points=g.away_score)
    OR (f.is_home=0 AND s.points=g.away_score AND o.points=g.home_score))
ORDER BY f.season,f.game_date,f.game_id,f.team_id
"""


def feature_analysis_prepare(raw: pd.DataFrame, min_history: int = 5) -> pd.DataFrame:
    """Use only completed D1 matchups (guaranteed by SQL) with past-game history.

    Any absent/non-prior adjusted snapshot is masked, NOT forward-filled.
    The associated feature never uses data from the completed game itself.
    """
    if raw.empty:
        return raw.copy()
    data = raw.copy()
    for col in ("games_before", "prior_adj_net", "prior_sos", "prior_offense",
                "prior_defense", "prior_raw_net", "prior_ppg", "prior_pace",
                "recent_margin", "rest_days", "points", "opp_points"):
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data["game_date"] = pd.to_datetime(data["game_date"], errors="coerce")
    ratings_date = pd.to_datetime(data["rating_snapshot_date"], errors="coerce")
    has_prior_rating = ratings_date.notna() & data["game_date"].notna() & (ratings_date < data["game_date"])
    data["invalid_prior_rating"] = ((data["prior_adj_net"].notna() | data["prior_sos"].notna())
                                     & ~has_prior_rating)
    data.loc[~has_prior_rating, ["prior_adj_net", "prior_sos"]] = float("nan")
    data = data.dropna(subset=["game_date", "games_before", "points", "opp_points"])
    data = data[(data["games_before"] >= min_history) &
                (data["points"] >= 0) & (data["opp_points"] >= 0)].copy()
    data["margin"] = data["points"] - data["opp_points"]
    data["total_points"] = data["points"] + data["opp_points"]
    data["rest_band"] = data["rest_days"].apply(
        lambda x: "Unknown" if pd.isna(x) else
        "0 days" if x == 0 else "1 day" if x == 1 else
        "2 days" if x == 2 else "3+ days" if x >= 3 else "Unknown")
    return data


def feature_analysis_pairs(data: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Validated numeric pairs for one historical association."""
    col, outcome, _ = FEATURE_ASSOCIATIONS[metric]
    if data.empty:
        return pd.DataFrame(columns=["season", "game_date", "team", "venue", col, outcome])
    part = data[["season", "game_date", "team", "venue", col, outcome]].copy()
    return part.dropna(subset=[col, outcome])


def feature_analysis_correlations(data: pd.DataFrame, minimum: int = 30) -> pd.DataFrame:
    """Descriptive pooled Pearson correlations; never causal or forecast accuracy."""
    items = []
    for metric, (col, target, title) in FEATURE_ASSOCIATIONS.items():
        valid = feature_analysis_pairs(data, metric)
        n = len(valid)
        correlation = (float(valid[col].corr(valid[target]))
                       if n >= minimum and valid[col].nunique() > 1
                       and valid[target].nunique() > 1 else float("nan"))
        items.append({"Statistic": metric, "Historical outcome": title,
                      "Team-game rows": n, "Pearson r": correlation})
    return pd.DataFrame(items)


def feature_analysis_bands(pairs: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Quartile grouping of past-game features, with no extrapolation."""
    col, target, _ = FEATURE_ASSOCIATIONS[metric]
    if len(pairs) < 4 or pairs[col].nunique() < 2:
        return pd.DataFrame(columns=["Prior-statistic range", "Team-game rows", "Mean outcome"])
    series = pd.qcut(pairs[col], q=4, duplicates="drop")
    grouped = pairs.assign(band=series).groupby("band", observed=True)[target].agg(["count", "mean"]).reset_index()
    grouped["band"] = grouped["band"].astype(str)
    return grouped.rename(columns={"band": "Prior-statistic range",
                                   "count": "Team-game rows", "mean": "Mean outcome"}).round(2)


def feature_analysis_venue_rest(data: pd.DataFrame, field: str) -> pd.DataFrame:
    """Descriptive group counts and scored/allowed points, not causal effects."""
    if data.empty:
        return pd.DataFrame(columns=[field, "Team-game rows", "Avg scored", "Avg allowed", "Avg margin"])
    g = data.groupby(field, dropna=False, observed=True).agg(
        count=("points", "size"),
        scored=("points", "mean"),
        allowed=("opp_points", "mean"),
        margin=("margin", "mean")
    ).reset_index()
    result = g.rename(columns={"count": "Team-game rows", "scored": "Avg scored",
                               "allowed": "Avg allowed", "margin": "Avg margin"})
    return result.round(2)


def feature_analysis_coverage(data: pd.DataFrame) -> pd.DataFrame:
    """Coverage is measured AFTER applying the configured minimum history filter."""
    count = len(data)
    rows = []
    for name, (col, _, _) in FEATURE_ASSOCIATIONS.items():
        n = int(data[col].notna().sum()) if count else 0
        rows.append({"Statistic": name, "Rows available": n,
                     "Rows reviewed": count,
                     "Coverage %": round(100 * n / count, 1) if count else 0.0})
    return pd.DataFrame(rows)



# Matchup Explorer is retrospective and descriptive. Season ratings and AP poll
# positions are end-of-season for archived years; never represent them as ratings
# known before a historical game or as forecasts for a future matchup.
ADVANCED_MATCHUP_RATINGS_SQL = """
SELECT r.team_id, t.name AS team, COALESCE(t.conference, '—') AS conference,
       t.logo_url, r.games_played AS gp,
       r.adjusted_offense AS adj_off, r.adjusted_defense AS adj_def,
       r.adjusted_net AS adj_net, r.opponent_sos AS opp_sos
FROM team_adjusted_ratings r
JOIN teams t ON t.team_id=r.team_id
WHERE r.season=%s AND t.is_d1=1
ORDER BY r.adjusted_net DESC, t.name ASC
"""

# Both team box scores must exist for a game to enter scoring/tempo trends.
# Only the selected teams' actual finished games are read, never future games.
ADVANCED_MATCHUP_GAMES_SQL = """
SELECT g.game_id, g.game_date, g.start_time_utc,
       s.team_id, o.team_id AS opponent_id,
       ot.name AS opponent_name, g.home_team_id, g.away_team_id,
       g.neutral_site, s.points, o.points AS opponent_points,
       s.possessions
FROM games g
JOIN team_game_stats s ON s.game_id=g.game_id
JOIN team_game_stats o ON o.game_id=g.game_id AND o.team_id<>s.team_id
JOIN teams ot ON ot.team_id=o.team_id
WHERE g.season=%s AND g.completed=1
  AND s.team_id IN (%s,%s)
  AND s.points IS NOT NULL AND o.points IS NOT NULL
  AND s.possessions>0 AND o.possessions>0
ORDER BY g.game_date,g.game_id,s.team_id
"""


def advanced_matchup_history(rows: pd.DataFrame, team_id: str) -> pd.DataFrame:
    """Chronological usable box scores for exactly one team; no upcoming games."""
    cols = ("game_id", "game_date", "team_id", "opponent_id", "opponent_name",
            "home_team_id", "neutral_site", "points", "opponent_points", "possessions")
    if rows.empty:
        return pd.DataFrame(columns=[*cols, "venue", "margin", "result", "full_rest_days"])
    data = rows.loc[rows["team_id"].astype(str) == str(team_id)].copy()
    if data.empty:
        return pd.DataFrame(columns=[*cols, "venue", "margin", "result", "full_rest_days"])
    for col in ("points", "opponent_points", "possessions"):
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data["game_date"] = pd.to_datetime(data["game_date"], errors="coerce")
    data = data.dropna(subset=["game_date", "points", "opponent_points", "possessions"])
    data = data[(data["points"] >= 0) & (data["opponent_points"] >= 0)
                & (data["possessions"] > 0)].copy()
    data = data.drop_duplicates(subset=["game_id", "team_id"])
    data = data.sort_values(["game_date", "game_id"], kind="stable").reset_index(drop=True)
    if data.empty:
        return data
    neutral = pd.to_numeric(data["neutral_site"], errors="coerce").fillna(0).eq(1)
    home = data["home_team_id"].astype(str).eq(str(team_id))
    data["venue"] = "Away"
    data.loc[home, "venue"] = "Home"
    data.loc[neutral, "venue"] = "Neutral"
    data["margin"] = data["points"] - data["opponent_points"]
    data["result"] = data["margin"].apply(lambda diff: "W" if diff > 0 else "L" if diff < 0 else "T")
    data["full_rest_days"] = (data["game_date"].diff().dt.days - 1).clip(lower=0)
    return data


def advanced_matchup_summary(history: pd.DataFrame, recent_n: int = 5) -> dict:
    """Summarize completed team games; no forecast or opponent-adjusted inference."""
    if recent_n < 1:
        raise ValueError("recent_n must be positive")
    if history.empty:
        return {"games": 0, "ppg": None, "allowed": None, "pace": None,
                "margin": None, "recent_ppg": None, "recent_allowed": None,
                "recent_margin": None, "recent_wins": 0, "recent_losses": 0,
                "recent_games": 0, "avg_rest": None, "last_date": None}
    recent = history.tail(recent_n)
    rest = pd.to_numeric(history["full_rest_days"], errors="coerce")
    return {
        "games": len(history),
        "ppg": float(history["points"].mean()),
        "allowed": float(history["opponent_points"].mean()),
        "pace": float(history["possessions"].mean()),
        "margin": float(history["margin"].mean()),
        "recent_ppg": float(recent["points"].mean()),
        "recent_allowed": float(recent["opponent_points"].mean()),
        "recent_margin": float(recent["margin"].mean()),
        "recent_wins": int(recent["result"].eq("W").sum()),
        "recent_losses": int(recent["result"].eq("L").sum()),
        "recent_games": len(recent),
        "avg_rest": float(rest.mean()) if rest.notna().any() else None,
        "last_date": history["game_date"].iloc[-1],
    }


def advanced_matchup_comparison_table(left: pd.Series, right: pd.Series,
                                      left_sum: dict, right_sum: dict,
                                      left_record: str, right_record: str) -> pd.DataFrame:
    """Display matching season summary metrics without claiming a winning team."""
    metric_fields = [
        ("Overall W–L (all completed games)", left_record, right_record, False),
        ("Games with usable box scores", left_sum["games"], right_sum["games"], False),
        ("Adjusted efficiency national rank", left["national_rank"], right["national_rank"], False),
        ("Adjusted offense / 100 possessions", left["adj_off"], right["adj_off"], True),
        ("Adjusted defense / 100 possessions", left["adj_def"], right["adj_def"], True),
        ("Adjusted net / 100 possessions", left["adj_net"], right["adj_net"], True),
        ("Opponent strength of schedule", left["opp_sos"], right["opp_sos"], True),
        ("Points scored per game", left_sum["ppg"], right_sum["ppg"], True),
        ("Points allowed per game", left_sum["allowed"], right_sum["allowed"], True),
        ("Possessions per game (estimated)", left_sum["pace"], right_sum["pace"], True),
        ("Average scoring margin", left_sum["margin"], right_sum["margin"], True),
        ("Recent points per game", left_sum["recent_ppg"], right_sum["recent_ppg"], True),
        ("Recent points allowed per game", left_sum["recent_allowed"], right_sum["recent_allowed"], True),
        ("Recent average scoring margin", left_sum["recent_margin"], right_sum["recent_margin"], True),
        ("Recent W–L (usable games)",
         f'{left_sum["recent_wins"]}–{left_sum["recent_losses"]}',
         f'{right_sum["recent_wins"]}–{right_sum["recent_losses"]}', False),
        ("Avg full days off between games", left_sum["avg_rest"], right_sum["avg_rest"], True),
    ]
    def show(value, numeric):
        if value is None or pd.isna(value):
            return "—"
        return f"{float(value):.1f}" if numeric else str(value)
    return pd.DataFrame([{"Statistic": label,
                          "Team A": show(a, rounded), "Team B": show(b, rounded)}
                         for label, a, b, rounded in metric_fields])


def advanced_matchup_recent_table(history: pd.DataFrame, n: int) -> pd.DataFrame:
    """Render a team's latest completed games, never upcoming schedule rows."""
    if history.empty:
        return pd.DataFrame(columns=["Date", "Opponent", "Venue", "Result", "Score", "Possessions"])
    recent = history.tail(n).iloc[::-1]
    return pd.DataFrame({
        "Date": recent["game_date"].dt.strftime("%Y-%m-%d").to_numpy(),
        "Opponent": recent["opponent_name"].to_numpy(),
        "Venue": recent["venue"].to_numpy(),
        "Result": recent["result"].to_numpy(),
        "Score": [f"{int(p)}–{int(o)}" for p, o in zip(recent["points"], recent["opponent_points"])],
        "Possessions": recent["possessions"].round(1).to_numpy(),
    }).reset_index(drop=True)

st.title("🏀 NCAA Men's Basketball Database")
st.caption("Division I • ESPN game statistics • opponent-adjusted SOS & historical data • Central Time")
with st.sidebar:
    st.header("Navigation")
    page = st.radio("Section", ["Game schedules", "Teams & conferences", "Team efficiency", "Team explorer",
                                    "Player season averages", "Adjusted SOS", "Historical team trends", "Historical Team Comparison", "Historical Data Quality", "Historical Scoring Review", "Statistical Feature Analysis", "Advanced Matchup Comparison", "Injuries", "Update status"], label_visibility="collapsed")
    if st.button("Refresh displayed data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

try:
    seasons = query("SELECT DISTINCT season FROM games WHERE season IS NOT NULL ORDER BY season DESC")
    if seasons.empty:
        st.info("No game records yet. After deployment, run the GitHub Actions importer once to populate your database.")
        st.stop()
    season_values = seasons["season"].astype(int).tolist()
    season = st.selectbox("Season (ESPN ending year)", season_values)
    latest_season = season_values[0]
except Exception as exc:
    # Log only a classified error category, not the raw exception: database
    # exceptions can occasionally contain credential-bearing connection details.
    detail = str(exc).lower()
    if "password authentication failed" in detail or "authentication failed" in detail:
        reason = "Supabase rejected the database username or password"
    elif "could not translate host name" in detail or "failed to resolve host" in detail:
        reason = "The database hostname in DATABASE_URL is malformed or unreachable"
    elif "timeout" in detail or "timed out" in detail:
        reason = "The database connection timed out"
    elif "connection refused" in detail:
        reason = "The database server refused the connection"
    elif "permission denied" in detail:
        reason = "Database user lacks permission to read the required table"
    elif "relation" in detail and "does not exist" in detail:
        reason = "A required database table does not exist"
    elif "database_url" in detail or "connection string" in detail:
        reason = "DATABASE_URL is missing or malformed"
    elif "invalid" in detail and ("port" in detail or "dsn" in detail):
        reason = "The PostgreSQL connection URL is malformed"
    else:
        reason = "An unexpected error occurred while reading the database"
    logging.error("NCAA dashboard database read failed. Error type: %s. Reason: %s.",
                  type(exc).__name__, reason)
    st.error("Cannot read cloud database yet.")
    st.warning(f"**Database diagnostic:** {reason}.")
    st.caption("This diagnostic does not display your password, database URL, or other credentials. "
               "If you change Streamlit secrets, save them and reboot the app.")
    st.stop()

if page == "Game schedules":
    st.subheader("Game schedules")
    try:
        efficiency_ranks, ap_ranks, _ = rank_maps(season, latest_season)
    except Exception:
        efficiency_ranks, ap_ranks = {}, {}
        st.warning("Efficiency ranks could not be loaded; schedules are still available.")
    st.caption("Teams display AP rank → team logo → name (latest ESPN AP Top 25, not the historical "
               "game-day rank). Eff. is this database's unadjusted net-efficiency rank.")
    conference_images = current_conference_logos()
    if not conference_images:
        st.caption("Conference artwork is temporarily unavailable from ESPN; showing names instead.")
    if season == latest_season and not ap_ranks:
        st.caption("Current AP poll rankings are temporarily unavailable from ESPN.")
    st.caption("Times shown in America/Chicago. ESPN data is saved by your GitHub importer, not streamed live. "
               "Use Refresh displayed data after an import finishes.")
    today = datetime.now(CENTRAL).date()
    today_str = today.isoformat()
    today_tab, upcoming_tab, browse_tab = st.tabs(["🏀 Today's games", "📅 Future schedule", "🔎 Browse dates"])

    with today_tab:
        st.markdown(f"**{today.strftime('%A, %B')} {today.day}, {today.year}**")
        today_games = load_schedule(season, today_str, today_str)
        st.metric("Games on today's schedule", len(today_games))
        if today_games.empty:
            st.info("No games for today in the database yet. If the importer hasn't run recently, "
                    "use GitHub Actions → Run workflow with both date fields blank to load recent "
                    "results and the next 14 days of schedules.")
        else:
            show_schedule(today_games, efficiency_ranks, ap_ranks, conference_images)

    with upcoming_tab:
        window = st.selectbox("Look ahead", [7, 14, 30], index=1,
                              format_func=lambda days: f"Next {days} days")
        start = today + timedelta(days=1)
        end = today + timedelta(days=window)
        upcoming = load_schedule(season, start.isoformat(), end.isoformat())
        if not upcoming.empty:
            upcoming = upcoming.loc[upcoming["completed"].eq(0)].copy()
        st.metric("Upcoming games in the database", len(upcoming))
        if upcoming.empty:
            st.info("No future games found for this range. Run the nightly ESPN updater "
                    "manually with both date fields blank to import the next 14 days. "
                    "Dates farther out may not have been imported yet.")
        else:
            for date_value, day_games in upcoming.groupby("game_date", sort=True):
                date_label = datetime.strptime(date_value, "%Y-%m-%d").strftime("%A, %B %d")
                st.markdown(f"**{date_label} · {len(day_games)} games**")
                show_schedule(day_games, efficiency_ranks, ap_ranks, conference_images)

    with browse_tab:
        dates = query("SELECT DISTINCT game_date FROM games WHERE season=%s "
                      "ORDER BY game_date DESC", (season,))
        if dates.empty:
            st.info("No games imported yet.")
        else:
            date_string = st.selectbox("Choose a date (Central Time)", dates["game_date"].tolist())
            games_on_date = load_schedule(season, date_string, date_string)
            st.caption(f"{len(games_on_date)} games")
            show_schedule(games_on_date, efficiency_ranks, ap_ranks, conference_images)

elif page == "Teams & conferences":
    st.subheader("Division I teams & conferences")
    teams = query("""SELECT team_id,name,abbreviation,conference,logo_url
                     FROM teams WHERE is_d1=1 ORDER BY name""")
    if teams.empty:
        st.info("No Division I teams have been imported yet.")
    else:
        try:
            efficiency_ranks, ap_ranks, _ = rank_maps(season, latest_season)
        except Exception:
            efficiency_ranks, ap_ranks = {}, {}
            st.warning("Efficiency ranks are temporarily unavailable.")
        records = season_records(season)
        positions = espn_conference_positions(season)
        teams["Conference"] = teams["conference"].fillna("—").replace("", "—")
        search_col, conference_col = st.columns([2, 1])
        with search_col:
            search = st.text_input("Search teams", placeholder="e.g. Iowa")
        with conference_col:
            conferences = ["All conferences"] + sorted(teams["Conference"].unique().tolist())
            conference = st.selectbox("Conference", conferences)
        if search:
            teams = teams[teams["name"].str.contains(search, case=False, regex=False, na=False)]
        if conference != "All conferences":
            teams = teams[teams["Conference"] == conference]
        st.caption(f"{len(teams)} Division I teams shown · AP rank → team logo → name → "
                   "overall record (from imported final scores).")
        if positions:
            st.caption("Sorted by conference, then ESPN's conference standings rank "
                       "(when available). Teams without ESPN positions appear afterward "
                       "in overall-record order.")
        else:
            st.warning("ESPN conference standings aren't available right now. "
                       "Teams are grouped by conference and ordered by imported overall "
                       "win percentage; this is NOT an official conference rank.")
        conference_images = current_conference_logos()
        if not conference_images:
            st.caption("Conference logos are temporarily unavailable from ESPN; showing names instead.")
        teams = sort_conference_teams(teams, positions, records)
        rows = []
        for _, team in teams.iterrows():
            official = team['official_rank']
            rows.append([
                safe_cell(int(official)) if pd.notna(official) else '—',
                team_badge(team["name"], team["team_id"], team["logo_url"],
                           ap_ranks, records),
                conference_badge(team["Conference"], conference_images),
                safe_cell(rank_label(team["team_id"], efficiency_ranks)),
            ])
        show_visual_table(["Conf. rank", "Team", "Conference", "Eff. rank"], rows)

elif page == "Team efficiency":
    st.subheader("Team efficiency rankings")
    st.caption("Eff. rank = unadjusted net efficiency from imported games. "
               "Team cells display AP rank → logo → name, when available.")
    try:
        efficiency_ranks, ap_ranks, ranks = rank_maps(season, latest_season)
    except Exception:
        st.error("Couldn't load rankings. Check the hosted app logs.")
        st.stop()
    if ranks.empty:
        st.info("No complete box scores imported for this season. Try running a historical backfill.")
    else:
        ranks.insert(0, "Eff. Rank", range(1, len(ranks)+1))
        ranks.insert(1, "Logo", ranks["logo_url"].map(logo_url))
        renames = {"name":"Team", "conference":"Conference", "games":"GP",
                   "wins":"W", "losses":"L", "off_eff":"Off / 100", "def_eff":"Def / 100",
                   "net_eff":"Net / 100", "efg_pct":"eFG %", "tov_pct":"TO %"}
        display = ranks.rename(columns=renames)
        display["Team"] = [ranked_team_name(name, ident, ap_ranks)
                           for name, ident in zip(ranks["name"], ranks["team_id"])]
        conference_names = display["Conference"].fillna("—").replace("", "—")
        conference_images = current_conference_logos()
        display["Conference"] = (conference_names.map(lambda n: conference_logo(n, conference_images))
                                 if conference_images else conference_names)
        if not conference_images:
            st.caption("Conference logos are temporarily unavailable from ESPN; showing names instead.")
        cols = ["Eff. Rank", "Logo", "Team", "Conference", "GP", "W", "L",
                "Off / 100", "Def / 100", "Net / 100", "eFG %", "TO %"]
        rows = []
        for i, team in ranks.iterrows():
            rows.append([
                safe_cell(team["Eff. Rank"]),
                team_badge(team["name"], team["team_id"], team["logo_url"], ap_ranks),
                conference_badge(team["conference"], conference_images),
                safe_cell(team["games"]),
                safe_cell(team["wins"]),
                safe_cell(team["losses"]),
                safe_cell(team["off_eff"]),
                safe_cell(team["def_eff"]),
                safe_cell(team["net_eff"]),
                safe_cell(team["efg_pct"]),
                safe_cell(team["tov_pct"]),
            ])
        show_visual_table(["Eff. Rank", "Team", "Conference", "GP", "W", "L",
                           "Off / 100", "Def / 100", "Net / 100", "eFG %", "TO %"], rows)
        # Keep conference names (not image URLs) in the downloadable CSV.
        csv_display = display[cols].copy()
        csv_display["Conference"] = conference_names
        st.download_button("Download rankings (CSV)", csv_display.to_csv(index=False),
                           "rankings.csv", "text/csv")

elif page == "Adjusted SOS":
    st.subheader("Opponent-adjusted strength of schedule")
    st.caption("Calculated from completed ESPN team box scores. Opp. SOS is the average "
               "opponent-adjusted NET efficiency of teams actually faced. Higher values "
               "mean a tougher schedule; it is not an official NCAA ranking. Adjusted "
               "offense/defense and SOS are measured in points per 100 possessions. "
               "Small-sample ratings are shrunk toward the season average.")
    available = query("SELECT to_regclass('public.team_adjusted_ratings') "
                      "IS NOT NULL AS available")
    if available.empty or not bool(available.iloc[0]["available"]):
        st.info("The analytics tables haven't been created yet. In GitHub Actions, "
                "run 'Update NCAA MBB Adjusted SOS and History' once.")
    else:
        sos = query("""
            SELECT r.team_id, t.name AS team, COALESCE(t.conference,'—') AS conference,
                   r.games_played AS gp,
                   ROUND(r.adjusted_offense::numeric,1)::double precision AS adj_off,
                   ROUND(r.adjusted_defense::numeric,1)::double precision AS adj_def,
                   ROUND(r.adjusted_net::numeric,1)::double precision AS adj_net,
                   ROUND(r.raw_net::numeric,1)::double precision AS raw_net,
                   ROUND(r.opponent_sos::numeric,1)::double precision AS opp_sos
            FROM team_adjusted_ratings r
            JOIN teams t ON t.team_id = r.team_id
            WHERE r.season=%s AND t.is_d1=1
            ORDER BY r.opponent_sos DESC, t.name ASC
        """, (season,))
        if sos.empty:
            st.info("No completed games with two full team box scores in this season yet. "
                    "Run a historical ESPN import and refresh the analytics workflow.")
        else:
            c1, c2, c3 = st.columns([2, 2, 1])
            with c1:
                name_filter = st.text_input("Search team", key="sos_team")
            with c2:
                conference_filter = st.selectbox("Conference",
                    ["All conferences"] + sorted(sos["conference"].unique().tolist()),
                    key="sos_conference")
            with c3:
                min_games = st.number_input("Minimum games", min_value=1, value=1,
                                            max_value=max(1, int(sos["gp"].max())),
                                            key="sos_min_gp")
            if name_filter:
                sos = sos[sos["team"].str.contains(name_filter, case=False,
                                                   na=False, regex=False)]
            if conference_filter != "All conferences":
                sos = sos[sos["conference"] == conference_filter]
            sos = sos[sos["gp"] >= min_games]
            sos = sos.reset_index(drop=True)
            sos.insert(0, "SOS rank", sos.index + 1)
            display = sos.rename(columns={"team":"Team", "conference":"Conference",
                "gp":"GP", "adj_off":"Adj Off / 100", "adj_def":"Adj Def / 100",
                "adj_net":"Adj Net / 100", "raw_net":"Raw Net / 100",
                "opp_sos":"Opponent SOS"}).drop(columns=["team_id"])
            st.caption("SOS rank is the current order of displayed results, not an "
                       "official conference or national ranking. Unimported opponents "
                       "and missing box scores can bias these figures.")
            show_table(display)
            st.download_button("Download adjusted ratings (CSV)",
                               display.to_csv(index=False),
                               f"adjusted_sos_{season}.csv", "text/csv")

elif page == "Team explorer":
    st.subheader("Team explorer")
    teams = query("SELECT team_id,name,conference,logo_url FROM teams WHERE is_d1=1 ORDER BY name")
    if teams.empty:
        st.info("The Division I team list hasn't been imported yet.")
    else:
        team_name = st.selectbox("Choose a team", teams["name"].tolist())
        selected_team = teams.loc[teams["name"] == team_name].iloc[0]
        team_id = str(selected_team["team_id"])
        records = season_records(season)
        try:
            efficiency_ranks, ap_ranks, _ = rank_maps(season, latest_season)
        except Exception:
            efficiency_ranks, ap_ranks = {}, {}
        conference_name = (selected_team["conference"]
                           if pd.notna(selected_team["conference"]) else "—")
        st.markdown(
            '<div class="mbb-team-card">' +
            team_badge(selected_team["name"], team_id, selected_team["logo_url"],
                       ap_ranks, records) + '</div>', unsafe_allow_html=True)
        st.caption(f"Conference: {conference_name} · "
                   f"Season efficiency rank: {rank_label(team_id, efficiency_ranks)} · "
                   "Records are based on imported final scores.")
        # Read-only opponent adjustment; separate analytics workflow owns this table.
        if query("SELECT to_regclass('public.team_adjusted_ratings') "
                 "IS NOT NULL AS available").iloc[0]["available"]:
            current_sos = query("""
                SELECT games_played, adjusted_net, opponent_sos
                FROM team_adjusted_ratings WHERE season=%s AND team_id=%s
            """, (season, team_id))
            if not current_sos.empty:
                ar = current_sos.iloc[0]
                st.caption("Opponent-adjusted net efficiency: "
                           f"{ar['adjusted_net']:+.1f} per 100 possessions · "
                           f"Opponent SOS: {ar['opponent_sos']:+.1f} · "
                           f"Based on {int(ar['games_played'])} complete box scores. "
                           "See Adjusted SOS for methodology.")
        st.markdown("#### Team season averages")
        team_averages = query(TEAM_SEASON_AVERAGES_QUERY, (team_id, season))
        if team_averages.empty or int(team_averages.iloc[0]["gp"]) == 0:
            st.info("No completed team box scores have been imported for this season yet.")
        else:
            st.caption("Per-game averages from completed ESPN team box scores. "
                       "GP may differ from the overall W–L record if a game is missing "
                       "its box score. Shooting percentages use season totals of "
                       "made shots divided by attempted shots.")
            season_summary = team_averages.iloc[0]
            fields = [
                ("GP", "gp"), ("PPG", "ppg"), ("Opp PPG", "opp_ppg"),
                ("RPG", "rpg"), ("APG", "apg"), ("SPG", "spg"),
                ("BPG", "bpg"), ("TO/G", "topg"),
                ("FG%", "fg_pct"), ("3P%", "three_pct"), ("FT%", "ft_pct"),
            ]
            show_visual_table([title for title, _ in fields],
                              [[safe_cell(season_summary[field])
                                for _, field in fields]])

        st.markdown("#### Next 5 games")
        # Use this season's imported future schedule; never fabricate games.
        # The nightly ESPN update normally imports 14 days ahead.
        upcoming_games = query("""
            SELECT g.game_id, g.game_date, g.start_time_utc,
                   g.home_team_id, g.away_team_id, g.venue,
                   CASE WHEN g.home_team_id=%s THEN a.team_id ELSE h.team_id END AS opponent_id,
                   CASE WHEN g.home_team_id=%s THEN a.name ELSE h.name END AS opponent,
                   CASE WHEN g.home_team_id=%s THEN a.logo_url ELSE h.logo_url END AS opponent_logo
            FROM games g
            JOIN teams h ON h.team_id=g.home_team_id
            JOIN teams a ON a.team_id=g.away_team_id
            WHERE (g.home_team_id=%s OR g.away_team_id=%s)
              AND g.season=%s AND g.completed=0 AND g.game_date >= %s
            ORDER BY g.game_date ASC, g.start_time_utc ASC NULLS LAST, g.game_id ASC
        """, (team_id, team_id, team_id, team_id, team_id, season,
              datetime.now(CENTRAL).date().isoformat()))
        upcoming_games = select_next_five_games(upcoming_games, datetime.now(timezone.utc))
        if upcoming_games.empty:
            st.info("No upcoming games have been imported for this team. "
                    "The nightly ESPN update normally imports the next 14 days. "
                    "Try running the GitHub Actions updater with both date fields blank.")
        else:
            st.caption("Upcoming games from the imported ESPN schedule. "
                       "Opponent W–L is their current overall record; start times are Central.")
            upcoming_rows = []
            for _, game in upcoming_games.iterrows():
                is_home = str(game["home_team_id"]) == team_id
                upcoming_rows.append([
                    safe_cell(game["game_date"]),
                    safe_cell(central_tipoff(game["start_time_utc"])),
                    "vs" if is_home else "@",
                    team_badge(game["opponent"], game["opponent_id"],
                               game["opponent_logo"], ap_ranks, records),
                    safe_cell(game["venue"]),
                ])
            show_visual_table(["Date", "Time (CT)", "Home/Away", "Opponent", "Venue"],
                              upcoming_rows)
            if len(upcoming_games) < 5:
                st.caption(f"Only {len(upcoming_games)} future game(s) currently in the database. "
                           "Additional games will appear as the ESPN updater imports them.")

        st.markdown("#### Previous games")
        games = query("""
            SELECT g.game_id, g.game_date, g.home_team_id, g.away_team_id,
                   g.home_score, g.away_score,
                   CASE WHEN g.home_team_id=%s THEN a.team_id ELSE h.team_id END AS opponent_id,
                   CASE WHEN g.home_team_id=%s THEN a.name ELSE h.name END AS opponent,
                   CASE WHEN g.home_team_id=%s THEN a.logo_url ELSE h.logo_url END AS opponent_logo,
                   s.offensive_efficiency,s.defensive_efficiency,s.net_efficiency
            FROM games g
            JOIN teams h ON h.team_id=g.home_team_id
            JOIN teams a ON a.team_id=g.away_team_id
            LEFT JOIN team_game_stats s ON s.game_id=g.game_id AND s.team_id=%s
            WHERE (g.home_team_id=%s OR g.away_team_id=%s)
              AND g.season=%s AND g.completed=1
              AND g.home_score IS NOT NULL AND g.away_score IS NOT NULL
            ORDER BY g.game_date DESC, g.game_id DESC
        """, (team_id, team_id, team_id, team_id, team_id, team_id, season))
        if games.empty:
            st.info("No completed games with final scores for this team yet.")
        else:
            st.caption("Past games: Result is shown from the selected team's viewpoint "
                       "(W/L, team score–opponent score). Opponent W–L is their current "
                       "overall record from imported games, not their record at tipoff.")
            rows = []
            for _, game in games.iterrows():
                rows.append([
                    safe_cell(game['game_date']),
                    team_badge(game['opponent'], game['opponent_id'],
                               game['opponent_logo'], ap_ranks, records),
                    matchup_result(game, team_id),
                    safe_cell(round(game['offensive_efficiency'], 1))
                        if pd.notna(game['offensive_efficiency']) else '—',
                    safe_cell(round(game['defensive_efficiency'], 1))
                        if pd.notna(game['defensive_efficiency']) else '—',
                    safe_cell(round(game['net_efficiency'], 1))
                        if pd.notna(game['net_efficiency']) else '—',
                ])
            show_visual_table(["Date", "Opponent", "Result", "Off / 100",
                               "Def / 100", "Net / 100"], rows)
            graph = (games.sort_values('game_date')
                     .set_index('game_date')[['offensive_efficiency',
                                              'defensive_efficiency']].dropna(how='all'))
            if not graph.empty:
                graph.columns = ['Offensive efficiency', 'Defensive efficiency']
                st.line_chart(graph)

elif page == "Historical Team Comparison":
    st.subheader("Historical Team Comparison")
    st.caption("Compare adjusted Division I team ratings using the five-season ESPN archive. "
               "Choose the basketball season with the Season selector above. "
               "All metrics come from completed games with usable ESPN box scores; "
               "they are descriptive, not forecasts or official NCAA NET rankings.")
    tables_exist = query("""
        SELECT to_regclass('public.team_adjusted_ratings') IS NOT NULL AS ratings,
               to_regclass('public.historical_team_snapshots') IS NOT NULL AS snapshots
    """)
    if (tables_exist.empty or not bool(tables_exist.iloc[0]["ratings"])
            or not bool(tables_exist.iloc[0]["snapshots"])):
        st.info("Historical ratings are not available yet. Run the existing "
                "'Update NCAA MBB Adjusted SOS and History' workflow in GitHub Actions.")
    else:
        full_ratings = query("""
            SELECT r.team_id, t.name AS team, t.logo_url,
                   COALESCE(t.conference, '—') AS conference,
                   r.games_played AS gp,
                   r.adjusted_offense AS adj_off,
                   r.adjusted_defense AS adj_def,
                   r.adjusted_net AS adj_net,
                   r.opponent_sos AS opp_sos
            FROM team_adjusted_ratings r
            JOIN teams t ON t.team_id=r.team_id
            WHERE r.season=%s AND t.is_d1=1
            ORDER BY r.adjusted_net DESC, t.name ASC
        """, (season,))
        ranked = historical_rankings(full_ratings)
        if ranked.empty:
            st.info(f"No adjusted ratings are available for the "
                    f"{historical_season_label(season)} season. Select an imported "
                    "historical season above, or rerun the analytics workflow.")
        else:
            ap_for_comparison, ap_poll_name = comparison_ap_poll(season, latest_season)
            st.caption(f"**Selected season:** {historical_season_label(season)} · "
                       f"{len(ranked):,} Division I teams rated")
            if ap_for_comparison:
                st.caption(f"Team names display AP rank → ESPN logo → name. "
                           f"AP source: {ap_poll_name}. 'NR' means outside the AP Top 25. "
                           "The AP poll is separate from our adjusted-efficiency ranking; "
                           "historical AP positions are NOT game-day rankings.")
            else:
                st.caption("ESPN AP rankings for this selected season are unavailable. "
                           "Team logos and names remain visible; adjusted-efficiency "
                           "rankings still work normally.")
            ranking_tab, matchup_tab, trends_tab, years_tab = st.tabs([
                "🏆 Top 25 rankings", "⚖️ Compare two teams",
                "📈 Performance trends", "🗓️ Compare seasons"
            ])
            with ranking_tab:
                st.markdown("#### Top 25 by adjusted net efficiency")
                st.caption("Higher adjusted net is better. Adjusted defense is better "
                           "when lower; SOS is higher for tougher opponents. "
                           "National rank is computed across every rated D1 team, "
                           "before displaying the top 25.")
                top25 = ranked.head(25).rename(columns={
                    "national_rank": "Rank", "team": "Team",
                    "conference": "Conference", "gp": "GP",
                    "adj_off": "Adj Off / 100", "adj_def": "Adj Def / 100",
                    "adj_net": "Adj Net / 100", "opp_sos": "Opponent SOS"
                })
                top25 = top25[["Rank", "Team", "Conference", "GP", "Adj Off / 100",
                               "Adj Def / 100", "Adj Net / 100", "Opponent SOS"]].copy()
                for col in ["Adj Off / 100", "Adj Def / 100", "Adj Net / 100", "Opponent SOS"]:
                    top25[col] = pd.to_numeric(top25[col], errors="coerce").round(1)
                top25_html = []
                for _, team_row in ranked.head(25).iterrows():
                    top25_html.append([
                        safe_cell(int(team_row["national_rank"])),
                        team_badge(team_row["team"], team_row["team_id"],
                                   team_row["logo_url"], ap_for_comparison),
                        safe_cell(team_row["conference"]),
                        safe_cell(int(team_row["gp"])),
                        historical_number(team_row["adj_off"]),
                        historical_number(team_row["adj_def"]),
                        historical_number(team_row["adj_net"], signed=True),
                        historical_number(team_row["opp_sos"], signed=True),
                    ])
                show_visual_table(["Adj rank", "Team (AP rank · logo · name)",
                                   "Conference", "GP", "Adj Off / 100", "Adj Def / 100",
                                   "Adj Net / 100", "Opponent SOS"], top25_html)
                st.caption("Adj rank is computed from adjusted net efficiency; "
                           "AP # beside a logo is from the Associated Press poll.")
                st.download_button("Download Top 25 (CSV)",
                                   top25.to_csv(index=False),
                                   f"ncaa_top25_adjusted_{season}.csv", "text/csv",
                                   key="historical_top25_download")

            # Both selectors are always constructed, so the following tabs
            # reuse the same two teams. Team IDs (not names) identify rows.
            selection = ranked[["team_id", "team", "conference", "logo_url"]].copy()
            selection["team_id"] = selection["team_id"].astype(str)
            choice_ids = selection["team_id"].tolist()
            choice_labels = {
                row["team_id"]: comparison_team_name(
                    row["team"], row["conference"], row["team_id"],
                    ap_for_comparison, bool(ap_for_comparison))
                for _, row in selection.iterrows()
            }
            with matchup_tab:
                st.markdown("#### Side-by-side team ratings")
                left, right = st.columns(2)
                with left:
                    first_id = st.selectbox("Team A", choice_ids, index=0,
                        format_func=lambda tid: choice_labels[tid],
                        key="historical_compare_first")
                    first_display = ranked.loc[
                        ranked["team_id"].astype(str) == first_id].iloc[0]
                    st.markdown('<div class="mbb-team-card">' + team_badge(
                        first_display["team"], first_id, first_display["logo_url"],
                        ap_for_comparison) + '</div>', unsafe_allow_html=True)
                with right:
                    second_id = st.selectbox("Team B", choice_ids,
                        index=min(1, len(choice_ids) - 1),
                        format_func=lambda tid: choice_labels[tid],
                        key="historical_compare_second")
                    second_display = ranked.loc[
                        ranked["team_id"].astype(str) == second_id].iloc[0]
                    st.markdown('<div class="mbb-team-card">' + team_badge(
                        second_display["team"], second_id, second_display["logo_url"],
                        ap_for_comparison) + '</div>', unsafe_allow_html=True)
                if first_id == second_id:
                    st.info("Select two different teams for a side-by-side comparison. "
                            "The trends tab will display the selected team.")
                compare_rows = ranked[ranked["team_id"].astype(str).isin(
                    [first_id, second_id])]
                by_id = {str(item["team_id"]): item
                         for _, item in compare_rows.iterrows()}
                # Pace is derived from the same completed, two-box-score games
                # used by the analytics builder, not from all scheduled games.
                pace_rows = query("""
                    SELECT s.team_id, COUNT(*)::integer AS tempo_games,
                           AVG(s.possessions)::double precision AS pace
                    FROM team_game_stats s
                    JOIN games g ON g.game_id=s.game_id
                    JOIN team_game_stats o ON o.game_id=s.game_id
                         AND o.team_id<>s.team_id
                    WHERE g.season=%s AND g.completed=1
                      AND g.home_score IS NOT NULL AND g.away_score IS NOT NULL
                      AND s.team_id IN (%s,%s)
                      AND s.possessions>0 AND o.possessions>0
                      AND s.points IS NOT NULL AND o.points IS NOT NULL
                    GROUP BY s.team_id
                """, (season, first_id, second_id))
                pace_map = {str(item["team_id"]): item["pace"]
                            for _, item in pace_rows.iterrows()}
                compare_fields = [
                    ("Adjusted offense / 100", "adj_off", False),
                    ("Adjusted defense / 100 (lower is better)", "adj_def", False),
                    ("Adjusted net / 100", "adj_net", True),
                    ("Opponent SOS / 100", "opp_sos", True),
                ]
                comparison = [[
                    "Team",
                    team_badge(by_id[first_id]["team"], first_id,
                               by_id[first_id]["logo_url"], ap_for_comparison),
                    team_badge(by_id[second_id]["team"], second_id,
                               by_id[second_id]["logo_url"], ap_for_comparison),
                ]]
                for label, col, signed in compare_fields:
                    comparison.append([
                        safe_cell(label),
                        historical_number(by_id[first_id][col], signed),
                        historical_number(by_id[second_id][col], signed),
                    ])
                for label, getter in [
                    ("Games in ratings", lambda tid: str(int(by_id[tid]["gp"]))),
                    ("Est. possessions / game", lambda tid: historical_number(
                        pace_map.get(tid)))
                ]:
                    comparison.append([safe_cell(label), getter(first_id),
                                       getter(second_id)])
                show_visual_table(["Statistic", "Team A", "Team B"], comparison)
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(f"{by_id[first_id]['team']} national rank",
                              f"#{int(by_id[first_id]['national_rank'])}")
                with col2:
                    st.metric(f"{by_id[second_id]['team']} national rank",
                              f"#{int(by_id[second_id]['national_rank'])}")
                st.caption("Tempo is estimated possessions per game for imported "
                           "completed games with box scores for both teams; it may "
                           "differ from the rating's GP if any game is incomplete.")

            with trends_tab:
                st.markdown("#### Ratings across the selected season")
                st.caption("Points are saved only on days when the team played. "
                           "Each snapshot is an END-OF-DAY value, including that "
                           "day's games—not a pregame rating.")
                metrics = {
                    "Adjusted net efficiency": "adjusted_net",
                    "Adjusted offense": "adjusted_offense",
                    "Adjusted defense": "adjusted_defense",
                    "Opponent strength of schedule": "opponent_sos",
                }
                trend_choice = st.selectbox("Show metric", list(metrics),
                                            key="historical_comparison_metric")
                trend_rows = query("""
                    SELECT snapshot_date, team_id, games_played,
                           adjusted_offense, adjusted_defense,
                           adjusted_net, opponent_sos
                    FROM historical_team_snapshots
                    WHERE season=%s AND team_id IN (%s,%s)
                    ORDER BY snapshot_date, team_id
                """, (season, first_id, second_id))
                trend_names = {first_id: choice_labels[first_id],
                               second_id: choice_labels[second_id]}
                history_line = historical_trend_chart(
                    trend_rows, metrics[trend_choice], trend_names)
                if history_line.empty:
                    st.info("No daily team-rating snapshots are stored for the "
                            "selected teams and season.")
                else:
                    st.line_chart(history_line)
                    downloadable = trend_rows.copy().rename(columns={
                        "snapshot_date": "Date", "team_id": "Team ID",
                        "games_played": "GP", "adjusted_offense": "Adj offense",
                        "adjusted_defense": "Adj defense", "adjusted_net": "Adj net",
                        "opponent_sos": "Opponent SOS"})
                    show_table(downloadable)
                    st.download_button("Download both teams' snapshots (CSV)",
                                       downloadable.to_csv(index=False),
                                       f"ncaa_team_trends_{season}.csv", "text/csv",
                                       key="historical_compare_trend_download")

            with years_tab:
                st.markdown("#### One team's ratings across seasons")
                all_teams = query("""
                    SELECT DISTINCT r.team_id, t.name AS team
                    FROM team_adjusted_ratings r
                    JOIN teams t ON t.team_id=r.team_id
                    WHERE t.is_d1=1
                    ORDER BY t.name, r.team_id
                """)
                if all_teams.empty:
                    st.info("No historical season ratings have been calculated yet.")
                else:
                    team_options = all_teams["team_id"].astype(str).tolist()
                    team_labels = dict(zip(team_options, all_teams["team"].tolist()))
                    index = team_options.index(first_id) if first_id in team_options else 0
                    year_team_id = st.selectbox("Team for year-over-year comparison",
                        team_options, index=index,
                        format_func=lambda tid: team_labels[tid],
                        key="historical_across_years_team")
                    year_rows = query("""
                        SELECT season, games_played AS gp,
                               adjusted_offense AS adj_off,
                               adjusted_defense AS adj_def,
                               adjusted_net AS adj_net,
                               opponent_sos AS opp_sos
                        FROM team_adjusted_ratings
                        WHERE team_id=%s
                        ORDER BY season
                    """, (year_team_id,))
                    if year_rows.empty:
                        st.info("No season ratings were found for this team.")
                    else:
                        display_years = year_rows.copy()
                        display_years["Season"] = display_years["season"].map(
                            historical_season_label)
                        display_years = display_years.rename(columns={
                            "gp": "GP", "adj_off": "Adj offense",
                            "adj_def": "Adj defense", "adj_net": "Adj net",
                            "opp_sos": "Opponent SOS"})
                        display_years = display_years[["Season", "GP", "Adj offense",
                            "Adj defense", "Adj net", "Opponent SOS"]]
                        for metric_col in ["Adj offense", "Adj defense", "Adj net",
                                           "Opponent SOS"]:
                            display_years[metric_col] = pd.to_numeric(
                                display_years[metric_col], errors="coerce").round(1)
                        show_table(display_years)
                        st.markdown("**Adjusted net efficiency by season**")
                        net_line = display_years.set_index("Season")[["Adj net"]]
                        st.line_chart(net_line)
                        st.markdown("**Opponent strength of schedule by season**")
                        sos_line = display_years.set_index("Season")[["Opponent SOS"]]
                        st.line_chart(sos_line)
                        st.download_button("Download season comparison (CSV)",
                                           display_years.to_csv(index=False),
                                           f"ncaa_years_{year_team_id}.csv", "text/csv",
                                           key="historical_years_download")
                        if len(display_years) < 2:
                            st.caption("Only one season of analytics is available "
                                       "for this team.")
                st.caption("Conference labels in other tables reflect the teams' "
                           "current imported conference metadata, not necessarily "
                           "historical conference membership. Ratings include "
                           "only games already imported with usable box scores.")

elif page == "Historical team trends":
    st.subheader("Historical team efficiency & schedule trends")
    st.caption("Date-by-date records of each team's opponent-adjusted efficiency "
               "and strength of schedule, updated after games on the displayed "
               "date. These are descriptive historical snapshots; they are not "
               "game forecasts or an automated prediction/backtesting model.")
    available = query("SELECT to_regclass('public.historical_team_snapshots') "
                      "IS NOT NULL AS available")
    if available.empty or not bool(available.iloc[0]["available"]):
        st.info("The historical trend table hasn't been created yet. In GitHub "
                "Actions, run 'Update NCAA MBB Adjusted SOS and History' once.")
    else:
        team_choices = query("""
            SELECT DISTINCT t.team_id, t.name
            FROM historical_team_snapshots s
            JOIN teams t ON t.team_id=s.team_id
            WHERE s.season=%s AND t.is_d1=1
            ORDER BY t.name
        """, (season,))
        if team_choices.empty:
            st.info("No historical snapshots have been calculated for this "
                    "season. Import historical completed games, then refresh the "
                    "analytics workflow.")
        else:
            chosen = st.selectbox("Select a team", team_choices["name"].tolist(),
                                  key="historical_trend_team")
            chosen_id = str(team_choices.loc[
                team_choices["name"] == chosen, "team_id"].iloc[0])
            snapshots = query("""
                SELECT snapshot_date, games_played,
                       ROUND(adjusted_offense::numeric,1)::double precision AS adj_off,
                       ROUND(adjusted_defense::numeric,1)::double precision AS adj_def,
                       ROUND(adjusted_net::numeric,1)::double precision AS adj_net,
                       ROUND(opponent_sos::numeric,1)::double precision AS opponent_sos
                FROM historical_team_snapshots
                WHERE season=%s AND team_id=%s
                ORDER BY snapshot_date
            """, (season,chosen_id))
            if not snapshots.empty:
                current = snapshots.iloc[-1]
                col1,col2,col3 = st.columns(3)
                col1.metric("Imported games", f"{int(current['games_played'])}")
                col2.metric("Latest adjusted net / 100",
                            f"{current['adj_net']:+.1f}")
                col3.metric("Latest opponent SOS",
                            f"{current['opponent_sos']:+.1f}")
                graph = snapshots.set_index('snapshot_date')[
                    ['adj_net', 'opponent_sos']].rename(columns={
                        'adj_net': 'Adjusted net / 100',
                        'opponent_sos': 'Opponent SOS / 100'})
                st.line_chart(graph)
                display = snapshots.rename(columns={
                    'snapshot_date':'Date', 'games_played':'GP',
                    'adj_off':'Adjusted offense', 'adj_def':'Adjusted defense',
                    'adj_net':'Adjusted net', 'opponent_sos':'Opponent SOS'})
                show_table(display)
                st.download_button("Download team trend history (CSV)",
                                   display.to_csv(index=False),
                                   f"historical_sos_{season}_{chosen_id}.csv",
                                   "text/csv")
                st.caption("Only dates on which this team has imported complete "
                           "box scores appear. If historical games haven't been "
                           "backfilled, the trend will be incomplete.")

elif page == "Player season averages":
    st.subheader("Player season averages")
    st.caption("Season-to-date NCAA Division I men's basketball averages from imported completed "
               "games. Sorted alphabetically by team, then by points per game within each "
               "team. DNP entries are excluded. Shooting percentages are based on total "
               "made / attempted shots.")
    averages = query(PLAYER_SEASON_AVERAGES_QUERY, (season,))
    if averages.empty:
        st.info("No player appearances are available for this season. "
                "Make sure the season has been imported in GitHub Actions.")
    else:
        # Find an exact team from a searchable list. A search for Iowa also matches
        # Iowa State, so provide a separate exact-team dropdown.
        all_teams = sorted(averages["team"].dropna().unique().tolist(),
                           key=str.casefold)
        team_col, player_col, gp_col = st.columns([2, 2, 1])
        with team_col:
            team_search = st.text_input(
                "Search teams", placeholder="Type a team name, e.g. Iowa")
            search_text = team_search.strip().casefold()
            matched_teams = [name for name in all_teams
                             if search_text in name.casefold()]
            all_label = "All matching teams" if search_text else "All teams"
            selected_team = st.selectbox(
                "Choose a team", [all_label] + matched_teams,
                help="Type above to narrow teams, then choose one for an exact match.")
        with player_col:
            player_search = st.text_input("Find a player", placeholder="Search by player name")
        with gp_col:
            minimum_gp = st.number_input("Minimum GP", min_value=1,
                                         max_value=max(1, int(averages["gp"].max())),
                                         value=1, step=1)
        filtered = averages
        if search_text:
            filtered = filtered.loc[filtered["team"].isin(matched_teams)]
        if selected_team != all_label:
            filtered = filtered.loc[filtered["team"] == selected_team]
        if player_search:
            filtered = filtered.loc[filtered["player"].str.contains(
                player_search, case=False, regex=False, na=False)]
        filtered = filtered.loc[filtered["gp"] >= minimum_gp]
        filtered = filtered.sort_values(["team", "ppg", "player"],
                                        ascending=[True, False, True],
                                        kind="stable")
        st.caption(f"{len(filtered):,} players shown across "
                   f"{filtered['team'].nunique():,} teams. "
                   "GP = games played; MPG = minutes; PPG = points; RPG = rebounds; "
                   "APG = assists; SPG = steals; BPG = blocks; TOPG = turnovers. "
                   "⛑️ means a participating conference reported a player as not fully "
                   "available for an upcoming conference game; it does not necessarily "
                   "mean the player is injured. No marker does not mean healthy.")
        if filtered.empty:
            st.info("No players match your selected filters.")
        else:
            profiles = {}
            # A selected team needs only one ESPN roster request. The existing
            # box-score athlete JSON supplies any already-stored height/weight.
            if season == latest_season and filtered["team_id"].nunique() == 1:
                profiles = current_roster_profiles(str(filtered.iloc[0]["team_id"]))
            elif season == latest_season:
                st.caption("Select one team to retrieve current roster height and weight "
                           "from ESPN. Missing details are shown as —.")
            # Official conference reports are GAME-SPECIFIC, not general diagnoses.
            # Display only reports for games today through two days ahead, so an
            # old "Out" designation cannot remain attached after that game.
            injury_by_player = {}
            if season == latest_season:
                today_central = datetime.now(CENTRAL).date()
                injury_rows = query("""
                    SELECT source_player_id, injury_detail, status, retrieved_at
                    FROM injury_reports
                    WHERE source = %s
                      AND raw_json::jsonb->>'game_date' >= %s
                      AND raw_json::jsonb->>'game_date' <= %s
                      AND source_player_id IS NOT NULL
                    ORDER BY raw_json::jsonb->>'game_date' ASC,
                             reported_at DESC, retrieved_at DESC
                """, ("Official conference report", today_central.isoformat(),
                      (today_central + timedelta(days=2)).isoformat()))
                for _, injury in injury_rows.iterrows():
                    player_id = str(injury["source_player_id"])
                    if player_id not in injury_by_player:
                        injury_by_player[player_id] = {
                            "injury_detail": injury["injury_detail"],
                            "status": injury["status"],
                        }
            shown = player_averages_display_rows(filtered, profiles, injury_by_player)
            renamed = shown.rename(columns={
                "team": "Team", "player": "Player", "position": "Pos",
                "height": "Height", "weight": "Weight",
                "gp": "GP", "mpg": "MPG", "ppg": "PPG", "rpg": "RPG",
                "apg": "APG", "spg": "SPG", "bpg": "BPG", "topg": "TOPG",
                "fg_pct": "FG%", "three_pct": "3P%", "ft_pct": "FT%",
            })
            columns = ["Team", "Player", "Pos", "Height", "Weight", "GP", "MPG", "PPG", "RPG",
                       "APG", "SPG", "BPG", "TOPG", "FG%", "3P%", "FT%"]
            table = renamed[columns]
            show_table(table)
            st.download_button("Download season averages (CSV)", table.to_csv(index=False),
                               f"ncaa_player_season_averages_{season}.csv", "text/csv")

elif page == "Historical Data Quality":
    st.subheader("Historical data quality & pregame statistics")
    st.caption("Coverage of completed ESPN games, strictly prior-date team features, "
               "and season-separated scoring-baseline accuracy. These are "
               "historical research diagnostics—not a trained game-outcome model.")
    has_tables = query("""
        SELECT to_regclass('public.historical_coverage_audit') IS NOT NULL AS coverage,
               to_regclass('public.pregame_team_features') IS NOT NULL AS features,
               to_regclass('public.historical_score_baseline_audit') IS NOT NULL AS accuracy
    """)
    if has_tables.empty or not all(bool(has_tables.iloc[0][name])
                                   for name in ("coverage", "features", "accuracy")):
        st.info("Historical QA tables haven't been created yet. In GitHub Actions, "
                "run 'Verify NCAA MBB Historical Pregame Data' after uploading "
                "pregame_quality.py and its workflow.")
    else:
        coverage_tab, pregame_tab, accuracy_tab = st.tabs([
            "Coverage audit", "Pregame-only statistics", "Holdout accuracy audit"
        ])
        with coverage_tab:
            coverage_data = query("""
                SELECT season, all_game_records, completed_games,
                       not_completed_games, d1_matchups, complete_scores,
                       both_box_scores, usable_stat_games, score_mismatches,
                       incomplete_possessions, missing_start_time
                FROM historical_coverage_audit
                ORDER BY season DESC
            """)
            if coverage_data.empty:
                st.info("No coverage audit has been generated. Run the workflow.")
            else:
                display = coverage_data.copy()
                display["Season"] = display["season"].map(historical_season_label)
                display = display.rename(columns={
                    "all_game_records": "All records",
                    "completed_games": "Completed", "not_completed_games": "Not completed",
                    "d1_matchups": "Completed D1 vs D1",
                    "complete_scores": "Final scores present",
                    "both_box_scores": "Two box scores",
                    "usable_stat_games": "Usable box scores",
                    "score_mismatches": "Score mismatches",
                    "incomplete_possessions": "Invalid/missing possessions",
                    "missing_start_time": "Missing tipoff timestamp"
                }).drop(columns=["season"])
                display = display[["Season"] + [c for c in display if c != "Season"]]
                show_table(display)
                st.caption("'Not completed' includes cancellations, postponements "
                           "and other uncompleted records; it is not the number of "
                           "missing results. Current is_D1 flags may not reflect "
                           "a program's historical classification.")
                st.download_button("Download coverage audit (CSV)",
                                   display.to_csv(index=False),
                                   "ncaa_coverage_audit.csv", "text/csv",
                                   key="qa_coverage_download")
        with pregame_tab:
            st.caption("Each row describes ONE team before a completed game. "
                       "Same-day games are excluded from the input history, "
                       "including when an earlier tipoff time was recorded. "
                       "Adjusted ratings come only from an earlier snapshot date.")
            total = query("""
                SELECT COUNT(*) AS rows_count,
                       COUNT(*) FILTER (WHERE games_before >= 5) AS rows_with_history,
                       COUNT(*) FILTER (WHERE rating_snapshot_date IS NOT NULL)
                           AS rows_with_adjusted_rating,
                       COUNT(*) FILTER (WHERE rating_snapshot_date >= game_date)
                           AS snapshot_leak_violations
                FROM pregame_team_features WHERE season=%s
            """, (season,))
            if total.empty or int(total.iloc[0]["rows_count"]) == 0:
                st.info("No pregame features for this season. Run the historical QA workflow.")
            else:
                x = total.iloc[0]
                c1,c2,c3,c4 = st.columns(4)
                c1.metric("Pregame team rows", f"{int(x['rows_count']):,}")
                c2.metric("5+ prior box scores", f"{int(x['rows_with_history']):,}")
                c3.metric("Prior adjusted ratings", f"{int(x['rows_with_adjusted_rating']):,}")
                c4.metric("Snapshot date violations", f"{int(x['snapshot_leak_violations']):,}")
                choices = query("""
                    SELECT DISTINCT f.team_id, t.name
                    FROM pregame_team_features f JOIN teams t ON t.team_id=f.team_id
                    WHERE f.season=%s
                    ORDER BY t.name,f.team_id
                """, (season,))
                if not choices.empty:
                    team_ids = choices["team_id"].astype(str).tolist()
                    names = dict(zip(team_ids,choices["name"].tolist()))
                    selected = st.selectbox("Inspect team's pregame history",team_ids,
                                            format_func=lambda tid:names[tid],
                                            key="qa_team")
                    rows = query("""
                        SELECT f.game_date AS "Date", ot.name AS "Opponent",
                               CASE WHEN f.neutral_site=1 THEN 'Neutral'
                                    WHEN f.is_home=1 THEN 'Home' ELSE 'Away' END
                                    AS "Location",
                               f.games_before AS "GP before",
                               ROUND(f.avg_points_for::numeric,1) AS "Prior PPG",
                               ROUND(f.avg_points_against::numeric,1) AS "Prior Opp PPG",
                               ROUND(f.raw_net::numeric,1) AS "Prior raw net/100",
                               ROUND(f.adjusted_net_before::numeric,1) AS "Prior adjusted net",
                               ROUND(f.opponent_sos_before::numeric,1) AS "Prior SOS",
                               f.rest_days AS "Full rest days",
                               f.rating_snapshot_date AS "Rating as-of"
                        FROM pregame_team_features f
                        JOIN teams ot ON ot.team_id=f.opponent_id
                        WHERE f.season=%s AND f.team_id=%s
                        ORDER BY f.game_date,f.game_id
                    """, (season,selected))
                    if not rows.empty:
                        show_table(rows)
                        st.download_button("Download selected team's pregame stats (CSV)",
                                           rows.to_csv(index=False),
                                           f"ncaa_pregame_{season}_{selected}.csv", "text/csv",
                                           key="qa_pregame_download")
                st.caption("GP before counts previous games with usable team box "
                           "scores, not all prior completed games. Missing early-season "
                           "averages and prior adjusted ratings are left blank on purpose.")
        with accuracy_tab:
            st.markdown("#### Historical scoring-baseline accuracy")
            st.caption("This simple benchmark uses each team's mean points "
                       "from at least five earlier usable games and checks its "
                       "absolute scoring error on later D1-vs-D1 games. It is "
                       "evaluated chronologically, with no same-day data. "
                       "It does NOT measure a trained game's winner accuracy "
                       "or prove the accuracy of a future prediction engine.")
            audits = query("""
                SELECT season, split_name, eligible_d1_games,
                       evaluated_games, evaluated_team_scores,
                       ROUND(points_mae::numeric,2) AS points_mae,
                       ROUND(points_rmse::numeric,2) AS points_rmse,
                       ROUND(points_bias::numeric,2) AS points_bias
                FROM historical_score_baseline_audit ORDER BY season
            """)
            if audits.empty:
                st.info("No historical scoring-baseline audit yet. Run the workflow.")
            else:
                display = audits.rename(columns={
                    "season": "ESPN season", "split_name": "Split",
                    "eligible_d1_games": "D1 games with scores",
                    "evaluated_games": "Evaluated games",
                    "evaluated_team_scores": "Team scores evaluated",
                    "points_mae": "Points MAE",
                    "points_rmse": "Points RMSE",
                    "points_bias": "Mean bias"
                })
                show_table(display)
                st.caption("Development = older seasons; validation = next-to-last "
                           "season; test = latest completed season in the workflow "
                           "range. MAE is in team points. These are out-of-season "
                           "checks of an unfitted rolling baseline, NOT a trained model.")
                st.download_button("Download accuracy audit (CSV)",
                                   display.to_csv(index=False),
                                   "ncaa_scoring_accuracy_audit.csv", "text/csv",
                                   key="qa_accuracy_download")


elif page == "Historical Scoring Review":
    st.subheader("Historical scoring review")
    st.caption("Retrospective descriptive analysis of completed games. This page does not "
               "forecast future games or generate winner probabilities, picks, or odds.")
    existing = query("SELECT to_regclass('public.pregame_team_features') IS NOT NULL AS exists")
    if existing.empty or not bool(existing.iloc[0]["exists"]):
        st.info("First run the Verify NCAA MBB Historical Pregame Data workflow.")
    else:
        source_rows = query("""
            SELECT f.game_date, f.team_id, t.name AS team,
                   CASE WHEN f.neutral_site=1 THEN 'Neutral'
                        WHEN f.is_home=1 THEN 'Home' ELSE 'Away' END AS location,
                   f.games_before,
                   f.avg_points_for AS prior_ppg,
                   s.points AS points
            FROM pregame_team_features f
            JOIN games g ON g.game_id=f.game_id
              AND g.season=f.season AND g.completed=1
            JOIN teams t ON t.team_id=f.team_id
            JOIN team_game_stats s ON s.game_id=f.game_id AND s.team_id=f.team_id
            WHERE f.season=%s AND f.games_before >= 5
              AND f.avg_points_for IS NOT NULL
              AND s.points IS NOT NULL
              AND s.possessions > 0
              AND s.points = CASE WHEN f.is_home=1 THEN g.home_score
                                  ELSE g.away_score END
            ORDER BY f.game_date, t.name
        """, (season,))
        patterns = scoring_pattern_metrics(source_rows)
        if patterns.empty:
            st.info("No eligible games in this season yet. Refresh the pregame history workflow.")
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("Historical team box scores", f"{len(patterns):,}")
            c2.metric("Average actual points", f"{patterns['points'].mean():.1f}")
            c3.metric("Typical scoring variation", f"{patterns['absolute_difference'].mean():.1f} points")
            st.caption("Typical variation is the mean absolute difference between a team's "
                       "earlier-season scoring average and its actual historical game score. "
                       "It is not a future-game accuracy guarantee.")

            location_tab, maturity_tab, timeline_tab, team_tab = st.tabs([
                "By location", "By season stage", "Monthly patterns", "Team scoring history"
            ])
            with location_tab:
                st.markdown("#### Historical scoring by venue")
                loc_summary = scoring_pattern_group_summary(patterns, "location")
                show_table(loc_summary)
                st.caption("Home/away/neutral are taken from the historical game data; "
                           "these aggregates do not adjust for opponent quality.")
            with maturity_tab:
                st.markdown("#### Historical scoring as the season develops")
                stage_summary = scoring_pattern_group_summary(patterns, "history_group")
                show_table(stage_summary)
                st.caption("Groups are based on how many earlier usable box scores "
                           "the team had before each completed game.")
            with timeline_tab:
                st.markdown("#### Variation over the historical season")
                monthly = scoring_pattern_group_summary(patterns, "month")
                show_table(monthly)
                if not monthly.empty:
                    st.line_chart(monthly.set_index("month")[["Typical variation"]])
            with team_tab:
                st.markdown("#### Explore a team's completed scoring history")
                names = (patterns[["team_id", "team"]].drop_duplicates()
                         .sort_values(["team", "team_id"]))
                ids = names["team_id"].astype(str).tolist()
                labels = dict(zip(ids, names["team"].astype(str)))
                if ids:
                    chosen = st.selectbox("Team", ids,
                                          format_func=lambda tid: labels[tid],
                                          key="scoring_review_team")
                    one_team = patterns[patterns["team_id"].astype(str)==chosen].copy()
                    chart = one_team[["game_date", "points", "prior_ppg"]].set_index("game_date")
                    st.line_chart(chart.rename(columns={"points": "Historical actual score",
                                                        "prior_ppg": "Average before game"}))
                    team_table = one_team[["game_date", "location", "games_before", "prior_ppg", "points",
                                           "absolute_difference"]].rename(columns={
                        "game_date": "Game date", "location": "Court", "games_before": "Earlier games",
                        "prior_ppg": "Prior PPG", "points": "Actual points",
                        "absolute_difference": "Difference from prior PPG"
                    })
                    show_table(team_table)
                    st.download_button("Download team scoring history (CSV)",
                                       team_table.to_csv(index=False),
                                       f"ncaa_historical_scoring_{season}_{chosen}.csv", "text/csv",
                                       key="scoring_review_csv")
    st.caption("This page uses existing historical tables, does not change Supabase "
               "records, and does not require a new workflow.")


elif page == "Statistical Feature Analysis":
    st.subheader("Statistical feature analysis")
    st.caption("Retrospective relationships between stats known BEFORE each game and actual "
               "completed-game performance. Correlations are descriptive, not causal, "
               "and are not a measure of a trained model's accuracy.")
    table_exists = query(
        "SELECT to_regclass('public.pregame_team_features') IS NOT NULL AS exists"
    )
    if table_exists.empty or not bool(table_exists.iloc[0]["exists"]):
        st.info("First run Verify NCAA MBB Historical Pregame Data in GitHub Actions.")
    else:
        scope = st.radio("Historical game range", ["Five completed seasons (2021–22 to 2025–26)",
                                                  "Selected ESPN season"], horizontal=True,
                         key="feature_analysis_scope")
        start_season, end_season = ((2022, 2026)
                                    if scope.startswith("Five completed") else (season, season))
        min_games = st.slider("Minimum prior usable games per team", min_value=1,
                              max_value=20, value=5, key="feature_min_prior_games")
        historical = query(FEATURE_ANALYSIS_SQL, (start_season, end_season))
        analysis = feature_analysis_prepare(historical, min_history=min_games)
        if analysis.empty:
            st.info("No eligible completed D1 games with enough prior history in this range. "
                    "Try selecting the five historical seasons or reducing the prior-games minimum.")
        else:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Historical team-game rows", f"{len(analysis):,}")
            col2.metric("Unique completed games", f"{analysis['game_id'].nunique():,}")
            col3.metric("Seasons included", f"{analysis['season'].nunique()}")
            col4.metric("Invalid/missing prior rating used", "0")
            if analysis["invalid_prior_rating"].any():
                st.warning(f"{int(analysis['invalid_prior_rating'].sum()):,} rows "
                           "had adjusted ratings without a strictly earlier snapshot. "
                           "Those values are EXCLUDED from adjusted-rating analysis.")
            st.caption("Each team contributes one row per completed game; two teams from "
                       "the same matchup are not independent observations. Only D1-vs-D1 "
                       "matchups with consistent final scores and usable box scores are included.")

            overview_tab, detail_tab, location_tab, coverage_tab = st.tabs([
                "Correlation overview", "Metric explorer", "Venue and rest", "Feature coverage"
            ])
            with overview_tab:
                st.markdown("#### Pregame statistics versus historical results")
                correlations = feature_analysis_correlations(analysis)
                display_correlations = correlations.copy()
                display_correlations["Pearson r"] = display_correlations["Pearson r"].round(3)
                show_table(display_correlations)
                chart_correlations = correlations.dropna(subset=["Pearson r"])
                if not chart_correlations.empty:
                    st.bar_chart(chart_correlations.set_index("Statistic")[["Pearson r"]])
                st.caption("Pearson r ranges from −1 to +1; its magnitude measures a "
                           "linear association, not explanatory certainty or predictive "
                           "accuracy. Metrics here have DIFFERENT historical outcomes "
                           "(scoring, points allowed, margin, and combined points), so "
                           "the bars should not be treated as a single feature ranking.")
            with detail_tab:
                st.markdown("#### Explore one pregame statistical relationship")
                metric = st.selectbox("Pregame statistic", list(FEATURE_ASSOCIATIONS),
                                      key="feature_analysis_metric")
                source_col, target_col, target_name = FEATURE_ASSOCIATIONS[metric]
                pairs = feature_analysis_pairs(analysis, metric)
                if pairs.empty:
                    st.info("This pregame statistic has no usable observations in the selected range.")
                else:
                    corr = (pairs[source_col].corr(pairs[target_col])
                            if len(pairs) >= 30 and pairs[source_col].nunique() > 1
                            and pairs[target_col].nunique() > 1 else float("nan"))
                    c1, c2 = st.columns(2)
                    c1.metric("Historical observations", f"{len(pairs):,}")
                    c2.metric("Pearson r", f"{corr:+.3f}" if pd.notna(corr) else "Insufficient variation")
                    st.caption("Scatter plot is deterministically sampled for readability; "
                               "tables and correlations use all eligible rows.")
                    stride = max(1, (len(pairs) + 1999) // 2000)
                    plot = pairs.iloc[::stride].rename(columns={source_col: metric,
                                                                 target_col: target_name})
                    st.scatter_chart(plot, x=metric, y=target_name)
                    st.markdown("##### Historical feature quartiles")
                    bands = feature_analysis_bands(pairs, metric)
                    if not bands.empty:
                        show_table(bands)
                    else:
                        st.caption("Insufficient unique values for quartile comparisons.")
                    st.markdown("##### Correlation by season")
                    per_season = []
                    for yr, subset in pairs.groupby("season"):
                        val = (subset[source_col].corr(subset[target_col])
                               if len(subset) >= 30 and subset[source_col].nunique() > 1
                               and subset[target_col].nunique() > 1 else float("nan"))
                        per_season.append({"Season": historical_season_label(int(yr)),
                                           "Team-game rows": len(subset), "Pearson r": val})
                    show_table(pd.DataFrame(per_season).round(3))
            with location_tab:
                st.markdown("#### Scoring by venue")
                venue = feature_analysis_venue_rest(analysis, "venue")
                show_table(venue)
                if not venue.empty:
                    st.bar_chart(venue.set_index("venue")[["Avg scored", "Avg allowed"]])
                st.markdown("#### Scoring by days of rest")
                rest = feature_analysis_venue_rest(analysis, "rest_band")
                show_table(rest)
                st.caption("Venue/rest differences are NOT adjusted for opponent strength, "
                           "team composition, schedule timing, or selection effects, "
                           "and must not be interpreted as causal home-court/rest effects.")
            with coverage_tab:
                st.markdown("#### Availability of each pregame metric")
                coverage = feature_analysis_coverage(analysis)
                show_table(coverage)
                st.caption("Missing adjusted ratings are intentionally left blank, "
                           "especially at the beginning of a season. No future or "
                           "same-day ratings are filled in.")
                st.download_button("Download feature coverage (CSV)",
                                   data=coverage.to_csv(index=False),
                                   file_name=f"ncaa_feature_coverage_{start_season}_{end_season}.csv",
                                   mime="text/csv", key="feature_coverage_download")
                st.download_button("Download eligible historical team-game rows (CSV)",
                                   data=analysis.drop(columns=["invalid_prior_rating"])
                                                .to_csv(index=False),
                                   file_name=f"ncaa_feature_history_{start_season}_{end_season}.csv",
                                   mime="text/csv", key="feature_history_download")
    st.caption("Data source: your existing Supabase pregame_team_features and completed "
               "ESPN games/box scores. This page reads data only. It does not modify "
               "Supabase or produce upcoming-matchup forecasts, picks, or wagering outputs.")


elif page == "Advanced Matchup Comparison":
    st.subheader("Advanced Matchup Comparison")
    st.caption("Compare two Division I teams from the selected season using imported, "
               "completed games. Ratings for past seasons describe the whole season, "
               "NOT what was known before a historical game. This page does not "
               "estimate outcomes of future games.")
    tables = query("SELECT to_regclass('public.team_adjusted_ratings') IS NOT NULL AS available")
    if tables.empty or not bool(tables.iloc[0]["available"]):
        st.info("Run 'Update NCAA MBB Adjusted SOS and History' in GitHub Actions first.")
    else:
        rated = historical_rankings(query(ADVANCED_MATCHUP_RATINGS_SQL, (season,)))
        if len(rated) < 2:
            st.info("At least two rated D1 teams are required. Refresh the season's "
                    "adjusted ratings in GitHub Actions.")
        else:
            ap_for_matchup, poll_source = comparison_ap_poll(season, latest_season)
            st.caption(f"Basketball season: {historical_season_label(season)}. "
                       f"AP poll: {poll_source or 'unavailable'}; historical AP ranks "
                       "are from the selected season's latest available regular-season poll, "
                       "not the date of a specific game.")
            labels = {str(r["team_id"]): comparison_team_name(
                          r["team"], r["conference"], r["team_id"],
                          ap_for_matchup, bool(ap_for_matchup))
                      for _, r in rated.iterrows()}
            team_ids = list(labels)
            a_selector, b_selector = st.columns(2)
            with a_selector:
                a_id = st.selectbox("Team A", team_ids, index=0,
                                    format_func=lambda x: labels[x],
                                    key="advanced_matchup_a")
            with b_selector:
                b_id = st.selectbox("Team B", team_ids, index=1,
                                    format_func=lambda x: labels[x],
                                    key="advanced_matchup_b")
            if a_id == b_id:
                st.info("Choose two different teams to see a comparison.")
            else:
                a = rated.loc[rated["team_id"].astype(str).eq(a_id)].iloc[0]
                b = rated.loc[rated["team_id"].astype(str).eq(b_id)].iloc[0]
                records = season_records(season)
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown('<div class="mbb-team-card">' + team_badge(
                        a["team"], a_id, a["logo_url"], ap_for_matchup, records) +
                        '</div>', unsafe_allow_html=True)
                    st.metric("Adjusted net / 100 possessions",
                              historical_number(a["adj_net"], signed=True))
                    st.caption(f"Adjusted efficiency rank #{int(a['national_rank'])}")
                with c2:
                    st.markdown('<div class="mbb-team-card">' + team_badge(
                        b["team"], b_id, b["logo_url"], ap_for_matchup, records) +
                        '</div>', unsafe_allow_html=True)
                    st.metric("Adjusted net / 100 possessions",
                              historical_number(b["adj_net"], signed=True))
                    st.caption(f"Adjusted efficiency rank #{int(b['national_rank'])}")
                recent_n = st.select_slider("Recent completed games to compare",
                                            options=[3, 5, 10], value=5,
                                            key="advanced_matchup_recent_n")
                boxscores = query(ADVANCED_MATCHUP_GAMES_SQL, (season, a_id, b_id))
                a_games = advanced_matchup_history(boxscores, a_id)
                b_games = advanced_matchup_history(boxscores, b_id)
                a_sum = advanced_matchup_summary(a_games, recent_n)
                b_sum = advanced_matchup_summary(b_games, recent_n)
                stats_tab, games_tab, head_to_head_tab = st.tabs([
                    "📊 Side-by-side stats", "🗓️ Recent game history", "🤝 Head-to-head"
                ])
                with stats_tab:
                    st.markdown("#### Team performance and schedule quality")
                    st.caption("'Team A' and 'Team B' follow the selectors above. "
                               "Adjusted metrics and SOS come from season-to-date "
                               "ratings; scoring and tempo use games with complete "
                               "ESPN team box scores. The full-season numbers for "
                               "archived seasons include all completed games of that year.")
                    comparison = advanced_matchup_comparison_table(
                        a, b, a_sum, b_sum, record_label(a_id, records),
                        record_label(b_id, records))
                    show_table(comparison)
                    st.download_button("Download matchup statistics (CSV)",
                                       comparison.to_csv(index=False),
                                       f"ncaa_matchup_{season}_{a_id}_{b_id}.csv", "text/csv",
                                       key="advanced_matchup_download")
                    st.caption("A lower adjusted defensive efficiency is better. "
                               "SOS is a measure of prior opponent quality, not a "
                               "guarantee of better results. Missing values are —.")
                with games_tab:
                    left_games, right_games = st.columns(2)
                    with left_games:
                        st.markdown(f"#### {a['team']} — last {recent_n}")
                        show_table(advanced_matchup_recent_table(a_games, recent_n))
                    with right_games:
                        st.markdown(f"#### {b['team']} — last {recent_n}")
                        show_table(advanced_matchup_recent_table(b_games, recent_n))
                    st.caption("W/L and scores are from each selected team's perspective. "
                               "Only finished games with both teams' box scores appear; "
                               "canceled and postponed games are excluded.")
                with head_to_head_tab:
                    st.markdown("#### Direct meetings during the selected season")
                    meetings = a_games.loc[a_games["opponent_id"].astype(str).eq(b_id)]
                    if meetings.empty:
                        st.info("No completed meeting with usable box scores was "
                                "found for these teams in this season.")
                    else:
                        show_table(advanced_matchup_recent_table(meetings, len(meetings)))
                    st.caption("This is a record of games already played, "
                               "not a prediction for a future matchup.")

elif page == "Injuries":
    st.subheader("Official conference player availability")
    st.caption("Statuses come from manually verified official conference reports "
               "for individual conference games. Out, Questionable, or Doubtful "
               "does not necessarily mean an injury. Missing reports are not proof "
               "that a player is healthy or available.")
    st.markdown(
        "Official sources: "
        "[SEC](https://www.secsports.com/reports) · "
        "[ACC](https://theacc.com/sports/2025/12/29/availability-reporting-mbb.aspx) · "
        "[Big 12](https://big12sports.com/sports/2026/1/1/MBBreporting.aspx) · "
        "[Big East](https://www.bigeast.com/news/2025/12/12/general-big-east-to-provide-public-availability-reports-for-mens-and-womens-basketball.aspx)"
    )
    # Automatically monitored publication links: these are not player statuses.
    index_path = Path(__file__).resolve().parent / "data" / "conference_report_index.json"
    if index_path.is_file():
        try:
            source_index = json.loads(index_path.read_text(encoding="utf-8"))
            with st.expander("Latest official report links (checked automatically)", expanded=False):
                st.caption("Last link check: " + str(source_index.get("checked_at", "Unknown"))
                           + " UTC. Embedded report viewers may have no directly linked documents. "
                           + "These links do not confirm any player is injured or unavailable.")
                for conference_name, source in source_index.get("sources", {}).items():
                    st.markdown(f"**{escape(conference_name)}** — "
                                f"[{source.get('status', 'unknown')}]({source.get('page', '#')})")
                    for entry in source.get("links", [])[:8]:
                        st.markdown(f"- [{escape(entry.get('title', 'Official report'))}]"
                                    f"({entry.get('url', '#')})")
        except (OSError, ValueError, TypeError):
            st.caption("Official report link index is not available yet.")
    today_central = datetime.now(CENTRAL).date()
    left, right = st.columns(2)
    with left:
        first_date = st.date_input("From game date", value=today_central, key="conf_start")
    with right:
        last_date = st.date_input("Through game date", value=today_central + timedelta(days=3),
                                  key="conf_end")
    if last_date < first_date:
        st.warning("The end game date must not be before the start date.")
    else:
        reports = query("""
            SELECT player_name, team_name, status,
                   raw_json::jsonb->>'conference' AS conference,
                   raw_json::jsonb->>'game_date' AS game_date,
                   raw_json::jsonb->>'source_url' AS source_url,
                   reported_at, retrieved_at
            FROM injury_reports
            WHERE source = %s
              AND raw_json::jsonb->>'game_date' >= %s
              AND raw_json::jsonb->>'game_date' <= %s
            ORDER BY raw_json::jsonb->>'game_date' ASC,
                     reported_at DESC, player_name ASC
            LIMIT 1000
        """, ("Official conference report", first_date.isoformat(), last_date.isoformat()))
        if reports.empty:
            st.info("No verified conference availability reports entered for these game dates. "
                    "This is common before conference play begins. Check the official sources "
                    "above; an empty list does not mean no players are unavailable.")
        else:
            st.caption(f"{len(reports):,} conference availability entries. "
                       "Statuses apply only to the corresponding game dates. "
                       "The source links open the original reports.")
            display = reports.rename(columns={
                "player_name": "Player", "team_name": "Team", "status": "Status",
                "conference": "Conference", "game_date": "Game date",
                "source_url": "Official source", "reported_at": "Report time (UTC)",
                "retrieved_at": "Imported (UTC)",
            })
            st.dataframe(display, use_container_width=True, hide_index=True,
                         column_config={"Official source": st.column_config.LinkColumn(
                             "Official source", display_text="Open report")})

else:
    st.subheader("Cloud update status")
    st.markdown("#### Automated database health")
    has_health = query("SELECT to_regclass('public.data_health_runs') IS NOT NULL AS exists")
    if has_health.empty or not bool(has_health.iloc[0]["exists"]):
        st.info("No automated health checks yet. In GitHub Actions, run "
                "'NCAA MBB Daily Data Health' once after uploading its workflow "
                "and daily_data_health.py.")
    else:
        health = query("""
            SELECT run_id, checked_at, season, window_start, window_end,
                   status, last_import_status, last_import_finished_at,
                   import_age_hours, recent_completed_d1_games, missing_scores,
                   missing_box_scores, invalid_possessions, score_mismatches,
                   missing_pregame_pairs, current_season_completed_d1_games,
                   latest_adjusted_at, latest_pregame_at, issues_json
            FROM data_health_runs ORDER BY run_id DESC LIMIT 20
        """)
        if health.empty:
            st.info("The audit table exists but no checks have run. "
                    "Run 'NCAA MBB Daily Data Health' in GitHub Actions.")
        else:
            h = health.iloc[0]
            h1, h2, h3, h4 = st.columns(4)
            h1.metric("Latest data health", str(h["status"]))
            h2.metric("Recent completed D1 games", f"{int(h['recent_completed_d1_games']):,}")
            h3.metric("Missing team box scores", f"{int(h['missing_box_scores']):,}")
            h4.metric("Score mismatches", f"{int(h['score_mismatches']):,}")
            st.caption(f"Checked {h['checked_at']} UTC · ESPN season {int(h['season'])} "
                       f"· reviewed {h['window_start']} through {h['window_end']} "
                       "(completed D1-vs-D1 games only; the most recent calendar day "
                       "is excluded to allow delayed box scores).")
            raw_issues = h.get("issues_json", "[]")
            try:
                issues = json.loads(raw_issues) if isinstance(raw_issues, str) else []
            except (json.JSONDecodeError, TypeError, ValueError):
                issues = ["Latest audit issues could not be decoded."]
            if issues:
                if str(h["status"]) == "ERROR":
                    st.error("Latest health check found a critical data issue.")
                else:
                    st.warning("Latest health check found something to review.")
                for issue in issues:
                    st.write(f"• {issue}")
            else:
                st.success("No problems detected by the configured health checks.")
            with st.expander("Last 20 health checks"):
                display_cols = ["checked_at", "season", "status", "last_import_status",
                                "recent_completed_d1_games", "missing_scores",
                                "missing_box_scores", "invalid_possessions",
                                "score_mismatches", "missing_pregame_pairs"]
                show_table(health[display_cols].rename(columns={
                    "checked_at": "Checked (UTC)", "season": "ESPN season",
                    "status": "Health", "last_import_status": "ESPN import",
                    "recent_completed_d1_games": "Recent D1 games",
                    "missing_scores": "Missing scores",
                    "missing_box_scores": "Box-score gaps",
                    "invalid_possessions": "Invalid possessions",
                    "score_mismatches": "Score mismatches",
                    "missing_pregame_pairs": "Missing pregame pairs",
                }))
    st.divider()
    st.markdown("#### ESPN import history")
    runs = query("""
        SELECT started_at,finished_at,status,dates_checked,games_seen,
               summaries_saved,errors FROM sync_runs ORDER BY run_id DESC LIMIT 20
    """)
    if runs.empty:
        st.info("No cloud imports recorded yet. Run the nightly workflow manually once.")
    else:
        latest = runs.iloc[0]
        st.metric("Latest import status", latest["status"].upper())
        st.caption(f"Last attempt finished (UTC): {latest['finished_at'] or '—'}")
        show_table(runs.drop(columns=["errors"]))
        if isinstance(latest.get("errors"), str) and latest["errors"].strip():
            with st.expander("Latest importer warnings"):
                st.code(latest["errors"])

st.divider()
st.caption("Raw Off/100 and Def/100 on existing game tables are unadjusted; "
           "the new Adjusted SOS page is opponent-adjusted and includes small-sample shrinkage. "
           "AP poll is fetched from ESPN when available; team logos and conferences come from imported ESPN data. "
           "ESPN endpoints are unofficial and may change; nightly updates and AP poll availability are not guaranteed. "
           "Official conference availability reports are separately verified and game-specific.")
