"""Browser/mobile dashboard for the hosted NCAA men's D1 basketball database.

Streamlit executes this code on its server. DATABASE_URL is never sent to a
browser; data is read from PostgreSQL using parameterized, read-only queries.
"""
from __future__ import annotations

import os
import logging
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


def player_averages_display_rows(frame: pd.DataFrame, profiles: dict[str, dict]) -> pd.DataFrame:
    """Add visible player jersey, height and weight without changing stat totals."""
    shown = frame.copy()
    visible_names = []
    heights = []
    weights = []
    for _, row in shown.iterrows():
        profile = profiles.get(str(row["player_id"]), {})
        number = _visible_jersey(row.get("jersey")) or _visible_jersey(profile.get("jersey"))
        visible_names.append(str(row["player"]) + (f" #{number}" if number else ""))
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


st.title("🏀 NCAA Men's Basketball Database")
st.caption("Division I • ESPN game statistics • unadjusted, possession-based efficiency • Central Time")
with st.sidebar:
    st.header("Navigation")
    page = st.radio("Section", ["Game schedules", "Teams & conferences", "Team efficiency", "Team explorer",
                                    "Player season averages", "Injuries", "Update status"], label_visibility="collapsed")
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
                   "APG = assists; SPG = steals; BPG = blocks; TOPG = turnovers.")
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
            shown = player_averages_display_rows(filtered, profiles)
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

elif page == "Injuries":
    st.subheader("Injury reports")
    st.warning("RotoWire injury reports are NOT connected yet. Missing entries do not mean players are healthy.")
    reports = query("""
        SELECT player_name,team_name,status,injury_detail,expected_return,reported_at
        FROM injury_reports ORDER BY retrieved_at DESC LIMIT 100
    """)
    if not reports.empty:
        show_table(reports)

else:
    st.subheader("Cloud update status")
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
st.caption("Off/100 and Def/100 are based on estimated possessions and are not opponent-adjusted. "
           "AP poll is fetched from ESPN when available; team logos and conferences come from imported ESPN data. "
           "ESPN endpoints are unofficial and may change; nightly updates and AP poll availability are not guaranteed.")
