"""Six-page NCAA D-I basketball schedule and prediction-analysis dashboard.

Server-side access to existing winner-engine forecasts; displays saved model
predictions only, never fabricates a prediction from team ratings.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo
import os
import logging
from html import escape
import requests

import pandas as pd
import streamlit as st

st.set_page_config(page_title="NCAA MBB | Schedule & Evaluation", page_icon="🏀", layout="wide")
CT = ZoneInfo("America/Chicago")
HERE = Path(__file__).resolve().parent
HISTORY = HERE / "data" / "historical_confidence_2026.csv"
FORECAST_MODEL = "winner-v1.0"
CONFIDENCE_BANDS = ["50–59%", "60–69%", "70–79%", "80–89%", "90–100%"]
DB_BANDS = ["50-59%", "60-69%", "70-79%", "80-89%", "90-100%"]

st.markdown("""
<style>
  .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
    background: #ffffff !important; color: #151515 !important;
  }
  [data-testid="stSidebar"], [data-testid="stSidebarContent"] {
    background: #f2f3f6 !important;
  }
  [data-testid="stMetric"] {background:#f4f5f8; padding:12px; border-radius:10px;}
  .stApp h1, .stApp h2, .stApp h3 {color:#151515 !important;}
</style>
""", unsafe_allow_html=True)


def now_ct() -> datetime:
    return datetime.now(CT)


def database_url() -> str:
    value = os.environ.get("DATABASE_URL", "")
    if not value:
        try:
            value = st.secrets.get("DATABASE_URL", "")
        except (FileNotFoundError, KeyError):
            value = ""
    if not value:
        st.error("DATABASE_URL is missing from Streamlit Cloud secrets.")
        st.stop()
    return value


@st.cache_data(ttl=180, show_spinner=False)
def run_query(sql: str, params: tuple = ()) -> pd.DataFrame:
    """Only server-side, read-only, parameterized SQL."""
    import psycopg
    from psycopg.rows import dict_row
    with psycopg.connect(database_url(), sslmode="require", connect_timeout=15,
                         prepare_threshold=None, row_factory=dict_row) as con:
        con.read_only = True
        with con.cursor() as cur:
            cur.execute(sql, params)
            return pd.DataFrame(cur.fetchall())


@st.cache_data(ttl=300, show_spinner=False)
def active_season() -> int:
    result = run_query("SELECT MAX(season) AS season FROM public.games")
    if result.empty or pd.isna(result.iloc[0]["season"]):
        raise RuntimeError("No seasons are present in public.games")
    return int(result.iloc[0]["season"])


# The ESPN-supplied date can disagree with the local calendar date for late games.
# Query a one-day buffer on both sides, then classify by actual tipoff in CT.
SCHEDULE_SQL = """
SELECT g.game_id::text AS game_id, g.season, g.game_date::text AS game_date,
       g.start_time_utc::text AS start_time_utc, g.status,
       g.status_detail, g.completed,
       g.home_score, g.away_score, g.neutral_site,
       ht.team_id::text AS home_id, at.team_id::text AS away_id,
       ht.name AS home_team, at.name AS away_team,
       ht.logo_url AS home_logo, at.logo_url AS away_logo,
       hr.adjusted_net AS home_net, ar.adjusted_net AS away_net,
       wf.predicted_winner_team_id, wf.home_win_probability,
       wf.created_at AS forecast_created_at, wf.model_version AS forecast_model
FROM public.games g
JOIN public.teams ht ON ht.team_id = g.home_team_id AND ht.is_d1=1
JOIN public.teams at ON at.team_id = g.away_team_id AND at.is_d1=1
LEFT JOIN LATERAL (
    SELECT r.adjusted_net FROM public.team_adjusted_ratings r
    WHERE r.season=g.season AND r.team_id=g.home_team_id
    ORDER BY r.games_played DESC NULLS LAST LIMIT 1
) hr ON TRUE
LEFT JOIN LATERAL (
    SELECT r.adjusted_net FROM public.team_adjusted_ratings r
    WHERE r.season=g.season AND r.team_id=g.away_team_id
    ORDER BY r.games_played DESC NULLS LAST LIMIT 1
 ) ar ON TRUE
LEFT JOIN LATERAL (
    SELECT p.predicted_winner_team_id, p.home_win_probability,
           p.created_at, p.model_version
    FROM public.mbb_winner_forecasts p
    WHERE p.game_id = g.game_id::text
      AND p.model_version = %s
      AND p.home_team_id = g.home_team_id::text
      AND p.away_team_id = g.away_team_id::text
      AND p.created_at < p.tipoff_utc
      AND p.tipoff_utc = NULLIF(BTRIM(g.start_time_utc::text), '')::timestamptz
    ORDER BY p.created_at DESC LIMIT 1
) wf ON TRUE
WHERE g.season = %s AND g.game_date BETWEEN %s AND %s
ORDER BY g.game_date, g.start_time_utc NULLS LAST, g.game_id
"""

# A missing forecast is NOT a wrong forecast. Keep every game in the result list.
YESTERDAY_SQL = """
SELECT g.game_id::text AS game_id, g.season, g.game_date::text AS game_date,
       g.start_time_utc::text AS start_time_utc, g.completed,
       g.home_score, g.away_score, g.status_detail,
       ht.name AS home_team, at.name AS away_team,
       ht.logo_url AS home_logo, at.logo_url AS away_logo,
       ht.team_id::text AS home_id, at.team_id::text AS away_id,
       f.predicted_winner_team_id, f.home_win_probability,
       f.created_at, f.tipoff_utc, f.model_version
FROM public.games g
JOIN public.teams ht ON ht.team_id=g.home_team_id AND ht.is_d1=1
JOIN public.teams at ON at.team_id=g.away_team_id AND at.is_d1=1
LEFT JOIN LATERAL (
    SELECT p.predicted_winner_team_id, p.home_win_probability,
           p.created_at, p.tipoff_utc, p.model_version
    FROM public.mbb_winner_forecasts p
    WHERE p.game_id=g.game_id::text
      AND p.model_version=%s
      AND p.created_at < p.tipoff_utc
    ORDER BY p.created_at ASC LIMIT 1
) f ON TRUE
WHERE g.season=%s AND g.game_date BETWEEN %s AND %s
ORDER BY g.start_time_utc NULLS LAST, g.game_id
"""

SAVED_FORECAST_SQL = """
SELECT f.game_id::text AS game_id, f.home_win_probability, f.predicted_winner_team_id,
       f.home_team_id::text AS home_id, f.away_team_id::text AS away_id,
       g.home_score, g.away_score, g.completed, f.tipoff_utc, f.created_at
FROM public.mbb_winner_forecasts f
JOIN public.games g ON g.game_id::text=f.game_id
WHERE f.model_version=%s AND g.season=%s AND g.completed=1
  AND g.home_score IS NOT NULL AND g.away_score IS NOT NULL
  AND g.home_score<>g.away_score
  AND NOT ((g.home_score=0 AND g.away_score=2) OR (g.home_score=2 AND g.away_score=0))
  AND f.created_at < f.tipoff_utc
ORDER BY f.tipoff_utc,f.game_id
"""



# The current AP Top 25 is retrieved once per hour, independently of game forecasts.
# Team IDs are the ESPN IDs already stored in public.teams. We never infer ranks
# from names or substitute a different poll when the AP feed is unavailable.
def parse_ap_poll(payload: dict) -> tuple[dict[str, int], str]:
    if not isinstance(payload, dict):
        return {}, ""
    for poll in payload.get("rankings", []):
        if not isinstance(poll, dict):
            continue
        name = str(poll.get("shortName") or poll.get("name") or "")
        lower = name.lower().strip()
        if not (lower == "ap" or lower.startswith("ap ") or "associated press" in lower):
            continue
        ranks: dict[str, int] = {}
        for entry in poll.get("ranks", []):
            if not isinstance(entry, dict):
                continue
            team = entry.get("team") or {}
            if not isinstance(team, dict):
                continue
            team_id = team.get("id")
            try:
                position = int(entry.get("current"))
            except (TypeError, ValueError):
                continue
            if team_id is not None and 1 <= position <= 25:
                ranks[str(team_id)] = position
        if ranks:
            return ranks, name
    return {}, ""


@st.cache_data(ttl=3600, show_spinner=False)
def current_ap_poll() -> tuple[dict[str, int], str]:
    endpoint = ("https://site.api.espn.com/apis/site/v2/sports/"
                "basketball/mens-college-basketball/rankings")
    try:
        response = requests.get(endpoint, timeout=9,
                                headers={"Accept": "application/json",
                                         "User-Agent": "NCAA-MBB-Dashboard/1.0"})
        response.raise_for_status()
        return parse_ap_poll(response.json())
    except (requests.RequestException, ValueError, TypeError, KeyError):
        return {}, ""


def safe_text(value) -> str:
    if value is None:
        return "—"
    try:
        if pd.isna(value):
            return "—"
    except (TypeError, ValueError):
        return "—"
    return escape(str(value))


def team_metadata(game) -> dict:
    """Carry stable IDs and logos for in-cell rendering, not as extra columns."""
    home_id, away_id = str(game["home_id"]), str(game["away_id"])
    selected = str(game.get("predicted_winner_team_id"))
    selected = selected if selected in (home_id, away_id) else ""
    actual = ""
    if completed_legitimate(game):
        actual = home_id if float(game["home_score"]) > float(game["away_score"]) else away_id
    return {"_home_id": home_id, "_away_id": away_id,
            "_home_logo": game.get("home_logo"),
            "_away_logo": game.get("away_logo"),
            "_predicted_winner_id": selected, "_past_prediction_id": selected,
            "_actual_winner_id": actual}


def safe_logo_url(value) -> str:
    """Render only HTTPS logo URLs provided by the existing ESPN data importer."""
    return str(value) if isinstance(value, str) and value.startswith("https://") else ""


def team_cell(name, team_id, logo, ap_ranks: dict[str, int], have_poll: bool) -> str:
    """Render AP rank, ESPN team logo, and name in one visual table cell."""
    rank = ap_ranks.get(str(team_id))
    rank_label = f"#{rank}" if rank is not None else "NR" if have_poll else "—"
    url = safe_logo_url(logo)
    artwork = (f'<img class="mbb-logo" loading="lazy" width="28" height="28" '
               f'src="{escape(url, quote=True)}" alt="">' if url else
               '<span class="mbb-no-logo" aria-hidden="true"></span>')
    return (f'<span class="mbb-team"><span class="mbb-rank">{rank_label}</span>'
            f'{artwork}<span class="mbb-team-name">{safe_text(name)}</span></span>')


def render_game_table(data: pd.DataFrame) -> None:
    """Render team logos/ranks in-place on all game tables, preserving columns.

    Streamlit st.dataframe does not render mixed text+image in one cell, so use a
    horizontally scrollable, escaped HTML table for schedule/result/game pages.
    The confidence-summary tables (no teams) stay as ordinary dataframes.
    """
    if data.empty:
        st.info("No games match the selected filter.")
        return
    ap_ranks, poll_name = current_ap_poll()
    columns = [c for c in data.columns if not c.startswith("_")]
    headers = "".join(f'<th scope="col">{safe_text(c)}</th>' for c in columns)
    rows = []
    for _, record in data.iterrows():
        cells = []
        for column in columns:
            side = {
                "Away": "away", "Home": "home", "Predicted winner": "predicted_winner",
                "Past prediction": "past_prediction", "Actual winner": "actual_winner",
            }.get(column)
            team_id = record.get(f"_{side}_id", "") if side else ""
            if team_id and side and pd.notna(record[column]):
                if str(team_id) == str(record.get("_home_id")):
                    logo = record.get("_home_logo")
                elif str(team_id) == str(record.get("_away_id")):
                    logo = record.get("_away_logo")
                else:
                    logo = ""
                content = team_cell(record[column], team_id, logo, ap_ranks, bool(ap_ranks))
            else:
                content = safe_text(record[column])
            cells.append(f"<td>{content}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    css = """
    <style>
      .mbb-scroll {width:100%;overflow-x:auto;border:1px solid #d4d7dc;border-radius:10px;
                   margin:.5rem 0 1rem;}
      .mbb-games {border-collapse:collapse;width:100%;font-size:.88rem;
                  background:#fff;color:#151515;white-space:nowrap;}
      .mbb-games th {background:#edf0f4;font-weight:700;text-align:left;padding:11px 12px;
                     border-bottom:1px solid #c5cad2;}
      .mbb-games td {padding:10px 12px;border-bottom:1px solid #e3e6eb;vertical-align:middle;}
      .mbb-games tr:last-child td {border-bottom:0;}
      .mbb-games tbody tr:hover {background:#f5f7fa;}
      .mbb-team {display:inline-flex;align-items:center;gap:7px;min-width:145px;}
      .mbb-rank {display:inline-block;min-width:31px;font-weight:750;color:#374151;}
      .mbb-logo {width:28px;height:28px;object-fit:contain;flex:0 0 28px;}
      .mbb-no-logo {display:inline-block;width:28px;flex:0 0 28px;}
      .mbb-team-name {font-weight:600;}
    </style>
    """
    st.markdown(css + '<div class="mbb-scroll"><table class="mbb-games"><thead><tr>'
                + headers + '</tr></thead><tbody>' + "".join(rows)
                + '</tbody></table></div>', unsafe_allow_html=True)
    if ap_ranks:
        st.caption(f"Team labels show latest {poll_name or 'AP Top 25'} rank · ESPN logo · name. "
                   "NR means unranked. Rankings are current, not necessarily from game day.")
    else:
        st.caption("ESPN AP poll is temporarily unavailable; ranks display —. "
                   "Team logos are loaded from the existing database when available.")


def game_tipoff(value) -> datetime | None:
    if value is None or pd.isna(value) or not str(value).strip():
        return None
    try:
        dt = datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(CT)
    except (TypeError, ValueError, OverflowError):
        return None


def local_game_day(row) -> date | None:
    tip = game_tipoff(row["start_time_utc"])
    if tip is not None:
        return tip.date()
    try:
        return date.fromisoformat(str(row["game_date"])[:10])
    except (ValueError, TypeError):
        return None


def only_local_days(frame: pd.DataFrame, first: date, last: date) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    mask = frame.apply(lambda row: (day is not None and first <= day <= last)
                       if (day := local_game_day(row)) is not None else False, axis=1)
    return frame.loc[mask].copy()


def game_time_label(value) -> str:
    dt = game_tipoff(value)
    return dt.strftime("%-I:%M %p") if dt else "TBD"


def is_completed(value) -> bool:
    return str(value).strip().lower() in {"1", "true", "t", "yes"}


def matchup_type(home_net, away_net) -> str:
    h = pd.to_numeric(pd.Series([home_net]), errors="coerce").iloc[0]
    a = pd.to_numeric(pd.Series([away_net]), errors="coerce").iloc[0]
    if pd.isna(h) or pd.isna(a):
        return "Unrated"
    return "Competitive" if abs(float(h) - float(a)) <= 5 else "Noncompetitive"


def current_forecast(row) -> dict:
    """Display the engine's stored future-game forecast or explicit missing status.

    A numeric probability does not turn into a prediction unless the saved team ID
    is one of the fixture's participants and the probability is in [0, 1].
    """
    home_id, away_id = str(row["home_id"]), str(row["away_id"])
    pick = row.get("predicted_winner_team_id")
    raw_probability = row.get("home_win_probability")
    try:
        p = float(raw_probability)
    except (ValueError, TypeError):
        p = float('nan')
    if str(pick) not in (home_id, away_id) or not (0 <= p <= 1):
        return {"Predicted winner": "Not yet predicted", "Confidence": "—",
                "Forecast saved (CT)": "—"}
    selected_home = str(pick) == home_id
    confidence = p if selected_home else 1-p
    asof = game_tipoff(row.get("forecast_created_at"))
    return {"Predicted winner": str(row["home_team"] if selected_home else row["away_team"]),
            "Confidence": f"{confidence*100:.1f}%",
            "Forecast saved (CT)": asof.strftime("%b %d, %I:%M %p") if asof else "—"}


def confidence_band(probability: float) -> str:
    if not pd.notna(probability) or not (0 <= float(probability) <= 1):
        return "Unavailable"
    p = 100 * max(float(probability), 1 - float(probability))
    if p >= 90: return "90–100%"
    if p >= 80: return "80–89%"
    if p >= 70: return "70–79%"
    if p >= 60: return "60–69%"
    return "50–59%"


def completed_legitimate(row) -> bool:
    h = pd.to_numeric(pd.Series([row["home_score"]]), errors="coerce").iloc[0]
    a = pd.to_numeric(pd.Series([row["away_score"]]), errors="coerce").iloc[0]
    if not is_completed(row["completed"]) or pd.isna(h) or pd.isna(a) or h == a:
        return False
    return not ((h, a) in ((0, 2), (2, 0)))


def yesterday_result(row) -> dict:
    """A result can be Correct, Incorrect, Not saved, or Not final; never invent a pick."""
    actual = None
    if completed_legitimate(row):
        actual = str(row["home_id"] if float(row["home_score"]) > float(row["away_score"])
                     else row["away_id"])
    predicted = row.get("predicted_winner_team_id")
    prob = row.get("home_win_probability")
    valid_pick = predicted is not None and pd.notna(predicted) and str(predicted) in {
        str(row["home_id"]), str(row["away_id"])}
    if actual is None:
        state = "Not final"
    elif not valid_pick:
        state = "No saved forecast"
    else:
        state = "Correct" if str(predicted) == actual else "Incorrect"
    winner_name = (row["home_team"] if actual == str(row["home_id"]) else
                   row["away_team"] if actual == str(row["away_id"]) else "—")
    predicted_name = (row["home_team"] if valid_pick and str(predicted)==str(row["home_id"]) else
                      row["away_team"] if valid_pick else "—")
    return {"Result": state, "Actual winner": winner_name,
            "Past prediction": predicted_name,
            "Past confidence": (f"{max(float(prob), 1-float(prob))*100:.1f}%"
                                if valid_pick and pd.notna(prob) and 0 <= float(prob) <= 1 else "—")}


def fixtures(start: date, end: date, season: int) -> pd.DataFrame:
    return only_local_days(run_query(SCHEDULE_SQL, (FORECAST_MODEL, season,
        (start-timedelta(days=1)).isoformat(), (end+timedelta(days=1)).isoformat())), start, end)


def yesterday_fixtures(day: date, season: int) -> pd.DataFrame:
    return only_local_days(run_query(YESTERDAY_SQL, (FORECAST_MODEL, season,
        (day-timedelta(days=1)).isoformat(), (day+timedelta(days=1)).isoformat())), day, day)


def display_exception(where: str, exc: Exception):
    logging.warning("Dashboard read failed at %s (%s)", where, type(exc).__name__)
    st.error(f"Couldn't load {where} from Supabase. Check the DATABASE_URL permissions and required tables.")
    # Never expose raw SQL/connection exceptions; these can contain secrets.


def schedule_page() -> None:
    st.title("🏀 Game Schedule")
    st.caption("NCAA Division I men's basketball · Central Time · refreshed from your ESPN-imported Supabase schedule")
    today = now_ct().date()
    col1, col2 = st.columns(2)
    start = col1.date_input("Starting date", value=today, key="start_date")
    days = col2.selectbox("Show", [1, 3, 7, 14, 21], index=2, format_func=lambda n: f"{n} days")
    end = start + timedelta(days=days-1)
    try:
        data = fixtures(start, end, active_season())
    except Exception as exc:
        display_exception("the schedule", exc)
        return
    if data.empty:
        st.info("No games found in that date range.")
        return
    out = []
    for _, g in data.iterrows():
        finished = is_completed(g["completed"])
        score = (f"{int(g['away_score'])}–{int(g['home_score'])}"
                 if finished and pd.notna(g["away_score"]) and pd.notna(g["home_score"])
                 else "—")
        out.append({"Date (CT)": str(local_game_day(g)), "Time (CT)": game_time_label(g["start_time_utc"]),
                    "Away": g["away_team"], "Home": g["home_team"],
                    **team_metadata(g),
                    "Status": "Final" if finished else str(g.get("status_detail") or "Scheduled"),
                    "Score (away–home)": score})
    st.metric("Games", len(out))
    render_game_table(pd.DataFrame(out))


def yesterday_page() -> None:
    day = now_ct().date() - timedelta(days=1)
    st.title("✅ Yesterday's Prediction Results")
    st.caption(f"{day.strftime('%B %d, %Y')} · Central Time · only forecasts saved before tipoff qualify")
    try:
        games = yesterday_fixtures(day, active_season())
    except Exception as exc:
        display_exception("yesterday's results", exc)
        return
    if games.empty:
        st.info("No games found for yesterday.")
        return
    rows = []
    for _, g in games.iterrows():
        result = yesterday_result(g)
        score = (f"{int(g['away_score'])}–{int(g['home_score'])}"
                 if completed_legitimate(g) else "—")
        rows.append({"Time (CT)": game_time_label(g["start_time_utc"]),
                     "Away": g["away_team"], "Home": g["home_team"],
                     **team_metadata(g),
                     "Final (away–home)": score, **result})
    out = pd.DataFrame(rows)
    wins = int(out["Result"].eq("Correct").sum())
    losses = int(out["Result"].eq("Incorrect").sum())
    not_saved = int(out["Result"].eq("No saved forecast").sum())
    a, b, c, d = st.columns(4)
    a.metric("Correct", wins)
    b.metric("Incorrect", losses)
    c.metric("Past accuracy", f"{wins/(wins+losses)*100:.1f}%" if wins+losses else "—")
    d.metric("No saved forecast", not_saved)
    if not_saved:
        st.caption("Games without a saved prediction are shown separately, not counted as wins or losses.")
    render_game_table(out)


def upcoming_page() -> None:
    st.title("🏀 Upcoming Matchups")
    st.caption("Future NCAA Division I basketball games · Central Time · winner engine forecasts saved before tipoff")
    today = now_ct().date()
    horizon = st.selectbox("Upcoming window", [1, 3, 7, 14, 21], index=2,
                           format_func=lambda n: f"Next {n} day{'s' if n != 1 else ''}")
    end = today + timedelta(days=horizon-1)
    try:
        games = fixtures(today, end, active_season())
    except Exception as exc:
        display_exception("upcoming matchups", exc)
        return
    if games.empty:
        st.info("No upcoming games found in that date range.")
        return
    current = now_ct()
    rows = []
    for _, g in games.iterrows():
        if is_completed(g["completed"]):
            continue
        tip = game_tipoff(g["start_time_utc"])
        if tip is not None and tip <= current:
            continue
        rows.append({"Date (CT)": str(local_game_day(g)),
                     "Time (CT)": game_time_label(g["start_time_utc"]),
                     "Away": g["away_team"], "Home": g["home_team"],
                     **team_metadata(g),
                     **current_forecast(g),
                     "Matchup type": matchup_type(g["home_net"], g["away_net"]),
                     "Game status": str(g.get("status_detail") or "Scheduled")})
    if not rows:
        st.info("No games remain before the end of this window.")
        return
    out = pd.DataFrame(rows)
    filt = st.selectbox("Matchup filter", ["All", "Competitive", "Noncompetitive", "Unrated"])
    if filt != "All":
        out = out[out["Matchup type"].eq(filt)]
    st.metric("Upcoming games", len(out))
    render_game_table(out)
    st.caption("Predicted winner and confidence are taken from the stored winner-v1.0 forecast. "
               "Unpredicted games are explicitly labeled. Competitive means current adjusted net "
               "ratings differ by at most 5 points per 100 possessions. Historical confidence "
               "percentages are not guarantees; saved forecasts do not automatically refresh.")




def today_page() -> None:
    """Every game on today's Central-Time calendar, including saved predictions.

    Deliberately include completed/in-progress fixtures so that the day's saved
    forecasts do not disappear once the game begins or finishes.
    """
    today = now_ct().date()
    st.title("🏀 Today's Games")
    st.caption(f"{today.strftime('%B %d, %Y')} · Central Time · saved pregame winner forecasts")
    try:
        games = fixtures(today, today, active_season())
    except Exception as exc:
        display_exception("today's games", exc)
        return
    if games.empty:
        st.info("No games found for today. Check whether the schedule importer has loaded today's games.")
        return
    rows = []
    for _, game in games.iterrows():
        prediction = current_forecast(game)
        finished = completed_legitimate(game)
        if finished:
            score = f"{int(game['away_score'])}–{int(game['home_score'])}"
            pick = game.get("predicted_winner_team_id")
            if prediction["Predicted winner"] == "Not yet predicted":
                prediction_result = "No saved forecast"
            else:
                actual_id = str(game["home_id"] if float(game["home_score"]) > float(game["away_score"]) else game["away_id"])
                prediction_result = "Correct" if str(pick) == actual_id else "Incorrect"
        else:
            score = "—"
            prediction_result = "Pending" if prediction["Predicted winner"] != "Not yet predicted" else "Not predicted"
        rows.append({
            "Time (CT)": game_time_label(game["start_time_utc"]),
            "Away": game["away_team"],
            "Home": game["home_team"],
            **team_metadata(game),
            **prediction,
            "Matchup type": matchup_type(game["home_net"], game["away_net"]),
            "Game status": ("Final" if finished else str(game.get("status_detail") or "Scheduled")),
            "Score (away–home)": score,
            "Prediction result": prediction_result,
        })
    data = pd.DataFrame(rows)
    total = len(data)
    saved = int(data["Predicted winner"].ne("Not yet predicted").sum())
    competitive = int(data["Matchup type"].eq("Competitive").sum())
    a, b, c = st.columns(3)
    a.metric("Today's games", total)
    b.metric("Saved predictions", f"{saved}/{total}")
    c.metric("Competitive matchups", competitive)
    filt = st.selectbox("Today's matchup filter", ["All", "Competitive", "Noncompetitive", "Unrated"],
                        key="today_matchup_filter")
    if filt != "All":
        data = data.loc[data["Matchup type"].eq(filt)]
    render_game_table(data)
    st.caption("Predicted winner and confidence come from your saved winner-v1.0 forecast in Supabase. "
               "Games without a stored, valid pre-tipoff forecast show Not yet predicted. "
               "Completed games are shown with their result; predictions aren't recalculated after tipoff. "
               "Competitive means a team adjusted-net rating difference of 5 or less.")


def tomorrow_page() -> None:
    """Dedicated tomorrow-only snapshot, using the same descriptors as Upcoming Matchups."""
    tomorrow = now_ct().date() + timedelta(days=1)
    st.title("🏀 Tomorrow's Games")
    st.caption(f"{tomorrow.strftime('%B %d, %Y')} · Central Time · NCAA Division I men's basketball")
    try:
        games = fixtures(tomorrow, tomorrow, active_season())
    except Exception as exc:
        display_exception("tomorrow's games", exc)
        return
    if games.empty:
        st.info("No games found for tomorrow. Check whether the schedule importer has upcoming games.")
        return
    rows = []
    for _, g in games.iterrows():
        if is_completed(g["completed"]):
            continue
        rows.append({"Time (CT)": game_time_label(g["start_time_utc"]),
                     "Away": g["away_team"], "Home": g["home_team"],
                     **team_metadata(g),
                     **current_forecast(g),
                     "Matchup type": matchup_type(g["home_net"], g["away_net"]),
                     "Game status": str(g.get("status_detail") or "Scheduled")})
    if not rows:
        st.info("There are no scheduled, unfinished games tomorrow.")
        return
    out = pd.DataFrame(rows)
    filt = st.selectbox("Tomorrow's matchup filter", ["All", "Competitive", "Noncompetitive", "Unrated"],
                        key="tomorrow_matchup_filter")
    if filt != "All":
        out = out.loc[out["Matchup type"].eq(filt)]
    st.metric("Tomorrow's games", len(out))
    render_game_table(out)
    st.caption("Predicted winner and confidence reflect stored winner-v1.0 forecasts, not an "
               "automatically refreshed estimate. Missing forecasts appear as Not yet predicted. "
               "Competitive means the current adjusted net rating gap is at most 5.")


def load_historical() -> pd.DataFrame:
    if not HISTORY.is_file():
        return pd.DataFrame()
    df = pd.read_csv(HISTORY)
    req = {"group", "confidence_band", "games", "wins", "losses", "accuracy_pct"}
    if not req.issubset(df.columns):
        return pd.DataFrame()
    return df


def saved_grade_rows(data: pd.DataFrame) -> pd.DataFrame:
    """Grouped past, verified-pretip accuracy; never score invalid administrative results."""
    cols = ["Confidence", "Games", "Correct", "Incorrect", "Accuracy (%)"]
    if data.empty:
        return pd.DataFrame(columns=cols)
    d = data.copy()
    d["confidence_band"] = d["home_win_probability"].map(confidence_band)
    d["correct"] = d.apply(lambda r: str(r["predicted_winner_team_id"]) ==
                            str(r["home_id"] if float(r["home_score"]) > float(r["away_score"]) else r["away_id"]), axis=1)
    out=[]
    for band in CONFIDENCE_BANDS:
        subset=d[d["confidence_band"].eq(band)]
        wins=int(subset["correct"].sum()); n=len(subset)
        out.append({"Confidence": band, "Games": n, "Correct": wins,
                    "Incorrect": n-wins, "Accuracy (%)": round(wins/n*100,2) if n else None})
    return pd.DataFrame(out,columns=cols)


def accuracy_page() -> None:
    st.title("📊 Confidence Accuracy")
    st.caption("Historical evaluation and verified saved-before-tipoff results are shown separately.")
    historic_tab, saved_tab = st.tabs(["Historical replay (2025–26)", "Saved pregame results"])
    with historic_tab:
        data = load_historical()
        if data.empty:
            st.error("Missing data/historical_confidence_2026.csv. Install the data file from the ZIP.")
        else:
            opts={"all":"All games", "competitive_net_gap_le5":"Competitive", "noncompetitive_net_gap_gt5":"Noncompetitive"}
            cat=st.selectbox("Historical category", list(opts), format_func=lambda x:opts[x])
            d = data.loc[data["group"].eq(cat)].copy()
            d["confidence_band"] = pd.Categorical(d["confidence_band"], categories=DB_BANDS, ordered=True)
            d = d.sort_values("confidence_band")
            out=pd.DataFrame({"Confidence":d["confidence_band"].astype(str).to_numpy(),
                              "Games":d["games"].astype(int).to_numpy(),
                              "Correct":d["wins"].astype(int).to_numpy(),
                              "Incorrect":d["losses"].astype(int).to_numpy(),
                              "Accuracy (%)":pd.to_numeric(d["accuracy_pct"], errors="coerce").to_numpy()})
            st.dataframe(out, hide_index=True, use_container_width=True)
            total=int(out["Games"].sum()); correct=int(out["Correct"].sum())
            a,b=st.columns(2); a.metric("Historical games",f"{total:,}")
            b.metric("Historical accuracy", f"{100*correct/total:.2f}%" if total else "—")
            if not out.empty:
                st.bar_chart(out.set_index("Confidence")["Accuracy (%)"], y_label="Accuracy (%)")
            st.caption("Historical replay, not independently verified live predictions. "
                       "The confidence groups are the model's past estimates, not guarantees.")
    with saved_tab:
        try:
            data = run_query(SAVED_FORECAST_SQL, (FORECAST_MODEL, active_season()))
        except Exception as exc:
            display_exception("saved prediction accuracy", exc)
            return
        records = saved_grade_rows(data)
        st.dataframe(records, hide_index=True, use_container_width=True)
        if data.empty:
            st.info("No eligible saved pregame predictions have completed yet.")
        else:
            total=int(records["Games"].sum()); correct=int(records["Correct"].sum())
            a,b=st.columns(2); a.metric("Graded saved games",f"{total:,}")
            b.metric("Saved accuracy",f"{100*correct/total:.2f}%" if total else "—")
        st.caption("Only completed games with a valid final score and a forecast saved before tipoff are graded. "
                   "Other games remain outside the scored denominator; this is not full-schedule coverage.")


# st.navigation overrides legacy pages/ menus: six pages, and no other pages.
# Available in Streamlit >= 1.36; keep existing Streamlit Cloud main-file target cloud_dashboard.py.
selection=st.navigation([
    st.Page(schedule_page, title="Game Schedule", icon="📅", default=True),
    st.Page(yesterday_page, title="Yesterday's Results", icon="✅"),
    st.Page(today_page, title="Today’s Games", icon="☀️"),
    st.Page(upcoming_page, title="Upcoming Matchups", icon="🏀"),
    st.Page(tomorrow_page, title="Tomorrow’s Games", icon="📆"),
    st.Page(accuracy_page, title="Confidence Accuracy", icon="📊"),
])
selection.run()
