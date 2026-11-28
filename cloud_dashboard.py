"""Four-page NCAA D-I basketball schedule and retrospective model evaluation dashboard.

This is the Streamlit entrypoint (replace the repository-root cloud_dashboard.py).
Database credentials stay server-side. No live matchup-specific winning picks
or prediction probabilities are published here.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo
import os
import logging

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
       hr.adjusted_net AS home_net, ar.adjusted_net AS away_net
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
WHERE g.season = %s AND g.game_date BETWEEN %s AND %s
ORDER BY g.game_date, g.start_time_utc NULLS LAST, g.game_id
"""

# A missing forecast is NOT a wrong forecast. Keep every game in the result list.
YESTERDAY_SQL = """
SELECT g.game_id::text AS game_id, g.season, g.game_date::text AS game_date,
       g.start_time_utc::text AS start_time_utc, g.completed,
       g.home_score, g.away_score, g.status_detail,
       ht.name AS home_team, at.name AS away_team,
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


def rating_gap(home_net, away_net):
    h, a = pd.to_numeric(pd.Series([home_net, away_net]), errors="coerce")
    return round(abs(float(h) - float(a)), 1) if pd.notna(h) and pd.notna(a) else None


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
    return only_local_days(run_query(SCHEDULE_SQL, (season,
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
                    "Status": "Final" if finished else str(g.get("status_detail") or "Scheduled"),
                    "Score (away–home)": score})
    st.metric("Games", len(out))
    st.dataframe(pd.DataFrame(out), hide_index=True, use_container_width=True)


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
    st.dataframe(out, hide_index=True, use_container_width=True)


def today_page() -> None:
    st.title("📅 Today's Matchups")
    st.caption("Current team-strength comparison only · Central Time")
    st.info("This view shows every matchup and whether current ratings classify it as competitive. "
            "It does not publish live winner picks or matchup-specific prediction confidence.")
    today = now_ct().date()
    try:
        games = fixtures(today, today, active_season())
    except Exception as exc:
        display_exception("today's matchups", exc)
        return
    if games.empty:
        st.info("No games found today.")
        return
    rows=[]
    for _, g in games.iterrows():
        rows.append({"Time (CT)": game_time_label(g["start_time_utc"]),
                     "Away": g["away_team"], "Home": g["home_team"],
                     "Matchup type": matchup_type(g["home_net"], g["away_net"]),
                     "Rating gap": rating_gap(g["home_net"], g["away_net"]),
                     "Game status": "Final" if is_completed(g["completed"]) else str(g.get("status_detail") or "Scheduled")})
    out=pd.DataFrame(rows)
    filt = st.selectbox("Matchup filter", ["All", "Competitive", "Noncompetitive", "Unrated"])
    if filt != "All":
        out = out[out["Matchup type"].eq(filt)]
    st.metric("Listed games", len(out))
    st.dataframe(out, hide_index=True, use_container_width=True)
    st.caption("Competitive means |home adjusted net − away adjusted net| ≤ 5 points per 100 possessions. "
               "Ratings are current descriptive values, not frozen pre-tipoff forecasts. "
               "Games with unavailable ratings are marked Unrated.")


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


# st.navigation overrides legacy pages/ menus: four pages, and no other pages.
# Available in Streamlit >= 1.36; keep existing Streamlit Cloud main-file target cloud_dashboard.py.
selection=st.navigation([
    st.Page(schedule_page, title="Game Schedule", icon="📅", default=True),
    st.Page(yesterday_page, title="Yesterday's Results", icon="✅"),
    st.Page(today_page, title="Today's Matchups", icon="🏀"),
    st.Page(accuracy_page, title="Confidence Accuracy", icon="📊"),
])
selection.run()
