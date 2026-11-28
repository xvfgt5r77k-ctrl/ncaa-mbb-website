"""Read-only NCAA men's basketball matchup strength and historical model calibration.

Does not make new game-winner predictions, give betting picks, or publish
matchup-specific prediction probabilities.

Install under pages/ for Streamlit's native multi-page navigation. This page
never changes Supabase data and reads the user's existing tables.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

CENTRAL = ZoneInfo("America/Chicago")
HISTORICAL_CSV = Path(__file__).resolve().parents[1] / "data" / "historical_confidence_2026.csv"
BANDS = ("50-59%", "60-69%", "70-79%", "80-89%", "90-100%")

st.set_page_config(page_title="NCAA MBB | Matchup Analytics", page_icon="🏀", layout="wide")


def db_url() -> str:
    import os
    url = os.getenv("DATABASE_URL", "")
    if not url:
        try:
            url = st.secrets.get("DATABASE_URL", "")
        except (KeyError, FileNotFoundError):
            url = ""
    if not url:
        st.error("Missing DATABASE_URL in your Streamlit secrets.")
        st.stop()
    return url


@st.cache_data(ttl=300, show_spinner=False)
def fetch_rows(season: int, start: str, end: str) -> pd.DataFrame:
    """Descriptive season-to-date strengths; never historical as-of game claims."""
    import psycopg
    from psycopg.rows import dict_row
    sql = """
        SELECT g.game_id::text AS game_id, g.game_date::text AS game_date,
               g.start_time_utc, g.completed,
               h.name AS home_team, a.name AS away_team,
               h.team_id::text AS home_team_id, a.team_id::text AS away_team_id,
               rh.adjusted_net AS home_adjusted_net,
               ra.adjusted_net AS away_adjusted_net
        FROM public.games g
        JOIN public.teams h ON h.team_id=g.home_team_id AND h.is_d1=1
        JOIN public.teams a ON a.team_id=g.away_team_id AND a.is_d1=1
        LEFT JOIN LATERAL (
            SELECT adjusted_net FROM public.team_adjusted_ratings r
            WHERE r.season=g.season AND r.team_id=g.home_team_id
            ORDER BY r.games_played DESC NULLS LAST LIMIT 1
        ) rh ON TRUE
        LEFT JOIN LATERAL (
            SELECT adjusted_net FROM public.team_adjusted_ratings r
            WHERE r.season=g.season AND r.team_id=g.away_team_id
            ORDER BY r.games_played DESC NULLS LAST LIMIT 1
        ) ra ON TRUE
        WHERE g.season=%s AND g.game_date BETWEEN %s AND %s
        ORDER BY g.game_date, g.start_time_utc NULLS LAST, g.game_id
    """
    with psycopg.connect(db_url(), sslmode="require", connect_timeout=15,
                         prepare_threshold=None, row_factory=dict_row) as connection:
        connection.read_only = True
        with connection.cursor() as cur:
            cur.execute(sql, (season, start, end))
            return pd.DataFrame(cur.fetchall())


@st.cache_data(ttl=300, show_spinner=False)
def latest_season() -> int:
    import psycopg
    with psycopg.connect(db_url(), sslmode="require", connect_timeout=15,
                         prepare_threshold=None) as connection:
        connection.read_only=True
        with connection.cursor() as cur:
            cur.execute("SELECT MAX(season) FROM public.games")
            return int(cur.fetchone()[0])


def classify_competitiveness(home_net, away_net, cutoff: float = 5.0) -> str:
    """Existing research convention: adjusted net-gap magnitude <= 5."""
    h = pd.to_numeric(pd.Series([home_net]), errors="coerce").iloc[0]
    a = pd.to_numeric(pd.Series([away_net]), errors="coerce").iloc[0]
    if pd.isna(h) or pd.isna(a):
        return "Rating unavailable"
    return "Competitive" if abs(float(h) - float(a)) <= cutoff else "Noncompetitive"


def descriptive_schedule(raw: pd.DataFrame, now_utc: datetime) -> pd.DataFrame:
    cols = ["Date", "Time (CT)", "Away", "Home", "Adjusted net gap", "Matchup type"]
    if raw.empty:
        return pd.DataFrame(columns=cols)
    frame = raw.copy()
    completed = frame["completed"].astype(str).str.lower().isin(["true", "1", "t"])
    tipoffs = pd.to_datetime(frame["start_time_utc"], utc=True, errors="coerce", format="mixed")
    now = pd.Timestamp(now_utc)
    if now.tzinfo is None:
        raise ValueError("now_utc must be timezone-aware")
    frame = frame.loc[~completed & (tipoffs.isna() | (tipoffs > now))].copy()
    if frame.empty:
        return pd.DataFrame(columns=cols)
    times = pd.to_datetime(frame["start_time_utc"], utc=True, errors="coerce", format="mixed")
    gap = (pd.to_numeric(frame["home_adjusted_net"], errors="coerce")
           - pd.to_numeric(frame["away_adjusted_net"], errors="coerce")).abs()
    out = pd.DataFrame({
        "Date": frame["game_date"].astype(str).to_numpy(),
        "Time (CT)": times.dt.tz_convert(CENTRAL).dt.strftime("%-I:%M %p").fillna("TBD").to_numpy(),
        "Away": frame["away_team"].astype(str).to_numpy(),
        "Home": frame["home_team"].astype(str).to_numpy(),
        "Adjusted net gap": gap.round(1).to_numpy(),
        "Matchup type": [classify_competitiveness(h, a) for h, a in zip(
            frame["home_adjusted_net"], frame["away_adjusted_net"])],
    })
    return out


def historical_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    data = pd.read_csv(path)
    required={"group", "confidence_band", "games", "wins", "losses", "accuracy_pct"}
    if not required.issubset(data.columns):
        return pd.DataFrame()
    return data


st.title("🏀 NCAA MBB Matchup Analytics")
st.caption("Read-only team strength comparison and retrospective model evaluation. No new winner forecasts.")
matchups_tab, history_tab = st.tabs(["Upcoming matchup competitiveness", "Historical model accuracy"])

with matchups_tab:
    st.caption("Competitive = season-to-date adjusted net-efficiency gap of 5 points per 100 possessions or less. "
               "Noncompetitive = gap greater than 5. Missing ratings are shown as unavailable. "
               "These latest ratings are descriptive and can change before tipoff.")
    try:
        season=latest_season()
        lookahead=st.selectbox("Upcoming window", [7,14,30], index=1,
                              format_func=lambda days: f"Next {days} days")
        today=datetime.now(CENTRAL).date()
        scheduled=fetch_rows(season, today.isoformat(), (today+timedelta(days=lookahead)).isoformat())
        output=descriptive_schedule(scheduled, datetime.now(timezone.utc))
        category=st.selectbox("Show matchups", ["All", "Competitive", "Noncompetitive", "Rating unavailable"])
        if category != "All":
            output=output.loc[output["Matchup type"].eq(category)]
        st.metric("Upcoming matchups", len(output))
        if output.empty:
            st.info("No eligible matchups or ratings found for this filter.")
        else:
            st.dataframe(output, hide_index=True, use_container_width=True)
    except Exception as exc:
        st.warning("Couldn't load matchup ratings. Confirm the existing team_adjusted_ratings table is populated and DATABASE_URL has read access.")
        # Avoid accidentally displaying secret-bearing DB exception text.

with history_tab:
    st.caption("2025–26 season; historical replay of 5,634 completed games. "
               "This is not verified live-forecast performance and doesn't predict upcoming games.")
    history=historical_table(HISTORICAL_CSV)
    if history.empty:
        st.info("Historical confidence CSV is missing. Upload data/historical_confidence_2026.csv to enable this chart.")
    else:
        labels={"all":"All historical games", "competitive_net_gap_le5":"Competitive games",
                "noncompetitive_net_gap_gt5":"Noncompetitive games"}
        grp=st.selectbox("Historical group", list(labels), format_func=lambda k:labels[k])
        rows=history.loc[history.group.eq(grp)].copy()
        rows["confidence_band"]=pd.Categorical(rows["confidence_band"],categories=BANDS,ordered=True)
        rows=rows.sort_values("confidence_band")
        rows=rows.rename(columns={"confidence_band":"Predicted winner confidence (historical)",
                                  "games":"Games", "wins":"Correct", "losses":"Incorrect", "accuracy_pct":"Accuracy (%)"})
        st.dataframe(rows[["Predicted winner confidence (historical)","Games","Correct","Incorrect","Accuracy (%)"]],
                     use_container_width=True,hide_index=True)
        chart=rows.set_index("Predicted winner confidence (historical)")["Accuracy (%)"]
        st.bar_chart(chart, y_label="Historically correct (%)")
        if grp=="all":
            st.caption(f"Verified report totals: {int(rows['Games'].sum()):,} games; "
                       f"{int(rows['Correct'].sum()):,} correct; "
                       f"{int(rows['Incorrect'].sum()):,} incorrect.")
