"""Browser/mobile dashboard for the hosted NCAA men's D1 basketball database.

Streamlit executes this code on its server. DATABASE_URL is never sent to a
browser; data is read from PostgreSQL using parameterized, read-only queries.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

st.set_page_config(page_title="NCAA MBB | Game Schedules & Stats", page_icon="🏀", layout="wide")

CENTRAL = ZoneInfo("America/Chicago")


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
    SELECT t.team_id,t.name,t.abbreviation,t.conference,
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
    GROUP BY t.team_id,t.name,t.abbreviation,t.conference
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


def show_table(frame: pd.DataFrame, columns: list[str] | None = None):
    st.dataframe(frame[columns] if columns else frame, use_container_width=True,
                 hide_index=True)


SCHEDULE_QUERY = """
    SELECT g.game_id, g.game_date, g.start_time_utc, g.status_detail, g.status,
           g.completed, g.away_score, g.home_score,
           a.name AS away, h.name AS home, g.venue
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


def schedule_display(games: pd.DataFrame) -> pd.DataFrame:
    """Make a compact, phone-friendly schedule table without exposing raw IDs."""
    if games.empty:
        return pd.DataFrame(columns=["Time (CT)", "Matchup", "Score", "Status", "Venue"])
    output = pd.DataFrame()
    output["Time (CT)"] = games["start_time_utc"].map(central_tipoff)
    output["Matchup"] = games["away"] + " @ " + games["home"]
    def score(row):
        # A scheduled game is not a 0–0 final, even if ESPN supplies zeros.
        if row["status"] in ("STATUS_SCHEDULED", "STATUS_POSTPONED", "STATUS_CANCELED"):
            return "—"
        if pd.notna(row["away_score"]) and pd.notna(row["home_score"]):
            return f"{int(row['away_score'])}–{int(row['home_score'])}"
        return "—"

    output["Score"] = games.apply(score, axis=1)
    output["Status"] = games["status_detail"].fillna("")
    output.loc[output["Status"].eq(""), "Status"] = games["status"].fillna("TBD")
    output["Venue"] = games["venue"].fillna("—")
    return output


def load_schedule(season: int, start_date: str, end_date: str) -> pd.DataFrame:
    return query(SCHEDULE_QUERY, (season, start_date, end_date))


st.title("🏀 NCAA Men's Basketball Database")
st.caption("Division I • ESPN game statistics • unadjusted, possession-based efficiency • Central Time")
with st.sidebar:
    st.header("Navigation")
    page = st.radio("Section", ["Game schedules", "Team efficiency", "Team explorer",
                                    "Player box scores", "Injuries", "Update status"], label_visibility="collapsed")
    if st.button("Refresh displayed data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

try:
    seasons = query("SELECT DISTINCT season FROM games WHERE season IS NOT NULL ORDER BY season DESC")
    if seasons.empty:
        st.info("No game records yet. After deployment, run the GitHub Actions importer once to populate your database.")
        st.stop()
    season = st.selectbox("Season (ESPN ending year)", seasons["season"].astype(int).tolist())
except Exception:
    st.error("Cannot read cloud database yet. Confirm the database connection string and run the first import in GitHub Actions. Check deployment logs for details.")
    st.stop()

if page == "Game schedules":
    st.subheader("Game schedules")
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
            show_table(schedule_display(today_games))

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
                show_table(schedule_display(day_games))

    with browse_tab:
        dates = query("SELECT DISTINCT game_date FROM games WHERE season=%s "
                      "ORDER BY game_date DESC", (season,))
        if dates.empty:
            st.info("No games imported yet.")
        else:
            date_string = st.selectbox("Choose a date (Central Time)", dates["game_date"].tolist())
            games_on_date = load_schedule(season, date_string, date_string)
            st.caption(f"{len(games_on_date)} games")
            show_table(schedule_display(games_on_date))

elif page == "Team efficiency":
    st.subheader("Team efficiency rankings")
    try:
        ranks = query(RANKINGS_QUERY, (season,))
    except Exception:
        st.error("Couldn't load rankings. Check the hosted app logs.")
        st.stop()
    if ranks.empty:
        st.info("No complete box scores imported for this season. Try running a historical backfill.")
    else:
        ranks.insert(0, "Rank", range(1, len(ranks)+1))
        renames = {"name":"Team", "conference":"Conference", "games":"GP",
                   "wins":"W", "losses":"L", "off_eff":"Off / 100", "def_eff":"Def / 100",
                   "net_eff":"Net / 100", "efg_pct":"eFG %", "tov_pct":"TO %"}
        display = ranks.rename(columns=renames)
        cols = ["Rank","Team","Conference","GP","W","L","Off / 100","Def / 100","Net / 100","eFG %","TO %"]
        show_table(display, cols)
        st.download_button("Download rankings (CSV)", display[cols].to_csv(index=False), "rankings.csv", "text/csv")

elif page == "Team explorer":
    st.subheader("Team explorer")
    teams = query("SELECT team_id,name FROM teams WHERE is_d1=1 ORDER BY name")
    if teams.empty:
        st.info("The Division I team list hasn't been imported yet.")
    else:
        team_name = st.selectbox("Choose a team", teams["name"].tolist())
        team_id = teams.loc[teams["name"]==team_name, "team_id"].iloc[0]
        games = query("""
            SELECT g.game_date, CASE WHEN g.home_team_id=%s THEN a.name ELSE h.name END AS opponent,
                   s.points,s.offensive_efficiency,s.defensive_efficiency,s.net_efficiency
            FROM team_game_stats s JOIN games g ON g.game_id=s.game_id
            JOIN teams h ON h.team_id=g.home_team_id JOIN teams a ON a.team_id=g.away_team_id
            WHERE s.team_id=%s AND g.season=%s AND g.completed=1
            ORDER BY g.game_date,g.game_id
        """, (team_id,team_id,season))
        if games.empty:
            st.info("No complete game box scores for this team yet.")
        else:
            show_table(games.rename(columns={"game_date":"Date","opponent":"Opponent",
                "points":"Points","offensive_efficiency":"Off / 100",
                "defensive_efficiency":"Def / 100","net_efficiency":"Net / 100"}))
            graph = games.set_index("game_date")[["offensive_efficiency","defensive_efficiency"]]
            graph.columns = ["Offensive efficiency", "Defensive efficiency"]
            st.line_chart(graph)

elif page == "Player box scores":
    st.subheader("Player box scores")
    games = query("""
        SELECT g.game_id,g.game_date,a.name AS away,h.name AS home
        FROM games g JOIN teams a ON a.team_id=g.away_team_id
        JOIN teams h ON h.team_id=g.home_team_id
        WHERE g.season=%s AND g.completed=1
        ORDER BY g.game_date DESC,g.start_time_utc DESC
    """, (season,))
    if games.empty:
        st.info("No finished games in the database yet.")
    else:
        games["label"] = games["game_date"] + " • " + games["away"] + " @ " + games["home"] + " (" + games["game_id"] + ")"
        label = st.selectbox("Choose a game", games["label"].tolist())
        game_id = games.loc[games["label"]==label, "game_id"].iloc[0]
        box = query("""
            SELECT t.name AS team, p.name AS player,s.starter,s.minutes,s.points,s.rebounds,
                   s.assists,s.steals,s.blocks,s.turnovers,s.fg_m,s.fg_a,
                   s.three_m,s.three_a,s.ft_m,s.ft_a
            FROM player_game_stats s JOIN players p ON p.player_id=s.player_id
            JOIN teams t ON t.team_id=s.team_id
            WHERE s.game_id=%s ORDER BY t.name,s.points DESC
        """, (game_id,))
        if box.empty:
            st.info("No player box-score records for this game yet.")
        else:
            show_table(box)
            st.download_button("Download box score (CSV)", box.to_csv(index=False), "boxscore.csv", "text/csv")

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
st.caption("Off/100 and Def/100 are based on estimated possessions and are not opponent-adjusted. The ESPN feed is unofficial and can change; refresh times are not guaranteed.")
