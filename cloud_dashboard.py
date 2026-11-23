"""NCAA Men's Basketball – historical statistics dashboard.

Standalone replacement for the Streamlit entrypoint. The historical stats come
from the *read-only*, precomputed Supabase materialized view. Optional schedules
and team directory use the existing server-side DATABASE_URL if configured.

This file is a streamlined replacement, not a full copy of the previous site's
specialized player-statistics, efficiency, and validation-analysis pages.
No database password, service-role key, or write operation is required.
"""
from __future__ import annotations

import os
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import streamlit as st

st.set_page_config(
    page_title="NCAA MBB | Team Statistics",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# The publishable key is intentionally safe for client-facing, RLS/grant-limited
# use. NEVER put an sb_secret_ key, service-role key, or DB password here.
DEFAULT_SUPABASE_URL = "https://piuiukhgllpakblqbkgi.supabase.co"
DEFAULT_PUBLISHABLE_KEY = "sb_publishable_JHFn15pg1uOHsIlutSlv_w_4LtQ9alu"
VIEW_NAME = "ncaa_mbb_team_dashboard_fast_v1"
SEASONS = (2026, 2025, 2024, 2023, 2022)
CST = ZoneInfo("America/Chicago")


@st.cache_data(ttl=600, show_spinner=False)
def load_team_stats(season: int) -> pd.DataFrame:
    """Fetch one season; <400 rows means PostgREST's default row cap is safe."""
    url = f"{DEFAULT_SUPABASE_URL}/rest/v1/{VIEW_NAME}"
    fields = (
        "season,team_id,team_name,abbreviation,games_analyzed,wins,losses,"
        "win_percentage,avg_points_scored,avg_points_allowed,"
        "avg_scoring_margin,home_games,away_games,neutral_games"
    )
    response = requests.get(
        url,
        params={
            "select": fields,
            "season": f"eq.{int(season)}",
            "order": "avg_scoring_margin.desc.nullslast,team_name.asc",
            "limit": "700",
        },
        headers={
            "apikey": DEFAULT_PUBLISHABLE_KEY,
            "Accept": "application/json",
        },
        timeout=20,
    )
    response.raise_for_status()
    records = response.json()
    if not isinstance(records, list):
        raise ValueError("Supabase returned an unexpected response format")
    df = pd.DataFrame(records)
    if df.empty:
        return df
    numbers = (
        "season", "games_analyzed", "wins", "losses", "win_percentage",
        "avg_points_scored", "avg_points_allowed", "avg_scoring_margin",
        "home_games", "away_games", "neutral_games",
    )
    for col in numbers:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["team_name"] = df["team_name"].fillna("Unknown team")
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def load_all_seasons() -> pd.DataFrame:
    frames = [load_team_stats(yr) for yr in SEASONS]
    frames = [frame for frame in frames if not frame.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def existing_database_url() -> str:
    """Optional private DB connection for original schedule/team directory."""
    url = os.getenv("DATABASE_URL", "")
    if url:
        return url
    try:
        return st.secrets.get("DATABASE_URL", "")
    except (FileNotFoundError, KeyError):
        return ""


@st.cache_data(ttl=300, show_spinner=False)
def private_query(sql: str, params: tuple = ()) -> pd.DataFrame:
    """Read-only, parameterized query. Never print the connection string."""
    database_url = existing_database_url()
    if not database_url:
        raise RuntimeError("No server-side DATABASE_URL configured")
    import psycopg
    from psycopg.rows import dict_row

    with psycopg.connect(
        database_url,
        sslmode="require",
        connect_timeout=12,
        prepare_threshold=None,
        row_factory=dict_row,
    ) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SET TRANSACTION READ ONLY")
            cursor.execute(sql, params)
            return pd.DataFrame(cursor.fetchall())


def label_year(year: int) -> str:
    return f"{year - 1}–{str(year)[-2:]}"


def presentation_table(df: pd.DataFrame) -> pd.DataFrame:
    columns = {
        "team_name": "Team",
        "games_analyzed": "Games",
        "wins": "W",
        "losses": "L",
        "win_percentage": "Win %",
        "avg_points_scored": "Points / Game",
        "avg_points_allowed": "Allowed / Game",
        "avg_scoring_margin": "Scoring Margin",
        "home_games": "Home Games",
        "away_games": "Away Games",
        "neutral_games": "Neutral Games",
    }
    selected = [key for key in columns if key in df.columns]
    return df[selected].rename(columns=columns)


def table_and_download(df: pd.DataFrame, basename: str) -> None:
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.download_button(
        "Download CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name=f"{basename}.csv",
        mime="text/csv",
    )


def team_row(df: pd.DataFrame, team_id: str) -> pd.Series:
    # Coerce to string because REST IDs can be strings or integers.
    return df.loc[df["team_id"].astype(str).eq(team_id)].iloc[0]


def metric_str(value: object, suffix: str = "") -> str:
    if pd.isna(value):
        return "—"
    return f"{float(value):,.2f}{suffix}"


st.markdown(
    """
    <style>
      .stApp {background: #f8fafc;}
      h1, h2, h3 {color: #17263b;}
      [data-testid="stMetric"] {
        background: white; padding: 1rem; border: 1px solid #e2e8f0;
        border-radius: 12px;
      }
      [data-testid="stSidebar"] {background: #f0f4f8;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🏀 NCAA Men's Basketball")
st.caption("Historical NCAA team statistics · ESPN-derived database · Read-only analytics")

with st.sidebar:
    st.header("Dashboard")
    page = st.radio(
        "View",
        [
            "Overview", "Team rankings", "Team explorer", "Compare teams",
            "Historical trends", "Game schedules", "Teams & conferences",
            "Data notes",
        ],
    )
    chosen_season = st.selectbox(
        "Historical season",
        list(SEASONS),
        format_func=label_year,
    )
    if st.button("↻ Refresh displayed statistics"):
        load_team_stats.clear()
        load_all_seasons.clear()
        private_query.clear()
        st.rerun()
    st.caption("Published team totals update only after the Supabase fast view is refreshed.")

if page in (
    "Overview", "Team rankings", "Team explorer", "Compare teams",
    "Historical trends",
):
    try:
        season_df = load_team_stats(chosen_season)
    except (requests.RequestException, ValueError) as exc:
        st.error("Could not retrieve the published NCAA team statistics from Supabase.")
        st.info(
            "Check that the fast materialized view exists and its SELECT grant "
            "for the anonymous role is enabled. Then refresh the page."
        )
        st.caption(f"Error category: {type(exc).__name__}")
        st.stop()
    if season_df.empty:
        st.warning("No team records are available for this season.")
        st.stop()

if page == "Overview":
    st.subheader(f"{label_year(chosen_season)} · Season overview")
    games = int(season_df["games_analyzed"].sum() / 2)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Teams listed", f"{len(season_df):,}")
    c2.metric("Qualified games", f"{games:,}")
    c3.metric("Team-game records", f"{season_df['games_analyzed'].sum():,.0f}")
    c4.metric(
        "Avg. points scored",
        metric_str(
            (season_df["avg_points_scored"] * season_df["games_analyzed"]).sum()
            / max(season_df["games_analyzed"].sum(), 1),
        ),
    )
    st.caption("Qualified games only: both teams had at least five prior usable games.")
    left, right = st.columns([1.3, 1])
    with left:
        st.markdown("#### Largest average scoring margins")
        leaders = season_df.nlargest(15, "avg_scoring_margin")
        st.bar_chart(leaders.set_index("team_name")["avg_scoring_margin"])
    with right:
        st.markdown("#### Season leaders")
        table_and_download(
            presentation_table(season_df.head(15)),
            f"ncaa_leaders_{chosen_season}",
        )

elif page == "Team rankings":
    st.subheader(f"Team rankings · {label_year(chosen_season)}")
    st.caption("Sortable historical summaries; not official NCAA standings or future-game forecasts.")
    col_a, col_b, col_c = st.columns([2, 1.2, 1])
    with col_a:
        name_search = st.text_input("Find a team", placeholder="Iowa, Duke, Houston...")
    with col_b:
        metric = st.selectbox(
            "Sort by",
            ["Avg. margin", "Win percentage", "Points scored", "Points allowed"],
        )
    with col_c:
        min_games = st.slider("Minimum games", 0, 35, 5)
    filtered = season_df[season_df["games_analyzed"] >= min_games].copy()
    if name_search.strip():
        filtered = filtered[
            filtered["team_name"].str.contains(name_search, case=False, regex=False)
        ]
    sort_columns = {
        "Avg. margin": ("avg_scoring_margin", False),
        "Win percentage": ("win_percentage", False),
        "Points scored": ("avg_points_scored", False),
        "Points allowed": ("avg_points_allowed", True),
    }
    col, asc = sort_columns[metric]
    filtered = filtered.sort_values(col, ascending=asc, na_position="last")
    st.caption(f"{len(filtered):,} teams shown")
    table_and_download(presentation_table(filtered), f"ncaa_teams_{chosen_season}")

elif page == "Team explorer":
    st.subheader("Team explorer")
    teams = season_df.sort_values("team_name")
    ids = teams["team_id"].astype(str).tolist()
    names = dict(zip(ids, teams["team_name"].tolist()))
    selected_id = st.selectbox("Select a team", ids, format_func=lambda val: names[val])
    selected = team_row(teams, selected_id)
    st.markdown(f"### {selected['team_name']} · {label_year(chosen_season)}")
    a, b, c, d = st.columns(4)
    a.metric("Record (qualified games)", f"{int(selected['wins'])}–{int(selected['losses'])}")
    b.metric("Win percentage", metric_str(selected["win_percentage"], "%"))
    c.metric("Points scored / game", metric_str(selected["avg_points_scored"]))
    d.metric("Points allowed / game", metric_str(selected["avg_points_allowed"]))
    a, b, c, d = st.columns(4)
    a.metric("Average margin", metric_str(selected["avg_scoring_margin"]))
    b.metric("Home games", int(selected["home_games"]))
    c.metric("Away games", int(selected["away_games"]))
    d.metric("Neutral games", int(selected["neutral_games"]))
    st.markdown("#### Across all available seasons")
    try:
        all_df = load_all_seasons()
        history = all_df[all_df["team_id"].astype(str).eq(selected_id)].copy()
        if not history.empty:
            history = history.sort_values("season")
            display = history[["season", "avg_points_scored", "avg_points_allowed", "avg_scoring_margin"]].copy()
            display["season"] = display["season"].map(label_year)
            st.line_chart(display.set_index("season")[["avg_points_scored", "avg_points_allowed"]])
            table_and_download(presentation_table(history), f"ncaa_team_{selected_id}")
    except (requests.RequestException, ValueError):
        st.warning("Multi-season history could not be loaded right now.")

elif page == "Compare teams":
    st.subheader(f"Compare teams · {label_year(chosen_season)}")
    teams = season_df.sort_values("team_name")
    ids = teams["team_id"].astype(str).tolist()
    names = dict(zip(ids, teams["team_name"].tolist()))
    left, right = st.columns(2)
    with left:
        team_a = st.selectbox("Team A", ids, index=0, format_func=lambda x: names[x])
    with right:
        team_b = st.selectbox(
            "Team B", ids, index=min(1, len(ids) - 1), format_func=lambda x: names[x]
        )
    if team_a == team_b:
        st.info("Choose two different teams for a side-by-side comparison.")
    else:
        a, b = team_row(teams, team_a), team_row(teams, team_b)
        attributes = (
            ("Qualified games", "games_analyzed"),
            ("Wins", "wins"), ("Losses", "losses"),
            ("Win percentage", "win_percentage"),
            ("Points / game", "avg_points_scored"),
            ("Points allowed / game", "avg_points_allowed"),
            ("Scoring margin", "avg_scoring_margin"),
        )
        comparison = pd.DataFrame(
            [
                {"Statistic": label, names[team_a]: a[col], names[team_b]: b[col]}
                for label, col in attributes
            ]
        )
        table_and_download(comparison, f"ncaa_comparison_{chosen_season}")
        chart = pd.DataFrame(
            {
                names[team_a]: [a["avg_points_scored"], a["avg_points_allowed"]],
                names[team_b]: [b["avg_points_scored"], b["avg_points_allowed"]],
            },
            index=["Scored", "Allowed"],
        )
        st.bar_chart(chart)
        st.caption("Comparison is descriptive only and does not predict a head-to-head result.")

elif page == "Historical trends":
    st.subheader("Historical scoring trends")
    try:
        all_df = load_all_seasons()
    except (requests.RequestException, ValueError):
        st.error("Unable to load multi-season statistics.")
        st.stop()
    summary = all_df.groupby("season", as_index=False).apply(
        lambda d: pd.Series(
            {
                "team_count": len(d),
                "qualified_games": d["games_analyzed"].sum() / 2,
                "points_per_team_game": (
                    (d["avg_points_scored"] * d["games_analyzed"]).sum()
                    / max(d["games_analyzed"].sum(), 1)
                ),
            }
        ),
        include_groups=False,
    ).reset_index()
    # pandas' groupby.apply result shape varies by version; keep season as column.
    if "season" not in summary.columns:
        summary["season"] = sorted(all_df["season"].dropna().unique())
    summary = summary.sort_values("season")
    summary["season_label"] = summary["season"].map(label_year)
    st.line_chart(summary.set_index("season_label")["points_per_team_game"])
    table_and_download(
        summary.rename(columns={
            "season_label": "Season", "team_count": "Teams",
            "qualified_games": "Qualified games",
            "points_per_team_game": "Avg. points / team-game",
        })[["Season", "Teams", "Qualified games", "Avg. points / team-game"]],
        "ncaa_historical_scoring",
    )
    st.caption("Statistics represent qualified games, not complete official season records.")

elif page == "Game schedules":
    st.subheader("Game schedules")
    st.caption("Optional: requires the existing private DATABASE_URL in your Streamlit secrets.")
    if not existing_database_url():
        st.info("The optional schedule needs the existing Streamlit DATABASE_URL secret.")
    else:
        try:
            available = private_query(
                "SELECT DISTINCT season FROM games WHERE season IS NOT NULL ORDER BY season DESC"
            )
            years = available["season"].astype(int).tolist()
            schedule_season = st.selectbox(
                "Schedule season", years, format_func=label_year, key="schedule_season"
            )
            today = datetime.now(CST).date()
            start = st.date_input("From", value=today, key="schedule_start")
            end = st.date_input("Through", value=today + timedelta(days=7), key="schedule_end")
            if end < start:
                st.warning("Choose an end date on or after the start date.")
            elif (end - start).days > 31:
                st.warning("Select a date range of 31 days or less.")
            else:
                schedule = private_query(
                    """
                    SELECT g.game_date, g.start_time_utc, a.name AS away_team,
                           h.name AS home_team, g.away_score, g.home_score,
                           g.status_detail, g.venue
                    FROM games g
                    JOIN teams a ON a.team_id = g.away_team_id
                    JOIN teams h ON h.team_id = g.home_team_id
                    WHERE g.season = %s AND g.game_date BETWEEN %s AND %s
                    ORDER BY g.game_date, g.start_time_utc NULLS LAST, g.game_id
                    LIMIT 500
                    """,
                    (schedule_season, start.isoformat(), end.isoformat()),
                )
                if schedule.empty:
                    st.info("No imported games are available in the selected date range.")
                else:
                    table_and_download(schedule, "ncaa_schedule")
        except Exception as exc:
            st.warning("Schedules are unavailable through the private database connection.")
            st.caption(f"Error category: {type(exc).__name__}")

elif page == "Teams & conferences":
    st.subheader("Teams & conferences")
    if not existing_database_url():
        st.info("This optional directory needs the existing Streamlit DATABASE_URL secret.")
    else:
        try:
            directory = private_query(
                """SELECT team_id, name, abbreviation, conference
                   FROM teams WHERE is_d1 = 1 ORDER BY name"""
            )
            search = st.text_input("Search schools or conferences")
            if search:
                mask = (
                    directory["name"].fillna("").str.contains(search, case=False, regex=False)
                    | directory["conference"].fillna("").str.contains(search, case=False, regex=False)
                )
                directory = directory.loc[mask]
            st.caption(f"{len(directory)} teams")
            table_and_download(directory, "ncaa_team_directory")
        except Exception as exc:
            st.warning("The team directory could not be loaded.")
            st.caption(f"Error category: {type(exc).__name__}")

elif page == "Data notes":
    st.subheader("About this dataset")
    st.markdown(
        """
        **What is included**

        - ESPN-derived men's NCAA basketball game data for seasons ending 2022–2026.
        - Qualified games: both teams had at least five prior usable game histories.
        - Final scores, average scoring, records from those qualified games, and game-location counts.
        - A previously calculated, read-only Supabase materialized view.

        **What is not included**

        - These records are not complete official season standings; early games are omitted.
        - The dashboard does not predict future winners, scores, or outcomes.
        - Refreshing the website's cache **does not** refresh the materialized view
          in Supabase. After importing new records, an administrator must run
          `REFRESH MATERIALIZED VIEW public.ncaa_mbb_team_dashboard_fast_v1;`.
        - Historical data consistency checks do not independently verify every
          original source record.

        **Privacy and access**

        - The site uses only a Supabase publishable key for its public stats view.
        - Server-side database credentials, if provided through Streamlit secrets,
          are never placed in browser code or printed on the page.
        - No visitor-facing write operations are performed.
        """
    )

st.divider()
st.caption(
    "NCAA MBB historical analysis · Season numbers are ESPN season ending years · "
    "Data updates depend on the existing importer and Supabase view refresh."
)
