"""App-managed dollar-value ledger with cumulative charts.

Entries are recorded only through the Streamlit form and persist in Supabase.
Graph views never alter data, and all views use the same underlying ledger.
"""
from __future__ import annotations

from datetime import date, datetime, time, timedelta
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
import hmac
import os
from zoneinfo import ZoneInfo

import pandas as pd

LOCAL_TZ = ZoneInfo("America/Chicago")
TABLE = "public.mbb_dollar_tracker_entries"
CENT = Decimal("0.01")


def dollar_cents(value) -> int:
    """Convert a user-entered decimal to exact cents (without float drift)."""
    try:
        decimal_value = Decimal(str(value))
        if not decimal_value.is_finite():
            raise ValueError("Dollar amount must be finite.")
        return int((decimal_value.quantize(CENT, rounding=ROUND_HALF_UP) * 100))
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise ValueError("Please enter a valid dollar amount.") from exc


def money(cents: int) -> str:
    sign = "-" if cents < 0 else ""
    return f"{sign}${abs(int(cents)) / 100:,.2f}"


def normalize_entries(frame: pd.DataFrame) -> pd.DataFrame:
    """Oldest to newest with exact cumulative cents and local timestamps."""
    columns = ["id", "timestamp_utc", "timestamp_local", "delta_cents", "balance_cents", "note"]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    required = {"id", "delta_amount", "created_at", "note"}
    if not required.issubset(frame.columns):
        raise ValueError("Incomplete dollar-tracker ledger data.")
    df = frame.copy()
    df["timestamp_utc"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True)
    if df["timestamp_utc"].isna().any():
        raise ValueError("Tracker contains invalid dates.")
    df["id"] = df["id"].astype(int)
    df = df.sort_values(["timestamp_utc", "id"], kind="stable").reset_index(drop=True)
    df["delta_cents"] = df["delta_amount"].apply(dollar_cents).astype("int64")
    df["balance_cents"] = df["delta_cents"].cumsum()
    df["timestamp_local"] = df["timestamp_utc"].dt.tz_convert(LOCAL_TZ)
    return df[columns]


def history_chart_points(entries: pd.DataFrame, view: str, selected_day: date | None = None,
                         today: date | None = None) -> pd.DataFrame:
    """Return [Time, Value ($)] with a single cumulative dollar balance source.

    Hourly displays end-of-hour cumulative values for the selected local day,
    including the balance carried in at midnight. Other grouped views display
    end-of-period cumulative values at the start of their labelled periods.
    """
    if today is None:
        today = datetime.now(LOCAL_TZ).date()
    empty = pd.DataFrame(columns=["Time", "Value ($)"])
    if view == "Hourly (one day)":
        day = selected_day if selected_day is not None else today
        start = pd.Timestamp(datetime.combine(day, time.min), tz=LOCAL_TZ)
        end = pd.Timestamp(datetime.combine(day + timedelta(days=1), time.min), tz=LOCAL_TZ)
        # Includes 23/25-hour DST days correctly, without losing event amounts.
        ticks = pd.date_range(start=start, end=end, freq="h", inclusive="left")
        if entries.empty:
            vals = pd.Series(0, index=ticks, dtype="int64")
        else:
            previous = int(entries.loc[entries["timestamp_local"] < start, "delta_cents"].sum())
            during = entries.loc[(entries["timestamp_local"] >= start) &
                                 (entries["timestamp_local"] < end)]
            vals = during.set_index("timestamp_local")["delta_cents"].resample("h").sum()
            vals = vals.reindex(ticks, fill_value=0).cumsum() + previous
        return pd.DataFrame({"Time": ticks, "Value ($)": vals.to_numpy(dtype="int64") / 100.0})

    if entries.empty:
        return empty
    if view == "All time":
        start = entries.iloc[0]["timestamp_local"] - pd.Timedelta(seconds=1)
        return pd.DataFrame({"Time": [start, *entries["timestamp_local"].tolist()],
                             "Value ($)": [0.0, *(entries["balance_cents"] / 100.0).tolist()]})

    freq = {"Daily": "D", "Weekly": "W-SUN", "Monthly": "M"}.get(view)
    if freq is None:
        raise ValueError("Unknown date grouping.")
    # Periods use Central Time as calendar boundaries, independent of UTC midnight.
    times = entries["timestamp_local"].dt.tz_localize(None)
    periods = times.dt.to_period(freq)
    last_period = pd.Period(max(today, entries.iloc[-1]["timestamp_local"].date()), freq=freq)
    all_periods = pd.period_range(start=periods.iloc[0], end=last_period, freq=freq)
    sums = entries.groupby(periods)["delta_cents"].sum().reindex(all_periods, fill_value=0)
    cumulative = sums.cumsum().astype("int64")
    labels = [pd.Timestamp(per.start_time).tz_localize(LOCAL_TZ) for per in all_periods]
    return pd.DataFrame({"Time": labels, "Value ($)": cumulative.to_numpy() / 100.0})


def get_secret(name: str) -> str:
    value = os.environ.get(name, "")
    if not value:
        import streamlit as st
        try:
            value = str(st.secrets.get(name, ""))
        except (FileNotFoundError, KeyError):
            value = ""
    return value


def db_url() -> str:
    return get_secret("TRACKER_DATABASE_URL") or get_secret("DATABASE_URL")


def read_entries() -> pd.DataFrame:
    """Read-only connection, separate from the main dashboard's query cache."""
    import psycopg
    from psycopg.rows import dict_row
    with psycopg.connect(db_url(), sslmode="require", connect_timeout=15,
                         prepare_threshold=None, row_factory=dict_row) as conn:
        conn.read_only = True
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, delta_amount::text AS delta_amount,
                       created_at, note
                FROM public.mbb_dollar_tracker_entries
                ORDER BY created_at, id
            """)
            return pd.DataFrame(cur.fetchall(), columns=["id", "delta_amount", "created_at", "note"])


def insert_entry(delta_cents: int, note: str) -> None:
    """Parameterized insert. The only code path that changes tracker data."""
    import psycopg
    if not delta_cents:
        raise ValueError("The change must be different from zero.")
    if abs(delta_cents) > 999_999_999_999_99:
        raise ValueError("The change exceeds the permitted amount.")
    amount = Decimal(delta_cents) / Decimal(100)
    with psycopg.connect(db_url(), sslmode="require", connect_timeout=15,
                         prepare_threshold=None) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO public.mbb_dollar_tracker_entries (delta_amount, note)
                VALUES (%s, %s)
            """, (amount, note[:180]))
        conn.commit()


def dollar_tracker_page() -> None:
    import streamlit as st
    import altair as alt

    st.title("💵 Dollar Tracker")
    st.caption("One saved value history · All time / hourly / daily / weekly / monthly · Central Time")
    try:
        raw = read_entries()
        entries = normalize_entries(raw)
    except Exception:
        st.error("Dollar Tracker could not load. Run the included SQL setup once, then check "
                 "the Streamlit database connection permissions.")
        return
    current_cents = int(entries.iloc[-1]["balance_cents"]) if len(entries) else 0
    a, b = st.columns(2)
    a.metric("Current value", money(current_cents))
    b.metric("Recorded changes", f"{len(entries):,}")

    view = st.selectbox("Graph view", ["All time", "Hourly (one day)", "Daily", "Weekly", "Monthly"],
                        key="dollar_tracker_view")
    chosen = None
    if view == "Hourly (one day)":
        today = datetime.now(LOCAL_TZ).date()
        default_date = (entries.iloc[-1]["timestamp_local"].date()
                        if len(entries) else today)
        chosen = st.date_input("Choose day (Central Time)", value=default_date,
                               key="dollar_tracker_day")
    points = history_chart_points(entries, view, selected_day=chosen)
    if points.empty:
        st.info("No entries yet. Add your first value change below to start your graph.")
    else:
        fmt = "%I %p" if view == "Hourly (one day)" else "%b %d, %Y"
        chart = (alt.Chart(points).mark_line(interpolate="step-after", point=True)
                 .encode(x=alt.X("Time:T", title="Date / time (Central)",
                                 axis=alt.Axis(format=fmt)),
                         y=alt.Y("Value ($):Q", title="Value ($)",
                                 axis=alt.Axis(format="$,.2f"), scale=alt.Scale(zero=False)),
                         tooltip=[alt.Tooltip("Time:T", title="Date / time"),
                                  alt.Tooltip("Value ($):Q", format="$,.2f", title="Value")])
                 .properties(height=360))
        st.altair_chart(chart, use_container_width=True)
        st.caption("Each entry affects every graph. Hourly and grouped charts show the balance "
                   "at the end of each period, including changes from earlier periods.")

    st.subheader("Change the value")
    password = get_secret("TRACKER_EDIT_PASSWORD")
    if not password:
        st.warning("Editing is locked. Add TRACKER_EDIT_PASSWORD to Streamlit app secrets "
                   "to enable in-app changes; no password is stored in the code.")
        return

    if not st.session_state.get("dollar_tracker_unlocked", False):
        with st.form("tracker_unlock_form"):
            entered = st.text_input("Edit password", type="password")
            unlock = st.form_submit_button("Unlock editing")
        if unlock:
            if hmac.compare_digest(entered, password):
                st.session_state["dollar_tracker_unlocked"] = True
                st.rerun()
            else:
                st.error("Incorrect edit password.")
        return

    st.caption("Editing unlocked for this session. All adjustments are saved to Supabase.")
    if st.button("Lock editing", key="tracker_relock"):
        st.session_state["dollar_tracker_unlocked"] = False
        st.rerun()

    with st.form("dollar_tracker_new_change", clear_on_submit=False):
        method = st.radio("Change type", ["Increase value", "Decrease value", "Set exact value"],
                          horizontal=True)
        if method == "Set exact value":
            number = st.number_input("New value ($)", value=current_cents / 100,
                                     step=1.0, format="%.2f")
        else:
            number = st.number_input("Change amount ($)", min_value=0.01,
                                     value=1.0, step=1.0, format="%.2f")
        note = st.text_input("Description (optional)", max_chars=180)
        submitted = st.form_submit_button("Save value change", type="primary")
    if submitted:
        try:
            cents = dollar_cents(number)
            delta = cents - current_cents if method == "Set exact value" else (
                cents if method == "Increase value" else -cents)
            insert_entry(delta, note.strip())
        except ValueError as exc:
            st.error(str(exc))
        except Exception:
            st.error("Could not save this change. Check database write permissions, then try again. "
                     "The chart has not been updated.")
        else:
            st.success("Value change saved to your database.")
            st.rerun()

    if not entries.empty:
        with st.expander("View saved entries"):
            visible = entries.iloc[::-1].copy()
            visible["When (Central)"] = visible["timestamp_local"].dt.strftime("%Y-%m-%d %I:%M %p")
            visible["Change"] = visible["delta_cents"].map(money)
            visible["Value after"] = visible["balance_cents"].map(money)
            visible = visible.rename(columns={"note": "Description"})
            st.dataframe(visible[["When (Central)", "Change", "Value after", "Description"]],
                         hide_index=True, use_container_width=True)
