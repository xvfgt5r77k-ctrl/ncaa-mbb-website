"""NCAA MBB V2.0 feature store.

Reconstructs prior-calendar-day team/player features for completed historical
matchups, and freezes new prospective snapshots for future scheduled games.
No wagers, live-game picks, or existing projection-table writes.

IMPORTANT: historical_replay is NOT a forecast saved before tipoff.
Actual injury/availability data is NOT inferred from absent box-score rows.
"""
from __future__ import annotations

import argparse
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
import json
import math
import os
from pathlib import Path
from typing import Any

VERSION = "v2.0"
BASE_EFF = 100.0
EFF_PRIOR_GAMES = 8.0
PLAYER_PRIOR_MINUTES = 100.0
PLAYER_PRIOR_POINTS_PER_MIN = 0.30
RECENT_PLAYER_GAMES = 5
MAX_PLAYER_ROWS_PER_TEAM = 12

TEAM_ALIASES = {
    "points": ("points", "pts"),
    "possessions": ("possessions", "possession_estimate"),
    "fg_m": ("fg_m", "field_goals_made"),
    "fg_a": ("fg_a", "field_goals_attempted"),
    "three_m": ("three_m", "three_point_field_goals_made", "three_pt_made"),
    "three_a": ("three_a", "three_point_field_goals_attempted", "three_pt_attempted"),
    "turnovers": ("turnovers", "turnover", "tov", "to_count"),
    "off_rebounds": ("off_rebounds", "offensive_rebounds", "orb", "oreb"),
    "ft_a": ("ft_a", "fta", "free_throws_attempted", "free_throw_attempts"),
}
PLAYER_ALIASES = {
    "minutes": ("minutes", "minutes_played", "min"),
    "points": ("points", "pts"),
}
TEAM_FIELDS = (
    "game_id", "team_id", "season", "game_day", "tipoff_utc", "snapshot_kind", "feature_version",
    "games_before", "elo_rating", "adjusted_offense_per100", "adjusted_defense_per100",
    "adjusted_net_per100", "opponent_elo_mean", "estimated_possessions",
    "average_points_for", "average_points_against", "effective_fg_pct", "three_point_rate",
    "turnover_per_possession", "free_throw_rate", "rest_days", "prior_player_box_games",
    "player_coverage", "expected_rotation_points_per40", "rotation_top3_minutes_share",
    "rotation_continuity", "quality_flags"
)
PLAYER_FIELDS = (
    "game_id", "team_id", "player_id", "season", "game_day", "tipoff_utc",
    "snapshot_kind", "feature_version", "prior_games_seen", "prior_minutes",
    "previous_five_expected_minutes", "smoothed_points_per_minute", "prior_box_score_only"
)


def number(x: Any) -> float | None:
    try:
        z = float(x)
        return z if math.isfinite(z) else None
    except (TypeError, ValueError, OverflowError):
        return None


def parse_minutes(value: Any) -> float | None:
    if value is None:
        return None
    try:
        s = str(value).strip()
        if ":" in s:
            m, sec = s.split(":", 1)
            minutes = float(m) + float(sec) / 60.0
        else:
            minutes = float(s)
        return minutes if 0 < minutes <= 60 else None
    except (ValueError, TypeError):
        return None


def game_day(value: Any) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


def tipoff(value: Any) -> datetime | None:
    if not value:
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        try:
            dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            return None
    if dt.tzinfo is None:
        # A timezone-free string cannot establish a trustworthy cutoff.
        return None
    return dt.astimezone(timezone.utc)


def mean(sum_val: float, n: int) -> float | None:
    return round(sum_val / n, 6) if n else None


def pct(num: float, den: float) -> float | None:
    return round(num / den, 6) if den > 0 else None


@dataclass
class Player:
    minutes: float = 0.0
    points: float = 0.0
    games_seen: int = 0


@dataclass
class Team:
    n: int = 0
    elo: float = 1500.0
    off_adjusted_sum: float = 0.0
    def_adjusted_sum: float = 0.0
    efficiency_games: int = 0
    opponent_elo_sum: float = 0.0
    points_for: float = 0.0
    points_against: float = 0.0
    poss_sum: float = 0.0
    poss_games: int = 0
    fgm: float = 0.0
    fga: float = 0.0
    three_m: float = 0.0
    three_a: float = 0.0
    tov: float = 0.0
    poss_for_tov: float = 0.0
    fta: float = 0.0
    fta_fga: float = 0.0
    last_day: date | None = None
    players: dict[str, Player] = field(default_factory=dict)
    recent_rosters: deque = field(default_factory=lambda: deque(maxlen=RECENT_PLAYER_GAMES))
    player_box_games: int = 0

    def eff(self) -> tuple[float, float]:
        den = EFF_PRIOR_GAMES + self.efficiency_games
        off = (BASE_EFF * EFF_PRIOR_GAMES + self.off_adjusted_sum) / den
        defense = (BASE_EFF * EFF_PRIOR_GAMES + self.def_adjusted_sum) / den
        return off, defense

    def rotation(self) -> tuple[float | None, float | None, float | None, list[tuple]]:
        recent = list(self.recent_rosters)
        if not recent:
            return None, None, None, []
        planned: dict[str, float] = defaultdict(float)
        for roster in recent:
            for pid, mins in roster.items():
                planned[pid] += mins / len(recent)
        total = sum(planned.values())
        if total <= 0:
            return None, None, None, []
        ranked = sorted(planned.items(), key=lambda x: (-x[1], x[0]))
        top3_share = sum(v for _, v in ranked[:3]) / total
        weighted_rate = 0.0
        player_rows = []
        for pid, expected_mins in ranked[:MAX_PLAYER_ROWS_PER_TEAM]:
            p = self.players[pid]
            rate = ((p.points + PLAYER_PRIOR_MINUTES * PLAYER_PRIOR_POINTS_PER_MIN) /
                    (p.minutes + PLAYER_PRIOR_MINUTES))
            player_rows.append((pid, p.games_seen, p.minutes, expected_mins, rate))
        for pid, exp_mins in planned.items():
            p = self.players[pid]
            rate = ((p.points + PLAYER_PRIOR_MINUTES * PLAYER_PRIOR_POINTS_PER_MIN) /
                    (p.minutes + PLAYER_PRIOR_MINUTES))
            weighted_rate += exp_mins * rate
        continuity = None
        if len(recent) >= 2:
            prior, current = recent[-2], recent[-1]
            den = sum(current.values())
            if den > 0:
                continuity = sum(min(minutes, prior.get(pid, 0.0))
                                 for pid, minutes in current.items()) / den
        return 40.0 * weighted_rate / total, top3_share, continuity, player_rows


def d1_games_sorted(games: list[dict]) -> list[dict]:
    normalized = []
    ids: set[str] = set()
    for g in games:
        gid = str(g["game_id"])
        if gid in ids:
            raise RuntimeError(f"Duplicate game_id in source: {gid}")
        ids.add(gid)
        item = dict(g)
        item["game_id"] = gid
        item["home_team_id"] = str(g["home_team_id"])
        item["away_team_id"] = str(g["away_team_id"])
        item["game_day"] = game_day(g["game_date"])
        item["tipoff_utc"] = tipoff(g.get("start_time_utc"))
        item["season"] = int(g["season"])
        item["completed"] = str(g.get("completed", "0")).lower() in ("1", "true", "t")
        normalized.append(item)
    return sorted(normalized, key=lambda x: (x["season"], x["game_day"], x["game_id"]))


def valid_score(g: dict) -> bool:
    h, a = number(g.get("home_score")), number(g.get("away_score"))
    return (g["completed"] and h is not None and a is not None and h >= 0 and a >= 0
            and h != a and (h, a) not in ((0.0, 2.0), (2.0, 0.0)))


def choose_kind(g: dict, now: datetime) -> str | None:
    if g["completed"]:
        return "historical_replay"
    when = g["tipoff_utc"]
    if when and when > now + timedelta(minutes=10):
        return "prospective"
    return None


def feature_values(team: Team, day: date) -> dict:
    off, defense = team.eff()
    rest = (day - team.last_day).days - 1 if team.last_day is not None else None
    rot_pts, rot_top3, rot_cont, athletes = team.rotation()
    if team.player_box_games == 0:
        coverage = "unavailable"
    elif team.player_box_games >= 3 and team.player_box_games >= 0.7 * max(team.n, 1):
        coverage = "sufficient"
    else:
        coverage = "partial"
    return {
        "games_before": team.n,
        "elo_rating": round(team.elo, 5),
        "adjusted_offense_per100": round(off, 5),
        "adjusted_defense_per100": round(defense, 5),
        "adjusted_net_per100": round(off - defense, 5),
        "opponent_elo_mean": mean(team.opponent_elo_sum, team.n),
        "estimated_possessions": mean(team.poss_sum, team.poss_games),
        "average_points_for": mean(team.points_for, team.n),
        "average_points_against": mean(team.points_against, team.n),
        "effective_fg_pct": pct(team.fgm + 0.5 * team.three_m, team.fga),
        "three_point_rate": pct(team.three_a, team.fga),
        "turnover_per_possession": pct(team.tov, team.poss_for_tov),
        "free_throw_rate": pct(team.fta, team.fta_fga),
        "rest_days": max(0, min(30, rest)) if rest is not None else None,
        "prior_player_box_games": team.player_box_games,
        "player_coverage": coverage,
        "expected_rotation_points_per40": round(rot_pts, 6) if rot_pts is not None else None,
        "rotation_top3_minutes_share": round(rot_top3, 6) if rot_top3 is not None else None,
        "rotation_continuity": round(rot_cont, 6) if rot_cont is not None else None,
        "player_rows": athletes,
    }


def get_possessions(box: dict) -> float | None:
    p = number(box.get("possessions"))
    if p is not None and 30 <= p <= 130:
        return p
    fga = number(box.get("fg_a"))
    tov = number(box.get("turnovers"))
    orebs = number(box.get("off_rebounds"))
    fta = number(box.get("ft_a"))
    if None not in (fga, tov, orebs, fta):
        estimate = fga - orebs + tov + 0.475 * fta
        return estimate if 30 <= estimate <= 130 else None
    return None


def player_roster_from_game(player_rows: list[dict]) -> dict[str, tuple[float, float]] | None:
    roster = {}
    for row in player_rows:
        mins = parse_minutes(row.get("minutes"))
        pts = number(row.get("points"))
        if mins is None or pts is None or pts < 0:
            continue
        pid = str(row["player_id"])
        if pid in roster:
            return None  # inconsistent duplicate player box
        roster[pid] = (mins, pts)
    combined = sum(v[0] for v in roster.values())
    return roster if len(roster) >= 5 and 150 <= combined <= 300 else None


def generate(games: list[dict], boxes: dict[tuple[str,str], dict],
             player_boxes: dict[tuple[str,str], list[dict]],
             now: datetime | None = None) -> tuple[list[dict], list[dict], dict]:
    """Build snapshots BEFORE applying all results from that calendar day.

    Changing any game's final score or box must not change its own pregame
    feature row. Same-day matches also cannot see earlier same-day results.
    """
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    source = d1_games_sorted(games)
    team_out, player_out = [], []
    statuses = defaultdict(int)
    from itertools import groupby
    for season, season_group in groupby(source, key=lambda x: x["season"]):
        states: dict[str, Team] = defaultdict(Team)
        for day, day_group in groupby(season_group, key=lambda x: x["game_day"]):
            today = list(day_group)
            updates: list[tuple] = []
            for g in today:
                gid, h_id, a_id = g["game_id"], g["home_team_id"], g["away_team_id"]
                hst, ast = states[h_id], states[a_id]
                kind = choose_kind(g, now)
                if kind is None:
                    statuses["unfinished_or_unscheduled_skipped"] += 1
                else:
                    for tid, st in ((h_id, hst), (a_id, ast)):
                        f = feature_values(st, day)
                        flags = []
                        if f["games_before"] < 5:
                            flags.append("under_five_prior_games")
                        if f["player_coverage"] != "sufficient":
                            flags.append("player_history_incomplete")
                        if f["estimated_possessions"] is None:
                            flags.append("no_prior_possessions")
                        if kind == "historical_replay":
                            flags.append("reconstructed_not_saved_pretipoff")
                        if kind == "prospective" and g["tipoff_utc"] is None:
                            raise RuntimeError("prospective snapshot missing tipoff")
                        team_out.append({
                            "game_id": gid, "team_id": tid, "season": season,
                            "game_day": day, "tipoff_utc": g["tipoff_utc"],
                            "snapshot_kind": kind, "feature_version": VERSION,
                            **{k: v for k, v in f.items() if k != "player_rows"},
                            "quality_flags": flags,
                        })
                        if f["player_coverage"] == "sufficient":
                            for pid, seen, mins, expected, rate in f["player_rows"]:
                                player_out.append({
                                    "game_id": gid, "team_id": tid, "player_id": pid,
                                    "season": season, "game_day": day,
                                    "tipoff_utc": g["tipoff_utc"], "snapshot_kind": kind,
                                    "feature_version": VERSION,
                                    "prior_games_seen": seen, "prior_minutes": round(mins, 5),
                                    "previous_five_expected_minutes": round(expected, 6),
                                    "smoothed_points_per_minute": round(rate, 6),
                                    "prior_box_score_only": True,
                                })
                    statuses[kind + "_games"] += 1
                if not valid_score(g):
                    statuses["unusable_final_score"] += int(g["completed"])
                    continue
                home_score, away_score = float(g["home_score"]), float(g["away_score"])
                margin = abs(home_score - away_score)
                pre_h = hst.elo
                pre_a = ast.elo
                location = 0.0 if number(g.get("neutral_site")) == 1 else 60.0
                p_home = 1.0/(1.0 + 10.0**((pre_a - pre_h - location)/400.0))
                elo_delta = 22.0 * min(1.6, .75 + .27 * math.log1p(margin)) * ((1.0 if home_score > away_score else 0.0) - p_home)
                hbox = boxes.get((gid, h_id), {})
                abox = boxes.get((gid, a_id), {})
                hp = get_possessions(hbox)
                ap = get_possessions(abox)
                for bx, correct_score in ((hbox, home_score), (abox, away_score)):
                    bx_pts = number(bx.get("points"))
                    if bx_pts is not None and abs(bx_pts - correct_score) > 0.01:
                        hp = ap = None  # do not use mismatched box-score efficiency
                        statuses["box_score_mismatch"] += 1
                        break
                # use the two teams' ratings PREVIOUS to this day's results.
                updates.append((g, home_score, away_score, hp, ap, pre_h, pre_a,
                                hst.eff(), ast.eff(), elo_delta, hbox, abox))
            for g, hs, as_, hp, ap, pre_h, pre_a, h_eff, a_eff, delta, hbox, abox in updates:
                gid, hid, aid = g["game_id"], g["home_team_id"], g["away_team_id"]
                for tid, oid, scored, conceded, p, own_box, opp_eff, opp_elo, elo_change in (
                    (hid, aid, hs, as_, hp, hbox, a_eff, pre_a, delta),
                    (aid, hid, as_, hs, ap, abox, h_eff, pre_h, -delta)
                ):
                    st = states[tid]
                    st.elo += elo_change
                    st.n += 1
                    st.points_for += scored
                    st.points_against += conceded
                    st.opponent_elo_sum += opp_elo
                    st.last_day = day
                    if p is not None:
                        opp_def, opp_off = opp_eff[1], opp_eff[0]
                        st.off_adjusted_sum += 100.0*scored/p - (opp_def - BASE_EFF)
                        st.def_adjusted_sum += 100.0*conceded/p - (opp_off - BASE_EFF)
                        st.efficiency_games += 1
                        st.poss_games += 1
                        st.poss_sum += p
                        tov = number(own_box.get("turnovers"))
                        if tov is not None and tov >= 0:
                            st.tov += tov
                            st.poss_for_tov += p
                    fga = number(own_box.get("fg_a"))
                    fgm = number(own_box.get("fg_m"))
                    three_m = number(own_box.get("three_m"))
                    three_a = number(own_box.get("three_a"))
                    ft_a = number(own_box.get("ft_a"))
                    if (None not in (fga, fgm, three_a, three_m) and fga > 0
                            and 0 <= fgm <= fga and 0 <= three_m <= three_a <= fga):
                        st.fgm += fgm
                        st.fga += fga
                        st.three_m += three_m
                        st.three_a += three_a
                        if ft_a is not None and ft_a >= 0:
                            st.fta += ft_a
                            st.fta_fga += fga
                    roster = player_roster_from_game(player_boxes.get((gid, tid), []))
                    if roster:
                        prior = st.players
                        for pid, (mins, pts) in roster.items():
                            pstate = prior.setdefault(pid, Player())
                            pstate.minutes += mins
                            pstate.points += pts
                            pstate.games_seen += 1
                        st.recent_rosters.append({pid: mins for pid, (mins, _) in roster.items()})
                        st.player_box_games += 1
    stats = dict(statuses)
    stats.update({"source_games": len(source), "feature_team_rows": len(team_out),
                  "feature_player_rows": len(player_out),
                  "source_player_team_boxes": len(player_boxes)})
    return team_out, player_out, stats


def fetch_source(url: str, start_season: int, end_season: int):
    import psycopg
    from psycopg import sql
    from psycopg.rows import dict_row

    with psycopg.connect(url, sslmode="require", connect_timeout=30,
                         prepare_threshold=None, row_factory=dict_row) as conn:
        conn.read_only = True
        conn.isolation_level = psycopg.IsolationLevel.REPEATABLE_READ
        with conn.cursor() as cur:
            cur.execute("""
                SELECT g.game_id::text game_id, g.season, g.game_date::text game_date,
                  g.start_time_utc::text start_time_utc, g.home_team_id::text home_team_id,
                  g.away_team_id::text away_team_id, g.neutral_site, g.home_score,
                  g.away_score, g.completed
                FROM public.games g
                JOIN public.teams ht ON ht.team_id=g.home_team_id AND ht.is_d1=1
                JOIN public.teams at ON at.team_id=g.away_team_id AND at.is_d1=1
                WHERE g.season BETWEEN %s AND %s AND g.game_date IS NOT NULL
                ORDER BY g.season, g.game_date, g.game_id
            """, (start_season, end_season))
            games = cur.fetchall()
            if not games:
                raise RuntimeError("No D1 games found. Verify the ESPN importer and selected seasons.")
            cur.execute("""SELECT table_name, column_name FROM information_schema.columns
                           WHERE table_schema='public' AND table_name IN
                           ('team_game_stats','player_game_stats')""")
            existing: dict[str, set] = defaultdict(set)
            for row in cur.fetchall():
                existing[row["table_name"]].add(row["column_name"])
            required = {"game_id", "team_id"}
            if not required <= existing["team_game_stats"]:
                raise RuntimeError("team_game_stats must have game_id and team_id")
            tmap = {name: next((a for a in options if a in existing["team_game_stats"]), None)
                    for name, options in TEAM_ALIASES.items()}
            tfields = [sql.SQL("s.game_id::text game_id"), sql.SQL("s.team_id::text team_id")]
            for name, column in tmap.items():
                expr = (sql.SQL("NULL::text") if column is None else
                        sql.SQL("s.{}::text").format(sql.Identifier(column)))
                tfields.append(sql.SQL("{} AS {}").format(expr, sql.Identifier(name)))
            query = sql.SQL("""SELECT {} FROM public.team_game_stats s
                     JOIN public.games g ON g.game_id = s.game_id
                     WHERE g.season BETWEEN %s AND %s AND g.completed=1""").format(
                         sql.SQL(", ").join(tfields))
            cur.execute(query, (start_season, end_season))
            team_raw = cur.fetchall()
            boxes: dict[tuple[str,str], dict] = {}
            for row in team_raw:
                key = (row["game_id"], row["team_id"])
                if key in boxes:
                    raise RuntimeError(f"Duplicate team_game_stats for {key}")
                boxes[key] = dict(row)
            players: dict[tuple[str,str], list[dict]] = defaultdict(list)
            required_player = {"game_id", "team_id", "player_id"}
            pmap = {name: next((a for a in options if a in existing["player_game_stats"]), None)
                    for name, options in PLAYER_ALIASES.items()}
            optional_audit = {
                "team_columns": tmap, "player_columns": pmap,
                "players_usable": required_player <= existing["player_game_stats"] and all(pmap.values()),
            }
            if optional_audit["players_usable"]:
                pfields = [sql.SQL("p.game_id::text game_id"), sql.SQL("p.team_id::text team_id"),
                           sql.SQL("p.player_id::text player_id")]
                for name, column in pmap.items():
                    pfields.append(sql.SQL("p.{}::text AS {}").format(sql.Identifier(column), sql.Identifier(name)))
                statement = sql.SQL("""SELECT {} FROM public.player_game_stats p
                    JOIN public.games g ON g.game_id=p.game_id
                    WHERE g.season BETWEEN %s AND %s AND g.completed=1""").format(
                        sql.SQL(", ").join(pfields))
                cur.execute(statement, (start_season, end_season))
                for row in cur:
                    players[(row["game_id"], row["team_id"])].append(dict(row))
            return games, boxes, players, optional_audit


def store(url: str, team_rows: list[dict], player_rows: list[dict],
          stats: dict, schema_audit: dict) -> tuple[int, int, int]:
    import psycopg
    from psycopg import sql
    from psycopg.rows import dict_row

    with psycopg.connect(url, sslmode="require", connect_timeout=30,
                         prepare_threshold=None, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            # Fail early if SQL installation hasn't been done. Never migrate
            # automatically using the nightly connection.
            cur.execute("SELECT to_regclass('public.mbb_v2_team_pregame') AS tbl")
            if cur.fetchone()["tbl"] is None:
                raise RuntimeError("Run sql/01_create_feature_tables.sql in Supabase SQL Editor first")
            cur.execute("INSERT INTO public.mbb_v2_refresh_log(status, source_games, source_team_boxes, source_player_boxes, details) "
                        "VALUES ('running', %s, %s, %s, %s::jsonb) RETURNING id",
                        (stats["source_games"], stats.get("source_team_boxes", 0),
                         stats["source_player_team_boxes"], json.dumps(schema_audit)))
            run_id = int(cur.fetchone()["id"])
            cur.execute("CREATE TEMP TABLE _mbb_stage_team "
                        "(LIKE public.mbb_v2_team_pregame INCLUDING DEFAULTS) ON COMMIT DROP")
            cur.execute("CREATE TEMP TABLE _mbb_stage_player "
                        "(LIKE public.mbb_v2_player_pregame INCLUDING DEFAULTS) ON COMMIT DROP")
            for table, columns, rows in (("_mbb_stage_team", TEAM_FIELDS, team_rows),
                                         ("_mbb_stage_player", PLAYER_FIELDS, player_rows)):
                fields = sql.SQL(", ").join(map(sql.Identifier, columns))
                cmd = sql.SQL("COPY {} ({}) FROM STDIN").format(sql.Identifier(table), fields)
                with cur.copy(cmd) as copy:
                    for row in rows:
                        values = []
                        for name in columns:
                            val = row.get(name)
                            if name == "quality_flags" and val is not None:
                                # psycopg's native list adapter handles text[] in COPY.
                                val = list(val)
                            values.append(val)
                        copy.write_row(tuple(values))
            cur.execute("""
                INSERT INTO public.mbb_v2_team_pregame ({})
                SELECT {} FROM _mbb_stage_team s
                WHERE s.snapshot_kind='historical_replay'
                   OR (s.snapshot_kind='prospective' AND
                       s.tipoff_utc > now() + interval '10 minutes'
                       AND EXISTS (SELECT 1 FROM public.games g
                         WHERE g.game_id::text=s.game_id AND g.completed=0))
                ON CONFLICT (game_id,team_id,snapshot_kind,feature_version) DO NOTHING
            """.format(", ".join(TEAM_FIELDS), ", ".join("s." + v for v in TEAM_FIELDS)))
            team_inserted = cur.rowcount
            cur.execute("""
                INSERT INTO public.mbb_v2_player_pregame ({})
                SELECT {} FROM _mbb_stage_player s
                WHERE s.snapshot_kind='historical_replay'
                   OR (s.snapshot_kind='prospective' AND
                       s.tipoff_utc > now() + interval '10 minutes'
                       AND EXISTS (SELECT 1 FROM public.games g
                         WHERE g.game_id::text=s.game_id AND g.completed=0))
                ON CONFLICT (game_id,team_id,player_id,snapshot_kind,feature_version) DO NOTHING
            """.format(", ".join(PLAYER_FIELDS), ", ".join("s." + v for v in PLAYER_FIELDS)))
            player_inserted = cur.rowcount
            cur.execute("""UPDATE public.mbb_v2_refresh_log SET finished_at=now(),
                 status='success', team_rows_inserted=%s, player_rows_inserted=%s
                 WHERE id=%s""", (team_inserted, player_inserted, run_id))
        conn.commit()
    return run_id, team_inserted, player_inserted


def main():
    parser = argparse.ArgumentParser(description="NCAA MBB prior-day feature store upgrade")
    parser.add_argument("--output-dir", default="feature_store_audit")
    parser.add_argument("--min-season", type=int, default=2027)
    parser.add_argument("--max-season", type=int, default=2027)
    parser.add_argument("--dry-run", action="store_true", help="Build audit; do not write Supabase")
    args = parser.parse_args()
    if args.min_season > args.max_season:
        parser.error("--min-season must be <= --max-season")
    path = Path(args.output_dir)
    path.mkdir(exist_ok=True, parents=True)
    url = os.getenv("DATABASE_URL")
    if not url:
        raise SystemExit("Missing GitHub Actions secret DATABASE_URL")
    games, boxes, players, audit = fetch_source(url, args.min_season, args.max_season)
    team_rows, player_rows, stats = generate(games, boxes, players)
    stats["source_team_boxes"] = len(boxes)
    audit.update(stats)
    audit["feature_version"] = VERSION
    audit["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    audit["historical_replay_is_prospective"] = False
    audit["injury_status_inferred_from_missing_players"] = False
    audit["security_note"] = "Snapshots are private; GitHub DATABASE_URL is server-side only."
    if not args.dry_run:
        run_id, tn, pn = store(url, team_rows, player_rows, stats, audit)
        audit.update({"run_id": run_id, "team_rows_inserted": tn, "player_rows_inserted": pn})
    else:
        audit["dry_run_no_database_writes"] = True
    (path / "audit.json").write_text(json.dumps(audit, indent=2, default=str), encoding="utf-8")
    (path / "summary.txt").write_text(
        "NCAA MBB V2.0 feature store\n"
        + "Source games: {}\nTeam pregame feature rows calculated: {}\n".format(len(games), len(team_rows))
        + "Player pregame feature rows calculated: {}\n".format(len(player_rows))
        + "Historical replay is NOT saved-in-advance accuracy.\n"
        + "Prospective rows are saved ONLY before scheduled tipoff.\n", encoding="utf-8")
    print("Feature-store build completed. See", path / "audit.json")
    print("Games:", len(games), "team feature rows:", len(team_rows),
          "player feature rows:", len(player_rows))


if __name__ == "__main__":
    main()
