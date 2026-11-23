"""Command-line interface: python main.py --help"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, timedelta

from db import TEAM_RANKINGS, connect, init
from pipeline import sync_range


def date_arg(value):
    try:
        return date.fromisoformat(value)
    except ValueError:
        raise argparse.ArgumentTypeError("Use YYYY-MM-DD")


def print_table(headers, records):
    records = [["" if item is None else str(item) for item in row] for row in records]
    widths = [max(len(h), *(len(r[i]) for r in records)) for i, h in enumerate(headers)]
    print("  ".join(h.ljust(widths[i]) for i, h in enumerate(headers)))
    print("  ".join("-" * w for w in widths))
    for record in records:
        print("  ".join(c.ljust(widths[i]) for i, c in enumerate(record)))


def main():
    parser = argparse.ArgumentParser(description="NCAA men's Division I basketball database")
    parser.add_argument("--db", default=None, help="Optional path to SQLite database")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("init", help="Create database tables")
    sync = sub.add_parser("sync", help="Pull recent results and upcoming schedule from ESPN")
    sync.add_argument("--days-back", type=int, default=4)
    sync.add_argument("--days-forward", type=int, default=14)
    backfill = sub.add_parser("backfill", help="Load every day in a historical date range")
    backfill.add_argument("--start", required=True, type=date_arg)
    backfill.add_argument("--end", required=True, type=date_arg)
    sub.add_parser("status", help="Show database counts and last update")
    report = sub.add_parser("rankings", help="Show raw, possession-based efficiency rankings")
    report.add_argument("--limit", type=int, default=20)
    schedule = sub.add_parser("schedule", help="Show schedule and final scores for a date")
    schedule.add_argument("--date", type=date_arg, default=date.today())
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    db_arg = {"db_path": args.db} if args.db else {}

    if args.command in ("sync", "backfill"):
        if args.command == "sync":
            if args.days_back < 0 or args.days_forward < 0:
                parser.error("days-back and days-forward cannot be negative")
            start = date.today() - timedelta(days=args.days_back)
            end = date.today() + timedelta(days=args.days_forward)
            result = sync_range(start, end, force_recent=True, **db_arg)
        else:
            result = sync_range(args.start, args.end, **db_arg)
        print(f"Status: {result['status']} | dates: {result['days']} | games: {result['games']} | boxscores added/refreshed: {result['boxscores']}")
        if result["errors"]:
            print("Warnings (first 5):")
            for message in result["errors"][:5]:
                print(" -", message)
            return 1
        return 0

    conn = connect(args.db) if args.db else connect()
    init(conn)
    try:
        if args.command == "init":
            print("Database created successfully.")
        elif args.command == "status":
            for table in ("teams", "games", "team_game_stats", "players", "player_game_stats", "injury_reports"):
                n = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                print(f"{table}: {n:,}")
            last = conn.execute("SELECT * FROM sync_runs ORDER BY run_id DESC LIMIT 1").fetchone()
            if last:
                print(f"Last sync: {last['finished_at']} ({last['status']})")
                if last["errors"]:
                    print("Warnings:", last["errors"][:800])
            else:
                print("No downloads yet. Run: python main.py sync")
        elif args.command == "rankings":
            rows = conn.execute(TEAM_RANKINGS).fetchall()[:max(1, args.limit)]
            print("Raw efficiency (not adjusted for strength of schedule)")
            print_table(["Team", "W-L", "Off", "Def", "Net", "eFG%"],
                        [(r["name"], f"{r['wins']}-{r['losses']}", r["off_eff"], r["def_eff"], r["net_eff"], r["efg_pct"]) for r in rows])
        elif args.command == "schedule":
            rows = conn.execute("""SELECT g.*,h.name AS home,a.name AS away FROM games g
                 JOIN teams h ON h.team_id=g.home_team_id
                 JOIN teams a ON a.team_id=g.away_team_id
                 WHERE g.game_date=? ORDER BY g.start_time_utc""", (args.date.isoformat(),)).fetchall()
            print_table(["Time (UTC)", "Away", "Home", "Score (A-H)", "Status"],
                        [(r["start_time_utc"] or "", r["away"], r["home"],
                          f"{r['away_score']}-{r['home_score']}" if r["completed"] else "-", r["status_detail"] or r["status"])
                         for r in rows])
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
