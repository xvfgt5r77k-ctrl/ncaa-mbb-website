"""Nightly GitHub Actions job: download ESPN to temp SQLite, merge into Supabase."""
from __future__ import annotations

import argparse
import logging
from datetime import date, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

from cloud_store import push_sqlite, database_url
from pipeline import sync_range


def parse_args():
    parser = argparse.ArgumentParser(description="Synchronize NCAA men's basketball to the cloud")
    parser.add_argument("--days-back", type=int, default=4)
    parser.add_argument("--days-forward", type=int, default=14)
    parser.add_argument("--start-date", type=date.fromisoformat, default=None,
                        help="Optional inclusive YYYY-MM-DD backfill start")
    parser.add_argument("--end-date", type=date.fromisoformat, default=None,
                        help="Optional inclusive YYYY-MM-DD backfill end")
    return parser.parse_args()


def select_range(args, today=None):
    today = today or date.today()
    if not 0 <= args.days_back <= 365 or not 0 <= args.days_forward <= 365:
        raise ValueError("Days-back/forward must be in range 0..365")
    if args.start_date:
        start = args.start_date
        end = args.end_date or today
    elif args.end_date:
        raise ValueError("--end-date requires --start-date")
    else:
        start, end = today - timedelta(days=args.days_back), today + timedelta(days=args.days_forward)
    if end < start or (end-start).days > 365:
        raise ValueError("Selected date range is invalid or longer than one year")
    return start, end


def main():
    args = parse_args()
    start, end = select_range(args)
    database_url()  # Fail before attempting ESPN if secret not set.
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    print(f"Downloading ESPN men's basketball data from {start} through {end}")
    with TemporaryDirectory() as temporary:
        result = sync_range(start, end, db_path=Path(temporary)/"ingest.sqlite3", force_recent=True)
        if result["status"] == "failed":
            raise RuntimeError("ESPN sync failed; no cloud data was changed. Review the log.")
        counts = push_sqlite(Path(temporary)/"ingest.sqlite3", result=result)
        print("Published cloud rows:", counts)
        print(f"Sync status: {result['status']}; games={result['games']}; boxscores={result['boxscores']}")
        if result["errors"]:
            # Mark action as failed while retaining successful partial rows for the dashboard.
            raise RuntimeError(f"ESPN sync partially succeeded, with {len(result['errors'])} problem(s). Check logs.")


if __name__ == "__main__":
    main()
