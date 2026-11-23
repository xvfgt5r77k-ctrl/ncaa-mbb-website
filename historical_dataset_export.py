"""Read-only, reproducible NCAA MBB historical data export.

The output is a ZIP of sorted CSVs plus a SHA-256 manifest.  Export only
sports analytics fields (no secrets, injury reports or raw ESPN JSON).

Run on GitHub Actions with the existing DATABASE_URL repository secret:
    python historical_dataset_export.py --start-season 2022 --end-season 2026

Requires psycopg and the repository's existing cloud_store.py.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import date, datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import tempfile
import zipfile


@dataclass(frozen=True)
class TableExport:
    name: str
    columns: tuple[str, ...]
    sql: str


TABLES = (
    TableExport(
        "games",
        ("game_id", "season", "game_date", "start_time_utc", "home_team_id",
         "away_team_id", "status", "status_detail", "home_score", "away_score",
         "venue", "neutral_site", "completed"),
        """SELECT game_id,season,game_date,start_time_utc,home_team_id,away_team_id,
                  status,status_detail,home_score,away_score,venue,neutral_site,completed
             FROM games WHERE season BETWEEN %s AND %s
             ORDER BY season,game_date,game_id"""
    ),
    TableExport(
        "teams",
        ("team_id", "name", "abbreviation", "conference", "logo_url", "is_d1"),
        """SELECT team_id,name,abbreviation,conference,logo_url,is_d1
             FROM teams
             WHERE team_id IN (
               SELECT home_team_id FROM games WHERE season BETWEEN %s AND %s
               UNION SELECT away_team_id FROM games WHERE season BETWEEN %s AND %s
             ) ORDER BY team_id"""
    ),
    TableExport(
        "team_game_stats",
        ("game_id", "team_id", "home_away", "points", "fg_m", "fg_a",
         "three_m", "three_a", "ft_m", "ft_a", "offensive_rebounds",
         "defensive_rebounds", "rebounds", "assists", "steals", "blocks",
         "turnovers", "fouls", "possessions", "offensive_efficiency",
         "defensive_efficiency", "net_efficiency"),
        """SELECT s.game_id,s.team_id,s.home_away,s.points,s.fg_m,s.fg_a,
                  s.three_m,s.three_a,s.ft_m,s.ft_a,s.offensive_rebounds,
                  s.defensive_rebounds,s.rebounds,s.assists,s.steals,s.blocks,
                  s.turnovers,s.fouls,s.possessions,s.offensive_efficiency,
                  s.defensive_efficiency,s.net_efficiency
             FROM team_game_stats s JOIN games g ON g.game_id=s.game_id
             WHERE g.season BETWEEN %s AND %s ORDER BY g.season,s.game_id,s.team_id"""
    ),
    TableExport(
        "pregame_team_features",
        ("season", "game_id", "game_date", "team_id", "opponent_id",
         "is_home", "neutral_site", "games_before", "avg_points_for",
         "avg_points_against", "avg_possessions", "raw_offense", "raw_defense",
         "raw_net", "recent_five_margin", "rest_days", "adjusted_net_before",
         "opponent_sos_before", "rating_snapshot_date"),
        """SELECT season,game_id,game_date,team_id,opponent_id,is_home,
                  neutral_site,games_before,avg_points_for,avg_points_against,
                  avg_possessions,raw_offense,raw_defense,raw_net,recent_five_margin,
                  rest_days,adjusted_net_before,opponent_sos_before,rating_snapshot_date
             FROM pregame_team_features WHERE season BETWEEN %s AND %s
             ORDER BY season,game_date,game_id,team_id"""
    ),
    TableExport(
        "historical_team_snapshots",
        ("season", "snapshot_date", "team_id", "games_played", "adjusted_offense",
         "adjusted_defense", "adjusted_net", "raw_net", "opponent_sos"),
        """SELECT season,snapshot_date,team_id,games_played,adjusted_offense,
                  adjusted_defense,adjusted_net,raw_net,opponent_sos
             FROM historical_team_snapshots WHERE season BETWEEN %s AND %s
             ORDER BY season,snapshot_date,team_id"""
    ),
    TableExport(
        "team_adjusted_ratings",
        ("season", "team_id", "games_played", "adjusted_offense",
         "adjusted_defense", "adjusted_net", "raw_net", "opponent_sos"),
        """SELECT season,team_id,games_played,adjusted_offense,
                  adjusted_defense,adjusted_net,raw_net,opponent_sos
             FROM team_adjusted_ratings WHERE season BETWEEN %s AND %s
             ORDER BY season,team_id"""
    ),
    TableExport(
        "historical_coverage_audit",
        ("season", "all_game_records", "completed_games", "not_completed_games",
         "d1_matchups", "complete_scores", "both_box_scores", "usable_stat_games",
         "score_mismatches", "incomplete_possessions", "missing_start_time"),
        """SELECT season,all_game_records,completed_games,not_completed_games,
                  d1_matchups,complete_scores,both_box_scores,usable_stat_games,
                  score_mismatches,incomplete_possessions,missing_start_time
             FROM historical_coverage_audit WHERE season BETWEEN %s AND %s
             ORDER BY season"""
    ),
    TableExport(
        "historical_score_baseline_audit",
        ("season", "split_name", "eligible_d1_games", "evaluated_games",
         "evaluated_team_scores", "points_mae", "points_rmse", "points_bias"),
        """SELECT season,split_name,eligible_d1_games,evaluated_games,
                  evaluated_team_scores,points_mae,points_rmse,points_bias
             FROM historical_score_baseline_audit WHERE season BETWEEN %s AND %s
             ORDER BY season"""
    ),
)


def count_placeholders(spec: TableExport) -> int:
    return spec.sql.count("%s")


def date_only(value: object) -> str:
    """Reject missing or malformed dates rather than trusting lexical sorting."""
    if value is None:
        raise ValueError("Missing required date")
    s = str(value)[:10]
    return date.fromisoformat(s).isoformat()


def validate_pregame_row(row: dict) -> None:
    snapshot = row.get("rating_snapshot_date")
    if snapshot not in (None, ""):
        game_day = date_only(row["game_date"])
        prior_day = date_only(snapshot)
        if prior_day >= game_day:
            raise ValueError(f"Non-pregame rating found for game {row['game_id']}")
    if int(row["games_before"]) < 0:
        raise ValueError(f"Negative games_before in {row['game_id']}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_dataset(cursor, spec: TableExport, start: int, end: int, folder: Path) -> dict:
    path = folder / f"{spec.name}.csv"
    args = tuple(x for _ in range(count_placeholders(spec)//2) for x in (start, end))
    # The teams query has 4 placeholders (the home/away UNION subqueries).
    cursor.execute(spec.sql, args)
    row_count = 0
    seasons = set()
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=spec.columns, lineterminator="\n")
        writer.writeheader()
        while True:
            batch = cursor.fetchmany(2000)
            if not batch:
                break
            for row in batch:
                if spec.name == "pregame_team_features":
                    validate_pregame_row(row)
                if "season" in row and row["season"] is not None:
                    seasons.add(int(row["season"]))
                writer.writerow({k: row.get(k) for k in spec.columns})
                row_count += 1
    return {
        "file": path.name,
        "rows": row_count,
        "sha256": sha256_file(path),
        "columns": list(spec.columns),
        "seasons_present": sorted(seasons),
    }


def validate_coverage(manifest_tables: dict, start: int, end: int) -> None:
    expected = set(range(start, end+1))
    for table in ("games", "pregame_team_features", "historical_team_snapshots",
                  "historical_coverage_audit", "historical_score_baseline_audit"):
        actual = set(manifest_tables[table]["seasons_present"])
        missing = expected - actual
        if missing:
            raise ValueError(f"{table} missing seasons: {sorted(missing)}. Export aborted.")
    if manifest_tables["team_game_stats"]["rows"] == 0:
        raise ValueError("No team game box scores. Export aborted.")
    if manifest_tables["teams"]["rows"] == 0:
        raise ValueError("No team metadata. Export aborted.")


def zip_and_manifest(folder: Path, output_dir: Path, start: int, end: int, tables: dict) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    content_hash = hashlib.sha256(
        json.dumps({k: v["sha256"] for k, v in sorted(tables.items())},
                   sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]
    archive_name = f"ncaa_mbb_history_{start}-{end}_{timestamp}_{content_hash}.zip"
    manifest = {
        "format_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "season_range_inclusive": [start, end],
        "data_content_id": content_hash,
        "purpose": "Frozen historical NCAA men's basketball analytics for independent review",
        "notes": [
            "Export is read-only and includes no secrets, injury reports, or raw JSON.",
            "Games include canceled/postponed records where stored; filter completed=1 for results.",
            "Team conference metadata reflects the current teams table, not necessarily historical membership.",
            "Historical team snapshots are END-OF-DAY, not pregame.",
            "Pregame features may be missing at the beginning of a season.",
            "Historical scoring baseline errors are descriptive, not model accuracy.",
            "Generated_at and updated_at source timestamps are intentionally excluded for stable data digests.",
        ],
        "files": tables,
    }
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    archive = output_dir / archive_name
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zip_out:
        for entry in sorted(folder.iterdir(), key=lambda p: p.name):
            zip_out.write(entry, arcname=entry.name)
    return archive


def verify_archive(path: Path) -> dict:
    """Recheck ZIP CRCs and each CSV's SHA256 against its manifest."""
    with zipfile.ZipFile(path) as zip_in:
        if zip_in.testzip() is not None:
            raise ValueError("ZIP has a failed CRC check")
        manifest = json.loads(zip_in.read("manifest.json"))
        for key, info in manifest["files"].items():
            entry = info["file"]
            if entry != f"{key}.csv":
                raise ValueError(f"Unexpected file path for {key}")
            actual_sha = hashlib.sha256(zip_in.read(entry)).hexdigest()
            if actual_sha != info["sha256"]:
                raise ValueError(f"SHA-256 mismatch for {entry}")
        return manifest


def export(start: int, end: int, output_dir: Path) -> Path:
    if start < 2000 or end > 2100 or start > end or end-start > 15:
        raise ValueError("Season years must be ordered, between 2000 and 2100, and span no more than 16 seasons")
    from psycopg.rows import dict_row
    from cloud_store import open_cloud_connection
    with tempfile.TemporaryDirectory(prefix="ncaa_mbb_dataset_") as folder_s:
        folder = Path(folder_s)
        with open_cloud_connection() as conn:
            with conn.cursor(row_factory=dict_row) as cursor:
                # Repeatable read ensures each table uses the SAME frozen database view.
                # No DDL, INSERT, UPDATE, or DELETE is executed.
                cursor.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY")
                details = {}
                for spec in TABLES:
                    details[spec.name] = write_dataset(cursor, spec, start, end, folder)
                    print(f"Exported {spec.name}: {details[spec.name]['rows']:,} rows")
        validate_coverage(details, start, end)
        archive = zip_and_manifest(folder, output_dir, start, end, details)
    verify_archive(archive)
    print(f"Archive verified: {archive}")
    return archive


def main():
    parser = argparse.ArgumentParser(description="Create a verified, frozen NCAA basketball analytics archive")
    parser.add_argument("--start-season", type=int, default=2022)
    parser.add_argument("--end-season", type=int, default=2026)
    parser.add_argument("--output-dir", default="historical_exports")
    parser.add_argument("--verify", type=Path, help="Verify an existing dataset ZIP instead of exporting")
    args = parser.parse_args()
    if args.verify is not None:
        manifest = verify_archive(args.verify)
        print(f"Verified archive; content ID: {manifest['data_content_id']}")
    else:
        export(args.start_season, args.end_season, Path(args.output_dir))


if __name__ == "__main__":
    main()
