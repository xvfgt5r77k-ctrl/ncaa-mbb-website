"""Automatic NCAA men's basketball data-quality monitoring.

Reads ESPN/Supabase data, creates ONLY the data_health_runs audit table, and
reports missing recent D1 box scores, score mismatches, ingestion freshness,
and whether current-season derived analytics have been refreshed.

No raw basketball tables are modified. Cancellations/postponements are excluded.
Run: python daily_data_health.py [--season 2027]
"""
from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

SCHEMA = """
CREATE TABLE IF NOT EXISTS data_health_runs (
    run_id BIGSERIAL PRIMARY KEY,
    checked_at TEXT NOT NULL,
    season INTEGER NOT NULL,
    window_start TEXT NOT NULL,
    window_end TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('OK','WARNING','ERROR')),
    last_import_status TEXT,
    last_import_finished_at TEXT,
    import_age_hours DOUBLE PRECISION,
    recent_completed_d1_games INTEGER NOT NULL,
    missing_scores INTEGER NOT NULL,
    missing_box_scores INTEGER NOT NULL,
    invalid_possessions INTEGER NOT NULL,
    score_mismatches INTEGER NOT NULL,
    missing_pregame_pairs INTEGER NOT NULL,
    current_season_completed_d1_games INTEGER NOT NULL,
    latest_adjusted_at TEXT,
    latest_pregame_at TEXT,
    issues_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_data_health_runs_checked ON data_health_runs(run_id DESC);
ALTER TABLE data_health_runs ENABLE ROW LEVEL SECURITY;
"""

# At 7 AM Central, allow at least one calendar day for ESPN box scores to settle.
# All games in the sample are completed D1-vs-D1. Never count canceled games.
RECENT_SQL = """
SELECT
  COUNT(*)::integer AS recent_completed_d1_games,
  COUNT(*) FILTER (WHERE g.home_score IS NULL OR g.away_score IS NULL)::integer AS missing_scores,
  COUNT(*) FILTER (WHERE hs.game_id IS NULL OR ax.game_id IS NULL
                    OR hs.points IS NULL OR ax.points IS NULL)::integer AS missing_box_scores,
  COUNT(*) FILTER (WHERE hs.game_id IS NOT NULL AND ax.game_id IS NOT NULL
                    AND hs.points IS NOT NULL AND ax.points IS NOT NULL
                    AND (hs.possessions IS NULL OR hs.possessions <= 0
                      OR ax.possessions IS NULL OR ax.possessions <= 0))::integer AS invalid_possessions,
  COUNT(*) FILTER (WHERE hs.points IS NOT NULL AND ax.points IS NOT NULL
                    AND g.home_score IS NOT NULL AND g.away_score IS NOT NULL
                    AND (hs.points <> g.home_score OR ax.points <> g.away_score))::integer AS score_mismatches,
  COUNT(*) FILTER (WHERE g.home_score IS NOT NULL AND g.away_score IS NOT NULL
                    AND (ph.game_id IS NULL OR pa.game_id IS NULL))::integer AS missing_pregame_pairs
FROM games g
JOIN teams th ON th.team_id = g.home_team_id AND th.is_d1 = 1
JOIN teams ta ON ta.team_id = g.away_team_id AND ta.is_d1 = 1
LEFT JOIN team_game_stats hs ON hs.game_id = g.game_id AND hs.team_id = g.home_team_id
LEFT JOIN team_game_stats ax ON ax.game_id = g.game_id AND ax.team_id = g.away_team_id
LEFT JOIN pregame_team_features ph ON ph.game_id = g.game_id AND ph.team_id = g.home_team_id
LEFT JOIN pregame_team_features pa ON pa.game_id = g.game_id AND pa.team_id = g.away_team_id
WHERE g.season = %s AND g.completed = 1
  AND g.game_date >= %s AND g.game_date <= %s
"""

SEASON_SQL = """
SELECT COUNT(*)::integer AS current_season_completed_d1_games
FROM games g
JOIN teams th ON th.team_id = g.home_team_id AND th.is_d1 = 1
JOIN teams ta ON ta.team_id = g.away_team_id AND ta.is_d1 = 1
WHERE g.season = %s AND g.completed = 1
"""

STAMP_SQL = """
SELECT
  (SELECT MAX(updated_at) FROM team_adjusted_ratings WHERE season = %s) AS latest_adjusted_at,
  (SELECT MAX(generated_at) FROM pregame_team_features WHERE season = %s) AS latest_pregame_at
"""

FIELDS = (
    'checked_at','season','window_start','window_end','status',
    'last_import_status','last_import_finished_at','import_age_hours',
    'recent_completed_d1_games','missing_scores','missing_box_scores',
    'invalid_possessions','score_mismatches','missing_pregame_pairs',
    'current_season_completed_d1_games','latest_adjusted_at','latest_pregame_at','issues_json'
)


def ending_year(day: date) -> int:
    """ESPN season 2027 is Nov 2026–Apr 2027."""
    return day.year + (1 if day.month >= 7 else 0)


def timestamp(value):
    """UTC-aware datetime from database TEXT; returns None for missing/bad strings."""
    if value is None:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
        return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)
    except (ValueError, TypeError, OverflowError):
        return None


def evaluate_health(metrics: dict, last_import: dict | None,
                    adjusted_at: str | None, pregame_at: str | None,
                    now: datetime, season_active: bool) -> tuple[str, list[str], float | None]:
    """Pure, testable rule evaluation. Zero games != missing game data."""
    errors, warnings = [], []
    recent = int(metrics.get('recent_completed_d1_games') or 0)
    season_total = int(metrics.get('current_season_completed_d1_games') or 0)
    last_end = timestamp(last_import.get('finished_at')) if last_import else None
    age_hours = max(0.0, (now.astimezone(timezone.utc) - last_end).total_seconds()/3600) if last_end else None

    if not last_import:
        errors.append('No ESPN sync_runs entry. Run the nightly ESPN import.')
    else:
        imported = str(last_import.get('status') or '').lower()
        if imported in ('failed','running'):
            errors.append(f'Latest recorded ESPN import status is {imported}.')
        elif imported != 'ok':
            warnings.append(f'Latest recorded ESPN import status is {imported or "unknown"}.')
        if not last_end:
            errors.append('Latest ESPN import has no valid completion timestamp.')
        elif season_active and age_hours is not None:
            if age_hours > 72:
                errors.append(f'Latest recorded ESPN import is {age_hours:.1f} hours old (over 72h).')
            elif age_hours > 40:
                warnings.append(f'Latest recorded ESPN import is {age_hours:.1f} hours old (over 40h).')

    if int(metrics.get('score_mismatches') or 0):
        errors.append(f'{metrics["score_mismatches"]} recent completed D1 games have score mismatches.')
    if int(metrics.get('missing_scores') or 0):
        errors.append(f'{metrics["missing_scores"]} recent completed D1 games lack final scores.')
    if int(metrics.get('missing_box_scores') or 0):
        warnings.append(f'{metrics["missing_box_scores"]} recent completed D1 games lack a team box score.')
    if int(metrics.get('invalid_possessions') or 0):
        warnings.append(f'{metrics["invalid_possessions"]} recent completed D1 games have invalid possessions.')
    if int(metrics.get('missing_pregame_pairs') or 0):
        warnings.append(f'{metrics["missing_pregame_pairs"]} recent completed D1 games are missing pregame team-feature rows.')
    if season_total:
        if not timestamp(adjusted_at):
            warnings.append('No adjusted ratings have been calculated for the current season.')
        elif last_end and timestamp(adjusted_at) < last_end - timedelta(minutes=10):
            warnings.append('Adjusted ratings were last generated before the latest recorded ESPN import.')
        if not timestamp(pregame_at):
            warnings.append('No pregame features have been calculated for the current season.')
        elif last_end and timestamp(pregame_at) < last_end - timedelta(minutes=10):
            warnings.append('Pregame features were last generated before the latest recorded ESPN import.')
    # An empty recent window is normal during the offseason / holiday breaks.
    # It must never be reported as missing games without independent schedule evidence.
    return ('ERROR' if errors else 'WARNING' if warnings else 'OK', errors + warnings, age_hours)


def publish(day: date | None = None, season: int | None = None):
    from psycopg.rows import dict_row
    from cloud_store import open_cloud_connection

    day = day or datetime.now(ZoneInfo('America/Chicago')).date()
    season = season or ending_year(day)
    start = (day - timedelta(days=8)).isoformat()
    end = (day - timedelta(days=2)).isoformat()
    now = datetime.now(timezone.utc)

    with open_cloud_connection() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            # Creation is the only schema change: existing ESPN and analytics
            # tables remain untouched by this script.
            cur.execute(SCHEMA)
            cur.execute("SELECT started_at, finished_at, status FROM sync_runs ORDER BY run_id DESC LIMIT 1")
            latest = cur.fetchone()
            cur.execute(RECENT_SQL, (season, start, end))
            metrics = dict(cur.fetchone())
            cur.execute(SEASON_SQL, (season,))
            metrics.update(dict(cur.fetchone()))
            cur.execute(STAMP_SQL, (season, season))
            stamps = dict(cur.fetchone())
            # NCAA play typically November-April. Outside that range, no
            # strict nightly-freshness failure; imports may intentionally pause.
            active = day.month in (11, 12, 1, 2, 3, 4)
            status, issues, age = evaluate_health(
                metrics, latest, stamps.get('latest_adjusted_at'),
                stamps.get('latest_pregame_at'), now, active)
            report = dict(checked_at=now.isoformat(timespec='seconds'), season=season,
                          window_start=start, window_end=end, status=status,
                          last_import_status=latest['status'] if latest else None,
                          last_import_finished_at=latest['finished_at'] if latest else None,
                          import_age_hours=round(age, 2) if age is not None else None,
                          latest_adjusted_at=stamps.get('latest_adjusted_at'),
                          latest_pregame_at=stamps.get('latest_pregame_at'),
                          issues_json=json.dumps(issues), **metrics)
            placeholders = ','.join(['%s']*len(FIELDS))
            cur.execute(f"INSERT INTO data_health_runs ({','.join(FIELDS)}) VALUES ({placeholders})",
                        tuple(report[f] for f in FIELDS))
    print(f'Data health: {status}; season {season}; recent D1 completed games {metrics["recent_completed_d1_games"]}; '
          f'box-score gaps {metrics["missing_box_scores"]}; score mismatches {metrics["score_mismatches"]}.')
    for issue in issues:
        print('  - ' + issue)
    # GitHub red X only for critical issues; WARNINGS display in logs/UI.
    return 1 if status == 'ERROR' else 0


def main():
    parser = argparse.ArgumentParser(description='Check daily NCAA MBB import and analytics health')
    parser.add_argument('--season', type=int, default=None,
                        help='Optional ESPN ending year; current season is used by default')
    args = parser.parse_args()
    raise SystemExit(publish(season=args.season))


if __name__ == '__main__':
    main()
