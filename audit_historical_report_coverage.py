#!/usr/bin/env python3
"""READ-ONLY completed-game historical report coverage audit.
Does NOT collect real-time player statuses, predict winners, or alter a database.
"""
import argparse
import csv
import json
import os
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

OFFICIAL_TYPES = {'official_conference', 'official_ncaa'}
FIELD_NAMES = ['season', 'game_id', 'team_id', 'source_id', 'source_url',
               'verified', 'published_at_utc', 'notes']


def load_sources(path):
    with open(path, newline='', encoding='utf-8') as f:
        sources = list(csv.DictReader(f))
    if not sources or any(not row.get('source_id') for row in sources):
        raise ValueError('Source registry is empty or missing source_id')
    return {row['source_id']: row for row in sources}


def read_evidence(path):
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        if not set(FIELD_NAMES).issubset(reader.fieldnames or []):
            raise ValueError('Evidence CSV is missing required columns')
        return list(reader)


def official_url_matches(url, source):
    expected = urlparse(source['official_entry_url']).hostname
    actual = urlparse(url or '').hostname
    return bool(expected and actual and
                (actual == expected or actual.endswith('.' + expected)))


def analyze(games, evidence, sources):
    lookup = {(str(g['season']), str(g['game_id'])): g for g in games}
    evidence_by_game = {}
    rejected = Counter()
    valid_rows = 0
    for e in evidence:
        key = (str(e.get('season', '')), str(e.get('game_id', '')))
        game = lookup.get(key)
        src = sources.get(e.get('source_id', ''))
        if game is None:
            rejected['no_matching_completed_game'] += 1
            continue
        if str(e.get('team_id', '')) not in (str(game['home_team_id']), str(game['away_team_id'])):
            rejected['team_not_in_game'] += 1
            continue
        if src is None:
            rejected['unknown_source_id'] += 1
            continue
        if e.get('verified', '').strip().lower() != 'yes':
            rejected['not_manually_verified'] += 1
            continue
        if src['source_type'] not in OFFICIAL_TYPES:
            rejected['not_official_availability_report'] += 1
            continue
        if not official_url_matches(e.get('source_url', ''), src):
            rejected['untrusted_official_url'] += 1
            continue
        valid_rows += 1
        evidence_by_game.setdefault(key, set()).add(str(e['team_id']))

    output = []
    buckets = Counter()
    for key, game in sorted(lookup.items()):
        team_ids = {str(game['home_team_id']), str(game['away_team_id'])}
        n = len(evidence_by_game.get(key, set()) & team_ids)
        buckets[n] += 1
        output.append({
            'season': key[0], 'game_id': key[1],
            'home_team_id': str(game['home_team_id']),
            'away_team_id': str(game['away_team_id']),
            'official_teams_with_documented_report': n,
            'coverage': 'both_teams' if n == 2 else ('one_team' if n == 1 else 'unknown')
        })
    summary = {
        'completed_games': len(output),
        'both_teams_documented': buckets[2],
        'one_team_documented': buckets[1],
        'no_official_report_evidence': buckets[0],
        'accepted_official_evidence_rows': valid_rows,
        'rejected_evidence_rows': dict(rejected),
        'critical_note': 'This measures indexed historical official reports, not player availability itself. Zero evidence is UNKNOWN; this script does not scrape or infer status.'
    }
    return summary, output


def fetch_completed_games(database_url, season):
    try:
        import psycopg
        from psycopg.rows import dict_row
    except ImportError as exc:
        raise RuntimeError('Install requirements_historical_coverage.txt first') from exc
    with psycopg.connect(database_url, row_factory=dict_row, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute('BEGIN READ ONLY')
            cur.execute('''SELECT g.season, g.game_id::text game_id,
                          g.home_team_id::text home_team_id,
                          g.away_team_id::text away_team_id
                     FROM public.games g
                    WHERE g.season = %s
                      AND g.completed = 1
                      AND g.home_score IS NOT NULL
                      AND g.away_score IS NOT NULL
                      AND g.home_score <> g.away_score''', (season,))
            results = cur.fetchall()
            cur.execute('ROLLBACK')
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--season', type=int, default=2026)
    p.add_argument('--sources', default='reference/source_registry.csv')
    p.add_argument('--evidence', default='reference/historical_report_evidence.csv')
    p.add_argument('--output', default='historical_coverage_report')
    args = p.parse_args()
    db_url = os.getenv('DATABASE_URL')
    if not db_url:
        raise SystemExit('DATABASE_URL secret is required; no modifications will be made')
    sources = load_sources(args.sources)
    evidence = read_evidence(args.evidence)
    games = fetch_completed_games(db_url, args.season)
    summary, per_game = analyze(games, evidence, sources)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'summary.json').open('w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    with (out / 'game_report_coverage.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['season','game_id','home_team_id','away_team_id',
                                         'official_teams_with_documented_report','coverage'])
        w.writeheader()
        w.writerows(per_game)
    with (out / 'source_registry_audit.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=next(iter(sources.values())).keys())
        w.writeheader()
        w.writerows(sources.values())
    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()
