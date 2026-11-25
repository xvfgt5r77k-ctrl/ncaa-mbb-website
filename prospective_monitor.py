#!/usr/bin/env python3
"""Read-only, postgame evaluation of forecasts *already* frozen in Supabase.

No model is trained or modified. No unplayed-game projections are exported.
The database's snapshot-recording timestamps are not cryptographic evidence of
upstream feature freshness; see the limitations in every generated report.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
import statistics
from collections import Counter, defaultdict
from bisect import bisect_right
from datetime import datetime, timedelta, timezone
from pathlib import Path

SNAPSHOT_SQL = """
SELECT
    s.game_id::text AS game_id, s.season, s.model_version, s.snapshot_stage,
    s.recorded_at, s.tipoff_utc,
    s.home_team_id::text AS snap_home_id,
    s.away_team_id::text AS snap_away_id,
    s.neutral_site AS snap_neutral_site,
    s.home_games, s.away_games,
    s.projected_home_points, s.projected_away_points,
    s.projected_margin, s.projected_total,
    g.game_id::text AS matched_game_id,
    g.start_time_utc::text AS game_tipoff_utc,
    g.home_team_id::text AS game_home_id,
    g.away_team_id::text AS game_away_id,
    g.neutral_site AS game_neutral_site,
    g.completed::text AS game_completed,
    g.home_score, g.away_score
FROM public.ncaa_mbb_projection_snapshots_v1 s
LEFT JOIN public.games g ON g.game_id::text = s.game_id
WHERE s.season = %s
ORDER BY s.model_version, s.snapshot_stage, s.game_id
"""

SCHEDULE_SQL = """
SELECT game_id::text AS game_id, start_time_utc::text AS game_tipoff_utc,
       completed::text AS game_completed,
       home_team_id::text AS home_team_id, away_team_id::text AS away_team_id
FROM public.games
WHERE season = %s
"""

DISCLAIMER = (
    "Snapshot timestamps show when predictions were saved, not when every "
    "upstream input was collected or computed. This report does not certify "
    "that the original model features were leakage-free. Historical 2026 model "
    "comparisons are not a prospective baseline for V5/V8/V12 until those "
    "specific models have independently frozen pregame records."
)


def iso_dt(value):
    if value is None or value == '':
        return None
    if isinstance(value, datetime):
        out = value
    else:
        try:
            out = datetime.fromisoformat(str(value).strip().replace('Z', '+00:00'))
        except (ValueError, TypeError):
            return None
    if out.tzinfo is None:
        # Column names explicitly indicate UTC; never assume machine time zone.
        out = out.replace(tzinfo=timezone.utc)
    return out.astimezone(timezone.utc)


def num(value):
    try:
        v = float(value)
        return v if math.isfinite(v) else None
    except (ValueError, TypeError):
        return None


def yes(value):
    if value is True:
        return True
    if value is False or value is None:
        return False
    return str(value).strip().lower() in ('1', 'true', 't', 'yes')


def same_identifier(a, b):
    return a is not None and b is not None and str(a) == str(b)


def assess_snapshot(row, now, duplicates=False):
    """Classify a saved record BEFORE comparing its prediction to a final score."""
    base = {
        'game_id': str(row.get('game_id') or ''),
        'model_version': str(row.get('model_version') or ''),
        'snapshot_stage': str(row.get('snapshot_stage') or ''),
        'reason': '', 'status': '',
    }

    def reject(status, reason):
        return {**base, 'status': status, 'reason': reason}

    if duplicates:
        return reject('excluded', 'duplicate_model_stage_game')
    if row.get('matched_game_id') is None:
        return reject('excluded', 'game_missing_from_schedule')
    if not (same_identifier(row.get('snap_home_id'), row.get('game_home_id'))
            and same_identifier(row.get('snap_away_id'), row.get('game_away_id'))):
        return reject('excluded', 'opponent_ids_changed')
    if row.get('snap_neutral_site') is not None and row.get('game_neutral_site') is not None:
        if str(row['snap_neutral_site']) != str(row['game_neutral_site']):
            return reject('excluded', 'venue_changed')
    recorded = iso_dt(row.get('recorded_at'))
    scheduled = iso_dt(row.get('tipoff_utc'))
    actual_tipoff = iso_dt(row.get('game_tipoff_utc'))
    if not (recorded and scheduled and actual_tipoff):
        return reject('excluded', 'missing_or_invalid_times')
    if abs((actual_tipoff - scheduled).total_seconds()) > 10 * 60:
        return reject('excluded', 'rescheduled_since_snapshot')
    if recorded >= min(scheduled, actual_tipoff):
        return reject('excluded', 'snapshot_not_pre_tipoff')

    h = num(row.get('projected_home_points'))
    a = num(row.get('projected_away_points'))
    m = num(row.get('projected_margin'))
    t = num(row.get('projected_total'))
    if any(v is None for v in (h, a, m, t)):
        return reject('excluded', 'missing_or_invalid_forecast')
    if h < 0 or a < 0 or abs(h - a - m) > .11 or abs(h + a - t) > .11:
        return reject('excluded', 'inconsistent_forecast')

    lead = (scheduled - recorded).total_seconds() / 3600
    if row.get('snapshot_stage') == 'within_24h' and (lead < 0 or lead > 24.1):
        return reject('excluded', 'invalid_24h_stage')
    if not yes(row.get('game_completed')):
        return reject('awaiting_result' if actual_tipoff <= now else 'upcoming',
                      'past_tipoff_no_verified_result' if actual_tipoff <= now else 'scheduled')
    ah = num(row.get('home_score'))
    aa = num(row.get('away_score'))
    if ah is None or aa is None or ah < 0 or aa < 0:
        return reject('excluded', 'invalid_final_score')
    if (ah, aa) in ((0., 2.), (2., 0.)):
        return reject('excluded', 'administrative_score')
    if ah == aa:
        return reject('excluded', 'tied_final_score')
    if actual_tipoff > now:
        return reject('excluded', 'premature_completed_game')

    win = (m > 0 and ah > aa) or (m < 0 and ah < aa)
    return {
        **base,
        'status': 'evaluated',
        'reason': 'valid_pretipoff_completed_game',
        'game_tipoff_utc': actual_tipoff.isoformat(),
        'snapshot_recorded_at': recorded.isoformat(),
        'lead_hours': round(lead, 3),
        'neutral_site': str(row.get('snap_neutral_site')) == '1',
        'winner_correct': int(win) if m != 0 else None,
        'projected_tie': m == 0,
        'total_absolute_error': abs(t - (ah + aa)),
        'margin_absolute_error': abs(m - (ah - aa)),
        'home_absolute_error': abs(h - ah),
        'away_absolute_error': abs(a - aa),
        'home_baseline_correct': int(ah > aa),
    }


def average(vals):
    return round(statistics.mean(vals), 4) if vals else None


def metrics(rows):
    ok = [r for r in rows if r['status'] == 'evaluated']
    valid_winner = [r for r in ok if r['winner_correct'] is not None]
    return {
        'evaluated_games': len(ok),
        'winner_evaluable': len(valid_winner),
        'predicted_ties': len(ok) - len(valid_winner),
        'correct_winners': sum(r['winner_correct'] for r in valid_winner),
        'winner_accuracy_pct': round(100 * average([r['winner_correct'] for r in valid_winner]), 2) if valid_winner else None,
        'always_home_accuracy_pct': round(100 * average([r['home_baseline_correct'] for r in valid_winner]), 2) if valid_winner else None,
        'total_mae': average([r['total_absolute_error'] for r in ok]),
        'margin_mae': average([r['margin_absolute_error'] for r in ok]),
        'home_mae': average([r['home_absolute_error'] for r in ok]),
        'away_mae': average([r['away_absolute_error'] for r in ok]),
    }


def schedule_audit(schedule, snapshots, now):
    game_ids = {str(r['game_id']) for r in snapshots if r.get('game_id') is not None}
    counts = Counter()
    for row in schedule:
        dt = iso_dt(row.get('game_tipoff_utc'))
        if not dt:
            counts['unknown_schedule_date'] += 1
        elif dt <= now:
            counts['past_tipoff_all_schedule'] += 1
            if str(row.get('game_id')) not in game_ids:
                counts['past_tipoff_without_any_snapshot'] += 1
        else:
            counts['future_all_schedule'] += 1
            if str(row.get('game_id')) not in game_ids:
                counts['future_without_any_snapshot'] += 1
        counts['schedule_games'] += 1
    return dict(counts)


def retrospective_feature_count_flags(snapshots, schedule):
    """Advisory upper-bound warning, NOT proof of training-data leakage.

    Count finalized games *currently known* whose start was at least 3 hours
    before the timestamp the snapshot claims. We have no historical ingestion
    times and must not certify that the game's stats existed at the time.
    """
    times = defaultdict(list)
    for game in schedule:
        dt = iso_dt(game.get('game_tipoff_utc'))
        if dt is None or not yes(game.get('game_completed')):
            continue
        for team in (game.get('home_team_id'), game.get('away_team_id')):
            if team is not None:
                times[str(team)].append(dt)
    for values in times.values():
        values.sort()
    examined = 0
    suspicious = 0
    examples = []
    for item in snapshots:
        recorded = iso_dt(item.get('recorded_at'))
        if recorded is None:
            continue
        cutoff = recorded - timedelta(hours=3)
        for side, key in (('home','snap_home_id'),('away','snap_away_id')):
            number = num(item.get(f'{side}_games'))
            team = item.get(key)
            if number is None or team is None:
                continue
            examined += 1
            known = bisect_right(times.get(str(team), []), cutoff)
            if number > known + .1:
                suspicious += 1
                if len(examples) < 12:
                    examples.append({'game_id':str(item.get('game_id')),
                                     'side':side, 'saved_prior_games':number,
                                     'earlier_recorded_schedule_games':known})
    return {'snapshot_team_sides_examined':examined,
            'count_above_known_prior_schedule':suspicious,
            'sample_review_items':examples,
            'interpretation': (
                'Advisory only: earlier-game schedule coverage can be incomplete, '
                'and a previously played game may have been imported later. '
                'This is neither proof of a data leak nor a complete point-in-time audit.')}


def make_report(snapshot_rows, schedule_rows, season, now):
    keys = Counter((str(s.get('game_id')), str(s.get('model_version')),
                    str(s.get('snapshot_stage'))) for s in snapshot_rows)
    annotated = [assess_snapshot(s, now,
                   duplicates=keys[(str(s.get('game_id')), str(s.get('model_version')),
                                    str(s.get('snapshot_stage')))] > 1)
                 for s in snapshot_rows]
    by_group = defaultdict(list)
    for row in annotated:
        by_group[(row['model_version'], row['snapshot_stage'])].append(row)
    stats = []
    for (model, stage), data in sorted(by_group.items()):
        state_counts = Counter(r['status'] for r in data)
        stats.append({'model_version': model, 'stage': stage,
                      'total_snapshots': len(data), 'upcoming': state_counts['upcoming'],
                      'awaiting_result': state_counts['awaiting_result'],
                      'excluded': state_counts['excluded'], **metrics(data)})
    audit = Counter((r['status'], r['reason']) for r in annotated)
    return {
        'season': season,
        'created_at_utc': now.isoformat(),
        'snapshot_count': len(snapshot_rows),
        'unique_snapshot_games': len({str(r.get('game_id')) for r in snapshot_rows}),
        'snapshot_statuses': [
            {'status': status, 'reason': reason, 'count': count}
            for (status, reason), count in sorted(audit.items())],
        'schedule_audit': schedule_audit(schedule_rows, snapshot_rows, now),
        'feature_count_warning': retrospective_feature_count_flags(snapshot_rows, schedule_rows),
        'per_model_and_stage': stats,
        'headline_initial_v1': metrics([r for r in annotated if r['model_version'] == 'V1' and r['snapshot_stage'] == 'initial']),
        'limitations': DISCLAIMER,
    }, [r for r in annotated if r['status'] == 'evaluated']


def html_report(summary):
    esc = lambda x: html.escape('—' if x is None else str(x))
    columns = ['model_version','stage','total_snapshots','evaluated_games','winner_accuracy_pct',
               'total_mae','margin_mae','upcoming','awaiting_result','excluded']
    trs = ''.join('<tr>'+''.join(f'<td>{esc(row.get(c))}</td>' for c in columns)+'</tr>'
                  for row in summary['per_model_and_stage'])
    status_rows = ''.join('<tr><td>'+esc(r['status'])+'</td><td>'+esc(r['reason'])+'</td><td>'+esc(r['count'])+'</td></tr>'
                          for r in summary['snapshot_statuses'])
    s = summary['schedule_audit']
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><title>2027 Prospective NCAA Basketball Research Audit</title>
<style>body{{font-family:system-ui,sans-serif;max-width:1120px;margin:28px auto;padding:0 18px;line-height:1.5;color:#15202b}}
table{{width:100%;border-collapse:collapse}}td,th{{text-align:left;border-bottom:1px solid #ddd;padding:9px}}
th{{background:#eef2f5}}.note{{padding:14px;background:#eef2f5;border-left:4px solid #64748b}}</style></head><body>
<h1>NCAA Men's Basketball — Prospective Accuracy Audit</h1>
<p>ESPN season {esc(summary['season'])} · Generated {esc(summary['created_at_utc'])}</p>
<p class="note">Only already-completed games are scored. No unplayed-game selections or scoring forecasts are published. {esc(summary['limitations'])}</p>
<h2>Saved snapshots</h2><p>{esc(summary['snapshot_count'])} snapshots across {esc(summary['unique_snapshot_games'])} games.</p>
<table><thead><tr>{''.join('<th>'+esc(c.replace('_',' ').title())+'</th>' for c in columns)}</tr></thead><tbody>{trs}</tbody></table>
<h2>Data-integrity classifications</h2><table><thead><tr><th>Status</th><th>Reason</th><th>Records</th></tr></thead><tbody>{status_rows}</tbody></table>
<h2>Pregame feature-count warning</h2><p>{esc(summary['feature_count_warning']['count_above_known_prior_schedule'])} suspicious snapshot-team counts among {esc(summary['feature_count_warning']['snapshot_team_sides_examined'])} examined; review `summary.json` for examples. This is a warning, not proof of data leakage.</p>
<h2>Schedule coverage (diagnostic only)</h2><p>{esc(s.get('schedule_games',0))} schedule games, {esc(s.get('future_without_any_snapshot',0))} future games without a snapshot, and {esc(s.get('past_tipoff_without_any_snapshot',0))} past-tipoff games without a snapshot.</p>
<p><strong>Note:</strong> The schedule-coverage counts are not filtered for the model's minimum-prior-game eligibility. A missing forecast is therefore a review item, not proof of a missed eligible prediction.</p>
<h2>Method</h2><p>Require recorded_at strictly before both stored and current scheduled tipoff; match both team IDs and venue; exclude rescheduled games (&gt;10 minute shift), malformed predictions, administrative 0–2 scores, missing results, and ties. Group each model and snapshot stage separately. 24-hour snapshots are not counted as additional independent games in initial-stage metrics.</p>
</body></html>'''


def write_output(out, summary, completed):
    out.mkdir(parents=True, exist_ok=True)
    (out / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf-8')
    (out / 'report.html').write_text(html_report(summary), encoding='utf-8')
    columns = ['game_id', 'model_version', 'snapshot_stage', 'game_tipoff_utc',
               'snapshot_recorded_at', 'lead_hours', 'neutral_site', 'winner_correct',
               'projected_tie', 'total_absolute_error', 'margin_absolute_error',
               'home_absolute_error', 'away_absolute_error', 'home_baseline_correct']
    with (out / 'evaluated_completed_games.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in completed:
            writer.writerow({c: row.get(c) for c in columns})


def main():
    parser = argparse.ArgumentParser(description='Read-only prospective accuracy audit on final NCAA games')
    parser.add_argument('--season', type=int, default=2027)
    parser.add_argument('--output-dir', default='prospective_2027_results')
    parser.add_argument('--demo', action='store_true', help='Generate safe synthetic output for testing')
    args = parser.parse_args()
    now = datetime.now(timezone.utc)
    if args.demo:
        snapshots = []
        schedule = []
    else:
        url = os.getenv('DATABASE_URL')
        if not url:
            raise SystemExit('Missing DATABASE_URL GitHub Actions secret.')
        import psycopg
        from psycopg.rows import dict_row
        with psycopg.connect(url, sslmode='require', connect_timeout=20,
                             row_factory=dict_row, prepare_threshold=None) as conn:
            with conn.transaction():
                with conn.cursor() as cur:
                    cur.execute('SET TRANSACTION READ ONLY')
                    cur.execute(SNAPSHOT_SQL, (args.season,))
                    snapshots = cur.fetchall()
                    cur.execute(SCHEDULE_SQL, (args.season,))
                    schedule = cur.fetchall()
    summary, completed = make_report(snapshots, schedule, args.season, now)
    write_output(Path(args.output_dir), summary, completed)
    print('2027 retrospective audit:')
    for row in summary['per_model_and_stage']:
        print(f"  {row['model_version']}/{row['stage']}: {row['evaluated_games']} evaluated, "
              f"{row['winner_accuracy_pct']}% winner accuracy, total MAE {row['total_mae']}")
    print(f"Wrote read-only result artifacts to {args.output_dir}/")


if __name__ == '__main__':
    main()
