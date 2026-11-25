#!/usr/bin/env python3
"""Read-only RECONSTRUCTED 2027 completed-game V8/V12 research comparison.

V8/V12 outcomes are computed retrospectively, NOT saved pregame predictions.
V1 original snapshots are assessed using their genuine saved timestamps.
Produces NO future/current-game predictions and never writes to Supabase.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import v3_research as v3
import v4_research as v4
import v5_research as v5
import v6_research as v6
import v8_research as v8
import v9_research as v9
import v10_research as v10
import v11_research as v11
import v12_hybrid_research as v12
import prospective_monitor as prospect

SEASON = 2027
TRAIN_YEARS = (2022, 2023, 2024)
SPECS = Path(__file__).with_name('archived_2026_selected_models.json')


def extend_sql(original: str) -> str:
    """Extend old SELECT statements to 2027; assert exact original cutoff."""
    needle = 'BETWEEN 2022 AND 2026'
    if original.count(needle) != 1:
        raise RuntimeError('Historical SQL cutoff changed; stop for review')
    return original.replace(needle, 'BETWEEN 2022 AND 2027')


def check_specifications(archived: dict):
    selected_v8 = archived['v8']['hybrid']
    if selected_v8['general'] != 'V8 hybrid interaction_plus_shooting logistic C=0.3':
        raise RuntimeError('Archived V8 Hybrid configuration is not the expected model')
    recomputed = list(dict.fromkeys(v5.HYBRID_FIELDS + v8.MATCHUP_FIELDS + v6.SHOOTING))
    if recomputed != selected_v8['features']:
        raise RuntimeError('V8 historical input specification changed; no comparable replay')
    selected_v12 = archived['v12']
    if (selected_v12['winning_model'] != 'V12 component v5_shooting' or
            selected_v12['total_model'] != 'V12 component v11_residual_boost' or
            selected_v12['winner_weights'].get('v5_shooting') != 1 or
            selected_v12['total_weights'].get('v11_residual_boost') != 1 or
            sum(selected_v12['winner_weights'].values()) != 1 or
            sum(selected_v12['total_weights'].values()) != 1):
        raise RuntimeError('Archived V12 model selection changed; stop for review')
    if v12.WIN_COMPONENTS['v5_shooting'] != v5.STRICT_FIELDS:
        raise RuntimeError('V12 winner feature list differs from archived selection')
    return recomputed


def fetch_database():
    import psycopg
    from psycopg import sql
    from psycopg.rows import dict_row

    url = os.getenv('DATABASE_URL')
    if not url:
        raise RuntimeError('Missing repository secret DATABASE_URL')
    with psycopg.connect(url, sslmode='require', connect_timeout=30,
                         prepare_threshold=None, row_factory=dict_row) as conn:
        conn.read_only = True
        with conn.cursor() as cur:
            # Historical queries are extended without modifying imported scripts.
            cur.execute(extend_sql(v3.SQL))
            raw = pd.DataFrame(cur.fetchall())
            cur.execute(extend_sql(v4.SCORE_SQL))
            scores = pd.DataFrame(cur.fetchall())
            cur.execute(extend_sql(v5.BOX_SQL))
            boxes = pd.DataFrame(cur.fetchall())
            # Safely discover optional columns using the existing V6 mapping.
            cur.execute("""SELECT column_name FROM information_schema.columns
                WHERE table_schema='public' AND table_name='team_game_stats'""")
            existing = {r['column_name'] for r in cur.fetchall()}
            mapped = {k: next((c for c in choices if c in existing), None)
                      for k, choices in v6.ALIASES.items()}
            columns = [sql.SQL('g.season'), sql.SQL('g.game_id::text AS game_id')]
            for prefix, alias in (('home', 'hs'), ('away', 'aws')):
                for friendly, source in mapped.items():
                    expr = (sql.SQL('NULL::numeric') if source is None else
                            sql.SQL('{}.{}').format(sql.Identifier(alias), sql.Identifier(source)))
                    columns.append(sql.SQL('{} AS {}').format(expr, sql.Identifier(f'{prefix}_{friendly}')))
            opt_sql = sql.SQL('''
                SELECT {} FROM public.games g
                LEFT JOIN public.team_game_stats hs
                  ON hs.game_id=g.game_id AND hs.team_id=g.home_team_id
                LEFT JOIN public.team_game_stats aws
                  ON aws.game_id=g.game_id AND aws.team_id=g.away_team_id
                WHERE g.season BETWEEN 2022 AND 2027
                  AND g.completed=1 AND g.home_score IS NOT NULL AND g.away_score IS NOT NULL
                ORDER BY g.season,g.game_id
            ''').format(sql.SQL(', ').join(columns))
            cur.execute(opt_sql)
            optional = pd.DataFrame(cur.fetchall())
            cur.execute(prospect.SNAPSHOT_SQL, (SEASON,))
            snapshot_rows = cur.fetchall()
    if raw.empty or scores.empty or boxes.empty:
        raise RuntimeError('Historical NCAA source query returned no games')
    return raw, scores, boxes, optional, snapshot_rows, mapped


def make_replay_frame(raw, scores, boxes, optional):
    # The original V3 code restricts seasons to 2022-2026.
    # Extend its accepted season list only while replaying completed 2027 games.
    old_seasons = v3.SEASONS
    v3.SEASONS = tuple(range(2022, SEASON + 1))
    try:
        base, audit = v6.make_frame(raw, scores, boxes, optional, pd.DataFrame(),
                                    {'player_availability_replay_disabled': True})
    finally:
        v3.SEASONS = old_seasons
    # No historical player-availability rows are inferred from current status.
    frame = v8.add_matchup_features(base)
    scoring, scoring_audit = v9.scoring_replay(scores, boxes)
    frame = frame.merge(scoring, on=['season', 'game_id'], how='left', validate='one_to_one')
    # V9 definitions used by V10/V11 scoring inputs.  Derived from prior-day V5 inputs only.
    attempts = np.minimum(pd.to_numeric(frame['v5_min_box_games'], errors='coerce').fillna(0), 40)
    frame['v9_strength_uncertainty'] = 1 / np.sqrt(attempts + 1)
    frame['v9_style_three_clash'] = pd.to_numeric(frame['v5_three_matchup_gap'], errors='coerce') * (
        1 + pd.to_numeric(frame['v5_three_attempt_share_gap'], errors='coerce').abs())
    frame.replace([np.inf, -np.inf], np.nan, inplace=True)
    audit['v9_scoring_replay'] = scoring_audit
    return frame, audit


def evaluate_models(frame: pd.DataFrame, fields_v8: list[str]) -> pd.DataFrame:
    train = frame.loc[frame.season.isin(TRAIN_YEARS)].copy()
    review = frame.loc[frame.season == SEASON].copy().sort_values(['game_date','game_id']).reset_index(drop=True)
    if len(train) < 500:
        raise RuntimeError(f'Only {len(train)} 2022-2024 training games; refusing unstable retraining')
    if review.empty:
        raise RuntimeError('No eligible 2027 completed games available to compare yet')
    required = list(dict.fromkeys(fields_v8 + v12.WIN_COMPONENTS['v5_shooting'] + v11.SCORE_FIELDS
                                  + ['v9_scoring_baseline_total', 'home_win', 'home_score', 'away_score']))
    absent = [key for key in required if key not in frame]
    if absent:
        raise RuntimeError('Missing model input fields: ' + ', '.join(absent))
    # Frozen specifications selected in 2024-2025, retrained from historical 2022-2024 only.
    y = train['home_win'].astype(int)
    v8_model = v4.logistic(.3)
    v8_model.fit(train[fields_v8], y)
    v12_model = v12.model_for('v5_shooting')
    v12_model.fit(train[v12.WIN_COMPONENTS['v5_shooting']], y)
    total_model = v11.make_score_model('boost')
    residual = (train['home_score'].to_numpy(float) + train['away_score'].to_numpy(float)
                - train['v9_scoring_baseline_total'].to_numpy(float))
    if not np.isfinite(residual).all():
        raise RuntimeError('Historical score residuals contain invalid values')
    total_model.fit(train[v11.SCORE_FIELDS], residual)
    out = review[['season','game_id','game_date','home_score','away_score',
                  'home_win','neutral_site','elo_gap']].copy()
    out = out.rename(columns={'home_win':'home_actual_win'})
    out['actual_total'] = out.home_score.to_numpy(float) + out.away_score.to_numpy(float)
    out['v8_home_win_probability'] = v8_model.predict_proba(review[fields_v8])[:, 1]
    out['v12_home_win_probability'] = v12_model.predict_proba(
        review[v12.WIN_COMPONENTS['v5_shooting']])[:, 1]
    baseline = review['v9_scoring_baseline_total'].to_numpy(float)
    out['v12_projected_total'] = baseline + np.clip(total_model.predict(review[v11.SCORE_FIELDS]), -25, 25)
    out['elo_close'] = out.elo_gap.abs() <= v8.CLOSE_ELO_POINTS
    out['v8_winner_correct'] = (out.v8_home_win_probability.ge(.5) == out.home_actual_win.astype(bool)).astype(int)
    out['v12_winner_correct'] = (out.v12_home_win_probability.ge(.5) == out.home_actual_win.astype(bool)).astype(int)
    out['v12_total_absolute_error'] = (out.v12_projected_total - out.actual_total).abs()
    if out.duplicated('game_id').any():
        raise RuntimeError('Duplicate 2027 game ID; comparison not valid')
    return out


def attach_original_v1(replay: pd.DataFrame, snapshot_rows: list[dict], now: datetime):
    """V1 original only: assess actual recorded timestamps and changing opponents/tipoffs."""
    original = [r for r in snapshot_rows
                if str(r.get('model_version')) == 'V1' and str(r.get('snapshot_stage')) == 'initial']
    # Duplicate V1 original entries are excluded, not arbitrarily chosen.
    counts = pd.Series([str(r['game_id']) for r in original]).value_counts().to_dict()
    audit_counts: dict[str, int] = {}
    good: dict[str, dict] = {}
    for row in original:
        result = prospect.assess_snapshot(row, now, duplicates=(counts[str(row['game_id'])] > 1))
        status = str(result['status']) + ':' + str(result['reason'])
        audit_counts[status] = audit_counts.get(status, 0) + 1
        if result['status'] == 'evaluated':
            good[str(row['game_id'])] = {
                'game_id': str(row['game_id']),
                'v1_winner_correct': result['winner_correct'],
                'v1_projected_total': float(row['projected_total']),
                'v1_total_absolute_error': float(result['total_absolute_error']),
                'v1_recorded_at': result['snapshot_recorded_at'],
            }
    verified = pd.DataFrame(list(good.values()))
    if verified.empty:
        for col in ('game_id','v1_winner_correct','v1_projected_total','v1_total_absolute_error','v1_recorded_at'):
            verified[col] = pd.Series(dtype='object')
    joined = replay.merge(verified, on='game_id', how='left', validate='one_to_one')
    # An independently saved V1 snapshot is evidence of a historical saved forecast,
    # while reconstructed V8/V12 numbers are NOT.
    return joined, audit_counts, len(good)


def summary_frame(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    def append(model: str, scope: str, sub: pd.DataFrame):
        if sub.empty:
            return
        if model == 'V1 original pretipoff':
            s = sub[sub.v1_winner_correct.notna()]
            correct = s['v1_winner_correct'].astype(int)
            errors = pd.to_numeric(s['v1_total_absolute_error'], errors='coerce')
            kind = 'saved_pretipoff'
        elif model == 'V8 hybrid retrospective':
            s = sub
            correct = s['v8_winner_correct'].astype(int)
            errors = pd.Series(dtype=float)
            kind = 'reconstructed_after_game'
        else:
            s = sub
            correct = s['v12_winner_correct'].astype(int)
            errors = s['v12_total_absolute_error']
            kind = 'reconstructed_after_game'
        if s.empty:
            return
        rows.append({'scope': scope, 'model': model, 'evaluation_type': kind,
                     'games': len(s), 'correct_winners': int(correct.sum()),
                     'winner_accuracy_pct': round(100*float(correct.mean()), 2),
                     'total_mae': round(float(errors.mean()), 3) if len(errors) else None,
                     'always_home_pct': round(100*float(s.home_actual_win.mean()), 2)})
    all_three = frame[frame.v1_winner_correct.notna()].copy()
    for key, data in [('all_2027_replay_eligible',frame),
                      ('matched_three_models',all_three),
                      ('elo_close_replay_eligible',frame[frame.elo_close])]:
        if key == 'all_2027_replay_eligible' or key == 'elo_close_replay_eligible':
            for model in ('V8 hybrid retrospective','V12 selected retrospective'):
                append(model,key,data)
        else:
            for model in ('V1 original pretipoff','V8 hybrid retrospective','V12 selected retrospective'):
                append(model,key,data)
    return pd.DataFrame(rows, columns=['scope','model','evaluation_type','games',
                                      'correct_winners','winner_accuracy_pct','total_mae','always_home_pct'])


def create_report(summary: pd.DataFrame, audit: dict) -> str:
    summary_html = summary.to_html(index=False, na_rep='—', escape=True)
    audit_html = html.escape(json.dumps(audit,indent=2,default=str))
    return f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>2027 V1 / V8 / V12 Completed-Game Research</title>
<style>body{{font-family:system-ui,Arial;max-width:1100px;margin:32px auto;padding:0 20px;line-height:1.5;color:#1d2937}}
table{{border-collapse:collapse;width:100%;overflow-x:auto;display:block}}th,td{{padding:8px 10px;white-space:nowrap;border-bottom:1px solid #ddd;text-align:left}}th{{background:#ecf0f4}}
.notice{{background:#eef2f7;padding:14px;border-left:4px solid #64748b}}pre{{white-space:pre-wrap;overflow-wrap:anywhere}}</style>
<h1>2027 NCAA Men's Basketball: V1 vs V8 vs V12</h1><p>Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}</p>
<div class="notice"><b>Completed-game research comparison.</b> V1 original forecasts were saved before tipoff and checked against schedule changes.
V8 and V12 are reconstructed <i>after the games</i> using original historical model specifications and lagged/replayed team history.
They were not independently recorded before 2027 tipoff. V8 Hybrid includes upstream pregame ratings whose original publication timing has not been fully audited.
This report is not evidence of live V8/V12 prospective accuracy and contains no upcoming-game predictions.</div>
<h2>Accuracy (separate populations clearly labeled)</h2>{summary_html}
<p><b>matched_three_models</b> includes only games with valid V1 original pretipoff records and eligible reconstructed V8/V12 inputs.
<b>all_2027_replay_eligible</b> is a different population. Winner accuracy and lower total MAE should only be compared within the same scope.
V8 is a winner-only model in this comparison, so it has no projected-total MAE.</p>
<h2>Data and timing audit</h2><pre>{audit_html}</pre></html>'''


def save_outputs(frame: pd.DataFrame, summary: pd.DataFrame, audit: dict, output: Path, export: Path | None):
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output/'completed_2027_game_diagnostics.csv', index=False)
    summary.to_csv(output/'completed_2027_summary.csv', index=False)
    (output/'timing_and_coverage_audit.json').write_text(json.dumps(audit,indent=2,default=str)+'\n',encoding='utf-8')
    (output/'report.html').write_text(create_report(summary, audit), encoding='utf-8')
    if export is not None:
        export.mkdir(parents=True, exist_ok=True)
        summary.to_csv(export/'v1_v8_v12_completed_2027_summary.csv', index=False)
        frame.to_csv(export/'v1_v8_v12_completed_2027_games.csv', index=False)
        (export/'v1_v8_v12_audit.json').write_text(json.dumps(audit,indent=2,default=str)+'\n',encoding='utf-8')


def run(output: Path, export: Path | None):
    archived = json.loads(SPECS.read_text(encoding='utf-8'))
    fields_v8 = check_specifications(archived)
    raw, scores, boxes, optional, snapshots, mapped = fetch_database()
    frame, replay_audit = make_replay_frame(raw, scores, boxes, optional)
    result = evaluate_models(frame, fields_v8)
    result, snap_audit, valid_v1_all = attach_original_v1(result, snapshots, datetime.now(timezone.utc))
    summary = summary_frame(result)
    if summary.empty:
        raise RuntimeError('No model comparison rows generated')
    codes = ['v3_research.py','v4_research.py','v5_research.py','v6_research.py',
             'v8_research.py','v9_research.py','v10_research.py','v11_research.py',
             'v12_hybrid_research.py']
    hashes = {key:hashlib.sha256(Path(key).read_bytes()).hexdigest()[:16] for key in codes}
    audit = {'evaluation_season':SEASON,
             'generated_utc':datetime.now(timezone.utc).isoformat(),
             'replayed_2027_games':len(result),
             'v1_valid_original_snapshots_all_2027':valid_v1_all,
             'v1_valid_original_snapshots_on_replay_games':int(result.v1_winner_correct.notna().sum()),
             'v1_original_snapshot_record_status':snap_audit,
             'raw_2027_qualified':int((raw.season==SEASON).sum()),
             'scores_2027_replayed':int((scores.season==SEASON).sum()),
             'v8_model_reconstructed_not_2027_saved':True,
             'v12_model_reconstructed_not_2027_saved':True,
             'model_training_seasons':list(TRAIN_YEARS),
             'model_selection_reused_from_archived_2025_reports':True,
             'v8_uses_unverified_hybrid_upstream_ratings':True,
             'upstream_pregame_feature_publication_times_proven':False,
             'season_2026_previously_examined_during_model_design':True,
             'v12_selected_winner':'V12 component v5_shooting',
             'v12_selected_total':'V12 component v11_residual_boost',
             'optional_team_stats_column_map':mapped,
             'model_source_hashes_16':hashes,
             'replay_audit':replay_audit,
             'scope_is_completed_games_only':True,
             'no_database_writes':True}
    save_outputs(result, summary, audit, output, export)
    print('Completed 2027 retrospective V8/V12 vs pretipoff V1 research report')
    print(summary.to_string(index=False))
    print('Note: V8/V12 reconstructed retrospectively; do NOT claim prospective accuracy.')


def main():
    ap = argparse.ArgumentParser(description='Read-only completed-game 2027 comparison')
    ap.add_argument('--output-dir',default='completed_2027_comparison_results')
    ap.add_argument('--dashboard-data-dir',default='data')
    args = ap.parse_args()
    run(Path(args.output_dir), Path(args.dashboard_data_dir) if args.dashboard_data_dir else None)


if __name__ == '__main__':
    main()
