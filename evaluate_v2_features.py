"""NCAA MBB V2 research evaluation (read-only).

2022–24 fitting, 2025 validation/model selection, 2026 retrospective review.
Uses the archived 2026 V8 output for exact-match comparisons where available.
It does NOT make live forecasts, change database data, or establish prospective accuracy.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

VERSION = 'V2-feature-controlled-research-1.0'
REQUIRED = [
    'game_id','season','game_date','start_time_utc','neutral_site','home_score','away_score',
    'home_team_id','away_team_id',
]
ATTRS = [
    'game_day','tipoff_utc','games_before','elo_rating','adjusted_offense_per100',
    'adjusted_defense_per100','adjusted_net_per100','estimated_possessions','effective_fg_pct',
    'three_point_rate','turnover_per_possession','free_throw_rate','rest_days',
    'player_coverage','expected_rotation_points_per40','rotation_top3_minutes_share',
    'rotation_continuity','feature_version','quality_flags',
]
BASE = ['net_gap','nonneutral','net_by_venue']
CORE = BASE + ['elo_gap','games_gap','games_min','rest_gap']
PLAYER = CORE + ['both_player_sufficient','rotation_points_gap','top3_minutes_gap','continuity_gap']
STYLES = PLAYER + [
    'shooting_gap','three_rate_gap','turnover_gap','free_throw_gap',
    'pace_gap','offense_gap','defense_gap',
]
FEATURE_SETS = {
    'A adjusted net + venue': BASE,
    'B + Elo, experience, rest': CORE,
    'C + prior player contributions': PLAYER,
    'D + shooting and pace': STYLES,
}


def read_supabase():
    url = os.environ.get('DATABASE_URL')
    if not url:
        raise RuntimeError('Missing server-side DATABASE_URL GitHub Actions secret')
    import psycopg
    from psycopg.rows import dict_row
    select_features = ',\n '.join(
        f'h.{c} AS home_{c}, a.{c} AS away_{c}' for c in ATTRS
    )
    sql = f"""
    SELECT
      g.game_id::text AS game_id, g.season, g.game_date::text AS game_date,
      g.start_time_utc::text AS start_time_utc, g.neutral_site,
      g.home_score, g.away_score,
      g.home_team_id::text AS home_team_id,
      g.away_team_id::text AS away_team_id,
      {select_features}
    FROM public.games g
      JOIN public.teams ht ON ht.team_id = g.home_team_id AND ht.is_d1 = 1
      JOIN public.teams at ON at.team_id = g.away_team_id AND at.is_d1 = 1
      JOIN public.mbb_v2_team_pregame h
        ON h.game_id = g.game_id::text AND h.team_id = g.home_team_id::text
       AND h.snapshot_kind = 'historical_replay' AND h.feature_version = 'v2.0'
      JOIN public.mbb_v2_team_pregame a
        ON a.game_id = g.game_id::text AND a.team_id = g.away_team_id::text
       AND a.snapshot_kind = 'historical_replay' AND a.feature_version = 'v2.0'
    WHERE g.season BETWEEN 2022 AND 2026
      AND g.completed = 1
      AND g.home_score IS NOT NULL AND g.away_score IS NOT NULL
      AND g.neutral_site IN (0,1)
    ORDER BY g.season, g.game_date, g.game_id
    """
    with psycopg.connect(url, sslmode='require', connect_timeout=30,
                         prepare_threshold=None, row_factory=dict_row) as con:
        con.read_only = True
        con.isolation_level = psycopg.IsolationLevel.REPEATABLE_READ
        with con.cursor() as cur:
            cur.execute(sql)
            rows = cur.fetchall()
    if not rows:
        raise RuntimeError('No historical matchup rows found. Check that 2022-2026 backfill completed.')
    return pd.DataFrame(rows)


def _num(df, key):
    return pd.to_numeric(df[key], errors='coerce')


def prepare_games(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Validate matchup identity, chronology metadata and pregame-feature provenance.

    This detects structural inconsistencies, but cannot prove that upstream
    ESPN data was available by the historical date at which the game was played.
    """
    df = raw.copy()
    missing = set(REQUIRED).difference(df.columns)
    if missing:
        raise ValueError(f'Missing required columns: {sorted(missing)}')
    for prefix in ('home_', 'away_'):
        needed = {prefix + name for name in ATTRS}
        not_found = needed.difference(df.columns)
        if not_found:
            raise ValueError(f'Missing {prefix} attributes: {sorted(not_found)}')
    df['game_id'] = df['game_id'].astype(str)
    if df['game_id'].duplicated().any():
        raise RuntimeError('Duplicate game IDs in joined historical features: aborting')
    raw_count = len(df)
    df['home_score'] = _num(df, 'home_score')
    df['away_score'] = _num(df, 'away_score')
    df['neutral_site'] = _num(df, 'neutral_site')
    df['season'] = _num(df, 'season')
    for side in ('home','away'):
        for key in [
            'games_before','elo_rating','adjusted_offense_per100','adjusted_defense_per100',
            'adjusted_net_per100','estimated_possessions','effective_fg_pct','three_point_rate',
            'turnover_per_possession','free_throw_rate','rest_days',
            'expected_rotation_points_per40','rotation_top3_minutes_share','rotation_continuity',
        ]:
            col = f'{side}_{key}'
            df[col] = _num(df, col)
        df[f'{side}_tipoff_utc'] = pd.to_datetime(df[f'{side}_tipoff_utc'],utc=True,errors='coerce')
        df[f'{side}_game_day'] = pd.to_datetime(df[f'{side}_game_day'],errors='coerce')
    tipoff = pd.to_datetime(df['start_time_utc'], utc=True, errors='coerce')
    game_day = pd.to_datetime(df['game_date'],errors='coerce')
    score_valid = df['home_score'].ge(0) & df['away_score'].ge(0) & df['home_score'].ne(df['away_score'])
    placeholder = ((df['home_score'].eq(0) & df['away_score'].eq(2)) |
                   (df['home_score'].eq(2) & df['away_score'].eq(0)))
    both_history = df['home_games_before'].ge(5) & df['away_games_before'].ge(5)
    correct_flag = pd.Series(True, index=df.index)
    for side in ('home','away'):
        flags = df[f'{side}_quality_flags'].apply(
            lambda x: x if isinstance(x, (tuple,list)) else []
        )
        correct_flag &= flags.apply(lambda x: 'reconstructed_not_saved_pretipoff' in x)
    versions = (df['home_feature_version'].eq('v2.0') & df['away_feature_version'].eq('v2.0'))
    day_match = (df['home_game_day'].dt.normalize().eq(game_day.dt.normalize()) &
                 df['away_game_day'].dt.normalize().eq(game_day.dt.normalize()))
    tipoff_match = (tipoff.notna() &
        df['home_tipoff_utc'].eq(tipoff) & df['away_tipoff_utc'].eq(tipoff))
    net_consistent = pd.Series(True, index=df.index)
    for side in ('home','away'):
        actual_net = df[f'{side}_adjusted_net_per100']
        computed_net = df[f'{side}_adjusted_offense_per100'] - df[f'{side}_adjusted_defense_per100']
        net_consistent &= actual_net.notna() & computed_net.notna() & (actual_net - computed_net).abs().le(0.1)
    rating_ok = df['home_elo_rating'].notna() & df['away_elo_rating'].notna()
    sanity = df['neutral_site'].isin([0,1]) & df['season'].between(2022,2026)
    conditions = {
        'invalid_or_placeholder_scores': ~score_valid | placeholder,
        'under_five_prior_games': ~both_history,
        'missing_or_changed_tipoff': ~tipoff_match,
        'feature_game_day_mismatch': ~day_match,
        'missing_replay_provenance_flag': ~correct_flag,
        'wrong_feature_version': ~versions,
        'inconsistent_or_missing_adjusted_net': ~net_consistent,
        'missing_elo': ~rating_ok,
        'unknown_venue_or_season': ~sanity,
    }
    # Count individual flags, including overlapping reasons.
    audit = {name: int(mask.sum()) for name,mask in conditions.items()}
    excluded = pd.concat(conditions.values(),axis=1).any(axis=1)
    df = df.loc[~excluded].copy()
    df['home_win'] = (df['home_score'] > df['away_score']).astype(int)
    df['nonneutral'] = (1 - df['neutral_site']).astype(float)
    df['net_gap'] = df['home_adjusted_net_per100'] - df['away_adjusted_net_per100']
    df['net_by_venue'] = df['net_gap'] * df['nonneutral']
    df['elo_gap'] = df['home_elo_rating'] - df['away_elo_rating']
    df['games_gap'] = df['home_games_before'] - df['away_games_before']
    df['games_min'] = df[['home_games_before','away_games_before']].min(axis=1)
    for feature,out in [('rest_days','rest_gap'),
                        ('expected_rotation_points_per40','rotation_points_gap'),
                        ('rotation_top3_minutes_share','top3_minutes_gap'),
                        ('rotation_continuity','continuity_gap'),
                        ('effective_fg_pct','shooting_gap'),
                        ('three_point_rate','three_rate_gap'),
                        ('turnover_per_possession','turnover_gap'),
                        ('free_throw_rate','free_throw_gap'),
                        ('estimated_possessions','pace_gap'),
                        ('adjusted_offense_per100','offense_gap'),
                        ('adjusted_defense_per100','defense_gap')]:
        df[out] = df[f'home_{feature}'] - df[f'away_{feature}']
    player_good = (df['home_player_coverage'].eq('sufficient') &
                   df['away_player_coverage'].eq('sufficient'))
    df['both_player_sufficient'] = player_good.astype(float)
    for feature in ['rotation_points_gap','top3_minutes_gap','continuity_gap']:
        df.loc[~player_good,feature] = np.nan
    df['actual_total'] = df['home_score'] + df['away_score']
    df['player_complete_for_comparison'] = player_good
    audit['raw_joined_games'] = raw_count
    audit['excluded_total_unique_games'] = int(excluded.sum())
    audit['eligible_games'] = len(df)
    audit['both_player_sufficient_games'] = int(player_good.sum())
    audit['eligible_by_season'] = {str(int(k)):int(v) for k,v in df['season'].value_counts().sort_index().items()}
    return df, audit


def make_classifier():
    # All preprocessing is trained on 2022–2024 only. C fixed before 2025 validation.
    return make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),
                         StandardScaler(), LogisticRegression(C=0.5,max_iter=1500,random_state=17))


def scores(y, probs):
    y = np.asarray(y, dtype=int)
    probs = np.asarray(probs, dtype=float)
    if len(y) == 0:
        return {'games':0,'correct':None,'accuracy_pct':None,'brier':None,'log_loss':None}
    guess = (probs >= 0.5).astype(int)
    return {'games':len(y),'correct':int(np.sum(guess==y)),
            'accuracy_pct':round(float(100*accuracy_score(y,guess)),3),
            'brier':round(float(brier_score_loss(y,probs)),5),
            'log_loss':round(float(log_loss(y,np.clip(probs,1e-6,1-1e-6),labels=[0,1])),5)}


def archive_comparison(test_df, chosen_probs, archive: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    want = ['game_id','season','home_win','v8_hybrid_home_win_prob','v8_strict_home_win_prob']
    if not set(want).issubset(archive.columns):
        raise RuntimeError(f'V8 archived file missing columns: {set(want)-set(archive.columns)}')
    v8 = archive[want].copy()
    v8['game_id'] = v8['game_id'].astype(str)
    if v8['game_id'].duplicated().any():
        raise RuntimeError('V8 archive has duplicate game IDs: refuse paired comparison')
    merged = test_df[['game_id','home_win','neutral_site','player_complete_for_comparison']].copy()
    merged['new_probability'] = chosen_probs
    merged = merged.merge(v8,on='game_id',suffixes=('_current','_archive'),how='inner',validate='one_to_one')
    mismatch = merged['home_win_current'].ne(merged['home_win_archive'])
    count_mismatch = int(mismatch.sum())
    # Do not silently compare games with changed labels.
    merged = merged.loc[~mismatch].copy()
    merged['new_correct'] = (merged['new_probability']>=0.5).astype(int).eq(merged['home_win_current'])
    merged['v8_hybrid_correct'] = (merged['v8_hybrid_home_win_prob']>=0.5).astype(int).eq(merged['home_win_current'])
    merged['v8_strict_correct'] = (merged['v8_strict_home_win_prob']>=0.5).astype(int).eq(merged['home_win_current'])
    n = len(merged)
    audit = {'archive_rows':len(archive),'overlap_games':n,
             'archive_outcome_mismatches_excluded':count_mismatch,
             'archive_games_not_in_new_evaluation':len(v8)-n-count_mismatch}
    return merged, audit


def paired_accuracy_delta(first_correct, second_correct, resamples=2000):
    """Accuracy difference and simple paired bootstrap interval in percentage points.

    Descriptive uncertainty only; a reused retrospective season is not an
    untouched prospective holdout.
    """
    first = np.asarray(first_correct, dtype=float)
    second = np.asarray(second_correct, dtype=float)
    if len(first) != len(second) or len(first) == 0:
        return {'games':len(first),'delta_pp':None,'bootstrap_95pct_interval_pp':None}
    delta=(first-second)*100.0
    rng=np.random.default_rng(1207)
    vals=np.array([delta[rng.integers(0,len(delta),len(delta))].mean()
                   for _ in range(resamples)])
    return {'games':len(delta),'delta_pp':round(float(delta.mean()),3),
            'bootstrap_95pct_interval_pp':[
                round(float(np.quantile(vals,.025)),3),
                round(float(np.quantile(vals,.975)),3)]}


def evaluate(df, v8_reference: pd.DataFrame | None = None):
    train = df.loc[df.season.isin([2022,2023,2024])].copy()
    valid = df.loc[df.season.eq(2025)].copy()
    test = df.loc[df.season.eq(2026)].copy()
    if min(len(train),len(valid),len(test)) < 100:
        raise RuntimeError(f'Insufficient chronological coverage: train={len(train)}, '
                           f'validation={len(valid)}, retrospective={len(test)}')
    models, predictions = {}, {}
    for name, features in FEATURE_SETS.items():
        model = make_classifier()
        model.fit(train[features],train.home_win)
        models[name] = model
        predictions[(name,2025)] = model.predict_proba(valid[features])[:,1]
        predictions[(name,2026)] = model.predict_proba(test[features])[:,1]
    validation_rows = []
    test_rows = []
    coverage_rows = []
    for year, sample in ((2025,valid),(2026,test)):
        for name in FEATURE_SETS:
            prob = predictions[(name,year)]
            row = {'season':year,'model':name,**scores(sample.home_win,prob)}
            (validation_rows if year==2025 else test_rows).append(row)
            mask = sample.player_complete_for_comparison.to_numpy(dtype=bool)
            coverage_rows.append({'season':year,'model':name,'population':'both player histories sufficient',
                                  **scores(sample.home_win.to_numpy()[mask],prob[mask])})
        home_pred = np.ones(len(sample))
        rating_pick = (sample.net_gap >= 0).astype(float)
        (validation_rows if year==2025 else test_rows).extend([
            {'season':year,'model':'always designated home',**scores(sample.home_win,home_pred)},
            {'season':year,'model':'raw net gap (unadjusted for venue)',**scores(sample.home_win,rating_pick)},
        ])
    val = pd.DataFrame(validation_rows)
    ret = pd.DataFrame(test_rows)
    # Chosen exclusively using 2025, not 2026. Keep modest safety threshold.
    baseline_name = 'A adjusted net + venue'
    baseline = val.loc[val.model.eq(baseline_name)].iloc[0]
    permitted = val.loc[val.model.isin(FEATURE_SETS)].copy()
    permitted = permitted.loc[
        (permitted.brier <= baseline.brier - 0.001) &
        (permitted.accuracy_pct >= baseline.accuracy_pct - 0.50)
    ]
    selected = (permitted.sort_values(['brier','accuracy_pct','model'],
                                     ascending=[True,False,True]).iloc[0].model
                if len(permitted) else baseline_name)
    selected_probs = predictions[(selected,2026)]
    selected_validation_probs = predictions[(selected,2025)]
    final = pd.DataFrame({
        'game_id':test.game_id.astype(str).to_numpy(), 'season':2026,
        'game_date':test.game_date.astype(str).to_numpy(),
        'neutral_site':test.neutral_site.to_numpy(),
        'home_win':test.home_win.to_numpy(),
        'home_score':test.home_score.to_numpy(), 'away_score':test.away_score.to_numpy(),
        'actual_total':test.actual_total.to_numpy(),
        'both_player_sufficient':test.player_complete_for_comparison.to_numpy(),
        'selected_model':selected,
        'selected_home_win_probability':selected_probs,
        'adjusted_net_venue_probability':predictions[(baseline_name,2026)],
    })
    if v8_reference is not None:
        paired, archive_audit = archive_comparison(test,selected_probs,v8_reference)
    else:
        paired, archive_audit = pd.DataFrame(), {'archive_missing':True}
    # Preset neutral-site and competitive-game analyses, not used for selection.
    subgroup = []
    for venue, mask in [('home court',test.nonneutral.to_numpy()==1),
                        ('neutral',test.nonneutral.to_numpy()==0),
                        ('competitive (|net gap| <= 5)',test.net_gap.abs().to_numpy()<=5),
                        ('noncompetitive (|net gap| > 5)',test.net_gap.abs().to_numpy()>5)]:
        subgroup.append({'group':venue,'model':selected,
                         **scores(test.home_win.to_numpy()[mask],selected_probs[mask])})
        base_prob = predictions[(baseline_name,2026)]
        subgroup.append({'group':venue,'model':baseline_name,
                         **scores(test.home_win.to_numpy()[mask],base_prob[mask])})
    subgroup_frame = pd.DataFrame(subgroup)
    selected_correct = (selected_probs >= .5).astype(int) == test.home_win.to_numpy()
    base_correct = (predictions[(baseline_name,2026)]>=.5).astype(int)==test.home_win.to_numpy()
    base_delta = paired_accuracy_delta(selected_correct,base_correct)
    if len(paired):
        v8_delta = paired_accuracy_delta(paired.new_correct,paired.v8_hybrid_correct)
    else:
        v8_delta = {'games':0, 'delta_pp':None,'bootstrap_95pct_interval_pp':None}
    result = {
        'selection':selected,
        'train_games':len(train),'validation_games':len(valid),'retrospective_games':len(test),
        'validation_selected':scores(valid.home_win,selected_validation_probs),
        'retrospective_selected':scores(test.home_win,selected_probs),
        'retrospective_baseline':scores(test.home_win,predictions[(baseline_name,2026)]),
        'archived_v8_comparison':archive_audit,
        'selected_vs_base_2026':base_delta,
        'selected_vs_v8_hybrid_exact_games_2026':v8_delta,
    }
    return result,val,ret,pd.DataFrame(coverage_rows),final,paired,subgroup_frame


def write_report(out:Path, result:dict, audit:dict, val, ret, subset, paired, subgroup):
    def escape(v): return html.escape(str(v))
    def table(frame):
        return frame.to_html(index=False,border=0,classes='table',escape=True) if len(frame) else '<p>No eligible rows.</p>'
    text = '''<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
    <title>NCAA feature-store controlled historical evaluation</title><style>
    body{{font:16px/1.5 system-ui,sans-serif;max-width:1100px;margin:32px auto;padding:0 20px;color:#223;}}
    h1,h2{{color:#192b45}}.table{{border-collapse:collapse;font-size:13px;width:100%;}}
    .table td,.table th{{border-bottom:1px solid #ddd;padding:7px;text-align:right;}}
    .table td:first-child,.table th:first-child{{text-align:left}} .notice{{background:#eff3fa;padding:15px;border-radius:8px;}}
    </style></head><body>
    <h1>2022–2026 Feature-Store Controlled Research</h1>
    <p class="notice"><strong>Historical reconstruction, not prospective accuracy.</strong>
    2022–2024 were fitted, 2025 used for model selection, and 2026 was evaluated retrospectively.
    The 2026 season has been repeatedly examined in prior research; its results are not an untouched holdout.
    Reconstructed prior-day features do not prove the original ESPN data was available at those times.
    Archived V8 probabilities may use additional upstream ratings with unverified timing.
    No 2027 future predictions or wagering outputs are generated.</p>
    <h2>Selection</h2><p>Chosen only on 2025: <strong>{selection}</strong>.</p>
    <p>Training: {train} games; validation: {valid}; retrospective: {test}.</p>
    <h2>2025 validation: compare feature additions</h2>{validation}
    <h2>2026 retrospective: all eligible games</h2>{retrospective}
    <h2>Player-available subset (same games per model)</h2>{subset}
    <h2>Exact-game archived V8 comparison</h2>{paired}
    <h2>2026 venue and competitiveness (descriptive only)</h2>{subgroup}
    <h2>Audits</h2><pre>{audit}</pre>
    </body></html>'''.format(
        selection=escape(result['selection']),train=result['train_games'],valid=result['validation_games'],
        test=result['retrospective_games'],validation=table(val),retrospective=table(ret),
        subset=table(subset),paired=table(paired),subgroup=table(subgroup),
        audit=escape(json.dumps({'preparation':audit,'evaluation':result},indent=2,default=str)))
    (out/'report.html').write_text(text,encoding='utf-8')


def main():
    p=argparse.ArgumentParser(description='Historical only: V2 feature-store validation research')
    p.add_argument('--output-dir',default='feature_store_v2_evaluation')
    p.add_argument('--v8-reference',default='reference/v8_archived_2026.csv')
    args=p.parse_args()
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    raw=read_supabase()
    clean,audit=prepare_games(raw)
    ref_path=Path(args.v8_reference)
    reference=pd.read_csv(ref_path,dtype={'game_id':str}) if ref_path.exists() else None
    result,val,ret,subset,final,paired,subgroup=evaluate(clean,reference)
    audit['feature_pipeline_source'] = 'db_upgrade_v2.py as installed; historical prior-day reconstruction'
    audit['historical_replay_pretipoff_saved'] = False
    audit['historical_source_data_asof_verified'] = False
    audit['prospective_2027_data_touched'] = False
    audit['v8_reference_sha256'] = hashlib.sha256(ref_path.read_bytes()).hexdigest() if reference is not None else None
    for name, data in [('validation_2025.csv',val),('retrospective_2026.csv',ret),
                       ('player_complete_subset.csv',subset),
                       ('retrospective_game_diagnostics_2026.csv',final),
                       ('exact_game_v8_comparison_2026.csv',paired),
                       ('subgroup_2026.csv',subgroup)]:
        data.to_csv(out/name,index=False)
    (out/'data_quality_audit.json').write_text(json.dumps(audit,indent=2,default=str),encoding='utf-8')
    (out/'evaluation_summary.json').write_text(json.dumps(result,indent=2,default=str),encoding='utf-8')
    if len(paired):
        pair_summary=pd.DataFrame([
            {'model':'New V2 selected','games':len(paired),
             'accuracy_pct':round(100*paired.new_correct.mean(),3)},
            {'model':'Archived V8 Hybrid','games':len(paired),
             'accuracy_pct':round(100*paired.v8_hybrid_correct.mean(),3)},
            {'model':'Archived V8 Strict','games':len(paired),
             'accuracy_pct':round(100*paired.v8_strict_correct.mean(),3)},
        ])
    else: pair_summary=pd.DataFrame()
    pair_summary.to_csv(out/'same_games_summary.csv',index=False)
    write_report(out,result,audit,val,ret,subset,pair_summary,subgroup)
    print('Read-only historical evaluation complete. Selected:',result['selection'])
    print('2026 retrospective:',result['retrospective_selected'])
    print('Exact V8 overlapping games:',len(paired))
    print('Output:',out)


if __name__=='__main__': main()
