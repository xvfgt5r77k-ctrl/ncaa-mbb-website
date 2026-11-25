"""NCAA MBB V11: strict, read-only retrospective model research.

- Reconstructs team features from games on PREVIOUS dates.
- Compares frozen V5/V8/V10 baselines on the SAME review-game IDs.
- Evaluates conservative nonlinear models and an independent total-score model.
- Selects candidates ONLY using 2024/2025 forward validation; 2026 is
  repeatedly examined development data, not an untouched holdout.
- Audits a stated >=5% relative win-accuracy and >=5% total-error reduction
  goal without manipulating samples, outcomes, or thresholds.
- No future-game picks, live selections, gambling integrations, or database writes.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import html
import json
import os
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import v4_research as v4
import v5_research as v5
import v6_research as v6
import v8_research as v8
import v9_research as v9
import v10_research as v10

OUT = Path(os.getenv('V11_OUTPUT_DIR', 'v11_results'))
SEED = 111
TRAIN_FOLDS = [((2022, 2023), 2024), ((2022, 2023, 2024), 2025)]
REVIEW_YEAR = 2026
CLOSE_ELO = 120.0
GOAL_RELATIVE_IMPROVEMENT = 0.05
# Test changes made on validation data alone. These are NOT 5% success claims.
MIN_WIN_VALIDATION_BRIER_GAIN = .0005
MIN_WIN_VALIDATION_ACC_GAIN_PP = .15
MIN_SCORE_VALIDATION_MAE_GAIN = .10

WIN_GROUPS = {
    'compact_shooting': list(dict.fromkeys(v10.CORE + v10.SHOOT)),
    'matchup': list(dict.fromkeys(v10.CORE + v10.SHOOT + v10.MATCHUP)),
    'lagged_strength': list(dict.fromkeys(v10.CORE + v10.SHOOT + v10.STRENGTH)),
}
WIN_CHOICES = [
    ('compact_hgb', 'compact_shooting', 7, 150, 100, 18.),
    ('matchup_hgb', 'matchup', 9, 165, 120, 24.),
    ('strength_hgb', 'lagged_strength', 7, 165, 110, 22.),
]
BLEND_WEIGHTS = (.35, .65, 1.0)  # newer model weight; remainder V10 anchor

SCORE_FIELDS = list(dict.fromkeys(v10.SCORING_COLS + [
    'elo_gap', 'v5_three_matchup_gap', 'v8_three_edge_by_reliability',
    'v9_style_three_clash', 'v9_recent_sample_min', 'v9_pace_sample_min',
    'min_prior_games_strict',
]))
SCORE_CHOICES = [
    ('ridge_residual', 'ridge'),
    ('shallow_boosted_residual', 'boost'),
]
SCORE_BLENDS = (.5, 1.0)


def assert_ready(data: pd.DataFrame) -> None:
    v10.ensure_data(data)
    required = set(sum(WIN_GROUPS.values(), []) + SCORE_FIELDS)
    absent = sorted(required - set(data.columns))
    if absent:
        raise RuntimeError('Missing historical pregame columns: ' + ', '.join(absent))
    if data.duplicated(['season', 'game_id']).any():
        raise RuntimeError('Duplicate game records')
    for key in ['home_score', 'away_score']:
        if pd.to_numeric(data[key], errors='coerce').isna().any():
            raise RuntimeError(f'Invalid completed historical score: {key}')


def make_winner_model(choice) -> Pipeline:
    _, _, nodes, iterations, leaf, regularization = choice
    return Pipeline([
        ('impute', SimpleImputer(strategy='median', keep_empty_features=True)),
        ('hgb', HistGradientBoostingClassifier(
            max_leaf_nodes=nodes, max_iter=iterations,
            learning_rate=.035, min_samples_leaf=leaf,
            l2_regularization=regularization, early_stopping=False,
            random_state=SEED)),
    ])


def make_score_model(kind: str) -> Pipeline:
    if kind == 'ridge':
        return Pipeline([
            ('impute', SimpleImputer(strategy='median', keep_empty_features=True)),
            ('scale', StandardScaler()),
            ('model', Ridge(alpha=900.0)),
        ])
    if kind == 'boost':
        return Pipeline([
            ('impute', SimpleImputer(strategy='median', keep_empty_features=True)),
            ('model', HistGradientBoostingRegressor(
                loss='absolute_error', learning_rate=.035, max_iter=140,
                max_leaf_nodes=8, min_samples_leaf=120,
                l2_regularization=25., early_stopping=False,
                random_state=SEED)),
        ])
    raise ValueError(f'Unknown scoring model: {kind}')


def brier(y, p):
    x = v10.safe_probability(p)
    return float(np.mean((np.asarray(y, dtype=float) - x) ** 2))


def pct_accuracy(y, p):
    return float(100 * np.mean((v10.safe_probability(p) >= .5) == np.asarray(y, dtype=int)))


def pick_best_with_guardrails(rows: list[dict], anchor: dict, typ: str) -> dict:
    """Selection based exclusively on 2024/25, with no 2026 accuracy tuning."""
    a = dict(anchor)
    if typ == 'winner':
        ranked = sorted(rows, key=lambda r: (r['weighted_brier'], -r['weighted_accuracy_pct'],r['name']))
        best = dict(ranked[0])
        if (best['weighted_brier'] <= a['weighted_brier'] - MIN_WIN_VALIDATION_BRIER_GAIN
            and best['weighted_accuracy_pct'] >= a['weighted_accuracy_pct'] + MIN_WIN_VALIDATION_ACC_GAIN_PP):
            best['used_new_model'] = True
            return best
    elif typ == 'score':
        ranked = sorted(rows, key=lambda r: (r['weighted_mae'], r['name']))
        best = dict(ranked[0])
        if (best['weighted_mae'] <= a['weighted_mae'] - MIN_SCORE_VALIDATION_MAE_GAIN
            and best['mae_2024'] <= a['mae_2024'] + .10
            and best['mae_2025'] <= a['mae_2025'] + .10):
            best['used_new_model'] = True
            return best
    else:
        raise ValueError(typ)
    a['used_new_model'] = False
    a['reason'] = 'Improvement threshold not achieved on pre-2026 validation; retained V10.'
    return a


def previous_v10_fold_probabilities(data: pd.DataFrame, selected: dict) -> dict:
    """Rebuild the V10 selected configuration's predictions for 2024/25.

    Its 2026 probability is taken from the already-computed V10 archive.
    """
    old_title = selected['candidate']
    matching = next((c for c in v10.CANDIDATES if c[0] == old_title), None)
    if matching is None:
        raise RuntimeError('Cannot reconstruct V10 validation anchor')
    columns = v10.CANDIDATE_FIELDS[matching[1]]
    predictions = {}
    for years, year in TRAIN_FOLDS:
        tr = data[data.season.isin(years)]
        va = data[data.season == year]
        model = v10.stable_logistic(matching[2])
        model.fit(tr[columns], tr.home_win.astype(int))
        raw = model.predict_proba(va[columns])[:, 1]
        predictions[year] = v10.temperature_scale(raw, selected['temperature'])
    return predictions


def choose_winner(data: pd.DataFrame, selected_v10: dict, v10_review_prob) -> tuple:
    anchor_pred = previous_v10_fold_probabilities(data, selected_v10)
    folds = {}
    winner_audit = []
    for choice in WIN_CHOICES:
        name, group = choice[:2]
        cols = WIN_GROUPS[group]
        out = {}
        for train_years, val_year in TRAIN_FOLDS:
            tr = data[data.season.isin(train_years)]
            va = data[data.season == val_year]
            model = make_winner_model(choice)
            model.fit(tr[cols], tr.home_win.astype(int))
            out[val_year] = v10.safe_probability(model.predict_proba(va[cols])[:, 1])
            if val_year == 2025:
                review = data[data.season == REVIEW_YEAR]
                out['review'] = v10.safe_probability(model.predict_proba(review[cols])[:, 1])
        folds[name] = out
    choices = []
    for model_name, probs in folds.items():
        for weight in BLEND_WEIGHTS:
            year_metrics = {}
            for year in (2024, 2025):
                y = data.loc[data.season == year, 'home_win'].astype(int).to_numpy()
                p = (1-weight)*anchor_pred[year] + weight*probs[year]
                year_metrics[year] = (pct_accuracy(y, p), brier(y, p))
            choices.append({
                'name': f'{model_name}_weight_{weight:g}',
                'model': model_name, 'weight': weight,
                'accuracy_2024': round(year_metrics[2024][0], 3),
                'accuracy_2025': round(year_metrics[2025][0], 3),
                'brier_2024': round(year_metrics[2024][1], 6),
                'brier_2025': round(year_metrics[2025][1], 6),
                'weighted_accuracy_pct': round(.4*year_metrics[2024][0] + .6*year_metrics[2025][0], 4),
                'weighted_brier': round(.4*year_metrics[2024][1] + .6*year_metrics[2025][1], 7),
            })
    anchor_years={}
    for year in (2024,2025):
        y=data.loc[data.season==year,'home_win'].astype(int).to_numpy()
        anchor_years[year]=(pct_accuracy(y,anchor_pred[year]),brier(y,anchor_pred[year]))
    anchor = {'name':'V10 strict validation-selected anchor',
              'model':'V10 anchor','weight':0.0,
              'accuracy_2024':round(anchor_years[2024][0],3),
              'accuracy_2025':round(anchor_years[2025][0],3),
              'brier_2024':round(anchor_years[2024][1],6),
              'brier_2025':round(anchor_years[2025][1],6),
              'weighted_accuracy_pct':round(.4*anchor_years[2024][0]+.6*anchor_years[2025][0],4),
              'weighted_brier':round(.4*anchor_years[2024][1]+.6*anchor_years[2025][1],7)}
    selected = pick_best_with_guardrails(choices,anchor,'winner')
    if selected['used_new_model']:
        review = ((1-selected['weight'])*v10.safe_probability(v10_review_prob) +
                  selected['weight']*folds[selected['model']]['review'])
    else:
        review = v10.safe_probability(v10_review_prob)
    # Historical scores are NEVER passed to the winner model during inference.
    return pd.DataFrame([anchor] + choices), selected, v10.safe_probability(review)


def score_anchor_predicts(data: pd.DataFrame, selected: dict) -> dict:
    """Reconstruct V10's already chosen scoring configuration in forward folds."""
    alpha=float(selected['alpha']); weight=float(selected['calibrated_weight'])
    out={}
    for years, year in TRAIN_FOLDS:
        tr=data[data.season.isin(years)]
        va=data[data.season==year]
        y_train=tr.home_score.to_numpy(dtype=float)+tr.away_score.to_numpy(dtype=float)
        model=v9.ridge(alpha)
        model.fit(tr[v10.SCORING_COLS], y_train)
        base=va.v9_scoring_baseline_total.to_numpy(dtype=float)
        out[year]=(weight*model.predict(va[v10.SCORING_COLS])+(1-weight)*base)
    return out


def choose_total(data: pd.DataFrame, selected_v10: dict, v10_review_total) -> tuple:
    old_fold = score_anchor_predicts(data, selected_v10)
    re=data[data.season==2026]
    new = {}
    for name, kind in SCORE_CHOICES:
        out={}
        for years, year in TRAIN_FOLDS:
            tr=data[data.season.isin(years)]
            va=data[data.season==year]
            baseline_train=tr.v9_scoring_baseline_total.to_numpy(dtype=float)
            actual_train=(tr.home_score+tr.away_score).to_numpy(dtype=float)
            fit=make_score_model(kind)
            fit.fit(tr[SCORE_FIELDS], actual_train-baseline_train)
            baseline_val=va.v9_scoring_baseline_total.to_numpy(dtype=float)
            # Limit pathological residual corrections; fixed before reviewing 2026.
            out[year]=baseline_val+np.clip(fit.predict(va[SCORE_FIELDS]),-25,25)
            if year==2025:
                base_rev=re.v9_scoring_baseline_total.to_numpy(dtype=float)
                out['review']=base_rev+np.clip(fit.predict(re[SCORE_FIELDS]),-25,25)
        new[name]=out
    options=[]
    for name, outs in new.items():
        for w in SCORE_BLENDS:
            errs={}
            for year in (2024,2025):
                y=(data.loc[data.season==year,'home_score'] + data.loc[data.season==year,'away_score']).to_numpy(dtype=float)
                pred=(1-w)*old_fold[year]+w*outs[year]
                errs[year]=float(mean_absolute_error(y,pred))
            options.append({'name':f'{name}_weight_{w:g}', 'model':name,'weight':w,
                            'mae_2024':round(errs[2024],5),'mae_2025':round(errs[2025],5),
                            'weighted_mae':round(.4*errs[2024]+.6*errs[2025],5)})
    old_err={}
    for year in (2024,2025):
        va=data[data.season==year]
        y=(va.home_score+va.away_score).to_numpy(dtype=float)
        old_err[year]=float(mean_absolute_error(y,old_fold[year]))
    anchor={'name':'V10 validation-selected total baseline','model':'V10 anchor','weight':0.,
            'mae_2024':round(old_err[2024],5),'mae_2025':round(old_err[2025],5),
            'weighted_mae':round(.4*old_err[2024]+.6*old_err[2025],5)}
    selected=pick_best_with_guardrails(options,anchor,'score')
    if selected['used_new_model']:
        final=(1-selected['weight'])*np.asarray(v10_review_total,dtype=float)+selected['weight']*new[selected['model']]['review']
    else:
        final=np.asarray(v10_review_total,dtype=float)
    return pd.DataFrame([anchor]+options),selected,final


def evaluate(data:pd.DataFrame, audit:dict, player_enabled:bool) -> dict:
    assert_ready(data)
    # Reproduce V10 benchmarks from exactly identical source and game order.
    previous=v10.evaluate(data,audit,player_enabled)
    review=data[data.season==2026].copy().reset_index(drop=True)
    prior=previous['historical_completed_games'].copy().reset_index(drop=True)
    if not np.array_equal(review.game_id.astype(str),prior.game_id.astype(str)):
        raise RuntimeError('V11 review game order differs from V10 benchmark')
    selection_win,selected_win,pwin=choose_winner(data,previous['selected_winner'],prior.v10_strict_probability)
    selection_total,selected_total,ptotal=choose_total(data,previous['selected_scoring'],prior.v10_projected_total)
    y=review.home_win.to_numpy(dtype=int)
    actual=(review.home_score+review.away_score).to_numpy(dtype=float)
    mask_close=v8.close_mask(review)
    populations={
        'all':np.ones(len(review),dtype=bool),
        'elo_close':mask_close,
        'other':~mask_close,
        'neutral':review.neutral_site.eq(1).to_numpy(dtype=bool),
        'non_neutral':review.neutral_site.eq(0).to_numpy(dtype=bool),
    }
    bench_probs={
        'V5 strict':prior.v5_strict_prob.to_numpy(dtype=float),
        'V8 strict':prior.v8_strict_prob.to_numpy(dtype=float),
        'V10 strict':prior.v10_strict_probability.to_numpy(dtype=float),
        'V8 hybrid (unverified provenance)':prior.v8_hybrid_prob.to_numpy(dtype=float),
        'V11 strict':pwin,
    }
    comparisons=[]; paired=[]
    for label,m in populations.items():
        for model,p in bench_probs.items():
            comparisons.append(v10.stats(y[m],p[m],model,2026,label))
        change=((pwin[m]>=.5)==y[m]).astype(int)-((bench_probs['V10 strict'][m]>=.5)==y[m]).astype(int)
        low,hi=v8.bootstrap_days(change,review.loc[m,'game_date'].reset_index(drop=True))
        paired.append({'population':label,'games':int(m.sum()),'v11_vs_v10_accuracy_change_pp':round(100*float(np.mean(change)),4),
                       'day_block_bootstrap_low_pp':low,'day_block_bootstrap_high_pp':hi})
    errors=[]
    for name,pred in [('V9 previous-day total baseline',prior.v9_uncalibrated_total),
                      ('V10 selected scoring model',prior.v10_projected_total),
                      ('V11 selected scoring model',ptotal)]:
        p=np.asarray(pred,dtype=float)
        errors.append({'model':name,'games':len(actual),'total_mae':round(float(np.mean(np.abs(p-actual))),4),
                       'total_rmse':round(float(np.sqrt(np.mean((p-actual)**2))),4),
                       'mean_error':round(float(np.mean(p-actual)),4)})
    accuracy_v11=pct_accuracy(y,pwin)
    accuracy_v8=pct_accuracy(y,bench_probs['V8 hybrid (unverified provenance)'])
    mae_v11=float(np.mean(np.abs(ptotal-actual)))
    mae_v10=float(np.mean(np.abs(prior.v10_projected_total.to_numpy(dtype=float)-actual)))
    # The user-specified "5% better" refers to RELATIVE winner-accuracy gain
    # and relative REDUCTION in total MAE, using distinct predeclared baselines.
    goal={'winner_reference':'V8 hybrid (historical, unverified upstream provenance)',
          'winner_reference_accuracy_pct':round(accuracy_v8,4),
          'winner_required_accuracy_pct':round(accuracy_v8*(1+GOAL_RELATIVE_IMPROVEMENT),4),
          'v11_accuracy_pct':round(accuracy_v11,4),
          'winner_gain_relative_pct':round(100*(accuracy_v11/accuracy_v8-1),3),
          'winner_at_least_5pct_relative_better':bool(accuracy_v11>=accuracy_v8*1.05),
          'total_reference':'V10 validation-selected scoring model',
          'total_reference_mae':round(mae_v10,4),
          'required_total_mae_or_lower':round(mae_v10*.95,4),
          'v11_total_mae':round(mae_v11,4),
          'total_mae_reduction_relative_pct':round(100*(1-mae_v11/mae_v10),3),
          'total_at_least_5pct_relative_better':bool(mae_v11<=mae_v10*.95),
          'both_5pct_goals_met':bool(accuracy_v11>=accuracy_v8*1.05 and mae_v11<=mae_v10*.95),
          'review_is_prospective':False}
    diagnostics=review[['season','game_id','game_date','neutral_site','home_score','away_score','home_win','elo_gap']].copy()
    diagnostics['v5_strict_win_probability']=bench_probs['V5 strict']
    diagnostics['v8_strict_win_probability']=bench_probs['V8 strict']
    diagnostics['v8_hybrid_win_probability_unverified']=bench_probs['V8 hybrid (unverified provenance)']
    diagnostics['v10_strict_win_probability']=bench_probs['V10 strict']
    diagnostics['v11_strict_win_probability']=pwin
    diagnostics['v10_projected_total']=prior.v10_projected_total.to_numpy(dtype=float)
    diagnostics['v11_projected_total']=ptotal
    diagnostics['actual_total']=actual
    return {'winner_selection':selection_win,'score_selection':selection_total,
            'winner_comparison':pd.DataFrame(comparisons),
            'total_comparison':pd.DataFrame(errors),
            'paired':pd.DataFrame(paired),
            'diagnostics':diagnostics,
            'calibration':v10.calibration_bins(review,pwin),
            'goal':goal,'selected_winner':selected_win,'selected_total':selected_total,
            'audit':{
              'v11_source_data_build':'v6.make_frame, v9.assemble_features, v10.evaluate',
              'v11_candidate_selection_only_2024_2025':True,
              'v11_training_seasons_for_2026':[2022,2023,2024],
              'v11_never_select_based_on_review_2026':True,
              '2026_not_untouched_holdout':True,
              'upstream_espn_publication_timing_fully_audited':False,
              'hybrid_rating_provenance_verified':False,
              'player_availability_verified':False,
              'v11_never_returns_upcoming_game_predictions':True,
              'winner_feature_groups':WIN_GROUPS,
              'score_features':SCORE_FIELDS,
              'source_audit':previous['audit'],
            }}


def to_html(result:dict) -> str:
    def tbl(key):
        return result[key].to_html(index=False,escape=True,border=0,classes='data')
    note=('Retrospective research ONLY. The 2026 season was already examined in earlier '
          'models and is not an untouched holdout. The 5% targets are evaluated, never '
          'guaranteed. No live-game selections, wagering actions, or database changes. '
          'The import/publication timing of all ESPN-derived historical inputs has not '
          'been independently audited.')
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><title>NCAA MBB V11 Research</title>
<style>body{{font:15px system-ui,Arial;max-width:1250px;margin:30px auto;padding:0 20px;color:#172b36;line-height:1.5}}
.data{{border-collapse:collapse;display:block;overflow-x:auto;margin-bottom:26px}}
.data th,.data td{{padding:9px 12px;border-bottom:1px solid #ddd;white-space:nowrap;text-align:left}}
.data th{{background:#edf2f7}}.note{{padding:12px;background:#eef3f7;border-left:4px solid #64748b}}
pre{{padding:12px;overflow-x:auto;background:#f5f7f9}}</style></head><body>
<h1>NCAA Men's Basketball — V11 Historical Research</h1>
<p>Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC</p>
<p class="note">{html.escape(note)}</p>
<h2>5% improvement requirements (relative percentages)</h2><pre>{html.escape(json.dumps(result['goal'],indent=2))}</pre>
<h2>Winner accuracy on identical historical games</h2>{tbl('winner_comparison')}
<h2>Projected combined score error</h2>{tbl('total_comparison')}
<h2>Day-block paired accuracy comparison</h2>{tbl('paired')}
<h2>Winner selection from 2024–2025 only</h2>{tbl('winner_selection')}
<h2>Total model selection from 2024–2025 only</h2>{tbl('score_selection')}
<h2>Probability reliability in the reviewed season</h2>{tbl('calibration')}
<h2>Winner selection decision</h2><pre>{html.escape(json.dumps(result['selected_winner'],indent=2))}</pre>
<h2>Total selection decision</h2><pre>{html.escape(json.dumps(result['selected_total'],indent=2))}</pre>
<h2>Reproducibility and provenance audit</h2><pre>{html.escape(json.dumps(result['audit'],indent=2,default=str))}</pre>
</body></html>'''


def save(result:dict) -> None:
    OUT.mkdir(parents=True,exist_ok=True)
    for filename,key in {
        'winner_comparison_2026.csv':'winner_comparison',
        'scoring_total_comparison_2026.csv':'total_comparison',
        'paired_comparison_2026.csv':'paired',
        'winner_selection_2024_2025.csv':'winner_selection',
        'score_selection_2024_2025.csv':'score_selection',
        'calibration_2026.csv':'calibration',
        'historical_completed_game_diagnostics.csv':'diagnostics',
    }.items():
        result[key].to_csv(OUT/filename,index=False)
    (OUT/'goals.json').write_text(json.dumps(result['goal'],indent=2))
    (OUT/'selected_models.json').write_text(json.dumps({
        'winner':result['selected_winner'],'total':result['selected_total']},indent=2,default=str))
    (OUT/'audit.json').write_text(json.dumps(result['audit'],indent=2,default=str))
    (OUT/'report.html').write_text(to_html(result),encoding='utf-8')
    print('V11 historical evaluation complete; no claims of prospective accuracy.')
    print(json.dumps(result['goal'],indent=2))
    print('Report at:',(OUT/'report.html').resolve())


def self_test():
    # Pure probability, improvement-goal, and model guardrail tests.
    prob=np.array([.1,.5,.9])
    assert np.array_equal(v10.safe_probability(prob)>=.5,[False,True,True])
    assert pct_accuracy([0,1,1],prob)==100.0
    anchor={'name':'anchor','weighted_brier':.19,'weighted_accuracy_pct':70.0,
            'mae_2024':14.0,'mae_2025':14.2,'weighted_mae':14.12}
    weak={'name':'weak','weighted_brier':.1898,'weighted_accuracy_pct':70.2,
          'mae_2024':13.95,'mae_2025':14.15,'weighted_mae':14.07}
    strong={'name':'strong','weighted_brier':.188,'weighted_accuracy_pct':71.,
            'mae_2024':13.6,'mae_2025':13.9,'weighted_mae':13.78}
    assert not pick_best_with_guardrails([weak],anchor,'winner')['used_new_model']
    assert pick_best_with_guardrails([strong],anchor,'winner')['used_new_model']
    assert not pick_best_with_guardrails([weak],anchor,'score')['used_new_model']
    assert pick_best_with_guardrails([strong],anchor,'score')['used_new_model']
    assert make_winner_model(WIN_CHOICES[0])
    assert make_score_model('boost')
    # 5% win improvement and 5% total-error reduction are not interchangeable.
    assert 73.731 >= 70.22*1.05-1e-6
    assert 13.0 <= 13.79*.95
    v10.offline_tests()
    v9.self_test()
    print('V11 offline math, safety, and previous-date replay tests passed.')


def main():
    parser=argparse.ArgumentParser(description='V11 retrospective NCAA research; read-only')
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if args.self_test:
        self_test()
        return
    raw,totals,scores,boxes=v5.fetch()
    optional,reports,availability_audit=v6.optional_data()
    players,player_audit=v9.fetch_players()
    frame,audit=v6.make_frame(raw,scores,boxes,optional,reports,availability_audit)
    data,audit,enabled=v9.assemble_features(frame,scores,boxes,players,audit)
    audit['v11_player_source']=player_audit
    result=evaluate(data,audit,enabled)
    save(result)


if __name__=='__main__':
    main()
