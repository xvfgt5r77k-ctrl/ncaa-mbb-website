"""NCAA MBB V10: read-only, completed-game historical MODEL AUDIT.

V10 deliberately favors stable pregame-only features over feature accumulation.
- Replays V5/V8/V9 benchmarks on the SAME completed historical games.
- Fits simple candidate classifiers on 2022-23 and 2022-24.
- Uses 2024 and 2025 to select regularization and optional probability shrinkage.
- Reviews selected model in 2026 (ALREADY EXAMINED: not a fresh holdout).
- Separately evaluates a possession-based, uncertainty-aware scoring total model.
- Audits close-game accuracy, calibrated probability bins, missingness and provenance.

No upcoming-game selections, live forecasts, wagering integration, database writes,
or dashboard modifications. Point-in-time source provenance is not fully audited.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import html
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import v4_research as v4
import v5_research as v5
import v6_research as v6
import v8_research as v8
import v9_research as v9

OUTPUT = Path(os.getenv('V10_OUTPUT_DIR', 'v10_results'))
RANDOM_SEED = 110
TRAIN_SEASONS = (2022, 2023, 2024)
VALIDATION_YEARS = (2024, 2025)
REVIEW_SEASON = 2026
CLOSE_ELO_THRESHOLD = 120.0  # predeclared; NOT tuned on reviewed games

# Small predeclared, point-in-time reconstructed feature groups.  Exclude
# v3.V3_FEATURES / V5 HYBRID (unverified upstream feature provenance).
CORE = [
    'elo_gap', 'elo_home_probability', 'strict_ppg_margin', 'strict_margin_gap',
    'past_opponent_elo_gap', 'recent5_margin_gap', 'rest_gap',
    'min_prior_games_strict', 'neutral_site',
]
SHOOT = [
    'v5_three_matchup_gap', 'v5_three_attempt_share_gap',
    'v5_three_volatility_sum', 'v5_shooting_games_min',
]
MATCHUP = [
    'v8_three_edge_by_reliability', 'v8_two_point_clash',
    'v8_shot_turnover_interaction', 'v8_shot_rebounding_interaction',
]
STRENGTH = [
    'v9_reliable_adj_net_gap', 'v9_strength_uncertainty',
    'v9_style_three_clash', 'v9_form_weighted_gap',
]
CANDIDATE_FIELDS = {
    'stable_core': CORE,
    'core_plus_shooting': CORE + SHOOT,
    'core_plus_matchups': CORE + SHOOT + MATCHUP,
    'core_plus_lagged_strength': CORE + SHOOT + STRENGTH,
}
# Deliberately limited model search to reduce historical overfitting.
CANDIDATES = [
    ('stable_core_C0.3', 'stable_core', .3),
    ('shoot_C0.1', 'core_plus_shooting', .1),
    ('shoot_C1', 'core_plus_shooting', 1.),
    ('matchup_C0.1', 'core_plus_matchups', .1),
    ('matchup_C1', 'core_plus_matchups', 1.),
    ('strength_C0.1', 'core_plus_lagged_strength', .1),
    ('strength_C1', 'core_plus_lagged_strength', 1.),
]
# Temperature > 1 shrinks extreme probabilities toward 50%. No calibration
# curve fitted to the 2026 outcome labels.
TEMPERATURES = (1.0, 1.10, 1.25)
SCORING_COLS = [
    'v9_scoring_baseline_total', 'v9_expected_possessions', 'v9_pace_gap',
    'v9_form_weighted_gap', 'v9_strength_uncertainty', 'neutral_site',
    'v5_three_volatility_sum',
]
SCORE_ALPHAS = (100., 1000., 3000.)
# The baseline is allowed to win if no calibrated regression improves validation MAE.
SCORE_WEIGHTS = (0.0, 0.5, 1.0)


def stable_logistic(c: float) -> Pipeline:
    return Pipeline([
        ('fill', SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=True)),
        ('scale', StandardScaler()),
        ('model', LogisticRegression(C=c, max_iter=2500, random_state=RANDOM_SEED)),
    ])


def safe_probability(p) -> np.ndarray:
    arr = np.asarray(p, dtype=float)
    if not np.isfinite(arr).all():
        raise RuntimeError('Non-finite model probabilities: no silent replacement')
    return np.clip(arr, 1e-6, 1-1e-6)


def temperature_scale(prob, temperature):
    p = safe_probability(prob)
    logit = np.log(p / (1-p))
    z = np.clip(logit/float(temperature), -25, 25)
    return safe_probability(1/(1+np.exp(-z)))


def stats(y, probs, label, season=2026, population='all'):
    yy = np.asarray(y, dtype=int)
    pp = safe_probability(probs)
    if len(pp) != len(yy):
        raise ValueError('Probability and outcome row count mismatch')
    if not len(yy):
        return {'model':label,'season':season,'population':population,'games':0,
                'correct':0,'winner_accuracy_pct':None,'brier_score':None,'log_loss':None}
    return {'model':label, 'season':season, 'population':population,
            'games':len(yy), 'correct':int(np.sum((pp>=.5)==yy)),
            'winner_accuracy_pct':round(float(100*accuracy_score(yy,pp>=.5)),3),
            'brier_score':round(float(brier_score_loss(yy,pp)),6),
            'log_loss':round(float(log_loss(yy,pp,labels=[0,1])),6)}


def ensure_data(data):
    if data.duplicated(['season','game_id']).any():
        raise RuntimeError('Ambiguous game keys: duplicate season/game_id')
    for name in set(sum(CANDIDATE_FIELDS.values(),[]) + SCORING_COLS +
                    ['game_date','home_win','home_score','away_score','game_id','season']):
        if name not in data:
            raise RuntimeError(f'V10 requires feature {name!r} from V9 replay pipeline')
    sizes = data.groupby('season').size().to_dict()
    for y in [2022,2023,2024,2025,2026]:
        if sizes.get(y,0)<150:
            raise RuntimeError(f'Missing or insufficient data in {y}: {sizes.get(y,0)}')
    if not data['home_win'].isin([0,1]).all():
        raise RuntimeError('Invalid winner labels')
    if pd.to_datetime(data['game_date'],errors='coerce').isna().any():
        raise RuntimeError('Missing game_date; cannot group errors chronologically')


def choose_classification(data):
    """Select ONLY on 2024 + 2025 time-forward folds, never on 2026."""
    ensure_data(data)
    # Two forward folds: (2022-23)->2024 and (2022-24)->2025.
    folds = [((2022,2023), 2024), ((2022,2023,2024),2025)]
    train_predicts = {}; candidates = []
    for title, group, c in CANDIDATES:
        columns = CANDIDATE_FIELDS[group]
        generated = {}
        for train_years, validation_year in folds:
            tr = data[data.season.isin(train_years)]
            va = data[data.season==validation_year]
            clf = stable_logistic(c)
            clf.fit(tr[columns],tr.home_win.astype(int))
            generated[validation_year] = (
                va.home_win.astype(int).to_numpy(),
                safe_probability(clf.predict_proba(va[columns])[:,1]))
            # This is EXACTLY the 2022-24 model eligible to evaluate 2026.
            if validation_year==2025:
                review=data[data.season==REVIEW_SEASON]
                generated['review_raw']=safe_probability(clf.predict_proba(review[columns])[:,1])
        for temp in TEMPERATURES:
            for year in VALIDATION_YEARS:
                yy, pp = generated[year]
                stat = stats(yy,temperature_scale(pp,temp),f'{title}_T{temp:g}',season=year)
                candidates.append({'candidate':title,'feature_group':group,'features':len(columns),
                  'regularization_C':c,'temperature':temp,'season':year,
                  **{k:stat[k] for k in ('games','winner_accuracy_pct','brier_score','log_loss')}})
        train_predicts[title] = generated
    leaderboard=pd.DataFrame(candidates)
    # Avoid indirect selection on the reviewed 2026 season. The selection metric
    # gives 2025 somewhat more weight and favors compact feature sets on ties.
    ranking=[]
    for (title,temp), rows in leaderboard.groupby(['candidate','temperature'],sort=False):
        by_year=rows.set_index('season')
        score=.4*float(by_year.loc[2024,'brier_score'])+.6*float(by_year.loc[2025,'brier_score'])
        accuracy=.4*float(by_year.loc[2024,'winner_accuracy_pct'])+.6*float(by_year.loc[2025,'winner_accuracy_pct'])
        candidate=next(x for x in CANDIDATES if x[0]==title)
        ranking.append({'candidate':title,'temperature':float(temp),'combined_brier':score,
                        'combined_accuracy_pct':accuracy,'features':len(CANDIDATE_FIELDS[candidate[1]])})
    ranking=pd.DataFrame(ranking).sort_values(['combined_brier','features','candidate','temperature'])
    selected=ranking.iloc[0].to_dict()
    # Guardrail: if a complex challenger fails to improve the simple anchor's
    # Brier score by at least .0004, retain the anchor (including its own best T).
    anchors=ranking[ranking.candidate=='stable_core_C0.3']
    anchor=anchors.iloc[0].to_dict()
    if (selected['candidate']!='stable_core_C0.3' and
        (selected['combined_brier']>anchor['combined_brier']-.0004 or
         selected['combined_accuracy_pct']<anchor['combined_accuracy_pct']-.30)):
        selected=anchor.copy()
        selected['guardrail_reverted_to_anchor']=True
    else:
        selected['guardrail_reverted_to_anchor']=False
    raw_review=train_predicts[selected['candidate']]['review_raw']
    p_review=temperature_scale(raw_review,selected['temperature'])
    return leaderboard, ranking, selected, p_review


def calibrated_score_model(alpha):
    return v9.ridge(alpha)


def choose_scoring(data):
    """Conservative total-score regression; select before reviewing 2026."""
    # V9 replay constructs this baseline from each team's previous-day games.
    choices=[]; outputs={}
    for alpha in SCORE_ALPHAS:
        predicts={}
        for train_years,validation in [((2022,2023),2024),((2022,2023,2024),2025)]:
            tr=data[data.season.isin(train_years)]
            va=data[data.season==validation]
            y=tr.home_score.astype(float).to_numpy()+tr.away_score.astype(float).to_numpy()
            model=calibrated_score_model(alpha)
            model.fit(tr[SCORING_COLS],y)
            regression=np.asarray(model.predict(va[SCORING_COLS]),dtype=float)
            base=va.v9_scoring_baseline_total.astype(float).to_numpy()
            predicts[validation]=(regression,base,va.home_score.astype(float).to_numpy()+va.away_score.astype(float).to_numpy())
            if validation==2025:
                review=data[data.season==2026]
                predicts['review']=(np.asarray(model.predict(review[SCORING_COLS]),dtype=float),
                   review.v9_scoring_baseline_total.astype(float).to_numpy(),
                   review.home_score.astype(float).to_numpy()+review.away_score.astype(float).to_numpy())
        for w in SCORE_WEIGHTS:
            residuals=[]; maelist=[]
            for year in VALIDATION_YEARS:
                model_pred,baseline,y=predicts[year]
                actual_pred=w*model_pred+(1-w)*baseline
                maelist.append(float(mean_absolute_error(y,actual_pred)))
                residuals.extend((y-actual_pred).tolist())
            choices.append({'alpha':alpha,'calibrated_weight':w,
                    'mae_2024':round(maelist[0],4),'mae_2025':round(maelist[1],4),
                    'weighted_validation_mae':.4*maelist[0]+.6*maelist[1],
                    'abs_error_p80':float(np.quantile(np.abs(residuals),.8))})
        outputs[alpha]=predicts
    search=pd.DataFrame(choices).sort_values(['weighted_validation_mae','calibrated_weight','alpha'])
    selected=search.iloc[0].to_dict()
    a=selected['alpha']; w=selected['calibrated_weight']
    model_pred,base,actual=outputs[a]['review']
    pred=w*model_pred+(1-w)*base
    half_width=selected['abs_error_p80']
    metrics=pd.DataFrame([
     {'model':'V9 raw prior-day possession baseline','games':len(actual),
       'total_mae':round(float(mean_absolute_error(actual,base)),3)},
     {'model':'V10 validation-selected scoring estimate','games':len(actual),
       'total_mae':round(float(mean_absolute_error(actual,pred)),3),
       'review_80pct_interval_coverage_pct':round(float(100*np.mean(np.abs(actual-pred)<=half_width)),2),
       'interval_half_width_points':round(float(half_width),2)},
    ])
    return search,selected,metrics,pred,base,actual


def calibration_bins(frame, probs):
    out=frame[['season','game_id','game_date','home_win']].copy().reset_index(drop=True)
    out['home_win_probability']=safe_probability(probs)
    bins=[0,.2,.4,.6,.8,1.0001]
    out['probability_bin']=pd.cut(out.home_win_probability,bins=bins,include_lowest=True)
    grouped=out.groupby('probability_bin',observed=True).agg(
        games=('home_win','size'),average_projected_home_win=('home_win_probability','mean'),
        observed_home_win_rate=('home_win','mean')).reset_index()
    grouped['probability_bin']=grouped.probability_bin.astype(str)
    for col in ('average_projected_home_win','observed_home_win_rate'):
        grouped[col]=(100*grouped[col]).round(2)
    return grouped


def evaluate(data,audit,player_enabled):
    ensure_data(data)
    # Benchmarks are rebuilt from the exact same game IDs. The V5/V8 Hybrid
    # columns are included as exploratory comparisons only (provenance caveat).
    comparisons_v9,_,_,_,scores_v9,_,hist,_,old_audit=v9.evaluate(data,audit,player_enabled)
    review=data[data.season==2026].copy().reset_index(drop=True)
    if not np.array_equal(hist.game_id.astype(str).to_numpy(), review.game_id.astype(str).to_numpy()):
        raise RuntimeError('V9 and V10 game IDs not in the same order')
    selection,rank,chosen,p_v10=choose_classification(data)
    y=review.home_win.astype(int).to_numpy()
    masks={'all':np.ones(len(review),dtype=bool),
           'elo_close':v8.close_mask(review), 'other_games':~v8.close_mask(review),
           'neutral':review.neutral_site.eq(1).to_numpy(dtype=bool),
           'non_neutral':review.neutral_site.eq(0).to_numpy(dtype=bool)}
    baseline_cols={'V5 strict':'v5_strict_prob','V8 strict':'v8_strict_prob',
                   'V9 strict':'v9_strict_home_win_prob','V5 hybrid (unverified upstream)':'v5_hybrid_prob',
                   'V8 hybrid (unverified upstream)':'v8_hybrid_prob',
                   'V9 hybrid (unverified upstream)':'v9_hybrid_home_win_prob'}
    # Names of columns in V9 exported frame are historical experimental results.
    for lbl,col in baseline_cols.items():
        if col not in hist:
            raise RuntimeError(f'Missing V9 comparison column {col}')
    metrics=[]; paired=[]
    archive=review[['season','game_id','game_date','neutral_site','home_score','away_score','home_win','elo_gap']].copy()
    archive['v10_strict_probability']=p_v10
    for label,col in baseline_cols.items():
        archive[col]=hist[col].to_numpy(dtype=float)
    for population,mask in masks.items():
        for label,probs in [('V10 strict',p_v10)]+[(label,hist[col].to_numpy(dtype=float)) for label,col in baseline_cols.items()]:
            metrics.append(stats(y[mask],probs[mask],label,2026,population))
        newest=(p_v10[mask]>=.5)==y[mask]
        for label in ('V5 strict','V8 strict','V9 strict'):
            old=(hist[baseline_cols[label]].to_numpy(dtype=float)[mask]>=.5)==y[mask]
            delta=newest.astype(int)-old.astype(int)
            low,high=v8.bootstrap_days(delta,review.loc[mask,'game_date'].reset_index(drop=True))
            paired.append({'comparison':'V10 strict - '+label,'population':population,
                'games':int(mask.sum()),'accuracy_change_pp':round(float(np.mean(delta)*100),3) if len(delta) else None,
                'day_block_bootstrap_low_pp':low,'day_block_bootstrap_high_pp':high,
                'v10_right_old_wrong':int(np.sum(newest & ~old)),
                'old_right_v10_wrong':int(np.sum(old & ~newest))})
    scoring_search,scoring_selected,scoring_results,predicted_total,baseline_total,actual_total=choose_scoring(data)
    archive['v10_projected_total']=predicted_total
    archive['v9_uncalibrated_total']=baseline_total
    archive['actual_total']=actual_total
    archive['absolute_total_error']=np.abs(predicted_total-actual_total)
    return {'comparison':pd.DataFrame(metrics),'paired':pd.DataFrame(paired),
            'model_selection_folds':selection,'model_selection_rank':rank,
            'scoring_selection':scoring_search,'scoring_comparison':scoring_results,
            'calibration':calibration_bins(review,p_v10),
            'historical_completed_games':archive,'selected_winner':chosen,
            'selected_scoring':scoring_selected,
            'audit':{**old_audit,'v10_selected_from_years':[2024,2025],
              'v10_2026_outcomes_used_for_selection':False,
              'v10_2026_untouched_holdout':False,
              'v10_hybrid_upstream_provenance_verified':False,
              'v10_player_availability_verified':False,
              'v10_close_game_definition':'abs(previous-day Elo gap) <= 120',
              'v10_score_interval_is_validation_error_quantile_not_prospective_guarantee':True}}


def report_html(result):
    def tbl(d): return d.to_html(index=False,border=0,escape=True,classes='data')
    warning=("Retrospective research ONLY. 2026 has been repeatedly examined; not an independent holdout. "
             "All 2026 scores are used solely for retrospective metrics. Historical ESPN game-box import "
             "timing and V5/V8 Hybrid upstream ratings are not independently verified point-in-time. "
             "This tool does not issue upcoming-game selections or operate a betting system.")
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><title>NCAA MBB V10 Research</title>
<style>body{{font:15px system-ui,Arial;max-width:1200px;margin:35px auto;padding:0 22px;color:#192935;line-height:1.5}}
h1,h2{{line-height:1.2}}.note{{background:#f0f4f8;padding:14px;border-left:4px solid #52637c}}
.data{{border-collapse:collapse;display:block;overflow-x:auto;margin:12px 0 32px}}
.data th,.data td{{padding:9px 12px;text-align:left;white-space:nowrap;border-bottom:1px solid #dae1e9}}
.data th{{background:#eff3f7}}pre{{background:#f4f6f8;padding:12px;overflow:auto}}</style></head><body>
<h1>NCAA MBB V10 — Stability, Calibration, and Chronological Evaluation</h1>
<p>Created {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC</p>
<div class="note">{html.escape(warning)}</div>
<h2>Same-game winner accuracy (2026 historical review)</h2>{tbl(result['comparison'])}
<h2>Paired V10 strict versus prior strict models; day-block uncertainty</h2>{tbl(result['paired'])}
<h2>2026 probability calibration</h2>{tbl(result['calibration'])}
<h2>Scoring totals (2026) and interval coverage</h2>{tbl(result['scoring_comparison'])}
<h2>Winner model selection: 2024 and 2025 only</h2>{tbl(result['model_selection_rank'])}
<h2>Scoring model selection: 2024 and 2025 only</h2>{tbl(result['scoring_selection'])}
<h2>Selected winner configuration</h2><pre>{html.escape(json.dumps(result['selected_winner'],indent=2))}</pre>
<h2>Selected scoring configuration</h2><pre>{html.escape(json.dumps(result['selected_scoring'],indent=2))}</pre>
<h2>Audit and limitations</h2><pre>{html.escape(json.dumps(result['audit'],indent=2,default=str))}</pre>
<p>Do not select another revision just because its 2026 score is higher. Use prospective frozen forecasts to establish genuine reliability.</p></body></html>'''


def write_output(result):
    OUTPUT.mkdir(parents=True,exist_ok=True)
    names={'model_comparison.csv':'comparison','paired_differences_2026.csv':'paired',
           'classification_2024_2025_selection.csv':'model_selection_folds',
           'classification_candidate_rank.csv':'model_selection_rank',
           'score_total_2024_2025_selection.csv':'scoring_selection',
           'score_total_evaluation.csv':'scoring_comparison',
           'probability_calibration_2026.csv':'calibration',
           'historical_2026_completed_game_diagnostics.csv':'historical_completed_games'}
    for filename,key in names.items():
        result[key].to_csv(OUTPUT/filename,index=False)
    summary={'selected_winner':result['selected_winner'],
             'selected_scoring':result['selected_scoring'],
             'winner_accuracy_2026':result['comparison'].query("population == 'all'").to_dict('records'),
             'score_total':result['scoring_comparison'].to_dict('records'),
             'prospective_accuracy_established':False,
             'label_2026_untouched_holdout':False}
    (OUTPUT/'summary.json').write_text(json.dumps(summary,indent=2,default=str),encoding='utf-8')
    (OUTPUT/'audit.json').write_text(json.dumps(result['audit'],indent=2,default=str),encoding='utf-8')
    (OUTPUT/'report.html').write_text(report_html(result),encoding='utf-8')
    print('V10 retrospective research completed. Not a prospective winner-prediction feed.')
    print(result['comparison'].query("population == 'all'").to_string(index=False))
    print('Scoring totals:')
    print(result['scoring_comparison'].to_string(index=False))
    print('Report:',(OUTPUT/'report.html').resolve())


def offline_tests():
    assert np.allclose(temperature_scale([.5,.8],1.), [.5,.8])
    assert np.all(temperature_scale([.1,.9],1.25) > 0)
    assert temperature_scale([.9],1.25)[0] < .9
    sample=pd.DataFrame({'season':[2024,2024,2025], 'game_id':['a','b','c'],
                         'game_date':['2024-01-01','2024-01-01','2025-01-01'],
                         'home_win':[1,0,1]})
    assert calibration_bins(sample,[.8,.3,.6]).games.sum()==3
    assert stats([1,0],[.9,.2],'test')['correct']==2
    assert np.isfinite(stats([1,0],[.9,.2],'test')['brier_score'])
    assert v8.close_mask(pd.DataFrame({'elo_gap':[0.,140.,-100.]})).tolist()==[True,False,True]
    print('V10 offline math, confidence, close-group, and comparison tests passed (synthetic).')


def run(raw, totals, scores, boxes, optional, reports, availability_audit, players, player_audit):
    frame,audit=v6.make_frame(raw,scores,boxes,optional,reports,availability_audit)
    data,audit,enabled=v9.assemble_features(frame,scores,boxes,players,audit)
    audit['v10_player_source']=player_audit
    result=evaluate(data,audit,enabled)
    write_output(result)
    return result


def main():
    parser=argparse.ArgumentParser(description='V10 read-only historical research')
    parser.add_argument('--self-test',action='store_true')
    a=parser.parse_args()
    if a.self_test:
        offline_tests(); v8.self_test(); v9.self_test()
        return
    raw, totals, scores, boxes=v5.fetch()
    optional, reports, availability_audit=v6.optional_data()
    players, player_audit=v9.fetch_players()
    run(raw,totals,scores,boxes,optional,reports,availability_audit,players,player_audit)


if __name__=='__main__':
    main()
