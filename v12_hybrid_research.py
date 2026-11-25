"""NCAA MBB V12: guarded multi-model research ensemble.

Historical completed games only. No future-game predictions, wagering logic,
API keys in output, or database writes. Uses historical team inputs replayed
strictly from prior calendar days; imported upstream timestamps remain unaudited.

Train 2022-23 -> validate 2024; train 2022-24 -> validate 2025;
select blend on the validation years only and REVIEW 2026 without selection.
2026 is not an untouched holdout (earlier model designs were informed by it).
"""
from __future__ import annotations

import argparse
import html
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

import v4_research as v4
import v5_research as v5
import v6_research as v6
import v8_research as v8
import v9_research as v9
import v10_research as v10
import v11_research as v11

OUTPUT = Path(os.getenv('V12_OUTPUT_DIR', 'v12_results'))
REFERENCE = Path(__file__).with_name('reference_2026_v11.csv')
FOLDS = [((2022, 2023), 2024), ((2022, 2023, 2024), 2025)]
WEIGHTS_YEARS = {2024: 0.40, 2025: 0.60}
REVIEW = 2026
SEED = 1212

# Stable models that represent the most useful earlier-version families.
# These are not the frozen configurations from each archived research version;
# the exported old-version probabilities remain separate reference benchmarks.
WIN_COMPONENTS = {
    'elo_strength': ['elo_gap', 'elo_home_probability', 'neutral_site'],
    'v5_shooting': list(dict.fromkeys(v5.STRICT_FIELDS)),
    'v8_matchups': list(dict.fromkeys(v5.STRICT_FIELDS + v8.MATCHUP_FIELDS)),
    'v10_stability': list(dict.fromkeys(v10.CORE + v10.SHOOT)),
    'v11_nonlinear': list(dict.fromkeys(v11.WIN_GROUPS['compact_shooting'])),
    'exploratory_v8_hybrid': list(dict.fromkeys(v5.HYBRID_FIELDS + v8.MATCHUP_FIELDS)),
}
STRICT_WIN_NAMES = ('elo_strength', 'v5_shooting', 'v8_matchups', 'v10_stability', 'v11_nonlinear')
EXPLORATORY = 'exploratory_v8_hybrid'  # not eligible to become the recommended model

# PREDECLARED weights, selected without consulting 2026 outcomes.
WIN_BLENDS = {
    'V12 equal strict blend': [0.20, 0.20, 0.20, 0.20, 0.20],
    'V12 stable strength blend': [0.20, 0.20, 0.20, 0.30, 0.10],
    'V12 strong matchups blend': [0.10, 0.25, 0.30, 0.25, 0.10],
    'V12 regularized blend': [0.15, 0.15, 0.25, 0.40, 0.05],
    'V12 linear-only blend': [0.15, 0.25, 0.25, 0.35, 0.00],
    'V12 robust older mix': [0.25, 0.30, 0.15, 0.30, 0.00],
}

SCORE_NAMES = ('v9_pace_baseline', 'v10_ridge', 'v11_residual_ridge', 'v11_residual_boost')
SCORE_BLENDS = {
    'V12 equal scoring blend': [0.25, 0.25, 0.25, 0.25],
    'V12 calibrated scoring blend': [0.10, 0.40, 0.35, 0.15],
    'V12 defensive scoring blend': [0.15, 0.60, 0.25, 0.00],
    'V12 smooth scoring blend': [0.30, 0.40, 0.30, 0.00],
    'V12 residual scoring blend': [0.10, 0.25, 0.45, 0.20],
}


def validate_weights(weights: dict[str, list[float]], component_names: tuple[str, ...]):
    for label, vector in weights.items():
        arr = np.asarray(vector, dtype=float)
        if len(arr) != len(component_names) or (arr < 0).any() or not np.isclose(arr.sum(), 1.0):
            raise ValueError(f'Invalid fixed weights in {label}: {vector}')


def model_for(name: str):
    if name == 'elo_strength':
        return v10.stable_logistic(.30)
    if name == 'v5_shooting':
        return v4.logistic(3.0)
    if name == 'v8_matchups':
        return v4.logistic(.30)
    if name == 'v10_stability':
        return v10.stable_logistic(1.0)
    if name == 'v11_nonlinear':
        return v11.make_winner_model(v11.WIN_CHOICES[0])
    if name == EXPLORATORY:
        return v4.logistic(.30)
    raise ValueError(f'Unknown winner component: {name}')


def score_model_for(name: str):
    if name == 'v10_ridge':
        return v9.ridge(1000.0), v10.SCORING_COLS, False
    if name == 'v11_residual_ridge':
        return v11.make_score_model('ridge'), v11.SCORE_FIELDS, True
    if name == 'v11_residual_boost':
        return v11.make_score_model('boost'), v11.SCORE_FIELDS, True
    raise ValueError(name)


def score_target(df: pd.DataFrame) -> np.ndarray:
    return df.home_score.to_numpy(dtype=float) + df.away_score.to_numpy(dtype=float)


def train_and_predict(data: pd.DataFrame):
    v11.assert_ready(data)
    validate_weights(WIN_BLENDS, STRICT_WIN_NAMES)
    validate_weights(SCORE_BLENDS, SCORE_NAMES)
    for _, fields in WIN_COMPONENTS.items():
        for f in fields:
            if f not in data:
                raise RuntimeError(f'Missing required pregame feature {f}')
    if not data.season.isin([2022, 2023, 2024, 2025, 2026]).all():
        raise RuntimeError('Unexpected season in source frame')
    if data.duplicated(['season', 'game_id']).any():
        raise RuntimeError('Duplicate historical game ID')

    results: dict[int, dict[str, dict[str, np.ndarray]]] = {}
    # Using the 2022-24 model for both 2025 and 2026 intentionally matches
    # the older research configurations and prevents any training on 2026.
    for tr_years, year in FOLDS:
        tr = data[data.season.isin(tr_years)]
        va = data[data.season == year]
        target_years = (year, 2026) if year == 2025 else (year,)
        if len(tr) < 250 or len(va) < 100:
            raise RuntimeError(f'Insufficient historical sample for forward validation {year}')
        models = {out: {} for out in target_years}
        for name, fields in WIN_COMPONENTS.items():
            clf = model_for(name)
            clf.fit(tr[fields], tr.home_win.astype(int))
            for target in target_years:
                f = data[data.season == target]
                models[target][name] = v10.safe_probability(clf.predict_proba(f[fields])[:, 1])
        for name in SCORE_NAMES:
            for target in target_years:
                models[target][name] = None
            if name == 'v9_pace_baseline':
                for target in target_years:
                    models[target][name] = data.loc[data.season == target, 'v9_scoring_baseline_total'].to_numpy(dtype=float)
                continue
            model, fields, residual = score_model_for(name)
            actual = score_target(tr)
            base_tr = tr.v9_scoring_baseline_total.to_numpy(dtype=float)
            model.fit(tr[fields], actual - base_tr if residual else actual)
            for target in target_years:
                f = data[data.season == target]
                base = f.v9_scoring_baseline_total.to_numpy(dtype=float)
                pred = np.asarray(model.predict(f[fields]), dtype=float)
                models[target][name] = base + np.clip(pred, -25, 25) if residual else pred
        for target in target_years:
            if target in results:
                raise RuntimeError(f'Duplicate forecast fold for {target}')
            results[target] = {'components': models[target],
                               'frame': data[data.season == target].reset_index(drop=True)}
    return results


def weighted_blend(models: dict[str, np.ndarray], names, weights):
    components = [np.asarray(models[name], dtype=float) for name in names]
    lengths = {len(arr) for arr in components}
    if len(lengths) != 1 or not all(np.isfinite(a).all() for a in components):
        raise RuntimeError('Missing, invalid or misaligned component predictions')
    return np.average(np.vstack(components), axis=0, weights=weights)


def win_quality(frame, p):
    y=frame.home_win.to_numpy(dtype=int)
    p=v10.safe_probability(p)
    return {'accuracy_pct':100*float(((p >= .5) == y).mean()),
            'correct':int(((p >= .5) == y).sum()),
            'brier':float(((p-y)**2).mean())}


def blend_grid(packs, typ):
    if typ == 'win':
        names=STRICT_WIN_NAMES
        weights={**{f'V12 component {x}':[float(x==n) for n in names] for x in names},**WIN_BLENDS}
    else:
        names=SCORE_NAMES
        weights={**{f'V12 component {x}':[float(x==n) for n in names] for x in names},**SCORE_BLENDS}
    rows=[]
    for label, weight in weights.items():
        metrics={}
        for year in (2024,2025):
            pack=packs[year]
            p=weighted_blend(pack['components'],names,weight)
            if typ=='win':
                metrics[year]=win_quality(pack['frame'],p)
            else:
                mae=float(mean_absolute_error(score_target(pack['frame']),p))
                metrics[year]={'mae':mae}
        out={'model':label,'weights':weight}
        if typ=='win':
            out.update({'accuracy_2024':metrics[2024]['accuracy_pct'],
                        'accuracy_2025':metrics[2025]['accuracy_pct'],
                        'brier_2024':metrics[2024]['brier'],
                        'brier_2025':metrics[2025]['brier'],
                        'weighted_accuracy':sum(WEIGHTS_YEARS[y]*metrics[y]['accuracy_pct'] for y in (2024,2025)),
                        'weighted_brier':sum(WEIGHTS_YEARS[y]*metrics[y]['brier'] for y in (2024,2025))})
        else:
            out.update({'mae_2024':metrics[2024]['mae'],'mae_2025':metrics[2025]['mae'],
                        'weighted_mae':sum(WEIGHTS_YEARS[y]*metrics[y]['mae'] for y in (2024,2025))})
        rows.append(out)
    return pd.DataFrame(rows)


def choose_blends(win_grid,score_grid):
    # Guarded selection: an ensemble must outperform the best single-component
    # validation result, not merely fit reviewed 2026 outcomes.
    singles=win_grid[win_grid.model.str.startswith('V12 component')]
    best_single=singles.sort_values(['weighted_accuracy','weighted_brier','model'], ascending=[False,True,True]).iloc[0]
    # Reject models with notably worse probability quality than the anchor.
    acceptable=win_grid[(win_grid.weighted_brier <= best_single.weighted_brier + .0015) &
                        (win_grid.accuracy_2024 >= best_single.accuracy_2024 - .75) &
                        (win_grid.accuracy_2025 >= best_single.accuracy_2025 - .75)]
    if acceptable.empty:
        winner=best_single.copy()
    else:
        best=acceptable.sort_values(['weighted_accuracy','weighted_brier','model'],ascending=[False,True,True]).iloc[0]
        winner=best if (best.model.startswith('V12 component') or
                        best.weighted_accuracy >= best_single.weighted_accuracy + .15) else best_single

    score_single=score_grid[score_grid.model.str.startswith('V12 component')].sort_values('weighted_mae').iloc[0]
    score_best=score_grid.sort_values(['weighted_mae','model']).iloc[0]
    # Must improve validation MAE by >=0.10 points to use a blend.
    scorer=score_best if (score_best.model.startswith('V12 component') or
                           score_best.weighted_mae <= score_single.weighted_mae - .10) else score_single
    return winner, scorer


def comparative_metrics(rev, win_p, total, archive=None):
    review=rev['frame']; y=review.home_win.to_numpy(dtype=int); actual=score_target(review)
    close=v8.close_mask(review)
    populations={'all':np.ones(len(review),bool),'competitive_elo_120':close,
                 'other_games':~close,'neutral':review.neutral_site.eq(1).to_numpy(),
                 'non_neutral':review.neutral_site.eq(0).to_numpy()}
    compar=[]
    preds={'V12 selected strict prediction':win_p}
    if archive is not None:
        for col in ['V5_strict_archive','V8_strict_archive','V8_hybrid_archive','V10_strict_archive','V11_strict_archive']:
            preds[col]=archive[col].to_numpy(dtype=float)
    for group, mask in populations.items():
        if not np.any(mask): continue
        for name,p in preds.items():
            s=win_quality(review.loc[mask],np.asarray(p)[mask])
            compar.append({'population':group,'model':name,'games':int(mask.sum()),
                           'correct':s['correct'],'accuracy_pct':round(s['accuracy_pct'],4),
                           'brier':round(s['brier'],6)})
    sc=[]
    totals={'V12 validation-selected total estimate':total}
    if archive is not None:
        totals['V10 archive total']=archive['V10_total_archive'].to_numpy(dtype=float)
        totals['V11 archive total']=archive['V11_total_archive'].to_numpy(dtype=float)
    for model,p in totals.items():
        pp=np.asarray(p,dtype=float)
        sc.append({'model':model,'games':len(actual),'total_mae':round(float(np.abs(pp-actual).mean()),4),
                   'total_rmse':round(float(np.sqrt(np.mean((pp-actual)**2))),4),
                   'mean_error':round(float(np.mean(pp-actual)),4)})
    pairs=[]
    if archive is not None:
        for baseline in ('V8_strict_archive','V8_hybrid_archive'):
            p=archive[baseline].to_numpy(dtype=float)
            delta=((win_p>=.5)==y).astype(int)-((p>=.5)==y).astype(int)
            lo,hi=v8.bootstrap_days(delta,review.game_date.reset_index(drop=True),SEED)
            pairs.append({'baseline':baseline,'games':len(y),
                          'accuracy_difference_pp':round(float(100*delta.mean()),4),
                          'day_bootstrap_95pct_low_pp':lo,'day_bootstrap_95pct_high_pp':hi,
                          'hybrid_right_old_wrong':int(((delta==1)).sum()),
                          'old_right_hybrid_wrong':int(((delta==-1)).sum())})
    return pd.DataFrame(compar),pd.DataFrame(sc),pd.DataFrame(pairs)


def align_reference(review):
    if not REFERENCE.is_file():
        return None,'Missing reference_2026_v11.csv; independent old-version comparison omitted.'
    archive=pd.read_csv(REFERENCE, dtype={'game_id':'string'})
    ids=review['game_id'].astype(str)
    if archive.duplicated(['season','game_id']).any():
        raise RuntimeError('Reference CSV has duplicated game IDs')
    archive=archive.set_index(['season','game_id'])
    keys=pd.MultiIndex.from_arrays([review.season.to_numpy(), ids.to_numpy()], names=['season','game_id'])
    matched=archive.reindex(keys)
    if matched.isnull().any().any():
        return None,'Historical reference game IDs do not fully align; old-version comparison omitted.'
    if not (matched.home_win.to_numpy(dtype=int)==review.home_win.to_numpy(dtype=int)).all():
        raise RuntimeError('Reference winners differ from database; refusing misleading comparison')
    if not np.allclose(matched.actual_total.to_numpy(float),score_target(review),atol=0,rtol=0):
        raise RuntimeError('Reference scores differ from database; refusing misleading comparison')
    return matched.reset_index(),'Archived 2026 model outputs aligned exactly by game ID and final scores.'


def evaluate(data:pd.DataFrame, audit:dict):
    packs=train_and_predict(data)
    selection_win=blend_grid(packs,'win')
    selection_score=blend_grid(packs,'score')
    winner,scorer=choose_blends(selection_win,selection_score)
    re=packs[REVIEW]; frame=re['frame']
    wp=v10.safe_probability(weighted_blend(re['components'], STRICT_WIN_NAMES, winner['weights']))
    tp=weighted_blend(re['components'],SCORE_NAMES,scorer['weights'])
    archive,ref_status=align_reference(frame)
    comp,score_comp,paired=comparative_metrics(re,wp,tp,archive)
    diag=frame[['season','game_id','game_date','home_win','home_score','away_score','neutral_site','elo_gap']].copy()
    diag['hybrid_home_win_probability']=wp
    diag['hybrid_projected_total']=tp
    diag['actual_total']=score_target(frame)
    for col in ['elo_strength','v5_shooting','v8_matchups','v10_stability','v11_nonlinear']:
        diag[f'component_{col}']=re['components'][col]
    diag['exploratory_hybrid_probability_unverified']=re['components'][EXPLORATORY]
    if archive is not None:
        for col in ('V8_hybrid_archive','V10_total_archive'):
            diag[col]=archive[col].to_numpy(float)
    detail={'winning_model':winner['model'],'winner_weights':dict(zip(STRICT_WIN_NAMES,winner['weights'])),
            'total_model':scorer['model'],'total_weights':dict(zip(SCORE_NAMES,scorer['weights'])),
            'validation_only':True,'ref_status':ref_status}
    audit_summary={'replay_audit':audit, 'selected':detail,
       'upstream_publication_timestamp_verified':False,
       'selected_winner_uses_only_reconstructed_strict_feature_groups':True,
       'exploratory_unverified_hybrid_excluded_from_model_selection':True,
       '2026_review_is_untouched_holdout':False,
       '2026_results_not_used_to_select_components_or_weights':True,
       '2027_live_accuracy_not_established':True,
       'future_game_or_live_pick_generation':False,
       'no_database_writes':True,
       'evaluation_seasons':[2024,2025,2026],
       'matching_archive_reference_status':ref_status}
    return {'winner_selection':selection_win,'score_selection':selection_score,
            'winner_comparison':comp,'score_comparison':score_comp,'paired':paired,
            'historical':diag,'selected':detail,'audit':audit_summary}


def report_html(r):
    def table(name):
        x=r[name]
        return x.drop(columns=['weights'],errors='ignore').round(5).to_html(index=False,border=0,classes='data',escape=True)
    note=('Research-only historical evaluation. 2026 has been repeatedly analyzed and is not a fresh holdout. '
          'Validation and weight selection use 2024–2025 only. Historical input timestamps and earlier '
          'upstream rating provenance are not independently audited; no future-game forecasts or accuracy promises.')
    return f'''<!doctype html><html><head><meta charset="utf-8"><title>NCAA MBB V12 Guarded Hybrid</title>
<style>body{{font:15px system-ui,Arial;max-width:1150px;margin:32px auto;padding:0 22px;line-height:1.5;color:#203047}}
.data{{border-collapse:collapse;display:block;overflow-x:auto;max-width:100%;margin-bottom:28px}}
.data th,.data td{{padding:8px 10px;border-bottom:1px solid #dde5ee;white-space:nowrap;text-align:left}}
.data th{{background:#edf3f8}} .note{{border-left:4px solid #5c748f;padding:14px;background:#f0f5f8}}
pre{{white-space:pre-wrap;word-wrap:break-word;background:#f3f5f8;padding:14px}}</style></head><body>
<h1>NCAA MBB V12: Guarded Historical Hybrid</h1><p>Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC</p>
<p class="note">{html.escape(note)}</p>
<h2>Validation-selected components</h2><pre>{html.escape(json.dumps(r['selected'],indent=2))}</pre>
<h2>2026 winner accuracy, same historical games</h2>{table('winner_comparison')}
<h2>2026 combined-score error</h2>{table('score_comparison')}
<h2>Paired winner accuracy vs archived V8</h2>{table('paired')}
<h2>2024–2025 winner blend selection</h2>{table('winner_selection')}
<h2>2024–2025 total model selection</h2>{table('score_selection')}
<h2>Source, timing, and data checks</h2><pre>{html.escape(json.dumps(r['audit'],indent=2,default=str))}</pre>
</body></html>'''


def save(r):
    OUTPUT.mkdir(parents=True,exist_ok=True)
    for name,key in [('winner_comparison_2026.csv','winner_comparison'),
                     ('total_comparison_2026.csv','score_comparison'),
                     ('paired_v8_2026.csv','paired'),
                     ('winner_selection_2024_2025.csv','winner_selection'),
                     ('total_selection_2024_2025.csv','score_selection'),
                     ('historical_2026_diagnostics.csv','historical')]:
        r[key].to_csv(OUTPUT/name,index=False)
    (OUTPUT/'selected_models.json').write_text(json.dumps(r['selected'],indent=2,default=str))
    (OUTPUT/'audit.json').write_text(json.dumps(r['audit'],indent=2,default=str))
    (OUTPUT/'report.html').write_text(report_html(r),encoding='utf8')
    print('V12 completed. Historical evaluation only; not a future-game accuracy claim.')
    print(r['winner_comparison'].query("population == 'all'").to_string(index=False))
    print(r['score_comparison'].to_string(index=False))
    print('Report:', (OUTPUT/'report.html').resolve())


def self_test():
    validate_weights(WIN_BLENDS, STRICT_WIN_NAMES)
    validate_weights(SCORE_BLENDS,SCORE_NAMES)
    sample={'a':np.array([.2,.5,.7]),'b':np.array([.4,.5,.9])}
    assert np.allclose(weighted_blend(sample,('a','b'),[.5,.5]),[.3,.5,.8])
    assert win_quality(pd.DataFrame({'home_win':[0,1,1]}),np.array([.1,.55,.75]))['correct']==3
    old=pd.DataFrame([
        {'model':'V12 component a','weights':[1.,0.],'accuracy_2024':68.,'accuracy_2025':68.,'brier_2024':.21,'brier_2025':.20,'weighted_accuracy':68.,'weighted_brier':.204},
        {'model':'V12 ensemble','weights':[.5,.5],'accuracy_2024':69.,'accuracy_2025':69.,'brier_2024':.20,'brier_2025':.19,'weighted_accuracy':69.,'weighted_brier':.194},
    ])
    score=pd.DataFrame([
        {'model':'V12 component a','weights':[1.,0.],'mae_2024':14.,'mae_2025':14.,'weighted_mae':14.},
        {'model':'V12 ensemble','weights':[.5,.5],'mae_2024':13.7,'mae_2025':13.7,'weighted_mae':13.7},
    ])
    w,s=choose_blends(old,score)
    assert w['model']=='V12 ensemble' and s['model']=='V12 ensemble'
    print('V12 selection, nonnegative weights, and non-peeking checks passed.')


def main():
    parser=argparse.ArgumentParser(description='V12 historical NCAA research hybrid')
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if args.self_test:
        self_test(); return
    raw,totals,scores,boxes=v5.fetch()
    optional,reports,availability_audit=v6.optional_data()
    players,player_audit=v9.fetch_players()
    frame,audit=v6.make_frame(raw,scores,boxes,optional,reports,availability_audit)
    data,audit,player_enabled=v9.assemble_features(frame,scores,boxes,players,audit)
    audit['source_player_fetch']=player_audit
    audit['player_features_enabled']=player_enabled
    # Selected strict set does not use missing injury-report status.
    save(evaluate(data,audit))


if __name__=='__main__':
    main()
