"""NCAA Men's Basketball V8: retrospective matchup + close-game research.

READ-ONLY, historical completed games ONLY (2022-2026). No future-game picks,
score feeds, bet placement, or current-game winner estimates.
Requires the exact pre-existing v3/v4/v5/v6 scripts from this repository.

2026 has been repeatedly inspected and IS NOT an untouched holdout.
All new features are computed from earlier-day reconstructed V4-V6 features.
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
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss

import v4_research as v4
import v5_research as v5
import v6_research as v6

OUT = Path(os.getenv('V8_OUTPUT_DIR', 'v8_results'))
SEED = 83
# Definition based exclusively on game-day strength available before tipoff.
# Do not adjust this threshold based on 2026 outcomes.
CLOSE_ELO_POINTS = 120.0
MIN_CLOSE_TRAIN = 150
MIN_CLOSE_VALIDATION = 60
MIN_IMPROVEMENT_CLOSE_PP = 0.5
# Historical injury reports are only used if both teams have >=80% coverage
# in training and validation, with source IDs + observed-at timestamps verified.
MIN_VERIFIED_LINEUP_COVERAGE = 80.0

MATCHUP_FIELDS = [
    'v8_three_edge_by_style', 'v8_three_edge_by_reliability',
    'v8_two_point_clash', 'v8_shot_turnover_interaction',
    'v8_shot_rebounding_interaction', 'v8_elo_shooting_interaction',
    'v8_three_uncertainty',
]
OPTIONAL_SHOOT_FIELDS = list(v6.SHOOTING)


def numeric(frame: pd.DataFrame, name: str) -> pd.Series:
    if name not in frame:
        raise RuntimeError(f'Required pregame field missing: {name}')
    return pd.to_numeric(frame[name], errors='coerce').astype(float)


def add_matchup_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Nonlinear interactions of pre-existing, prior-day-only features.

    No use of current-game scores or shooting results. Missing data remain NaN.
    The interactions are signed to preserve home/away differences where possible.
    """
    f = frame.copy()
    three = numeric(f, 'v5_three_matchup_gap')
    share = numeric(f, 'v5_three_attempt_share_gap')
    volatility = numeric(f, 'v5_three_volatility_sum')
    n_shoot = numeric(f, 'v5_shooting_games_min')
    two = numeric(f, 'v6_two_matchup_gap')
    efg = numeric(f, 'v6_efg_matchup_gap')
    turnover = numeric(f, 'v6_tov_matchup_gap')
    rebounding = numeric(f, 'v6_orb_matchup_gap')
    elo = numeric(f, 'elo_gap')
    reliability = np.clip(n_shoot, 0, 40) / (np.clip(n_shoot, 0, 40) + 8.0)
    # Scale differences: percentages in [0,1]; Elo in points.
    f['v8_three_edge_by_style'] = three * np.abs(share)
    f['v8_three_edge_by_reliability'] = three * reliability
    f['v8_two_point_clash'] = two * np.abs(efg)
    f['v8_shot_turnover_interaction'] = efg * np.abs(turnover)
    f['v8_shot_rebounding_interaction'] = efg * np.abs(rebounding)
    f['v8_elo_shooting_interaction'] = (elo / 200.0) * np.abs(three)
    f['v8_three_uncertainty'] = volatility / (np.sqrt(n_shoot.clip(lower=0) + 1))
    f.replace([np.inf, -np.inf], np.nan, inplace=True)
    return f


def close_mask(frame: pd.DataFrame) -> np.ndarray:
    # The threshold and eligibility are independent of outcomes or model errors.
    gap = numeric(frame, 'elo_gap').abs()
    return gap.le(CLOSE_ELO_POINTS).fillna(False).to_numpy(dtype=bool)


def lineup_gate(audit: dict) -> tuple[bool, str]:
    if not audit.get('player_feature_enabled', False):
        return False, 'Historical reports unavailable or below minimum basic eligibility.'
    train = float(audit.get('two_sided_coverage_training_pct', 0) or 0)
    val = float(audit.get('two_sided_coverage_2025_pct', 0) or 0)
    if min(train, val) < MIN_VERIFIED_LINEUP_COVERAGE:
        return False, (f'Historical two-sided coverage below {MIN_VERIFIED_LINEUP_COVERAGE:.0f}% '
                       f'(training {train:.1f}%, validation {val:.1f}%).')
    return True, ('Strict pretipoff game/team/player IDs and timestamps were found; '
                  'interpret availability as reported status, not verified participation.')


def candidate_model(kind: str, c: float = 3.0):
    if kind == 'logistic':
        return v4.logistic(c)
    if kind == 'boost':
        return v4.boosted(3, 220, 100, .045)
    raise ValueError('Unknown candidate family')


def metrics(y: np.ndarray, p: np.ndarray, name: str, season: int, population: str) -> dict:
    y = np.asarray(y, dtype=int)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1-1e-6)
    if not len(y):
        return dict(model=name, season=season, population=population, games=0,
                    correct=0, accuracy_pct=None, brier_score=None, log_loss=None)
    return dict(model=name, season=season, population=population, games=int(len(y)),
                correct=int(((p >= .5) == y).sum()),
                accuracy_pct=round(100*accuracy_score(y, p >= .5), 2),
                brier_score=round(float(brier_score_loss(y, p)), 5),
                log_loss=round(float(log_loss(y, p, labels=[0,1])), 5))


def bootstrap_days(differences: np.ndarray, days: pd.Series, seed: int = SEED):
    diff = np.asarray(differences, dtype=float)
    day = pd.to_datetime(days, errors='coerce').dt.date
    sample = pd.DataFrame({'day':day, 'diff':diff}).dropna(subset=['day'])
    grouped = sample.groupby('day').agg(total=('diff','sum'), n=('diff','size'))
    if grouped.empty:
        return None, None
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(grouped), size=(1000, len(grouped)))
    s = grouped.total.to_numpy()[draws].sum(axis=1)
    n = grouped.n.to_numpy()[draws].sum(axis=1)
    values = 100*s/n
    return (round(float(np.quantile(values, .025)), 3),
            round(float(np.quantile(values, .975)), 3))


def evaluate(frame: pd.DataFrame, audit: dict):
    if frame.duplicated(['season','game_id']).any():
        raise RuntimeError('Duplicate historical game IDs; refusing ambiguous comparison')
    frame = add_matchup_features(frame)
    enabled, reason = lineup_gate(audit)
    audit['v8_lineup_feature_used'] = enabled
    audit['v8_lineup_feature_reason'] = reason
    audit['v8_opponent_interactions_from_pregame_replay_only'] = True
    audit['v8_close_definition'] = f'absolute prior-day Elo gap <= {CLOSE_ELO_POINTS:g}'
    audit['v8_no_2026_data_used_to_select_models'] = True
    audit['v8_2026_previously_examined'] = True
    audit['v8_2027_not_claimed_prospectively_validated'] = True
    audit['v8_hybrid_has_unverified_upstream_features'] = True

    # Rebuild V5 benchmark on identical game IDs; V5 selected only on 2025.
    old_metrics, _, _, old_history, old_selection, _, _ = v5.predict_and_evaluate(frame)
    _ = old_metrics, old_selection

    train = frame[frame.season.isin(v4.TRAIN)]
    val = frame[frame.season == v4.VALIDATE]
    review = frame[frame.season == v4.REVIEW]
    if len(train) < 500 or len(val) < 200 or len(review) < 200:
        raise RuntimeError('Insufficient historical train/validation/review games')
    if not np.array_equal(old_history.game_id.astype(str), review.game_id.astype(str)):
        raise RuntimeError('V5 historical benchmark not aligned by game ID')
    tr_y, va_y, re_y = [t.home_win.astype(int).to_numpy() for t in (train,val,review)]
    tr_close, va_close, re_close = map(close_mask,(train,val,review))
    if tr_close.sum() < MIN_CLOSE_TRAIN or va_close.sum() < MIN_CLOSE_VALIDATION:
        raise RuntimeError(f'Not enough Elo-defined close games: training {tr_close.sum()}, validation {va_close.sum()}')

    chosen = {}
    selections = []
    abst = []
    comparison = []
    review_export = review[['season','game_id','game_date','home_win','neutral_site']].copy().reset_index(drop=True)
    review_export['elo_close_game'] = re_close
    for grp in ('strict','hybrid'):
        base = list(v5.STRICT_FIELDS if grp == 'strict' else v5.HYBRID_FIELDS)
        groups = {'interaction_only':MATCHUP_FIELDS,
                  'interaction_plus_shooting':MATCHUP_FIELDS+OPTIONAL_SHOOT_FIELDS}
        if enabled:
            groups['interaction_plus_shooting_and_reports'] = MATCHUP_FIELDS+OPTIONAL_SHOOT_FIELDS+list(v6.PLAYER)
        candidates = []
        for label, extras in groups.items():
            fields = list(dict.fromkeys(base+extras))
            for kind, c in (('logistic', .3), ('logistic',3.0), ('boost', 0)):
                name=f'V8 {grp} {label} {kind}'+(f' C={c:g}' if kind=='logistic' else '')
                model = candidate_model(kind,c)
                model.fit(train[fields],tr_y)
                p_val=model.predict_proba(val[fields])[:,1]
                p_rev=model.predict_proba(review[fields])[:,1]
                stat=metrics(va_y,p_val,name,2025,'all')
                selections.append(dict(group=grp, **stat))
                candidates.append((name, fields, p_val, p_rev, stat))
                # Exploratory feature-group comparisons with a fixed model.
                if kind=='logistic' and c==3.0:
                    abst.append(metrics(va_y,p_val,name,2025,'all'))
                    abst.append(metrics(re_y,p_rev,name,2026,'all'))
        # Choose using 2025 winner accuracy; break ties by Brier and log loss.
        name, fields, p_val, p_rev, stat = min(candidates,
            key=lambda it: (-it[4]['accuracy_pct'],it[4]['brier_score'],it[4]['log_loss'],it[0]))
        baseline = old_history[f'v5_{grp}_home_win_prob'].to_numpy(dtype=float)
        # Train specialist ONLY on the predeclared Elo-close subset from 2022-24.
        specialists = []
        for c in (.3,3.0):
            spec=candidate_model('logistic',c)
            spec.fit(train.loc[tr_close,fields],tr_y[tr_close])
            spec_val=spec.predict_proba(val.loc[va_close,fields])[:,1]
            spec_rev=spec.predict_proba(review.loc[re_close,fields])[:,1]
            st=metrics(va_y[va_close],spec_val,f'V8 {grp} specialist C={c:g}',2025,'close_elo')
            selections.append(dict(group=grp, **st))
            specialists.append((c,spec_val,spec_rev,st))
        best_spec=min(specialists,key=lambda x:(-x[3]['accuracy_pct'],x[3]['brier_score'],x[0]))
        base_close = metrics(va_y[va_close],p_val[va_close],name,2025,'close_elo')
        specialist_gain=best_spec[3]['accuracy_pct']-base_close['accuracy_pct']
        # Require meaningful validation improvement and no severe probability degradation.
        accept=(specialist_gain >= MIN_IMPROVEMENT_CLOSE_PP and
                best_spec[3]['log_loss'] <= base_close['log_loss']+.02)
        p_val_final=p_val.copy()
        p_rev_final=p_rev.copy()
        if accept:
            p_val_final[va_close]=best_spec[1]
            p_rev_final[re_close]=best_spec[2]
        selected_name=name+ (' + CLOSE specialist' if accept else ' (specialist not adopted)')
        chosen[grp]={'general':name,'close_specialist_adopted':bool(accept),
                     'close_validation_gain_pp':round(float(specialist_gain),3),
                     'specialist_close_val_games':int(va_close.sum()),
                     'features':fields}
        for yr,y,old,new,close in ((2026,re_y,baseline,p_rev_final,re_close),):
            for population, mask in [('all',np.ones(len(y),bool)),('close_elo',close),('other_games',~close)]:
                comparison += [metrics(y[mask],old[mask],f'V5 {grp} (same games)',yr,population),
                               metrics(y[mask],new[mask],f'V8 {grp} selected',yr,population)]
        review_export[f'v5_{grp}_home_win_prob'] = baseline
        review_export[f'v8_{grp}_home_win_prob'] = p_rev_final
        review_export[f'v8_{grp}_model_selected'] = selected_name
        review_export[f'v8_{grp}_winner_correct'] = ((p_rev_final >= .5)==re_y).astype(int)
        review_export[f'v5_{grp}_winner_correct'] = ((baseline >= .5)==re_y).astype(int)
    paired = []
    for grp in ('strict','hybrid'):
        old_ok=review_export[f'v5_{grp}_winner_correct'].to_numpy(dtype=int)
        new_ok=review_export[f'v8_{grp}_winner_correct'].to_numpy(dtype=int)
        for population,mask in [('all',np.ones(len(review),bool)),('close_elo',re_close),('other_games',~re_close)]:
            diff=new_ok[mask]-old_ok[mask]
            lo,hi=bootstrap_days(diff,review.game_date.loc[mask].reset_index(drop=True))
            paired.append({'group':grp,'population':population,'games':int(mask.sum()),
                'accuracy_gain_pp':round(float(diff.mean()*100),3) if mask.any() else None,
                'day_bootstrap_95pct_low_pp':lo,'day_bootstrap_95pct_high_pp':hi,
                'v8_correct_v5_wrong':int(((new_ok[mask]==1)&(old_ok[mask]==0)).sum()),
                'v5_correct_v8_wrong':int(((new_ok[mask]==0)&(old_ok[mask]==1)).sum())})
    return (pd.DataFrame(comparison),pd.DataFrame(selections),pd.DataFrame(abst),
            pd.DataFrame(paired),review_export,chosen,audit)


def html_table(df: pd.DataFrame) -> str:
    return df.to_html(index=False, border=0, escape=True, classes='results')


def report_html(comp, selection, ablation, paired, audit, chosen):
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><title>NCAA MBB V8 Research</title>
<style>body{{font:15px system-ui,Arial;max-width:1200px;margin:40px auto;padding:0 22px;color:#17212b;line-height:1.5}}
h1,h2{{line-height:1.2}}.note{{background:#edf2f7;border-left:4px solid #567089;padding:15px}}
.results{{border-collapse:collapse;display:block;overflow-x:auto;margin:14px 0 30px}}
.results td,.results th{{padding:8px 12px;border-bottom:1px solid #ddd;white-space:nowrap;text-align:left}}
.results th{{background:#f0f3f7}}pre{{white-space:pre-wrap;background:#f5f6f7;padding:18px}}</style></head><body>
<h1>NCAA MBB — V8 Matchup and Close-Game Research</h1>
<p>Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}</p>
<div class="note"><b>Historical research only.</b> These are completed games, not future-game selections.
2022–2024 fit; 2025 candidate selection; 2026 retrospective review (previously seen, NOT untouched).
Every new interaction is calculated from prior-day statistics, but V5 hybrid retains upstream fields
that have not been fully audited. Results are not verified prospective accuracy.</div>
<h2>V8 versus V5, exactly matched 2026 games</h2>{html_table(comp)}
<h2>Matched accuracy differences and uncertainty</h2>{html_table(paired)}
<h2>2025 model selection (never selected using 2026)</h2>{html_table(selection)}
<h2>Feature group ablations (exploratory)</h2>{html_table(ablation)}
<h2>Chosen configurations and close-game gate</h2><pre>{html.escape(json.dumps(chosen,indent=2))}</pre>
<h2>Historical feature provenance and availability</h2><pre>{html.escape(json.dumps(audit,indent=2,default=str))}</pre>
<p><b>Important:</b> V8's close-game subset means prior-day Elo differences of at most {CLOSE_ELO_POINTS:g} points.
It does not mean that those games were objectively 50/50 matchups. Selection on 2025 and previous
iterations against 2026 can create overfitting. Establish new-season performance using forecasts frozen
before tipoff and independently audited source inputs.</p></body></html>'''


def run():
    raw, totals, scores, boxes = v5.fetch()
    optional, reports, availability_audit = v6.optional_data()
    frame, audit = v6.make_frame(raw,scores,boxes,optional,reports,availability_audit)
    comp, selection, ablation, paired, rows, chosen, audit = evaluate(frame,audit)
    OUT.mkdir(exist_ok=True,parents=True)
    for fn, df in [('model_comparison.csv',comp),('selection_2025.csv',selection),
                   ('feature_groups.csv',ablation),('paired_differences_2026.csv',paired),
                   ('historical_2026_diagnostics.csv',rows)]:
        df.to_csv(OUT/fn,index=False)
    (OUT/'chronology_and_availability_audit.json').write_text(json.dumps(audit,indent=2,default=str),encoding='utf-8')
    (OUT/'summary.json').write_text(json.dumps({'selected':chosen,'metrics':comp.to_dict('records'),
                                                'paired':paired.to_dict('records'),
                                                'prospective_accuracy_confirmed':False},indent=2),encoding='utf-8')
    (OUT/'report.html').write_text(report_html(comp,selection,ablation,paired,audit,chosen),encoding='utf-8')
    print('V8 HISTORICAL RESEARCH — no live selections')
    print(comp.to_string(index=False))
    print('Report:',(OUT/'report.html').resolve())


def self_test():
    # Verify a game's labels or same-day results are NOT used by feature engineering.
    base=pd.DataFrame({'v5_three_matchup_gap':[.04,-.08,.01],
        'v5_three_attempt_share_gap':[.25,.10,.20],
        'v5_three_volatility_sum':[.12,.11,np.nan],
        'v5_shooting_games_min':[10,20,2],
        'v6_two_matchup_gap':[.03,-.01,.04],
        'v6_efg_matchup_gap':[.02,-.05,.01],
        'v6_tov_matchup_gap':[.01,-.02,.015],
        'v6_orb_matchup_gap':[.03,-.04,.01],
        'elo_gap':[30,-150,90],
        'home_score':[65,75,72], 'away_score':[64,83,60],
        'home_win':[1,0,1]})
    out=add_matchup_features(base)
    modified=base.copy()
    modified[['home_score','away_score','home_win']]=[[1,250,0],[100,3,1],[10,50,0]]
    out2=add_matchup_features(modified)
    for col in MATCHUP_FIELDS:
        assert np.allclose(out[col],out2[col],equal_nan=True),col
    assert close_mask(base).tolist()==[True,False,True]
    disabled,reason=lineup_gate({'player_feature_enabled':True,
         'two_sided_coverage_training_pct':79,'two_sided_coverage_2025_pct':90})
    assert not disabled and 'coverage' in reason.lower()
    # Generate independent labels to verify close-game diagnostics never index errors.
    p=np.array([.6,.38,.9]); y=np.array([1,0,1]); close=close_mask(base)
    assert metrics(y[close],p[close],'test',2025,'close')['games']==2
    lo,hi=bootstrap_days(np.array([0,1,-1]),pd.Series(['2026-01-01','2026-01-02','2026-01-02']))
    assert lo is not None and hi is not None
    print('V8 SYNTHETIC FEATURE AND SAFETY TESTS PASSED (no real accuracy measured).')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description='Historical NCAA basketball model diagnostics')
    parser.add_argument('--self-test', action='store_true', help='offline unit tests')
    args=parser.parse_args()
    if args.self_test:
        self_test()
    else:
        run()
