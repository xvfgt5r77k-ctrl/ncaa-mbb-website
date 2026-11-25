"""V7 NCAA MBB retrospective score/winner research.

Read-only Supabase input, no upcoming or live picks. Winner model: focused
lagged features from V5/V6, chosen using 2025 only. Score-total model: lagged
prior-day scoring and allowance histories. 2026 previously reviewed, exploratory.

Install alongside v3_research.py, v4_research.py, v5_research.py, v6_research.py.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import html
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, log_loss, brier_score_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import v4_research as v4
import v5_research as v5
import v6_research as v6

OUT = Path(os.getenv('V7_OUTPUT_DIR', 'v7_results'))

# Predeclared candidates. Selection strictly by 2025 log-loss, NOT 2026 results.
WINNER_SETS = {
    'V7 compact shooting': list(dict.fromkeys(v5.STRICT_FIELDS + v6.SHOOTING)),
    'V7 shooting + opponent': list(dict.fromkeys(v5.STRICT_FIELDS + v6.SHOOTING + v6.OPPONENT)),
    'V7 shooting + context': list(dict.fromkeys(v5.STRICT_FIELDS + v6.SHOOTING + v6.CONTEXT)),
    'V7 shooting + opponent + context': list(dict.fromkeys(v5.STRICT_FIELDS + v6.SHOOTING + v6.OPPONENT + v6.CONTEXT)),
}
TOTAL_FIELDS = ['v7_total_baseline', 'v7_total_recent', 'v7_total_sample_min',
                'neutral_site', 'combined_scoring_rate']


def chronological_total_features(scores: pd.DataFrame) -> pd.DataFrame:
    """Only use games from strictly earlier calendar dates for every team.

    For a game day, all features are computed before updating any game from
    that day, preventing intra-day score leakage and preserving causal replay.
    """
    required = {'season','game_id','game_date','home_team_id','away_team_id',
                'home_score','away_score','neutral_site'}
    if not required.issubset(scores.columns):
        raise ValueError('Historical results missing necessary columns')
    data = scores.copy()
    data['day'] = pd.to_datetime(data['game_date'], errors='coerce').dt.date
    data['season'] = pd.to_numeric(data['season'], errors='coerce')
    data = data.dropna(subset=['day','season','home_team_id','away_team_id',
                               'home_score','away_score']).copy()
    if data.duplicated(['season', 'game_id']).any():
        raise ValueError('Duplicate game IDs in historical results')
    data = data.sort_values(['season','day','game_id'], kind='stable')
    out = []
    for season, season_rows in data.groupby('season', sort=True):
        played = defaultdict(lambda: {'scored':[], 'allowed':[], 'totals':[]})
        for day, today in season_rows.groupby('day', sort=True):
            updates = []
            for r in today.itertuples(index=False):
                h_id = str(r.home_team_id)
                a_id = str(r.away_team_id)
                h = played[h_id]
                a = played[a_id]
                n = min(len(h['scored']),len(a['scored']))
                if n:
                    hp = np.mean(h['scored'])
                    hd = np.mean(h['allowed'])
                    ap = np.mean(a['scored'])
                    ad = np.mean(a['allowed'])
                    baseline = .5*(hp+ad) + .5*(ap+hd)
                    recent = np.mean((h['totals'][-5:]+a['totals'][-5:]))
                else:
                    baseline, recent = np.nan, np.nan
                out.append({'season':int(season), 'game_id':str(r.game_id),
                            'v7_total_baseline':baseline, 'v7_total_recent':recent,
                            'v7_total_sample_min':int(n)})
                updates.append((h_id,a_id,float(r.home_score),float(r.away_score)))
            # Batch apply updates AFTER every game that day has been estimated.
            for h_id,a_id,h_points,a_points in updates:
                played[h_id]['scored'].append(h_points)
                played[h_id]['allowed'].append(a_points)
                played[h_id]['totals'].append(h_points+a_points)
                played[a_id]['scored'].append(a_points)
                played[a_id]['allowed'].append(h_points)
                played[a_id]['totals'].append(h_points+a_points)
    return pd.DataFrame(out)


def score_pipe(alpha: float):
    return Pipeline([('impute',SimpleImputer(strategy='median',keep_empty_features=True)),
                     ('scale', StandardScaler()), ('ridge',Ridge(alpha=alpha))])


def train_and_evaluate(frame: pd.DataFrame, scores: pd.DataFrame):
    replay = chronological_total_features(scores)
    data = frame.merge(replay, on=['season','game_id'], how='left',validate='one_to_one')
    # V3's historical feature query does not select team IDs; get their IDs
    # from the completed-game scoreboard solely for post-evaluation labels.
    identities = scores[['season','game_id','home_team_id','away_team_id']].drop_duplicates(['season','game_id'])
    data = data.merge(identities, on=['season','game_id'], how='left',validate='one_to_one')
    if data.duplicated(['season','game_id']).any():
        raise ValueError('Duplicate training games')
    for c in TOTAL_FIELDS:
        data[c] = pd.to_numeric(data[c], errors='coerce')
    data['actual_total'] = pd.to_numeric(data.home_score,errors='raise') + pd.to_numeric(data.away_score,errors='raise')
    train = data[data.season.isin(v4.TRAIN)]
    val = data[data.season == v4.VALIDATE]
    review = data[data.season == v4.REVIEW]
    if len(train)<100 or len(val)<100 or len(review)<100:
        raise ValueError('Insufficient historical 2022-26 data to evaluate V7')
    for f in WINNER_SETS.values():
        absent = [c for c in f if c not in data.columns]
        if absent:
            raise ValueError('Missing V7 fields: '+','.join(absent))
    yr_tr=train.home_win.astype(int).to_numpy()
    yr_val=val.home_win.astype(int).to_numpy()
    yr_rev=review.home_win.astype(int).to_numpy()

    # Model selection uses *only* predeclared 2025 log-loss.
    trials=[]
    for label, cols in WINNER_SETS.items():
        for c in (.3, 3.):
            name=f'{label} C={c:g}'
            m = v4.logistic(c)
            m.fit(train[cols],yr_tr)
            pv=m.predict_proba(val[cols])[:,1]
            trials.append((float(log_loss(yr_val,pv,labels=[0,1])),name,cols,m))
    trials.sort(key=lambda row:(row[0],row[1]))
    _, winner_name, chosen_cols, winner = trials[0]
    pv=winner.predict_proba(review[chosen_cols])[:,1]

    total_trials=[]
    for alpha in (1., 20., 200.):
        model=score_pipe(alpha)
        model.fit(train[TOTAL_FIELDS],train.actual_total)
        err=mean_absolute_error(val.actual_total,model.predict(val[TOTAL_FIELDS]))
        total_trials.append((float(err),alpha,model))
    total_trials.sort(key=lambda t:(t[0],t[1]))
    _, selected_alpha, total_model=total_trials[0]
    projected_total=np.clip(total_model.predict(review[TOTAL_FIELDS]),70.,240.)
    correct=((pv>=.5)==yr_rev)

    # Baseline selection (V5 Hybrid) and original V6 hybrid can be reconstructed
    # separately by their report; here compare only against matched 2026 rows
    # loaded by V6's existing V5 evaluation, avoiding any population mismatch.
    old_metrics,old_board,old_paired,old_hist,old_selected,*_=v5.predict_and_evaluate(frame)
    matched=old_hist.set_index('game_id')
    v5_probs=matched.loc[review.game_id.astype(str),'v5_hybrid_home_win_prob'].to_numpy()
    v5_correct=(v5_probs>=.5)==yr_rev

    report=pd.DataFrame({
        'season':review.season.astype(int).to_numpy(),
        'game_id':review.game_id.astype(str).to_numpy(),
        'game_date':review.game_date.astype(str).to_numpy(),
        'home_team_id':review.home_team_id.astype(str).to_numpy(),
        'away_team_id':review.away_team_id.astype(str).to_numpy(),
        'home_win_prob':pv,
        'predicted_winner_side':np.where(pv>=.5,'Home','Away'),
        'predicted_winner_correct':correct.astype(int),
        'projected_total':np.round(projected_total,2),
        'actual_total':review.actual_total.to_numpy(),
        'total_abs_error':np.round(abs(projected_total-review.actual_total.to_numpy()),2),
        'home_score':review.home_score.to_numpy(),
        'away_score':review.away_score.to_numpy(),
        'v5_hybrid_home_win_prob':v5_probs,
    })
    stats={
        'v7_winner_model':winner_name,
        'v7_total_ridge_alpha':selected_alpha,
        '2026_games':int(len(report)),
        'v7_winner_accuracy_pct':round(100*float(correct.mean()),2),
        'v5_hybrid_same_game_accuracy_pct':round(100*float(v5_correct.mean()),2),
        'v7_correct':int(correct.sum()),
        'v5_correct_same_games':int(v5_correct.sum()),
        'v7_log_loss':round(float(log_loss(yr_rev,pv,labels=[0,1])),5),
        'v7_brier':round(float(brier_score_loss(yr_rev,pv)),5),
        'v7_total_mae':round(float(mean_absolute_error(review.actual_total,projected_total)),2),
        'v7_total_baseline_mae':round(float(mean_absolute_error(
             review.actual_total,
             review.v7_total_baseline.fillna(train.actual_total.median()))),2),
        '2026_holdout_untouched':False,
        'historical_game_results_only':True,
        'note': ('Historical 2026 was repeatedly reviewed in V3-V6; not a fresh holdout. '
                 'Winner feature selection and total score regression use 2025 only. '
                 'Future games are never selected by this script.'),
    }
    selection=pd.DataFrame([{'model':t[1], 'validation_2025_log_loss':round(t[0],5)} for t in trials])
    totals=pd.DataFrame([{'ridge_alpha':a, 'validation_2025_mae':round(e,3)} for e,a,_ in total_trials])
    return report,stats,selection,totals


def read_teams():
    import psycopg
    from psycopg.rows import dict_row
    url=os.getenv('DATABASE_URL')
    if not url: return {}
    with psycopg.connect(url, sslmode='require', connect_timeout=25,
                         prepare_threshold=None,row_factory=dict_row) as conn:
        conn.read_only=True
        with conn.cursor() as c:
            c.execute('SELECT team_id::text AS team_id, name FROM public.teams')
            return {row['team_id']:row['name'] for row in c.fetchall()}


def write_results(report, stats, selection, totals, team_names):
    OUT.mkdir(exist_ok=True,parents=True)
    report=report.copy()
    report.insert(4,'home_team',report.home_team_id.map(team_names).fillna(report.home_team_id))
    report.insert(5,'away_team',report.away_team_id.map(team_names).fillna(report.away_team_id))
    report.insert(9,'predicted_winner',np.where(report.predicted_winner_side=='Home',report.home_team,report.away_team))
    report.to_csv(OUT/'historical_2026_predictions.csv',index=False)
    selection.to_csv(OUT/'winner_selection_2025.csv',index=False)
    totals.to_csv(OUT/'total_selection_2025.csv',index=False)
    (OUT/'summary.json').write_text(json.dumps(stats,indent=2),encoding='utf-8')
    rows=''.join(f'<tr><th>{html.escape(str(key))}</th><td>{html.escape(str(value))}</td></tr>' for key,value in stats.items())
    report_html=f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><title>V7 Retrospective Research</title>
<style>body{{font:16px system-ui;max-width:850px;margin:32px auto;padding:0 20px;line-height:1.5}}
th,td{{border-bottom:1px solid #ddd;text-align:left;padding:9px}}</style></head><body>
<h1>V7 Retrospective Winner and Total Research</h1>
<p>Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}</p>
<p><b>Completed games only.</b> 2026 was previously inspected and is NOT a new untouched holdout.
All model candidate choices use 2025 validation. Score totals are historical reconstructed estimates;
no in-progress or upcoming-game selections are produced.</p>
<table>{rows}</table><p>See historical_2026_predictions.csv for the predictions and recorded final totals.</p>
</body></html>'''
    (OUT/'report.html').write_text(report_html,encoding='utf-8')


def test_total_time_cutoff():
    items=[]
    for day in range(1,15):
        for i in range(2):
            items.append({'season':2026,'game_id':f'{day}-{i}',
                 'game_date':f'2025-11-{day:02d}',
                 'home_team_id':str(2*i),'away_team_id':str(2*i+1),
                 'home_score':65+day, 'away_score':60+day,
                 'neutral_site':0})
    a=pd.DataFrame(items)
    b=a.copy()
    b.loc[b.game_date=='2025-11-09','home_score']+=100
    orig=chronological_total_features(a).set_index('game_id')
    changed=chronological_total_features(b).set_index('game_id')
    for gid in a.loc[a.game_date=='2025-11-09','game_id']:
        for col in ['v7_total_baseline','v7_total_recent','v7_total_sample_min']:
            x,y=orig.loc[gid,col],changed.loc[gid,col]
            assert (pd.isna(x) and pd.isna(y)) or np.isclose(x,y), (gid,col,x,y)
    # Changes do affect next day, to prove the test checks for relevant data.
    assert not np.isclose(orig.loc['10-0','v7_total_baseline'],
                          changed.loc['10-0','v7_total_baseline'])
    print('V7 historical score totals use only earlier calendar days: PASS')


def main():
    if '--self-test' in sys.argv:
        test_total_time_cutoff()
        return
    raw,denoms,scores,boxes=v5.fetch()
    optional,reports,audit=v6.optional_data()
    frame,_=v6.make_frame(raw,scores,boxes,optional,reports,audit)
    results,stats,winners,totals=train_and_evaluate(frame,scores)
    write_results(results,stats,winners,totals,read_teams())
    print('V7 historical 2026 research results:',stats)
    print('Output:',str(OUT.resolve()))


if __name__=='__main__':
    try: main()
    except Exception as e:
        print('V7 research failed:',type(e).__name__,str(e),file=sys.stderr)
        raise SystemExit(1) from e
