#!/usr/bin/env python3
"""NCAA men's basketball WINNER-only prediction engine.

Inputs: immutable V2 previous-day team snapshots from Supabase.
Selection: 2022-24 training, 2025 validation, 2026 retrospective evaluation.
Deployment: fit 2022-26, score only FUTURE 2027 prospective snapshots.
No wagering, market, injury, or in-game information.

Critical: historical_replay rows were reconstructed after games; not evidence
that their entire upstream input source existed in exactly this form pre-tip.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone, timedelta
import json
import os
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RESEARCH_VERSION = 'winner-v1.0'
TEAM_FIELDS = (
    'games_before', 'elo_rating', 'adjusted_net_per100',
    'adjusted_offense_per100', 'adjusted_defense_per100', 'opponent_elo_mean',
    'average_points_for', 'average_points_against', 'estimated_possessions',
    'effective_fg_pct', 'three_point_rate', 'turnover_per_possession',
    'free_throw_rate', 'rest_days', 'recorded_at', 'tipoff_utc', 'game_day',
    'quality_flags'
)
BASIC = ['elo_gap', 'net_gap', 'home_court']
CORE = BASIC + ['min_games', 'games_gap', 'rest_gap', 'form_margin_gap', 'sos_elo_gap']
FULL = CORE + ['off_gap', 'def_gap', 'pace_gap', 'efg_gap', 'three_gap', 'turnovers_gap', 'free_throw_gap']


def matchup_sql() -> str:
    attrs = ',\n '.join(f'h.{c} AS home_{c}, a.{c} AS away_{c}' for c in TEAM_FIELDS)
    # Historical training exclusively uses reconstructed prior-day snapshots.
    # Upcoming games are recomputed ON DEMAND from the latest source box scores
    # in db_upgrade_v2.py, never from stale frozen prospective feature rows.
    return f"""
    SELECT g.game_id::text AS game_id, g.season, g.game_date::text AS game_date,
           g.start_time_utc, g.home_team_id::text AS home_team_id,
           g.away_team_id::text AS away_team_id,
           g.neutral_site, g.home_score, g.away_score, g.completed,
           {attrs}
    FROM public.games g
    JOIN public.teams ht ON ht.team_id=g.home_team_id AND ht.is_d1=1
    JOIN public.teams at ON at.team_id=g.away_team_id AND at.is_d1=1
    JOIN public.mbb_v2_team_pregame h
      ON h.game_id=g.game_id::text AND h.team_id=g.home_team_id::text
     AND h.feature_version='v2.0'
     AND h.snapshot_kind='historical_replay'
    JOIN public.mbb_v2_team_pregame a
      ON a.game_id=g.game_id::text AND a.team_id=g.away_team_id::text
     AND a.feature_version='v2.0'
     AND a.snapshot_kind='historical_replay'
    WHERE g.season BETWEEN %s AND %s
      AND g.neutral_site IN (0,1)
      AND g.completed=1
    ORDER BY g.season, g.game_date, g.game_id
    """


def read_db(database_url: str, min_season: int, target_season: int) -> tuple[pd.DataFrame, dict]:
    import psycopg
    from psycopg.rows import dict_row
    with psycopg.connect(database_url, sslmode='require', connect_timeout=30,
                         prepare_threshold=None, row_factory=dict_row) as con:
        con.read_only = True
        con.isolation_level = psycopg.IsolationLevel.REPEATABLE_READ
        with con.cursor() as cur:
            cur.execute(matchup_sql(), (min_season, target_season))
            rows = cur.fetchall()
            cur.execute("""
                SELECT g.season,
                    COUNT(*) FILTER (WHERE g.completed=1 AND g.home_score IS NOT NULL
                    AND g.away_score IS NOT NULL AND g.home_score<>g.away_score
                    AND NOT (g.home_score=0 AND g.away_score=2)
                    AND NOT (g.home_score=2 AND g.away_score=0)) AS completed_d1_games,
                    COUNT(*) FILTER (WHERE g.completed=0 AND
                    NULLIF(BTRIM(g.start_time_utc), '')::timestamptz > now()+interval '10 minutes') AS future_d1_games
                FROM public.games g
                JOIN public.teams ht ON ht.team_id=g.home_team_id AND ht.is_d1=1
                JOIN public.teams at ON at.team_id=g.away_team_id AND at.is_d1=1
                WHERE g.season BETWEEN %s AND %s AND g.neutral_site IN (0,1)
                GROUP BY g.season ORDER BY g.season
            """, (min_season, target_season))
            coverage = [dict(x) for x in cur.fetchall()]
    return pd.DataFrame(rows), {'db_season_coverage': coverage}


def fresh_future_inputs(database_url: str, target_season: int, now: datetime,
                        horizon_hours: float = 48.0) -> tuple[pd.DataFrame, dict]:
    """Regenerate pregame inputs at execution time from current season's results.

    No database modification. Reuses the proven previous-day snapshot generator;
    avoids the earlier V2 prospective ON CONFLICT freezing months in advance.
    """
    from db_upgrade_v2 import fetch_source, generate
    games, boxes, players, source_audit = fetch_source(database_url, target_season, target_season)
    team_rows, _, generation_audit = generate(games, boxes, players, now=now)
    cutoff = pd.Timestamp(now)+pd.Timedelta(hours=horizon_hours)
    eligible=[]
    for g in games:
        if str(g.get('completed','0')).lower() in ('1','true','t'):
            continue
        tip=pd.to_datetime(g.get('start_time_utc'),utc=True,errors='coerce')
        if pd.isna(tip) or not (pd.Timestamp(now)+pd.Timedelta(minutes=10) < tip <= cutoff):
            continue
        if str(g.get('neutral_site')) not in ('0','1','False','True') and g.get('neutral_site') not in (0,1,False,True):
            continue
        eligible.append(g)
    if not eligible:
        return pd.DataFrame(), {'d1_games_in_forecast_window':0,'feature_generation':generation_audit}
    by_team={ (str(t['game_id']),str(t['team_id'])):t for t in team_rows if t['snapshot_kind']=='prospective'}
    cols=[]
    for game in eligible:
        gid=str(game['game_id']);hid=str(game['home_team_id']);aid=str(game['away_team_id'])
        h=by_team.get((gid,hid)); a=by_team.get((gid,aid))
        if h is None or a is None: continue
        row=dict(game)
        row['game_id']=gid
        row['season']=int(game['season'])
        for prefix,t in [('home',h),('away',a)]:
            for field in TEAM_FIELDS:
                row[prefix+'_'+field]= now if field=='recorded_at' else t.get(field)
        cols.append(row)
    return pd.DataFrame(cols), {'d1_games_in_forecast_window':len(eligible),
                                 'fresh_feature_pairs_generated':len(cols),
                                 'feature_generation':generation_audit}


def _num(df: pd.DataFrame, name: str) -> pd.Series:
    return pd.to_numeric(df[name], errors='coerce')


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    required = {'game_id', 'season', 'neutral_site', 'completed', 'home_score', 'away_score',
                'game_date', 'start_time_utc'} | {f'{side}_{c}' for side in ('home','away') for c in TEAM_FIELDS}
    missing = required-set(df.columns)
    if missing:
        raise ValueError('Required V2 source columns missing: '+str(sorted(missing)))
    x = df.copy()
    x['game_id'] = x['game_id'].astype(str)
    if x.game_id.duplicated().any():
        raise ValueError('Duplicate game IDs (join ambiguity), refusing to score')
    for c in ['season', 'neutral_site', 'home_score', 'away_score']:
        x[c] = _num(x, c)
    x['completed'] = x.completed.astype(str).str.lower().isin(['1','true','t'])
    x['tipoff'] = pd.to_datetime(x.start_time_utc, errors='coerce', utc=True, format='mixed')
    x['game_date_dt'] = pd.to_datetime(x.game_date, errors='coerce', utc=True)
    x['home_court'] = 1. - x.neutral_site
    if not x.neutral_site.isin([0,1]).all():
        raise ValueError('Unknown neutral_site value')
    for side in ('home','away'):
        for col in TEAM_FIELDS:
            if col not in ('recorded_at','tipoff_utc','game_day','quality_flags'):
                x[f'{side}_{col}'] = _num(x, f'{side}_{col}')
        x[f'{side}_recorded_at'] = pd.to_datetime(x[f'{side}_recorded_at'],errors='coerce',utc=True,format='mixed')
        x[f'{side}_tipoff_utc'] = pd.to_datetime(x[f'{side}_tipoff_utc'],errors='coerce',utc=True,format='mixed')
        x[f'{side}_game_day'] = pd.to_datetime(x[f'{side}_game_day'],errors='coerce',utc=True)
    def gap(k):
        return x['home_'+k] - x['away_'+k]
    x['elo_gap'] = gap('elo_rating')
    x['net_gap'] = gap('adjusted_net_per100')
    x['games_gap'] = gap('games_before')
    x['min_games'] = x[['home_games_before','away_games_before']].min(axis=1)
    x['rest_gap'] = gap('rest_days')
    x['form_margin_gap'] = (x.home_average_points_for - x.home_average_points_against) - (x.away_average_points_for-x.away_average_points_against)
    x['sos_elo_gap'] = gap('opponent_elo_mean')
    x['off_gap'] = gap('adjusted_offense_per100')
    x['def_gap'] = gap('adjusted_defense_per100')
    x['pace_gap'] = gap('estimated_possessions')
    x['efg_gap'] = gap('effective_fg_pct')
    x['three_gap'] = gap('three_point_rate')
    x['turnovers_gap'] = gap('turnover_per_possession')
    x['free_throw_gap'] = gap('free_throw_rate')
    x['home_win'] = np.where(x.completed & x.home_score.notna() & x.away_score.notna(),
                             (x.home_score>x.away_score).astype(int), np.nan)
    for c in FULL:
        x[c] = pd.to_numeric(x[c],errors='coerce')
        x[c] = x[c].replace([np.inf,-np.inf],np.nan)
    return x


def valid_history(df: pd.DataFrame) -> pd.DataFrame:
    has_results = df.completed & df.home_score.notna() & df.away_score.notna()
    not_tied = df.home_score.ne(df.away_score)
    not_admin = ~(((df.home_score==0)&(df.away_score==2))|((df.home_score==2)&(df.away_score==0)))
    days_match = (df.home_game_day.dt.normalize().eq(df.game_date_dt.dt.normalize()) &
                  df.away_game_day.dt.normalize().eq(df.game_date_dt.dt.normalize()))
    # Replay is generated retrospectively from previous-day statistics. Do not
    # treat its recorded_at time as the historical prediction timestamp.
    return df.loc[has_results & not_tied & not_admin & days_match].copy()


def valid_future(df: pd.DataFrame, target_season: int, now=None) -> pd.DataFrame:
    now = pd.Timestamp.now(tz='UTC') if now is None else pd.Timestamp(now)
    if now.tzinfo is None:
        raise ValueError('now must be timezone aware')
    x=df.loc[(df.season==target_season) & ~df.completed &
             df.tipoff.gt(now + pd.Timedelta(minutes=10))].copy()
    if x.empty:
        return x
    valid = (x.home_recorded_at.notna() & x.away_recorded_at.notna() &
             x.home_recorded_at.le(now) & x.away_recorded_at.le(now) &
             x.home_recorded_at.lt(x.tipoff-pd.Timedelta(minutes=10)) &
             x.away_recorded_at.lt(x.tipoff-pd.Timedelta(minutes=10)) &
             x.home_tipoff_utc.eq(x.tipoff) & x.away_tipoff_utc.eq(x.tipoff) &
             x.home_game_day.dt.normalize().eq(x.game_date_dt.dt.normalize()) &
             x.away_game_day.dt.normalize().eq(x.game_date_dt.dt.normalize()))
    return x.loc[valid].copy()


def candidates():
    def logistic(c):
        return make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),
                             StandardScaler(), LogisticRegression(C=c, max_iter=1200,random_state=23))
    return {
        'basic_logistic': (BASIC,logistic(0.2)),
        'core_logistic': (CORE,logistic(0.2)),
        'core_boost': (CORE,HistGradientBoostingClassifier(max_leaf_nodes=8,max_iter=120,
                        learning_rate=.04,l2_regularization=15,min_samples_leaf=45,random_state=23)),
        'full_logistic': (FULL,logistic(0.06)),
        'full_boost': (FULL,HistGradientBoostingClassifier(max_leaf_nodes=10,max_iter=120,
                        learning_rate=.035,l2_regularization=30,min_samples_leaf=60,random_state=23)),
    }


def describe_predictions(data: pd.DataFrame, proba: np.ndarray, model: str) -> dict:
    actual=data.home_win.astype(int).to_numpy()
    prob=np.clip(np.asarray(proba,dtype=float),1e-8,1-1e-8)
    predicted=(prob>=.5).astype(int)
    return {'model':model,'games':len(actual),'correct':int((actual==predicted).sum()),
            'accuracy_pct':round(100*float((actual==predicted).mean()),3),
            'brier':round(float(brier_score_loss(actual,prob)),5),
            'log_loss':round(float(log_loss(actual,prob,labels=[0,1])),5)}


CONFIDENCE_BANDS = ('50-59%', '60-69%', '70-79%', '80-89%', '90-100%')


def winner_confidence_bands(home_win_probabilities) -> pd.Categorical:
    """The probability assigned to the predicted winner, NOT always the home team.

    Bins are [50,60), [60,70), [70,80), [80,90), [90,100].
    """
    p = np.asarray(home_win_probabilities, dtype=float)
    if not np.isfinite(p).all() or not ((p >= 0) & (p <= 1)).all():
        raise ValueError('Probability must be finite and in [0,1]')
    confidence = np.maximum(p, 1-p)
    return pd.cut(confidence, bins=[.5,.6,.7,.8,.9,1.0000000001],
                  right=False,labels=CONFIDENCE_BANDS,include_lowest=True)


def historical_game_level(test: pd.DataFrame, p_home: np.ndarray, model: str) -> pd.DataFrame:
    if len(test) != len(p_home):
        raise ValueError('Mismatch between games and probabilities')
    pred_home = np.asarray(p_home, dtype=float) >= .5
    actual_home = test.home_win.to_numpy(dtype=int) == 1
    result = pd.DataFrame({
        'game_id':test.game_id.to_numpy(), 'season':test.season.to_numpy(),
        'game_date':test.game_date.to_numpy(),
        'home_team_id':test.home_team_id.to_numpy(),
        'away_team_id':test.away_team_id.to_numpy(),
        'home_score':test.home_score.to_numpy(),
        'away_score':test.away_score.to_numpy(),
        'home_win_probability':np.asarray(p_home,dtype=float),
        'predicted_winner_side':np.where(pred_home,'home','away'),
        'predicted_winner_team_id':np.where(pred_home,test.home_team_id.to_numpy(),test.away_team_id.to_numpy()),
        'actual_winner_side':np.where(actual_home,'home','away'),
        'actual_winner_team_id':np.where(actual_home,test.home_team_id.to_numpy(),test.away_team_id.to_numpy()),
        'confidence_pct':100*np.maximum(p_home,1-p_home),
        'confidence_band':winner_confidence_bands(p_home).astype(str),
        'correct':pred_home==actual_home,
        'competitive_net_gap_le5':test.net_gap.abs().le(5).to_numpy(),
        'model':model,
        'evaluation_type':'historical_replay_not_live_verified',
    })
    return result


def historical_confidence_records(games: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for group, subset in (
        ('all',games),
        ('competitive_net_gap_le5',games.loc[games.competitive_net_gap_le5]),
        ('noncompetitive_net_gap_gt5',games.loc[~games.competitive_net_gap_le5]),
    ):
        for band in CONFIDENCE_BANDS:
            d=subset.loc[subset.confidence_band.eq(band)]
            n=len(d); wins=int(d.correct.sum()); losses=n-wins
            rows.append({'group':group,'confidence_band':band,'games':n,
                         'wins':wins,'losses':losses,
                         'accuracy_pct':round(100*wins/n,3) if n else None,
                         'average_confidence_pct':round(float(d.confidence_pct.mean()),3) if n else None,
                         'evaluation_type':'historical_replay_not_live_verified'})
    return pd.DataFrame(rows)


def model_search(data: pd.DataFrame):
    tr=data.loc[data.season.isin([2022,2023,2024])]
    val=data.loc[data.season==2025]
    test=data.loc[data.season==2026]
    if min(len(tr),len(val),len(test)) < 1000:
        raise ValueError('Insufficient history for strict 2022-24 / 2025 / 2026 evaluation: '
                         +str({'train':len(tr),'validation':len(val),'test':len(test)}))
    scores=[]
    for name,(cols,m) in candidates().items():
        m.fit(tr[cols],tr.home_win.astype(int))
        p=m.predict_proba(val[cols])[:,1]
        scores.append(describe_predictions(val,p,name))
    results=pd.DataFrame(scores).sort_values(['accuracy_pct','brier','model'],ascending=[False,True,True])
    chosen=results.iloc[0].model
    # 2026 is an evaluation season, not used to choose the champion.
    cols, candidate=candidates()[chosen]
    trainval=data.loc[data.season.isin([2022,2023,2024,2025])]
    candidate.fit(trainval[cols],trainval.home_win.astype(int))
    p_test=candidate.predict_proba(test[cols])[:,1]
    test_scores=[describe_predictions(test,p_test,chosen+'_evaluation_2026')]
    for group, mask in [('competitive_net_gap_le5',test.net_gap.abs().le(5)),
                        ('noncompetitive_net_gap_gt5',test.net_gap.abs().gt(5)),
                        ('early_either_team_fewer5_games',test.min_games.lt(5)),
                        ('other_games',test.min_games.ge(5)),
                        ('neutral_site',test.neutral_site.eq(1)),
                        ('home_venue',test.neutral_site.eq(0))]:
        if mask.any():
            r=describe_predictions(test.loc[mask],p_test[mask.to_numpy()],chosen)
            test_scores.append({'group':group,**r})
    # No test-period tuning. Refit for next-season deployment.
    champion_cols, champion = candidates()[chosen]
    champion.fit(data.loc[data.season.le(2026),champion_cols],
                 data.loc[data.season.le(2026),'home_win'].astype(int))
    game_level=historical_game_level(test,p_test,chosen)
    return chosen,champion_cols,champion,results,test_scores,game_level


def save_forecasts(database_url: str, p: pd.DataFrame, created: datetime) -> int:
    if p.empty:
        return 0
    import psycopg
    # SQL enforces the input timestamps, official future tipoff and immutable
    # forecast: ON CONFLICT DO NOTHING. Store server-only; no public RLS bypass.
    query = """
      INSERT INTO public.mbb_winner_forecasts
      (game_id,season,home_team_id,away_team_id,tipoff_utc,feature_version,
       model_version,home_win_probability,predicted_winner_team_id,
       home_feature_recorded_at,away_feature_recorded_at,created_at)
      SELECT %s,%s,%s,%s,%s,'v2.0',%s,%s,%s,%s,%s,now()
      WHERE EXISTS (
        SELECT 1 FROM public.games g WHERE g.game_id::text=%s
          AND g.completed=0 AND NULLIF(BTRIM(g.start_time_utc), '')::timestamptz > now() + interval '10 minutes'
          AND g.home_team_id::text=%s AND g.away_team_id::text=%s
          AND NULLIF(BTRIM(g.start_time_utc), '')::timestamptz=%s::timestamptz
      )
      AND %s::timestamptz < %s::timestamptz - interval '10 minutes'
      AND %s::timestamptz < %s::timestamptz - interval '10 minutes'
      ON CONFLICT (game_id,model_version) DO NOTHING
    """
    count=0
    with psycopg.connect(database_url,sslmode='require',connect_timeout=30,prepare_threshold=None) as con:
        with con.cursor() as cur:
            for r in p.itertuples(index=False):
                vals = (r.game_id,int(r.season),r.home_team_id,r.away_team_id,r.tipoff,
                        RESEARCH_VERSION,float(r.home_win_probability),r.predicted_winner_team_id,
                        r.home_recorded_at,r.away_recorded_at,r.game_id,r.home_team_id,
                        r.away_team_id,r.tipoff,r.home_recorded_at,r.tipoff,r.away_recorded_at,r.tipoff)
                cur.execute(query,vals)
                count+=cur.rowcount
        con.commit()
    return count


def run(db_url: str, outdir: Path, target_season: int=2027, save: bool=False, input_csv: str|None=None):
    outdir.mkdir(parents=True,exist_ok=True)
    now=datetime.now(timezone.utc)
    if input_csv:
        raw=pd.read_csv(input_csv,dtype={'game_id':str,'home_team_id':str,'away_team_id':str})
        audit={'source':'offline_local_fixture', 'd1_games_in_forecast_window':int((raw['completed']==0).sum())}
    else:
        history_rows,audit=read_db(db_url,2022,target_season)
        future_rows,now_audit=fresh_future_inputs(db_url,target_season,now,horizon_hours=48.0)
        audit.update(now_audit)
        raw=pd.concat([history_rows,future_rows],ignore_index=True) if len(future_rows) else history_rows
    if raw.empty:
        raise ValueError('No historical games with V2 feature rows. Run the V2 historical refresh first.')
    df=make_features(raw)
    history=valid_history(df)
    chosen,cols,champion,validation,test,game_level=model_search(history)
    future=valid_future(df,target_season,now=pd.Timestamp(now))
    if len(future):
        proba=champion.predict_proba(future[cols])[:,1]
        future['home_win_probability']=proba
        future['predicted_winner_team_id']=np.where(proba>=.5,future.home_team_id,future.away_team_id)
        future['predicted_winner_side']=np.where(proba>=.5,'home','away')
        future['model_version']=RESEARCH_VERSION
        future['snapshot_generated_at_utc']=pd.Timestamp(now).isoformat()
    display=['game_id','season','game_date','tipoff','home_team_id','away_team_id',
             'neutral_site','predicted_winner_side','predicted_winner_team_id',
             'home_win_probability','model_version','snapshot_generated_at_utc',
             'home_recorded_at','away_recorded_at']
    if future.empty:
        pd.DataFrame(columns=display).to_csv(outdir/'next_game_winners.csv',index=False)
    else:
        future[display].to_csv(outdir/'next_game_winners.csv',index=False)
    validation.to_csv(outdir/'candidate_validation_2025.csv',index=False)
    pd.DataFrame(test).to_csv(outdir/'winner_accuracy_2026.csv',index=False)
    game_level.to_csv(outdir/'winner_game_level_2026.csv',index=False)
    confidence_summary=historical_confidence_records(game_level)
    confidence_summary.to_csv(outdir/'confidence_records_2026.csv',index=False)
    inserted=0
    if save and not future.empty:
        inserted=save_forecasts(db_url,future,now)
    scored_2026=next(r for r in test if r.get('group') is None)
    available=int(audit.get('d1_games_in_forecast_window',0))
    audit.update({'research_version':RESEARCH_VERSION,
                  'goal_winner_accuracy_pct':90,
                  'target_achieved_on_2026_replay':scored_2026['accuracy_pct']>=90,
                  'selected_by':'2025 historical winner accuracy; Brier tiebreak',
                  'validation_winner_accuracy_pct':validation.iloc[0]['accuracy_pct'],
                  'evaluation_2026':scored_2026,
                  'confidence_report_2026':{'games':len(game_level),'bins':list(CONFIDENCE_BANDS),'method':'predicted winner probability'},
                  'evaluation_not_pristine':'2026 results were previously inspected in this project',
                  'historical_games_joined_feature_store':int(df.completed.sum()),
                  'history_games_used':len(history),
                  'historical_games_by_season':{str(k):int(v) for k,v in history.season.value_counts().sort_index().items()},
                  'future_games_scored':len(future),
                  'future_games_missing_features':max(0,available-len(future)),
                  'near_tipoff_coverage_pct':round(100*len(future)/available,2) if available else None,
                  'future_forecasts_inserted':inserted,
                  'prospective_source':'fresh read-only recomputation using db_upgrade_v2.generate',
                  'predict_close_to_tipoff_window_hours':48,
                  'prospective_source_requires_recorded_before_tipoff':True,
                  'historical_replay_is_live_verified':False,
                  'all_competitive_and_noncompetitive_scored':True,
                  'important':'No guarantee of 90%; historical replay does not establish prospective accuracy.'})
    (outdir/'run_audit.json').write_text(json.dumps(audit,indent=2,default=str)+'\n')
    print(json.dumps(audit,indent=2,default=str))

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output-dir',default='winner_engine_results')
    parser.add_argument('--target-season',type=int,default=2027)
    parser.add_argument('--save-prospective',action='store_true')
    parser.add_argument('--input-csv',default=None,help='Offline fixture only, same columns as Supabase matchups')
    args=parser.parse_args()
    database_url=os.environ.get('DATABASE_URL','')
    if not database_url and not args.input_csv:
        parser.error('DATABASE_URL must be provided by a server-side GitHub Actions secret')
    if args.save_prospective and not database_url:
        parser.error('Cannot save forecasts without DATABASE_URL')
    run(database_url,Path(args.output_dir),args.target_season,args.save_prospective,args.input_csv)
