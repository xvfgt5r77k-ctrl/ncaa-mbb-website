"""Read-only NCAA MBB ESPN game-projection feasibility and calibration pilot.

This study retrieves TODAY's ESPN game-summary predictor for completed 2026 games.
The returned predictor IS NOT a verified pre-tipoff snapshot. Results are descriptive
and MUST NOT be called pregame/live forecasting performance.

No market data, odds, wagering recommendations or database writes.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time
from urllib.parse import urlencode

import pandas as pd
import requests

ESPN_BASE = 'https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/summary'
REQUIRED = ['game_id','season','game_date','neutral_site','home_win',
            'selected_home_win_probability','v8_hybrid_home_win_prob','v8_strict_home_win_prob']
MODEL_COLUMNS = {
    'ESPN retrieved predictor': 'espn_home_probability',
    'V2 selected adjusted efficiency': 'selected_home_win_probability',
    'V8 hybrid (archived)': 'v8_hybrid_home_win_prob',
    'V8 strict (archived)': 'v8_strict_home_win_prob',
}


def fetch_identity_and_rating_gaps(game_ids: list[str]) -> pd.DataFrame:
    """Single read-only DB query: team IDs and relative adjusted efficiencies."""
    db_url = os.environ.get('DATABASE_URL')
    if not db_url:
        raise RuntimeError('Missing DATABASE_URL secret. This study needs team IDs to verify ESPN orientation.')
    import psycopg
    from psycopg.rows import dict_row
    sql = '''
    SELECT g.game_id::text AS game_id,
           g.home_team_id::text AS home_team_id,
           g.away_team_id::text AS away_team_id,
           g.start_time_utc::text AS tipoff_utc,
           h.adjusted_net_per100 AS home_net,
           a.adjusted_net_per100 AS away_net
    FROM public.games g
    JOIN public.mbb_v2_team_pregame h
      ON h.game_id = g.game_id::text
     AND h.team_id = g.home_team_id::text
     AND h.snapshot_kind = 'historical_replay'
     AND h.feature_version = 'v2.0'
    JOIN public.mbb_v2_team_pregame a
      ON a.game_id = g.game_id::text
     AND a.team_id = g.away_team_id::text
     AND a.snapshot_kind = 'historical_replay'
     AND a.feature_version = 'v2.0'
    WHERE g.season = 2026 AND g.completed = 1
      AND g.game_id::text = ANY(%s)
    '''
    with psycopg.connect(db_url, sslmode='require', connect_timeout=25,
                         prepare_threshold=None, row_factory=dict_row) as con:
        con.read_only = True
        con.isolation_level = psycopg.IsolationLevel.REPEATABLE_READ
        with con.cursor() as cur:
            cur.execute(sql, (game_ids,))
            rows = cur.fetchall()
    identity = pd.DataFrame(rows)
    if identity.empty:
        raise RuntimeError('No completed historical-replay games found in Supabase.')
    identity['game_id'] = identity['game_id'].astype(str)
    if identity['game_id'].duplicated().any():
        raise RuntimeError('Database returned duplicate games. Refusing ambiguous team mapping.')
    return identity


def load_eligible(reference_path: Path, identity: pd.DataFrame) -> pd.DataFrame:
    df = pd.read_csv(reference_path, dtype={'game_id':str})
    absent = set(REQUIRED) - set(df.columns)
    if absent:
        raise RuntimeError(f'Missing model reference columns: {sorted(absent)}')
    if df.game_id.duplicated().any():
        raise RuntimeError('Reference game IDs are duplicated.')
    df = df.loc[df.season.eq(2026)].merge(identity,on='game_id',validate='one_to_one',how='inner')
    if df.empty:
        raise RuntimeError('Zero overlap between reference file and database.')
    for col in MODEL_COLUMNS.values():
        if col == 'espn_home_probability':
            continue
        df[col] = pd.to_numeric(df[col],errors='coerce')
        if df[col].isna().any() or not df[col].between(0,1).all():
            raise RuntimeError(f'Invalid archived model probabilities: {col}')
    for col in ('home_net','away_net'):
        df[col] = pd.to_numeric(df[col],errors='coerce')
    df['net_difference'] = df.home_net - df.away_net
    df['competitive'] = df.net_difference.abs().le(5)
    df['neutral_site'] = pd.to_numeric(df.neutral_site,errors='coerce')
    df['home_win'] = pd.to_numeric(df.home_win,errors='coerce')
    df['tipoff_utc'] = pd.to_datetime(df.tipoff_utc, utc=True, errors='coerce')
    if df.tipoff_utc.isna().any() or not df.home_win.isin([0,1]).all():
        raise RuntimeError('Database tipoff timestamps or game outcomes missing.')
    df = df.loc[df.home_net.notna() & df.away_net.notna() & df.neutral_site.isin([0,1])]
    if df.empty:
        raise RuntimeError('No games have valid adjusted-efficiency and venue features.')
    return df


def sample_balanced(df: pd.DataFrame, max_games: int) -> pd.DataFrame:
    """Deterministic 50:50 competitive/noncompetitive feasibility pilot.

    Not a representative random draw of the entire 2026 schedule.
    """
    if not 2 <= max_games <= 300:
        raise ValueError('Pilot max_games must be 2..300 for polite ESPN request volume.')
    temp=df.copy()
    temp['_priority'] = temp.game_id.apply(
        lambda gid: hashlib.sha256(('espn-research-2026-v1:'+str(gid)).encode()).hexdigest())
    competitive = temp.loc[temp.competitive].sort_values('_priority')
    other = temp.loc[~temp.competitive].sort_values('_priority')
    cap=min(max_games,len(temp))
    n1=min(cap//2,len(competitive))
    n2=min(cap//2,len(other))
    remainder=cap-n1-n2
    extra1=min(remainder,len(competitive)-n1)
    n1+=extra1
    remainder-=extra1
    n2+=min(remainder,len(other)-n2)
    return pd.concat([competitive.head(n1),other.head(n2)]).sort_values('_priority').drop(columns='_priority').reset_index(drop=True)


def as_probability(home_projection, away_projection) -> float | None:
    """Accept values adding approximately to 1 or to 100; otherwise reject."""
    try:
        def numeric(v):
            if isinstance(v,bool) or v is None:
                return None
            return float(str(v).strip().rstrip('%'))
        h=numeric(home_projection)
        a=numeric(away_projection)
        if h is None or a is None or not all(map(math.isfinite,[h,a])):
            return None
        if 0<=h<=1 and 0<=a<=1 and abs(h+a-1)<=0.025:
            return h
        if 0<=h<=100 and 0<=a<=100 and abs(h+a-100)<=2.5:
            return h/100
    except (ValueError,TypeError):
        pass
    return None


def parse_espn_summary(payload: dict, home_id: str, away_id: str) -> tuple[float|None,str]:
    """Only the pregame-style `predictor` field, NEVER in-play winprobability.

    Requires ESPN team IDs to exactly match our home/away assignment.
    """
    if not isinstance(payload, dict):
        return None, 'not_an_object'
    predictor=payload.get('predictor')
    if not isinstance(predictor,dict):
        return None, 'no_predictor'
    home=predictor.get('homeTeam')
    away=predictor.get('awayTeam')
    if not isinstance(home,dict) or not isinstance(away,dict):
        return None, 'missing_predictor_teams'
    if home.get('id') is None or away.get('id') is None:
        return None, 'missing_predictor_team_ids'
    if str(home['id'])!=str(home_id) or str(away['id'])!=str(away_id):
        return None, 'team_id_mismatch'
    p=as_probability(home.get('gameProjection'),away.get('gameProjection'))
    if p is None:
        return None, 'invalid_or_missing_projection'
    return p, 'found'


def fetch_forecast(session: requests.Session, game_id: str,
                   home_id: str, away_id: str, pause: float) -> dict:
    now = datetime.now(timezone.utc).isoformat()
    url = f'{ESPN_BASE}?{urlencode({"event":str(game_id)})}'
    record = {'game_id':str(game_id), 'retrieved_at_utc':now,
              'source_url':url,'source_field':'predictor.homeTeam.gameProjection',
              'historical_timing_verified':False, 'status':'unknown',
              'espn_home_probability':None}
    try:
        response=session.get(url,timeout=18)
        if response.status_code == 429:
            record['status']='rate_limited'
            return record
        if response.status_code != 200:
            record['status']=f'http_{response.status_code}'
            return record
        payload=response.json()
        p,status=parse_espn_summary(payload,home_id,away_id)
        record.update(status=status,espn_home_probability=p)
    except requests.RequestException:
        record['status']='request_error'
    except (ValueError,TypeError):
        record['status']='invalid_json'
    finally:
        if pause:
            time.sleep(pause)
    return record


def measure(y: pd.Series, p: pd.Series) -> dict:
    yi=y.astype(float).to_numpy()
    pi=p.astype(float).to_numpy()
    if len(pi)==0:
        return {'n':0,'accuracy_pct':None,'brier_score':None,'log_loss':None}
    pi=pi.clip(1e-6,1-1e-6)
    correct=int(((pi>=0.5).astype(int)==yi).sum())
    return {'n':int(len(pi)), 'correct':correct,
            'accuracy_pct':round(100*correct/len(pi),3),
            'brier_score':round(float(((pi-yi)**2).mean()),5),
            'log_loss':round(float((-yi*pd.Series(pi).map(math.log).to_numpy()-(1-yi)*pd.Series(1-pi).map(math.log).to_numpy()).mean()),5)}


def evaluate(found: pd.DataFrame) -> tuple[pd.DataFrame,pd.DataFrame]:
    summaries=[]
    calibrations=[]
    if found.empty:
        return pd.DataFrame(columns=['group','model','n','correct','accuracy_pct','brier_score','log_loss']),pd.DataFrame()
    for group,group_df in [('all_pilot_matches',found),
                           ('competitive',found.loc[found.competitive]),
                           ('other',found.loc[~found.competitive]),
                           ('neutral',found.loc[found.neutral_site.eq(1)]),
                           ('nonneutral',found.loc[found.neutral_site.eq(0)])]:
        if len(group_df)==0:
            continue
        for name,col in MODEL_COLUMNS.items():
            stats=measure(group_df.home_win,group_df[col])
            summaries.append({'group':group,'model':name,**stats})
    for name,col in MODEL_COLUMNS.items():
        copy=found[['home_win',col]].copy()
        copy['bucket']=(copy[col]*10).clip(upper=9.99999).astype(int)
        for bucket,group in copy.groupby('bucket'):
            calibrations.append({'model':name,'bin_min':round(bucket/10,1),
               'bin_max':round((bucket+1)/10,1), 'games':len(group),
               'mean_predicted_home_win_prob':round(group[col].mean(),4),
               'actual_home_win_rate':round(group.home_win.mean(),4)})
    return pd.DataFrame(summaries), pd.DataFrame(calibrations)


def write_report(output_dir: Path, pilot: pd.DataFrame, retrieval: pd.DataFrame,
                 results: pd.DataFrame, calibration: pd.DataFrame, total_pool: pd.DataFrame):
    output_dir.mkdir(parents=True,exist_ok=True)
    retrieval.to_csv(output_dir/'espn_retrieval_audit.csv',index=False)
    paired=pilot.merge(retrieval[['game_id','status','espn_home_probability','retrieved_at_utc']],
                       on='game_id',how='left',validate='one_to_one')
    paired.to_csv(output_dir/'pilot_games_all_statuses.csv',index=False)
    found=paired.loc[paired.status.eq('found') & paired.espn_home_probability.notna()].copy()
    found.to_csv(output_dir/'exact_game_forecast_comparison.csv',index=False)
    results.to_csv(output_dir/'paired_model_metrics.csv',index=False)
    calibration.to_csv(output_dir/'calibration_by_probability_bin.csv',index=False)
    statuses={str(k):int(v) for k,v in retrieval.status.value_counts().items()}
    audit={
        'purpose':'retrospective feasibility and calibration research (not verified pregame snapshots)',
        'source':'ESPN site game summary predictor only (not winprobability and not market data)',
        'source_asof_tipoff_verified':False,
        'all_historical_espn_values_retrieved_postgame':True,
        'interpretation_warning':'Do not claim future/pregame performance from these ESPN retrospective observations.',
        'selection':'Fixed-hash, balanced 50/50 competitive vs other; not schedule-representative',
        'competitive_definition':'absolute home-away adjusted net efficiency <=5 per 100 possessions',
        'database_writes':False,
        '2026_historical_reference_pool':int(len(total_pool)),
        '2026_pilot_attempted':int(len(pilot)),
        'competitive_attempted':int(pilot.competitive.sum()),
        'noncompetitive_attempted':int((~pilot.competitive).sum()),
        'espn_forecasts_found':int(len(found)),
        'coverage_percent':round(100*len(found)/len(pilot),2) if len(pilot) else None,
        'retrieval_statuses':statuses,
    }
    (output_dir/'pilot_summary.json').write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8')
    lines=[
        '# NCAA MBB ESPN Probability Feasibility Pilot',
        '',
        '**CAUTION:** These ESPN predictor values were retrieved after the 2026 games ended. '
        'Their historical pre-tipoff publication timestamps are NOT verified. '
        'This is an exploratory historical comparison, NOT a prospective accuracy test.',
        '',
        f'- Historical evaluation games available: **{audit["2026_historical_reference_pool"]}**',
        f'- Stratified games sampled: **{len(pilot)}**',
        f'- Competitive games sampled (efficiency gap <=5): **{audit["competitive_attempted"]}**',
        f'- ESPN forecasts found: **{len(found)}** (**{audit["coverage_percent"]}%**) ',
        '- Retrieval statuses: '+', '.join(f'{key}={value}' for key,value in statuses.items()),
        '',
        '## Exact-game paired comparisons',
    ]
    if results.empty:
        lines+=['','ESPN predictor data was unavailable for all queried historical games.',
                'No accuracy or calibration claim is possible; see `espn_retrieval_audit.csv`.']
    else:
        lines+=['','All three models below use the exact same subset where ESPN predictor values were found.',
                'This small stratified pilot is not representative of the full season.','',
                results.to_markdown(index=False), '',
                'See `calibration_by_probability_bin.csv` for sample counts and observed win rates.']
    lines+=['','## Next phase',
        'For validated pretipoff probabilities, record ESPN forecasts with retrieval timestamps '
        'BEFORE future games begin, then grade those observations only after the games finish.',
        'Historical ESPN values must remain explicitly marked as timing-unverified.',
    ]
    (output_dir/'README_RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return audit


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',default='reference/2026_model_reference.csv')
    parser.add_argument('--output-dir',default='espn_probability_pilot_results')
    parser.add_argument('--max-games',type=int,default=120)
    parser.add_argument('--pause-seconds',type=float,default=0.5)
    args=parser.parse_args(argv)
    if not 0.3<=args.pause_seconds<=10:
        parser.error('--pause-seconds must be at least 0.3')
    reference=pd.read_csv(args.reference,usecols=['game_id'],dtype={'game_id':str})
    identity=fetch_identity_and_rating_gaps(reference.game_id.tolist())
    full=load_eligible(Path(args.reference),identity)
    sample=sample_balanced(full,args.max_games)
    print(f'Found {len(full)} eligible 2026 historical records; requesting {len(sample)} ESPN game summaries.')
    client=requests.Session()
    client.headers.update({'User-Agent':'NCAA-MBB-academic-forecast-audit/1.0 (read-only research)',
                           'Accept':'application/json'})
    retrieval=[]
    for i,row in enumerate(sample.itertuples(index=False),1):
        out=fetch_forecast(client,row.game_id,row.home_team_id,row.away_team_id,args.pause_seconds)
        retrieval.append(out)
        if i%20==0:
            print(f'Checked {i}/{len(sample)}; predictor found in {sum(x["status"]=="found" for x in retrieval)} games.')
        if out['status']=='rate_limited':
            print('ESPN rate limit encountered. Stopping requests and exporting partial audit.')
            break
    audit_frame=pd.DataFrame(retrieval,columns=['game_id','retrieved_at_utc','source_url','source_field',
                                                 'historical_timing_verified','status','espn_home_probability'])
    paired=sample.merge(audit_frame[['game_id','status','espn_home_probability']],on='game_id',how='inner')
    found=paired.loc[paired.status.eq('found')].copy()
    metrics,calibration=evaluate(found)
    audit=write_report(Path(args.output_dir),sample,audit_frame,metrics,calibration,full)
    print(json.dumps(audit,indent=2))
    print(f'Artifacts written to {args.output_dir}')


if __name__=='__main__':
    main()
