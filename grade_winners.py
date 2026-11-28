#!/usr/bin/env python3
"""Score immutable, pretip-off game-winner forecasts against completed games."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import pandas as pd


def read_results(url: str):
    import psycopg
    from psycopg.rows import dict_row
    sql = """
        SELECT f.game_id, f.season, f.tipoff_utc, f.created_at,
          f.home_team_id, f.away_team_id, f.predicted_winner_team_id,
          f.home_win_probability, f.model_version, g.completed,
          g.home_score, g.away_score
        FROM public.mbb_winner_forecasts f
        JOIN public.games g ON f.game_id=g.game_id::text
        WHERE f.model_version='winner-v1.0'
        ORDER BY f.tipoff_utc,f.game_id
    """
    with psycopg.connect(url,sslmode='require',connect_timeout=30,
                         prepare_threshold=None,row_factory=dict_row) as con:
        con.read_only=True
        with con.cursor() as cur:
            cur.execute(sql)
            return pd.DataFrame(cur.fetchall())


def evaluate(df: pd.DataFrame) -> tuple[pd.DataFrame,dict]:
    if df.empty:
        return df,{'saved_forecasts':0,'graded_games':0,'correct':0,'winner_accuracy_pct':None,'goal_pct':90,'goal_met':None}
    x=df.copy()
    x['created_at']=pd.to_datetime(x.created_at,utc=True,errors='coerce')
    x['tipoff_utc']=pd.to_datetime(x.tipoff_utc,utc=True,errors='coerce')
    x['home_score']=pd.to_numeric(x.home_score,errors='coerce')
    x['away_score']=pd.to_numeric(x.away_score,errors='coerce')
    x['completed']=x.completed.astype(str).str.lower().isin(('1','true','t'))
    if (x.created_at.isna()|x.tipoff_utc.isna()|x.created_at.ge(x.tipoff_utc)).any():
        raise ValueError('Forecast timestamp missing or not before tipoff; aborting evaluation')
    if x.game_id.duplicated().any():
        raise ValueError('Duplicate forecast IDs')
    valid=(x.completed & x.home_score.notna() & x.away_score.notna() &
           x.home_score.ne(x.away_score) &
           ~(((x.home_score==0)&(x.away_score==2)) | ((x.home_score==2)&(x.away_score==0))))
    scored=x.loc[valid].copy()
    winner=scored.home_team_id.where(scored.home_score>scored.away_score,scored.away_team_id)
    scored['actual_winner_team_id']=winner.astype(str)
    scored['correct']=scored.predicted_winner_team_id.astype(str).eq(scored.actual_winner_team_id)
    acc=round(100*float(scored.correct.mean()),3) if len(scored) else None
    return scored,{'saved_forecasts':len(x),'graded_games':len(scored),
                   'ungraded_games':len(x)-len(scored),'correct':int(scored.correct.sum()),
                   'winner_accuracy_pct':acc,'goal_pct':90,
                   'goal_met':None if acc is None else bool(acc>=90),
                   'strictly_pretip_forecasts':True}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output-dir',default='winner_engine_results')
    args=parser.parse_args()
    url=os.environ.get('DATABASE_URL')
    if not url:parser.error('Missing DATABASE_URL GitHub secret')
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    scored,summary=evaluate(read_results(url))
    scored.to_csv(out/'graded_pretip_forecasts.csv',index=False)
    (out/'prospective_accuracy.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
