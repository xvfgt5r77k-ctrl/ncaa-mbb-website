"""Read-only NCAA MBB prospective feature audit, after final results only.

No upcoming-game forecasts, wagering integrations, or database modifications.
"""
from __future__ import annotations
import csv
import json
import os
from pathlib import Path
from collections import defaultdict

import psycopg2

QUERY = """
WITH first_snapshots AS (
    SELECT DISTINCT ON (season, game_id::text, team_id::text)
       season, game_id::text AS game_id, team_id::text AS team_id,
       recorded_at, tipoff_utc, adjusted_net_per100
    FROM public.mbb_v2_team_pregame
    WHERE season = %s AND snapshot_kind = 'prospective'
    ORDER BY season, game_id::text, team_id::text, recorded_at ASC
)
SELECT g.game_id::text, g.home_score, g.away_score,
       COALESCE(g.neutral_site, 0),
       h.adjusted_net_per100, a.adjusted_net_per100
FROM public.games AS g
JOIN first_snapshots AS h
  ON h.game_id = g.game_id::text
 AND h.season = g.season
 AND h.team_id = g.home_team_id::text
JOIN first_snapshots AS a
  ON a.game_id = g.game_id::text
 AND a.season = g.season
 AND a.team_id = g.away_team_id::text
WHERE g.season = %s
  AND g.completed = 1
  AND g.home_score IS NOT NULL
  AND g.away_score IS NOT NULL
  AND g.home_score <> g.away_score
  AND h.adjusted_net_per100 IS NOT NULL
  AND a.adjusted_net_per100 IS NOT NULL
  AND h.adjusted_net_per100 <> a.adjusted_net_per100
  AND h.recorded_at < h.tipoff_utc - INTERVAL '30 minutes'
  AND a.recorded_at < a.tipoff_utc - INTERVAL '30 minutes'
  AND h.tipoff_utc <= clock_timestamp()
  AND a.tipoff_utc <= clock_timestamp()
ORDER BY g.game_id
"""


def evaluate(rows):
    evaluated=[]
    stats=defaultdict(lambda: {'games':0,'correct':0})
    for game_id, home_score, away_score, neutral, home_net, away_net in rows:
        home_win = home_score > away_score
        favors_home = home_net > away_net
        competitive = abs(float(home_net)-float(away_net)) <= 5
        group='competitive' if competitive else 'other'
        venue='neutral' if neutral else 'home_court'
        correct=home_win == favors_home
        evaluated.append({
            'game_id':str(game_id),
            'home_score':int(home_score),
            'away_score':int(away_score),
            'venue':venue,
            'competitive':competitive,
            'rating_result':'correct' if correct else 'incorrect',
        })
        for key in ('all', group, venue):
            stats[key]['games']+=1
            stats[key]['correct']+=int(correct)
    summary={
        'study':'Completed-game retrospective audit of saved prospective team ratings',
        'count':len(evaluated),
        'caution':'This checks only prior saved adjusted-efficiency ratings. It is not an ESPN or V8 probability evaluation. Original upstream availability is unverified.',
        'groups':{k:{**v,'accuracy_pct':round(100*v['correct']/v['games'],2)} for k,v in sorted(stats.items())},
    }
    return evaluated,summary


def main():
    url=os.getenv('DATABASE_URL')
    if not url:
        raise SystemExit('Missing DATABASE_URL secret')
    out=Path('postgame_snapshot_audit')
    out.mkdir(exist_ok=True)
    with psycopg2.connect(url,connect_timeout=20,sslmode='require') as conn:
        conn.set_session(readonly=True)
        with conn.cursor() as cur:
            cur.execute(QUERY,(2027,2027))
            rows=cur.fetchall()
    evaluated,summary=evaluate(rows)
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    with (out/'completed_games_only.csv').open('w',newline='',encoding='utf-8') as fp:
        writer=csv.DictWriter(fp,fieldnames=['game_id','home_score','away_score','venue','competitive','rating_result'])
        writer.writeheader()
        writer.writerows(evaluated)
    print(json.dumps({'completed_games_evaluated':summary['count'],'group_accuracy':summary['groups']},indent=2))

if __name__=='__main__':
    main()
