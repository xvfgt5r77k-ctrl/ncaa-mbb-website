-- Shows ALL completed games with a pre-tipoff saved winner forecast.
-- Coverage is reported separately: never call model accuracy 'overall' if games were missed.
SELECT f.model_version,
       COUNT(*) AS graded_games,
       COUNT(*) FILTER (WHERE f.predicted_winner_team_id =
           CASE WHEN g.home_score > g.away_score THEN g.home_team_id::text
                ELSE g.away_team_id::text END) AS winners_correct,
       ROUND(100.0 * COUNT(*) FILTER (WHERE f.predicted_winner_team_id =
           CASE WHEN g.home_score > g.away_score THEN g.home_team_id::text
                ELSE g.away_team_id::text END) / NULLIF(COUNT(*),0),2) AS winner_accuracy_pct
FROM public.mbb_winner_forecasts f
JOIN public.games g ON f.game_id = g.game_id::text
WHERE g.completed=1 AND g.home_score IS NOT NULL AND g.away_score IS NOT NULL
  AND g.home_score<>g.away_score
  AND NOT ((g.home_score=0 AND g.away_score=2) OR (g.home_score=2 AND g.away_score=0))
  AND f.created_at < f.tipoff_utc
GROUP BY f.model_version ORDER BY f.model_version;
