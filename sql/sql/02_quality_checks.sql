-- Read-only verification after the first GitHub Feature Store V2 refresh.

-- 1. Which seasons and snapshot types have been populated?
SELECT season, snapshot_kind,
       COUNT(*) AS team_feature_rows,
       COUNT(*) FILTER (WHERE games_before >= 5) AS established_team_history,
       COUNT(*) FILTER (WHERE player_coverage = 'sufficient') AS player_coverage_sufficient,
       COUNT(*) FILTER (WHERE 'player_history_incomplete' = ANY(quality_flags)) AS player_coverage_incomplete
FROM public.mbb_v2_team_pregame
GROUP BY season, snapshot_kind ORDER BY season, snapshot_kind;

-- 2. Ensure genuine prospective features were created before tipoff.
SELECT COUNT(*) AS saved_prospective_team_rows,
       COUNT(*) FILTER (WHERE recorded_at >= tipoff_utc) AS late_or_invalid_saves,
       COUNT(*) FILTER (WHERE tipoff_utc IS NULL) AS missing_tipoff
FROM public.mbb_v2_team_pregame
WHERE snapshot_kind = 'prospective';

-- 3. Recent refresh outcomes.
SELECT id, started_at, finished_at, status, source_games,
       source_team_boxes, source_player_boxes,
       team_rows_inserted, player_rows_inserted
FROM public.mbb_v2_refresh_log
ORDER BY id DESC LIMIT 10;

-- 4. Historical rows must not be confused with pregame-saved forecasts.
SELECT snapshot_kind, COUNT(*) AS records,
       COUNT(*) FILTER (WHERE 'reconstructed_not_saved_pretipoff' = ANY(quality_flags)) AS flagged_replay
FROM public.mbb_v2_team_pregame
GROUP BY snapshot_kind ORDER BY snapshot_kind;
