-- Run ONCE in Supabase SQL Editor to add the dashboard-managed dollar ledger.
-- All subsequent changes are entered in Streamlit; no SQL edits are needed.
CREATE TABLE IF NOT EXISTS public.mbb_dollar_tracker_entries (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    delta_amount NUMERIC(14,2) NOT NULL CHECK (delta_amount <> 0),
    note TEXT NOT NULL DEFAULT '' CHECK (char_length(note) <= 180),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS mbb_dollar_tracker_time_idx
    ON public.mbb_dollar_tracker_entries (created_at, id);

-- Avoid accidentally exposing financial values through Supabase public APIs.
ALTER TABLE public.mbb_dollar_tracker_entries ENABLE ROW LEVEL SECURITY;
REVOKE ALL ON TABLE public.mbb_dollar_tracker_entries FROM PUBLIC, anon, authenticated;
REVOKE ALL ON SEQUENCE public.mbb_dollar_tracker_entries_id_seq FROM PUBLIC, anon, authenticated;

-- The Streamlit server's DATABASE_URL (or TRACKER_DATABASE_URL) must connect as
-- the table owner / other appropriately authorized DB role able to SELECT and INSERT.
-- Do not use the publishable/anon key or put the connection string in GitHub source.
