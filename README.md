# NCAA Men's Basketball — Cloud Website Starter

**Goal:** A website you can open on your phone, laptop or tablet, with basketball data updating overnight **without your computer being turned on**.

This is the **cloud conversion** of your prior ESPN-based starter project. It includes source code, a mobile-friendly Streamlit website, a persistent PostgreSQL database connector, and a nightly GitHub Actions update. **It is not deployed yet.** You must create/connect your own hosting and database accounts once. A nightly run will only work after deployment, credentials and a first import are configured.

## How it works

```text
ESPN public-facing college basketball endpoints
       |
GitHub Actions Python importer (nightly, 3:37 AM America/Chicago)
       |
Supabase managed PostgreSQL database
       |
Streamlit Community Cloud website (shareable HTTPS link)
       |
Phone / Tablet / Laptop
```

### What it can display

- Men's **NCAA Division I** team efficiency rankings (offensive, defensive, net efficiency), estimated possessions, shooting and turnover metrics
- Game schedules, scores, completed results, team explorer and game-by-game player box scores
- Season selector, CSV downloads and importer status/warnings
- Injury tab reserved for authorized **RotoWire** integration (not active; no placeholder injury claims)

## Initial setup (no programming required)

You need three free-tier-capable services: **GitHub** (stores code and runs updates), **Supabase** (cloud database), **Streamlit Community Cloud** (hosts the website). Service terms, availability, and free-tier limits can change.

1. **Create the database.** Open https://supabase.com/ and create an account, then create a new project. Save your database password in a password manager. In its dashboard click **Connect → Session pooler**. Copy its complete PostgreSQL connection string (typically ending in `:5432/postgres`). Replace `[YOUR-PASSWORD]` with your password, **percent-encoding special characters** if needed. Do not share or publish this connection string. This project connects with encryption. You do **not** need a Supabase API key.
2. **Upload code to GitHub.** Open https://github.com/new and create a repository named e.g. `ncaa-mbb-website`. Extract this ZIP **first**, then upload the **contents** of its folder into the root of the new repository (not the ZIP itself). Make sure `.github/workflows/nightly.yml` and `cloud_dashboard.py` are at their correct paths. If the GitHub website upload tool doesn't preserve the `.github` nested folder, use **GitHub Desktop** to commit and publish this directory: https://desktop.github.com/ . You can use a public repository for sports-statistics code; don't put passwords in repository files.
3. **Add the GitHub secret.** In your repository open **Settings → Secrets and variables → Actions → New repository secret**. Name it exactly **`DATABASE_URL`** and paste the Supabase **Session pooler** connection string as its value. You will never need an ESPN API key for this project.
4. **Run the first importer.** In the repository choose **Actions → NCAA MBB Nightly ESPN Update → Run workflow**. For an initial historical import, enter `start_date: 2026-11-01` and `end_date: 2026-11-23` (or your desired dates). Check the run's output and confirm the job finishes successfully. Long imports can hit ESPN rate limits or the 45-minute workflow limit; if that happens, choose shorter date ranges and retry. Once your first import succeeds, the cloud database keeps the records even though GitHub's temporary job machine is deleted. On subsequent nights, new and updated records merge into the same database rather than replacing history.
5. **Publish the website.** Visit https://share.streamlit.io/ and sign in with GitHub. Select your repository and branch, and set the **main file path** to **`cloud_dashboard.py`**. In **Advanced settings → Secrets**, paste:
   ```toml
   DATABASE_URL = "postgresql://YOUR_USER:YOUR_PASSWORD@YOUR_SESSION_POOLER_HOST:5432/postgres"
   ```
   Use the same real, properly encoded connection string you entered in GitHub. Click **Deploy**. Streamlit will provide a `*.streamlit.app` address that you can bookmark and open from any device with internet access. Streamlit stores app secrets server-side; do not print this value in the app or commit `.streamlit/secrets.toml` to GitHub.
6. **Check it.** Open your deployed app, select **Update status**, and confirm you see a completed import. Open **Team efficiency**, **Games & schedules** and **Player box scores**. If you see no complete games yet, run a historical backfill for a period with finished games. You can refresh the page via the button in the app; page data has a five-minute cache.

**Important:** You will need the database connection string in *two places*: GitHub Actions repository secrets (writer) and Streamlit Cloud app secrets (website reader). The connection string is like a password. **Never paste it into chat, a public GitHub file, or a screenshot.** If compromised, reset the database password and update both secrets.

## Nightly updates

The GitHub Actions schedule is in `.github/workflows/nightly.yml`: every day at **3:37 AM America/Chicago** (automatically adjusts for daylight saving time). This runs on GitHub's cloud computers: your home PC can be completely turned off. The workflow also supports **Run workflow** on demand, with optional historical date range.

The default update checks the past **4 calendar days** and next **14 days**. This captures recent game results/stat corrections and upcoming schedules. If a nightly run fails, inspect **Actions** logs and rerun manually. If it hasn't updated for more than 4 days, do a historical backfill to catch missed games. GitHub schedules can run late or occasionally be missed; they are **not a guaranteed execution time**.

**Free-tier limitations:** GitHub automatically disables scheduled workflows in some public repositories after **60 days without repository activity**. A paused Supabase free project or unavailable Streamlit app can also interrupt service. Check the app's **Update status** and GitHub Actions regularly. For guaranteed 24/7 uptime and scheduling, paid hosting/service agreements may be necessary.

## Files

| File | What it's for |
|---|---|
| `cloud_dashboard.py` | Hosted website view on phone/computer |
| `cloud_sync.py` | Nightly / manual ESPN downloader and cloud publisher |
| `cloud_store.py` | Supabase PostgreSQL schema, protected connection, upserts |
| `.github/workflows/nightly.yml` | Nightly cloud scheduler and manual backfill action |
| `espn.py` | ESPN fetching and game/box-score parsing |
| `pipeline.py` | SQLite staging importer; preserves incomplete data checks |
| `db.py` | SQLite staging schema and unadjusted efficiency formulas |
| `main.py` | Original offline CLI and optional local testing |
| `requirements.txt` | Python dependencies |
| `tests/` | Offline data-integrity and SQL-generation tests |

## Data integrity

- ESPN endpoints are **public-facing but undocumented/unofficial**. They may change, fail or restrict automated access. ESPN is not affiliated with this project. This is a personal learning project, not a licensed ESPN data feed.
- Team and player game stats are stored by stable ESPN game/player IDs with UPSERT logic, so repeated jobs don't create duplicate games.
- Team efficiency is calculated using estimated possessions `FGA - offensive rebounds + turnovers + 0.475 × FTA`, averaged across the two teams in each game. Efficiency is unadjusted for opponents or venue and cannot be described as KenPom rankings.
- `sync_runs` records success, partial failure, and source errors. A nightly update may be incomplete even if some games appear.
- Cloud tables have **Row-Level Security enabled with no anonymous policies**, so Supabase's public API cannot expose them without a trusted server connection. The Streamlit app uses server-side secrets and parameterized read-only SQL queries, but the database account in this starter is still powerful: use a least-privilege read-only database login for production.
- RotoWire injuries are **not connected**. Don't interpret an empty injury table as an absence of injuries. An authorized, paid/contracted feed may be needed later.

## Development & local validation (optional)

Python 3.11+:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

Local ESPN staging import (not needed for hosted website):

```bash
python main.py sync --days-back 2 --days-forward 3
```

Run the hosted dashboard locally (requires a real `DATABASE_URL` environmental variable):

```bash
streamlit run cloud_dashboard.py
```

## Docs

- Streamlit hosting/deployment: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy
- Supabase connection strings: https://supabase.com/docs/guides/database/connecting-to-postgres
- GitHub Actions cron with timezone: https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-syntax#onschedule
- GitHub scheduled workflow inactivity: https://docs.github.com/en/actions/how-tos/manage-workflow-runs/disable-and-enable-workflows
- ESPN endpoint community reference (unofficial): https://github.com/sportsdataverse/sportsdataverse-js/blob/main/src/services/mbb.service.ts
