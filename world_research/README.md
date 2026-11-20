# VRChat world research

Use Python 3.11 or newer. From the workspace root, run `python world_research/collector.py init` to create `world-research.sqlite3`.

The collector can request one page of active worlds and one page sorted by publication date. Both default to 50 results. Pass `--tag TAG:LIMIT` for each additional tag search, with a larger limit for a broad search and a smaller one for a temporary theme. Use tag values verified against a world response. The active, published, and tag results identify the query that found a world. VRChat's in-client world rows can use additional curation rules. [VRChat Worlds wiki](https://wiki.vrchat.com/wiki/Worlds/en) · [World search reference](https://vrchat.community/reference/search-worlds)

## First live check

Set `VRCHAT_API_CONTACT` locally to the app's contact email or project URL. This value is sent to VRChat in the User-Agent. The collector makes anonymous GET requests. VRChat's community reference lists authentication for discovery, so a discovery request may return HTTP 401. Test a known world by ID first. Anonymous detail responses can mask some counts with zeros, so the collector records those zeros as missing. [VRChat Creator Guidelines](https://hello.vrchat.com/creator-guidelines#api-usage-bots) · [World details reference](https://vrchat.community/reference/get-world)

```powershell
$env:VRCHAT_API_CONTACT = '<app contact email or project URL>'
python world_research/collector.py watch <world-id> --reason 'Compare activity over time'
python world_research/collector.py scan --watch-only
python world_research/collector.py scan --limit 20
```

The watchlist defaults to a 28-day review period. Re-run `watch` to renew an entry, or pass `--review-on YYYY-MM-DD`. Each scan is manual while access and count reliability are being checked. When a request returns a rate-limit or server error, the collector ends the pass. Let the service recover before starting another pass and respect any server-provided retry interval.

## On-demand scan using the website connection

From the workspace root in PowerShell 7, run `./world_research/scan-live.ps1`. Use `-Database PATH` to import into a different SQLite file. The command pulls up to ten public worlds from the Escape Through Reality Worker and imports the results. The Worker keeps its existing VRChat session and shares its upstream rate limit with profile lookups. A scan within a minute may return the same snapshot. The importer keeps the original fetch time and skips duplicate imports.

The command uses a dedicated token from `.world-research-token`, encrypted for the current Windows user. The Worker stores the matching value as `WORLD_RESEARCH_TOKEN`. It authorizes world-research scans and buffer sync. If the Worker reports that VRChat needs to be reconnected, use the existing website connection setup to renew its session.

The owner can also download a snapshot in the website's signed-in browser at `/api/admin/world-research/active` and import the file with `python world_research/collector.py import-active <path-to-json>`.

## Discovery and paging pilot

Run `./world_research/scan-pilot.ps1`. The default pilot fetches two pages of up to 100 worlds each for the active list, games tagged `author_tag_game`, and worlds tagged `author_tag_social`. It also fetches 50 recently published worlds sorted by publication date. The seven requests take at least six minutes under the shared Worker rate limit. Each page is saved as it completes, with the original rank range and capture time. Use `-Pools social` to scan just the social tag. The optional `hangouts` pool queries both social and hangout tags together.

Run `python -X utf8 world_research/pilot_report.py <pages.ndjson> --summary` to import the pages and summarize results. Give multiple files to combine complementary pools in one sweep. Pass `--chart PATH` to create an interactive report and `--floor N` to evaluate a different activity cutoff. Each world is counted once per pool and once overall. Among known counts for the combined distribution, the active list has precedence, followed by games, social pools, and recent publications. Within a pool, the earliest returned rank with a known count wins. Counts across pages were sampled at different times. Pools may overlap and their totals should not be added together.

Optional adaptive mode `./world_research/scan-pilot.ps1 -ActivityFloor N` skips page two after a short first page or when fewer than half of the last 20 worlds meet the floor and all 20 counts are known. Omitting `-ActivityFloor` always checks both pages for calibration. The recent-publication sample has one page in both modes.

## Hosted sweeps and local history

The website Worker starts a Cloudflare Workflow every two hours when `WORLD_RESEARCH_SCHEDULE` is `enabled`. It offsets each start by 30 seconds to nine minutes and spaces upstream requests by 65 seconds. The Workflow applies the 20-player paging rule to Active, Games, and Social, audits page two once a day, and records up to 50 recently published worlds. It saves every page outcome and each successful capture time into one batch in the separate D1 buffer. The site profile integration and the sweeps share the upstream rate limit.

Run `python world_research/hosted_sync.py` to catch up manually. `register-sync-task.ps1` installs the same sync at Windows sign-in and every 30 minutes while that user is signed in. It writes a rotating log to `world-history-sync.log`. The Windows user must be able to unlock the existing encrypted token. The host continues to collect with the computer powered off.

The sync commits each batch to SQLite before acknowledging it to Cloudflare. A repeat import checks the batch ID. Use `python world_research/hosted_sync.py --replay` to recover from a restored backup using any retained batches. Acknowledged batches are removed after seven days. Pending batches stay in the buffer. The `hosted_batches`, `collection_queries`, `world_samples`, and `world_metadata_history` tables or views preserve import IDs, coverage, player and engagement history, and available world updates, platforms, and capacity changes.

## Investigation queue

`python -X utf8 world_research/investigations.py queue` shows flags, their reasons, the latest detail snapshot, exploration status, and eligibility dates as JSON. The Windows sync evaluates flags and attempts at most one due detail lookup per invocation through the shared Worker limiter. Use `--skip-investigations` with `hosted_sync.py` to import only scans.

The initial rules admit up to five automatic flags per UTC day. The pass gives each reason a chance to contribute a candidate, then fills any remaining slots using the number of qualifying reasons and average recent player count. Existing flags retain their reasons and the timestamps of the observations behind them. These are starting thresholds to review after more history accumulates.

- **Sustained activity** requires at least three samples with 20 or more players, at least half of measured samples at or above that floor over the last seven days, and at least 20 hours between the first and last measured samples.
- **Promising new release** covers worlds public for 7–29 days. Growth in active players, visits, or favorites must clear a minimum and beat the upper quartile of at least three sampled peers published within seven days of that world. Visit and favorite growth use observed changes per day.
- **Recurring gathering** looks for an above-usual crowd in the same two-hour Central time window across at least three distinct weeks in the last six weeks. The queue lists the window and the peak observations behind it.
- **Growing established world** compares the last seven days with the prior two weeks for worlds public at least 30 days. The median in matching weekday and time slots must rise by at least eight players and 50%, with enough observations in each period.

- `investigations.py flag WORLD_ID "Reason"` flags a known world manually. `dismiss WORLD_ID` removes it from the active queue and prevents automatic re-enrollment.
- `details WORLD_ID` returns saved descriptions, changed fields, and visit notes. The first detail fetch waits until seven days after publication. Further fetches require a newer observed update time. A missing publication date holds the check until we know the date.
- `visit WORLD_ID "Observations"` records an actual exploration. Use `--at` with a time zone to record a past visit.
- `review SNAPSHOT_ID major "Evidence"` confirms a major change. `minor` closes the review without re-queuing. A confirmed update after the last visit is eligible for a revisit once 30 days have elapsed since that visit.

## Data for comparing worlds and launches

The hosted sweep saves a Community Labs discovery page in addition to the public-release and activity pages. It preserves both reported launch dates, observed tags and discovery rank, creator ID, platform support and the featured flag when returned. Every sweep can enroll one fresh Labs or public launch from the release feeds, using launch time and rotating milestone preference. Enrollment does not use player counts. Up to 26 recent enrollments can be active at once. Each follows the same world for about two days, with a direct lookup on sweeps where the world has no usable player count in discovery. The study allows measurements through hour 50 so that a slightly delayed two-hour sweep can still provide an end-of-window reading.

- `world_research_observations` joins the original observations with pool, rank, tags, creator, platform and featured context. Use the same sampled pool and time windows when comparing genres. Worlds can have several tags or appear in multiple pools.
- `launch_study_samples` and `launch_study_outcomes` report launch age, sample coverage, first observed crossing of 20 players, first-day observed peak and second-day activity. Original visits and favorites are available at each sample. Missing samples are gaps. Day-two measures are conditional on having a day-two sample.
- `launch_longitudinal_samples` includes discovery observations and direct follow-ups around days 7, 14, and 28. Each follow-up window lasts four hours, and each sweep can process up to eight older enrollments in addition to the initial launch cohort. A quiet world remains eligible for follow-up regardless of its rank.
- `python world_research/research_data.py coverage` reports what has actually been recorded. To log a known announcement or launch event, run `python world_research/research_data.py event WORLD_ID 2026-11-17T20:00:00-06:00 announcement "Posted announcement"`.

## Reading the data

Run `python world_research/explorer.py` and open `http://127.0.0.1:8768/` on the collection computer. Refresh to read new imports. The explorer opens SQLite read-only and listens on the local loopback interface.

- **Launch timing** compares Community Labs and public release milestones separately. Choose a weekday, target for concurrent players, visits gained or net favorites gained, and follow-up horizon. Gains use the earliest measured total within two hours of release as their baseline. For the most recent 90 days, it shows known release times and enrolled outcomes in four-hour Central Time windows. A known release can appear without any enrolled follow-up. Each window opens its underlying enrolled launches. The median time is conditional on observing a hit.
- **Topic changes** compares two selected sweeps using worlds that had measured players and the same creator tag in both. Total player change and median change per world help distinguish a single popular world from broader movement.
- **World search** filters one sweep by name, world ID, discovery pool or creator tag. Select a world to see its counts and discovery history across the latest 12 sampled runs. The local database retains older runs.
- **Investigations** lists the current flags with evidence, reasons and visit stages.

- `worlds` stores one record per VRChat world ID.
- `collection_runs` and `collection_queries` show when a scan ran and which requests succeeded. Each observation includes the returned tags and the world's position in that query, when available.
- `world_samples` provides one sample per world per scan for trend analysis. If a world appears in several results, it prefers a sample with an occupant count and then prefers discovery results over an anonymous detail result. Use `observations` and `collection_queries` to inspect every source individually.
