PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS worlds (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    tags_json TEXT,
    published_at TEXT
);

CREATE TABLE IF NOT EXISTS collection_runs (
    id INTEGER PRIMARY KEY,
    started_at TEXT NOT NULL,
    finished_at TEXT
);

CREATE TABLE IF NOT EXISTS collection_queries (
    id INTEGER PRIMARY KEY,
    run_id INTEGER NOT NULL REFERENCES collection_runs(id),
    source TEXT NOT NULL,
    target TEXT,
    finished_at TEXT NOT NULL,
    status TEXT NOT NULL,
    error TEXT
);

CREATE TABLE IF NOT EXISTS observations (
    query_id INTEGER NOT NULL REFERENCES collection_queries(id),
    world_id TEXT NOT NULL REFERENCES worlds(id),
    rank INTEGER,
    tags_json TEXT,
    occupants INTEGER,
    public_occupants INTEGER,
    visits INTEGER,
    favorites INTEGER,
    PRIMARY KEY (query_id, world_id)
);

CREATE INDEX IF NOT EXISTS observations_world ON observations(world_id, query_id);

CREATE VIEW IF NOT EXISTS world_samples AS
WITH ranked_samples AS (
    SELECT q.run_id, o.world_id, q.finished_at AS observed_at, q.source, q.target,
           o.occupants, o.public_occupants, o.visits, o.favorites,
           row_number() OVER (
               PARTITION BY q.run_id, o.world_id
               ORDER BY o.occupants IS NULL,
                   CASE q.source WHEN 'active' THEN 0 WHEN 'published' THEN 1
                       WHEN 'tag_search' THEN 2 ELSE 3 END,
                   q.id
           ) AS preference
    FROM observations o JOIN collection_queries q ON q.id = o.query_id
)
SELECT run_id, world_id, observed_at, source, target,
       occupants, public_occupants, visits, favorites
FROM ranked_samples WHERE preference = 1;

CREATE TABLE IF NOT EXISTS watchlist (
    world_id TEXT PRIMARY KEY,
    reason TEXT NOT NULL,
    added_at TEXT NOT NULL,
    review_on TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS hosted_batches (
    batch_id TEXT PRIMARY KEY,
    sequence INTEGER NOT NULL,
    run_id INTEGER NOT NULL UNIQUE REFERENCES collection_runs(id),
    imported_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS world_metadata_history (
    world_id TEXT NOT NULL REFERENCES worlds(id),
    observed_at TEXT NOT NULL,
    updated_at TEXT,
    platforms_json TEXT,
    capacity INTEGER,
    PRIMARY KEY (world_id, observed_at)
);

CREATE TABLE IF NOT EXISTS investigation_flags (
    world_id TEXT PRIMARY KEY REFERENCES worlds(id),
    reason TEXT NOT NULL,
    origin TEXT NOT NULL CHECK (origin IN ('manual', 'automatic')),
    priority REAL NOT NULL,
    flagged_at TEXT NOT NULL,
    dismissed INTEGER NOT NULL DEFAULT 0,
    retry_after TEXT
);

CREATE TABLE IF NOT EXISTS investigation_details (
    id INTEGER PRIMARY KEY,
    world_id TEXT NOT NULL REFERENCES investigation_flags(world_id),
    version_at TEXT NOT NULL,
    captured_at TEXT NOT NULL,
    details_json TEXT NOT NULL,
    changed_fields_json TEXT NOT NULL,
    classification TEXT NOT NULL CHECK (classification IN ('baseline', 'unclassified', 'minor', 'major')),
    review_reason TEXT,
    UNIQUE (world_id, version_at)
);

CREATE TABLE IF NOT EXISTS investigation_reasons (
    world_id TEXT NOT NULL REFERENCES investigation_flags(world_id),
    kind TEXT NOT NULL,
    summary TEXT NOT NULL,
    evidence_json TEXT NOT NULL,
    first_detected_at TEXT NOT NULL,
    observed_at TEXT NOT NULL,
    PRIMARY KEY (world_id, kind)
);

CREATE TABLE IF NOT EXISTS investigation_visits (
    id INTEGER PRIMARY KEY,
    world_id TEXT NOT NULL REFERENCES investigation_flags(world_id),
    visited_at TEXT NOT NULL,
    notes TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS investigation_visits_world ON investigation_visits(world_id, visited_at);

CREATE TABLE IF NOT EXISTS world_launch_times (
    world_id TEXT NOT NULL REFERENCES worlds(id),
    milestone TEXT NOT NULL CHECK (milestone IN ('labs','public')),
    launched_at TEXT NOT NULL,
    first_observed_at TEXT NOT NULL,
    PRIMARY KEY (world_id, milestone, launched_at)
);

CREATE TABLE IF NOT EXISTS launch_sample_context (
    query_id INTEGER NOT NULL,
    world_id TEXT NOT NULL,
    creator_id TEXT,
    featured INTEGER,
    platforms_json TEXT,
    PRIMARY KEY (query_id, world_id),
    FOREIGN KEY (query_id, world_id) REFERENCES observations(query_id, world_id)
);

CREATE TABLE IF NOT EXISTS launch_study_enrollment (
    world_id TEXT NOT NULL,
    milestone TEXT NOT NULL CHECK (milestone IN ('labs','public')),
    launched_at TEXT NOT NULL,
    enrolled_at TEXT NOT NULL,
    PRIMARY KEY (world_id, milestone, launched_at)
);

CREATE TABLE IF NOT EXISTS world_context_events (
    id INTEGER PRIMARY KEY,
    world_id TEXT NOT NULL REFERENCES worlds(id),
    event_at TEXT NOT NULL,
    kind TEXT NOT NULL,
    notes TEXT NOT NULL
);

CREATE VIEW IF NOT EXISTS launch_study_samples AS
SELECT e.world_id,e.milestone,e.launched_at,e.enrolled_at,s.run_id,s.observed_at,
       (julianday(s.observed_at)-julianday(e.launched_at))*24 AS hours_since_launch,
       s.source,s.occupants,s.visits,s.favorites
FROM launch_study_enrollment e JOIN world_samples s USING(world_id)
WHERE julianday(s.observed_at) BETWEEN julianday(e.launched_at) AND julianday(e.launched_at)+50.0/24;

CREATE VIEW IF NOT EXISTS world_research_observations AS
SELECT q.run_id,o.query_id,o.world_id,w.name,q.finished_at AS observed_at,q.source,
       CASE WHEN json_valid(q.target) THEN coalesce(json_extract(q.target,'$.pool'),q.source)
            ELSE q.source END AS pool,
       o.rank,o.tags_json,o.occupants,o.visits,o.favorites,
       c.creator_id,c.featured,c.platforms_json
FROM observations o JOIN collection_queries q ON q.id=o.query_id
JOIN worlds w ON w.id=o.world_id
LEFT JOIN launch_sample_context c USING(query_id,world_id);

CREATE VIEW IF NOT EXISTS launch_study_outcomes AS
SELECT e.world_id,e.milestone,e.launched_at,e.enrolled_at,
       count(s.occupants) AS measured_samples,
       min(CASE WHEN s.occupants IS NOT NULL THEN s.hours_since_launch END) AS first_measured_hours,
       min(CASE WHEN s.occupants >= 20 THEN s.hours_since_launch END) AS first_observed_20_hours,
       max(CASE WHEN s.hours_since_launch <= 24 THEN s.occupants END) AS sampled_day_one_peak,
       avg(CASE WHEN s.hours_since_launch > 24 AND s.hours_since_launch <= 48 THEN s.occupants END) AS sampled_day_two_mean,
       count(CASE WHEN s.hours_since_launch > 24 AND s.hours_since_launch <= 48 THEN s.occupants END) AS day_two_samples,
       max(CASE WHEN s.occupants IS NOT NULL THEN s.hours_since_launch END) AS last_measured_hours
FROM launch_study_enrollment e LEFT JOIN launch_study_samples s USING(world_id,milestone,launched_at,enrolled_at)
GROUP BY e.world_id,e.milestone,e.launched_at,e.enrolled_at;

CREATE VIEW IF NOT EXISTS launch_longitudinal_samples AS
SELECT e.world_id,e.milestone,e.launched_at,s.run_id,s.observed_at,s.source,s.occupants,s.visits,s.favorites,
       (julianday(s.observed_at)-julianday(e.launched_at))*24 AS hours_since_launch
FROM launch_study_enrollment e JOIN world_samples s USING(world_id)
WHERE julianday(s.observed_at) BETWEEN julianday(e.launched_at) AND julianday(e.launched_at)+28+4.0/24;
