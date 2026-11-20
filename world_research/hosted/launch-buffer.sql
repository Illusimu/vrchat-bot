CREATE TABLE IF NOT EXISTS launch_cohort (
    world_id TEXT NOT NULL,
    milestone TEXT NOT NULL CHECK (milestone IN ('labs', 'public')),
    launched_at INTEGER NOT NULL,
    enrolled_at INTEGER NOT NULL,
    sweep_id TEXT NOT NULL UNIQUE,
    PRIMARY KEY (world_id, milestone, launched_at)
);

CREATE INDEX IF NOT EXISTS launch_cohort_by_time ON launch_cohort(launched_at);
