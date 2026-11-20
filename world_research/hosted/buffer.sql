CREATE TABLE IF NOT EXISTS world_batches (
    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_id TEXT NOT NULL UNIQUE,
    payload TEXT NOT NULL,
    acknowledged_at INTEGER
);

CREATE INDEX IF NOT EXISTS world_batches_pending
    ON world_batches(sequence) WHERE acknowledged_at IS NULL;
CREATE INDEX IF NOT EXISTS world_batches_acknowledged
    ON world_batches(acknowledged_at) WHERE acknowledged_at IS NOT NULL;
