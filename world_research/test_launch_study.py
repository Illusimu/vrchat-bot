import sqlite3
import unittest
from contextlib import closing
from copy import deepcopy
from datetime import UTC, datetime, timedelta

from collector import SCHEMA
from hosted_sync import SWEEP_PAGES, import_batch


LAUNCH = datetime(2026, 11, 17, 12, tzinfo=UTC)
WORLD = "wrld_aabbccdd-1111-2222-3333-000000000000"


def batch(batch_id, observed, occupants, study=True):
    capture = observed.isoformat()
    world = {"id": WORLD, "name": "A launch", "releaseStatus": "public", "occupants": occupants,
             "visits": 7 + occupants, "favorites": 2, "tags": ["author_tag_game"],
             "labsPublicationDate": LAUNCH.isoformat(), "featured": False, "platforms": ["android"]}
    all_pages = SWEEP_PAGES | ({("labs", 1)} if study else set())
    result = {"version": 1, "batchId": batch_id, "startedAt": capture, "finishedAt": capture,
              "results": [{"status": "skipped", "pool": pool, "page": page, "reason": "Trial"}
                          for pool, page in sorted(all_pages)]}
    if study:
        result["launchStudy"] = {
            "enrollment": [{"worldId": WORLD, "milestone": "labs", "launchedAt": LAUNCH.isoformat(),
                            "enrolledAt": (LAUNCH + timedelta(hours=1)).isoformat()}],
            "followUps": [{"worldId": WORLD, "status": "ok", "observedAt": capture, "world": world}],
        }
    return result


class LaunchStudyTest(unittest.TestCase):
    def test_zero_activity_and_followup_attach_to_same_launch(self):
        with closing(sqlite3.connect(":memory:")) as db:
            db.executescript(SCHEMA.read_text(encoding="utf-8"))
            early = batch("study-1", LAUNCH + timedelta(hours=1), 0)
            day_two = batch("study-2", LAUNCH + timedelta(hours=36), 23)
            self.assertTrue(import_batch(db, 1, deepcopy(early)))
            self.assertTrue(import_batch(db, 2, deepcopy(day_two)))
            self.assertFalse(import_batch(db, 2, deepcopy(day_two)))
            counts = db.execute("SELECT occupants FROM launch_study_samples ORDER BY observed_at").fetchall()
            self.assertEqual(counts, [(0,), (23,)])
            result = db.execute("""SELECT measured_samples,round(first_measured_hours),round(first_observed_20_hours),
                                   sampled_day_one_peak,sampled_day_two_mean,day_two_samples
                                   FROM launch_study_outcomes""").fetchone()
            self.assertEqual(result, (2, 1, 36, 0, 23, 1))
            self.assertEqual(db.execute("SELECT DISTINCT featured FROM world_research_observations").fetchone()[0], 0)
            self.assertEqual(db.execute("SELECT count(*) FROM world_launch_times").fetchone()[0], 1)
            self.assertTrue(import_batch(db, 3, batch("legacy", LAUNCH, 0, study=False)))
            self.assertTrue(import_batch(db, 4, batch("week-one", LAUNCH + timedelta(days=7, hours=1), 0)))
            self.assertEqual(db.execute("SELECT count(*) FROM launch_study_samples").fetchone()[0], 2)
            self.assertEqual(db.execute("""SELECT occupants FROM launch_longitudinal_samples
                                           WHERE hours_since_launch BETWEEN 168 AND 172""").fetchone()[0], 0)

    def test_wrong_followup_id_rolls_back_batch(self):
        with closing(sqlite3.connect(":memory:")) as db:
            db.executescript(SCHEMA.read_text(encoding="utf-8"))
            payload = batch("invalid-batch", LAUNCH + timedelta(hours=1), 0)
            payload["launchStudy"]["followUps"][0]["world"]["id"] = "wrld_aabbccdd-1111-2222-3333-000000000001"
            with self.assertRaises(ValueError):
                import_batch(db, 1, payload)
            self.assertEqual(db.execute("SELECT count(*) FROM collection_runs").fetchone()[0], 0)
            self.assertEqual(db.execute("SELECT count(*) FROM launch_study_enrollment").fetchone()[0], 0)
