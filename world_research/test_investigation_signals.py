import sqlite3
import unittest
from contextlib import closing
from datetime import UTC, datetime, timedelta

from collector import SCHEMA
from investigation_signals import Sample, detect_reasons
from investigations import add_candidates, queue, time_text


class InvestigationSignalsTest(unittest.TestCase):
    now = datetime(2026, 11, 17, 16, tzinfo=UTC)

    def reasons(self, world, samples, published):
        return detect_reasons({world: sorted(samples, key=lambda s: s.at)}, {world: published}, self.now).get(world, [])

    def test_release_uses_publication_age_and_peer_growth(self):
        recent = [self.now - timedelta(days=3), self.now - timedelta(days=1)]
        history = {world: [Sample(recent[0], 4, 100, 10), Sample(recent[1], 6, 100 + increase, 10)]
                   for world, increase in (("new", 140), ("peer-1", 18), ("peer-2", 25), ("peer-3", 30),
                                           ("old", 260), ("too-young", 300))}
        published = {world: self.now - timedelta(days=12) for world in history}
        published["old"] = self.now - timedelta(days=80)
        published["too-young"] = self.now - timedelta(days=6)
        reasons = detect_reasons(history, published, self.now)
        self.assertEqual(set(reasons), {"new"})
        self.assertEqual(reasons["new"][0].kind, "promising_release")
        self.assertEqual(reasons["new"][0].evidence["metrics"]["visits"]["peers"], 3)

    def test_recurring_central_window_survives_clock_change(self):
        peaks = [Sample(datetime(2026, 10, 30, 1, tzinfo=UTC), 21, None, None),
                 Sample(datetime(2026, 11, 6, 2, tzinfo=UTC), 24, None, None),
                 Sample(datetime(2026, 11, 13, 2, tzinfo=UTC), 22, None, None)]
        quiet = [Sample(peak.at + timedelta(hours=12, days=day), 3, None, None)
                 for peak in peaks for day in (0, 1)]
        reasons = self.reasons("recurring", peaks + quiet, None)
        self.assertEqual([reason.kind for reason in reasons], ["recurring_gathering"])
        self.assertEqual((reasons[0].evidence["weekday"], reasons[0].evidence["start_hour"]), (3, 20))
        self.assertEqual(self.reasons("recurring", peaks[:2] + quiet, None), [])

    def test_matched_growth_and_sustained_activity_coexist_in_queue(self):
        at = [self.now - timedelta(days=day, hours=hour) for day in (1, 2, 3) for hour in (1, 4)]
        history = [Sample(item, 30, None, None) for item in at]
        history += [Sample(item - timedelta(days=7), 10, None, None) for item in at]
        wid = "wrld_aabbccdd-1111-2222-3333-000000000000"
        with closing(sqlite3.connect(":memory:")) as db:
            db.executescript(SCHEMA.read_text(encoding="utf-8"))
            with db:
                db.execute("INSERT INTO worlds(id,name,published_at) VALUES (?,?,?)",
                           (wid, "Example", time_text(self.now - timedelta(days=60))))
                for sample in history:
                    observed = time_text(sample.at)
                    run = db.execute("INSERT INTO collection_runs(started_at,finished_at) VALUES (?,?)",
                                     (observed, observed)).lastrowid
                    query = db.execute("INSERT INTO collection_queries(run_id,source,finished_at,status) VALUES (?,'active',?,'ok')",
                                       (run, observed)).lastrowid
                    db.execute("INSERT INTO observations(query_id,world_id,occupants) VALUES (?,?,?)",
                               (query, wid, sample.occupants))
            self.assertEqual(add_candidates(db, self.now), 1)
            self.assertEqual(add_candidates(db, self.now), 0)
            flag = queue(db, self.now)[0]
            self.assertEqual(flag["stage"], "waiting_for_details")
            self.assertEqual({reason["kind"] for reason in flag["reasons"]}, {"growing_established", "sustained_activity"})
            with db:
                db.execute("UPDATE investigation_flags SET dismissed=1 WHERE world_id=?", (wid,))
            self.assertEqual(add_candidates(db, self.now + timedelta(days=1)), 0)
            self.assertEqual(queue(db, self.now), [])
