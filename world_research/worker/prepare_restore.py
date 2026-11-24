"""Convert a research D1 export into statements small enough for Wrangler."""

import argparse
from contextlib import closing
from pathlib import Path
import sqlite3


def write_restore(source, output, high_water):
    schema = Path(__file__).resolve().parent.parent / "hosted"
    with closing(sqlite3.connect(":memory:")) as db:
        db.executescript(source.read_text(encoding="utf-8-sig"))
        if db.execute("PRAGMA quick_check").fetchall() != [("ok",)]:
            raise ValueError("The exported research database failed its integrity check")
        saved = db.execute("SELECT seq FROM sqlite_sequence WHERE name = 'world_batches'").fetchone()
        sequence = max(saved[0] if saved else 0, high_water or 0)

        def literal(value):
            return db.execute("SELECT quote(?)", (value,)).fetchone()[0]

        with output.open("x", encoding="utf-8", newline="\n") as script:
            for name in ("buffer.sql", "launch-buffer.sql"):
                script.write((schema / name).read_text(encoding="utf-8") + "\n")
            for number, batch_id, payload, acknowledged in db.execute(
                "SELECT sequence, batch_id, payload, acknowledged_at FROM world_batches ORDER BY sequence"
            ):
                identity = ", ".join(map(literal, (number, batch_id, "", acknowledged)))
                script.write(f"INSERT INTO world_batches VALUES ({identity});\n")
                for start in range(0, len(payload), 12_000):
                    chunk = literal(payload[start:start + 12_000])
                    script.write(f"UPDATE world_batches SET payload = payload || {chunk} WHERE sequence = {number};\n")
            for row in db.execute(
                "SELECT world_id, milestone, launched_at, enrolled_at, sweep_id FROM launch_cohort"
            ):
                script.write(f"INSERT INTO launch_cohort VALUES ({', '.join(map(literal, row))});\n")
            script.write(
                "INSERT INTO sqlite_sequence(name, seq) "
                f"SELECT 'world_batches', {sequence} WHERE NOT EXISTS "
                "(SELECT 1 FROM sqlite_sequence WHERE name = 'world_batches');\n"
                f"UPDATE sqlite_sequence SET seq = max(seq, {sequence}) WHERE name = 'world_batches';\n"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--high-water", type=int, help="Last assigned sequence from the export metadata")
    args = parser.parse_args()
    if args.high_water is not None and args.high_water < 0:
        parser.error("--high-water cannot be negative")
    write_restore(args.source, args.output, args.high_water)
