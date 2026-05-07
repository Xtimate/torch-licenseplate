import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from api.database import get_conn


def migrate():
    conn = get_conn()
    existing = {row[1] for row in conn.execute("PRAGMA table_info(plates)").fetchall()}

    added = []

    if "lat" not in existing:
        conn.execute("ALTER TABLE plates ADD COLUMN lat REAL")
        added.append("lat")

    if "lng" not in existing:
        conn.execute("ALTER TABLE plates ADD COLUMN lng REAL")
        added.append("lng")

    if "location_name" not in existing:
        conn.execute("ALTER TABLE plates ADD COLUMN location_name TEXT")
        added.append("location_name")

    conn.commit()
    conn.close()

    if added:
        print(f"Migration complete. Added columns: {', '.join(added)}")

    else:
        print("Nothing to migrate, columns already exist.")


if __name__ == "__main__":
    print("Running migration...")
    migrate()
