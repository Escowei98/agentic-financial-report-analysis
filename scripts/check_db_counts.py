"""Zeigt die Anzahl der Einträge in allen Tabellen der ablation_study.db."""
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "ablation_study.db"

def main():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Alle Tabellen auflisten
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
    tables = [row[0] for row in cursor.fetchall()]

    if not tables:
        print("Keine Tabellen in der Datenbank gefunden.")
        conn.close()
        return

    print(f"Datenbank: {DB_PATH}\n")
    print(f"{'Tabelle':<40} {'Einträge':>10}")
    print("-" * 52)

    total = 0
    for table in tables:
        cursor.execute(f'SELECT COUNT(*) FROM "{table}";')
        count = cursor.fetchone()[0]
        total += count
        print(f"{table:<40} {count:>10}")

    print("-" * 52)
    print(f"{'GESAMT':<40} {total:>10}")

    conn.close()

if __name__ == "__main__":
    main()
