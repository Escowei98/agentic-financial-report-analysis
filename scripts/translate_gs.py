import csv
import json
from pathlib import Path

from tqdm import tqdm

from src.common.llm_client import get_llm


def translate_row(row, llm):
    prompt = f"""
You are an expert financial translator. Translate the following fields from German to English.
Ensure financial terms (like Net Income, Operating Margin) remain accurate.
Convert units in the 'gt_value' and 'rationale' (e.g. 'Mrd' -> 'billion', 'Millionen' -> 'million', 'Prozentpunkte' -> 'percentage points').
If 'gt_value' is a number/unit, make sure it matches the English convention (e.g. '$93.7 Mrd' -> '$93.7 billion', '+2.0pp' stays '+2.0pp').

Input:
query: {row.get('query', '')}
gt_value: {row.get('gt_value', '')}
rationale: {row.get('rationale', '')}

Output ONLY a JSON object with the keys "query", "gt_value", "rationale":
{{
  "query": "...",
  "gt_value": "...",
  "rationale": "..."
}}
"""
    try:
        response = llm.invoke(prompt)
        text = response.content.strip()
        if text.startswith("```json"):
            text = text[7:-3]
        data = json.loads(text)

        row["query"] = data.get("query", row["query"])
        row["gt_value"] = data.get("gt_value", row["gt_value"])
        row["rationale"] = data.get("rationale", row["rationale"])
    except Exception as e:
        print(f"Error on id {row.get('id')}: {e}")

    return row

def main():
    in_csv = Path("data/gold_standard/gold_standard_v3.csv")
    out_csv = Path("data/gold_standard/gold_standard_v3_en.csv")

    with open(in_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        rows = list(reader)
        fieldnames = reader.fieldnames

    print(f"Loaded {len(rows)} rows.")

    from src.common.config import load_config
    config = load_config()
    model = config.get("llm", {}).get("model", "gemini-2.5-flash")
    llm = get_llm(model_name=model, temperature=0.0)

    translated_rows = []
    for row in tqdm(rows):
        translated_rows.append(translate_row(row, llm))

    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        writer.writerows(translated_rows)

    print(f"Saved translated CSV to {out_csv}")

if __name__ == "__main__":
    main()
