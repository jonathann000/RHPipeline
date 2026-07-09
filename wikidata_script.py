import time
import pandas as pd
import requests

# Define Wikidata endpoint and a descriptive User-Agent (required by Wikidata)
WIKIDATA_URL = "https://query.wikidata.org/sparql"
HEADERS = {
    "User-Agent": "SwedishMedicalDeidBot/1.0 (contact: your-email@domain.com) Python/requests"
}

# Mapping of Wikidata Q-IDs to English text for clean saving
CATEGORIES = {
    "Q56061": "Administrative_Entity",
    "Q34442": "Street",
    "Q16917": "Hospital",
    "Q3914": "School",
}


def fetch_category_data(class_id):
    """Fetches all entities for a specific class in Sweden using pagination."""
    all_results = []
    limit = 5000
    offset = 0

    print(f"Starting download for class {class_id}...")

    while True:
        # Optimized SPARQL template using pagination (LIMIT/OFFSET)
        sparql_query = f"""
        SELECT ?entity ?entityLabel ?entityLabelSv WHERE {{
          {{
            SELECT DISTINCT ?entity WHERE {{
              ?entity wdt:P17 wd:Q34 .
              ?entity wdt:P31/wdt:P279* wd:{class_id} .
            }}
            LIMIT {limit} OFFSET {offset}
          }}
          OPTIONAL {{ 
            ?entity rdfs:label ?entityLabel . 
            FILTER(LANG(?entityLabel) = "en") 
          }}
          OPTIONAL {{ 
            ?entity rdfs:label ?entityLabelSv . 
            FILTER(LANG(?entityLabelSv) = "sv") 
          }}
        }}
        """

        try:
            response = requests.get(
                WIKIDATA_URL,
                params={"query": sparql_query, "format": "json"},
                headers=HEADERS,
                timeout=60,
            )
            response.raise_for_status()
            data = response.json()
            results = data["results"]["bindings"]

            if not results:
                break

            all_results.extend(results)
            print(f"  Fetched {len(results)} rows (Total: {len(all_results)})")

            if len(results) < limit:
                break  # Reached the end of available data

            offset += limit
            time.sleep(1)  # Respectful delay between hits to avoid rate-limiting

        except Exception as e:
            print(f"  Error fetching batch at offset {offset}: {e}")
            break

    return all_results


def main():
    all_records = []
    for qid, cat_name in CATEGORIES.items():
        raw_data = fetch_category_data(qid)

        for row in raw_data:
            all_records.append(
                {
                    "wikidata_id": row["entity"]["value"].split("/")[-1],
                    "label_en": row.get("entityLabel", {}).get("value", ""),
                    "label_sv": row.get("entityLabelSv", {}).get("value", ""),
                    "category": cat_name,
                }
            )

    df = pd.DataFrame(all_records)
    df.to_csv("sweden_entities_deid.csv", index=False, encoding="utf-8")
    print(f"\nDone! Saved {len(df)} entities to sweden_entities_deid.csv")


if __name__ == "__main__":
    main()
