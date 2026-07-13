import time
import pandas as pd
import requests

# Define Wikidata endpoint and a descriptive User-Agent (required by Wikidata)
WIKIDATA_URL = "https://query.wikidata.org/sparql"
HEADERS = {
    "User-Agent": "SwedishMedicalDeidBot/1.0 (contact: your-email@domain.com) Python/requests"
}

# Places: filtered by "located in Sweden" (P17) — straightforward, since a
# hospital/street/school is physically located somewhere.
PLACE_CATEGORIES = {
    "Q56061": "Administrative_Entity",
    "Q34442": "Street",
    "Q16917": "Hospital",
    "Q3914": "School",
}
PLACE_FILTER = "?entity wdt:P17 wd:Q34 ."

# Names: a name isn't "located in" a country, so filtering by P17 doesn't
# apply. Also, "has a Swedish-language label" turned out NOT to be a useful
# proxy — Wikidata auto-fills sv labels on names regardless of actual usage
# (tested empirically: returned entries like "Nebahat", "Femmina", clearly
# not Swedish). What actually works: P407 "language of work or name" = Q9027
# (Swedish) — tags the name itself as linguistically Swedish, verified to
# return genuinely Nordic names (Börje, Karolina, Anders Gustaf, ...).
NAME_CATEGORIES = {
    "Q202444": "Given_Name",
    "Q101352": "Family_Name",
}
NAME_FILTER = "?entity wdt:P407 wd:Q9027 ."


def fetch_entities(class_id, filter_clause, limit=5000):
    """
    Fetches all entities for a specific Wikidata class using pagination.
    filter_clause: SPARQL condition(s) narrowing results to ones relevant
    for Swedish text redaction (see PLACE_FILTER / NAME_FILTER above).
    """
    all_results = []
    offset = 0
    retries_left = 2

    print(f"Starting download for class {class_id}...")

    while True:
        sparql_query = f"""
        SELECT ?entity ?entityLabel ?entityLabelSv WHERE {{
          {{
            SELECT DISTINCT ?entity WHERE {{
              ?entity wdt:P31/wdt:P279* wd:{class_id} .
              {filter_clause}
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
            retries_left = 2  # reset after a successful batch

            if len(results) < limit:
                break  # Reached the end of available data

            offset += limit
            time.sleep(1)  # Respectful delay between hits to avoid rate-limiting

        except requests.exceptions.Timeout:
            if retries_left > 0:
                retries_left -= 1
                print(f"  Timeout at offset {offset}, retrying ({retries_left} left)...")
                continue
            print(f"  Timeout at offset {offset}, giving up after retries — data may be incomplete")
            break

        except Exception as e:
            print(f"  Error fetching batch at offset {offset}: {e}")
            break

    return all_results


def main():
    all_records = []
    for qid, cat_name, filter_clause in [
        (qid, cat_name, PLACE_FILTER) for qid, cat_name in PLACE_CATEGORIES.items()
    ] + [
        (qid, cat_name, NAME_FILTER) for qid, cat_name in NAME_CATEGORIES.items()
    ]:
        raw_data = fetch_entities(qid, filter_clause)

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
