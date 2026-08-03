# -*- coding: utf-8 -*-
import csv
import json
import argparse
import os
import random

def process_and_write(data_subset, output_csv_file, system_prompt, direct_identifiers):
    """Helper function to process a subset of data and write it to a CSV."""
    with open(output_csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["system_prompt", "user_input", "assistant_output"])

        for item in data_subset:
            user_input = item["data"]["text"]

            annotations = item.get("annotations", [])
            assistant_output_list = []

            if annotations and "result" in annotations[0]:
                for result in annotations[0]["result"]:
                    val = result.get("value", {})
                    exact_text = val.get("text", "").replace("\n", " ").strip()
                    labels = val.get("labels", [])

                    # Extract generalization from meta if present
                    meta = result.get("meta", {})
                    meta_texts = meta.get("text", [])
                    generalization = None
                    for m in meta_texts:
                        if m.startswith("generalized:"):
                            generalization = m.replace("generalized:", "").strip()

                    if exact_text and labels:
                        label = labels[0]
                        inferred_risk = "high" if label in direct_identifiers else "medium"

                        entity_entry = {
                            "text": exact_text,
                            "label": label,
                            "risk": inferred_risk,
                        }

                        if generalization:
                            entity_entry["generalized"] = generalization

                        assistant_output_list.append(entity_entry)

            assistant_output = json.dumps(assistant_output_list, ensure_ascii=False)
            writer.writerow([system_prompt, user_input, assistant_output])

    print(f"Successfully processed {len(data_subset)} records and saved to {output_csv_file}!")

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Convert Label Studio JSON to training/validation CSVs.")
    parser.add_argument("-i", "--input", required=True, help="Path to the input JSON file")
    parser.add_argument("-o", "--output", required=True, help="Base path to the output CSV file")
    parser.add_argument("-s", "--split", type=float, default=0.8, help="Train split ratio (default: 0.8 for 80% train / 20% val)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible shuffling (default: 42)")
    args = parser.parse_args()

    input_json_file = args.input
    output_base_name = args.output

    # 1. Load the Label Studio JSON data
    with open(input_json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 2. Construct the static system prompt using your Swedish prompt
    system_prompt = (
        "Du är ett system för att identifiera ALL personlig och känslig"
        " information i svenska journalanteckningar, inklusive direkta"
        " identifierare och quasi-identifierare.\n\nDirekta identifierare (hög"
        " risk — alltid maskera):\n- private_person:  namn, titel\n-"
        " private_email:   e-postadresser\n- private_phone:   telefonnummer\n-"
        " account_number:  personnummer, passnummer, körkort, kontonummer\n-"
        " private_address: gatuadress, postnummer, stad\n- private_date:    "
        "födelsedatum, specifika vårddatum\n- secret:          lösenord,"
        " PIN-koder\n\nQuasi-identifierare (kontextberoende risk) — gäller även"
        " ovanlig eller\nspecifik information om anhöriga (make/maka, förälder,"
        " barn), inte bara\npatienten själv, eftersom det indirekt kan identifiera"
        " patienten.\n\nCentral fråga för VARJE uppgift du överväger, oavsett"
        " kategori: hur många\nandra personer i en svensk kommun eller på detta"
        " sjukhus skulle troligen ha\nexakt samma egenskap? Väldigt få (ett ovanligt"
        " eller framstående yrke — hos\npatienten ELLER en anhörig — en sällsynt"
        " diagnos, en ovanlig kombination av\nfakta) betyder att det är en"
        " quasi-identifierare, även om den inte liknar\nnågot exempel nedan. De"
        " flesta patienter med liknande vårdbehov (normala\nvitalparametrar, vanliga"
        " sjukdomar som högt blodtryck eller depression,\nvanliga mediciner, normala"
        " eller negativa undersökningsfynd) betyder att\ndet INTE är en"
        " quasi-identifierare, hur specifikt eller tekniskt det än\nlåter — ett"
        " exakt tal är inte i sig identifierande bara för att det är ett\ntal.\n\n-"
        " demographics:    ålder, kön, etnicitet, yrke, familjesituation\n-"
        " medical:         sällsynta diagnoser, ovanliga ingrepp\n- temporal:        "
        " exakta tidpunkter, vårdlängd\n- social:          arbetsgivare,"
        " boendesituation, religiös tillhörighet, yrke eller titel hos"
        " anhöriga\n- medication:      läkemedelsnamn — flagga för granskningsloggen;"
        " vanliga läkemedelsnamn är sällan\n"
        "                   identifierande och är ofta viktiga att bevara"
        " oförändrade\n                   för analys- eller"
        " forskningssyften\n\nReturnera ENBART giltig JSON — inga förklaringar,"
        " inga markdown-block.\nVarje entitet ska ha: text, label, risk"
        " (low/medium/high)."
    )

    # Lists to help auto-format the missing 'risk' fields in your training data
    direct_identifiers = [
        "private_person",
        "private_email",
        "private_phone",
        "account_number",
        "private_address",
        "private_date",
        "secret",
    ]

    # 3. Shuffle and split the data
    random.seed(args.seed)
    random.shuffle(data)
    
    split_index = int(len(data) * args.split)
    train_data = data[:split_index]
    val_data = data[split_index:]

    # 4. Generate filenames for train and validation
    base_name, ext = os.path.splitext(output_base_name)
    if not ext:
        ext = ".csv"
        
    train_csv_file = f"{base_name}_train{ext}"
    val_csv_file = f"{base_name}_validation{ext}"

    # 5. Process and write both files
    print(f"Splitting {len(data)} total records into Train ({len(train_data)}) and Validation ({len(val_data)})...")
    process_and_write(train_data, train_csv_file, system_prompt, direct_identifiers)
    process_and_write(val_data, val_csv_file, system_prompt, direct_identifiers)

if __name__ == "__main__":
    main()
