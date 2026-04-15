# import json
# from collections import Counter, defaultdict

# # =========================
# # CONFIG
# # =========================
# INPUT_FILE = "Datasets\\test2.json"
# OUTPUT_CLEAN = "Datasets\\test2_clean.json"
# OUTPUT_FINAL = "Datasets\\test2_final.json"
# TARGET_PER_TOPIC = 20  # change if needed

# # =========================
# # STEP 1: LOAD DATA
# # =========================
# with open(INPUT_FILE, "r", encoding="utf-8") as f:
#     data = json.load(f)

# print(f"✅ Loaded {len(data)} samples")


# # =========================
# # STEP 2: CLEAN UNICODE
# # =========================
# # (Handled automatically when saving with ensure_ascii=False)


# # =========================
# # STEP 3: COUNT TOPICS
# # =========================
# topic_counts = Counter(sample["topic"] for sample in data)

# print("\n📊 Topic Distribution:")
# for topic, count in topic_counts.items():
#     print(f"{topic}: {count}")


# # =========================
# # STEP 4: FIND IMBALANCE
# # =========================
# deficit = {}
# excess = {}

# for topic, count in topic_counts.items():
#     if count < TARGET_PER_TOPIC:
#         deficit[topic] = TARGET_PER_TOPIC - count
#     elif count > TARGET_PER_TOPIC:
#         excess[topic] = count - TARGET_PER_TOPIC

# print("\n📉 Topics needing MORE samples:")
# for topic, val in deficit.items():
#     print(f"{topic}: +{val}")

# print("\n📈 Topics with EXTRA samples:")
# for topic, val in excess.items():
#     print(f"{topic}: -{val}")


# # =========================
# # STEP 5: CHECK DUPLICATE IDS
# # =========================
# ids = set()
# duplicates = []

# for sample in data:
#     if sample["id"] in ids:
#         duplicates.append(sample["id"])
#     ids.add(sample["id"])

# if duplicates:
#     print("\n⚠️ Duplicate IDs found:", duplicates)
# else:
#     print("\n✅ No duplicate IDs")


# # =========================
# # STEP 6: SAVE CLEAN DATASET
# # =========================
# with open(OUTPUT_CLEAN, "w", encoding="utf-8") as f:
#     json.dump(data, f, ensure_ascii=False, indent=2)

# print(f"\n✅ Clean dataset saved as {OUTPUT_CLEAN}")


# # =========================
# # STEP 7: PREPARE PROMPTS FOR MISSING DATA
# # =========================
# print("\n🧠 GENERATION PROMPTS FOR GPT/CLAUDE:\n")

# for topic, missing in deficit.items():
#     print("--------------------------------------------------")
#     print(f"Topic: {topic}")
#     print(f"Generate: {missing} samples\n")
#     print(f"""Generate {missing} dataset samples for topic: {topic}

# Requirements:
# - Follow exact JSON structure:
#   id, topic, attack_type, injection_turn, detection_turn, success, turns[]
# - Include mix of attack and benign samples
# - Maintain realistic multi-turn conversations
# - Vary attack types (direct, indirect, multi_turn)
# - Keep consistent with existing dataset style

# Return ONLY JSON array.
# """)
#     print("--------------------------------------------------\n")


# # =========================
# # STEP 8: MERGE (RUN AFTER YOU GENERATE NEW DATA)
# # =========================
# def merge_new_data(existing_file, new_file, output_file):
#     with open(existing_file, "r", encoding="utf-8") as f:
#         existing = json.load(f)

#     with open(new_file, "r", encoding="utf-8") as f:
#         new = json.load(f)

#     # Check duplicate IDs before merging
#     existing_ids = set(sample["id"] for sample in existing)

#     for sample in new:
#         if sample["id"] in existing_ids:
#             print(f"⚠️ Duplicate ID skipped: {sample['id']}")
#         else:
#             existing.append(sample)

#     with open(output_file, "w", encoding="utf-8") as f:
#         json.dump(existing, f, ensure_ascii=False, indent=2)

#     print(f"\n✅ Final dataset saved as {output_file}")


# # =========================
# # HOW TO USE MERGE:
# # =========================
# # After generating new samples:
# merge_new_data(
#     "Datasets/test2_clean.json",
#     "Datasets/new_samples.json",
#     "Datasets/test2_final.json"
# )


import json
from collections import defaultdict

FILE = "Datasets/test2_final.json"  # change if needed

with open(FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

# structure: {topic: {"attack": x, "benign": y}}
stats = defaultdict(lambda: {"attack": 0, "benign": 0})

# count
for sample in data:
    topic = sample["topic"]

    # check if sample is attack or benign
    # (based on your dataset: attack_type = none or not)
    if sample.get("attack_type") == "none":
        stats[topic]["benign"] += 1
    else:
        stats[topic]["attack"] += 1

# total per topic
print("\n📊 FINAL TOPIC DISTRIBUTION TABLE\n")
print(f"{'Topic':<25}{'Total':<10}{'Attacks':<10}{'Benign':<10}")

for topic, counts in stats.items():
    total = counts["attack"] + counts["benign"]
    print(f"{topic:<25}{total:<10}{counts['attack']:<10}{counts['benign']:<10}")

# grand totals
total_all = sum(v["attack"] + v["benign"] for v in stats.values())
total_attack = sum(v["attack"] for v in stats.values())
total_benign = sum(v["benign"] for v in stats.values())

print("\n========================")
print(f"TOTAL SAMPLES: {total_all}")
print(f"TOTAL ATTACK: {total_attack}")
print(f"TOTAL BENIGN: {total_benign}")
print("========================\n")