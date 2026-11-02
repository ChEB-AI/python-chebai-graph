import json

import pandas as pd

file1 = "/home/aditya/python-chebai/data/chebi_v252/ChEBI50/processed/gatv2_baseline_classes_test.json"
file2 = "/home/aditya/python-chebai/data/chebi_v252/ChEBI50/processed/gatv2_aug_classes_test.json"

# Output CSV file
output_file = "combined_f1_scores.csv"

# Load JSON data
with open(file1, "r") as f:
    data1 = json.load(f)

with open(file2, "r") as f:
    data2 = json.load(f)

# Create a combined set of labels from both files
labels = sorted(set(data1.keys()) | set(data2.keys()))

# Create output rows
rows = []
for label in labels:
    rows.append(
        {
            "label": label,
            file1: data1.get(label, {}).get("f1", None),
            file2: data2.get(label, {}).get("f1", None),
        }
    )

# Create DataFrame and save to CSV
df = pd.DataFrame(rows)
df.to_csv(output_file, index=False)

print(f"CSV file created: {output_file}")
print(df)
