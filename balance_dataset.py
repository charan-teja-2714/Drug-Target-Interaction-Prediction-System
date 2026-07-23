import pandas as pd

INPUT_FILE = "data/raw/bindingdb_filtered.csv"
OUTPUT_FILE = "data/raw/bindingdb_balanced_20000.csv"

SAMPLES_PER_CLASS = 10_000  # 10k positive + 10k negative

print("Loading dataset...")
df = pd.read_csv(INPUT_FILE)

# Split classes
df_pos = df[df["interaction"] == 1]
df_neg = df[df["interaction"] == 0]

print("Original counts:")
print("Positive:", len(df_pos))
print("Negative:", len(df_neg))

# Sample equally
df_pos_sampled = df_pos.sample(n=SAMPLES_PER_CLASS, random_state=42)
df_neg_sampled = df_neg.sample(n=SAMPLES_PER_CLASS, random_state=42)

# Combine and shuffle
df_balanced = pd.concat([df_pos_sampled, df_neg_sampled])
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

# Save
df_balanced.to_csv(OUTPUT_FILE, index=False)

print("Balanced dataset saved:", OUTPUT_FILE)
print("Final size:", len(df_balanced))
print("Positive:", df_balanced["interaction"].sum())
print("Negative:", len(df_balanced) - df_balanced["interaction"].sum())
