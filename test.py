
import pandas as pd

df = pd.read_csv("data/raw/bindingdb_filtered.csv")

print("Total samples:", len(df))
print("Positive interactions:", df["interaction"].sum())
print("Negative interactions:", len(df) - df["interaction"].sum())


# import pandas as pd


# # File paths

# INPUT_FILE = "data/raw/BindingDB_All.tsv"
# OUTPUT_FILE = "data/raw/bindingdb_filtered.csv"


# # Chunk size (safe for Windows)

# CHUNK_SIZE = 100_000


# # Required BindingDB columns

# SMILES_COL = "Ligand SMILES"
# SEQUENCE_COL = "BindingDB Target Chain Sequence 1"
# AFFINITY_COLS = ["IC50 (nM)", "Ki (nM)", "Kd (nM)"]

# filtered_chunks = []

# print("Starting BindingDB filtering...")


# # Chunk-wise reading

# for chunk in pd.read_csv(
#     INPUT_FILE,
#     sep="\t",
#     chunksize=CHUNK_SIZE,
#     low_memory=True
# ):
    
#     # Keep only required columns (if present)
    
#     keep_cols = [c for c in [SMILES_COL, SEQUENCE_COL] if c in chunk.columns]
#     keep_cols += [c for c in AFFINITY_COLS if c in chunk.columns]

#     chunk = chunk[keep_cols]

    
#     # Drop rows with missing SMILES or protein sequence
    
#     chunk = chunk.dropna(subset=[SMILES_COL, SEQUENCE_COL])

    
#     # Create interaction label
    
#     def interaction_label(row):
#         for col in AFFINITY_COLS:
#             if col in row:
#                 val = row[col]
#                 if pd.notna(val):
#                     try:
#                         if float(val) <= 1000:
#                             return 1
#                     except:
#                         pass
#         return 0

#     chunk["interaction"] = chunk.apply(interaction_label, axis=1)

    
#     # Rename columns to match ML pipeline
    
#     chunk = chunk.rename(columns={
#         SMILES_COL: "smiles",
#         SEQUENCE_COL: "protein_sequence"
#     })

#     filtered_chunks.append(
#         chunk[["smiles", "protein_sequence", "interaction"]]
#     )


# # Combine all chunks

# final_df = pd.concat(filtered_chunks, ignore_index=True)


# # Save final dataset

# final_df.to_csv(OUTPUT_FILE, index=False)

# print("✅ Filtering completed")
# print("Saved file:", OUTPUT_FILE)
# print("Total samples:", len(final_df))
# print("Positive interactions:", final_df["interaction"].sum())
# print("Negative interactions:", len(final_df) - final_df["interaction"].sum())


# import pandas as pd

# for chunk in pd.read_csv(
#     "data/raw/BindingDB_All.tsv",
#     sep="\t",
#     chunksize=1,
#     low_memory=True
# ):
#     print(chunk.columns.tolist())
#     break
