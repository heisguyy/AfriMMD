#!/usr/bin/env python3
import pandas as pd
import evaluate

# --- Config ---
candidates_csv_path = "pangea_amh_ibo_captions_cleaned.csv"
references_csv_path = "test_data.csv"
output_path = "finetune/pangea_bleu_scores.csv"

print(f"Loading candidates from: {candidates_csv_path}")
print(f"Loading references from: {references_csv_path}")

# --- Load CSVs ---
try:
    candidates_df = pd.read_csv(candidates_csv_path)
    references_df = pd.read_csv(references_csv_path)
    
    print(f"Candidates shape: {candidates_df.shape}")
    print(f"References shape: {references_df.shape}")
except Exception as e:
    print(f"Error loading CSV files: {e}")
    exit(1)

# --- Prepare merged dataframe ---
# Rename columns for clarity
references_df = references_df.rename(columns={'caption': 'reference'})
candidates_df = candidates_df.rename(columns={'generated_caption': 'candidate'})

# Perform merge on image_id and language
merged_df = pd.merge(
    candidates_df[['image_id', 'language', 'candidate']], 
    references_df[['image_id', 'language', 'reference']], 
    on=['image_id', 'language'], 
    how='inner'
)

print(f"Merged dataframe shape: {merged_df.shape}")

if merged_df.empty:
    print("No matching entries found. Check image_id and language values in both files.")
    exit(1)

# --- Load BLEU metric ---
bleu = evaluate.load("bleu")

# --- Compute overall BLEU ---
predictions = merged_df['candidate'].tolist()
references = [[ref] for ref in merged_df['reference'].tolist()]  # wrap each reference in a list

overall_result = bleu.compute(predictions=predictions, references=references)
overall_score = overall_result['bleu']
print(f"✅ Overall BLEU: {overall_score:.4f}")

# --- Compute per-language BLEU ---
language_scores = []
for lang, group in merged_df.groupby('language'):
    lang_preds = group['candidate'].tolist()
    lang_refs = [[r] for r in group['reference'].tolist()]
    
    try:
        result = bleu.compute(predictions=lang_preds, references=lang_refs)
        score = result['bleu']
    except Exception as e:
        print(f"[!] Skipped {lang} due to error: {e}")
        score = None
    
    language_scores.append({'language': lang, 'BLEU': score})

# --- Save results ---
results_df = pd.DataFrame(language_scores)
results_df.loc[len(results_df.index)] = {'language': 'Overall', 'BLEU': overall_score}
results_df.to_csv(output_path, index=False)

print(f"\n📁 BLEU scores saved to: {output_path}")
print(results_df) 