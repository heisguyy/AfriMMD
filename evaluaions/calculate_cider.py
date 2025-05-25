#!/usr/bin/env python3
import pandas as pd
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

# --- Config ---
candidates_csv_path = "pangea_amh_ibo_captions_cleaned.csv"
references_csv_path = "test_data.csv"
output_path = "finetune/pangea_cider_scores.csv"

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

# --- Prepare inputs for CIDEr ---
# Format data for pycocoevalcap
references = {}
predictions = {}

for idx, (_, row) in enumerate(merged_df.iterrows()):
    image_id = str(idx)
    references[image_id] = [{'image_id': image_id, 'caption': row['reference']}]
    predictions[image_id] = [{'image_id': image_id, 'caption': row['candidate']}]

# Create tokenizer
tokenizer = PTBTokenizer()

# Tokenize
tokenized_refs = tokenizer.tokenize(references)
tokenized_preds = tokenizer.tokenize(predictions)

# --- Load CIDEr metric ---
cider_scorer = Cider()

# --- Compute overall CIDEr score ---
overall_score, _ = cider_scorer.compute_score(tokenized_refs, tokenized_preds)
print(f"✅ Overall CIDEr: {overall_score:.4f}")

# --- Compute CIDEr per language ---
language_scores = []
for lang, group in merged_df.groupby('language'):
    lang_refs = {}
    lang_preds = {}
    
    for idx, (_, row) in enumerate(group.iterrows()):
        image_id = str(idx)
        lang_refs[image_id] = [{'image_id': image_id, 'caption': row['reference']}]
        lang_preds[image_id] = [{'image_id': image_id, 'caption': row['candidate']}]
    
    try:
        # Tokenize
        tokenized_lang_refs = tokenizer.tokenize(lang_refs)
        tokenized_lang_preds = tokenizer.tokenize(lang_preds)
        
        # Compute CIDEr
        score, _ = cider_scorer.compute_score(tokenized_lang_refs, tokenized_lang_preds)
    except Exception as e:
        print(f"[!] Skipped {lang} due to error: {e}")
        score = None
    
    language_scores.append({'language': lang, 'CIDEr': score})

# --- Save results ---
results_df = pd.DataFrame(language_scores)
results_df.loc[len(results_df.index)] = {'language': 'Overall', 'CIDEr': overall_score}
results_df.to_csv(output_path, index=False)

print(f"\n📁 CIDEr scores saved to: {output_path}")
print(results_df) 