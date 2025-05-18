import pandas as pd
import json
import os
from pycocoevalcap.spice.spice import Spice
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

# --- Config ---
csv_path = "finetune/test_predictions.csv"
output_path = "finetune/SPICE_scores.csv"

# --- Load CSV ---
df = pd.read_csv(csv_path)

# --- Prepare inputs for SPICE ---
# Format data for pycocoevalcap
references = {}
predictions = {}

for idx, (_, row) in enumerate(df.iterrows()):
    image_id = str(idx)
    references[image_id] = [{'image_id': image_id, 'caption': row['references']}]
    predictions[image_id] = [{'image_id': image_id, 'caption': row['candidates']}]

# Create tokenizer
tokenizer = PTBTokenizer()

# Tokenize
tokenized_refs = tokenizer.tokenize(references)
tokenized_preds = tokenizer.tokenize(predictions)

# --- Load SPICE metric ---
spice_scorer = Spice()

# --- Compute overall SPICE score ---
overall_score, _ = spice_scorer.compute_score(tokenized_refs, tokenized_preds)
print(f"✅ Overall SPICE: {overall_score:.4f}")

# --- Compute SPICE per language ---
language_scores = []
for lang, group in df.groupby('language'):
    lang_refs = {}
    lang_preds = {}
    
    for idx, (_, row) in enumerate(group.iterrows()):
        image_id = str(idx)
        lang_refs[image_id] = [{'image_id': image_id, 'caption': row['references']}]
        lang_preds[image_id] = [{'image_id': image_id, 'caption': row['candidates']}]
    
    try:
        # Tokenize
        tokenized_lang_refs = tokenizer.tokenize(lang_refs)
        tokenized_lang_preds = tokenizer.tokenize(lang_preds)
        
        # Compute SPICE
        score, _ = spice_scorer.compute_score(tokenized_lang_refs, tokenized_lang_preds)
    except Exception as e:
        print(f"[!] Skipping {lang} due to error: {e}")
        score = None
    
    language_scores.append({'language': lang, 'SPICE': score})

# --- Save results ---
results_df = pd.DataFrame(language_scores)
results_df.loc[len(results_df.index)] = {'language': 'Overall', 'SPICE': overall_score}
results_df.to_csv(output_path, index=False)

print(f"\n📁 SPICE scores saved to: {output_path}")
print(results_df)
