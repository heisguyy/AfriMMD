import pandas as pd
import evaluate

# --- Config ---
csv_path = "finetune/test_predictions.csv"
output_path = "Africaption_BLEU_scores.csv"

# --- Load CSV ---
df = pd.read_csv(csv_path)

# --- Prepare inputs ---
predictions = df['candidates'].tolist()
references = [[ref] for ref in df['references'].tolist()]  # wrap each reference in a list

# --- Load BLEU metric ---
bleu = evaluate.load("bleu")

# --- Compute overall BLEU ---
overall_result = bleu.compute(predictions=predictions, references=references)
overall_score = overall_result['bleu']
print(f"✅ Overall BLEU: {overall_score:.4f}")

# --- Compute per-language BLEU ---
language_scores = []
for lang, group in df.groupby('language'):
    lang_preds = group['candidates'].tolist()
    lang_refs = [[r] for r in group['references'].tolist()]
    
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
