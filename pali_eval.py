import os
import pandas as pd
from PIL import Image
from typing import Optional
from transformers import AutoProcessor, MllamaForConditionalGeneration
import torch
from tqdm import tqdm
import glob

def caption_images_with_llama(
    csv_path: str,
    image_root_dir: str,
    output_csv_path: str,
    image_column: str = "image_path",
    reference_column: Optional[str] = None,
    target_language: str = "yoruba",
    batch_size: int = 4,
    model_name: str = "meta-llama/Llama-3.2-11B-Vision"
):
    """
    Generate image captions using PaliGemma 2 in a target African language and save results to CSV.

    Args:
        csv_path (str): Path to the CSV containing image paths.
        image_root_dir (str): Root directory where images are stored.
        output_csv_path (str): Output CSV file path to save image_id, reference, prediction.
        image_column (str): Column in CSV with image paths.
        reference_column (Optional[str]): Column in CSV with reference captions (optional).
        target_language_code (str): Language to caption in (e.g., 'sw' for Swahili, 'yo' for Yoruba).
        batch_size (int): Number of images to process in a batch.
        model_name (str): Model ID on Hugging Face (default is PaliGemma 2).
    """
    has_cuda = torch.cuda.is_available()
    
    processor = AutoProcessor.from_pretrained(model_name)
    model = MllamaForConditionalGeneration.from_pretrained(
        model_name, 
        torch_dtype=torch.bfloat16,
        device_map="auto"
    ).eval()
    
    # Determine the device for input tensors
    device = torch.device("cuda")
    
    df = pd.read_csv(csv_path)
    all_results = []

    for i in tqdm(range(0, len(df), batch_size), desc="Captioning Batches"):
        batch_df = df.iloc[i:i+batch_size]
        image_paths = [
            os.path.join(image_root_dir, path) for path in batch_df[image_column]
        ]
        images = [
            [Image.open(p).convert("RGB")] for p in image_paths if os.path.exists(p)
        ]
        print(len(image_paths) - len(images), "images not found")
        
        prompt = f"<|image|><|begin_of_text|>Caption this image in {target_language} language"
        inputs = processor(images=images, text=[prompt]*len(images), return_tensors="pt", padding=True)
        
        # Move inputs to appropriate device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.inference_mode():
            input_len = inputs["input_ids"].shape[-1]
            outputs = model.generate(**inputs, max_new_tokens=100, do_sample=False)
            
            # Process each generated output
            captions = []
            for output in outputs:
                generation = output[input_len:]
                caption = processor.decode(generation, skip_special_tokens=True)
                captions.append(caption)

        for idx, row in batch_df.iterrows():
            result = {
                "image_id": os.path.basename(row[image_column]),
                "reference": row[reference_column] if reference_column else "",
                "prediction": captions[idx - i]
            }
            all_results.append(result)

    output_df = pd.DataFrame(all_results)
    output_df.to_csv(output_csv_path, index=False)
    print(f"✅ Captioning complete. Results saved to: {output_csv_path}")

languages = [
    "Amharic",
    "Hausa",
    "Igbo",
    "Yoruba"
]

for language in languages:
    caption_images_with_llama(
        csv_path="test_data.csv",
        image_root_dir="data/Images",
        output_csv_path=f"finetune/llama_predictions/{language}.csv",
        image_column="image_id",
        reference_column="caption",
        target_language=language.lower(),
        batch_size=4,
    )



# Path to the predictions directory
predictions_dir = "finetune/llama_predictions"

# Get all CSV files in the directory
csv_files = glob.glob(os.path.join(predictions_dir, "*.csv"))

# List to hold all dataframes
all_dfs = []

# Process each CSV file
for csv_file in csv_files:
    if os.path.basename(csv_file) != "all.csv":  # Skip the output file if it exists
        language = os.path.splitext(os.path.basename(csv_file))[0]  # Extract language name from filename
        print(f"Processing {os.path.basename(csv_file)}")
        df = pd.read_csv(csv_file)
        # Add language column
        df['language'] = language
        # Drop duplicates based on image_id
        df = df.drop_duplicates(subset=['image_id'])
        all_dfs.append(df)

# Concatenate all dataframes
combined_df = pd.concat(all_dfs, ignore_index=True)
combined_df = combined_df.rename(columns={"prediction": "candidates", "reference": "references"})
# Drop rows with any NaN values
combined_df = combined_df.dropna()
print(f"Dropped {len(all_dfs) - len(combined_df)} rows with NaN values")

# Save the combined dataframe
output_path = os.path.join(predictions_dir, "all.csv")
combined_df.to_csv(output_path, index=False)
print(f"✅ Combined data saved to: {output_path}")
print(f"Total rows in combined dataset: {len(combined_df)}")