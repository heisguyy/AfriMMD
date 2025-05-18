import os
import pandas as pd
from PIL import Image
from typing import Optional, List
from transformers import AutoProcessor, AutoModelForVision2Seq
import torch
from tqdm import tqdm

def caption_images_with_paliGemma2(
    csv_path: str,
    image_root_dir: str,
    output_csv_path: str,
    image_column: str = "image_path",
    reference_column: Optional[str] = None,
    target_language_code: str = "sw",
    batch_size: int = 4,
    model_name: str = "google/paligemma-3b-pt-224"
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForVision2Seq.from_pretrained(model_name).to(device)
    
    df = pd.read_csv(csv_path)
    all_results = []

    for i in tqdm(range(0, len(df), batch_size), desc="Captioning Batches"):
        batch_df = df.iloc[i:i+batch_size]
        image_paths = [
            os.path.join(image_root_dir, path) for path in batch_df[image_column]
        ]
        images = [Image.open(p).convert("RGB") for p in image_paths]
        
        prompt = f"Describe the image in {target_language_code}:"
        inputs = processor(images=images, text=[prompt]*len(images), return_tensors="pt", padding=True).to(device)
        
        with torch.no_grad():
            outputs = model.generate(**inputs)
        
        captions = processor.batch_decode(outputs, skip_special_tokens=True)

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


caption_images_with_paliGemma2(
    csv_path="test_data.csv",
    image_root_dir="data/images",
    output_csv_path="finetune/paligemma_predictions.csv",
    image_column="image_id",
    reference_column="caption",
    target_language_code="sw",
    batch_size=8
)
