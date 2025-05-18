import os
import pandas as pd
from PIL import Image
from typing import Optional, List
from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration
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
    model_name: str = "google/paligemma2-3b-pt-224"
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
    
    processor = PaliGemmaProcessor.from_pretrained(model_name)
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        model_name, 
        torch_dtype=torch.bfloat16 if has_cuda else torch.float32,
        device_map="auto"
    ).eval()
    
    # Determine the device for input tensors
    device = model.device if not isinstance(model.device, str) else torch.device("cuda" if has_cuda else "cpu")
    
    df = pd.read_csv(csv_path)
    all_results = []

    for i in tqdm(range(0, len(df), batch_size), desc="Captioning Batches"):
        batch_df = df.iloc[i:i+batch_size]
        image_paths = [
            os.path.join(image_root_dir, path) for path in batch_df[image_column]
        ]
        images = [Image.open(p).convert("RGB") for p in image_paths]
        
        prompt = f"Describe the image in {target_language_code}:"
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


caption_images_with_paliGemma2(
    csv_path="test_data.csv",
    image_root_dir="data/images",
    output_csv_path="finetune/paligemma_predictions.csv",
    image_column="image_id",
    reference_column="caption",
    target_language_code="sw",
    batch_size=8
)
