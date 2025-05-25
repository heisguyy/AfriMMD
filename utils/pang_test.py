from transformers import LlavaNextForConditionalGeneration, AutoProcessor
import torch
from PIL import Image
import pandas as pd
import os
import csv
from tqdm import tqdm

# Load model and processor
print("Loading Pangea-7B model...")
model = LlavaNextForConditionalGeneration.from_pretrained(
            "neulab/Pangea-7B-hf", 
            torch_dtype=torch.float16
        ).to(0)
processor = AutoProcessor.from_pretrained("neulab/Pangea-7B-hf")
# Fix the patch_size directly on the processor
processor.patch_size = 14
model.resize_token_embeddings(len(processor.tokenizer))
print("Model loaded successfully.")

# Read the input CSV with test data
test_data = pd.read_csv("test_data.csv")
print(f"Loaded {len(test_data)} test samples")

# Filter to only include Amharic and Igbo languages
test_data = test_data[test_data['language'].isin(['amh', 'ibo'])]
print(f"Filtered to {len(test_data)} samples with languages 'amh' (Amharic) and 'ibo' (Igbo)")

# Count by language
amh_count = len(test_data[test_data['language'] == 'amh'])
ibo_count = len(test_data[test_data['language'] == 'ibo'])
print(f"Distribution: {amh_count} Amharic samples, {ibo_count} Igbo samples")

# Read the existing output file to check which images are already captioned
existing_captions = {}
try:
    output_df = pd.read_csv("pangea_amh_ibo_captions.csv")
    # Create a dictionary to track which images already have captions for each language
    for _, row in output_df.iterrows():
        image_id = row['image_id']
        language = row['language'].lower()
        if image_id not in existing_captions:
            existing_captions[image_id] = []
        existing_captions[image_id].append(language)
    
    # Count already processed images by language
    amh_processed = sum(1 for img_id, langs in existing_captions.items() if 'amh' in langs)
    ibo_processed = sum(1 for img_id, langs in existing_captions.items() if 'ibo' in langs)
    print(f"Already processed: {amh_processed}/{amh_count} Amharic, {ibo_processed}/{ibo_count} Igbo")
    print(f"Remaining to process: {amh_count - amh_processed} Amharic, {ibo_count - ibo_processed} Igbo")

except FileNotFoundError:
    # If the file doesn't exist yet, create an empty dataframe
    output_df = pd.DataFrame(columns=['image_id', 'language', 'generated_caption'])
    print("No existing captions found. Starting from scratch.")

# Language prompts
languages = {
    "amh": "Caption the image in Amharic language (አማርኛ). Be descriptive and focus on the main subjects and actions.",
    "ibo": "Caption the image in Igbo language (Igbo). Be descriptive and focus on the main subjects and actions."
}

# Create a list of tasks to process (image_id, language)
tasks = []
for idx, row in test_data.iterrows():
    image_id = row['image_id']
    language = row['language']
    # Only add tasks that haven't been processed yet
    if image_id not in existing_captions or language not in existing_captions[image_id]:
        tasks.append((image_id, language))

print(f"Total tasks to process: {len(tasks)}")

# Process each task with a progress bar
for image_id, language in tqdm(tasks, desc="Processing images"):
    # Construct the correct image path by prepending "data/images/"
    image_path = os.path.join("data/images", image_id)
    
    try:
        # Open and process the image
        image_input = Image.open(image_path).convert('RGB')
        
        # Get the prompt for this language
        lang_prompt = languages[language]
        
        # Create the prompt
        text_input = lang_prompt
        text_input = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<image>\n{text_input}<|im_end|>\n<|im_start|>assistant\n"
        
        # Generate the caption
        model_inputs = processor(images=image_input, text=text_input, return_tensors='pt').to("cuda", torch.float16)
        output = model.generate(**model_inputs, max_new_tokens=1024, min_new_tokens=32, temperature=1.0, top_p=0.9, do_sample=True)
        output = output[0]
        result = processor.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        
        # Get language display name for CSV
        language_display_name = "Amharic" if language == "amh" else "Igbo"
        
        # Create a new row for the CSV
        new_row = {
            'image_id': image_id,
            'language': language,
            'generated_caption': result
        }
        
        # Append to the CSV file immediately to save progress
        with open('pangea_amh_ibo_captions.csv', 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['image_id', 'language', 'generated_caption'])
            # Write headers if it's a new file
            if f.tell() == 0:
                writer.writeheader()
            writer.writerow(new_row)
            
        # Update the in-memory tracking of completed images
        if image_id not in existing_captions:
            existing_captions[image_id] = []
        existing_captions[image_id].append(language)
        
        # Clear CUDA cache to prevent memory issues
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"\nError processing {image_id} for {language}: {str(e)}")

print("Processing complete!")
