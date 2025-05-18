from transformers import LlavaNextForConditionalGeneration, AutoProcessor
import torch
from PIL import Image
import pandas as pd
import os
import csv
from tqdm import tqdm

# Check if CUDA is available
if not torch.cuda.is_available():
    print("WARNING: CUDA is not available. Using CPU instead.")
    device = torch.device("cpu")
else:
    device = torch.device("cuda:0")
    print(f"Using CUDA device: {torch.cuda.get_device_name(device)}")

# Load the test data
test_data = pd.read_csv('test_data.csv')
print(f"Loaded {len(test_data)} test samples")

# Filter to only include Amharic and Igbo languages
test_data = test_data[test_data['language'].isin(['amh', 'ibo'])]
print(f"Filtered to {len(test_data)} samples with languages 'amh' (Amharic) and 'ibo' (Igbo)")

# Load the model and processor (only once to save memory)
model = LlavaNextForConditionalGeneration.from_pretrained(
    "neulab/Pangea-7B-hf", 
    torch_dtype=torch.float16
).to(device)
processor = AutoProcessor.from_pretrained("neulab/Pangea-7B-hf")
model.resize_token_embeddings(len(processor.tokenizer))

# Create output file name
output_file = 'pangea_amh_ibo_captions.csv'

# Create output CSV file
with open(output_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['image_id', 'language', 'generated_caption'])

# Process each image
for idx, row in tqdm(test_data.iterrows(), total=len(test_data), desc="Processing images"):
    image_id = row['image_id']
    language = row['language']
    image_path = os.path.join("data/Images", image_id)
    
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        continue
        
    try:
        image = Image.open(image_path)
        
        # Prepare prompt with the specific language
        if language == "amh":
            text_input = "Caption the image in Amharic language (አማርኛ). Be descriptive and focus on the main subjects and actions."
        elif language == "ibo":
            text_input = "Caption the image in Igbo language (Asụsụ Igbo). Be descriptive and focus on the main subjects and actions."
        
        prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<image>\n{text_input}<|im_end|>\n<|im_start|>assistant\n"
        
        # Process the image and generate caption
        model_inputs = processor(images=image, text=prompt, return_tensors='pt').to(device, torch.float16)
        
        with torch.cuda.amp.autocast():
            output = model.generate(
                **model_inputs, 
                max_new_tokens=1024, 
                min_new_tokens=32, 
                temperature=1.0, 
                top_p=0.9, 
                do_sample=True
            )
        
        # Decode the output
        result = processor.decode(output[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        
        # Save result to CSV
        with open(output_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([image_id, language, result])
        
        print(f"Processed {image_id} in {language}")
        
        # Clear CUDA cache to prevent memory issues
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    except Exception as e:
        print(f"Error processing {image_id}: {str(e)}")

print(f"Captions generated and saved to {output_file}")
