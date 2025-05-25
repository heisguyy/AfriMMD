#!/usr/bin/env python3
import csv
import re

# Define the patterns to remove
patterns = [
    r'system\nYou are a helpful assistant\.\nuser\n\nCaption the image in (Amharic|Igbo) language.*?\nassistant\n',
    r'system\nYou are a helpful assistant\.\nuser\n\nCaption the image in (አማርኛ|Igbo).*?\nassistant\n'
]

# Compile the patterns
compiled_patterns = [re.compile(pattern, re.DOTALL) for pattern in patterns]

# Input and output files
input_file = 'pangea_amh_ibo_captions.csv'
output_file = 'pangea_amh_ibo_captions_cleaned.csv'

# Process the CSV file
with open(input_file, 'r', newline='', encoding='utf-8') as infile, \
     open(output_file, 'w', newline='', encoding='utf-8') as outfile:
    
    reader = csv.reader(infile)
    writer = csv.writer(outfile)
    
    # Read the header
    header = next(reader)
    writer.writerow(header)
    
    # Process each row
    for row in reader:
        if len(row) >= 3:  # Make sure we have at least 3 columns
            image_id, language, caption = row[0], row[1], row[2]
            
            # Clean the caption by removing the patterns
            for pattern in compiled_patterns:
                caption = pattern.sub('', caption)
            
            # Write the cleaned row
            writer.writerow([image_id, language, caption])
        else:
            # Write the row as is if it doesn't have enough columns
            writer.writerow(row)

print(f"Cleaned captions saved to {output_file}") 