import base64
import json
import os
import random
import time

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

load_dotenv()

client = OpenAI(api_key="")

SYSTEM_PROMPT = """You would be given images to caption in yoruba, igbo, hause and amharic.

Your response should be in this format:
yoruba caption: <yoruba caption>
igbo caption: <igbo caption>
hausa caption: <hausa caption>
amharic caption: <amharic caption>
"""


class Captions(BaseModel):
    """Captions for the images in yoruba, igbo, hause and amharic"""

    yoruba_caption: str
    igbo_caption: str
    hausa_caption: str
    amharic_caption: str


dataframe = pd.read_csv("test_data.csv")
image_ids = dataframe.image_id.unique().tolist()

script_dir = os.path.dirname(os.path.abspath(__file__))
image_ids = [
    os.path.join(script_dir, "data", "Images", os.path.basename(image_id))
    for image_id in image_ids
]

random.shuffle(image_ids)

jsonl_dir = os.path.join(script_dir, "jsonl_batches")
results_dir = os.path.join(script_dir, "batch_results")
os.makedirs(jsonl_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

batch_size = 150
chunks = [image_ids[i : i + batch_size] for i in range(0, len(image_ids), batch_size)]


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


batch_jobs = []

# For each chunk, create a jsonl file and create a batch job.
for batch_no, chunk in enumerate(chunks, start=1):
    jsonl_filename = os.path.join(jsonl_dir, f"batch_{batch_no}.jsonl")
    with open(jsonl_filename, "w", encoding="utf-8") as f:
        for i, image_path in enumerate(chunk, start=1):
            base64_image = encode_image(image_path)
            request_payload = {
                "custom_id": f"batch_{batch_no}_request_{i}",
                "method": "POST",
                "url": "/v1/responses",
                "body": {
                    "model": "gpt-4.1-2025-04-14",
                    "instructions": SYSTEM_PROMPT,
                    "input": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "input_image",
                                    "image_url": (
                                        f"data:image/jpeg;base64,{base64_image}"
                                    ),
                                    "detail": "low",
                                }
                            ],
                        }
                    ],
                    "text": {
                        "type": "json_schema",
                        "schema": Captions.model_json_schema(),
                    },
                },
            }
            f.write(json.dumps(request_payload) + "\n")
    with open(jsonl_filename, mode="rb") as file:
        batch_input_file = client.files.create(file=file, purpose="batch")
    print(f"Created input file for batch {batch_no}: {batch_input_file.id}")

    # Create batch job for current jsonl file
    batch = client.batches.create(
        input_file_id=batch_input_file.id,
        endpoint="/v1/responses",
        completion_window="24h",
        metadata={"description": f"Image captioning job for batch {batch_no}."},
    )
    print(f"Created Batch ID for batch {batch_no}: {batch.id}")
    # Store batch info along with the jsonl filename for later use.
    batch_jobs.append(
        {"batch_no": batch_no, "batch_id": batch.id, "status": None}
    )

# Poll all batches until each one is complete or failed.
pending_batches = {job["batch_id"]: job for job in batch_jobs}
while pending_batches:
    for batch_id, job in list(pending_batches.items()):
        batch = client.batches.retrieve(batch_id)
        if batch.status == "completed":
            file_response = client.files.content(batch.output_file_id)
            result_filename = os.path.join(
                results_dir, f"batch_{job['batch_no']}_result.txt"
            )
            with open(result_filename, "w", encoding="utf-8") as file:
                file.write(file_response.text)
            print(
                f"Batch {job['batch_no']} completed successfully."
                f" Result saved to {result_filename}"
            )
            pending_batches.pop(batch_id)
        elif batch.status == "failed":
            file_response = client.files.content(batch.output_file_id)
            result_filename = os.path.join(
                results_dir, f"batch_{job['batch_no']}_result.txt"
            )
            with open(result_filename, "w", encoding="utf-8") as file:
                file.write(file_response.text)
            print(
                f"Batch {job['batch_no']} FAILED. Check {result_filename} for details."
            )
            pending_batches.pop(batch_id)
        else:
            print(f"Batch {job['batch_no']} is {batch.status}.")
    if pending_batches:
        print("Sleeping for 5 minutes before checking again...")
        time.sleep(300)
