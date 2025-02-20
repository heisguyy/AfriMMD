import os
import torch
import pandas as pd
from tqdm import tqdm
from siglip_nllb import SiglipNllb, Tokenizer
import torch
from dataset import DatasetProcessor
from datasets import load_dataset, DatasetDict
from huggingface_hub import hf_hub_download


# set important parameters
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
OUTPUT_DIR = "investigation"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# load test data
processor = DatasetProcessor()
# raw_data = load_dataset("AfriMM/AfriMMD")
# raw_data = raw_data["train"]
# processed_data = processor.process(raw_data)
# processed_data.save_to_disk("processed_data")
processed_data = DatasetDict.load_from_disk("processed_data")

test_dataloader = torch.utils.data.DataLoader(
    processed_data["test"],
    batch_size=BATCH_SIZE,
    collate_fn=processor.collate_batch,
)


# load model
weights_path = hf_hub_download(
    repo_id="AfriMM/SiglipNllb",
    filename="model.pth"
)
model = SiglipNllb()
checkpoint = torch.load(weights_path)
fixed_state_dict = {
    k.replace("_orig_mod.", ""): v
    for k, v in checkpoint["model_state_dict"].items()
}
model.load_state_dict(fixed_state_dict)
model.to(DEVICE)


model.eval()
val_loss = 0
num_steps = len(test_dataloader)
prediction = []
language_codes = []
references = []
with torch.no_grad():
    pbar = tqdm(test_dataloader, desc="Evaluating")
    for batch in pbar:
        batch = {
            k: v if k in ["lang_code", "caption"] else v.to(DEVICE)
            for k, v in batch.items()
        }

        outputs = model(batch)
        pred_tokens = torch.argmax(outputs, dim=-1)

        prediction.extend(pred_tokens.cpu().tolist())
        references.extend(batch["caption"])
        language_codes.extend(batch["lang_code"])


dataframe = pd.DataFrame(
    {
        "predictions": prediction,
        "references": references,
        "language": language_codes,
    }
)
dataframe["candidates"] = ""
for lang in dataframe["language"].unique():
    tokenizer = processor.get_tokenizer(lang)
    lang_df = dataframe[dataframe["language"] == lang]
    prediction = tokenizer.text_tokenizer.batch_decode(
        lang_df.predictions.to_list(),
        skip_special_tokens=True,
    )
    dataframe.loc[
        dataframe["language"] == lang, "candidates"
    ] = prediction
dataframe.to_csv(os.path.join(OUTPUT_DIR, "test_predictions.csv"))