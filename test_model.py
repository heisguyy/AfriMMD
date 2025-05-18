from siglip_nllb import Tokenizer
import torch
from torch import nn
from PIL import Image
from model import VisionEncoderDecoderModel
from transformers import (SiglipVisionModel, M2M100ForConditionalGeneration)
from huggingface_hub import hf_hub_download

vision_encoder = SiglipVisionModel.from_pretrained("google/siglip-base-patch16-256-multilingual")
decoder = M2M100ForConditionalGeneration.from_pretrained("facebook/nllb-200-distilled-600M").model.decoder
model = VisionEncoderDecoderModel(encoder=vision_encoder, decoder=decoder)

model.config.bos_token_id = decoder.config.bos_token_id
model.config.eos_token_id = decoder.config.eos_token_id
model.config.pad_token_id = decoder.config.pad_token_id
model.config.decoder_start_token_id = model.config.bos_token_id

device = torch.device("cpu")

model_path = hf_hub_download(repo_id="AfriMM/SiglipNllb", filename="model.pth")

# Load the model
checkpoint = torch.load(model_path, weights_only=True)
fixed_state_dict = {k.replace('_orig_mod.', ''): v for k, v in checkpoint['model_state_dict'].items()}
model.load_state_dict(fixed_state_dict)
model.to(device)
model = torch.compile(model)

loss_fn = nn.CrossEntropyLoss()
tokenizer = Tokenizer()
image_ = Image.open("data/Images/41999070_838089137e.jpg").convert("RGB")
tokenized_input = tokenizer(image_)

# Move inputs to the same device as the model
for key, value in tokenized_input.items():
    if isinstance(value, torch.Tensor):
        tokenized_input[key] = value.to(device)

# Check the first few tokens that would be produced
yor_id = tokenizer.text_tokenizer.convert_tokens_to_ids("yor_Latn")
print(f"Yoruba token ID: {yor_id}")

result = model.generate(
    **tokenized_input,
    forced_bos_token_id=tokenizer.text_tokenizer.convert_tokens_to_ids(
        "yor_Latn"
    ),
    max_length=100,
    num_beams=5,
    no_repeat_ngram_size=2,
    temperature=0.7,
    do_sample=True,
    top_k=50,
    top_p=0.95
)
print(
    tokenizer.detokenize(
        result.squeeze().tolist(), skip_special_tokens=False
    )
)
