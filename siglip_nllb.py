"""
Script to train a multilingual image captioning model
combining the power of siglip and nllb
"""

from PIL import Image
from torch import nn
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    SiglipVisionModel,
    M2M100ForConditionalGeneration,
)
from model import VisionEncoderDecoderModel
import torch
import os

# The tokenization method is `<tokens> <eos> <language code>` for source
# language documents, and `<language code>
# <tokens> <eos>` for target language documents.


class Tokenizer:
    """
    Tokenizer class
    """

    def __init__(self):
        self.image_processor, self.text_tokenizer = (
            self.__load_from_huggingface()
        )
        self.text_tokenizer.add_bos_token = False
        self.text_tokenizer.add_eos_token = False

    def __call__(self, image: Image.Image, target: str = None):
        """
        Tokenize the image and text
        """
        pixel_values = self.image_processor(
            images=image, return_tensors="pt"
        ).pixel_values

        return_data = {"pixel_values": pixel_values}
        if target is not None:
            labels = self.text_tokenizer(
                text_target=target,
                return_tensors="pt",
                return_attention_mask=False,
            )
            return_data["labels"] = labels["input_ids"]
        return return_data

    def detokenize(self, input_ids, skip_special_tokens=False):
        """
        Detokenize the input_ids
        """
        return self.text_tokenizer.decode(
            input_ids, skip_special_tokens=skip_special_tokens
        )

    def __load_from_huggingface(self):
        siglip_image_processor = AutoProcessor.from_pretrained(
            "google/siglip-base-patch16-256-multilingual"
        ).image_processor
        nllb_tokenizer = AutoTokenizer.from_pretrained(
            "facebook/nllb-200-distilled-600M"
        )
        return siglip_image_processor, nllb_tokenizer


vision_encoder = SiglipVisionModel.from_pretrained(
    "google/siglip-base-patch16-256-multilingual"
)
decoder = M2M100ForConditionalGeneration.from_pretrained(
    "facebook/nllb-200-distilled-600M"
).model.decoder
model = VisionEncoderDecoderModel(encoder=vision_encoder, decoder=decoder)

model.config.bos_token_id = decoder.config.bos_token_id
model.config.eos_token_id = decoder.config.eos_token_id
model.config.pad_token_id = decoder.config.pad_token_id

# Load the model state dictionary from the checkpoint

# Check if the checkpoint file exists
checkpoint_path = "finetune/checkpoint_epoch_10.pth"
if os.path.exists(checkpoint_path):
    # Load the state dictionary
    state_dict = torch.load(checkpoint_path)["model_state_dict"]
    # Load the state dictionary into the model
    # Remove "_orig_mod." prefix from keys if present (added when model is compiled)
    clean_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("_orig_mod."):
            clean_state_dict[key[len("_orig_mod."):]] = value
        else:
            clean_state_dict[key] = value
    state_dict = clean_state_dict
    model.load_state_dict(state_dict)
    print(f"Model loaded from {checkpoint_path}")


if __name__ == "__main__":
    loss_fn = nn.CrossEntropyLoss()
    tokenizer = Tokenizer()
    image_ = Image.open("data/Images/10815824_2997e03d76.jpg").convert("RGB")
    tokenized_input = tokenizer(image_)
    result = model.generate(
        **tokenized_input,
        forced_bos_token_id=tokenizer.text_tokenizer.convert_tokens_to_ids(
            "eng_Latn"
        ),
    )
    print(
        tokenizer.detokenize(
            result.squeeze().tolist(), skip_special_tokens=False
        )
    )
