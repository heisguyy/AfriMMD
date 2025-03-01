"""
Script to train a multilingual image captioning model
combining the power of siglip and nllb
"""

from typing import Optional

from PIL import Image
from torch import nn
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    SiglipVisionModel,
)
from model import VisionEncoderDecoderModel

from transformers.models.m2m_100.modeling_m2m_100 import M2M100Decoder

# The tokenization method is `<tokens> <eos> <language code>` for source
# language documents, and `<language code>
# <tokens> <eos>` for target language documents.


class Tokenizer:
    """
    Tokenizer class
    """

    def __init__(self, target_lang: str):
        self.image_processor, self.text_tokenizer = (
            self.__load_from_huggingface(target_lang)
        )
        self.text_tokenizer.add_bos_token = False
        self.text_tokenizer.add_eos_token = False

    def __call__(self, image: Image.Image, target_language: str):
        """
        Tokenize the image and text
        """
        pixel_values = self.image_processor(
            images=image, return_tensors="pt"
        ).pixel_values

        inputs = self.text_tokenizer(
            text_target=target_language, return_tensors="pt", return_attention_mask=True,
            add_special_tokens=False
        )
        inputs = {f"decoder_{k}": v for k, v in inputs.items()}
        return_data = {"pixel_values": pixel_values, **inputs}
        return return_data

    def detokenize(self, input_ids, skip_special_tokens=False):
        """
        Detokenize the input_ids
        """
        return self.text_tokenizer.decode(
            input_ids, skip_special_tokens=skip_special_tokens
        )

    def __load_from_huggingface(self, target_lang):
        siglip_image_processor = AutoProcessor.from_pretrained(
            "google/siglip-base-patch16-256-multilingual"
        ).image_processor
        nllb_tokenizer = AutoTokenizer.from_pretrained(
            "facebook/nllb-200-distilled-600M", tgt_lang=target_lang
        )
        return siglip_image_processor, nllb_tokenizer

vision_encoder = SiglipVisionModel.from_pretrained(
    "google/siglip-base-patch16-256-multilingual"
)
decoder = M2M100Decoder.from_pretrained("facebook/nllb-200-distilled-600M")
model = VisionEncoderDecoderModel(encoder=vision_encoder, decoder=decoder)

model.config.bos_token_id = decoder.config.bos_token_id
model.config.eos_token_id = decoder.config.eos_token_id
model.config.pad_token_id = decoder.config.pad_token_id



if __name__ == "__main__":
    loss_fn = nn.CrossEntropyLoss()
    tokenizer = Tokenizer("yor_Latn")
    image_ = Image.open("data/Images/10815824_2997e03d76.jpg").convert("RGB")
    tokenized_input= tokenizer(image_, "ibo")
    result = model.generate(**tokenized_input)
    print(tokenizer.detokenize(result.squeeze().tolist(), skip_special_tokens=True))