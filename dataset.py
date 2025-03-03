import torch
from PIL import Image
from siglip_nllb import Tokenizer
from datasets import DatasetDict
from typing import Dict, List, Any

class DatasetProcessor:
    lang_mapper = {
        'kab': 'kab_Latn',
        'afr': 'afr_Latn',
        'aka': 'aka_Latn',
        'amh': 'amh_Ethi',
        'ary': 'ary_Arab',
        'arz': 'arz_Arab',
        'bem': 'bem_Latn',
        'cjk': 'cjk_Latn',
        'dik': 'dik_Latn',
        'dyu': 'dyu_Latn',
        'eng': 'eng_Latn',
        'ewe': 'ewe_Latn',
        'fuv': 'fuv_Latn',
        'hau': 'hau_Latn',
        'ibo': 'ibo_Latn',
        'kam': 'kam_Latn',
        'kik': 'kik_Latn',
        'kin': 'kin_Latn',
        'kmb': 'kmb_Latn',
        'knc': 'knc_Latn',
        'kon': 'kon_Latn',
        'lin': 'lin_Latn',
        'lua': 'lua_Latn',
        'lug': 'lug_Latn',
        'yor': 'yor_Latn',
    }
    def __init__(self, train_size: float = 0.8, test_size: float = 0.1):
        self.train_size = train_size
        self.test_size = test_size
        self.tokenizer = Tokenizer()
        
    def tokenize(self, example: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Tokenize a single example"""
        image = Image.open(f"data/Images/{example['image_id']}").convert("RGB")
        return self.tokenizer(image,example["lang_code"], example['caption'])

    def collate_batch(self, batch) -> Dict[str, torch.Tensor]:
        """Collate batch with proper padding per language"""
        pixel_values = torch.stack([item["pixel_values"] for item in batch]).squeeze(1)
        input_ids = [item["decoder_input_ids"].squeeze(0) for item in batch]
        attention_mask = [item["decoder_attention_mask"].squeeze(0) for item in batch]
        labels = [item["labels"].squeeze(0) for item in batch]
        
        # Get language codes and pad tokens for this batch
        lang_codes = [item["lang_code"] for item in batch]
        captions = [item["caption"] for item in batch]
        pad_token = self.tokenizer.text_tokenizer.pad_token_id
        
        # Pad sequences using language-specific pad tokens
        max_len = max(len(ids) for ids in labels)
        padded_input_ids = torch.stack(input_ids)
        padded_labels = torch.stack([
            torch.cat([ids, torch.full((max_len - len(ids),), pad_token)]) 
            for _, ids in enumerate(labels)
        ])
        
        padded_attention_mask = torch.nn.utils.rnn.pad_sequence(
            attention_mask, batch_first=True, padding_value=0
        )

        return {
            "pixel_values": pixel_values,
            "decoder_input_ids": padded_input_ids,
            "decoder_attention_mask": padded_attention_mask,
            "labels": padded_labels,
            "lang_code": lang_codes,
            "caption": captions,
        }

    def transform_dataset(self, dataset) -> DatasetDict:
        """Transform dataset to include language codes and captions"""
        dataset = dataset.remove_columns(["id"])
        lang_columns = [col for col in dataset.column_names if col not in ["image_id"]]

        def transform_row(batch):
            new_image_ids = []
            new_lang_codes = []
            new_captions = []

            for image_id, row in zip(batch["image_id"], zip(*[batch[col] for col in lang_columns])):
                for lang_code, caption in zip(lang_columns, row):
                    new_image_ids.append(image_id)
                    new_lang_codes.append(self.lang_mapper[lang_code])
                    new_captions.append(caption)

            return {
                "image_id": new_image_ids,
                "lang_code": new_lang_codes,
                "caption": new_captions,
            }

        dataset = dataset.map(transform_row, batched=True, remove_columns=lang_columns)
        
        # Split dataset
        train_test = dataset.train_test_split(train_size=self.train_size, test_size=self.test_size)
        test_valid = train_test["test"].train_test_split(train_size=0.5)

        return DatasetDict({
            "train": train_test["train"],
            "test": test_valid["train"],
            "validation": test_valid["test"]
        })
    def process(self, dataset) -> DatasetDict:
        """Complete dataset processing pipeline"""
        transformed = self.transform_dataset(dataset)
        tokenized = transformed.map(self.tokenize)
        # Convert to PyTorch tensors
        for split in tokenized.keys():
            tokenized[split].set_format(
                "torch",
                columns=[
                    "pixel_values",
                    "decoder_input_ids",
                    "decoder_attention_mask",
                    "labels"
                ],
                output_all_columns=True,
            )
        return tokenized