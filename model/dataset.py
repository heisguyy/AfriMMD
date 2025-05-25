import torch
from PIL import Image
from siglip_nllb import Tokenizer
from datasets import DatasetDict, concatenate_datasets
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
        self.tokenizers = {}  # Cache for tokenizers
        
    def get_tokenizer(self, lang_code: str) -> Tokenizer:
        """Get or create tokenizer for a language"""
        if lang_code not in self.tokenizers:
            self.tokenizers[lang_code] = Tokenizer()
            self.tokenizers[lang_code].text_tokenizer.tgt_lang = lang_code
        return self.tokenizers[lang_code]
        
    def tokenize(self, example: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Tokenize a single example"""
        tokenizer = self.get_tokenizer(example['lang_code'])
        image = Image.open(f"data/Images/{example['image_id']}").convert("RGB")
        return tokenizer(image, example['caption'])

    def collate_batch(self, batch) -> Dict[str, torch.Tensor]:
        """Collate batch with proper padding per language"""
        pixel_values = torch.stack([item["pixel_values"] for item in batch]).squeeze(1)
        labels = [item["labels"].squeeze(0) for item in batch]
        
        # Get language codes and pad tokens for this batch
        lang_codes = [item["lang_code"] for item in batch]
        captions = [item["caption"] for item in batch]
        pad_tokens = [self.get_tokenizer(lang).text_tokenizer.pad_token_id for lang in lang_codes]
        
        # Pad sequences using language-specific pad tokens
        max_len = max(len(ids) for ids in labels)
        padded_labels = torch.stack([
            torch.cat([ids, torch.full((max_len - len(ids),), pad_tokens[i])])  
            for i, ids in enumerate(labels)
        ])

        return {
            "pixel_values": pixel_values,
            "labels": padded_labels,
            "lang_code": lang_codes,
            "caption": captions,
        }

    def transform_dataset(self, dataset) -> DatasetDict:
        """Transform dataset to include language codes and captions"""

        def transform(data):
            combined_datasets = []

            # Process each language split and add language code
            for lang, dataset in data.items():
                # Add language code to each example
                dataset = dataset.map(lambda example: {
                    "image_id": example["image_id"],
                    "caption": example["caption"],
                    "lang_code": self.lang_mapper[lang]
                })
                combined_datasets.append(dataset)

            # Combine all datasets
            combined_data = concatenate_datasets(combined_datasets)
            # Shuffle the combined dataset
            combined_data = combined_data.shuffle(seed=42)
            return combined_data

        dataset = transform(dataset)
        
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
                    "labels"
                ],
                output_all_columns=True,
            )
        return tokenized
