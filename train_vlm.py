import argparse
import json
import os
import random
import shutil
import zipfile

import gdown
import numpy as np
import torch
import wandb
from datasets import load_dataset
from torch import nn, optim
from tqdm.auto import tqdm
from transformers import get_inverse_sqrt_schedule
from transformers import (
    M2M100ForConditionalGeneration, SiglipVisionModel
)
from model import VisionEncoderDecoderModel

from dataset import DatasetProcessor
from siglip_nllb import model
from utils import *


def get_args_parser():
    parser = argparse.ArgumentParser(
        "AFRIMMD Pretraining script", add_help=True
    )
    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument("--epochs", default=15, type=int)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--warmup_steps", default=1000, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--wandb_logging", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        help="Path to the checkpoint to resume training"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where model checkpoints and logs will be saved",
    )
    parser.add_argument(
        "--lr", default=2.0e-5, type=float
    )  # tried 2e-4, 0.01 earlier,
    # * Optimizer parameters
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help="weight decay (default: 0.05)",
    )
    return parser


def train_one_epoch(
    args, model, train_data, epoch, optimizer, lr_scheduler):
    model.train()
    train_loss = 0
    epoch_metrics = {}

    num_steps = len(train_data)

    pbar = tqdm(enumerate(train_data), total=num_steps,
        desc=f"Epoch {epoch+1}/{args.epochs}")

    for _, batch in pbar:
        batch = {k: v.to(args.device)
            for k, v in batch.items()
            if k not in ["lang_code", "caption"]}
        optimizer.zero_grad()
        # Forward pass
        outputs = model(
            pixel_values=batch["pixel_values"],
            labels=batch["labels"])
        loss = outputs.loss
        # Backward pass
        loss.backward()
        train_loss += loss.item()

        optimizer.step()  # Update the model parameters
        lr_scheduler.step()

    # Average epoch metrics
    epoch_metrics["loss"] = train_loss / num_steps
    epoch_metrics["perplexity"] = torch.exp(
        torch.tensor(epoch_metrics["loss"])
    ).item()
    epoch_metrics["lr"] = lr_scheduler.get_last_lr()[0]
    # Logging
    if args.wandb_logging:
        wandb.log(epoch_metrics)
    return epoch_metrics


def evaluate(args, model, eval_dataloader):
    model.eval()
    val_loss = 0
    num_steps = len(eval_dataloader)
    with torch.no_grad():
        pbar = tqdm(eval_dataloader, desc="Evaluating")
        for batch in pbar:
            batch = {k: v if k in ["lang_code", "caption"] else v.to(args.device)
                for k, v in batch.items()}
            outputs = model(pixel_values=batch["pixel_values"],
                labels=batch["labels"],)
            loss = outputs.loss
            val_loss += loss.item()

    # Average metrics
    val_loss_avg = val_loss / num_steps
    perplexity = torch.exp(torch.tensor(val_loss_avg))
    return val_loss_avg, perplexity.item()


def main(args):
    # Check if data/Images directory exists
    if not os.path.exists("data/Images"):
        os.makedirs("data/Images")

        url = "https://drive.google.com/uc?id=1LnJICLTAOE3r-JE5kwvOCaT7HCpf6TqZ"
        zip_path = "data/images.zip"
        gdown.download(url, zip_path, quiet=False)

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall("data")

        os.remove(zip_path)
        if os.path.exists("data/__MACOSX"):
            shutil.rmtree("data/__MACOSX")

    if args.wandb_logging:
        wandb.init(
            project="afrimmd-pretraining",
            config=vars(args),
            dir=args.output_dir,
        )

    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = torch.device(args.device)
    output_dir = args.output_dir

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
    model.config.decoder_start_token_id = decoder.config.bos_token_id

    optimizer = optim.Adam(
        model.parameters(), args.lr, weight_decay=args.weight_decay
    )
    last_epoch = -1
    if args.resume:
        state_dict = torch.load(args.checkpoint_path, map_location=device)
        last_epoch = state_dict["epoch"]
        clean_state_dict = {}
        for key, value in state_dict["model_state_dict"].items():
            if key.startswith("_orig_mod."):
                clean_state_dict[key[len("_orig_mod."):]] = value
            else:
                clean_state_dict[key] = value
        model.load_state_dict(clean_state_dict)
        print(f"Model loaded from {args.checkpoint_path}")
        optimizer.load_state_dict(state_dict["optimizer_state_dict"])
        print(f"Optimizer loaded from {args.checkpoint_path}")
    
    lr_scheduler = get_inverse_sqrt_schedule(
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        last_epoch=last_epoch,
    )


    for parameters in model.encoder.parameters():
        parameters.requires_grad = False
    for parameters in model.decoder.parameters():
        parameters.requires_grad = False
    for parameters in model.lm_head.parameters():
        parameters.requires_grad = False
    for parameters in model.encoder.vision_model.head.mlp.parameters():
        parameters.requires_grad = True
    model.to(device)
    model = torch.compile(model)

    # Process dataset
    processor = DatasetProcessor()
    raw_data = load_dataset("AfriMM/AfriMMD")
    raw_data = raw_data["train"]
    processed_data = processor.process(raw_data)

    # Create dataloaders
    dataloaders = {
        "train": torch.utils.data.DataLoader(
            processed_data["train"],
            shuffle=True,
            batch_size=args.batch_size,
            collate_fn=processor.collate_batch,
        ),
        "val": torch.utils.data.DataLoader(
            processed_data["validation"],
            batch_size=args.batch_size,
            collate_fn=processor.collate_batch,
        ),
        "test": torch.utils.data.DataLoader(
            processed_data["test"],
            batch_size=args.batch_size,
            collate_fn=processor.collate_batch,
        ),
    }

    print(f"Starting SIGLIP and NLLB Pretraining on AFRIMMD dataset")
    print(f"Outputs will be saved to: {args.output_dir}")
    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        log_data = {}
        train_metrics = train_one_epoch(
            args,
            model,
            dataloaders["train"],
            epoch,
            optimizer,
            lr_scheduler,
        )

        val_loss_avg, perplexity = evaluate(
            args, model, dataloaders["val"]
        )

        print(f"\nEpoch {epoch+1} results:")
        print(f"Train metrics: {train_metrics}")
        print(f"Val metrics: {val_loss_avg, perplexity}")

        log_data["epoch"] = epoch + 1
        log_data["train_loss"] = train_metrics["loss"]
        log_data["train_perplexity"] = train_metrics["perplexity"]
        log_data["val_loss"] = val_loss_avg
        log_data["val_perplexity"] = perplexity
        log_data["train_lr"] = train_metrics["lr"]

        with open(os.path.join(output_dir, "log_file.txt"), "a") as f:
            f.write(json.dumps(log_data) + "\n")

        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            save_checkpoint(
                output_dir, model, optimizer, epoch, val_loss_avg, is_best=True
            )

        if epoch == args.epochs - 1:
            save_checkpoint(
                output_dir, model, optimizer, epoch, val_loss_avg, is_best=False
            )

    if args.eval:
        test_metrics = evaluate(args, model, dataloaders["test"], loss_fn)
        with open(os.path.join(args.output_dir, "test_metrics.txt"), "w") as f:
            json.dump(test_metrics, f, indent=4)


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    main(args)
