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
from datasets import DatasetDict, load_dataset
from sacrebleu import corpus_bleu
from torch import nn, optim
from tqdm.auto import tqdm
from transformers import get_inverse_sqrt_schedule

from dataset import DatasetProcessor
from siglip_nllb import SiglipNllb, Tokenizer
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
    args, model, train_data, epoch, optimizer, loss_fn, lr_scheduler
):
    model.train()
    train_loss = 0
    # loss_fct = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX,label_smoothing=0.2)
    epoch_metrics = {}

    num_steps = len(train_data)

    pbar = tqdm(
        enumerate(train_data),
        total=num_steps,
        desc=f"Epoch {epoch+1}/{args.epochs}",
    )

    for steps, batch in pbar:
        batch = {
            k: v.to(args.device) for k, v in batch.items() if k != "lang_code"
        }
        optimizer.zero_grad()
        # Forward pass
        outputs = model(batch)
        logits = outputs.view(-1, outputs.size(-1))
        targets = batch["input_ids"].view(-1)
        loss = loss_fn(logits, targets)
        # Backward pass
        loss.backward()
        train_loss += loss.item()

        optimizer.step()  # Update the model parameters
        lr_scheduler.step()

    # Average epoch metrics
    epoch_metrics["loss"] = train_loss / num_steps
    epoch_metrics["perplexity"] = torch.exp(torch.tensor(epoch_metrics["loss"]))
    epoch_metrics["lr"] = lr_scheduler.get_last_lr()[0]
    # Logging
    if args.wandb_logging:
        wandb.log(epoch_metrics)
    return epoch_metrics


def evaluate(args, model, eval_dataloader, loss_fn):
    model.eval()
    val_loss = 0
    num_steps = len(eval_dataloader)
    with torch.no_grad():
        pbar = tqdm(eval_dataloader, desc="Evaluating")
        for batch in pbar:
            batch = {
                k: v.to(args.device) for k, v in batch.items()
                if k != "lang_code"
            }

            outputs = model(batch)
            logits = outputs.view(-1, outputs.size(-1))
            targets = batch["input_ids"].view(-1)

            loss = loss_fn(logits, targets)
            val_loss += loss.item()

    # Average metrics
    val_loss_avg = val_loss / num_steps
    perplexity = torch.exp(torch.tensor(val_loss_avg))
    return val_loss_avg, perplexity


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
    loss_fn = nn.CrossEntropyLoss(ignore_index=1)

    device = torch.device(args.device)
    output_dir = args.output_dir

    model = SiglipNllb()
    # for parameters in model.vit.parameters():
    #     parameters.requires_grad = False
    # for parameters in model.lm.parameters():
    #     parameters.requires_grad = False
    # for parameters in model.lm_head.parameters():
    #     parameters.requires_grad = False
    # for parameters in model.vit.head.mlp.parameters():
    #     parameters.requires_grad = True
    model.to(device)

    # Process dataset
    processor = DatasetProcessor()
    # raw_data = load_dataset("AfriMM/AfriMMD")
    # raw_data = raw_data["train"]
    # processed_data = processor.process(raw_data)
    # processed_data.save_to_disk("processed_data")
    processed_data = DatasetDict.load_from_disk("processed_data")

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

    optimizer = optim.Adam(
        model.parameters(), args.lr, weight_decay=args.weight_decay
    )
    lr_scheduler = get_inverse_sqrt_schedule(
        optimizer=optimizer, num_warmup_steps=args.warmup_steps, last_epoch=-1
    )

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
            loss_fn,
            lr_scheduler,
        )

        val_loss_avg, perplexity = evaluate(
            args, model, dataloaders["val"], loss_fn
        )

        print(f"\nEpoch {epoch+1} results:")
        print(f"Train metrics: {train_metrics}")
        print(f"Val metrics: {val_loss_avg, perplexity}")

        log_data["epoch"] = epoch + 1
        log_data["train_loss"] = train_metrics["loss"]
        log_data["train_perplexity"] = train_metrics["perplexity"].item()
        log_data["val_loss"] = val_loss_avg
        log_data["val_perplexity"] = perplexity.item()
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
        if epoch % 2 == 0:
            predicted_text = []
            references = []
            with torch.no_grad():
                pbar = tqdm(dataloaders["test"], desc="Evaluating")
                for batch in pbar:
                    batch = {
                        k: v if k == "lang_code" else v.to(args.device)
                        for k, v in batch.items()
                    }
                    outputs = model(batch)
                    pred_tokens = torch.argmax(outputs, dim=-1)
                    targets = batch["input_ids"].view(-1)
                    for pred, language_code, target in zip(
                        pred_tokens, batch["lang_code"], targets
                    ):
                        tokenizer = Tokenizer(language_code)
                        pred = pred[pred != tokenizer.text_tokenizer.pad_token_id]
                        target = target[target != tokenizer.text_tokenizer.pad_token_id]

                        pred_text = tokenizer.detokenize(pred, skip_special_tokens=True)
                        target_text = tokenizer.detokenize(target, skip_special_tokens=True)

                        predicted_text.append(pred_text)
                        references.append([target_text])
            bleu = corpus_bleu(predicted_text, references)
            print(f"BLEU score: {bleu.score}")


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
