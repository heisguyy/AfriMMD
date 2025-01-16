import wandb
import torch
from utils import *
import random
import numpy as np
from torch import nn
from typing import Dict
from tqdm.auto import tqdm
import argparse, json, os, csv
from datasets import load_dataset
from siglip_nllb import SiglipNllb
from dataset import DatasetProcessor
from transformers import AdamW, get_scheduler

def get_args_parser():
    parser = argparse.ArgumentParser('AFRIMMD Pretraining script', add_help=True)
    parser.add_argument('--batch-size', default=16, type=int)
    parser.add_argument('--epochs', default=5, type=int)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--warmup_steps', default=1000, type=int)
    parser.add_argument('--save_steps', default=1000, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--wandb_logging', action='store_true')
    parser.add_argument('--output_dir', type=str, required=True, 
                       help='Directory where model checkpoints and logs will be saved')
    parser.add_argument('--freeze_vision_encoder', action='store_true')
    parser.add_argument('--freeze_language_decoder', action='store_true')
    parser.add_argument('--lr', default=0.01, type=float) #tried 2e-4 earlier
       # * Optimizer parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1.0e-09, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1.0e-09)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: [0.9, 0.98], use opt default)')
    parser.add_argument('--clip-grad', type=float, default=1.0, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.0,
                        help='weight decay (default: 0.05)')
    return parser


def train_one_epoch(args, model, train_data, epoch, loss_scaler, optimizer, lr_scheduler):
    model.train()
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    # loss_fct = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX,label_smoothing=0.2)
    
    epoch_metrics = {
        'loss': 0,
        'perplexity': 0,
        'accuracy': 0,
        'vocab_usage': 0,
        'grad_norm': 0}
    num_steps = len(train_data)
    
    pbar = tqdm(enumerate(train_data), total=num_steps, 
                desc=f"Epoch {epoch+1}/{args.epochs}")
    
    for step, batch in pbar:
        batch = {k: v.to(args.device) for k, v in batch.items()}
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            # Forward pass
            outputs = model(batch)
            logits = outputs.view(-1, outputs.size(-1))
            targets = batch["input_ids"].view(-1)
            loss = loss_fn(logits, targets)
            
        # Backward pass
        loss_scaler._scaler.scale(loss).backward()
        
        # Compute training metrics
        step_metrics = compute_training_metrics(logits, targets, loss)
       
        loss_scaler(loss, optimizer, clip_grad=args.clip_grad, parameters=model.parameters())
    

        # Update metrics
        for k, v in step_metrics.items():
            epoch_metrics[k] += v
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{step_metrics['loss']:.4f}",
            'perp': f"{step_metrics['perplexity']:.4f}",
            'lr': f"{lr_scheduler.get_last_lr()[0]:.7f}"})
        
        # Logging
        if args.wandb_logging:
            wandb.log(step_metrics)
    
    # Average epoch metrics
    epoch_metrics = {k: v/num_steps for k, v in epoch_metrics.items()}
    return epoch_metrics, lr_scheduler.get_last_lr()[0]

def evaluate(args, model, eval_dataloader):
    model.eval()
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    eval_metrics = {
        'loss': 0,
        'perplexity': 0,
        'accuracy': 0,
        'vocab_usage': 0
    }
    num_steps = len(eval_dataloader)
    
    with torch.no_grad():
        pbar = tqdm(eval_dataloader, desc="Evaluating")
        for batch in pbar:
            batch = {k: v.to(args.device) for k, v in batch.items()}
            
            outputs = model(batch)
            logits = outputs.view(-1, outputs.size(-1))
            targets = batch["input_ids"].view(-1)
            
            loss = loss_fn(logits, targets)
            
            # Compute metrics
            step_metrics = compute_training_metrics(logits, targets, loss)
            
            # Update metrics
            for k, v in step_metrics.items():
                eval_metrics[k] += v
    
    # Average metrics
    eval_metrics = {k: v/num_steps for k, v in eval_metrics.items()}
    return eval_metrics

def main(args):
    
    if args.wandb_logging:
        wandb.init(project="afrimmd-pretraining", config=vars(args), dir=args.output_dir)
    
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    loss_scaler = NativeScaler()
    device = torch.device(args.device)
    output_dir = (args.output_dir)
    
    model = SiglipNllb()
    model.to(device)
    
    # Freeze layers based on args
    if args.freeze_vision_encoder:
        for param in model.vit.parameters():
            param.requires_grad = False
    if args.freeze_language_decoder:
        for param in model.lm.parameters():
            param.requires_grad = False
    model.connector.requires_grad = True
    
    # Process dataset
    processor = DatasetProcessor()
    raw_data = load_dataset("AfriMM/AfriMMD")
    raw_data = raw_data['train']
    processed_data = processor.process(raw_data)
    
    # Create dataloaders
    dataloaders = {
        'train': torch.utils.data.DataLoader(
            processed_data["train"],
            shuffle=True,
            batch_size=args.batch_size,
            collate_fn=processor.collate_batch
        ),
        'val': torch.utils.data.DataLoader(
            processed_data["validation"],
            batch_size=args.batch_size,
            collate_fn=processor.collate_batch
        ),
        'test': torch.utils.data.DataLoader(
            processed_data["test"],
            batch_size=args.batch_size,
            collate_fn=processor.collate_batch
        )
    }
    
    optimizer = AdamW(model.parameters(), args.lr, weight_decay=args.weight_decay)
    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.epochs * len(dataloaders['train'])
    )
    print(f"Starting SIGLIP and NLLB Pretraining on AFRIMMD dataset")
    print(f"Outputs will be saved to: {args.output_dir}")
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        train_metrics, _ = train_one_epoch(args, model, dataloaders['train'], epoch, loss_scaler, optimizer, lr_scheduler)
        
        val_metrics = evaluate(args, model, dataloaders['val'])
        log_combined_metrics(output_dir, epoch + 1, train_metrics, val_metrics, 
                           optimizer, model)
        
        print(f"\nEpoch {epoch+1} results:")
        print(f"Train metrics: {train_metrics}")
        print(f"Val metrics: {val_metrics}")
        
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            save_checkpoint(output_dir, model, optimizer, epoch, val_metrics, is_best=True)
        
        if epoch == args.epochs - 1:
            save_checkpoint(output_dir, model, optimizer, epoch, val_metrics, is_best=False)
    
    if args.eval:
        test_metrics = evaluate(args, model, dataloaders['test'])
        with open(os.path.join(args.output_dir, 'test_metrics.txt'), 'w') as f:
            json.dump(test_metrics, f, indent=4)

if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    main(args)