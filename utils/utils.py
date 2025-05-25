import torch, os, json
from typing import Dict
from torch import nn

def save_checkpoint(output_dir: str, model, optimizer, epoch: int, metrics: float, 
                   is_best: bool = False):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': metrics}
    
    if is_best:
        save_path = os.path.join(output_dir, 'best_model.pth')
        print(f"\n=== Saving new best model at epoch {epoch+1} with validation loss: {metrics:.4f} ===")
    else:
        save_path = os.path.join(output_dir, f'checkpoint_epoch_{epoch+1}.pth')
        print(f"\nSaving checkpoint for epoch {epoch+1}")
        
    torch.save(checkpoint, save_path)