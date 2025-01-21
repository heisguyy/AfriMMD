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
    
# class NativeScaler:
#     state_dict_key = "amp_scaler"

#     def __init__(self, device='cuda'):
#         try:
#             self._scaler = torch.amp.GradScaler(device=device)
#         except (AttributeError, TypeError) as e:
#             self._scaler = torch.cuda.amp.GradScaler()

#     def __call__(self, loss, optimizer, clip_grad=None, clip_mode='norm', parameters=None, create_graph=False, need_update=True):  
#         # self._scaler.scale(loss).backward(create_graph=create_graph)
#         #took out the loss.backward pass from here
#         if need_update:
#             if clip_grad is not None:
#                 assert parameters is not None
#                 self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
#                 torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
#                 # dispatch_clip_grad(parameters, clip_grad, mode=clip_mode)
#             self._scaler.step(optimizer)
#             self._scaler.update()

#     def state_dict(self):
#         return self._scaler.state_dict()

#     def load_state_dict(self, state_dict):
#         self._scaler.load_state_dict(state_dict) 
        