import torch, os, json
from typing import Dict
from torch import nn

def compute_training_metrics(logits: torch.Tensor, targets: torch.Tensor, loss: torch.Tensor) -> Dict[str, float]:
    """
    Compute pretraining-specific metrics
    """
    with torch.no_grad():
        perplexity = torch.exp(loss)
        
        # Vision-Language alignment score (cosine similarity between vision and language features)
        pred_probs = torch.softmax(logits, dim=-1)
        accuracy = (torch.argmax(pred_probs, dim=-1) == targets).float().mean()
        
        # Vocabulary usage statistics
        vocab_usage = torch.sum(pred_probs > 0.1, dim=-1).float().mean()
    return {
        'loss': loss.item(),
        'perplexity': perplexity.item(),
        'accuracy': accuracy.item(),
        'vocab_usage': vocab_usage.item()}

def log_combined_metrics(output_dir: str, epoch: int, 
                        train_metrics: Dict[str, float], 
                        test_metrics: Dict[str, float],
                        optimizer: torch.optim.Optimizer,
                        model: nn.Module,):
    """Log train and test metrics in JSON format"""
    
    current_lr = optimizer.param_groups[0]['lr']
    
    log_data = {
        "epoch": epoch,
        "train_lr": current_lr,
        "train_loss": train_metrics['loss'],
        "train_perplexity": train_metrics['perplexity'],
        "train_accuracy": train_metrics['accuracy'],
        "train_vocab_usage": train_metrics['vocab_usage'],
        "test_loss": test_metrics['loss'],
        "test_perplexity": test_metrics['perplexity'],
        "test_accuracy": test_metrics['accuracy'],
        "test_vocab_usage": test_metrics['vocab_usage'],
    }

    # Append log data as a JSON object
    with open(os.path.join(output_dir, "log_file.txt"), 'a') as f:
        f.write(json.dumps(log_data) + '\n')
        
def save_checkpoint(output_dir: str, model, optimizer, epoch: int, metrics: Dict[str, float], 
                   is_best: bool = False):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics}
    
    if is_best:
        save_path = os.path.join(output_dir, 'best_model.pt')
        print(f"\n=== Saving new best model at epoch {epoch+1} with validation loss: {metrics['loss']:.4f} ===")
    else:
        save_path = os.path.join(output_dir, f'checkpoint_epoch_{epoch+1}.pt')
        print(f"\nSaving checkpoint for epoch {epoch+1}")
        
    torch.save(checkpoint, save_path)
    
class NativeScaler:
    state_dict_key = "amp_scaler"

    def __init__(self, device='cuda'):
        try:
            self._scaler = torch.amp.GradScaler(device=device)
        except (AttributeError, TypeError) as e:
            self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, clip_mode='norm', parameters=None, create_graph=False, need_update=True):  
        # self._scaler.scale(loss).backward(create_graph=create_graph)
        #took out the loss.backward pass from here
        if need_update:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
                # dispatch_clip_grad(parameters, clip_grad, mode=clip_mode)
            self._scaler.step(optimizer)
            self._scaler.update()

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict) 
        