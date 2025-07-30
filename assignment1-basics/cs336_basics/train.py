import argparse
import numpy as np
import torch
import wandb

from cs336_basics.model import TransformerLM
from cs336_basics.training_utils import data_loading, cross_entropy, gradient_clipping, save_checkpoint, load_checkpoint
from cs336_basics.optimizer import AdamW
import os
import time
torch.autograd.set_detect_anomaly(True)

def train(args):
    wandb.init(project=args.wandb_project, config=vars(args))
    config = wandb.config
    
    train_data = np.load(config.train_data_path, mmap_mode="r")
    val_data = np.load(config.val_data_path, mmap_mode="r")

    model = TransformerLM(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        context_length=config.context_length,
        d_ff=config.d_ff,
        rope_theta=config.rope_theta,
    ).to(config.device)

    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay, eps=config.adamw_eps, betas=(config.beta_1, config.beta_2))
    
    start_step = 0
    if config.resume_from:
        start_step = load_checkpoint(config.resume_from, model, optimizer)
        print(f"✅ Resumed from checkpoint {config.resume_from} at step {start_step}")
        wandb.run.name = f"resumed-{os.path.basename(config.resume_from)}"
    
    loss_fn = cross_entropy
    start_time = time.time()

    for step in range(start_step, config.max_steps):
        model.train()
        x, y = data_loading(train_data, config.batch_size, config.context_length, config.device)

        logits = model(x)
        loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))

        optimizer.zero_grad()
        loss.backward()
        if config.grad_clip:
            gradient_clipping(model.parameters(), config.max_grad_norm, eps=config.grad_clip_eps)
        
        optimizer.step()

        if step % config.log_interval == 0:
            elapsed_time = time.time() - start_time
            print(f"[{step}] Train loss: {loss.item():.4f} | Time: {elapsed_time:.2f}s")
            wandb.log({
                "train_loss": loss.item(),
                "step": step,
                "wallclock_time": elapsed_time,
            })

        if step % config.eval_interval == 0 and step > 0:
            model.eval()
            with torch.no_grad():
                xb_val, yb_val = data_loading(val_data, config.batch_size, config.context_length, config.device)
                val_logits = model(xb_val)
                val_loss = loss_fn(val_logits.view(-1, val_logits.size(-1)), yb_val.view(-1))
                elapsed_time = time.time() - start_time
                print(f"[{step}] Validation loss: {val_loss.item():.4f} | Time: {elapsed_time:.2f}s")
                wandb.log({
                    "val_loss": val_loss.item(),
                    "step": step,
                    "wallclock_time": elapsed_time,
                })
                
        if config.ckpt_path and step % config.ckpt_interval == 0:
            ckpt_file = f"{config.ckpt_path}/step_{step}.pt"
            save_checkpoint(model, optimizer, step, ckpt_file)
            print(f"Saved checkpoint to {ckpt_file}")
            wandb.save(ckpt_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data_path", type=str, required=True)
    parser.add_argument("--val_data_path", type=str, required=True)
    
    parser.add_argument("--vocab_size", type=int, default=50000)
    parser.add_argument("--context_length", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--d_ff", type=int, default=1366)
    parser.add_argument("--rope_theta", type=int, default=10000)
    
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--beta_1", type=float, default=0.9)
    parser.add_argument("--beta_2", type=float, default=0.999)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--adamw_eps", type=float, default=1e-8)
    
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Max grad norm (None to disable)")
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--grad_clip_eps", type=float, default=1e-6)
    
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--eval_interval", type=int, default=500)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--ckpt_interval", type=int, default=1000)
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--resume_from", type=str, default=None, help="Path to checkpoint to resume from")
    
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--wandb_project", type=str, default="transformer-lm")

    args = parser.parse_args()
    train(args)


"""
uv run cs336_basics/train.py \
  --train_data_path ./results/tiny_sample_5M.memmap.npy \
  --val_data_path ./results/tiny_sample_5M.memmap.npy \
  --max_steps 4000 \
  --vocab_size 10000 \
  --context_length 128 \
  --batch_size 64 \
  --d_model 768 \
  --num_heads 12 \
  --num_layers 12 \
  --d_ff 3072 \
  --ckpt_path ./checkpoints \
  --wandb_project transformer-test

python train.py \
  --train_data_path ./data/tokens_train.npy \
  --val_data_path ./data/tokens_val.npy \
  --vocab_size 10000 \
  --context_length 128 \
  --batch_size 64 \
  --d_model 768 \
  --num_heads 12 \
  --num_layers 12 \
  --d_ff 3072 \
  --ckpt_path ./checkpoints \
  --wandb_project transformer-baseline

python train.py \
  --train_data_path ./data/tokens_train.npy \
  --val_data_path ./data/tokens_val.npy \
  --resume_from ./checkpoints/step_4000.pt \
  --ckpt_path ./checkpoints \
  --wandb_project transformer-resume
"""