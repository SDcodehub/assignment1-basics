import os
import time
import math
import argparse
import numpy as np
import torch
import wandb

# Import your custom modules
from cs336_basics.nn import functional as F
from cs336_basics.nn.modules import TransformerLM
from cs336_basics.nn.optim import AdamW, get_lr_cosine_schedule, gradient_clipping
from cs336_basics.data_load import get_batch
from cs336_basics.utils import save_checkpoint, load_checkpoint

def parse_args():
    parser = argparse.ArgumentParser(description="Train a Transformer LM")
    
    # Data params
    parser.add_argument("--train_data", type=str, required=True, help="Path to training .npy file")
    parser.add_argument("--val_data", type=str, required=True, help="Path to validation .npy file")
    parser.add_argument("--out_dir", type=str, default="out", help="Directory to save checkpoints")
    
    # Model params (Defaults for TinyStories-ish size)
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--d_ff", type=int, default=2048) # 4 * d_model
    
    # Training params
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_iters", type=int, default=5000)
    parser.add_argument("--eval_interval", type=int, default=200)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--eval_iters", type=int, default=50, help="Num batches to average for eval")
    
    # Optimization params
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_iters", type=int, default=100)
    
    # Misc
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--wandb", action="store_true", help="Log to Weights & Biases")
    parser.add_argument("--wandb_project", type=str, default="cs336-assignment1")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")

    return parser.parse_args()


@torch.no_grad()
def estimate_loss(model, data, batch_size, context_length, device, eval_iters):
    """
    Estimates the loss on the dataset by averaging over a few random batches.
    Helps reduce noise in the validation metric.
    """
    out = {}
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        X, Y = get_batch(data, batch_size, context_length, device)
        logits = model(X)
        loss = F.cross_entropy(logits, Y)
        losses[k] = loss.item()
    model.train()
    return losses.mean()


def main():
    args = parse_args()
    
    # 1. Setup Device
    device = args.device
    print(f"Using device: {device}")

    # 2. Setup Logging
    if args.wandb:
        wandb.init(project=args.wandb_project, config=vars(args))
    
    os.makedirs(args.out_dir, exist_ok=True)

    # 3. Load Data (Memory Mapped)
    # We assume the data was saved as uint16 (standard for vocab < 65k)
    train_data = np.load(args.train_data, mmap_mode='r').astype(np.int64)
    val_data = np.load(args.val_data, mmap_mode='r').astype(np.int64)
    
    print(f"Loaded training data: {len(train_data)} tokens")
    print(f"Loaded validation data: {len(val_data)} tokens")

    # 4. Initialize Model
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        num_layers=args.num_layers,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        device=device
    )
    model.to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    # 5. Initialize Optimizer
    optimizer = AdamW(
        model.parameters(), 
        lr=args.lr, 
        betas=(args.beta1, args.beta2), 
        weight_decay=args.weight_decay
    )

    iter_num = 0
    if args.resume:
        print(f"Resuming from {args.resume}...")
        iter_num = load_checkpoint(args.resume, model, optimizer)

    # 6. Training Loop
    t0 = time.time()

    while iter_num < args.max_iters:

        # --- Validation & Checkpointing ---
        if iter_num % args.eval_interval == 0:
            val_loss = estimate_loss(model, val_data, args.batch_size, args.context_length, device, args.eval_iters)
            print(f"Step {iter_num}: val loss {val_loss:.4f}")
            
            if args.wandb:
                wandb.log({"val/loss": val_loss, "step": iter_num})
            
            # Save checkpoint
            ckpt_path = os.path.join(args.out_dir, f"ckpt_{iter_num:05d}.pt")
            save_checkpoint(model, optimizer, iter_num, ckpt_path)

            
        # --- Training Step ---
        
        # 1. Update Learning Rate
        lr = get_lr_cosine_schedule(iter_num, args.lr, 0.1 * args.lr, args.warmup_iters, args.max_iters)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # 2. Get Batch
        X, Y = get_batch(train_data, args.batch_size, args.context_length, device)


        # 3. Forward Pass
        logits = model(X)
        loss = F.cross_entropy(logits, Y)

        # 4. Backward Pass
        optimizer.zero_grad()
        loss.backward()

        # 5. Clip Gradients
        if args.grad_clip > 0.0:
            gradient_clipping(model.parameters(), args.grad_clip)

        # 6. Optimizer Step
        optimizer.step()

        # --- Logging ---
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        
        if iter_num % args.log_interval == 0:
            loss_f = loss.item()
            print(f"Step {iter_num}: train loss {loss_f:.4f} | lr {lr:.4e} | time {dt*1000:.2f}ms")
            if args.wandb:
                wandb.log({
                    "train/loss": loss_f,
                    "train/lr": lr,
                    "step": iter_num
                })
        
        iter_num += 1

    # Save final checkpoint
    save_checkpoint(model, optimizer, iter_num, os.path.join(args.out_dir, "ckpt_final.pt"))
    print("Training complete!")


if __name__ == "__main__":
    main()
