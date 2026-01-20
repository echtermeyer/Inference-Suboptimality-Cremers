from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import sys
import os
import argparse
from tqdm import tqdm
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from utils.hparams import HParams
from utils.helper import get_model, get_loaders


parser = argparse.ArgumentParser(
    description="Train encoder with frozen decoder from pretrained model"
)

# Model paths
parser.add_argument(
    "--pretrained-path",
    "-pp",
    type=str,
    required=True,
    help="path to pretrained model (decoder will be frozen from this)",
)
parser.add_argument(
    "--save-name",
    "-sn",
    type=str,
    default="encoder_frozen_dec.pth",
    help="name to save trained model",
)

# Architecture
parser.add_argument(
    "--dataset",
    "-d",
    type=str,
    default="mnist",
    choices=["mnist", "fashion", "cifar"],
    help="dataset (default: mnist)",
)
parser.add_argument(
    "--wide-encoder",
    "-we",
    action="store_true",
    help="use wide encoder (500 units vs 200)",
)
parser.add_argument("--z-size", "-zs", type=int, default=50)

# Training
parser.add_argument(
    "--epochs", "-e", type=int, default=5000, help="max epochs (default: 5000)"
)
parser.add_argument("--batch-size", "-bs", type=int, default=100)
parser.add_argument("--display-epoch", "-de", type=int, default=10)
parser.add_argument("--warmup", "-w", action="store_true", help="apply warmup")
parser.add_argument("--lr-schedule", "-lrs", action="store_true")
parser.add_argument(
    "--early-stopping",
    "-es",
    action="store_true",
    help="apply early stopping (paper's implementation)",
)
parser.add_argument(
    "--patience", type=int, default=10, help="patience for early stopping (default: 10)"
)
parser.add_argument("--no-cuda", "-nc", action="store_true")

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


def freeze_decoder(model):
    """Freeze all decoder parameters."""
    for name, param in model.named_parameters():
        if any(x in name for x in ["fc4", "fc5", "fc6"]):
            param.requires_grad = False
    print("Decoder frozen (fc4, fc5, fc6)")


def reinitialize_encoder(model):
    """Reinitialize encoder parameters with Xavier initialization."""
    for name, param in model.named_parameters():
        if any(x in name for x in ["fc1", "fc2", "fc3", "x_info"]):
            if "weight" in name:
                torch.nn.init.xavier_uniform_(param)
            elif "bias" in name:
                torch.nn.init.zeros_(param)
    print("Encoder reinitialized (fc1, fc2, fc3, x_info_layer)")


def count_parameters(model):
    """Count trainable vs frozen parameters."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    total = trainable + frozen

    print(f"\nParameter counts:")
    print(f"  Trainable: {trainable:,} ({trainable/total*100:.1f}%)")
    print(f"  Frozen:    {frozen:,} ({frozen/total*100:.1f}%)")
    print(f"  Total:     {total:,}")

    return trainable, frozen


def train_encoder_with_frozen_decoder(
    model,
    train_loader,
    test_loader,
    epochs=5000,
    display_epoch=10,
    lr_schedule=True,
    warmup=True,
    early_stopping=False,
    patience=10,
    save_path="checkpoints/",
    save_name="model.pth",
):
    """Train encoder while keeping decoder frozen. Uses early stopping exactly as in paper."""

    print("\n" + "=" * 80)
    print("TRAINING ENCODER WITH FROZEN DECODER")
    print("=" * 80)

    # Warmup schedule
    warmup_thres = 400.0 if warmup else None

    # Checkpoints to save
    checkpoints = [1] + list(range(0, epochs, display_epoch))[1:] + [epochs]

    # Optimizer (only for trainable parameters)
    if lr_schedule:
        current_lr = 1e-3
        pow_exp = 0
        epoch_elapsed = 0
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=current_lr,
            eps=1e-4,
        )
    else:
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, eps=1e-4
        )

    # Early stopping tracking (exactly as in paper)
    num_worse = 0
    prev_valid_err = None

    print(f"\nTraining for up to {epochs} epochs")
    print(f"Warmup: {warmup} (threshold: {warmup_thres})")
    print(f"LR schedule: {lr_schedule}")
    print(f"Early stopping: {early_stopping} (patience: {patience})")
    print(f"Save path: {save_path}")
    print("=" * 80 + "\n")

    time_ = time.time()

    for epoch in tqdm(range(1, epochs + 1)):
        warmup_const = min(1.0, epoch / warmup_thres) if warmup else 1.0

        # LR schedule from IWAE paper
        if lr_schedule:
            if epoch_elapsed >= 3**pow_exp:
                current_lr *= 10.0 ** (-1.0 / 7.0)
                pow_exp += 1
                epoch_elapsed = 0
                for param_group in optimizer.param_groups:
                    param_group["lr"] = current_lr
            epoch_elapsed += 1

        # Training
        model.train()
        for _, (batch, _) in enumerate(train_loader):
            batch = Variable(batch)
            if args.cuda:
                batch = batch.cuda()

            optimizer.zero_grad()
            elbo, _, _, _ = model.forward(batch, k=1, warmup_const=warmup_const)
            loss = -elbo
            loss.backward()
            optimizer.step()

        # Evaluation
        if epoch % display_epoch == 0:
            model.eval()

            train_stats, test_stats = [], []

            for _, (batch, _) in enumerate(train_loader):
                batch = Variable(batch)
                if args.cuda:
                    batch = batch.cuda()
                with torch.no_grad():
                    elbo, logpx, logpz, logqz = model(batch, k=1)
                train_stats.append(elbo.item())

            for _, (batch, _) in enumerate(test_loader):
                batch = Variable(batch)
                if args.cuda:
                    batch = batch.cuda()
                with torch.no_grad():
                    elbo, logpx, logpz, logqz = model(batch, k=1)
                test_stats.append(elbo.item())

            print(
                f"Epoch [{epoch}/{epochs}] "
                f"Train ELBO: {np.mean(train_stats):.4f} "
                f"Test ELBO: {np.mean(test_stats):.4f} "
                f"Time: {time.time() - time_:.2f}s"
            )
            time_ = time.time()

            # Early stopping logic (exactly from paper's main.py)
            if early_stopping:
                curr_valid_err = np.mean(test_stats)

                if prev_valid_err is None:  # don't have history yet
                    prev_valid_err = curr_valid_err
                elif curr_valid_err >= prev_valid_err:  # performance improved
                    prev_valid_err = curr_valid_err
                    num_worse = 0
                else:
                    num_worse += 1

                if num_worse >= patience:
                    print(f"\nEarly stopping at epoch {epoch}")
                    print(
                        f"No improvement for {patience} consecutive checks ({patience * display_epoch} epochs)"
                    )
                    break

        # Save checkpoints
        if epoch in checkpoints:
            checkpoint_path = os.path.join(save_path, f"{epoch}_{save_name}")
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                checkpoint_path,
            )
            if epoch == epochs or (early_stopping and num_worse >= patience):
                print(f"\n✓ Final model saved to {checkpoint_path}")


def main():
    """Main training routine."""

    print("\n" + "=" * 80)
    print("ENCODER TRAINING WITH FROZEN DECODER")
    print("=" * 80)
    print(f"Pretrained model: {args.pretrained_path}")
    print(f"Wide encoder: {args.wide_encoder}")
    print(f"Dataset: {args.dataset}")
    print("=" * 80 + "\n")

    # Load data
    train_loader, test_loader = get_loaders(
        dataset=args.dataset,
        evaluate=False,
        batch_size=args.batch_size,
    )

    # Create model with desired encoder architecture
    hps = HParams(
        z_size=args.z_size,
        act_func=F.elu,
        has_flow=False,
        n_flows=0,
        wide_encoder=args.wide_encoder,
        cuda=args.cuda,
        hamiltonian_flow=False,
    )

    model = get_model(args.dataset, hps)

    print(f"\nEncoder architecture:")
    print(f"  fc1: {model.fc1.in_features} → {model.fc1.out_features}")
    print(f"  fc2: {model.fc2.in_features} → {model.fc2.out_features}")

    # Load pretrained model (to get decoder weights)
    print(f"Loading pretrained model from {args.pretrained_path}...")
    pretrained_state = torch.load(args.pretrained_path, map_location="cpu")
    pretrained_weights = pretrained_state["state_dict"]

    # Copy decoder weights from pretrained model
    print("Copying decoder weights from pretrained model...")
    decoder_keys = [
        k
        for k in pretrained_weights.keys()
        if any(x in k for x in ["fc4", "fc5", "fc6"])
    ]

    model_state = model.state_dict()
    for key in decoder_keys:
        if key in model_state:
            model_state[key] = pretrained_weights[key]
            print(f"  Copied {key}: {pretrained_weights[key].shape}")

    model.load_state_dict(model_state)

    # Freeze decoder
    freeze_decoder(model)

    # Reinitialize encoder (fresh random weights)
    print("\nReinitializing encoder...")
    reinitialize_encoder(model)

    # Count parameters
    count_parameters(model)

    if args.cuda:
        model.cuda()

    # Create save directory
    encoder_type = "wide" if args.wide_encoder else "standard"
    save_path = f"checkpoints/{args.dataset}/frozen_decoder/{encoder_type}_encoder/"
    os.makedirs(save_path, exist_ok=True)

    # Train
    train_encoder_with_frozen_decoder(
        model,
        train_loader,
        test_loader,
        epochs=args.epochs,
        display_epoch=args.display_epoch,
        lr_schedule=args.lr_schedule,
        warmup=args.warmup,
        early_stopping=args.early_stopping,
        patience=args.patience,
        save_path=save_path,
        save_name=args.save_name,
    )

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Models saved in: {save_path}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
