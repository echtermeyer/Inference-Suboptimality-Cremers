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
    for name, param in model.named_parameters():
        if any(x in name for x in ["fc4", "fc5", "fc6"]):
            param.requires_grad = False


def reinitialize_encoder(model):
    for name, param in model.named_parameters():
        if any(x in name for x in ["fc1", "fc2", "fc3", "x_info"]):
            if "weight" in name:
                torch.nn.init.xavier_uniform_(param)
            elif "bias" in name:
                torch.nn.init.zeros_(param)


def count_parameters(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"Count params: {trainable + frozen}")

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
    print("starting training...")

    warmup_thres = 400.0 if warmup else None
    checkpoints = [1] + list(range(0, epochs, display_epoch))[1:] + [epochs]

    if lr_schedule:
        current_lr = 1e-3
        pow_exp = 0
        epoch_elapsed = 0
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=current_lr,
        )
    else:
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, eps=1e-4
        )

    num_worse = 0
    prev_valid_err = None

    time_ = time.time()

    for epoch in tqdm(range(1, epochs + 1)):
        warmup_const = min(1.0, epoch / warmup_thres) if warmup else 1.0

        # lr schedule as in the paper
        if lr_schedule:
            if epoch_elapsed >= 3**pow_exp:
                current_lr *= 10.0 ** (-1.0 / 7.0)
                pow_exp += 1
                epoch_elapsed = 0
                for param_group in optimizer.param_groups:
                    param_group["lr"] = current_lr
            epoch_elapsed += 1

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

            if early_stopping:
                curr_valid_err = np.mean(test_stats)

                if prev_valid_err is None:
                    prev_valid_err = curr_valid_err
                elif curr_valid_err >= prev_valid_err:
                    prev_valid_err = curr_valid_err
                    num_worse = 0
                else:
                    num_worse += 1

                if num_worse >= patience:
                    print(f"\stopping at epoch {epoch}")
                    print(
                        f"missing improvement for {patience} consecutive checks ({patience * display_epoch} epochs)"
                    )
                    break

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
    train_loader, test_loader = get_loaders(
        dataset=args.dataset,
        evaluate=False,
        batch_size=args.batch_size,
    )

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

    print(f"  fc1: {model.fc1.in_features} -> {model.fc1.out_features}")
    print(f"  fc2: {model.fc2.in_features} -> {model.fc2.out_features}")

    pretrained_state = torch.load(args.pretrained_path, map_location="cpu")
    pretrained_weights = pretrained_state["state_dict"]

    # copy weights
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

    freeze_decoder(model)
    reinitialize_encoder(model)
    count_parameters(model)

    if args.cuda:
        model.cuda()

    encoder_type = "wide" if args.wide_encoder else "standard"
    save_path = f"checkpoints/{args.dataset}/frozen_decoder/{encoder_type}_encoder/"
    os.makedirs(save_path, exist_ok=True)

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

    print(f"Model saved in: {save_path}")


if __name__ == "__main__":
    main()
