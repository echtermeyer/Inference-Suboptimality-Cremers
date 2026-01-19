from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import sys
import os
from tqdm import tqdm
import argparse
import numpy as np
import pickle
from collections import defaultdict

import torch
import torch.utils.data
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from utils.math_ops import log_bernoulli, log_normal, log_mean_exp, safe_repeat
from utils.hparams import HParams
from utils.helper import get_model, get_loaders


parser = argparse.ArgumentParser(
    description="Experiment 1: Q* Optimization Reliability Analysis"
)
# action configuration flags
parser.add_argument("--no-cuda", "-nc", action="store_true")
parser.add_argument(
    "--debug", action="store_true", help="debug mode with verbose output"
)
parser.add_argument(
    "--pilot", action="store_true", help="pilot mode: 5 datapoints, R=3, K=3"
)

# model configuration flags
parser.add_argument("--z-size", "-zs", type=int, default=50)
parser.add_argument(
    "--batch-size", "-bs", type=int, default=100, help="batch size for evaluation"
)
parser.add_argument(
    "--eval-path",
    "-ep",
    type=str,
    default="model.pth",
    help="path to load evaluation ckpt (default: model.pth)",
)
parser.add_argument(
    "--dataset",
    "-d",
    type=str,
    default="mnist",
    choices=["mnist", "fashion", "cifar"],
    help="dataset to train and evaluate on (default: mnist)",
)
parser.add_argument(
    "--wide-encoder",
    "-we",
    action="store_true",
    help="use wider layer (more hidden units for FC, more channels for CIFAR)",
)

# experiment configuration
parser.add_argument(
    "--num-repetitions",
    "-R",
    type=int,
    default=10,
    help="number of repetitions (outer loop) for statistical analysis (default: 10)",
)
parser.add_argument(
    "--num-restarts",
    "-K",
    type=int,
    default=5,
    help="number of restarts (inner loop) per repetition (default: 5)",
)
parser.add_argument(
    "--k-iwae",
    "-ki",
    type=int,
    default=100,
    help="number of importance samples for IWAE evaluation (default: 100 from paper)",
)
parser.add_argument(
    "--num-datapoints",
    "-nd",
    type=int,
    default=100,
    help="number of test datapoints to evaluate (default: 100)",
)
parser.add_argument(
    "--output-dir",
    "-o",
    type=str,
    default="results/experiment1",
    help="directory to save results (default: results/experiment1/)",
)
parser.add_argument(
    "--check-every",
    type=int,
    default=100,
    help="check convergence every N epochs (default: 100 from paper)",
)
parser.add_argument(
    "--sentinel-thres",
    type=int,
    default=10,
    help="early stopping threshold (default: 10 from paper)",
)
parser.add_argument(
    "--init-std",
    type=float,
    default=0.1,
    help="std for perturbed initialization from prior (default: 0.1)",
)
parser.add_argument(
    "--seed",
    type=int,
    default=42,
    help="random seed for reproducibility (default: 42)",
)

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


def get_default_hparams():
    return HParams(
        z_size=args.z_size,
        act_func=F.elu,
        has_flow=False,  # Use FFG only for experiment 1
        n_flows=0,
        wide_encoder=args.wide_encoder,
        cuda=args.cuda,
        hamiltonian_flow=False,
    )


def optimize_local_gaussian_with_tracking(
    log_likelihood,
    model,
    data_var,
    init_mean=None,
    init_logvar=None,
    k=100,
    check_every=100,
    sentinel_thres=10,
    debug=False,
):
    """
    Optimize local factorized Gaussian posterior q*(z|x) per datapoint.
    Based on paper's Section 3.4 methodology with added convergence tracking.

    Args:
        log_likelihood: Likelihood function
        model: Trained VAE model (decoder frozen)
        data_var: Single datapoint [1, X]
        init_mean: Initial mean [B*k, z_size] or None (will sample from N(0, init_std^2))
        init_logvar: Initial logvar [B*k, z_size] or None (will sample from N(0, init_std^2))
        k: Number of MC samples for ELBO estimation (default: 100 from paper)
        check_every: Check convergence every N steps (default: 100 from paper)
        sentinel_thres: Early stopping threshold (default: 10 from paper)
        debug: Verbose output

    Returns:
        vae_elbo: Final VAE ELBO value
        iwae_elbo: Final IWAE bound value
        convergence_steps: Total number of optimization steps
        converged: Whether optimization converged (True) or hit max iterations (False)
        trajectory: List of ELBO values at each checkpoint
        init_mean: Initial mean parameters [z_size]
        init_logvar: Initial logvar parameters [z_size]
        final_mean: Final mean parameters [z_size]
        final_logvar: Final logvar parameters [z_size]
        final_mean_grad_norm: Final gradient norm for mean
        final_logvar_grad_norm: Final gradient norm for logvar
    """

    B = data_var.size()[0]
    z_size = model.z_size

    data_var = safe_repeat(data_var, k)
    zeros = Variable(torch.zeros(B * k, z_size).type(model.dtype))

    # Initialize mean and logvar (perturbed from prior)
    if init_mean is None:
        mean = Variable(
            torch.randn(B * k, z_size).type(model.dtype) * args.init_std,
            requires_grad=True,
        )
    else:
        mean = Variable(init_mean.clone(), requires_grad=True)

    if init_logvar is None:
        logvar = Variable(
            torch.randn(B * k, z_size).type(model.dtype) * args.init_std,
            requires_grad=True,
        )
    else:
        logvar = Variable(init_logvar.clone(), requires_grad=True)

    # Store initial parameters (before optimization)
    init_mean_np = mean[:z_size].data.cpu().numpy().copy()
    init_logvar_np = logvar[:z_size].data.cpu().numpy().copy()

    # Adam optimizer with lr=1e-3 (from paper Section 3.4)
    optimizer = optim.Adam([mean, logvar], lr=1e-3)
    best_avg, sentinel, prev_seq = 999999, 0, []
    trajectory = []  # Track ELBO at each checkpoint

    # Perform local optimization
    time_ = time.time()
    converged = False

    for epoch in range(1, 999999):
        # Reparameterization trick
        eps = Variable(torch.FloatTensor(mean.size()).normal_().type(model.dtype))
        z = eps.mul(logvar.mul(0.5).exp_()).add_(mean)
        x_logits = model.decode(z)

        # Compute ELBO components
        logpz = log_normal(z, zeros, zeros)
        logqz = log_normal(z, mean, logvar)
        logpx = log_likelihood(x_logits, data_var)

        # Optimize VAE ELBO (standard objective, not IWAE)
        optimizer.zero_grad()
        loss = -torch.mean(logpx + logpz - logqz)
        loss_np = loss.data.cpu().numpy()
        loss.backward()
        optimizer.step()

        prev_seq.append(loss_np)

        # Check convergence every `check_every` steps
        if epoch % check_every == 0:
            last_avg = np.mean(prev_seq)
            trajectory.append(-last_avg)  # Store negative loss (= ELBO)

            if debug:  # debugging helper
                sys.stderr.write(
                    "Epoch %d, time elapse %.4f, last avg %.4f, prev best %.4f\n"
                    % (epoch, time.time() - time_, -last_avg, -best_avg)
                )

            if last_avg < best_avg:
                sentinel, best_avg = 0, last_avg
            else:
                sentinel += 1

            # Early stopping based on sentinel
            if sentinel > sentinel_thres:
                convergence_steps = epoch
                converged = True
                break

            prev_seq = []
            time_ = time.time()
    else:
        # If we hit max iterations without breaking
        convergence_steps = 999999
        converged = False

    # Final evaluation with IWAE bound
    eps = Variable(torch.FloatTensor(B * k, z_size).normal_().type(model.dtype))
    z = eps.mul(logvar.mul(0.5).exp_()).add_(mean)

    logpz = log_normal(z, zeros, zeros)
    logqz = log_normal(z, mean, logvar)
    logpx = log_likelihood(model.decode(z), data_var)
    elbo = logpx + logpz - logqz

    vae_elbo = torch.mean(elbo)
    iwae_elbo = torch.mean(log_mean_exp(elbo.view(k, -1).transpose(0, 1)))

    # Compute final gradients for convergence diagnostics
    optimizer.zero_grad()
    final_loss = -vae_elbo
    final_loss.backward()

    final_mean_grad_norm = torch.norm(mean.grad).item()
    final_logvar_grad_norm = torch.norm(logvar.grad).item()

    # Extract final parameters (only for first item in batch)
    final_mean = mean[:z_size].data.cpu().numpy()
    final_logvar = logvar[:z_size].data.cpu().numpy()

    return (
        vae_elbo.item(),
        iwae_elbo.item(),
        convergence_steps,
        converged,
        trajectory,
        init_mean_np,
        init_logvar_np,
        final_mean,
        final_logvar,
        final_mean_grad_norm,
        final_logvar_grad_norm,
    )


def run_experiment():
    """Main experiment function implementing R repetitions × K restarts design."""

    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # Configure experiment based on mode
    if args.pilot:
        num_datapoints = 5
        num_repetitions = 3
        num_restarts = 3
        print("=" * 80)
        print("PILOT MODE: 5 datapoints, R=3 repetitions, K=3 restarts")
        print("=" * 80)
    else:
        num_datapoints = args.num_datapoints
        num_repetitions = args.num_repetitions
        num_restarts = args.num_restarts

    print("\nExperiment 1: Q* Optimization Reliability Analysis")
    print("=" * 80)
    print(f"Dataset: {args.dataset}")
    print(f"Model checkpoint: {args.eval_path}")
    print(f"Random seed: {args.seed}")
    print(f"Number of datapoints: {num_datapoints}")
    print(f"Repetitions (R): {num_repetitions}")
    print(f"Restarts per repetition (K): {num_restarts}")
    print(f"Total optimizations per datapoint: {num_repetitions * num_restarts}")
    print(f"IWAE samples (k): {args.k_iwae}")
    print(f"Initialization: μ, log σ ~ N(0, {args.init_std}²I)")
    print(f"Check every: {args.check_every} epochs")
    print(f"Sentinel threshold: {args.sentinel_thres}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 80 + "\n")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load data and model
    train_loader, test_loader = get_loaders(
        dataset=args.dataset, evaluate=True, batch_size=args.batch_size
    )
    model = get_model(args.dataset, get_default_hparams())
    model.load_state_dict(torch.load(args.eval_path)["state_dict"])
    model.eval()  # Freeze model (decoder frozen during local optimization)

    print(f"Model loaded successfully from {args.eval_path}")
    print(f"Z dimension: {model.z_size}")
    print(f"Using CUDA: {args.cuda}\n")

    # Storage for results
    # Structure: [num_datapoints, num_repetitions, num_restarts]
    results = {
        "vae_elbo": [],  # [N, R, K]
        "iwae_elbo": [],  # [N, R, K]
        "convergence_steps": [],  # [N, R, K]
        "converged_flags": [],  # [N, R, K] - bool
        "trajectories": [],  # [N, R, K, variable_length]
        "init_means": [],  # [N, R, K, z_size]
        "init_logvars": [],  # [N, R, K, z_size]
        "final_means": [],  # [N, R, K, z_size]
        "final_logvars": [],  # [N, R, K, z_size]
        "final_mean_grad_norms": [],  # [N, R, K]
        "final_logvar_grad_norms": [],  # [N, R, K]
        "datapoint_indices": [],  # [N] - global index in test set for reproducibility
        "metadata": {
            "dataset": args.dataset,
            "eval_path": args.eval_path,
            "random_seed": args.seed,
            "num_datapoints": num_datapoints,
            "num_repetitions": num_repetitions,
            "num_restarts": num_restarts,
            "k_iwae": args.k_iwae,
            "z_size": args.z_size,
            "check_every": args.check_every,
            "sentinel_thres": args.sentinel_thres,
            "init_std": args.init_std,
            "batch_size": args.batch_size,
            "pilot_mode": args.pilot,
        },
    }

    # Main experiment loop
    total_time = time.time()
    datapoint_count = 0

    for batch_idx, (batch, _) in enumerate(test_loader):
        if datapoint_count >= num_datapoints:
            break

        batch = Variable(batch.type(model.dtype))
        current_batch_size = batch.size(0)

        # Process each datapoint in the batch
        for i in range(current_batch_size):
            if datapoint_count >= num_datapoints:
                break

            datapoint = batch[i : i + 1]  # Keep batch dimension [1, X]
            datapoint_global_idx = (
                batch_idx * args.batch_size + i
            )  # Global index in test set
            datapoint_time = time.time()

            print(f"\n{'='*80}")
            print(
                f"Datapoint {datapoint_count + 1}/{num_datapoints} (Test Set Index: {datapoint_global_idx})"
            )
            print(f"{'='*80}")

            # Storage for this datapoint: [R, K]
            dp_vae_elbo = []
            dp_iwae_elbo = []
            dp_convergence_steps = []
            dp_converged_flags = []
            dp_trajectories = []
            dp_init_means = []
            dp_init_logvars = []
            dp_final_means = []
            dp_final_logvars = []
            dp_final_mean_grad_norms = []
            dp_final_logvar_grad_norms = []

            # Outer loop: R repetitions
            for rep_idx in range(num_repetitions):
                print(f"\nRepetition {rep_idx + 1}/{num_repetitions}")
                print(f"{'-'*80}")

                # Storage for this repetition: [K]
                rep_vae_elbo = []
                rep_iwae_elbo = []
                rep_convergence_steps = []
                rep_converged_flags = []
                rep_trajectories = []
                rep_init_means = []
                rep_init_logvars = []
                rep_final_means = []
                rep_final_logvars = []
                rep_final_mean_grad_norms = []
                rep_final_logvar_grad_norms = []

                # Inner loop: K restarts per repetition
                for restart_idx in range(num_restarts):
                    print(
                        f"  Restart {restart_idx + 1}/{num_restarts}...",
                        end=" ",
                        flush=True,
                    )
                    restart_time = time.time()

                    # Set random seed for reproducibility (different for each restart)
                    seed = datapoint_count * 1000 + rep_idx * 100 + restart_idx
                    torch.manual_seed(seed)
                    np.random.seed(seed)

                    # Run optimization with fresh perturbed initialization
                    (
                        vae_elbo,
                        iwae_elbo,
                        conv_steps,
                        converged,
                        trajectory,
                        init_mean,
                        init_logvar,
                        final_mean,
                        final_logvar,
                        mean_grad_norm,
                        logvar_grad_norm,
                    ) = optimize_local_gaussian_with_tracking(
                        log_bernoulli,
                        model,
                        datapoint,
                        init_mean=None,  # Will sample N(0, init_std^2)
                        init_logvar=None,  # Will sample N(0, init_std^2)
                        k=args.k_iwae,
                        check_every=args.check_every,
                        sentinel_thres=args.sentinel_thres,
                        debug=args.debug,
                    )

                    # Store results for this restart
                    rep_vae_elbo.append(vae_elbo)
                    rep_iwae_elbo.append(iwae_elbo)
                    rep_convergence_steps.append(conv_steps)
                    rep_converged_flags.append(converged)
                    rep_trajectories.append(trajectory)
                    rep_init_means.append(init_mean)
                    rep_init_logvars.append(init_logvar)
                    rep_final_means.append(final_mean)
                    rep_final_logvars.append(final_logvar)
                    rep_final_mean_grad_norms.append(mean_grad_norm)
                    rep_final_logvar_grad_norms.append(logvar_grad_norm)

                    conv_status = "✓" if converged else "✗ (max iter)"
                    print(
                        f"{conv_status} {time.time() - restart_time:.1f}s | "
                        f"IWAE: {iwae_elbo:.4f} | Steps: {conv_steps} | "
                        f"Grad: {mean_grad_norm:.6f}"
                    )

                # Compute statistics for this repetition
                max_iwae = np.max(rep_iwae_elbo)
                print(f"  → Repetition {rep_idx + 1} max IWAE: {max_iwae:.4f}")

                # Store repetition results
                dp_vae_elbo.append(rep_vae_elbo)
                dp_iwae_elbo.append(rep_iwae_elbo)
                dp_convergence_steps.append(rep_convergence_steps)
                dp_converged_flags.append(rep_converged_flags)
                dp_trajectories.append(rep_trajectories)
                dp_init_means.append(rep_init_means)
                dp_init_logvars.append(rep_init_logvars)
                dp_final_means.append(rep_final_means)
                dp_final_logvars.append(rep_final_logvars)
                dp_final_mean_grad_norms.append(rep_final_mean_grad_norms)
                dp_final_logvar_grad_norms.append(rep_final_logvar_grad_norms)

            # Compute statistics for this datapoint
            dp_iwae_array = np.array(dp_iwae_elbo)  # [R, K]
            maxima = np.max(dp_iwae_array, axis=1)  # [R] - max of each repetition
            all_values = dp_iwae_array.flatten()  # [R*K] - all values

            print(f"\n{'-'*80}")
            print(f"Statistics for datapoint {datapoint_count + 1}:")
            print(f"  Maxima (R={num_repetitions}):")
            print(f"    Mean: {np.mean(maxima):.4f}, Std: {np.std(maxima):.6f}")
            print(f"    Min: {np.min(maxima):.4f}, Max: {np.max(maxima):.4f}")
            print(f"    Range: {np.max(maxima) - np.min(maxima):.6f}")
            cv_max = np.std(maxima) / abs(np.mean(maxima))
            print(f"    CV: {cv_max:.6f} ({cv_max*100:.3f}%)")

            print(f"  All values (N={num_repetitions * num_restarts}):")
            print(f"    Mean: {np.mean(all_values):.4f}, Std: {np.std(all_values):.6f}")
            print(f"    Min: {np.min(all_values):.4f}, Max: {np.max(all_values):.4f}")
            cv_all = np.std(all_values) / abs(np.mean(all_values))
            print(f"    CV: {cv_all:.6f} ({cv_all*100:.3f}%)")

            # Convergence statistics
            converged_array = np.array(dp_converged_flags)
            convergence_rate = np.mean(converged_array) * 100
            print(
                f"  Convergence: {convergence_rate:.1f}% ({np.sum(converged_array)}/{converged_array.size})"
            )

            print(f"  Time: {time.time() - datapoint_time:.2f}s")
            print(f"{'-'*80}")

            # Store results for this datapoint
            results["vae_elbo"].append(dp_vae_elbo)
            results["iwae_elbo"].append(dp_iwae_elbo)
            results["convergence_steps"].append(dp_convergence_steps)
            results["converged_flags"].append(dp_converged_flags)
            results["trajectories"].append(dp_trajectories)
            results["init_means"].append(dp_init_means)
            results["init_logvars"].append(dp_init_logvars)
            results["final_means"].append(dp_final_means)
            results["final_logvars"].append(dp_final_logvars)
            results["final_mean_grad_norms"].append(dp_final_mean_grad_norms)
            results["final_logvar_grad_norms"].append(dp_final_logvar_grad_norms)
            results["datapoint_indices"].append(datapoint_global_idx)

            datapoint_count += 1

            # Save intermediate results every 10 datapoints
            if datapoint_count % 10 == 0:
                temp_path = os.path.join(
                    args.output_dir, f"intermediate_results_{datapoint_count}.pkl"
                )
                with open(temp_path, "wb") as f:
                    pickle.dump(results, f)
                print(f"\nIntermediate results saved to {temp_path}")

    # Convert lists to numpy arrays for easier analysis
    # Shape: [num_datapoints, num_repetitions, num_restarts]
    results["vae_elbo"] = np.array(results["vae_elbo"])
    results["iwae_elbo"] = np.array(results["iwae_elbo"])
    results["convergence_steps"] = np.array(results["convergence_steps"])
    results["converged_flags"] = np.array(results["converged_flags"])
    results["final_mean_grad_norms"] = np.array(results["final_mean_grad_norms"])
    results["final_logvar_grad_norms"] = np.array(results["final_logvar_grad_norms"])
    results["datapoint_indices"] = np.array(
        results["datapoint_indices"]
    )  # [num_datapoints]
    # Shape: [num_datapoints, num_repetitions, num_restarts, z_size]
    results["init_means"] = np.array(results["init_means"])
    results["init_logvars"] = np.array(results["init_logvars"])
    results["final_means"] = np.array(results["final_means"])
    results["final_logvars"] = np.array(results["final_logvars"])

    # Final summary statistics
    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE - OVERALL SUMMARY")
    print("=" * 80)
    print(f"Total datapoints processed: {datapoint_count}")
    print(f"Total optimizations: {datapoint_count * num_repetitions * num_restarts}")
    print(f"Total time: {time.time() - total_time:.2f}s")

    # Compute aggregate statistics
    iwae_maxima = np.max(results["iwae_elbo"], axis=2)  # [N, R] - max per repetition
    iwae_all = results["iwae_elbo"].reshape(-1)  # Flatten all values

    print(f"\nAggregate Statistics:")
    print(f"  IWAE ELBO (maxima per repetition):")
    print(f"    Mean: {np.mean(iwae_maxima):.4f}, Std: {np.std(iwae_maxima):.6f}")

    print(f"  IWAE ELBO (all values):")
    print(f"    Mean: {np.mean(iwae_all):.4f}, Std: {np.std(iwae_all):.6f}")

    # Per-datapoint statistics
    cv_maxima = []
    cv_all = []
    for i in range(datapoint_count):
        maxima = np.max(results["iwae_elbo"][i], axis=1)  # [R]
        all_vals = results["iwae_elbo"][i].flatten()  # [R*K]

        cv_max = np.std(maxima) / abs(np.mean(maxima))
        cv_a = np.std(all_vals) / abs(np.mean(all_vals))

        cv_maxima.append(cv_max)
        cv_all.append(cv_a)

    cv_maxima = np.array(cv_maxima)
    cv_all = np.array(cv_all)

    print(f"\n  Coefficient of Variation (CV):")
    print(
        f"    CV of maxima - Mean: {np.mean(cv_maxima):.6f}, Median: {np.median(cv_maxima):.6f}"
    )
    print(
        f"    CV of all values - Mean: {np.mean(cv_all):.6f}, Median: {np.median(cv_all):.6f}"
    )

    # Reliability assessment (CV > 0.03 = unreliable)
    unreliable_maxima = np.sum(cv_maxima > 0.03)
    unreliable_all = np.sum(cv_all > 0.03)
    print(f"\n  Reliability (CV > 3% threshold):")
    print(
        f"    Unreliable maxima: {unreliable_maxima}/{datapoint_count} ({unreliable_maxima/datapoint_count*100:.1f}%)"
    )
    print(
        f"    Unreliable overall: {unreliable_all}/{datapoint_count} ({unreliable_all/datapoint_count*100:.1f}%)"
    )

    # Convergence statistics
    convergence_rate = np.mean(results["converged_flags"]) * 100
    print(f"\n  Convergence rate: {convergence_rate:.1f}%")

    # Gradient norms (convergence diagnostic)
    mean_grad_norms = results["final_mean_grad_norms"].flatten()
    print(f"  Final gradient norms (mean param):")
    print(
        f"    Mean: {np.mean(mean_grad_norms):.6f}, Median: {np.median(mean_grad_norms):.6f}"
    )

    # Save final results
    output_path = os.path.join(args.output_dir, "experiment1_results.pkl")
    with open(output_path, "wb") as f:
        pickle.dump(results, f)
    print(f"\nFinal results saved to {output_path}")

    # Also save as numpy compressed format for easier loading
    npz_path = os.path.join(args.output_dir, "experiment1_results.npz")
    np.savez_compressed(
        npz_path,
        vae_elbo=results["vae_elbo"],
        iwae_elbo=results["iwae_elbo"],
        convergence_steps=results["convergence_steps"],
        converged_flags=results["converged_flags"],
        init_means=results["init_means"],
        init_logvars=results["init_logvars"],
        final_means=results["final_means"],
        final_logvars=results["final_logvars"],
        final_mean_grad_norms=results["final_mean_grad_norms"],
        final_logvar_grad_norms=results["final_logvar_grad_norms"],
        datapoint_indices=results["datapoint_indices"],
    )
    print(f"Numpy arrays saved to {npz_path}")

    print("=" * 80 + "\n")

    return results


if __name__ == "__main__":
    results = run_experiment()
