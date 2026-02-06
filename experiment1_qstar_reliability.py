# NOTE: This script is basically copy pasted from the original script with the changes needed for my experiment 1
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import sys
import os
import argparse
import numpy as np
import pickle
import multiprocessing as mp
from functools import partial

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

from utils.math_ops import log_bernoulli, log_normal, log_mean_exp, safe_repeat
from utils.hparams import HParams
from utils.helper import get_model, get_loaders


parser = argparse.ArgumentParser(
    description="Experiment 1: Q* Optimization Reliability Analysis"
)
parser.add_argument("--no-cuda", "-nc", action="store_true")
parser.add_argument(
    "--debug", action="store_true", help="debug mode with verbose output"
)
parser.add_argument(
    "--pilot", action="store_true", help="pilot mode: 5 datapoints, R=3, K=3"
)
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
parser.add_argument(
    "--num-repetitions",
    "-R",
    type=int,
    default=10,  # this is old and not really used anymore, but does not impact the experiments
    help="number of repetitions (outer loop) for statistical analysis",
)
parser.add_argument(
    "--num-restarts",
    "-K",
    type=int,
    default=5,  # this is old and not really used anymore, but does not impact the experiments
    help="number of restarts (inner loop) per repetition",
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
    help="std for perturbed initialization from prior. I only use 0.1 for my experiments",
)
parser.add_argument(
    "--seed",
    type=int,
    default=42,
    help="random seed for reproducibility (default: 42)",
)

# use parallel calls to make it faster (hehe)
parser.add_argument(
    "--num-workers",
    "-nw",
    type=int,
    default=4,
    help="number of parallel workers (default: 4)",
)

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


def get_default_hparams():
    return HParams(
        z_size=args.z_size,
        act_func=F.elu,
        has_flow=False,  # i only need FFG
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
    init_std=0.1,
    debug=False,
):
    B = data_var.size()[0]
    z_size = model.z_size

    data_var = safe_repeat(data_var, k)
    zeros_single = Variable(torch.zeros(1, z_size).type(model.dtype))

    if init_mean is None:
        mean = Variable(
            torch.randn(1, z_size).type(model.dtype) * init_std,
            requires_grad=True,
        )
    else:
        mean = Variable(init_mean.clone(), requires_grad=True)

    if init_logvar is None:
        logvar = Variable(
            torch.randn(1, z_size).type(model.dtype) * init_std,
            requires_grad=True,
        )
    else:
        logvar = Variable(init_logvar.clone(), requires_grad=True)

    # initial params
    init_mean_np = mean.data.cpu().numpy().copy()
    init_logvar_np = logvar.data.cpu().numpy().copy()

    optimizer = optim.Adam([mean, logvar], lr=1e-3)
    best_avg, sentinel, prev_seq = 999999, 0, []
    trajectory = []

    time_ = time.time()
    converged = False

    for epoch in range(1, 999999):
        mean_expanded = mean.expand(k, z_size)
        logvar_expanded = logvar.expand(k, z_size)
        zeros_expanded = zeros_single.expand(k, z_size)

        eps = Variable(torch.FloatTensor(k, z_size).normal_().type(model.dtype))
        z = eps.mul(logvar_expanded.mul(0.5).exp_()).add_(mean_expanded)

        x_logits = model.decode(z)  # [k, X]

        logpz = log_normal(z, zeros_expanded, zeros_expanded)  # [k]
        logqz = log_normal(z, mean_expanded, logvar_expanded)  # [k]
        logpx = log_likelihood(x_logits, data_var)  # [k]

        # optimize ELBO
        optimizer.zero_grad()
        loss = -torch.mean(logpx + logpz - logqz)
        loss_np = loss.data.cpu().numpy()
        loss.backward()
        optimizer.step()

        prev_seq.append(loss_np)

        # check if converrged
        if epoch % check_every == 0:
            last_avg = np.mean(prev_seq)
            trajectory.append(-last_avg)

            if debug:
                sys.stderr.write(
                    "Epoch %d, time elapse %.4f, last avg %.4f, prev best %.4f\n"
                    % (epoch, time.time() - time_, -last_avg, -best_avg)
                )

            if last_avg < best_avg:
                sentinel, best_avg = 0, last_avg
            else:
                sentinel += 1

            if sentinel > sentinel_thres:
                convergence_steps = epoch
                converged = True
                break

            prev_seq = []
            time_ = time.time()
    else:
        convergence_steps = 999999
        converged = False

    # final evaluation (but we do it again in analysis)
    mean_expanded = mean.expand(k, z_size)
    logvar_expanded = logvar.expand(k, z_size)
    zeros_expanded = zeros_single.expand(k, z_size)

    eps = Variable(torch.FloatTensor(k, z_size).normal_().type(model.dtype))
    z = eps.mul(logvar_expanded.mul(0.5).exp_()).add_(mean_expanded)

    logpz = log_normal(z, zeros_expanded, zeros_expanded)
    logqz = log_normal(z, mean_expanded, logvar_expanded)
    logpx = log_likelihood(model.decode(z), data_var)
    elbo = logpx + logpz - logqz

    vae_elbo = torch.mean(elbo)
    iwae_elbo = torch.mean(log_mean_exp(elbo.view(k, -1).transpose(0, 1)))

    optimizer.zero_grad()
    final_loss = -vae_elbo
    final_loss.backward()

    final_mean_grad_norm = torch.norm(mean.grad).item()
    final_logvar_grad_norm = torch.norm(logvar.grad).item()

    final_mean = mean.data.cpu().numpy()
    final_logvar = logvar.data.cpu().numpy()

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


def compute_amortized_elbo(model, data_var, k=100):
    B = data_var.size()[0]
    z_size = model.z_size

    data_var_expanded = safe_repeat(data_var, k)  # [k, X]
    zeros = Variable(torch.zeros(k, z_size).type(model.dtype))

    encode_output = model.encode(data_var_expanded)
    mean, logvar = encode_output[0], encode_output[1]

    eps = Variable(torch.FloatTensor(k, z_size).normal_().type(model.dtype))
    z = eps.mul(logvar.mul(0.5).exp_()).add_(mean)

    x_logits = model.decode(z)
    logpz = log_normal(z, zeros, zeros)
    logqz = log_normal(z, mean, logvar)
    logpx = log_bernoulli(x_logits, data_var_expanded)
    elbo = logpx + logpz - logqz

    vae_elbo = torch.mean(elbo)
    iwae_elbo = torch.mean(log_mean_exp(elbo.view(k, -1).transpose(0, 1)))

    return vae_elbo.item(), iwae_elbo.item()


def worker_optimize(
    task_data,
    model_state_dict,
    dataset,
    hparams_dict,
    k_iwae,
    check_every,
    sentinel_thres,
    init_std,
    debug,
):
    # function used for parallel optimization (speeds things up a lot, but sometimes CUDA pickle issues)
    datapoint_tensor, rep_idx, restart_idx, seed = task_data

    torch.manual_seed(seed)
    np.random.seed(seed)
    hparams = HParams(**hparams_dict)
    model = get_model(dataset, hparams)
    model.load_state_dict(model_state_dict)
    model.eval()

    datapoint = Variable(datapoint_tensor.type(model.dtype))

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
        init_mean=None,
        init_logvar=None,
        k=k_iwae,
        check_every=check_every,
        sentinel_thres=sentinel_thres,
        init_std=init_std,
        debug=debug,
    )

    results = {
        "vae_elbo": vae_elbo,
        "iwae_elbo": iwae_elbo,
        "convergence_steps": conv_steps,
        "converged": converged,
        "trajectory": trajectory,
        "init_mean": init_mean,
        "init_logvar": init_logvar,
        "final_mean": final_mean,
        "final_logvar": final_logvar,
        "mean_grad_norm": mean_grad_norm,
        "logvar_grad_norm": logvar_grad_norm,
    }

    return (rep_idx, restart_idx, results)


def run_experiment():
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    if args.pilot:
        num_datapoints = 2
        num_repetitions = 3
        num_restarts = 3
        print("=" * 80)
        print("pilot mode")
        print("=" * 80)
    else:
        num_datapoints = args.num_datapoints
        num_repetitions = args.num_repetitions
        num_restarts = args.num_restarts

    print("\experiment 1: q* optimization")
    # print the config so i can check it if necessary

    print("=" * 80)
    print(f"dataset: {args.dataset}")
    print(f"seed: {args.seed}")
    print(f"# datapoints: {num_datapoints}")
    print(f"# optimizations: {num_repetitions * num_restarts + 1} per datapoint")

    os.makedirs(args.output_dir, exist_ok=True)

    # we only use train data
    train_loader, test_loader = get_loaders(
        dataset=args.dataset, evaluate=True, batch_size=args.batch_size
    )
    model = get_model(args.dataset, get_default_hparams())
    model.load_state_dict(torch.load(args.eval_path)["state_dict"])
    model.eval()

    model_state_dict = model.state_dict()
    hparams_dict = {
        "z_size": args.z_size,
        "act_func": F.elu,
        "has_flow": False,
        "n_flows": 0,
        "wide_encoder": args.wide_encoder,
        "cuda": args.cuda,
        "hamiltonian_flow": False,
    }

    # try to save as much as possible (maybe we need it again later)
    results = {
        "vae_elbo": [],
        "iwae_elbo": [],
        "amortized_vae_elbo": [],
        "amortized_iwae_elbo": [],
        "baseline_vae_elbo": [],
        "baseline_iwae_elbo": [],
        "baseline_convergence_steps": [],
        "baseline_converged": [],
        "baseline_trajectory": [],
        "baseline_final_mean": [],
        "baseline_final_logvar": [],
        "baseline_mean_grad_norm": [],
        "baseline_logvar_grad_norm": [],
        "convergence_steps": [],
        "converged_flags": [],
        "trajectories": [],
        "init_means": [],
        "init_logvars": [],
        "final_means": [],
        "final_logvars": [],
        "final_mean_grad_norms": [],
        "final_logvar_grad_norms": [],
        "datapoint_indices": [],
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
            "num_workers": args.num_workers,
        },
    }

    pool = mp.Pool(processes=args.num_workers)
    total_time = time.time()
    datapoint_count = 0

    for batch_idx, (batch, _) in enumerate(train_loader):
        if datapoint_count >= num_datapoints:
            break

        batch = Variable(batch.type(model.dtype))
        current_batch_size = batch.size(0)

        for i in range(current_batch_size):
            if datapoint_count >= num_datapoints:
                break

            datapoint = batch[i : i + 1]
            datapoint_global_idx = batch_idx * args.batch_size + i
            datapoint_time = time.time()

            print(f"\n{'='*80}")
            print(f"Datapoint {datapoint_count + 1}/{num_datapoints}")
            print(f"{'='*80}")

            amortized_vae, amortized_iwae = compute_amortized_elbo(
                model, datapoint, k=args.k_iwae
            )
            print(f"Amortized ELBO: VAE={amortized_vae:.4f}")

            print(
                f"\nBaseline q* optimization...",
                end=" ",
                flush=True,
            )
            baseline_time = time.time()

            torch.manual_seed(datapoint_count * 1000)
            np.random.seed(datapoint_count * 1000)

            init_mean_zeros = torch.zeros(1, model.z_size).type(model.dtype)
            init_logvar_zeros = torch.zeros(1, model.z_size).type(model.dtype)

            (
                baseline_vae,
                baseline_iwae,
                baseline_conv_steps,
                baseline_converged,
                baseline_traj,
                _,
                _,
                baseline_final_mean,
                baseline_final_logvar,
                baseline_mean_grad,
                baseline_logvar_grad,
            ) = optimize_local_gaussian_with_tracking(
                log_bernoulli,
                model,
                datapoint,
                init_mean=init_mean_zeros,
                init_logvar=init_logvar_zeros,
                k=args.k_iwae,
                check_every=args.check_every,
                sentinel_thres=args.sentinel_thres,
                init_std=args.init_std,
                debug=args.debug,
            )

            tasks = []
            for rep_idx in range(num_repetitions):
                for restart_idx in range(num_restarts):
                    seed = datapoint_count * 1000 + rep_idx * 100 + restart_idx
                    datapoint_cpu = datapoint.data.cpu()
                    tasks.append((datapoint_cpu, rep_idx, restart_idx, seed))

            worker_func = partial(
                worker_optimize,
                model_state_dict=model_state_dict,
                dataset=args.dataset,
                hparams_dict=hparams_dict,
                k_iwae=args.k_iwae,
                check_every=args.check_every,
                sentinel_thres=args.sentinel_thres,
                init_std=args.init_std,
                debug=args.debug,
            )

            parallel_results = pool.map(worker_func, tasks)

            dp_vae_elbo = [[None] * num_restarts for _ in range(num_repetitions)]
            dp_iwae_elbo = [[None] * num_restarts for _ in range(num_repetitions)]
            dp_convergence_steps = [
                [None] * num_restarts for _ in range(num_repetitions)
            ]
            dp_converged_flags = [[None] * num_restarts for _ in range(num_repetitions)]
            dp_trajectories = [[None] * num_restarts for _ in range(num_repetitions)]
            dp_init_means = [[None] * num_restarts for _ in range(num_repetitions)]
            dp_init_logvars = [[None] * num_restarts for _ in range(num_repetitions)]
            dp_final_means = [[None] * num_restarts for _ in range(num_repetitions)]
            dp_final_logvars = [[None] * num_restarts for _ in range(num_repetitions)]
            dp_final_mean_grad_norms = [
                [None] * num_restarts for _ in range(num_repetitions)
            ]
            dp_final_logvar_grad_norms = [
                [None] * num_restarts for _ in range(num_repetitions)
            ]

            for rep_idx, restart_idx, result_dict in parallel_results:
                dp_vae_elbo[rep_idx][restart_idx] = result_dict["vae_elbo"]
                dp_iwae_elbo[rep_idx][restart_idx] = result_dict["iwae_elbo"]
                dp_convergence_steps[rep_idx][restart_idx] = result_dict[
                    "convergence_steps"
                ]
                dp_converged_flags[rep_idx][restart_idx] = result_dict["converged"]
                dp_trajectories[rep_idx][restart_idx] = result_dict["trajectory"]
                dp_init_means[rep_idx][restart_idx] = result_dict["init_mean"]
                dp_init_logvars[rep_idx][restart_idx] = result_dict["init_logvar"]
                dp_final_means[rep_idx][restart_idx] = result_dict["final_mean"]
                dp_final_logvars[rep_idx][restart_idx] = result_dict["final_logvar"]
                dp_final_mean_grad_norms[rep_idx][restart_idx] = result_dict[
                    "mean_grad_norm"
                ]
                dp_final_logvar_grad_norms[rep_idx][restart_idx] = result_dict[
                    "logvar_grad_norm"
                ]

            dp_iwae_array = np.array(dp_iwae_elbo)
            maxima = np.max(dp_iwae_array, axis=1)

            # get amortization gap (but we recalculate it again later anyways)
            q_star_best_perturbed = np.max(maxima)
            amor_gap_baseline = baseline_iwae - amortized_iwae
            amor_gap_perturbed_best = q_star_best_perturbed - amortized_iwae

            print(
                f"\nParallel execution completed in {time.time() - datapoint_time:.2f}s"
            )

            results["vae_elbo"].append(dp_vae_elbo)
            results["iwae_elbo"].append(dp_iwae_elbo)
            results["amortized_vae_elbo"].append(amortized_vae)
            results["amortized_iwae_elbo"].append(amortized_iwae)
            results["baseline_vae_elbo"].append(baseline_vae)
            results["baseline_iwae_elbo"].append(baseline_iwae)
            results["baseline_convergence_steps"].append(baseline_conv_steps)
            results["baseline_converged"].append(baseline_converged)
            results["baseline_trajectory"].append(baseline_traj)
            results["baseline_final_mean"].append(baseline_final_mean)
            results["baseline_final_logvar"].append(baseline_final_logvar)
            results["baseline_mean_grad_norm"].append(baseline_mean_grad)
            results["baseline_logvar_grad_norm"].append(baseline_logvar_grad)
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

            # some checkpointing
            if datapoint_count % 10 == 0:
                temp_path = os.path.join(
                    args.output_dir, f"intermediate_results_{datapoint_count}.pkl"
                )
                with open(temp_path, "wb") as f:
                    pickle.dump(results, f)
                print(f"\nIntermediate results saved to {temp_path}")

    pool.close()
    pool.join()

    results["vae_elbo"] = np.array(results["vae_elbo"])
    results["iwae_elbo"] = np.array(results["iwae_elbo"])
    results["amortized_vae_elbo"] = np.array(results["amortized_vae_elbo"])
    results["amortized_iwae_elbo"] = np.array(results["amortized_iwae_elbo"])
    results["baseline_vae_elbo"] = np.array(results["baseline_vae_elbo"])
    results["baseline_iwae_elbo"] = np.array(results["baseline_iwae_elbo"])
    results["baseline_convergence_steps"] = np.array(
        results["baseline_convergence_steps"]
    )
    results["baseline_converged"] = np.array(results["baseline_converged"])
    results["baseline_mean_grad_norm"] = np.array(results["baseline_mean_grad_norm"])
    results["baseline_logvar_grad_norm"] = np.array(
        results["baseline_logvar_grad_norm"]
    )
    results["baseline_final_mean"] = np.array(results["baseline_final_mean"])
    results["baseline_final_logvar"] = np.array(results["baseline_final_logvar"])
    results["convergence_steps"] = np.array(results["convergence_steps"])
    results["converged_flags"] = np.array(results["converged_flags"])
    results["final_mean_grad_norms"] = np.array(results["final_mean_grad_norms"])
    results["final_logvar_grad_norms"] = np.array(results["final_logvar_grad_norms"])
    results["datapoint_indices"] = np.array(results["datapoint_indices"])
    results["init_means"] = np.array(results["init_means"])
    results["init_logvars"] = np.array(results["init_logvars"])
    results["final_means"] = np.array(results["final_means"])
    results["final_logvars"] = np.array(results["final_logvars"])

    output_path = os.path.join(args.output_dir, "experiment1_results.pkl")
    with open(output_path, "wb") as f:
        pickle.dump(results, f)
    print(f"\nResults saved to {output_path}")

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
        amortized_vae_elbo=results["amortized_vae_elbo"],
        amortized_iwae_elbo=results["amortized_iwae_elbo"],
        baseline_vae_elbo=results["baseline_vae_elbo"],
        baseline_iwae_elbo=results["baseline_iwae_elbo"],
        baseline_convergence_steps=results["baseline_convergence_steps"],
        baseline_converged=results["baseline_converged"],
        baseline_final_mean=results["baseline_final_mean"],
        baseline_final_logvar=results["baseline_final_logvar"],
        baseline_mean_grad_norm=results["baseline_mean_grad_norm"],
        baseline_logvar_grad_norm=results["baseline_logvar_grad_norm"],
        datapoint_indices=results["datapoint_indices"],
    )

    return results


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    results = run_experiment()
