from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import sys
import os
import argparse
import numpy as np
import pickle
from collections import defaultdict
from multiprocessing import Pool

import pandas as pd
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

from utils.math_ops import log_bernoulli, log_normal, log_mean_exp, safe_repeat
from utils.hparams import HParams
from utils.helper import get_model, get_loaders


parser = argparse.ArgumentParser(
    description="Experiment 2: Per-Digit Amortization Gap Analysis"
)
parser.add_argument("--no-cuda", "-nc", action="store_true")
parser.add_argument(
    "--debug", action="store_true", help="debug mode with verbose output"
)
parser.add_argument(
    "--pilot", action="store_true", help="pilot mode: 2 datapoints per digit"
)
parser.add_argument("--z-size", "-zs", type=int, default=50)
parser.add_argument(
    "--batch-size", "-bs", type=int, default=100, help="batch size for data loading"
)
parser.add_argument(
    "--standard-model-path",
    "-smp",
    type=str,
    required=True,
    help="path to standard encoder model checkpoint",
)
parser.add_argument(
    "--large-model-path",
    "-lmp",
    type=str,
    required=True,
    help="path to large encoder model checkpoint",
)
parser.add_argument(
    "--dataset",
    "-d",
    type=str,
    default="mnist",
    choices=["mnist", "fashion", "cifar"],
    help="dataset to evaluate on (default: mnist)",
)
parser.add_argument(
    "--k-iwae",
    "-ki",
    type=int,
    default=5000,
    help="number of importance samples for IWAE evaluation (default: 5000 from paper)",
)
parser.add_argument(
    "--k-mc",
    "-km",
    type=int,
    default=100,
    help="number of MC samples for q* optimization (default: 100 from paper). Final evaluation uses k-iwae.",
)
parser.add_argument(
    "--datapoints-per-digit",
    "-dpd",
    type=int,
    default=100,
    help="number of datapoints per digit class (default: 100)",
)
parser.add_argument(
    "--output-dir",
    "-o",
    type=str,
    default="results/experiment2",
    help="directory to save results (default: results/experiment2/)",
)
parser.add_argument(
    "--check-every",
    type=int,
    default=100,
    help="check convergence every N epochs for q* optimization (default: 100 from paper)",
)
parser.add_argument(
    "--sentinel-thres",
    type=int,
    default=10,
    help="early stopping threshold for q* optimization (default: 10 from paper)",
)
parser.add_argument(
    "--seed",
    type=int,
    default=42,
    help="random seed for reproducibility (default: 42)",
)
parser.add_argument(
    "--num-workers",
    "-nw",
    type=int,
    default=0,
    help="number of parallel workers (0 = auto-detect, 1 = sequential, >1 = parallel, default: 0)",
)

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


def get_default_hparams(wide_encoder=False):
    """Get default hyperparameters for model."""
    return HParams(
        z_size=args.z_size,
        act_func=F.elu,
        has_flow=False,
        n_flows=0,
        wide_encoder=wide_encoder,
        cuda=args.cuda,
        hamiltonian_flow=False,
    )


def optimize_q_star(
    log_likelihood,
    model,
    data_var,
    k_optimization=100,
    k_evaluation=5000,
    check_every=100,
    sentinel_thres=10,
    debug=False,
):
    B = data_var.size()[0]
    z_size = model.z_size

    data_var_opt = safe_repeat(data_var, k_optimization)
    zeros_opt = Variable(torch.zeros(B * k_optimization, z_size).type(model.dtype))

    mean = Variable(
        torch.zeros(B * k_optimization, z_size).type(model.dtype), requires_grad=True
    )
    logvar = Variable(
        torch.zeros(B * k_optimization, z_size).type(model.dtype), requires_grad=True
    )

    optimizer = optim.Adam([mean, logvar], lr=1e-3)
    best_avg, sentinel, prev_seq = 999999, 0, []
    converged = False
    time_ = time.time()

    for epoch in range(1, 999999):
        eps = Variable(torch.FloatTensor(mean.size()).normal_().type(model.dtype))
        z = eps.mul(logvar.mul(0.5).exp_()).add_(mean)
        x_logits = model.decode(z)

        logpz = log_normal(z, zeros_opt, zeros_opt)
        logqz = log_normal(z, mean, logvar)
        logpx = log_likelihood(x_logits, data_var_opt)

        optimizer.zero_grad()
        loss = -torch.mean(logpx + logpz - logqz)
        loss_np = loss.data.cpu().numpy()
        loss.backward()
        optimizer.step()

        prev_seq.append(loss_np)

        if epoch % check_every == 0:
            last_avg = np.mean(prev_seq)

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
                converged = True
                break

            prev_seq = []
            time_ = time.time()

    mean_opt = mean[0:1, :].data
    logvar_opt = logvar[0:1, :].data

    data_var_eval = safe_repeat(data_var, k_evaluation)
    zeros_eval = Variable(torch.zeros(B * k_evaluation, z_size).type(model.dtype))

    mean_eval = safe_repeat(Variable(mean_opt), k_evaluation)
    logvar_eval = safe_repeat(Variable(logvar_opt), k_evaluation)

    # k evaluation samples
    eps = Variable(
        torch.FloatTensor(B * k_evaluation, z_size).normal_().type(model.dtype)
    )
    z = eps.mul(logvar_eval.mul(0.5).exp_()).add_(mean_eval)

    logpz = log_normal(z, zeros_eval, zeros_eval)
    logqz = log_normal(z, mean_eval, logvar_eval)
    logpx = log_likelihood(model.decode(z), data_var_eval)
    elbo = logpx + logpz - logqz

    L_q_star = torch.mean(elbo).item()

    optimizer.zero_grad()

    eps_opt = Variable(torch.FloatTensor(mean.size()).normal_().type(model.dtype))
    z_opt = eps_opt.mul(logvar.mul(0.5).exp_()).add_(mean)
    logpz_opt = log_normal(z_opt, zeros_opt, zeros_opt)
    logqz_opt = log_normal(z_opt, mean, logvar)
    logpx_opt = log_likelihood(model.decode(z_opt), data_var_opt)
    elbo_opt = logpx_opt + logpz_opt - logqz_opt

    final_loss = -torch.mean(elbo_opt)
    final_loss.backward()
    final_grad_norm = torch.norm(mean.grad).item()

    return L_q_star, converged, final_grad_norm


def compute_iwae_bound(model, data_var, k=5000):
    B = data_var.size()[0]
    z_size = model.z_size

    encode_output = model.encode(data_var)
    if len(encode_output) == 3:
        mu, logvar, _ = encode_output
    else:
        mu, logvar = encode_output

    mu = safe_repeat(mu, k)
    logvar = safe_repeat(logvar, k)
    data_var_repeated = safe_repeat(data_var, k)
    zeros = Variable(torch.zeros(B * k, z_size).type(model.dtype))

    eps = Variable(torch.FloatTensor(B * k, z_size).normal_().type(model.dtype))
    z = eps.mul(logvar.mul(0.5).exp_()).add_(mu)

    logpz = log_normal(z, zeros, zeros)
    logqz = log_normal(z, mu, logvar)
    logpx = log_bernoulli(model.decode(z), data_var_repeated)

    elbo = logpx + logpz - logqz
    log_p_x = log_mean_exp(elbo.view(k, -1).transpose(0, 1))

    return log_p_x.item()


def compute_amortized_elbo(model, data_var, k=5000):
    B = data_var.size()[0]
    z_size = model.z_size

    encode_output = model.encode(data_var)
    if len(encode_output) == 3:
        mu, logvar, _ = encode_output
    else:
        mu, logvar = encode_output

    mu = safe_repeat(mu, k)
    logvar = safe_repeat(logvar, k)
    data_var_repeated = safe_repeat(data_var, k)
    zeros = Variable(torch.zeros(B * k, z_size).type(model.dtype))

    eps = Variable(torch.FloatTensor(B * k, z_size).normal_().type(model.dtype))
    z = eps.mul(logvar.mul(0.5).exp_()).add_(mu)

    logpz = log_normal(z, zeros, zeros)
    logqz = log_normal(z, mu, logvar)
    logpx = log_bernoulli(model.decode(z), data_var_repeated)

    elbo = logpx + logpz - logqz
    L_q_phi = torch.mean(elbo).item()

    return L_q_phi


def select_datapoints_by_digit(loader, datapoints_per_digit=100):
    selected_data = defaultdict(list)
    global_idx = 0

    for batch, labels in loader:
        batch_size = batch.size(0)

        for i in range(batch_size):
            digit = labels[i].item()

            if len(selected_data[digit]) < datapoints_per_digit:
                selected_data[digit].append((batch[i : i + 1], global_idx))

            global_idx += 1

            if all(len(selected_data[d]) >= datapoints_per_digit for d in range(10)):
                return selected_data

    return selected_data


def process_single_datapoint(args_tuple):

    (
        datapoint_np,
        global_index,
        digit,
        standard_state_dict,
        large_state_dict,
        dataset,
        global_args,
    ) = args_tuple

    hps_standard = HParams(
        z_size=global_args["z_size"],
        act_func=F.elu,
        has_flow=False,
        n_flows=0,
        wide_encoder=False,
        cuda=False,  # Force CPU for parallel workers
        hamiltonian_flow=False,
    )

    hps_large = HParams(
        z_size=global_args["z_size"],
        act_func=F.elu,
        has_flow=False,
        n_flows=0,
        wide_encoder=True,
        cuda=False,
        hamiltonian_flow=False,
    )

    standard_model = get_model(dataset, hps_standard)
    standard_model.load_state_dict(standard_state_dict)
    standard_model.eval()

    large_model = get_model(dataset, hps_large)
    large_model.load_state_dict(large_state_dict)
    large_model.eval()

    datapoint = Variable(torch.from_numpy(datapoint_np).type(standard_model.dtype))

    log_p_x = compute_iwae_bound(standard_model, datapoint, k=global_args["k_iwae"])
    L_q_star, converged, grad_norm = optimize_q_star(
        log_bernoulli,
        standard_model,
        datapoint,
        k_optimization=global_args["k_mc"],
        k_evaluation=global_args["k_iwae"],
        check_every=global_args["check_every"],
        sentinel_thres=global_args["sentinel_thres"],
        debug=False,
    )

    L_q_phi_standard = compute_amortized_elbo(
        standard_model, datapoint, k=global_args["k_iwae"]
    )
    L_q_phi_large = compute_amortized_elbo(
        large_model, datapoint, k=global_args["k_iwae"]
    )

    G_app = log_p_x - L_q_star

    G_amor_standard = L_q_star - L_q_phi_standard
    G_inf_standard = G_app + G_amor_standard

    G_amor_large = L_q_star - L_q_phi_large
    G_inf_large = G_app + G_amor_large

    return {
        "datapoint_idx": global_index,
        "digit_label": digit,
        "log_p_x": log_p_x,
        "L_q_star": L_q_star,
        "q_star_converged": converged,
        "q_star_final_grad_norm": grad_norm,
        "standard": {
            "L_q_phi": L_q_phi_standard,
            "G_app": G_app,
            "G_amor": G_amor_standard,
            "G_inf": G_inf_standard,
        },
        "large": {
            "L_q_phi": L_q_phi_large,
            "G_app": G_app,
            "G_amor": G_amor_large,
            "G_inf": G_inf_large,
        },
    }


def evaluate_both_models(
    standard_model, large_model, selected_data, num_classes=10, num_workers=1
):
    standard_model.eval()
    large_model.eval()

    results = {"standard": defaultdict(list), "large": defaultdict(list)}
    total_datapoints = sum(len(selected_data[d]) for d in range(num_classes))

    print(f"Evaluating both encoder models")

    global_args = {
        "z_size": args.z_size,
        "k_iwae": args.k_iwae,
        "k_mc": args.k_mc,
        "check_every": args.check_every,
        "sentinel_thres": args.sentinel_thres,
    }

    standard_state_dict = {k: v.cpu() for k, v in standard_model.state_dict().items()}
    large_state_dict = {k: v.cpu() for k, v in large_model.state_dict().items()}

    if num_workers > 1:
        tasks = []
        for digit in range(num_classes):
            for datapoint, global_index in selected_data[digit]:
                # Convert datapoint to numpy for pickling
                datapoint_np = datapoint.cpu().numpy()
                task = (
                    datapoint_np,
                    global_index,
                    digit,
                    standard_state_dict,
                    large_state_dict,
                    args.dataset,
                    global_args,
                )
                tasks.append(task)

        print(f"Processing {len(tasks)} datapoints in parallel...")
        experiment_start_time = time.time()

        with Pool(processes=num_workers) as pool:
            task_results = []
            for i, result in enumerate(pool.imap(process_single_datapoint, tasks)):
                task_results.append(result)

                if (i + 1) % 50 == 0 or (i + 1) == len(tasks):
                    elapsed = time.time() - experiment_start_time
                    rate = (i + 1) / elapsed
                    remaining = (len(tasks) - i - 1) / rate if rate > 0 else 0
                    print(
                        f"  Progress: {i+1}/{len(tasks)} ({(i+1)/len(tasks)*100:.1f}%) | "
                        f"Rate: {rate:.2f} points/s | "
                        f"Elapsed: {elapsed/60:.1f}m | "
                        f"Remaining: ~{remaining/60:.1f}m"
                    )

        for result in task_results:
            digit = result["digit_label"]

            results["standard"][digit].append(
                {
                    "datapoint_idx": result["datapoint_idx"],
                    "digit_label": digit,
                    "log_p_x": result["log_p_x"],
                    "L_q_star": result["L_q_star"],
                    "L_q_phi": result["standard"]["L_q_phi"],
                    "G_app": result["standard"]["G_app"],
                    "G_amor": result["standard"]["G_amor"],
                    "G_inf": result["standard"]["G_inf"],
                    "q_star_converged": result["q_star_converged"],
                    "q_star_final_grad_norm": result["q_star_final_grad_norm"],
                }
            )
            results["large"][digit].append(
                {
                    "datapoint_idx": result["datapoint_idx"],
                    "digit_label": digit,
                    "log_p_x": result["log_p_x"],
                    "L_q_star": result["L_q_star"],
                    "L_q_phi": result["large"]["L_q_phi"],
                    "G_app": result["large"]["G_app"],
                    "G_amor": result["large"]["G_amor"],
                    "G_inf": result["large"]["G_inf"],
                    "q_star_converged": result["q_star_converged"],
                    "q_star_final_grad_norm": result["q_star_final_grad_norm"],
                }
            )

    else:
        raise ValueError("please use parallel execution")

    return results


def run_experiment():
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    if args.pilot:
        datapoints_per_digit = 2
        print("Pilot mode")
    else:
        datapoints_per_digit = args.datapoints_per_digit

    print("\nExperiment 2: Per-Digit Amortization Gap Analysis")

    os.makedirs(args.output_dir, exist_ok=True)
    # test loader not used
    train_loader, test_loader = get_loaders(
        dataset=args.dataset,
        evaluate=False,
        batch_size=args.batch_size,
    )
    selected_data = select_datapoints_by_digit(train_loader, datapoints_per_digit)

    counts = {digit: len(selected_data[digit]) for digit in range(10)}
    min_count = min(counts.values())
    max_count = max(counts.values())

    if min_count < max_count:
        for digit in range(10):
            selected_data[digit] = selected_data[digit][:min_count]

        datapoints_per_digit = min_count

    standard_model = get_model(args.dataset, get_default_hparams(wide_encoder=False))
    standard_model.load_state_dict(torch.load(args.standard_model_path)["state_dict"])
    standard_model.eval()

    large_model = get_model(args.dataset, get_default_hparams(wide_encoder=True))
    large_model.load_state_dict(torch.load(args.large_model_path)["state_dict"])
    large_model.eval()

    num_workers = args.num_workers
    if num_workers <= 0:
        raise ValueError("please use >=1 workers")

    experiment_start_time = time.time()

    results = evaluate_both_models(
        standard_model,
        large_model,
        selected_data,
        num_classes=10,
        num_workers=num_workers,
    )

    results["metadata"] = {
        "dataset": args.dataset,
        "standard_model_path": args.standard_model_path,
        "large_model_path": args.large_model_path,
        "random_seed": args.seed,
        "datapoints_per_digit_requested": datapoints_per_digit,
        "datapoints_per_digit_actual": {d: len(selected_data[d]) for d in range(10)},
        "total_datapoints": sum(len(selected_data[d]) for d in range(10)),
        "k_iwae": args.k_iwae,
        "k_mc": args.k_mc,
        "z_size": args.z_size,
        "check_every": args.check_every,
        "sentinel_thres": args.sentinel_thres,
        "batch_size": args.batch_size,
        "pilot_mode": args.pilot,
        "num_workers": num_workers,
        "shared_q_star": True,
    }

    print(f"Total time: {time.time() - experiment_start_time:.2f}s\n")

    output_path = os.path.join(args.output_dir, "experiment2_results.pkl")
    with open(output_path, "wb") as f:
        pickle.dump(results, f)
    print(f"\nResults saved to {output_path}")

    rows = []
    for model_name in ["standard", "large"]:
        for digit in range(10):
            for result in results[model_name][digit]:
                row = result.copy()
                row["model"] = model_name
                rows.append(row)

    df = pd.DataFrame(rows)
    csv_path = os.path.join(args.output_dir, "experiment2_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"CSV saved to {csv_path}")

    return results


if __name__ == "__main__":
    results = run_experiment()
