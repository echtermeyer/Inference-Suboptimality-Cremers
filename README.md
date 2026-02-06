# Inference Suboptimality in VAEs

This repository is adapted from [lxuechen/inference-suboptimality](https://github.com/lxuechen/inference-suboptimality) for a seminar thesis on the paper *Inference Suboptimality in Variational Autoencoders* by Cremer et al.

## Overview

This project investigates the amortization gap in VAEs through custom experiments building on the original codebase. The experiments focus on analyzing inference suboptimality and encoder capacity effects.

## Project Structure

- `run.py` - Main training and evaluation script for VAE models
- `cvae.py` - Convolutional VAE architecture implementation
- `experiment1_qstar_reliability.py` - Analysis of q* optimization convergence and reliability
- `experiment2_freezed_decoder_training.py` - Training encoders with frozen decoders
- `experiment2_amortization_gap.py` - Per-class amortization gap analysis
- `utils/` - Helper functions for AIS, HMC, math operations, etc.
- `checkpoints/` - Saved model checkpoints
- `results/` - Experiment outputs

## Dependencies

See `requirements.txt`

## Usage

### Training a VAE
```bash
python run.py --train --dataset mnist --warmup --lr-schedule
```

### Experiment 1: Q* Reliability Analysis

Pilot run (to verify setup):
```bash
python experiment1_qstar_reliability.py \
    --pilot \
    --dataset mnist \
    --eval-path checkpoints/mnist/warmup/ffg/3280_model.pth
```

Full run:
```bash
python experiment1_qstar_reliability.py \
    --dataset mnist \
    --eval-path checkpoints/mnist/warmup/ffg/3280_model.pth \
    --num-datapoints 50
```

### Experiment 2: Encoder Capacity & Amortization Gap

First, train encoders with frozen decoder:
```bash
python experiment2_freezed_decoder_training.py \
    --pretrained-path checkpoints/mnist/warmup/ffg/3280_model.pth \
    --dataset mnist \
    --epochs 5000 \
    --warmup --lr-schedule --early-stopping \
    --save-name standard_encoder.pth

python experiment2_freezed_decoder_training.py \
    --pretrained-path checkpoints/mnist/warmup/ffg/3280_model.pth \
    --dataset mnist \
    --wide-encoder \
    --epochs 5000 \
    --warmup --lr-schedule --early-stopping \
    --save-name large_encoder.pth
```

Then run the amortization gap analysis:
```bash
python experiment2_amortization_gap.py \
    -smp checkpoints/mnist/frozen_decoder/standard_encoder/790_standard_encoder.pth \
    -lmp checkpoints/mnist/frozen_decoder/wide_encoder/490_large_encoder.pth \
    -d mnist --pilot
```

