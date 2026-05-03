# market-representation-lab

Experimental research sandbox for neural sequence models on NIFTY spot and NIFTY futures data.

This repo is intentionally separate from any production quant or execution system. It is for hidden feature extraction, latent market-state learning, pattern discovery, and learned signal research only.

## Project Aim

The goal is to learn useful latent representations of market state from numeric NIFTY spot and futures sequences. The workflow is:

- align spot and futures market data on a 1-minute exchange-session grid
- build deterministic numeric features for returns, basis, volatility, open interest, expiry, and session context
- train sequence encoders to predict short-horizon forward market statistics
- extract hidden states from trained models
- cluster hidden states into candidate market regimes
- run simple research-only signal tests from predictions or cluster assignments

This is best described as a market representation learning sandbox using LSTM baselines and lightweight Mamba-inspired numeric sequence blocks.

## Boundaries

- No live trading.
- No broker API integration.
- No pretrained Mamba language-model checkpoints.
- No tokenizing financial data as text.
- Mamba-style modules here are small numeric sequence blocks that consume tensors shaped `[batch, seq_len, num_features]`.
- The current Mamba2 and Mamba3 modules are inspired by Mamba-style sequence mixing, but they are not official `mamba-ssm` implementations.

## Mamba Faithfulness Note

The `Mamba2Encoder` and `Mamba3Encoder` classes in this repo are experimental, from-scratch PyTorch blocks for numeric time-series tensors. They use ideas such as gated projections, causal depthwise convolution, residual mixing, normalization, and multi-scale filters.

They do not currently implement the official Mamba-2 or Mamba-3 algorithms from `state-spaces/mamba`. In particular, the current blocks do not use the official selective scan or SSD kernels, `d_state`/`headdim` state parameterization, Mamba-3 SISO/MIMO recurrence, complex-valued state updates, fused CUDA/Triton kernels, pretrained language-model checkpoints, or token-generation code.

This is intentional for now: the first objective is to create a Windows-friendly research scaffold for market representation experiments. A future Linux/CUDA path can add an `OfficialMamba3Encoder` backed by `mamba_ssm.Mamba3` for closer architectural fidelity.

## Data Layout

Place files in `data/raw`:

- `nifty_spot_1m.csv`: `timestamp, open, high, low, close`
- `nifty_futures_1m.csv`: `timestamp, contract, expiry, open, high, low, close, volume, open_interest`
- `expiry_calendar.csv`
- `holiday_calendar.csv`

Timestamps are normalized to `Asia/Kolkata`. The data pipeline builds a 1-minute exchange session grid, removes holidays, aligns spot and futures by timestamp, and creates a deterministic near-month continuous futures series.

## Feature Set

Feature construction lives in `src/features/build_features.py` and currently produces:

- returns: `spot_ret_1m`, `spot_ret_5m`, `spot_ret_15m`, `fut_ret_1m`, `fut_ret_5m`, `fut_ret_15m`
- ranges and basis: `spot_range`, `fut_range`, `basis`, `basis_pct`, `basis_change`, `basis_zscore_60m`
- flow/positioning: `fut_volume_zscore_by_time_of_day`, `oi_change_1m`, `oi_change_15m`
- volatility: `realized_vol_15m`, `realized_vol_30m`
- calendar/session: `minutes_from_open`, `minutes_to_close`, `day_of_week`, `days_to_expiry`, `is_expiry_day`, `is_expiry_week`

Labels are built in `src/features/labels.py`:

- `future_ret_15m`
- `future_ret_30m`
- `future_realized_vol_30m`
- `future_range_30m`
- `future_basis_change_30m`
- `max_adverse_excursion_30m`
- `max_favorable_excursion_30m`

## Models

The first six encoders are implemented in `src/models/encoders.py`:

- `LSTMEncoder`
- `VSNLSTMEncoder`
- `Mamba2Encoder`
- `VSNMamba2Encoder`
- `Mamba3Encoder`
- `VSNMamba3Encoder`

All models include an input adapter from `num_features` to `d_model`. VSN variants use a feature-wise variable selection network and return `vsn_weights`.

Each forward pass returns:

- `hidden_state`
- `pred_return_15m`
- `pred_return_30m`
- `pred_vol_30m`
- `pred_range_30m`
- `pred_basis_change_30m`
- optional `vsn_weights`

## Install

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
```

## Train

Edit the relevant YAML config in `configs`, then run:

```bash
python -m src.training.train --config configs/lstm.yaml
python -m src.training.train --config configs/vsn_lstm.yaml
python -m src.training.train --config configs/mamba2.yaml
python -m src.training.train --config configs/vsn_mamba2.yaml
python -m src.training.train --config configs/mamba3.yaml
python -m src.training.train --config configs/vsn_mamba3.yaml
```

The default objective is a weighted multi-task Smooth L1 loss across return, volatility, range, and basis-change predictions.

## Evaluate Hidden States

Use `src/evaluation/hidden_states.py` to extract model hidden states for validation/test windows, then cluster with `src/evaluation/clustering.py`.

Cluster summaries include:

- count
- average future 15m return
- average future 30m return
- average future 30m range
- average future 30m volatility
- win rate
- max adverse excursion
- max favorable excursion

## Simple Signal Tests

`src/evaluation/trading.py` contains two intentionally simple research tests:

- pure model signal: long if `pred_return_30m > threshold`, short if `< -threshold`, flat otherwise
- cluster signal: choose long/short clusters using training data only, then apply those rules to validation/test

Metrics include total return, Sharpe, max drawdown, hit rate, average trade, turnover, and estimated cost impact.

## Tests

```bash
pytest
```

The test suite uses dummy tensors and synthetic market data. It does not require real NIFTY files.
