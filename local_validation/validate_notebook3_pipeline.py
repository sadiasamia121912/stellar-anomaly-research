"""
Local, CPU-only pre-flight validation of Notebook 3's VAE data pipeline.

Purpose: catch shape/logic bugs against REAL synced data (not synthetic) before
spending any GPU time on vast.ai. This deliberately duplicates the relevant
logic from the rebuilt `All Notebooks/Notebook_3.ipynb` cells 1-11 (data
loading through Dataset/DataLoader + one forward pass) rather than importing
it, since the notebook is meant to be a self-contained Colab artifact. If a
bug is found and fixed here, the same fix must be applied to the notebook.

This script does NOT run the tuning grid or full training (those need a GPU
per the resource memo's own timing estimates) -- it only proves the pipeline
runs end-to-end on real data with correct shapes, and reports a real
wall-clock cost per batch on this CPU as a sanity cross-check against the
resource memo's GPU estimates.

Usage: python local_validation/validate_notebook3_pipeline.py [--n-stars N]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

LOCAL_BASE = Path(os.environ.get("STELLAR_LOCAL_DATA", r"C:\Users\User\stellar-anomaly-data-local"))


def load_manifests():
    split_assignment = pd.read_csv(LOCAL_BASE / "splits" / "star_split_assignment.csv")
    split_assignment["TIC_ID_num"] = split_assignment["TIC_ID_num"].astype(str)
    injection_manifest = pd.read_csv(LOCAL_BASE / "injections" / "injection_manifest.csv")
    with open(LOCAL_BASE / "provenance" / "normalization_stats.json") as f:
        norm_stats = json.load(f)
    return split_assignment, injection_manifest, norm_stats


# --- injection function library (verbatim from Notebook 2 Cell 7 / Notebook 3 Cell 2) ---
TRANSIT_DURATION_DAYS = 0.125
TRANSIT_RAMP_FRAC = 0.10
FLARE_RISE_TAU_DAYS = 0.01
FLARE_DECAY_TAU_DAYS = 0.05


def inject_transit(time, flux, depth, t0, duration=TRANSIT_DURATION_DAYS, ramp_frac=TRANSIT_RAMP_FRAC):
    flux = flux.copy()
    ramp = duration * ramp_frac
    t_end = t0 + duration
    in_flat = (time >= t0 + ramp) & (time <= t_end - ramp)
    in_ingress = (time >= t0) & (time < t0 + ramp)
    in_egress = (time > t_end - ramp) & (time <= t_end)
    flux[in_flat] -= depth
    if ramp > 0:
        frac_in = (time[in_ingress] - t0) / ramp
        flux[in_ingress] -= depth * frac_in
        frac_out = (t_end - time[in_egress]) / ramp
        flux[in_egress] -= depth * frac_out
    return flux


def inject_flare(time, flux, amplitude, t_peak, rise_tau=FLARE_RISE_TAU_DAYS, decay_tau=FLARE_DECAY_TAU_DAYS):
    flux = flux.copy()
    dt = time - t_peak
    rise_mask = dt < 0
    decay_mask = dt >= 0
    flux[rise_mask] += amplitude * np.exp(dt[rise_mask] / rise_tau)
    flux[decay_mask] += amplitude * np.exp(-dt[decay_mask] / decay_tau)
    return flux


def inject_periodic(time, flux, amplitude, period_days, phase):
    flux = flux.copy()
    flux += amplitude * np.sin(2 * np.pi * time / period_days + phase)
    return flux


INJECTION_FUNCS = {"transit": inject_transit, "flare": inject_flare, "periodic": inject_periodic}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-stars", type=int, default=8, help="number of train-split stars to spot-check")
    args = parser.parse_args()

    print(f"Local data root: {LOCAL_BASE}")
    for required in ["splits/star_split_assignment.csv", "injections/injection_manifest.csv",
                      "provenance/normalization_stats.json"]:
        p = LOCAL_BASE / required
        if not p.exists():
            print(f"MISSING: {p} -- sync not complete yet, aborting.")
            return

    split_assignment, injection_manifest, norm_stats = load_manifests()
    GLOBAL_NORM_MEAN = norm_stats["global_norm_mean"]
    GLOBAL_NORM_STD = norm_stats["global_norm_std"]
    print(f"Loaded {len(split_assignment)} stars, {len(injection_manifest)} injection-manifest rows.")
    print(f"Normalization stats: mean={GLOBAL_NORM_MEAN:.6f}, std={GLOBAL_NORM_STD:.6f}, "
          f"fitted_on={norm_stats.get('fitted_on')}")

    processed_dir = LOCAL_BASE / "processed" / "general"
    if not processed_dir.exists():
        print(f"MISSING: {processed_dir} -- bulk data sync not complete yet, aborting.")
        return

    # --- pick a handful of train-split stars that actually have local .npz files ---
    train_ids = set(split_assignment.loc[split_assignment["split"] == "train", "TIC_ID_num"])
    available = {os.path.basename(p).replace("TIC", "").replace("_processed.npz", "")
                 for p in glob.glob(str(processed_dir / "*.npz"))}
    usable_train_ids = sorted(train_ids & available)
    print(f"Train-split stars with local .npz available: {len(usable_train_ids)} / {len(train_ids)}")
    if not usable_train_ids:
        print("No usable train-split .npz files synced yet -- aborting (sync still running?).")
        return

    sample_ids = usable_train_ids[: args.n_stars]

    # --- materialize_instance + binning, on real manifest rows for these stars ---
    def load_raw_star(tic_id):
        npz_path = processed_dir / f"TIC{tic_id}_processed.npz"
        with np.load(npz_path) as d:
            return d["time"].copy(), d["flux"].copy()

    def materialize_instance(row):
        tic_id = str(row["tic_id"])
        time, flux = load_raw_star(tic_id)
        if row["is_injected"]:
            params = json.loads(row["params_json"])
            inject_fn = INJECTION_FUNCS[row["anomaly_type"]]
            flux = inject_fn(time, flux, **params)
        norm_flux = (flux - GLOBAL_NORM_MEAN) / GLOBAL_NORM_STD
        return time, norm_flux

    rows = injection_manifest[
        injection_manifest["tic_id"].astype(str).isin(sample_ids) & (injection_manifest["split"] == "train")
    ]
    print(f"Manifest rows available for sampled stars: {len(rows)}")
    if len(rows) == 0:
        print("No manifest rows matched the sampled stars -- aborting.")
        return

    # windowing constants, fit on this same train-only sample (mirrors Notebook 3 Cell 5's
    # logic; the real notebook run fits this over the FULL train split, not just this
    # local spot-check sample -- this script only proves the mechanism works)
    durations = []
    for tic_id in sample_ids:
        t, _ = load_raw_star(tic_id)
        durations.append(t.max() - t.min())
    BIN_WIDTH_MINUTES = 10
    max_duration_days = max(durations)
    MAX_LENGTH = int(np.ceil(max_duration_days * 24 * 60 / BIN_WIDTH_MINUTES))
    print(f"Sample max duration: {max_duration_days:.2f} days -> MAX_LENGTH={MAX_LENGTH} bins "
          f"(spot-check only; real fit uses the full train split)")

    def bin_light_curve(time, flux, bin_width_minutes=BIN_WIDTH_MINUTES, max_length=MAX_LENGTH):
        t_start = time.min()
        bin_width_days = bin_width_minutes / (24 * 60)
        bin_idx = np.clip(np.floor((time - t_start) / bin_width_days).astype(int), 0, max_length - 1)
        binned_flux = np.zeros(max_length, dtype=np.float32)
        mask = np.zeros(max_length, dtype=np.float32)
        counts = np.zeros(max_length, dtype=np.int32)
        np.add.at(binned_flux, bin_idx, flux)
        np.add.at(counts, bin_idx, 1)
        valid = counts > 0
        binned_flux[valid] /= counts[valid]
        mask[valid] = 1.0
        return binned_flux, mask

    print("\nMaterializing + binning a few real manifest rows...")
    sample_rows = rows.head(4)
    fluxes, masks = [], []
    for _, r in sample_rows.iterrows():
        t0 = time.time()
        tt, nf = materialize_instance(r)
        bf, bm = bin_light_curve(tt, nf)
        fluxes.append(bf)
        masks.append(bm)
        print(f"  tic={r['tic_id']} injected={r['is_injected']} type={r['anomaly_type']} "
              f"-> valid_bins={int(bm.sum())}/{MAX_LENGTH} ({time.time()-t0:.3f}s)")

    # --- ConvVAE forward pass on real data, CPU ---
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class ConvVAE(nn.Module):
        def __init__(self, latent_channels=8, max_length=MAX_LENGTH):
            super().__init__()
            self.max_length = max_length
            enc_channels = [2, 16, 32, 64, 64, 128, 128, 128]
            enc_layers = []
            for i in range(len(enc_channels) - 1):
                enc_layers += [
                    nn.Conv1d(enc_channels[i], enc_channels[i + 1], kernel_size=5, stride=2, padding=2),
                    nn.GroupNorm(8, enc_channels[i + 1]),
                    nn.ReLU(inplace=True),
                ]
            self.encoder_conv = nn.Sequential(*enc_layers)
            self.conv_mu = nn.Conv1d(enc_channels[-1], latent_channels, kernel_size=1)
            self.conv_logvar = nn.Conv1d(enc_channels[-1], latent_channels, kernel_size=1)
            self.fc_decode = nn.Conv1d(latent_channels, enc_channels[-1], kernel_size=1)
            dec_channels = enc_channels[::-1]
            dec_layers = []
            for i in range(len(dec_channels) - 1):
                is_last = i == len(dec_channels) - 2
                dec_layers += [nn.ConvTranspose1d(dec_channels[i], dec_channels[i + 1], kernel_size=5,
                                                   stride=2, padding=2, output_padding=1)]
                if not is_last:
                    dec_layers += [nn.GroupNorm(8, dec_channels[i + 1]), nn.ReLU(inplace=True)]
            self.decoder_conv = nn.Sequential(*dec_layers)

        def encode(self, x):
            h = self.encoder_conv(x)
            return self.conv_mu(h), self.conv_logvar(h)

        def reparameterize(self, mu, logvar):
            std = torch.exp(0.5 * logvar)
            return mu + torch.randn_like(std) * std

        def decode(self, z):
            h = self.fc_decode(z)
            h = self.decoder_conv(h)
            recon_flux = h[:, 0:1, :]
            recon_flux = F.interpolate(recon_flux, size=self.max_length, mode="linear", align_corners=False)
            return recon_flux.squeeze(1)

        def forward(self, x):
            mu, logvar = self.encode(x)
            z = self.reparameterize(mu, logvar)
            return self.decode(z), mu, logvar

    print("\nRunning one forward pass on real, binned data (CPU)...")
    flux_batch = torch.tensor(np.stack(fluxes), dtype=torch.float32)
    mask_batch = torch.tensor(np.stack(masks), dtype=torch.float32)
    x_batch = torch.stack([flux_batch, mask_batch], dim=1)

    model = ConvVAE(latent_channels=8)
    t0 = time.time()
    with torch.no_grad():
        recon, mu, logvar = model(x_batch)
    elapsed = time.time() - t0

    print(f"  input shape:  {tuple(x_batch.shape)}")
    print(f"  recon shape:  {tuple(recon.shape)}")
    print(f"  mu shape:     {tuple(mu.shape)}")
    assert recon.shape == flux_batch.shape, "Reconstruction shape mismatch against real data!"
    print(f"  CPU forward pass: {elapsed:.3f}s for batch of {len(fluxes)} "
          f"(sequence length {MAX_LENGTH}) -- confirms the architecture is correct "
          f"end-to-end on real data; full training needs GPU per the resource memo.")
    print("\nPIPELINE VALIDATION: PASS")


if __name__ == "__main__":
    main()
