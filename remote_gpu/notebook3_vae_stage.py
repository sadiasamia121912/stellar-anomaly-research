"""
Notebook 3 (VAE stage) ported to a plain script for the vast.ai GPU box.

Faithful, line-for-line port of All Notebooks/Notebook_3.ipynb cells 1-18, with
ONLY these changes:
  - No google.colab drive mount / no lightkurve+astroquery pip install (both are
    Colab-environment artifacts; nothing in this notebook's actual cells uses them).
  - DRIVE_BASE points at the local uploaded-data directory on this box instead of
    a Drive mount path.
  - Print statements get flushed immediately (stdout buffering) so `nohup`/tee
    logs show live progress during a long training run.
  - Wrapped in --stage {trials,select,train,gate,all} so it can be resumed cell-
    group-by-cell-group if the box or connection drops mid-run.

All constants, formulas, gate thresholds, and file paths (relative to DRIVE_BASE)
are copied verbatim from the notebook -- this is not a re-implementation.
"""
import argparse
import functools
print = functools.partial(print, flush=True)

import numpy as np, pandas as pd, json, os, glob, time
from pathlib import Path

DRIVE_BASE = Path(os.environ.get("NB3_DATA_DIR", "/workspace/nb3_data"))
SEED = 42
S2_ID = '261136679'
S8_ID = '149603524'


def load_common():
    split_assignment = pd.read_csv(DRIVE_BASE / 'splits' / 'star_split_assignment.csv')
    split_assignment['TIC_ID_num'] = split_assignment['TIC_ID_num'].astype(str)
    vetted_pool = pd.read_csv(DRIVE_BASE / 'vetting' / 'vetted_pool.csv')
    reflib_manifest = pd.read_csv(DRIVE_BASE / 'reference_library' / 'reference_library_FINAL_manifest.csv')
    injection_manifest = pd.read_csv(DRIVE_BASE / 'injections' / 'injection_manifest.csv')

    with open(DRIVE_BASE / 'provenance' / 'gate_check_passed.json') as f:
        gate_check = json.load(f)
    assert gate_check['status'] == 'PASS', "Gate check failed -- stop and investigate"

    with open(DRIVE_BASE / 'provenance' / 'normalization_stats.json') as f:
        norm_stats = json.load(f)
    global_norm_mean = norm_stats['global_norm_mean']
    global_norm_std = norm_stats['global_norm_std']
    assert norm_stats['fitted_on'] == 'general_pool TRAIN split only (pre-injection)', \
        "Normalization stats were not fit train-only per R3 -- stop and investigate."

    print(f"Splits loaded: train={sum(split_assignment.split=='train')}, "
          f"val={sum(split_assignment.split=='val')}, "
          f"test={sum(split_assignment.split=='test')}")
    print(f"Injection manifest: {len(injection_manifest)} rows")
    print(f"Normalization stats (loaded, train-only fit): mean={global_norm_mean:.6f}, std={global_norm_std:.6f}")
    print("Gate check: PASS")

    return split_assignment, vetted_pool, reflib_manifest, injection_manifest, global_norm_mean, global_norm_std


# ============================================================
# CELL 2: Injection function library (verbatim from Notebook 2, Cell 7)
# ============================================================
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

_raw_cache = {}


def load_raw_star(tic_id):
    if tic_id not in _raw_cache:
        npz_path = DRIVE_BASE / 'processed' / 'general' / f'TIC{tic_id}_processed.npz'
        with np.load(npz_path) as d:
            _raw_cache[tic_id] = (d['time'].copy(), d['flux'].copy())
    return _raw_cache[tic_id]


def make_materialize_instance(global_norm_mean, global_norm_std):
    def materialize_instance(row):
        tic_id = str(row['tic_id'])
        time, flux = load_raw_star(tic_id)
        if row['is_injected']:
            params = json.loads(row['params_json'])
            inject_fn = INJECTION_FUNCS[row['anomaly_type']]
            flux = inject_fn(time, flux, **params)
        norm_flux = (flux - global_norm_mean) / global_norm_std
        return time, norm_flux
    return materialize_instance


# ============================================================
# CELL 5: Lock windowing constants from TRAIN-split data only
# ============================================================
def scan_durations(folder, label, tic_filter=None):
    rows = []
    for path in glob.glob(str(DRIVE_BASE / folder / '*.npz')):
        fname = os.path.basename(path)
        tic_id = fname.replace('TIC', '').replace('_processed.npz', '')
        if tic_filter is not None and tic_id not in tic_filter:
            continue
        with np.load(path) as d:
            t = d['time']
            rows.append({'tic_id': tic_id, 'duration_days': float(t.max() - t.min()), 'n_points': len(t)})
    df = pd.DataFrame(rows)
    df['source'] = label
    return df


def compute_windowing(split_assignment):
    train_tic_ids = set(split_assignment.loc[split_assignment['split'] == 'train', 'TIC_ID_num'])
    train_stats = scan_durations('processed/general', 'general_train_only', tic_filter=train_tic_ids)

    print(train_stats[['duration_days', 'n_points']].describe().T)
    print(f"\nTrain-split max duration: {train_stats['duration_days'].max():.2f} days "
          f"(over {len(train_stats)} train stars)")

    bin_width_minutes = 10
    max_duration_days = float(train_stats['duration_days'].max())
    max_length = int(np.ceil(max_duration_days * 24 * 60 / bin_width_minutes))

    print(f"BIN_WIDTH_MINUTES = {bin_width_minutes}")
    print(f"MAX_DURATION_DAYS = {max_duration_days:.2f} (train-split only)")
    print(f"MAX_LENGTH = {max_length} bins")

    # Diagnostic-only reference-library duration scan: skipped here because
    # processed/reference wasn't uploaded to this box (unused dead-code path in
    # the original notebook -- reflib_stats is computed but never printed or fed
    # into MAX_LENGTH or any saved provenance file there either). See session
    # notes: this doesn't change any saved output.
    reflib_dir_present = (DRIVE_BASE / 'processed' / 'reference').exists()
    if reflib_dir_present:
        scan_durations('processed/reference', 'reference')

    all_gen_stats = scan_durations('processed/general', 'general_all')
    n_exceeding = int((all_gen_stats['duration_days'] > max_duration_days).sum())
    if n_exceeding:
        print(f"NOTE: {n_exceeding} non-train star(s) exceed the train-fit max duration and "
              f"will be truncated to MAX_LENGTH bins at inference time -- this is the "
              f"correct, leakage-free behavior (val/test never influences MAX_LENGTH).")

    windowing_config = {
        "bin_width_minutes": bin_width_minutes,
        "max_length_bins": max_length,
        "based_on": "TRAIN split only (fixed: was general pool incl. val/test)",
        "max_duration_days_train_only": max_duration_days,
        "n_train_stars_used_for_fit": len(train_stats),
        "n_non_train_stars_exceeding_max": n_exceeding,
        "aggregation": "mean",
        "gap_policy": "masked, never interpolated (consistent with R3)",
        "reference_library_note": "Reference library curves (up to ~2900 days, CVZ stars) "
                                   "are NOT padded to this length; they pass through the "
                                   "same conv+global-pooling model at inference time using "
                                   "their own native length (truncated to MAX_LENGTH if "
                                   "longer), since pooling makes the architecture length-agnostic.",
    }
    with open(DRIVE_BASE / 'provenance' / 'windowing_config.json', 'w') as f:
        json.dump(windowing_config, f, indent=2)
    print("\nLogged to provenance/windowing_config.json")
    return bin_width_minutes, max_length


# ============================================================
# CELL 6: Fixed-width binning
# ============================================================
def bin_light_curve(time, flux, bin_width_minutes, max_length):
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


# ============================================================
# CELL 8: ConvVAE architecture (final -- spatial/conv latent)
# ============================================================
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvVAE(nn.Module):
    def __init__(self, latent_channels, max_length):
        super().__init__()
        self.latent_channels = latent_channels
        self.max_length = max_length
        enc_channels = [2, 16, 32, 64, 64, 128, 128, 128]
        enc_layers = []
        for i in range(len(enc_channels) - 1):
            enc_layers += [
                nn.Conv1d(enc_channels[i], enc_channels[i+1], kernel_size=5, stride=2, padding=2),
                nn.GroupNorm(8, enc_channels[i+1]),
                nn.ReLU(inplace=True),
            ]
        self.encoder_conv = nn.Sequential(*enc_layers)
        self.bottleneck_len = max_length // (2 ** (len(enc_channels) - 1))
        self.conv_mu = nn.Conv1d(enc_channels[-1], latent_channels, kernel_size=1)
        self.conv_logvar = nn.Conv1d(enc_channels[-1], latent_channels, kernel_size=1)
        self.fc_decode = nn.Conv1d(latent_channels, enc_channels[-1], kernel_size=1)

        dec_channels = enc_channels[::-1]
        dec_layers = []
        for i in range(len(dec_channels) - 1):
            is_last = (i == len(dec_channels) - 2)
            dec_layers += [nn.ConvTranspose1d(dec_channels[i], dec_channels[i+1], kernel_size=5,
                                               stride=2, padding=2, output_padding=1)]
            if not is_last:
                dec_layers += [nn.GroupNorm(8, dec_channels[i+1]), nn.ReLU(inplace=True)]
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
        recon_flux = F.interpolate(recon_flux, size=self.max_length, mode='linear', align_corners=False)
        return recon_flux.squeeze(1)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def vae_loss(recon_flux, target_flux, mask, mu, logvar, beta=0.01, free_bits=0.0):
    sq_err = (recon_flux - target_flux) ** 2 * mask
    recon_loss = sq_err.sum() / mask.sum().clamp(min=1)
    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    if free_bits > 0:
        kl_per_dim = torch.clamp(kl_per_dim, min=free_bits)
    kl_loss = kl_per_dim.mean()
    return recon_loss + beta * kl_loss, recon_loss.item(), kl_loss.item()


from torch.utils.data import Dataset, DataLoader


class LightCurveDataset(Dataset):
    def __init__(self, manifest_df, split, get_model_input_fn):
        self.rows = manifest_df[manifest_df['split'] == split].reset_index(drop=True)
        self.get_model_input_fn = get_model_input_fn

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows.iloc[idx]
        flux, mask = self.get_model_input_fn(row)
        return torch.tensor(flux, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)


EPOCHS_PER_TRIAL = 15
LATENT_CHANNELS_GRID = [4, 8, 16]
LR_GRID = [1e-3, 3e-4]
BATCH_SIZE = 8
BETA_MAX = 0.01
MAX_EPOCHS = 150
PATIENCE = 15
FREE_BITS = 0.25
WARMUP_FRAC = 1 / 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["setup", "trials", "select", "train", "gate", "all"], default="all")
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    split_assignment, vetted_pool, reflib_manifest, injection_manifest, gnm, gns = load_common()
    materialize_instance = make_materialize_instance(gnm, gns)

    bin_width_minutes, max_length = compute_windowing(split_assignment)

    def get_model_input(row):
        time, norm_flux = materialize_instance(row)
        return bin_light_curve(time, norm_flux, bin_width_minutes, max_length)

    train_dataset = LightCurveDataset(injection_manifest, 'train', get_model_input)
    val_dataset = LightCurveDataset(injection_manifest, 'val', get_model_input)
    print(f"Train instances: {len(train_dataset)}")
    print(f"Val instances: {len(val_dataset)}")

    if args.stage == "setup":
        return

    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    trial_configs = [(lc, lr) for lc in LATENT_CHANNELS_GRID for lr in LR_GRID]
    TRIAL_SCHEMA_TAG = f"latent_channels_v2|{LATENT_CHANNELS_GRID}|{LR_GRID}|{EPOCHS_PER_TRIAL}"

    def run_trial(latent_channels, lr, epochs=EPOCHS_PER_TRIAL, seed=SEED, beta_max=BETA_MAX, warmup_frac=WARMUP_FRAC):
        torch.manual_seed(seed)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
        model = ConvVAE(latent_channels=latent_channels, max_length=max_length).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        warmup_epochs = max(1, int(epochs * warmup_frac))

        t0 = time.time()
        for epoch in range(epochs):
            beta = beta_max * min(1.0, (epoch + 1) / warmup_epochs)
            model.train()
            for flux, mask in train_loader:
                flux, mask = flux.to(device), mask.to(device)
                x = torch.stack([flux, mask], dim=1)
                optimizer.zero_grad()
                recon, mu, logvar = model(x)
                loss, recon_l, kl_l = vae_loss(recon, flux, mask, mu, logvar, beta=beta)
                loss.backward()
                optimizer.step()
        wall_seconds = time.time() - t0

        model.eval()
        val_loss_total, val_recon_total, val_kl_total, n_batches = 0.0, 0.0, 0.0, 0
        with torch.no_grad():
            for flux, mask in val_loader:
                flux, mask = flux.to(device), mask.to(device)
                x = torch.stack([flux, mask], dim=1)
                recon, mu, logvar = model(x)
                loss, recon_l, kl_l = vae_loss(recon, flux, mask, mu, logvar, beta=1.0)
                val_loss_total += loss.item(); val_recon_total += recon_l
                val_kl_total += kl_l; n_batches += 1

        return {"schema_tag": TRIAL_SCHEMA_TAG,
                "latent_channels": latent_channels, "learning_rate": lr, "epochs": epochs,
                "val_loss": val_loss_total / n_batches, "val_recon_loss": val_recon_total / n_batches,
                "val_kl_loss": val_kl_total / n_batches, "wall_seconds": wall_seconds}

    results_path = DRIVE_BASE / 'provenance' / 'tuning_trial_results.json'

    if args.stage in ("trials", "all"):
        if results_path.exists():
            with open(results_path) as f:
                cached_results = json.load(f)
            stale = [r for r in cached_results if r.get("schema_tag") != TRIAL_SCHEMA_TAG]
            completed_results = [r for r in cached_results if r.get("schema_tag") == TRIAL_SCHEMA_TAG]
            if stale:
                print(f"Discarding {len(stale)} cached trial(s) from a stale/incompatible schema.")
            print(f"Resuming: {len(completed_results)} trial(s) already done under the current schema.")
        else:
            completed_results = []

        done_configs = {(r["latent_channels"], r["learning_rate"]) for r in completed_results}

        for latent_channels, lr in trial_configs:
            if (latent_channels, lr) in done_configs:
                continue
            print(f"Running trial: latent_channels={latent_channels}, lr={lr} ...")
            result = run_trial(latent_channels, lr)
            completed_results.append(result)
            with open(results_path, 'w') as f:
                json.dump(completed_results, f, indent=2)
            print(f"  -> val_loss={result['val_loss']:.4f} "
                  f"(recon={result['val_recon_loss']:.4f}, kl={result['val_kl_loss']:.4f}) "
                  f"in {result['wall_seconds']/60:.1f} min")

        print(f"\nAll {len(completed_results)}/{len(trial_configs)} trials complete.")

    if args.stage == "trials":
        return

    if args.stage in ("select", "train", "gate", "all"):
        with open(results_path) as f:
            all_results = json.load(f)
        all_results = [r for r in all_results if r.get("schema_tag") == TRIAL_SCHEMA_TAG]
        assert len(all_results) == len(trial_configs), \
            f"Search incomplete under current schema ({len(all_results)}/{len(trial_configs)}) -- run --stage trials first."

        best = min(all_results, key=lambda r: r["val_loss"])
        total_gpu_hours = sum(r["wall_seconds"] for r in all_results) / 3600
        LATENT_CHANNELS_FINAL = best["latent_channels"]
        LEARNING_RATE_FINAL = best["learning_rate"]

        tuning_budget_log = {
            "search_type": "fixed_budget_grid",
            "schema_tag": TRIAL_SCHEMA_TAG,
            "grid": {"latent_channels": LATENT_CHANNELS_GRID, "learning_rate": LR_GRID},
            "epochs_per_trial": EPOCHS_PER_TRIAL,
            "n_trials": len(all_results),
            "total_gpu_hours": round(total_gpu_hours, 3),
            "selection_metric": "validation total loss (recon + KL)",
            "selected_config": {"latent_channels": LATENT_CHANNELS_FINAL, "learning_rate": LEARNING_RATE_FINAL},
            "all_trial_results": all_results,
        }
        with open(DRIVE_BASE / 'provenance' / 'tuning_budget_log.json', 'w') as f:
            json.dump(tuning_budget_log, f, indent=2)
        print(f"Selected: latent_channels={LATENT_CHANNELS_FINAL}, learning_rate={LEARNING_RATE_FINAL}")
        print(f"Total GPU-hours spent on search: {total_gpu_hours:.2f}")
        print("Logged to provenance/tuning_budget_log.json")

    if args.stage == "select":
        return

    best_model_path = DRIVE_BASE / 'checkpoints' / 'vae_best.pt'

    if args.stage in ("train", "all"):
        warmup_epochs = max(1, int(MAX_EPOCHS * WARMUP_FRAC))
        torch.manual_seed(SEED)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
        model = ConvVAE(latent_channels=LATENT_CHANNELS_FINAL, max_length=max_length).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE_FINAL)

        history = []
        best_val_loss = float('inf')
        epochs_without_improvement = 0
        (DRIVE_BASE / 'checkpoints').mkdir(parents=True, exist_ok=True)

        for epoch in range(MAX_EPOCHS):
            beta = min(1.0, (epoch + 1) / warmup_epochs)

            model.train()
            train_loss_total, train_recon_total, train_kl_total, n_train_batches = 0.0, 0.0, 0.0, 0
            for flux, mask in train_loader:
                flux, mask = flux.to(device), mask.to(device)
                x = torch.stack([flux, mask], dim=1)
                optimizer.zero_grad()
                recon, mu, logvar = model(x)
                loss, recon_l, kl_l = vae_loss(recon, flux, mask, mu, logvar, beta=beta, free_bits=FREE_BITS)
                loss.backward()
                optimizer.step()
                train_loss_total += loss.item(); train_recon_total += recon_l
                train_kl_total += kl_l; n_train_batches += 1

            model.eval()
            val_loss_total, val_recon_total, val_kl_total, n_val_batches = 0.0, 0.0, 0.0, 0
            with torch.no_grad():
                for flux, mask in val_loader:
                    flux, mask = flux.to(device), mask.to(device)
                    x = torch.stack([flux, mask], dim=1)
                    recon, mu, logvar = model(x)
                    loss, recon_l, kl_l = vae_loss(recon, flux, mask, mu, logvar, beta=1.0, free_bits=0.0)
                    val_loss_total += loss.item(); val_recon_total += recon_l
                    val_kl_total += kl_l; n_val_batches += 1

            rec = {"epoch": epoch+1, "beta": beta,
                   "train_loss": train_loss_total/n_train_batches,
                   "train_recon_loss": train_recon_total/n_train_batches,
                   "train_kl_loss": train_kl_total/n_train_batches,
                   "val_loss": val_loss_total/n_val_batches,
                   "val_recon_loss": val_recon_total/n_val_batches,
                   "val_kl_loss": val_kl_total/n_val_batches}
            history.append(rec)

            if rec["val_loss"] < best_val_loss:
                best_val_loss = rec["val_loss"]; epochs_without_improvement = 0
                torch.save({"model_state_dict": model.state_dict(), "latent_channels": LATENT_CHANNELS_FINAL,
                            "learning_rate": LEARNING_RATE_FINAL, "epoch": epoch+1, "val_loss": best_val_loss},
                           best_model_path)
            else:
                epochs_without_improvement += 1

            print(f"Epoch {epoch+1}/{MAX_EPOCHS} (beta={beta:.2f}): train={rec['train_loss']:.4f}  "
                  f"val={rec['val_loss']:.4f} (recon={rec['val_recon_loss']:.4f}, kl={rec['val_kl_loss']:.4f}) "
                  f"{'*' if epochs_without_improvement==0 else ''}")

            with open(DRIVE_BASE / 'provenance' / 'training_history.json', 'w') as f:
                json.dump(history, f, indent=2)

            if epochs_without_improvement >= PATIENCE:
                print(f"\nEarly stopping: no improvement for {PATIENCE} epochs. Best val_loss={best_val_loss:.4f}.")
                break

        print(f"\nTraining complete. Best model saved to {best_model_path} (val_loss={best_val_loss:.4f}).")

    if args.stage == "train":
        return

    # ---- CELL 17: non-collapse hard gate ----
    ckpt = torch.load(best_model_path, map_location=device)
    model = ConvVAE(latent_channels=ckpt["latent_channels"], max_length=max_length).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    def masked_mse(a, b, mask):
        return (((a - b) ** 2) * mask).sum() / mask.sum().clamp(min=1)

    val_indices = np.linspace(0, len(val_dataset) - 1, 8, dtype=int)
    samples = [val_dataset[i] for i in val_indices]
    flux_stack = torch.stack([s[0] for s in samples]).to(device)
    mask_stack = torch.stack([s[1] for s in samples]).to(device)
    x_diverse = torch.stack([flux_stack, mask_stack], dim=1)

    with torch.no_grad():
        mu_diverse, _ = model.encode(x_diverse)
    mu_std_across_distinct_stars = mu_diverse.std(dim=0).mean().item()

    flux_batch, mask_batch = next(iter(val_loader))
    flux_batch, mask_batch = flux_batch.to(device), mask_batch.to(device)
    x_batch = torch.stack([flux_batch, mask_batch], dim=1)

    with torch.no_grad():
        mu, logvar = model.encode(x_batch)
        recon_normal = model.decode(mu)
        z_shuffled = mu[torch.randperm(mu.size(0))]
        recon_shuffled = model.decode(z_shuffled)

    decoder_sensitivity_to_z = masked_mse(recon_normal, recon_shuffled, mask_batch).item()
    reference_input_diff = masked_mse(flux_batch[0:1], flux_batch[1:2],
                                       mask_batch[0:1] * mask_batch[1:2]).item()
    sensitivity_ratio = decoder_sensitivity_to_z / max(reference_input_diff, 1e-8)

    MU_STD_MIN = 0.01
    SENSITIVITY_RATIO_MIN = 0.05

    print(f"(a) mu std across 8 distinct stars: {mu_std_across_distinct_stars:.6f} "
          f"(need > {MU_STD_MIN}) -> {'PASS' if mu_std_across_distinct_stars > MU_STD_MIN else 'FAIL'}")
    print(f"(b) decoder sensitivity to z: {decoder_sensitivity_to_z:.6f}, "
          f"reference input diff: {reference_input_diff:.6f}, ratio: {sensitivity_ratio:.4f} "
          f"(need > {SENSITIVITY_RATIO_MIN}) -> {'PASS' if sensitivity_ratio > SENSITIVITY_RATIO_MIN else 'FAIL'}")

    non_collapse_record = {
        "mu_std_across_distinct_stars": mu_std_across_distinct_stars,
        "decoder_sensitivity_to_z": decoder_sensitivity_to_z,
        "reference_input_diff": reference_input_diff,
        "sensitivity_ratio": sensitivity_ratio,
        "thresholds": {"mu_std_min": MU_STD_MIN, "sensitivity_ratio_min": SENSITIVITY_RATIO_MIN},
        "status": "PASS" if (mu_std_across_distinct_stars > MU_STD_MIN and sensitivity_ratio > SENSITIVITY_RATIO_MIN) else "FAIL",
    }
    with open(DRIVE_BASE / 'provenance' / 'vae_non_collapse_check.json', 'w') as f:
        json.dump(non_collapse_record, f, indent=2)

    assert non_collapse_record["status"] == "PASS", (
        "VAE POSTERIOR COLLAPSE DETECTED in the final trained checkpoint -- "
        "do not proceed to feature extraction / Model A-D with this model. "
        "See provenance/vae_non_collapse_check.json."
    )
    print("\nNon-collapse gate: PASS -- safe to use this checkpoint downstream.")

    print("\nVAE stage complete.")
    print(f"  Checkpoint: {best_model_path}")
    print(f"  latent_channels={LATENT_CHANNELS_FINAL}, learning_rate={LEARNING_RATE_FINAL}")
    print(f"  Non-collapse gate: {non_collapse_record['status']}")


if __name__ == "__main__":
    main()
