# MASTER PROJECT FILE
# Beyond Detection: Morphological Fingerprinting and
# Contrastive Characterization of Novel Stellar Anomalies in TESS Data

---

## 👤 Student Info
- **Name:** Samin
- **University:** Albukhary International University (AIU)
- **Student ID:** AIU24102090
- **Goal:** Masters-level AI research project for top university applications
- **Supervisor:** Claude (AI)
- **Timeline:** 3 months

---

## 🎯 Project Summary
An AI system that goes beyond detecting anomalies in NASA TESS star data
— it **characterizes** them. Finds unusual stellar light curves, creates a
morphological fingerprint for each anomaly, compares against known phenomena
using contrastive explanation, and assigns a novelty score.

### Why Novel
Previous work (CLARA, Villar et al. 2021, Webb et al. 2020) only detected
anomalies but could not explain what kind or how novel. This project adds:
1. Morphological Fingerprinting
2. Contrastive Explanation Engine
3. Novelty Score

---

## 🖥️ Environment
- **Platform:** Google Colab (T4 GPU)
- **Framework:** PyTorch
- **Notebook name:** My star.ipynb
- **Drive checkpoint path:** /content/drive/MyDrive/stellar_anomaly_checkpoints/

### Checkpoint Files (saved to Drive)
| File | Created by | Contains |
|------|-----------|---------|
| tess_dataset.npy | Cell 1 | 8 preprocessed light curves |
| vae_model.pth | Cell 2+3 | Trained VAE weights |
| anomaly_scores.npy | Cell 4 | Per-star anomaly scores |
| fingerprints.npy | Cell 5 | 13-dim morphological vectors |
| umap_embedding.npy | Cell 6 | 2D UMAP coordinates |
| cluster_labels.npy | Cell 6 | HDBSCAN cluster assignments |

---

## 🗺️ Roadmap

| Phase | Weeks | Cells | Status |
|-------|-------|-------|--------|
| Phase 1 | 1–2 | Cell 0–4 | 🔄 In Progress |
| Phase 2 | 3–5 | Cell 5–6 | ⏳ Pending |
| Phase 3 | 6–8 | Cell 7–9 | ⏳ Pending |
| Phase 4 | 9–12 | Cell 10+ | ⏳ Pending |

---

## ✅ CURRENT STATUS
- **Last completed:** Cell 0 (Master Setup) ✅
- **Next step:** Run Cell 1 — Download TESS Data
- **Drive status:** Empty — all cells need to run

---

## 📋 HOW TO CONTINUE IN NEW CONVERSATION
1. Download this MASTER.md from GitHub
2. Upload to new Claude conversation
3. Say: "আমি Samin। File দেখো, [Cell X] থেকে continue করো"
4. Claude will have full context immediately

---

## 💻 COMPLETE CODE — ALL CELLS

### 🔧 CELL 0 — Master Setup (Run Every Session First)
```python
import os, sys
print("="*55)
print("  STELLAR ANOMALY RESEARCH — SESSION STARTUP")
print("="*55)

print("\n[1/5] Installing libraries...")
!pip install lightkurve astroquery umap-learn hdbscan -q
print("✓ Done")

print("[2/5] Importing modules...")
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks
from scipy.stats import entropy, skew, kurtosis
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import umap, hdbscan
import warnings
warnings.filterwarnings('ignore')
print("✓ Done")

print("[3/5] Mounting Google Drive...")
from google.colab import drive
drive.mount('/content/drive', force_remount=False)
DRIVE_PATH = '/content/drive/MyDrive/stellar_anomaly_checkpoints'
os.makedirs(DRIVE_PATH, exist_ok=True)
print(f"✓ Drive mounted — {DRIVE_PATH}")

print("[4/5] Defining model and functions...")
class VAE(nn.Module):
    def __init__(self, input_dim=1000, latent_dim=32):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(),
            nn.Linear(512, 128), nn.ReLU()
        )
        self.fc_mu  = nn.Linear(128, latent_dim)
        self.fc_var = nn.Linear(128, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.ReLU(),
            nn.Linear(128, 512),        nn.ReLU(),
            nn.Linear(512, input_dim),  nn.Sigmoid()
        )
    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_var(h)
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        return mu + std * torch.randn_like(std)
    def decode(self, z):
        return self.decoder(z)
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var

def vae_loss(recon, original, mu, log_var):
    recon_loss = nn.MSELoss()(recon, original)
    kl_loss    = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
    return recon_loss + 0.001 * kl_loss

def compute_symmetry(curve):
    mid = len(curve) // 2
    left  = curve[:mid]
    right = curve[mid:mid+len(left)][::-1]
    return 1.0 - np.mean(np.abs(left - right))

def count_peaks(curve, prominence=0.05):
    peaks,   _ = find_peaks( curve, prominence=prominence, distance=20)
    troughs, _ = find_peaks(-curve, prominence=prominence, distance=20)
    return len(peaks), len(troughs)

def rise_fall_ratio(curve):
    peak_idx = np.argmax(curve)
    if peak_idx == 0 or peak_idx == len(curve)-1:
        return 1.0
    rise = (curve[peak_idx] - curve[0])  / (peak_idx + 1e-8)
    fall = (curve[peak_idx] - curve[-1]) / (len(curve) - peak_idx + 1e-8)
    return rise / (fall + 1e-8)

def compute_fft_features(curve):
    n          = len(curve)
    fft_vals   = np.abs(fft(curve - np.mean(curve)))[:n//2]
    freqs      = fftfreq(n)[:n//2]
    power      = fft_vals ** 2
    total      = np.sum(power) + 1e-10
    power_norm = power / total
    dom_idx    = np.argmax(power[1:]) + 1
    return freqs[dom_idx], entropy(power_norm[1:]+1e-10), np.sum(power[n//8:])/total

def compute_duration_features(curve, threshold=0.3):
    return (np.sum(curve < threshold)/len(curve),
            np.sum(curve > (1-threshold))/len(curve))

def compute_variability(curve):
    return np.std(curve), skew(curve), kurtosis(curve), np.sqrt(np.mean(curve**2))

def morphological_fingerprint(curve):
    s            = compute_symmetry(curve)
    np_, nt      = count_peaks(curve)
    rfr          = rise_fall_ratio(curve)
    df, se, hf   = compute_fft_features(curve)
    dim, bri     = compute_duration_features(curve)
    std, sk, ku, rms = compute_variability(curve)
    return np.array([s, np_, nt, rfr, df, se, hf, dim, bri, std, sk, ku, rms])

FEATURE_NAMES = [
    "Symmetry","N_Peaks","N_Troughs","Rise/Fall",
    "Dom_Freq","Spec_Entropy","HF_Power",
    "Dim_Frac","Bright_Frac","Flux_Std",
    "Skewness","Kurtosis","RMS"
]
print("✓ Done")

print("[5/5] Loading checkpoints from Drive...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CHECKPOINTS = {
    'dataset'       : f'{DRIVE_PATH}/tess_dataset.npy',
    'anomaly_scores': f'{DRIVE_PATH}/anomaly_scores.npy',
    'fingerprints'  : f'{DRIVE_PATH}/fingerprints.npy',
    'embedding'     : f'{DRIVE_PATH}/umap_embedding.npy',
    'cluster_labels': f'{DRIVE_PATH}/cluster_labels.npy',
    'vae_model'     : f'{DRIVE_PATH}/vae_model.pth',
}
loaded = {}
for name, path in CHECKPOINTS.items():
    if os.path.exists(path):
        if name == 'vae_model':
            model = VAE(input_dim=1000, latent_dim=32).to(device)
            model.load_state_dict(torch.load(path, map_location=device))
            model.eval()
        else:
            globals()[name] = np.load(path)
        loaded[name] = True
    else:
        loaded[name] = False

STATUS_MAP = {
    'dataset'       : ('Cell 1',   'Download TESS data'),
    'vae_model'     : ('Cell 2+3', 'Build + Train VAE'),
    'anomaly_scores': ('Cell 4',   'Anomaly Detection'),
    'fingerprints'  : ('Cell 5',   'Morphological Fingerprinting'),
    'embedding'     : ('Cell 6',   'UMAP + Clustering'),
    'cluster_labels': ('Cell 6',   'UMAP + Clustering'),
}
print("\n" + "="*45)
print("  CHECKPOINT STATUS")
print("="*45)
all_ok = True
for name, (cell, desc) in STATUS_MAP.items():
    icon = "✅" if loaded[name] else "❌"
    note = "loaded" if loaded[name] else f"→ run {cell}"
    print(f"  {icon}  {desc:<32} {note}")
    if not loaded[name]: all_ok = False
print("="*45)
if all_ok:
    print("  🟢 ALL LOADED — jump to your next cell!")
else:
    print("  🔴 Some files missing — run cells above")
print("="*45)
print(f"\n  Device: {device}")
print(f"  Drive:  {DRIVE_PATH}")
```

### 🔧 CELL 1 — Download TESS Data
**Status:** ⏳ Pending
**Run when:** Cell 0 shows ❌ for "Download TESS data"
```python
import lightkurve as lk

def preprocess_lightcurve(lc):
    try:
        lc   = lc.remove_nans().remove_outliers(sigma=5)
        time = lc.time.value
        flux = lc.flux.value
        if len(time) < 100: return None
        fmin, fmax = np.min(flux), np.max(flux)
        if (fmax - fmin) < 1e-10: return None
        flux_norm = (flux - fmin) / (fmax - fmin)
        t_uniform = np.linspace(time.min(), time.max(), 1000)
        interp    = interp1d(time, flux_norm,
                             kind='linear', fill_value='extrapolate')
        return np.clip(interp(t_uniform), 0, 1)
    except:
        return None

TIC_IDS = [
    "TIC 279741379","TIC 261136679","TIC 410214986",
    "TIC 144065872","TIC 231670397","TIC 267263253",
    "TIC 425934411","TIC 149603524","TIC 236445129",
    "TIC 198456033","TIC 307210830","TIC 166527623",
    "TIC 349118411","TIC 238196153","TIC 229945862"
]

dataset = []
print("Downloading TESS light curves...\n")
for tic in TIC_IDS:
    try:
        search = lk.search_lightcurve(
            tic, mission="TESS", sector=1, author="SPOC")
        if len(search) == 0: continue
        lc        = search[0].download()
        processed = preprocess_lightcurve(lc)
        if processed is not None:
            dataset.append(processed)
            print(f"  ✓ {tic}")
    except:
        print(f"  ✗ {tic} — skipped")

dataset = np.array(dataset)
np.save(CHECKPOINTS['dataset'], dataset)
print(f"\n✓ Dataset shape: {dataset.shape}")
print(f"✓ Saved to Drive!")
```

### 🔧 CELL 2+3 — Build + Train VAE
**Status:** ⏳ Pending
**Run when:** Cell 0 shows ❌ for "Build + Train VAE"
```python
X            = torch.FloatTensor(dataset)
train_loader = DataLoader(TensorDataset(X), batch_size=4, shuffle=True)
model        = VAE(input_dim=1000, latent_dim=32).to(device)
optimizer    = optim.Adam(model.parameters(), lr=1e-3)

print(f"Device: {device}")
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
print("\nTraining VAE...\n")

train_losses = []
for epoch in range(200):
    model.train()
    epoch_loss = 0
    for batch in train_loader:
        x = batch[0].to(device)
        optimizer.zero_grad()
        recon, mu, log_var = model(x)
        loss = vae_loss(recon, x, mu, log_var)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    avg = epoch_loss / len(train_loader)
    train_losses.append(avg)
    if (epoch+1) % 20 == 0:
        print(f"  Epoch {epoch+1:3d}/200 | Loss: {avg:.6f}")

torch.save(model.state_dict(), CHECKPOINTS['vae_model'])
print(f"\n✓ Final loss: {train_losses[-1]:.6f}")
print(f"✓ Saved to Drive!")
```

### 🔧 CELL 4 — Anomaly Detection
**Status:** ⏳ Pending
**Run when:** Cell 0 shows ❌ for "Anomaly Detection"
```python
model.eval()
anomaly_scores       = []
reconstructed_curves = []

with torch.no_grad():
    for i in range(len(dataset)):
        x     = torch.FloatTensor(dataset[i]).unsqueeze(0).to(device)
        recon, mu, log_var = model(x)
        score = nn.MSELoss()(recon, x).item()
        anomaly_scores.append(score)
        reconstructed_curves.append(recon.cpu().numpy()[0])

anomaly_scores       = np.array(anomaly_scores)
reconstructed_curves = np.array(reconstructed_curves)
threshold            = np.mean(anomaly_scores) + np.std(anomaly_scores)

print("Anomaly Scores:")
print("-"*45)
for i, score in enumerate(anomaly_scores):
    flag = " ← 🚨 ANOMALY" if score > threshold else ""
    print(f"  Star #{i+1:2d} | {score:.6f}{flag}")
print("-"*45)
print(f"  Threshold: {threshold:.6f}")

np.save(CHECKPOINTS['anomaly_scores'], anomaly_scores)
print(f"\n✓ Saved to Drive!")
```

### 🔧 CELL 5 — Morphological Fingerprinting
**Status:** ⏳ Pending
**Run when:** Cell 0 shows ❌ for "Morphological Fingerprinting"
```python
threshold    = np.mean(anomaly_scores) + np.std(anomaly_scores)
fingerprints = []

print("Computing fingerprints...\n")
for i, curve in enumerate(dataset):
    fp   = morphological_fingerprint(curve)
    fingerprints.append(fp)
    flag = " ← 🚨 ANOMALY" if anomaly_scores[i] > threshold else ""
    print(f"  Star #{i+1:2d} | {anomaly_scores[i]:.6f}{flag}")

fingerprints = np.array(fingerprints)
np.save(CHECKPOINTS['fingerprints'], fingerprints)

fp_vis = MinMaxScaler().fit_transform(fingerprints)
fig, ax = plt.subplots(figsize=(14,5))
im = ax.imshow(fp_vis.T, aspect='auto', cmap='viridis')
ax.set_yticks(range(len(FEATURE_NAMES)))
ax.set_yticklabels(FEATURE_NAMES, fontsize=9)
ax.set_xticks(range(len(dataset)))
ax.set_xticklabels([f"S{i+1}" for i in range(len(dataset))], fontsize=9)
ax.set_title("Morphological Fingerprint Heatmap")
plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.show()
print(f"\n✓ Shape: {fingerprints.shape}")
print(f"✓ Saved to Drive!")
```

### 🔧 CELL 6 — UMAP + HDBSCAN Clustering
**Status:** ⏳ Pending
**Run when:** Cell 0 shows ❌ for "UMAP + Clustering"
```python
fp_scaled = StandardScaler().fit_transform(fingerprints)
n_stars   = len(fingerprints)

reducer   = umap.UMAP(
    n_components=2, n_neighbors=min(5, n_stars-1),
    min_dist=0.1, metric='euclidean', random_state=42)
embedding = reducer.fit_transform(fp_scaled)

clusterer     = hdbscan.HDBSCAN(
    min_cluster_size=max(2, n_stars//4),
    min_samples=1, metric='euclidean')
cluster_labels = clusterer.fit_predict(embedding)
n_clusters     = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)

np.save(CHECKPOINTS['embedding'],      embedding)
np.save(CHECKPOINTS['cluster_labels'], cluster_labels)

fig, axes = plt.subplots(1, 2, figsize=(16,6))
unique_labels = sorted(set(cluster_labels))
palette   = plt.cm.tab10(np.linspace(0,1,max(len(unique_labels),2)))
color_map = {l: palette[i] for i,l in enumerate(unique_labels)}
color_map[-1] = (0.5,0.5,0.5,1.0)

axes[0].scatter(embedding[:,0], embedding[:,1],
    c=[color_map[l] for l in cluster_labels],
    s=120, edgecolors='white', linewidths=0.8)
for i,(x,y) in enumerate(embedding):
    axes[0].annotate(f"S{i+1}",(x,y),
        textcoords="offset points",xytext=(6,4),
        fontsize=8,color='white')
axes[0].set_facecolor('#1a1a2e')
axes[0].set_title("UMAP — by Cluster")
axes[0].grid(True,alpha=0.2)

sc = axes[1].scatter(embedding[:,0], embedding[:,1],
    c=anomaly_scores, cmap='hot', s=120,
    edgecolors='white', linewidths=0.8)
for i,(x,y) in enumerate(embedding):
    axes[1].annotate(f"S{i+1}",(x,y),
        textcoords="offset points",xytext=(6,4),
        fontsize=8,color='white')
plt.colorbar(sc, ax=axes[1], label="Anomaly Score")
axes[1].set_facecolor('#1a1a2e')
axes[1].set_title("UMAP — by Anomaly Score")
axes[1].grid(True,alpha=0.2)
plt.suptitle("Morphological Fingerprint Space", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()

print(f"✓ Clusters: {n_clusters}")
print(f"✓ Saved to Drive!")
print("\n→ Next: Cell 7 — Contrastive Explanation Engine")
```

### 🔧 CELL 7 — Contrastive Explanation Engine
**Status:** ⏳ Pending — NOT YET BUILT

### 🔧 CELL 8 — Novelty Scoring
**Status:** ⏳ Pending — NOT YET BUILT

### 🔧 CELL 9 — Results + Visualization
**Status:** ⏳ Pending — NOT YET BUILT

---

## 📊 Results Log
*(Update this section after each cell completes)*

| Cell | Key Output | Values |
|------|-----------|--------|
| Cell 0 | Setup | ✅ CUDA, Drive mounted |
| Cell 1 | Stars downloaded | TBD |
| Cell 2+3 | VAE final loss | TBD |
| Cell 4 | Anomalies found | TBD |
| Cell 5 | Fingerprint shape | TBD |
| Cell 6 | Clusters found | TBD |

---

## 🎯 Target Venues
- NeurIPS ML4PS Workshop
- Astronomy & Astrophysics journal
- ICML Workshop
