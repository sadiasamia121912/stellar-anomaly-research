# Remote GPU bridge (vast.ai, SSH-blocked workaround)

SSH is blocked at the network/ISP level on the dev machine used for this project
(confirmed across Defender-off, mobile hotspot, raw sockets — see project memory).
This directory holds the working alternative: drive the rented vast.ai box over
HTTPS via Jupyter's terminals + contents API instead of SSH.

## Files
- `gpu_conn.json.example` — copy to `gpu_conn.json` (gitignored) and fill in the
  current box's tunnel URL + token from the vast.ai console's "Open" button.
  **Never commit gpu_conn.json** — it holds a live auth token.
- `gpu_bridge.py` — the bridge itself:
  - `python gpu_bridge.py run "<cmd>" [--timeout N]` — run one shell command on
    the box, get its stdout back.
  - `python gpu_bridge.py upload <local> <remote>` — chunked upload via the
    Contents API (remote path is relative to `/`, the box's Jupyter root).
  - `python gpu_bridge.py download <remote> <local>` — the reverse.
- `notebook3_vae_stage.py` — Notebook 3 (VAE stage)'s cells 1-18 ported to a
  plain script, staged via `--stage {setup,trials,select,train,gate,all}` so a
  dropped connection doesn't lose progress (each stage reads/writes the same
  provenance JSON files the notebook does). Needs `NB3_DATA_DIR` env var
  pointing at the uploaded data directory (mirrors the Drive layout: splits/,
  vetting/, injections/, provenance/, processed/general/, reference_library/).

## Known gotcha this file's marker-detection code works around
The PTY echoes the typed command back as stdout, sometimes twice (raw echo +
post-bracketed-paste-mode echo) before the command's real output appears — so a
naive "wait for the marker string" trips on the echo, not the real completion.
`_find_real_marker_end()` in gpu_bridge.py distinguishes them: an echo occurrence
is always immediately preceded by "echo " (since the sent command is literally
`<cmd>; echo <marker>`); the real output occurrence never is.

## Data upload pattern
Do NOT copy rclone/GitHub OAuth credentials onto the rented box (it's reached
through a public tunnel URL — a real exposure risk). Instead: zip only the
files actually needed from the local Drive mirror
(`C:\Users\User\stellar-anomaly-data-local`), upload the zip via
`gpu_bridge.py upload` (chunked), unzip on the box, delete the zip.
(A Drive-public-link shortcut was tried once and blocked by Claude Code's own
permission classifier as an outward-facing action — the chunked direct-upload
path above is the one that's actually approved and working.)
