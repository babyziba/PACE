#!/usr/bin/env python3
"""
tcn_anomaly.py

End-to-end pipeline for pitcher anomaly detection with a Temporal CNN Autoencoder (TCN).

Steps:
1) Load per-pitch CSV (Statcast-like) with common column aliasing:
   - pitcher -> pitcher_id
   - release_extension -> extension
2) Optional one-hot of pitch_type (keeps a copy as 'pitch_type_raw' for grouping).
3) Choose time windows:
   - TRAIN/VAL: rows with game_date <= --train_end_date (if provided), else whole CSV.
   - SCORING:   rows with game_date >= --score_start_date (if provided), else whole CSV.
4) Per-group normalization: fit StandardScaler on TRAIN by pitcher_id
   (optionally by pitcher_id + pitch_type_raw with --group_by_pitch_type), then
   apply to VAL & SCORING.
5) Build sliding windows [B, C, T] from TRAIN/VAL; train TCN autoencoder (MSE).
6) Score chosen pitcher on SCORING:
   - end-of-window reconstruction error -> EMA smoothing -> z-scores (std or robust MAD)
   - percentile threshold (--pct); optional consecutive filter (--min_run)
   - optional per-feature error attribution (--save_drivers)
   - optional raw-feature delta columns vs baseline (--save_deltas)
7) Save: anomaly CSV + timeline plot.

Usage example:
  python tcn_anomaly.py \
    --csv /path/pitches.csv \
    --out_dir outputs \
    --pitcher_id 621111 \
    --seq_len 64 --epochs 30 --batch_size 128 \
    --one_hot_pitch_type \
    --train_end_date 2022-06-01 \
    --score_start_date 2022-06-01 \
    --pct 97.5

Requirements:
  pip install pandas numpy torch matplotlib scikit-learn
"""

import os, argparse, random
from typing import Dict, Tuple, List, Optional, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import matplotlib.dates as mdates

def _pretty_timeline(df, title, pct, out_path, injury_date=None, roll=5):
    thr = df["z_score"].quantile(pct/100.0)

    fig, ax = plt.subplots(figsize=(12, 4.2), dpi=160)
    ax.plot(df["game_date"], df["z_score"], lw=2, label="z-score")
    if roll and roll > 1:
        ax.plot(df["game_date"], df["z_score"].rolling(roll, center=True).median(),
                lw=1.6, alpha=0.7, label=f"Rolling median ({roll})")

    ax.axhline(thr, ls="--", lw=1.5, label=f"{int(pct)}th pct = {thr:.2f}")
    flags = df["z_score"] > thr
    ax.scatter(df.loc[flags,"game_date"], df.loc[flags,"z_score"], s=36, zorder=3, label="Flagged")

    peak_i = df["z_score"].idxmax()
    ax.annotate(f"Peak {df.loc[peak_i,'z_score']:.2f}\n{df.loc[peak_i,'game_date'].date()}",
                xy=(df.loc[peak_i,"game_date"], df.loc[peak_i,"z_score"]),
                xytext=(10, 12), textcoords="offset points", fontsize=9)

    if injury_date:
        dj = pd.to_datetime(injury_date)
        ax.axvline(dj, ls=":", lw=1.5, color="red", label="Injury/IL")

    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.grid(axis="y", ls=":", alpha=0.5)
    ax.set_title(title)
    ax.set_ylabel("z-score (reconstruction error)")
    ax.legend(loc="upper left", ncol=2, frameon=False)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[Saved] {out_path}")

def _per_game_max_plot(df, title, pct, out_path, injury_date=None):
    g = (df.groupby("game_date")["z_score"].max()
           .reset_index()
           .sort_values("game_date"))
    thr = g["z_score"].quantile(pct/100.0)

    fig, ax = plt.subplots(figsize=(12, 4.2), dpi=160)
    ax.plot(g["game_date"], g["z_score"], lw=2, marker="o", ms=4)
    ax.axhline(thr, ls="--", lw=1.5, label=f"{int(pct)}th pct = {thr:.2f}")
    flags = g["z_score"] > thr
    ax.scatter(g.loc[flags,"game_date"], g.loc[flags,"z_score"], s=40, zorder=3, label="Flagged")

    if injury_date:
        dj = pd.to_datetime(injury_date)
        ax.axvline(dj, ls=":", lw=1.5, color="red", label="Injury/IL")

    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.grid(axis="y", ls=":", alpha=0.5)
    ax.set_title(f"{title} — per-game max")
    ax.set_ylabel("z-score (reconstruction error)")
    ax.legend(loc="upper left", ncol=2, frameon=False)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[Saved] {out_path}")



# -----------------------
# Reproducibility helpers
# -----------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -----------------------
# TCN components
# -----------------------
class TcnBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, k: int = 3, d: int = 1, p: float = 0.1):
        super().__init__()
        pad = d * (k - 1) // 2
        self.net = nn.Sequential(
            nn.Conv1d(c_in, c_out, kernel_size=k, padding=pad, dilation=d),
            nn.BatchNorm1d(c_out),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Conv1d(c_out, c_out, kernel_size=k, padding=pad, dilation=d),
            nn.BatchNorm1d(c_out),
            nn.ReLU(),
        )
        self.res = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) + self.res(x)


class TcnAutoencoder(nn.Module):
    def __init__(self, c_in: int, hid: int = 64, p: float = 0.1):
        super().__init__()
        # Encoder
        self.e1 = TcnBlock(c_in, 64,  k=3, d=1, p=p)
        self.e2 = TcnBlock(64,  128, k=3, d=2, p=p)
        self.e3 = TcnBlock(128, 256, k=3, d=4, p=p)
        self.bot = nn.Conv1d(256, hid, kernel_size=1)
        # Decoder
        self.d1 = TcnBlock(hid, 256, k=3, d=4, p=p)
        self.d2 = TcnBlock(256, 128, k=3, d=2, p=p)
        self.d3 = TcnBlock(128, 64,  k=3, d=1, p=p)
        self.out = nn.Conv1d(64, c_in, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.e3(self.e2(self.e1(x)))
        z = self.bot(z)
        y = self.d3(self.d2(self.d1(z)))
        return self.out(y)


# -----------------------
# Dataset & windowing
# -----------------------
class PitchWindows(Dataset):
    """
    Takes a per-pitch dataframe for a SINGLE pitcher with:
      - columns 'game_date', 'pitch_number' for ordering
      - numeric feature columns
    Builds sliding windows of shape [C, T].
    """
    def __init__(self, df_one_pitcher: pd.DataFrame, feat_cols: List[str],
                 seq_len: int = 64, stride: int = 1):
        self.seq_len = seq_len
        self.stride = stride

        dd = df_one_pitcher.sort_values(
            ["game_date", "pitch_number"], kind="mergesort"
        ).reset_index(drop=True)

        X = dd[feat_cols].to_numpy(dtype=np.float32)
        self.meta = dd[["game_date", "pitch_number"]].reset_index(drop=True)

        # build windows
        self.windows = []
        self.indices = []  # end index per window for mapping back
        T = len(X)
        for end in range(seq_len, T + 1, stride):
            start = end - seq_len
            self.windows.append(X[start:end].T)  # [T, C] -> [C, T]
            self.indices.append(end - 1)
        self.windows = (
            np.stack(self.windows, axis=0)
            if self.windows
            else np.empty((0, len(feat_cols), seq_len), dtype=np.float32)
        )

    def __len__(self) -> int:
        return self.windows.shape[0]

    def __getitem__(self, idx: int):
        # Return tensor [C, T] and index in original series (the window's end)
        return torch.from_numpy(self.windows[idx]), self.indices[idx]


# -----------------------
# Utilities
# -----------------------
def one_hot_pitch_type(df: pd.DataFrame, col: str = "pitch_type") -> Tuple[pd.DataFrame, List[str]]:
    """One-hot encode pitch_type if present, returning (new df, added columns)."""
    if col not in df.columns:
        return df, []
    cats = sorted([c for c in df[col].astype(str).unique() if c.lower() != "nan"])
    if not cats:
        return df, []
    oh = pd.get_dummies(df[col].astype(str), prefix="pt")
    df2 = pd.concat([df.drop(columns=[col]), oh], axis=1)
    return df2, list(oh.columns)


def _group_key(pid: int, pt: Optional[str], use_pitch_type: bool) -> Union[int, Tuple[int, str]]:
    return (int(pid), str(pt)) if use_pitch_type and pt is not None else int(pid)


def fit_group_scalers(train_df: pd.DataFrame, feat_cols: List[str],
                      group_by_pitch_type: bool = False,
                      type_col: str = "pitch_type_raw") -> Tuple[Dict[Union[int, Tuple[int, str]], StandardScaler], StandardScaler]:
    """
    Fit a StandardScaler per group on TRAIN.
    Group is pitcher_id or (pitcher_id, pitch_type_raw) if group_by_pitch_type.
    Also returns a global fallback scaler.
    """
    pid2scaler: Dict[Union[int, Tuple[int, str]], StandardScaler] = {}
    if group_by_pitch_type and type_col not in train_df.columns:
        print("[Warn] --group_by_pitch_type set but no pitch_type column present; falling back to pitcher-only.")
        group_by_pitch_type = False

    if group_by_pitch_type:
        for (pid, pt), grp in train_df.groupby(["pitcher_id", type_col]):
            if grp.empty:  # safety
                continue
            sc = StandardScaler().fit(grp[feat_cols])
            pid2scaler[_group_key(pid, pt, True)] = sc
    else:
        for pid, grp in train_df.groupby("pitcher_id"):
            sc = StandardScaler().fit(grp[feat_cols])
            pid2scaler[_group_key(pid, None, False)] = sc

    global_scaler = StandardScaler().fit(train_df[feat_cols])
    return pid2scaler, global_scaler


def apply_group_scalers(df_src: pd.DataFrame, feat_cols: List[str],
                        pid2scaler: Dict[Union[int, Tuple[int, str]], StandardScaler],
                        global_scaler: StandardScaler,
                        group_by_pitch_type: bool = False,
                        type_col: str = "pitch_type_raw") -> pd.DataFrame:
    """Apply the appropriate scaler per group; fallback to global if unseen."""
    use_pt = group_by_pitch_type and (type_col in df_src.columns)
    out = []
    if use_pt:
        for (pid, pt), grp in df_src.groupby(["pitcher_id", type_col]):
            g = grp.copy()
            sc = pid2scaler.get(_group_key(pid, pt, True),
                                pid2scaler.get(_group_key(pid, None, False), global_scaler))
            g[feat_cols] = sc.transform(g[feat_cols])
            out.append(g)
        # handle rows where pitch_type_raw is NaN
        missing = df_src[df_src[type_col].isna()]
        if len(missing):
            for pid, grp in missing.groupby("pitcher_id"):
                g = grp.copy()
                sc = pid2scaler.get(_group_key(pid, None, False), global_scaler)
                g[feat_cols] = sc.transform(g[feat_cols])
                out.append(g)
    else:
        for pid, grp in df_src.groupby("pitcher_id"):
            g = grp.copy()
            sc = pid2scaler.get(_group_key(pid, None, False), global_scaler)
            g[feat_cols] = sc.transform(g[feat_cols])
            out.append(g)
    return pd.concat(out, ignore_index=True) if out else df_src.copy()


def split_train_val_by_date(df: pd.DataFrame, date_col: str = "game_date",
                            val_frac: float = 0.2, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split by unique dates to avoid leakage. Returns (train_df, val_df)."""
    dates = sorted(df[date_col].dropna().unique())
    rng = np.random.RandomState(seed)
    rng.shuffle(dates)
    k = max(1, int(len(dates) * (1 - val_frac)))
    train_dates = set(dates[:k])
    train_df = df[df[date_col].isin(train_dates)].copy()
    val_df = df[~df[date_col].isin(train_dates)].copy()
    return train_df, val_df


def mse_per_timestep(x_true: torch.Tensor, x_pred: torch.Tensor) -> torch.Tensor:
    """x_true, x_pred: [B, C, T] -> per-sample per-timestep MSE reduced over channels: [B, T]."""
    return ((x_true - x_pred) ** 2).mean(dim=1)


def per_feature_error_last_t(x_true: torch.Tensor, x_pred: torch.Tensor) -> torch.Tensor:
    """
    Return per-feature (channel) squared error at the LAST timestep: [B, C].
    """
    diff = x_true - x_pred  # [B, C, T]
    last = diff[:, :, -1]
    return last.pow(2)  # [B, C]


def consecutive_runs(mask: np.ndarray) -> np.ndarray:
    """
    Given boolean array, return run length of consecutive True values at each position.
    E.g., [0,1,1,0,1] -> [0,1,2,0,1]
    """
    runs = np.zeros_like(mask, dtype=int)
    c = 0
    for i, v in enumerate(mask):
        c = c + 1 if v else 0
        runs[i] = c
    return runs


# -----------------------
# Training
# -----------------------
def train_tcn(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, device: torch.device,
              epochs: int = 30, lr: float = 1e-3, wd: float = 1e-6, grad_clip: float = 1.0):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr * 3, steps_per_epoch=max(1, len(train_loader)), epochs=epochs, pct_start=0.3
    )
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    history = {"train": [], "val": []}

    for ep in range(1, epochs + 1):
        # ---- train
        model.train()
        tr_loss = 0.0
        for xb, _ in train_loader:
            xb = xb.to(device)  # [B, C, T]
            opt.zero_grad()
            yb = model(xb)
            loss = loss_fn(yb, xb)
            loss.backward()
            if grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            if len(train_loader) > 0:
                sched.step()
            tr_loss += loss.item()
        tr_loss /= max(1, len(train_loader))

        # ---- val
        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for xb, _ in val_loader:
                xb = xb.to(device)
                yb = model(xb)
                loss = loss_fn(yb, xb)
                va_loss += loss.item()
        va_loss /= max(1, len(val_loader))

        history["train"].append(tr_loss)
        history["val"].append(va_loss)
        print(f"Epoch {ep:03d} | Train {tr_loss:.4f} | Val {va_loss:.4f}")

        if va_loss < best_val:
            best_val = va_loss
            torch.save(model.state_dict(), os.path.join(args.out_dir, "best_tcn.pth"))

    return history


# -----------------------
# Anomaly scoring + plotting
# -----------------------
def score_anomalies_for_pitcher(
    model: nn.Module,
    df: pd.DataFrame,
    feat_cols: List[str],
    pitcher_id: int,
    seq_len: int,
    stride: int,
    device: torch.device,
    out_dir: str,
    pct: float = 97.5,
    ema_alpha: float = 0.2,
    z_method: str = "std",
    min_run: int = 1,
    save_drivers: bool = False,
    save_deltas: bool = False,
    raw_df_for_deltas: Optional[pd.DataFrame] = None,
    baseline_end_date: Optional[pd.Timestamp] = None,
) -> Tuple[str, str]:
    """
    Compute reconstruction error per timestep for the chosen pitcher,
    z-score it within that pitcher, and save CSV + timeline plot.
    Options:
      - z_method: "std" (mean/std) or "mad" (robust median/MAD)
      - min_run: require >=N consecutive anomaly windows to consider final flag
      - save_drivers: include per-feature end-step squared error columns (err_<feat>)
      - save_deltas: include delta_<feat> columns vs baseline means from raw_df_for_deltas
    """
    dfp = df[df["pitcher_id"] == pitcher_id].copy()
    if len(dfp) < seq_len:
        raise ValueError(f"Pitcher {pitcher_id} has only {len(dfp)} pitches (< seq_len={seq_len}).")

    print(f"[Scoring] pitcher_id={pitcher_id} rows={len(dfp)}  date range: "
          f"{dfp['game_date'].min().date()} → {dfp['game_date'].max().date()}")

    ds = PitchWindows(dfp, feat_cols, seq_len=seq_len, stride=stride)
    dl = DataLoader(ds, batch_size=128, shuffle=False)
    model.eval()

    all_err = []
    all_end_idx = []
    driver_chunks = []  # per-feature squared error at last timestep

    with torch.no_grad():
        for xb, idxs in dl:
            xb = xb.to(device)
            yb = model(xb)
            et = mse_per_timestep(xb, yb).cpu().numpy()  # [B, T]
            all_err.append(et)
            all_end_idx.extend(idxs.numpy().tolist())
            if save_drivers:
                per_feat = per_feature_error_last_t(xb, yb).cpu().numpy()  # [B, C]
                driver_chunks.append(per_feat)

    if not all_err:
        raise ValueError("No scoring windows were built. Try lowering --seq_len or ensure enough pitches exist.")

    err_mat = np.concatenate(all_err, axis=0)  # [num_windows, T]
    end_err = err_mat[:, -1]  # error aligned to window end

    # ---- EMA smoothing before z-scoring
    se = pd.Series(end_err)
    end_err_smooth = se.ewm(alpha=ema_alpha).mean().to_numpy()

    # Map to original rows at window end
    out = dfp.iloc[all_end_idx][["game_date", "pitch_number"]].reset_index(drop=True)
    out["recon_error"] = end_err_smooth

    # ---- z-scores
    x = out["recon_error"].values.astype(float)
    if z_method.lower() == "mad":
        med = np.median(x)
        mad = np.median(np.abs(x - med))
        scale = (mad * 1.4826) if mad > 0 else 1.0
        z = (x - med) / scale
        method_used = "MAD (robust)"
    else:
        mu = x.mean()
        sd = x.std(ddof=1) if len(x) > 1 else 0.0
        if sd == 0:
            sd = 1.0
        z = (x - mu) / sd
        method_used = "Mean/Std"
    out["z_score"] = z

    # ---- percentile threshold
    thr = np.percentile(out["z_score"].values, pct)
    out["is_anomaly"] = (out["z_score"] > thr).astype(int)

    # ---- consecutive filter (final flag)
    if min_run > 1:
        runs = consecutive_runs(out["is_anomaly"].values.astype(bool))
        out["run_len"] = runs
        out["is_anomaly_final"] = (out["run_len"] >= min_run).astype(int)
        final_count = int(out["is_anomaly_final"].sum())
    else:
        out["run_len"] = 0
        out["is_anomaly_final"] = out["is_anomaly"]
        final_count = int(out["is_anomaly"].sum())

    print(f"[Threshold] method={method_used} pct={pct:.2f} → z_thresh={thr:.3f}  "
          f"raw_anomalies={int(out['is_anomaly'].sum())}/{len(out)}  "
          f"final_anomalies(min_run={min_run})={final_count}/{len(out)}")

    # ---- attach per-feature error drivers (normalized scale), end-of-window
    if save_drivers and driver_chunks:
        drivers = np.concatenate(driver_chunks, axis=0)  # [N, C]
        for i, feat in enumerate(feat_cols):
            out[f"err_{feat}"] = drivers[:, i]

    # ---- optional raw-feature deltas vs baseline (raw_df_for_deltas required)
    if save_deltas and raw_df_for_deltas is not None:
        rawp = raw_df_for_deltas.copy()
        rawp = rawp[rawp["pitcher_id"] == pitcher_id]
        # baseline up to provided cutoff; default to earliest train cutoff or entire rawp
        if baseline_end_date is not None:
            base_df = rawp[rawp["game_date"] <= baseline_end_date]
        else:
            base_df = rawp
        if len(base_df) > 0:
            base_means = base_df.groupby("pitcher_id")[feat_cols].mean().iloc[0]
            # per-date means on scoring slice
            per_day = rawp.groupby("game_date")[feat_cols].mean().reset_index()
            # join to out by game_date
            out = out.merge(per_day, on="game_date", how="left", suffixes=("", "_day"))
            # compute deltas: day - baseline
            for feat in feat_cols:
                if feat in out.columns:
                    out[f"delta_{feat}"] = out[feat] - base_means.get(feat, np.nan)
        else:
            print("[Warn] Baseline set is empty for delta computation; skipping delta columns.")

    # Save CSV
    csv_path = os.path.join(out_dir, f"anomalies_pitcher_{pitcher_id}.csv")
    out.to_csv(csv_path, index=False)
    print(f"[Saved] {csv_path}")

    # Plot timeline (scatter final anomalies after min_run filter)
    plt.figure(figsize=(11, 4))
    plt.plot(out["z_score"].values)
    plt.axhline(thr, linestyle="--")
    mask = out["is_anomaly_final"].values.astype(bool)
    an_idx = np.where(mask)[0]
    if len(an_idx) > 0:
        plt.scatter(an_idx, out["z_score"].values[an_idx], marker="o")
    plt.title(f"Pitcher {pitcher_id} anomaly z-scores (end-of-window)")
    plt.xlabel("Window end (chronological)")
    plt.ylabel("z-score (reconstruction error)")
    plot_path = os.path.join(out_dir, f"anomaly_timeline_pitcher_{pitcher_id}.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=160)
    plt.close()
    print(f"[Saved] {plot_path}")

    return csv_path, plot_path


# -----------------------
# Main
# -----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretty_plots", action="store_true",
                    help="Also save nicer timeline/per-game plots with date axis.")
    parser.add_argument("--injury_date", type=str, default=None,
                        help="Optional YYYY-MM-DD vertical marker on plots.")
    parser.add_argument("--roll_median", type=int, default=5,
                        help="Rolling-median window for pretty timeline (0 to disable).")
    parser.add_argument("--title_suffix", type=str, default="",
                        help="Optional text appended to plot titles.")
    parser.add_argument("--per_game_plot", action="store_true",
                        help="Also save a per-game max z-score plot.")

    parser.add_argument("--csv", type=str, required=True, help="Path to per-pitch CSV")
    parser.add_argument("--out_dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--pitcher_id", type=int, required=True, help="Pitcher ID to plot/score")
    parser.add_argument("--seq_len", type=int, default=64, help="Sequence length (window size)")
    parser.add_argument("--stride", type=int, default=1, help="Window stride")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--hid", type=int, default=64, help="Bottleneck channels")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_frac", type=float, default=0.2, help="Fraction of TRAIN dates used as validation")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    # Sensitivity/robustness knobs
    parser.add_argument("--pct", type=float, default=97.5, help="Percentile for adaptive threshold (e.g., 97.5)")
    parser.add_argument("--ema_alpha", type=float, default=0.2, help="EMA smoothing alpha for errors (0..1)")
    parser.add_argument("--z_method", type=str, default="std", choices=["std", "mad"],
                        help="z-score method: 'std' (mean/std) or 'mad' (robust median/MAD)")
    parser.add_argument("--min_run", type=int, default=1, help="Require >=N consecutive windows to flag (final)")
    # Time filters
    parser.add_argument("--train_end_date", type=str, default=None,
                        help="Use rows with game_date <= this date for TRAIN/VAL (YYYY-MM-DD)")
    parser.add_argument("--score_start_date", type=str, default=None,
                        help="Score only rows with game_date >= this date (YYYY-MM-DD)")
    # Pitch-type awareness (optional)
    parser.add_argument("--group_by_pitch_type", action="store_true",
                        help="Fit/apply scalers per (pitcher_id, pitch_type_raw) instead of pitcher-only")
    # Interpretability outputs
    parser.add_argument("--save_drivers", action="store_true",
                        help="Add per-feature end-step squared error columns (err_<feat>) to anomalies CSV")
    parser.add_argument("--save_deltas", action="store_true",
                        help="Add delta_<feat> columns vs baseline raw means to anomalies CSV")
    # Feature selection
    parser.add_argument(
        "--feature_cols",
        type=str,
        default="release_speed,release_spin_rate,release_pos_x,release_pos_z,extension,pfx_x,pfx_z",
        help="Comma-separated numeric feature columns (before one-hot)"
    )
    parser.add_argument("--one_hot_pitch_type", action="store_true", help="One-hot encode pitch_type")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)

    device = torch.device(
        "cuda" if (args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())) else "cpu"
    )
    print(f"Using device: {device}")

    # 1) Load CSV
    df = pd.read_csv(args.csv)

    # Common aliases to reduce friction
    alias_map = {}
    if "pitcher_id" not in df.columns and "pitcher" in df.columns:
        alias_map["pitcher"] = "pitcher_id"
    if "extension" not in df.columns and "release_extension" in df.columns:
        alias_map["release_extension"] = "extension"
    if alias_map:
        df = df.rename(columns=alias_map)
        print(f"[Alias] Renamed columns: {alias_map}")

    # Coerce date
    if "game_date" not in df.columns:
        raise ValueError("CSV must include 'game_date' column.")
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")

    # Required columns
    for c in ["pitcher_id", "game_date", "pitch_number"]:
        if c not in df.columns:
            raise ValueError(f"CSV missing required column: {c}")

    # Preserve raw pitch_type for grouping (before one-hot)
    if "pitch_type" in df.columns:
        df["pitch_type_raw"] = df["pitch_type"].astype(str)

    # Optional one-hot
    added_cols: List[str] = []
    if args.one_hot_pitch_type:
        df, added_cols = one_hot_pitch_type(df, col="pitch_type")

    feat_cols = [c.strip() for c in args.feature_cols.split(",") if c.strip()]
    feat_cols += added_cols

    missing = [c for c in feat_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Feature column(s) not found in CSV: {missing}")

    # Drop NaNs only where needed
    before_rows = len(df)
    df = df.dropna(subset=feat_cols + ["pitcher_id", "game_date", "pitch_number"]).copy()
    print(f"[Load] rows={before_rows} -> after dropna={len(df)}  "
          f"date range: {df['game_date'].min().date()} → {df['game_date'].max().date()}")

    # 2) Choose time windows for TRAIN/VAL and SCORING
    if args.train_end_date:
        train_pool = df[df["game_date"] <= pd.to_datetime(args.train_end_date)].copy()
        print(f"[Train pool] ≤ {args.train_end_date}  rows={len(train_pool)}")
    else:
        train_pool = df.copy()
        print(f"[Train pool] entire CSV  rows={len(train_pool)}")

    if args.score_start_date:
        score_base = df[df["game_date"] >= pd.to_datetime(args.score_start_date)].copy()
        print(f"[Score base] ≥ {args.score_start_date}  rows={len(score_base)}")
    else:
        score_base = df.copy()
        print(f"[Score base] entire CSV  rows={len(score_base)}")

    if train_pool.empty:
        raise ValueError("Training pool is empty after applying --train_end_date filter.")
    if score_base.empty:
        raise ValueError("Scoring pool is empty after applying --score_start_date filter.")

    # 3) Split TRAIN/VAL by date within the training pool (avoid leakage)
    train_df, val_df = split_train_val_by_date(train_pool, date_col="game_date",
                                               val_frac=args.val_frac, seed=args.seed)

    print(f"[Split] TRAIN rows={len(train_df)}  VAL rows={len(val_df)}")

    # 4) Fit per-group scalers on TRAIN, apply to VAL and SCORING
    pid2scaler, global_scaler = fit_group_scalers(
        train_df, feat_cols, group_by_pitch_type=args.group_by_pitch_type, type_col="pitch_type_raw"
    )
    train_df_n = apply_group_scalers(train_df, feat_cols, pid2scaler, global_scaler,
                                     group_by_pitch_type=args.group_by_pitch_type, type_col="pitch_type_raw")
    val_df_n   = apply_group_scalers(val_df,   feat_cols, pid2scaler, global_scaler,
                                     group_by_pitch_type=args.group_by_pitch_type, type_col="pitch_type_raw")
    score_df_n = apply_group_scalers(score_base, feat_cols, pid2scaler, global_scaler,
                                     group_by_pitch_type=args.group_by_pitch_type, type_col="pitch_type_raw")

    # 5) Build datasets (concatenate windows from all pitchers)
    def build_concat_windows(df_src: pd.DataFrame, seq_len: int, stride: int) -> Dataset:
        all_wins = []
        for pid, grp in df_src.groupby("pitcher_id"):
            if len(grp) >= seq_len:
                ds = PitchWindows(grp, feat_cols, seq_len=seq_len, stride=stride)
                if len(ds) > 0:
                    all_wins.append(ds)
        if not all_wins:
            raise ValueError("No pitcher had enough pitches to build at least one window. Lower --seq_len?")
        X_list, idx_list = [], []
        for ds in all_wins:
            for i in range(len(ds)):
                x, idx = ds[i]
                X_list.append(x.unsqueeze(0))  # [1, C, T]
                idx_list.append(idx)
        X = torch.cat(X_list, dim=0)  # [N, C, T]
        idxs = torch.tensor(idx_list, dtype=torch.long)
        class _MemDS(Dataset):
            def __len__(self): return X.shape[0]
            def __getitem__(self, i): return X[i], idxs[i]
        return _MemDS()

    train_ds = build_concat_windows(train_df_n, args.seq_len, args.stride)
    val_ds   = build_concat_windows(val_df_n,   args.seq_len, args.stride)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, drop_last=False)

    # 6) Model
    c_in = len(feat_cols)
    model = TcnAutoencoder(c_in=c_in, hid=args.hid, p=args.dropout).to(device)
    print(model)

    # 7) Train
    _ = train_tcn(model, train_loader, val_loader, device,
                  epochs=args.epochs, lr=args.lr, wd=1e-6, grad_clip=1.0)

    # 8) Load best and compute anomalies for chosen pitcher on SCORING set
    model.load_state_dict(torch.load(os.path.join(args.out_dir, "best_tcn.pth"), map_location=device))

    # Prepare raw df for delta computation (optional)
    raw_df_for_deltas = None
    baseline_cutoff = None
    if args.save_deltas:
        raw_df_for_deltas = df.copy()
        baseline_cutoff = pd.to_datetime(args.train_end_date) if args.train_end_date else None

    csv_path, plot_path = score_anomalies_for_pitcher(
        model=model,
        df=score_df_n,
        feat_cols=feat_cols,
        pitcher_id=args.pitcher_id,
        seq_len=args.seq_len,
        stride=args.stride,
        device=device,
        out_dir=args.out_dir,
        pct=args.pct,
        ema_alpha=args.ema_alpha,
        z_method=args.z_method,
        min_run=args.min_run,
        save_drivers=args.save_drivers,
        save_deltas=args.save_deltas,
        raw_df_for_deltas=raw_df_for_deltas,
        baseline_end_date=baseline_cutoff,
    )

    print("Done.")
        # Optional: pretty plots that do NOT replace the original outputs
    if args.pretty_plots:
        df_out = pd.read_csv(csv_path, parse_dates=["game_date"])
        base_title = f"Pitcher {args.pitcher_id} anomaly z-scores{(' — ' + args.title_suffix) if args.title_suffix else ''}"
        pretty_path = os.path.join(args.out_dir, f"pretty_timeline_pitcher_{args.pitcher_id}.png")
        _pretty_timeline(df_out, base_title, args.pct, pretty_path,
                         injury_date=args.injury_date, roll=args.roll_median)

        if args.per_game_plot:
            per_game_path = os.path.join(args.out_dir, f"per_game_pitcher_{args.pitcher_id}.png")
            _per_game_max_plot(df_out, base_title, args.pct, per_game_path,
                               injury_date=args.injury_date)

