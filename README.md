# PACE
PACE is a deep learning based framework designed to detect anomalous behavior using a Temporal Convolutional Autoencoder trained on Statcast data.
PitchPulse / tcn_anomaly.py — Usage Guide
===========================================

SYNOPSIS
--------
Train a Temporal CNN Autoencoder on “normal” Statcast windows and flag anomalies.

Basic:
  python tcn_anomaly.py --csv <PATH.csv> --out_dir <OUTDIR> --pitcher_id <ID>     --seq_len 64 --epochs 30 --batch_size 128 --feature_cols "<comma-separated>"

With dates & adaptive threshold:
  python tcn_anomaly.py --csv <PATH.csv> --out_dir <OUTDIR> --pitcher_id <ID>     --train_end_date YYYY-MM-DD --score_start_date YYYY-MM-DD     --pct 97.5 --ema_alpha 0.20 --z_method std --min_run 1

Notes:
- The field name is --pitcher_id for both pitchers and batters (just an ID field).
- Script auto-aliases: pitcher -> pitcher_id, release_extension -> extension.
- Required columns: pitcher_id, game_date, pitch_number, and your selected features.


ARGUMENTS
---------
--csv PATH                  [REQUIRED] Input CSV of per-pitch (or per-PA) rows.
--out_dir PATH              Output directory (default: outputs). Will be created.
--pitcher_id INT            [REQUIRED] Player ID to score/plot (pitcher or batter).

# Windowing / training
--seq_len INT               Sliding window length in pitches (or PAs). Default: 64.
--stride INT                Window stride. Default: 1.
--batch_size INT            Training batch size. Default: 128.
--epochs INT                Training epochs. Default: 30.
--lr FLOAT                  Learning rate. Default: 1e-3.
--dropout FLOAT             Dropout in TCN blocks. Default: 0.10.
--hid INT                   Bottleneck channels (1×1 conv). Default: 64.
--seed INT                  Random seed. Default: 42.
--val_frac FLOAT            Fraction of TRAIN dates used for validation split. Default: 0.20.
--device {auto,cpu,cuda}    Device selection. Default: auto.

# Features
--feature_cols "CSV"        Comma-separated list of numeric features to use.
                            Pitcher default:
                              "release_speed,release_spin_rate,release_pos_x,release_pos_z,extension,pfx_x,pfx_z"
                            Batter (PA-level) example:
                              "pa_pitches,swing_rate,contact_rate,zone_rate,avg_ev,avg_la,bbe_rate,avg_xwoba"
--one_hot_pitch_type        Flag: one-hot-encode pitch_type column if present.
--group_by_pitch_type       Flag: build windows per pitch_type group (optional advanced).

# Time windows (to avoid leakage and focus scoring period)
--train_end_date YYYY-MM-DD  Use rows with date <= this for TRAIN/VAL. Optional.
--score_start_date YYYY-MM-DD Score rows with date >= this for SCORING. Optional.

# Scoring & flagging
--ema_alpha FLOAT           EMA smoothing alpha for reconstruction error (0..1).
                            Typical: 0.20–0.35. Default: 0.20.
--z_method {std,mad}        Z-score method: mean/STD ("std") or robust MAD ("mad").
                            Default: std.
--pct FLOAT                 Percentile threshold for anomalies (e.g., 95, 97.5). Default: 97.5.
--min_run INT               Require at least this many consecutive flagged windows.
                            Cuts single-window noise. Default: 1.

# Interpretability & visuals
--save_drivers              Flag: save per-feature error terms (err_<feature>) in CSV.
--save_deltas               Flag: save deltas vs baseline means (delta_<feature>) in CSV.
--pretty_plots              Flag: produce a styled timeline (plus the default one).
--title_suffix "TEXT"       Extra title text on the plot(s).

OUTPUTS
-------
OUTDIR/anomalies_pitcher_<ID>.csv
  Columns include:
    game_date, pitch_number, recon_error, z_score, is_anomaly
    (if --save_drivers) err_<feature> for each feature
    (if --save_deltas)  delta_<feature> for each feature

OUTDIR/anomaly_timeline_pitcher_<ID>.png
  Baseline timeline with threshold and flagged windows.

OUTDIR/pretty_timeline_pitcher_<ID>.png  (if --pretty_plots)
  Styled timeline with shaded anomaly bands and title suffix.


EXAMPLES
--------
# Walker Buehler (621111), 2022 case
python tcn_anomaly.py   --csv /path/buehler_2022.csv   --out_dir /path/outputs_buehler22   --pitcher_id 621111   --seq_len 64 --epochs 30 --batch_size 128   --one_hot_pitch_type   --train_end_date 2022-06-01 --score_start_date 2022-06-01   --pct 97.5 --ema_alpha 0.20 --z_method std --min_run 1   --feature_cols "release_speed,release_spin_rate,release_pos_x,release_pos_z,extension,pfx_x,pfx_z"

# Shohei Ohtani (660271), Aug 2023
python tcn_anomaly.py   --csv /path/ohtani_2023.csv   --out_dir /path/outputs_ohtani_2023   --pitcher_id 660271   --seq_len 32 --epochs 30 --batch_size 128   --one_hot_pitch_type   --train_end_date 2023-08-01 --score_start_date 2023-08-01   --pct 95.0 --ema_alpha 0.35 --z_method mad --min_run 2   --feature_cols "release_speed,release_spin_rate,release_pos_x,release_pos_z,extension,pfx_x,pfx_z"

# Batter example — Max Muncy (571970), 2024 (PA-level features)
python tcn_anomaly.py   --csv /path/muncy_2024_pa.csv   --out_dir /path/outputs_muncy_2024_batter   --pitcher_id 571970   --seq_len 24 --epochs 30 --batch_size 128   --train_end_date 2024-05-01 --score_start_date 2024-05-01   --pct 95.0 --ema_alpha 0.30 --z_method mad --min_run 2   --feature_cols "pa_pitches,swing_rate,contact_rate,zone_rate,avg_ev,avg_la,bbe_rate,avg_xwoba"   --pretty_plots --title_suffix "Max Muncy 2024"


TIPS
----
- If you see “No windows built” errors, lower --seq_len (e.g., 24) or widen the date range.
- For noisy periods, try --z_method mad and/or raise --min_run to 2.
- For mixed pitch types, prefer --one_hot_pitch_type (simple, effective).
- Ensure your CSV has no NaNs in required columns; the script drops rows with NaNs in selected features.
