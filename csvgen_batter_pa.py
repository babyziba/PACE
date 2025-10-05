
from pybaseball import statcast_batter
import pandas as pd
import numpy as np
import argparse

def make_pa_table(df):

    swing_set   = {"foul","foul_tip","foul_bunt","swinging_strike","swinging_strike_blocked","hit_into_play","hit_into_play_no_out","hit_into_play_score"}
    contact_set = {"foul","foul_tip","foul_bunt","hit_into_play","hit_into_play_no_out","hit_into_play_score"}

    called_strike_set = {"called_strike"}
    df["is_swing"]  = df["description"].isin(swing_set).astype(int)
    df["is_contact"]= df["description"].isin(contact_set).astype(int)
    df["is_called_strike"] = df["description"].isin(called_strike_set).astype(int)


    in_zone = []
    for _,r in df[["plate_x","plate_z","sz_top","sz_bot"]].fillna(0).iterrows():
        px, pz, zt, zb = r.plate_x, r.plate_z, r.sz_top or 3.5, r.sz_bot or 1.5
        in_zone.append(int(abs(px) <= 0.83 and zb <= pz <= zt))
    df["is_zone"] = in_zone


    df["has_bbe"] = (~df["launch_speed"].isna()).astype(int)
    df["launch_speed"] = df["launch_speed"].fillna(0.0)
    df["launch_angle"] = df["launch_angle"].fillna(0.0)

 
    grp_keys = ["game_pk","at_bat_number"]
    g = df.groupby(grp_keys, as_index=False)


    pa = g.agg({
        "game_date":"first",
        "batter":"first",
        "pitch_number":"count",         
        "is_swing":"mean",
        "is_contact":"mean",
        "is_zone":"mean",
        "launch_speed":"mean",
        "launch_angle":"mean",
        "has_bbe":"mean",
        "estimated_woba_using_speedangle":"mean",
    }).rename(columns={
        "pitch_number":"pa_pitches",
        "is_swing":"swing_rate",
        "is_contact":"contact_rate",
        "is_zone":"zone_rate",
        "launch_speed":"avg_ev",
        "launch_angle":"avg_la",
        "has_bbe":"bbe_rate",
        "estimated_woba_using_speedangle":"avg_xwoba"
    })

    pa = pa.sort_values(["game_date","game_pk","at_bat_number"]).reset_index(drop=True)
    pa["pa_order"] = pa.groupby("batter").cumcount() + 1


    pa = pa.rename(columns={"batter":"pitcher_id","pa_order":"pitch_number"})
    return pa[[
        "pitcher_id","game_date","pitch_number",
        "pa_pitches","swing_rate","contact_rate","zone_rate",
        "avg_ev","avg_la","bbe_rate","avg_xwoba"
    ]]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--batter_id", type=int, required=True)
    ap.add_argument("--start", type=str, required=True)
    ap.add_argument("--end", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    df = statcast_batter(start_dt=args.start, end_dt=args.end, player_id=args.batter_id)
    if "game_date" in df.columns:
        df["game_date"] = pd.to_datetime(df["game_date"])
    pa = make_pa_table(df)
    pa.to_csv(args.out, index=False)
    print(f"Wrote: {args.out} rows: {len(pa)} for batter_id={args.batter_id}")
