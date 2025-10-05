
from pybaseball import statcast
import pandas as pd

df = statcast(start_dt="2024-03-28", end_dt="2024-04-15") 
if "release_extension" in df.columns and "extension" not in df.columns:
    df = df.rename(columns={"release_extension":"extension"})

out = "/Users/babyziba/Desktop/pitches.csv"
df.to_csv(out, index=False)
print("Wrote:", out, "rows:", len(df))
