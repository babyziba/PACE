from pybaseball import statcast
import pandas as pd

# Example: 2024 season data for ALL pitchers between March 28 and April 15
df = statcast(start_dt="2024-03-28", end_dt="2024-04-15")

print(df.head())
df.to_csv("pitches.csv", index=False)