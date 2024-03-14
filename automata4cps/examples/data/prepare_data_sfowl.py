# This code was used to create log1.csv and log2.csv of the conveyors system from sfowl
import datetime
import os
import pandas as pd

log1 = pd.read_csv(os.path.join("log1.csv"))
log2 = pd.read_csv(os.path.join("log2.csv"))
data = [log1, log2]
discrete_cols = [c for c in data[0].columns if c.lower()[-5:] == '_ctrl']
cont_cols = [c for c in data[0].columns if c.lower()[-5:] != '_ctrl' and c != "timestamp"]

print(pd.to_datetime("0001-1-1"))#  + pd.to_timedelta(log1["timestamp"], unit="s"))
