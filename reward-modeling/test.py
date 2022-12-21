import pandas as pd
import wandb

data = {"users": ["geoff", "juergen", "ada"], "feature_01": [1, 117, 42]}
df = pd.DataFrame(data)

tbl = wandb.Table(data=df)
assert all(tbl.get_column("users") == df["users"])
assert all(tbl.get_column("feature_01") == df["feature_01"])