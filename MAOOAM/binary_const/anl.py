import numpy as np

files = [
    "mean_b_cov_0d8c560.npy",
    "mean_b_cov_6682eed.npy",
    "mean_b_cov_c2c7c0f.npy",
    "mean_b_cov_b0b2d64.npy"]

for f in files:
    a = np.load(f)
    print(f)
    print(np.linalg.cond(a))
