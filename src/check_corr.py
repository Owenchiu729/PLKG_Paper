import numpy as np
from scipy.stats import pearsonr

data = np.load("run_001_stacked.npy")
x, y = data[0], data[1]

corrs = []
for i in range(2000):  # sample first 2000 only
    c, _ = pearsonr(x[i], y[i])
    corrs.append(c)

print("Average correlation: ", np.nanmean(corrs))
