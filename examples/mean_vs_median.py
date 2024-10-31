"""
================
Mean vs Median
================
"""

import os
import sys
import numpy as np

import easy_mpl
from easy_mpl import plot

import SeqMetrics
from SeqMetrics import mse, med_seq_error, mae, median_abs_error, mape, mdape, me, mde

# %%
print('python version: ', sys.version)
print('OS Name: ', os.name)
print('numpy: ', np.__version__)
print('easy_mpl: ', easy_mpl.__version__)
print('SeqMetrics: ', SeqMetrics.__version__)

# %%

true = np.random.random(100)
pred = np.random.random(100)
pred_o = pred.copy()
pred_o[2] = 40

# %%

plot(true,'--.', show=False, label="True")
plot(pred,'--.', show=False, label="Predicted")
plot(pred_o, '--.', show=False, label="Predicted with outlier")

# %%

print("Simple Error")
print("Mean: ", round(me(true, pred), 2), '-->', round(me(true, pred_o), 2))
print("Median: ", round(mde(true, pred), 2), '-->', round(mde(true, pred_o), 2))

# %%

print("Squared Error")
print(round(mse(true, pred), 2), '-->', round(mse(true, pred_o), 2))
print(round(med_seq_error(true, pred),2), '-->', round(med_seq_error(true, pred_o), 2))

# %%

print("Asbolute Error")
print("Mean: ", round(mae(true, pred), 2), '-->', round(mae(true, pred_o), 2))
print("Median: ", round(median_abs_error(true, pred), 2), '-->', round(median_abs_error(true, pred_o), 2))

# %%

print("Abolute Percentage Error")
print("Mean:", round(mape(true, pred), 2), '-->', round(mape(true, pred_o), 2))
print("Median:", round(mdape(true, pred), 2), '-->', round(mdape(true, pred_o), 2))

# %%
# So the erros based on mean are more sensitive to outliers than the errors based upon median.