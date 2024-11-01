"""
================
Mean vs Median
================
This script compares the errors based on mean and median.
It shows that the errors based on mean are more sensitive to outliers than the errors based upon median.
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
# Consider that we have two outputs from our model. One output is normal and the other has an outlier.
# We will compare the errors based on mean and median.

true = np.random.random(100)
pred = np.random.random(100)
pred_o = pred.copy()
pred_o[2] = 40

# %%
# First see that how the array ``pred_o`` has just one outlier.
# The outlier is at index 2.

plot(true,'--.', show=False, label="True")
plot(pred,'--.', show=False, label="Predicted")
plot(pred_o, '--.', show=False, label="Predicted with outlier")

# %%
# See the change in mean and median errors. The change in mean error is very high
# but the change in median error is very low.
print("Simple Error")
print("Mean: ", round(me(true, pred), 2), '-->', round(me(true, pred_o), 2))
print("Median: ", round(mde(true, pred), 2), '-->', round(mde(true, pred_o), 2))

# %%
# Similarly the change in mean squared error is very large but the change in 
# median squared error is very small.

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
# Therefore, if the data has outliers and we want to ignore them, we should use median based errors.
# On the other hand, if we want to focus on the outliers, we should use mean based errors.