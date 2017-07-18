from __future__ import division
import numpy as np
from statsmodels import robust
import glob
import os
import sys

h_dir = '/home/peter/Desktop/Connect/rest/output/'

#Collect list of correlation matrices
corr_mat_list = glob.glob(h_dir + 'D_*_I/corrmat/global_correlation_aal_trans.csv')
corr_mat_list.sort()

subj_ids = [os.path.split(os.path.split(os.path.split(subj_info)[0])[0])[1] for subj_info in corr_mat_list]

#Read in correlation matrices. Return as list
corr_mats = [np.genfromtxt(corr_mat, delimiter = ',') for corr_mat in corr_mat_list]

#Vectorise lower correlation matrices without diagonal. Return as numpy array.
corr_vec = np.array([x[np.tril_indices_from(x, k = -1)] for x in corr_mats])

#Plot stacked histograms for corr_vec (helps to visualise issues)
fig, ax = plt.subplots()
[ax.hist(x, histtype = 'step', color = 'gray', alpha = 0.7) for x in corr_vec]
ax.vlines(np.mean(corr_vec), 0, 1, transform = ax.get_xaxis_transform(), colors='r', linestyles = 'dashed', label = 'Mean')
ax.vlines(np.median(corr_vec), 0, 1, transform = ax.get_xaxis_transform(), colors='b', linestyles = 'dashed', label = 'Median')
ax.legend()

#Plot vectors
plt.figure()
#plt.imshow(corr_vec, vmin = -1, vmax = 1, interpolation = 'None', cmap = 'RdYlBu_r'); plt.colorbar()
plt.imshow(corr_vec, vmin = -1, vmax = 1, interpolation = 'None', cmap = 'spectral_r'); plt.colorbar()
y_ticks = plt.yticks(range(len(subj_ids)), subj_ids, fontsize = 8)

#Compile variance measure
subj_var = np.var(corr_vec, axis = 1)

subj_var_median = np.median(subj_var) #Compute median
subj_var_mad = robust.mad(subj_var) #Compute MAD (mean absolute deviation)

#Plot
fig, ax = plt.subplots()
ax.plot(subj_var, '.')

#Plot lines showing +- 1 MAD from median
ax.plot(range(0,len(subj_var)), np.tile(subj_var_median + subj_var_mad, len(subj_var)), 'r-', label = 'median')  
ax.plot(range(0,len(subj_var)), np.tile(subj_var_median - subj_var_mad, len(subj_var)), 'r-')


#Plot lines showing +- 1 std from mean
ax.plot(range(0,len(subj_var)), np.tile(np.mean(subj_var) + np.std(subj_var), len(subj_var)), 'b-', label = 'mean')  
ax.plot(range(0,len(subj_var)), np.tile(np.mean(subj_var) - np.std(subj_var), len(subj_var)), 'b-')

ax.legend()

#Annote names
for i, txt in enumerate(subj_ids):
    ax.annotate(txt, [i, subj_var[i]], size = 'small', alpha = 0.5)














