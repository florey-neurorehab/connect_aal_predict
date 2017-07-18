from __future__ import division

import numpy as np
import nibabel
import pandas as pd
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.spatial.distance as distance
import os

d_dir = '/home/peter/Dropbox/post_doc/florey_leeanne/study_scripts/scratchpad/centroid'
#d_dir = '/home/orcasha/Dropbox/post_doc/florey_leeanne/study_scripts/scratchpad/centroid'

labels_file = os.path.join(d_dir, 'aal_pa_labels.txt')
aal_file = os.path.join(d_dir, 'aal_pa_3mm.nii')

#Load nii
im = nibabel.load(aal_file)
im_data = im.get_data()


#Make dataframe
aal_df = pd.DataFrame(data = np.genfromtxt(labels_file, dtype = str), columns=['region'])
aal_df['id'] = np.unique(im_data)[1:]
aal_df['xC'] = [int(np.median(np.where(im_data == n)[0])) for n in range(0, len(aal_df))]
aal_df['yC'] = [int(np.median(np.where(im_data == n)[1])) for n in range(0, len(aal_df))]
aal_df['zC'] = [int(np.median(np.where(im_data == n)[2])) for n in range(0, len(aal_df))]

#Plot 3d scatter
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(aal_df['xC'].as_matrix(), aal_df['yC'].as_matrix(), aal_df['zC'].as_matrix())

plt.show()

#Calculate distance matrix
pairwise_partial = distance.pdist(array(aal_df[['xC','yC','zC']]))
pairwise_mat = distance.squareform(pairwise_partial)

#Flatten, vectorise
distance_flat = pairwise_mat[np.tril_indices_from(pairwise_mat, k = -1)]


