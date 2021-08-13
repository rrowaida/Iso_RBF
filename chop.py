import sys
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

rbfunc = 1
dataset = int(sys.argv[1])
neighbors_num = int(sys.argv[2])
reduced_n = int(sys.argv[3])

tags = ["chairs", "lamps", "tables"] 
j=0

X = pd.read_csv('outputs' + '/' + tags[dataset-1] + '_neighbors_' + str(neighbors_num) + '_rbf_' + str(rbfunc) + '_reduced_' + str(reduced_n) + '_recons_1d_change_x5.csv', header = None).to_numpy()
X_new_df = np.zeros([100,128])
for i in range(0,400,4):
    X_new_df[j,:] = X[i,:]
    #print(X_new_df[j,1])
    j = j + 1
#print(X_new_df.shape)    
name2 = 'outputs' + '/' + tags[dataset-1] + '_neighbors_' + str(neighbors_num) + '_rbf_' + str(rbfunc) + '_reduced_' + str(reduced_n) + '_recons_1d_change_x5_chop.csv'
np.savetxt(name2, X_new_df, delimiter = ',')
