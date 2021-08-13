import sys
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

def euclidean(x, y):
    dist = np.sqrt(np.sum(np.square(np.subtract(x,y)),keepdims=True))
    return dist   

def rbf(x, y):
    eps = 0.001
    #return np.exp(-(euclidean(x, y)**2)*(eps**2))    
    return np.power(euclidean(x, y),1) 
rbfunc = 1
dataset = int(sys.argv[1])
neighbors_num = int(sys.argv[2])
reduced_n = int(sys.argv[3])

tags = ["chairs", "lamps", "tables"] 

X = pd.read_csv('data' + '/' + 'z_vectors_' + tags[dataset-1] + '.csv', header = None).to_numpy()
Y = pd.read_csv('data' + '/' + tags[dataset-1] + '_neighbors_' + str(neighbors_num) + '_isomap_reduced_' + str(reduced_n) +'.csv', header = None).to_numpy()

[n, D] = X.shape
[N2, d2] = Y.shape

K = np.zeros([n,n])
for i in range(n):
    for j in range(n):
        K[i,j] = rbf(Y[i,:], Y[j,:])
#print(K)
A = np.linalg.solve(K,X)
print(A.shape)
np.savetxt('data' + '/' + tags[dataset-1] + '_neighbors_' + str(neighbors_num) + '_rbf_' + str(rbfunc) + '_reduced_' + str(reduced_n) + '_A_matix.csv', A, delimiter = ',')
Y_new = pd.read_csv('data' + '/' + tags[dataset-1] + '_neighbors_' + str(neighbors_num) + '_reduced_' + str(reduced_n) + '_test_points_1d_change_x5.csv', header = None).to_numpy()    
X_new = np.zeros([n,D])

K_new = np.zeros([n,n])

for i in range(n):
    for j in range(n):
        K_new[i,j] = rbf(Y_new[i,:], Y[j,:])

for i in range(n):
    for j in range(D):
          X_new[i,j] = np.dot(np.transpose(A[:,j]),np.transpose(K_new[i,:]))
         
X_new_df = pd.DataFrame(X_new)

#Saving reconstructed test higher dimension data for error comparion
name2 = 'outputs' + '/' + tags[dataset-1] + '_neighbors_' + str(neighbors_num) + '_rbf_' + str(rbfunc) + '_reduced_' + str(reduced_n) + '_recons_1d_change_x5.csv'
np.savetxt(name2, X_new_df, delimiter = ',')
