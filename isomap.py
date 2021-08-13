import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt

from sklearn.manifold import Isomap

def plotting(arr, pad, label_x, label_y, x_col, y_col, txt, save_file_name):
    plt.plot(arr[:,x_col], arr[:,y_col], "r.", ms=5)
    plt.axis([min(min(arr[:,x_col]), min(arr[:,y_col]))-pad, max(max(arr[:,x_col]), max(arr[:,y_col]))+pad, min(min(arr[:,x_col]), min(arr[:,y_col]))-pad, max(max(arr[:,x_col]), max(arr[:,y_col]))+pad])
    plt.xlabel(label_x)
    plt.ylabel(label_y)
    #plt.show()
    plt.figtext(0.5, 0.9, txt, wrap=True, horizontalalignment='center', fontweight="bold", fontsize=12, color="green")
    p_lbl1 = save_file_name
    plt.savefig(p_lbl1)
    plt.clf()  
    
dataset = int(sys.argv[1])
reduced_n = int(sys.argv[2])
neighbors_num = int(sys.argv[3])
    
tags = ["chairs", "lamps", "tables"]  

txt = "Reduced 2D " + tags[dataset-1] + " using Isomap of neighbors " + str(neighbors_num)

z = pd.read_csv('data' + '/' + 'z_vectors_' + tags[dataset-1] + '.csv', header = None).to_numpy()

# Calling isomap function using n_components value to the expected reduced dimension (usually 2)
embedding = Isomap(n_neighbors = neighbors_num, n_components=reduced_n)
z_transformed = embedding.fit_transform(z)
#print(z_transformed.shape)

#plot and save data of reduced dimension 
p_lbl3 = 'outputs' + '/' + tags[dataset-1] + '_neighbors_' + str(neighbors_num) + '_isomap_reduced_' + str(reduced_n) +  '.png'
plotting(z_transformed, 10, "X1", "X2", 0, 1, txt, p_lbl3)

#save the reduced data information in .csv file in the output folder
f_name2 = 'data' + '/' + tags[dataset-1] + '_neighbors_' + str(neighbors_num) + '_isomap_reduced_' + str(reduced_n) +'.csv'
np.savetxt(f_name2, z_transformed, delimiter = ',')
