# Date: 18 Jan 2024
# Author: Linus Lind
# LICENSED UNDER: GNU General Public GPLv3 License
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math as m
from scipy.ndimage import gaussian_filter
from saturation_gradient import file_reader
from saturation_gradient import grad_avg
from saturation_gradient import sat_gradient
from saturation_gradient import min_max_scaler
################################################################################

def main():
    # measured degrees:
    degs = [0, 15, 30, 45, 60, 75, 88, 90, 92,\
            105, 120, 135, 150, 165, 180, 270]
    ############################################################################
    #creating lots of pd dataframes to store data:
    applied, intensity, intensity_grad, \
        intensity_grad_adj, intensity_min_max,\
            intensity_grad_min_max,\
            applied_dir            = [pd.DataFrame(), pd.DataFrame(), 
                                      pd.DataFrame(), pd.DataFrame(), 
                                      pd.DataFrame(), pd.DataFrame(),
                                      pd.DataFrame()]
    ############################################################################
    for deg in degs: # process data for each degree
        ########################################################################
        col = f'{deg}deg' # column name, 

        # read data
        applied[col] = file_reader(f'{col}/loop_data.txt', 0, 9)
        intensity[col] = file_reader(f'{deg}deg/loop_data.txt', 10, 24)

        ########################################################################
        # An insanely hacky trick of dividing the data into two parts and 
        # min max scaling them separately to reduce the drift effect.
        # First it determines, whether the applied field is increasing -> 1 or 
        # decreasing -> 0 by calculating a rolling average of the gradient and
        # converting it to a binary mapping. [0, 0, 0,... 1, 1, 1] etc.
        # Then zeros are mapped to nans so that the min_max_scaler function
        # scales only the values not nan.
        applied_dir[col] = np.gradient(applied[col])
        bin_grad = grad_avg(applied_dir[col].values, 5)
        for i, el in enumerate(bin_grad):
            if el<0: bin_grad[i] = 0
            else: bin_grad[i] = 1
        # Multiply with the data
        scale1st = np.multiply(intensity[col].values, bin_grad)
        # Map zeros to nans 
        scale1st[scale1st==0] = np.nan
        scale1st = min_max_scaler(pd.Series(scale1st))
        scale1st = scale1st.fillna(0)
        # binary inverse "bit flip" (1 - binary_mapping):
        scale2nd = np.multiply(intensity[col].values, 1 - bin_grad)
        scale2nd[scale2nd==0] = np.nan
        scale2nd = min_max_scaler(pd.Series(scale2nd))
        scale2nd[scale2nd==np.nan] = 0
        scale2nd = scale2nd.fillna(0)
        combined = pd.Series(scale1st.values + scale2nd.values)
        intensity_min_max[col] = min_max_scaler(intensity[col] * bin_grad)
        intensity_min_max[col] = min_max_scaler(combined)
        ########################################################################

        # calculate gradient
        intensity_grad[col] = np.gradient(intensity[col], applied[col])
        grads = sat_gradient(intensity_grad[col].values, 8)
        print(col, np.mean(grads))
        dir_grads = np.zeros(np.shape(intensity_grad[col].values))
        print(intensity_grad[col].values)
        roll_avg_grad = grad_avg(intensity_grad[col].values, 10)
        for i, el in enumerate(roll_avg_grad):
            if el > 0: dir_grads[i] = 1
            else: dir_grads[i] = -1
        print(dir_grads)
        grad = np.multiply(np.mean(grads), dir_grads)
        print(grad)
        #print(np.sort(np.abs(intensity_grad[col])))
        #sorted_grad = np.sort(np.abs(intensity_grad[col]))
        #avg_grad = np.mean(sorted_grad[3:13])
        #print(avg_grad)
        #intensity_grad_min_max[col] = min_max_scaler(intensity_grad[col])
        #grads = intensity_grad[col].values
        #grads_min_max = intensity_grad_min_max[col].values
        #print(grads_min_max)
       # grads = grad_avg(grads, 6)
        grad = np.multiply(applied[col], grad)

        intensity_grad_adj[col] = intensity[col].values - grad
        intensity_grad_adj[col] = min_max_scaler(intensity_grad_adj[col])
    plot = plt.plot(applied['0deg'], intensity_grad_adj['0deg'])
    plt.show()
    plot = plt.plot(applied['0deg'], intensity_min_max['0deg'])
    plt.show()
    # save to csv boilerplate:
    applied.to_csv('applied.csv', index=None)
    applied_dir.to_csv('applied_dir.csv', index=None)
    ############################################################################
    intensity.to_csv('intensity.csv', index=None)
    intensity_grad.to_csv('intensity_grad.csv', index=None)
    intensity_grad_adj.to_csv('intensity_grad_adj.csv', index=None)
    intensity_min_max.to_csv('intensity_min_max.csv', index=None)

if __name__ == "__main__":
	main()