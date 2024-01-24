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
from saturation_gradient import separate_scaler
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
        intensity_min_max[col] = separate_scaler(applied[col].values, \
                                            intensity[col].values)
        # calculate gradient
        intensity_grad[col] = np.gradient(intensity[col], applied[col])
        sat_grad = sat_gradient(intensity_grad[col].values, 8)
        grads = sat_grad
        grads = np.multiply(applied[col], grads)
        intensity_grad_adj[col] = intensity[col].values - grads
        intensity_grad_adj[col] = separate_scaler(applied[col].values,
                                                  intensity_grad_adj[col].values)
        #intensity_grad_adj[col] = min_max_scaler(intensity_grad_adj[col])
        plt.plot(applied[col], intensity_grad_adj[col])
        plt.savefig(f'{col}/grad_corr_plot.png')
        plt.close()
        plt.plot(applied[col], intensity_min_max[col])
        plt.savefig(f'{col}/grad_raw_plot.png')
        plt.close()
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