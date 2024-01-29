# Date: 18 Jan 2024
# Author: Linus Lind
# LICENSED UNDER: GNU General Public GPLv3 License
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math as m
from scipy.ndimage import gaussian_filter
from scipy.signal import savgol_filter
from saturation_gradient import file_reader
from saturation_gradient import grad_avg
from saturation_gradient import sat_gradient
from saturation_gradient import min_max_scaler
from saturation_gradient import separate_scaler
from saturation_gradient import find_roots
################################################################################

def main():
    # measured degrees:
    degs = [0, 15, 30, 45, 60, 75, 88, 90, 92,\
            105, 120, 135, 150, 165, 180, 270]
    ############################################################################
    #creating pandas dataframes to store data:
    applied, intensity, \
        intensity_grad_adj, intensity_min_max = [pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), \
                                                 pd.DataFrame()]
    ###########################################################################
    # parameters for gradient algorithm
    params = [(3, -0.4), (7, -0.5), (4, -0.85), (7, 0.1), (7, 0.1), (7, -0.4),\
              (3, 0.8), (3, 0.8), (3, 0.8), (7, 0.2), (5, 0.2), (5, 0.2),\
              (5, 0.2), (5, 0.2), (5, 0.2), (5, 0.2)]
    # filter amount
    gauss_amt = 0.5
    savgol_amt = 0.7
    degs_and_params = dict.fromkeys(degs, params)
    for i, deg in enumerate(degs):
        degs_and_params[deg] = params[i]
    remanence = []
    coercive_field = []
    for deg in degs: # process data for each degree
        ########################################################################
        col = f'{deg}deg' # column name, 

        # a width parameter for rolling average calculations. The algorithmic
        # approach needs the width and threshold parameter to determine gradients
        # and to detect sharp increases
        width = degs_and_params[deg][0]
        threshold = degs_and_params[deg][1]
        # read data
        applied[col] = file_reader(f'{col}/loop_data.txt', 0, 9)
        intensity[col] = file_reader(f'{deg}deg/loop_data.txt', 10, 24)
        intensity_min_max[col] = separate_scaler(applied[col].values, \
                                            intensity[col].values)
        # calculate gradient
        grads = sat_gradient(intensity[col].values,
                             applied[col].values, threshold, width)
        grads = np.multiply(applied[col].values, grads)
        #print(col, grads)
        intensity_grad_adj[col] = intensity[col].values
        if deg not in [88, 90, 92, 105, 120, 135, 270]:
            intensity_grad_adj[col] = intensity_grad_adj[col].values - grads
            intensity_grad_adj[col] = (1 - gauss_amt) * intensity_grad_adj[col].values\
                + gauss_amt * gaussian_filter(intensity_grad_adj[col].values, 2)
            intensity_grad_adj[col] = (1 - savgol_amt) * intensity_grad_adj[col].values\
                + savgol_amt * savgol_filter(intensity_grad_adj[col].values, 5, 3)
            intensity_grad_adj[col] = separate_scaler(applied[col].values,
                                                        intensity_grad_adj[col].values)
        else:
            intensity_grad_adj[col] = (1 - gauss_amt) * intensity_grad_adj[col].values\
                + gauss_amt * gaussian_filter(intensity_grad_adj[col].values, 2)
            intensity_grad_adj[col] = (1 - savgol_amt) * intensity_grad_adj[col].values\
                + savgol_amt * savgol_filter(intensity_grad_adj[col].values, 5, 3)
            intensity_grad_adj[col] = separate_scaler(applied[col].values,
                                                        intensity_grad_adj[col].values)
        plt.plot(applied[col], intensity_grad_adj[col])
        plt.xlabel('Applied field $H$ (mT)')
        plt.ylabel('Intensity $I$ ($M/M_s$)')
        plt.savefig(f'{col}/grad_corr_plot.png')
        plt.close()
        plt.plot(applied[col], intensity[col])
        plt.xlabel('Applied field $H$ (mT)')
        plt.ylabel('Intensity $I$ ($M$)')
        plt.savefig(f'{col}/grad_raw_plot.png')
        plt.close()
        # calculate remanent magnetization
        remanence.append(np.mean(np.abs(find_roots(intensity_grad_adj[col].values, applied[col].values))))
        # calculate coercive field:
        intercepts = find_roots(applied[col].values, intensity_grad_adj[col].values)
        if len(intercepts) == 4:
            intercepts = np.array([intercepts[1], intercepts[3]])
        print(deg, intercepts)
        coercive_field.append(np.mean(np.abs(intercepts)))
    print(remanence)
    linspace_degs = np.linspace(0,270, num=19)
    # plot remanence:
    plt.plot(degs, remanence, 'bo--')
    plt.xlabel('Angle $\psi$ ($^\circ$)')
    plt.ylabel('Remanence')
    plt.xticks(linspace_degs, rotation=70)
    plt.tight_layout()
    plt.savefig('remanence.png')
    plt.close()

    # plot coercive field
    plt.plot(degs, coercive_field, 'ro--')
    plt.xlabel('Angle $\psi$ ($^\circ$)')
    plt.ylabel('Coercivity field $H_s$')
    plt.xticks(linspace_degs, rotation=70)
    plt.tight_layout()
    plt.savefig('coercive_field.png')
    plt.close()
    
    intensity_grad_adj.to_csv('intensity_grad_adj.csv',index=False)
if __name__ == "__main__":
	main()