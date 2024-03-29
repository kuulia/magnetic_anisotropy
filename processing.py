# Date: 01 Feb 2024
# Author: Linus Lind
# LICENSED UNDER: GNU General Public GPLv3 License
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.signal import savgol_filter
from scipy import constants
from saturation_gradient import file_reader
from saturation_gradient import sat_gradient
from saturation_gradient import separate_scaler
from saturation_gradient import find_roots
################################################################################

def main():
    # measured degrees:
    degs = [0, 15, 30, 45, 60, 75, 90,\
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
    # set these to 0.0 for calculating remanence and coercivity 
    # for plotting gauss = 0.5, savgol = 0.7 were used
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
        applied[col] = file_reader(f'datafiles/{col}/loop_data.txt', 0, 9)
        intensity[col] = file_reader(f'datafiles/{col}/loop_data.txt', 10, 24)
        intensity_min_max[col] = separate_scaler(applied[col].values, \
                                            intensity[col].values)
        # calculate gradient
        grads = sat_gradient(intensity[col].values,
                             applied[col].values, threshold, width)
        grads = np.multiply(applied[col].values, grads)
        intensity_grad_adj[col] = intensity[col].values
        if deg not in [88, 90, 105, 120, 135, 270]:
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
        plt.savefig(f'datafiles/{col}/grad_corr_plot.png')
        plt.close()
        plt.plot(applied[col], intensity[col])
        plt.xlabel('Applied field $H$ (mT)')
        plt.ylabel('Intensity $I$ ($M$)')
        plt.savefig(f'datafiles/{col}/grad_raw_plot.png')
        plt.close()
        # calculate remanent magnetization
        print(f'The remanence at {deg} is:', \
              np.round(find_roots(intensity_grad_adj[col].values, applied[col].values), 3))
        remanence.append(np.mean(np.abs(find_roots(intensity_grad_adj[col].values, applied[col].values))))
        # calculate coercive field:
        intercepts = find_roots(applied[col].values, intensity_grad_adj[col].values)
        if len(intercepts) == 4:
            intercepts = np.array([intercepts[1], intercepts[3]])
        print(f'The coercivity at {deg} is:', np.round(intercepts, 3))
        print('---------------------------------------------------')
        coercive_field.append(np.mean(np.abs(intercepts)))
    linspace_degs = np.linspace(0,270, num=19)
    # plot remanence:
    plt.plot(degs, remanence, 'bo--')
    plt.xlabel('Angle $\psi$ ($^\circ$)')
    plt.ylabel('Remanence ($M/M_s$)')
    plt.xticks(linspace_degs, rotation=70)
    plt.tight_layout()
    plt.savefig('datafiles/remanence.png')
    plt.close()

    # plot coercive field
    plt.plot(degs, coercive_field, 'ro--')
    plt.xlabel('Angle $\psi$ ($^\circ$)')
    plt.ylabel('Coercivity field $H_c$')
    plt.xticks(linspace_degs, rotation=70)
    plt.tight_layout()
    plt.savefig('datafiles/coercive_field.png')
    plt.close()
    
    # to csv boilerplate
    intensity_grad_adj.to_csv('datafiles/intensity_grad_adj.csv',index=False)
    intensity.to_csv('datafiles/intensity_raw.csv',index=False)
    intensity_min_max.to_csv('datafiles/intensity_min_max.csv',index=False)
    applied.to_csv('datafiles/applied_H.csv',index=False)

    # calculate the anisotropy constant from the hard axis data 
    col = '90deg'
    n = len(intensity_grad_adj[col].values)
    y = intensity_grad_adj[col].values[0:round(n/2)]
    x = applied[col].values[0:round(n/2)]
    idx = np.where(np.logical_and(x>=-0.6, x<=0.8))
    x = x[idx] * 795.77471545947673925 # conversion mT -> A / m
    M_sat = 1.2 * 10**(6) # (units A / m)
    y = y[idx] * M_sat # conversion m = M/M_sat -> M = m*M_sat (units A / m)
    lin_fit = np.polyfit(x, y, 1)
    mu_0 = constants.value('vacuum mag. permeability')
    K_mu = mu_0 * M_sat * lin_fit[0]

    y = intensity_grad_adj[col].values[0:round(n/2)]
    x = applied[col].values[0:round(n/2)]
    x = x[idx]
    y = y[idx]
    plt.plot(x, y, 'r*')
    lin_fit = np.polyfit(x, y, 1)
    print(f'The anisotropy constant is: {K_mu}')
    plt.plot(applied[col], intensity_grad_adj[col])
    plt.plot(np.linspace(-2, 3), (np.linspace(-2, 3) * lin_fit[0]) + lin_fit[1], '--', 
            color='orange')
    plt.ylim([-1.05, 1.05])
    plt.xlabel('Applied field $H$ (mT)')
    plt.ylabel('Intensity $I$ ($M/M_s$)')
    plt.savefig(f'datafiles/{col}/fit.png')
    plt.close()
if __name__ == "__main__":
	main()