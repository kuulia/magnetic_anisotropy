# Date: 18 Jan 2024
# Author: Linus Lind
# LICENSED UNDER: GNU General Public GPLv3 License
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math as m
################################################################################
# helper function for reading files
def file_reader(file: str, start: int, end: int) -> pd.DataFrame:
    read = open(file)
    lines = read.readlines()
    data = []
    for line in lines: 
        #just read the chars of the line specified
        data.append(line[start:end].replace(' ', '').replace(',', '.'))
    #remove column header
    data.pop(0)
    df = pd.DataFrame()
    df['Field(mT)'] = data
    df = df.astype(float)
    return df

# min max scaling function. scales data from range [any, any] to [-1, 1],
# where max(data) -> 1, min(data) -> -1
def min_max_scaler(ser: pd.Series) -> pd.Series:
    max_val, min_val = np.max(ser), np.min(ser)
    return ser.apply(lambda x: 2 * (x - min_val) / (max_val - min_val) - 1)

def main():
    # measured degrees:
    degs = [0, 15, 30, 45, 60, 75, 88, 90, 92,\
            105, 120, 135, 150, 165, 180, 270]
    ############################################################################
    #creating lots of pd dataframes to store data:
    applied, intensity, intensity_grad, intensity_grad_adj,\
        intensity_min_max, = [pd.DataFrame(), pd.DataFrame(), pd.DataFrame(),
                              pd.DataFrame(), pd.DataFrame(),]
    ############################################################################
    for deg in degs: # process data for each degree
        col = f'{deg}deg' # column name, 
        ########################################################################
        # read data
        applied[col] = file_reader(f'{col}/loop_data.txt', 0, 9)
        ########################################################################
        # read data
        intensity[col] = file_reader(f'{deg}deg/loop_data.txt', 10, 24)
        # apply min max scaler
        intensity_min_max[col] = min_max_scaler(intensity[col])
        # calculate gradient
        intensity_grad[col] = np.gradient(intensity_min_max[col], 1)
        #x = np.where(np.abs(intensity_grad[col]) <= 0.2, 0, 1)
        grads = intensity_grad[col].values
        # hacky convolution trick
        N = len(grads)
        grads = np.convolve(grads, np.ones(N) / N, mode='valid')
        test = (np.max(applied[col].values) + 1) * np.ones(len(applied[col].values)) - np.abs(applied[col].values)
        #print(test)
        magic_number = 15
        grads = np.multiply(test, magic_number * grads)
        print(grads)
        # 3: substract saturation gradient
        intensity_grad_adj[col] = intensity_min_max[col] - grads
    plot = plt.plot(applied['90deg'], intensity_grad_adj['90deg'])
    plt.show()
    # save to csv boilerplate:
    applied.to_csv('applied.csv', index=None)
    ############################################################################
    intensity.to_csv('intensity.csv', index=None)
    intensity_grad.to_csv('intensity_grad.csv', index=None)
    intensity_grad_adj.to_csv('intensity_grad_adj.csv', index=None)
    intensity_min_max.to_csv('intensity_min_max.csv', index=None)

if __name__ == "__main__":
	main()