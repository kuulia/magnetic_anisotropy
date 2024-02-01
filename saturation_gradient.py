# Date: 01 Feb 2024
# Author: Linus Lind
# LICENSED UNDER: GNU General Public GPLv3 License
# Import this file
import pandas as pd
import numpy as np

# helper function for reading measurement files
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

def grad_avg(arr: np.ndarray, width: int):
    avgs = np.zeros(np.shape(arr))
    for idx, _ in enumerate(arr):
        mean = np.sum(arr[idx:idx+width]) / min(width, len(arr[idx:idx+width]))
        avgs[idx] = mean
    return avgs

# min max scaling function. scales data from range [any, any] to [-1, 1],
# where max(data) -> 1, min(data) -> -1
def min_max_scaler(ser: pd.Series) -> pd.Series:
    max_val, min_val = np.max(ser), np.min(ser)
    return ser.apply(lambda x: 2 * (x - min_val) / (max_val - min_val) - 1)

########################################################################
# An insanely hacky trick of dividing the data into two parts and 
# min max scaling them separately to reduce the drift effect.
# First it determines, whether the applied field is increasing -> 1 or 
# decreasing -> 0 by calculating a rolling average of the gradient and
# converting it to a binary mapping. [0, 0, 0,... 1, 1, 1] etc.
# Then zeros are mapped to nans so that the min_max_scaler function
# scales only the values not nan.
def separate_scaler(applied: np.ndarray, intensity: np.ndarray):
    applied_dir = np.gradient(applied)
    bin_grad = grad_avg(applied_dir, 5)
    for i, el in enumerate(bin_grad):
        if el<0: bin_grad[i] = 0
        else: bin_grad[i] = 1
    # Multiply with the data
    scale1st = np.multiply(intensity, bin_grad)
    # Map zeros to nans 
    scale1st[scale1st==0] = np.nan
    scale1st = min_max_scaler(pd.Series(scale1st))
    scale1st = scale1st.fillna(0)
    # binary inverse "bit flip" (1 - binary_mapping):
    scale2nd = np.multiply(intensity, 1 - bin_grad)
    scale2nd[scale2nd==0] = np.nan
    scale2nd = min_max_scaler(pd.Series(scale2nd))
    scale2nd[scale2nd==np.nan] = 0
    scale2nd = scale2nd.fillna(0)
    combined = pd.Series(scale1st + scale2nd)
    scaled = min_max_scaler(combined)
    return scaled

def sat_gradient(arr: np.ndarray, \
                 applied_arr: np.ndarray, \
                 threshold: float, \
                 width: int) -> np.ndarray:
    grad_arr = np.gradient(arr, applied_arr)
    grad = grad_avg(grad_arr, width)
    grad_mag = np.abs(grad)
    grad_mag = min_max_scaler(pd.Series(grad_mag))
    grad_dir = np.zeros(np.shape(grad_mag))
    for i, el in enumerate(grad_mag):
        if el<threshold: grad_dir[i] = 0
        else: grad_dir[i] = 1
    list_idx = []
    idx = []
    for i, el in enumerate(grad_dir):
        if el == 0:
            idx.append(i)
        else: 
            if len(idx) != 0:
                list_idx.append(idx)
            idx = []
    if len(idx) != 0:
        list_idx.append(idx)
    grads = np.zeros(np.shape(grad))
    for region in list_idx:
        grad_mean = np.polyfit(applied_arr[region], arr[region], 1)[0]
        grads[region] = grad_mean
    zero_regions = np.where(grads==0)[0]
    seq = []
    temp = []
    for i, el in enumerate(zero_regions):
        if i < len(zero_regions) - 1:
            if zero_regions[i + 1] == el + 1:
                temp.append(el)
            else:
                temp.append(el)
                seq.append(temp)
                temp = []
    temp.append(el)
    seq.append(temp)
    for region in seq:  
        middle = round(len(region) / 2)
        for i, el in enumerate(region):
            if i < middle:
                grads[el] = grads[region[0] - 1]
            else:
                grads[el] = grads[region[len(region) - 1] + 1]
    return grads

# find 0-intercepts
def find_roots(x,y):
    s = np.abs(np.diff(np.sign(y))).astype(bool)
    return x[:-1][s] + np.diff(x)[s]/(np.abs(y[1:][s]/y[:-1][s])+1)