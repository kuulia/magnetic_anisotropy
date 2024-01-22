import pandas as pd
import numpy as np

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


def sat_gradient(grad_arr: np.ndarray, \
                 width: int) -> np.ndarray:
    grad = grad_avg(grad_arr, width)
    grad_mag = np.abs(grad)
    
    grad_mag = min_max_scaler(pd.Series(grad_mag)).values
    grad_dir = np.zeros(np.shape(grad_mag))
    for i, el in enumerate(grad_mag):
        if el<0.2: grad_dir[i] = 0
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
    mean_grads = []
    for region in list_idx:
        mean_grads.append(np.mean(np.abs(grad[region])) / 2)
    return mean_grads

def main():

    intensity = pd.read_csv('intensity.csv')
    intensity_grad = pd.read_csv('intensity_grad.csv')
    #sat_gradient(intensity['0deg'])
    #print(grad_avg(intensity['0deg'].values, 10))
    sat_gradient(intensity_grad['0deg'].values, 5)
    #print(grad_avg(intensity_grad['0deg'].values, 5))
if __name__ == "__main__":
    main()