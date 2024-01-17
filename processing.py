import pandas as pd

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

def main():
    degs = [0, 15, 30, 45, 60, 75, 88, 90, 92,\
            105, 120, 135, 150, 165, 180, 270]
    applied = pd.DataFrame()
    intensity = pd.DataFrame()
    for deg in degs:
        applied[f'{deg}deg'] = file_reader(f'{deg}deg/loop_data.txt', 0, 9)
        intensity[f'{deg}deg'] = file_reader(f'{deg}deg/loop_data.txt', 10, 24)
    applied.to_csv('applied.csv', index=None)
    intensity.to_csv('intensity.csv', index=None)

if __name__ == "__main__":
	main()