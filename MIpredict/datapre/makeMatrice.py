import glob
import numpy as np
#np.set_printoptions(threshold=np.sys.maxsize)

data_files_all = sorted(glob.glob("/Users/fan/Documents/Data/EEG/BCI_CompetitionIV_2a/BCICIV_2a_gdf/sourcedata/cwttraindata/*.npy"))#读取数据

files_data = []
for i in data_files_all:
    print(np.load(i).shape)
    for j in range(0, np.load(i).shape[0]):
        np_data = np.load(i)
        files_data.append(np_data[j])

data = np.asarray(files_data)
np.save('/Users/fan/Documents/Data/EEG/BCI_CompetitionIV_2a/BCICIV_2a_gdf/sourcedata/cwttraindata/cwtfeatures', data)
print('-data-', data)
print('-data-', data.shape)

