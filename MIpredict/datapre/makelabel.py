import glob
import numpy as np
np.set_printoptions(threshold=np.sys.maxsize)

label_files_all = sorted(glob.glob("/Users/fan/Documents/Data/EEG/BCI_CompetitionIV_2a/BCICIV_2a_gdf/sourcedata/trainlabel/*.npy"))#读取数据

files_data = []
for i in label_files_all:
    print('--i--', np.load(i))
    for j in range(0, np.load(i).shape[0]):
        np_data = np.load(i)
        files_data.append(np_data[j])
label_data = np.asarray(files_data)
print(label_data)
print(label_data.shape)

#np.save('/Users/fan/Documents/Data/EEG/BCI_CompetitionIV_2a/BCICIV_2a_gdf/sourcedata/testlabel/testlabel', label_data)
box = np.load('/Users/fan/Documents/Data/EEG/BCI_CompetitionIV_2a/BCICIV_2a_gdf/sourcedata/trainlabel/label.npy')

res = box - 7
print('-res-', res)
np.save('/Users/fan/Documents/Data/EEG/BCI_CompetitionIV_2a/BCICIV_2a_gdf/sourcedata/trainlabel/trainlabel', res)

