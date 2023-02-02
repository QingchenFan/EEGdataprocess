
import numpy as np
import sklearn
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import cohen_kappa_score
from sklearn.datasets import load_digits
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
x_data = np.load('/Users/fan/Documents/Data/EEG/BCI_CompetitionIV_2a/BCICIV_2a_gdf/sourcedata/traindata/feature.npy')
y_label = np.load('/Users/fan/Documents/Data/EEG/BCI_CompetitionIV_2a/BCICIV_2a_gdf/sourcedata/trainlabel/trainlabel.npy')


kf = KFold(n_splits=5, shuffle=True)
acc_res = []
kappa_res = []
for train_index, test_index in kf.split(x_data):

    # split data
    X_train, X_test = x_data[train_index, :], x_data[test_index, :]
    X_train_2D = (X_train.reshape(X_train.shape[0], X_train.shape[1] * X_train.shape[2]))
    X_test_2D = (X_test.reshape(X_test.shape[0], X_test.shape[1] * X_test.shape[2]))

    y_train, y_test = y_label[train_index], y_label[test_index]


    # Model
    svmmodel = svm.SVC(kernel='poly')

    svmmodel.fit(X_train_2D, y_train)
    t_score = svmmodel.score(X_train_2D, y_train)
    print('t_score', t_score)
    Predict_Score = svmmodel.predict(X_test_2D)
    print('-Predict_Score-', Predict_Score)
    print('-y_test-', y_test)

    acc = accuracy_score(y_test, Predict_Score)
    print('-acc-', acc)
    acc_res.append(acc)

    kappa = cohen_kappa_score(np.array(y_test).reshape(-1, 1), np.array(Predict_Score).reshape(-1, 1))
    print('-kappa-', kappa)
    kappa_res.append(kappa)


print('Result: acc=%.3f, kappa=%.3f ' % (np.mean(acc_res), np.mean(kappa_res)))
