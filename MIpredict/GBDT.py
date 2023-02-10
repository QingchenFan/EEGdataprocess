
import numpy as np
import nibabel as nib
import glob
import pandas as pd
from sklearn import svm
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.ensemble import BaggingClassifier, GradientBoostingClassifier
import joblib
from sklearn.metrics import r2_score, make_scorer, accuracy_score, cohen_kappa_score
from datetime import datetime

x_data = np.load('/Users/fan/Documents/Data/EEG/BCI_CompetitionIV_2a/BCICIV_2a_gdf/sourcedata/traindata/feature.npy')
y_label = np.load('/Users/fan/Documents/Data/EEG/BCI_CompetitionIV_2a/BCICIV_2a_gdf/sourcedata/trainlabel/label.npy')
kf = KFold(n_splits=2, shuffle=True)

acc_res = []
kappa_res = []
for train_index, test_index in kf.split(x_data):

    # split data
    X_train, X_test = x_data[train_index, :], x_data[test_index, :]
    X_train_2D = (X_train.reshape(X_train.shape[0], X_train.shape[1] * X_train.shape[2]))
    X_test_2D = (X_test.reshape(X_test.shape[0], X_test.shape[1] * X_test.shape[2]))

    y_train, y_test = y_label[train_index], y_label[test_index]


    # Model
    # GBDT
    clf = GradientBoostingClassifier(n_estimators=100, max_depth=1, random_state=0)

    # 网格交叉验证
    cv_times = 2  # inner
    param_grid = {
        'base_estimator__learning_rate': [1.0, 0.1],
    }
    predict_model = GridSearchCV(clf, param_grid, scoring='accuracy', verbose=6, cv=cv_times)

    predict_model.fit(X_train_2D, y_train)
    best_model = predict_model.best_estimator_
    # # weight
    # feature_weight = np.zeros([np.shape(X_train)[1], 1])
    # for i, j in enumerate(predict_model.best_estimator_.estimators_):
    #     # print('第{}个模型的系数{}'.format(i, j.coef_))
    #     # test.to_csv('test_'+str(epoch)+'_'+str(i)+'.csv')
    #     feature_weight = np.add(j.coef_, feature_weight)
    #
    # num = len(predict_model.best_estimator_.estimators_)
    # feature_weight_mean = feature_weight / num
    # print('--feature_weight_mean--\n', feature_weight_mean)
    # feature_weight_res = np.add(feature_weight_mean, feature_weight_res)  # sum = sum + 1
    #

    Predict_Score = best_model.predict(X_test_2D)
    print('-Predict_Score-', Predict_Score)
    print('-y_test-', y_test)

    acc = accuracy_score(y_test, Predict_Score)
    print('-acc-', acc)
    acc_res.append(acc)
    kappa = cohen_kappa_score(np.array(y_test).reshape(-1, 1), np.array(Predict_Score).reshape(-1, 1))
    print('-kappa-', kappa)
    kappa_res.append(kappa)

print('Result: acc=%.3f, kappa=%.3f ' % (np.mean(acc_res), np.mean(kappa_res)))
