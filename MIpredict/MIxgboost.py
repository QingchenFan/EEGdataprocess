from sklearn.metrics import accuracy_score, cohen_kappa_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
import numpy as np


x_data = np.load('/Users/fan/Documents/Data/EEG/BCI_CompetitionIV_2a/BCICIV_2a_gdf/sourcedata/traindata/feature.npy')
y_label = np.load('/Users/fan/Documents/Data/EEG/BCI_CompetitionIV_2a/BCICIV_2a_gdf/sourcedata/trainlabel/trainlabel.npy')
acc_res = []
kappa_res = []
kf = KFold(n_splits=2, shuffle=True)
for train_index, test_index in kf.split(x_data):
    # split data
    X_train, X_test = x_data[train_index, :], x_data[test_index, :]
    y_train, y_test = y_label[train_index], y_label[test_index]
    X_train_2D = (X_train.reshape(X_train.shape[0], X_train.shape[1] * X_train.shape[2]))
    X_test_2D = (X_test.reshape(X_test.shape[0], X_test.shape[1] * X_test.shape[2]))
    # Model
    Hyper_param = {'max_depth': range(3, 6, 10)}

    predict_model = GridSearchCV(estimator=xgb.XGBClassifier(booster='gbtree',
                                                            learning_rate=0.1,
                                                            n_estimators=100,
                                                            verbosity=0,
                                                            objective='multi:softmax'),
                                 param_grid=Hyper_param,
                                 scoring='accuracy',
                                 verbose=6,
                                 cv=2)
    predict_model.fit(X_train_2D, y_train)

    Predict_Score = predict_model.predict(X_test_2D)
    print('-Predict_Score-', Predict_Score)
    print('-y_test-', y_test)

    acc = accuracy_score(y_test, Predict_Score)
    print('-acc-', acc)
    acc_res.append(acc)
    kappa = cohen_kappa_score(np.array(y_test).reshape(-1, 1), np.array(Predict_Score).reshape(-1, 1))
    print('-kappa-', kappa)
    kappa_res.append(kappa)
print('Result: acc=%.3f, kappa=%.3f ' % (np.mean(acc_res), np.mean(kappa_res)))

