'''
Created on 08-Mar-2016

@author: pankajrawat
'''
import random
tuned_params = {
                #'learning_rate': [0.001, 0.01, 0.05, 0.02, 0.1, 1.0],
                #'n_estimators': [400, 1000, 2000, 3000],
                #'max_depth': [3, 4, 5],
                #'nthread': [3]
#                   'loss': ['deviance', 'exponential'],
                #'max_features': ["sqrt"]
                #'metric': ['euclidean', 'manhattan', 'chebyshev', 'minkowski'],
                #'algorithm': ['auto', 'brute'],
                #'n_neighbors': [50, 60, 70, 100, 150],
                #'p': [1, 2],
                #'weights': ['distance', 'uniform'],
                #'n_estimators': [100, 1000, 2000, 5000, 10000],
                #'max_features': [None, 'sqrt', 'log2'],
                # 'penalty': ['l1', 'l2'],
                # 'C': [0.01, 1.0, 10.0, 100.0],
                #'gamma': ['auto', .01, 0.001, 0.0001],
                #'kernel': ['linear'],
                #'degree': [2, 3]
}


GaussianNBTuningParams = {}

LogisticRegressionTuningParams = {
                    'C': [0.001, 0.01, 0.05, 0.02, 0.1, 1.0],
                    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag'],
                    'penalty': ['l1', 'l2'],
                }

RandomForestClassifierTuningParams = {
                }