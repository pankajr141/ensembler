'''
Created on 08-Mar-2016

@author: pankajrawat
'''
import random

from scipy.stats import randint as sp_randint


ExtraTreesClassifierTuningParams = {
                    'n_estimators': sp_randint(400, 4000),
                    'max_depth': sp_randint(3, 50),  # None also an option
                    'min_samples_split': sp_randint(2, 10),
                    'min_samples_leaf': sp_randint(1, 10),
                    "max_features": [None, 'sqrt', 'log2']
                }


KNeighborsClassifierTuningParams = {
                    'n_neighbors': sp_randint(3, 300),
                    'weights': ['uniform', 'distance'],
                    'metric': ['minkowski', 'euclidean', 'manhattan', 'chebyshev'],
                    #'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                    'algorithm': ['auto', 'ball_tree', 'kd_tree'],

                    #'n_jobs': [3]
                }

LogisticRegressionTuningParams = {
                    'C': [random.uniform(0.001, 2) for i in range(0, 100)],
                    'solver' : ['newton-cg', 'lbfgs', 'liblinear', 'sag'],
                    'penalty': ['l1', 'l2'],
                }


XGBClassifierTuningParams = {
                     'n_estimators':sp_randint(40, 2000),
                     'learning_rate': [random.uniform(0.001, 1) for i in range(0, 100)],
                     'max_depth': sp_randint(3, 10),
                     'subsample': [random.uniform(0.6, 1) for i in range(0, 100)],
                     'colsample_bytree': [random.uniform(0.6, 1) for i in range(0, 100)],
                }

GaussianNBTuningParams = {}

RandomForestClassifierTuningParams = {
                    'n_estimators': sp_randint(40, 2000),
                    'max_features': [random.uniform(0.001, 1) for i in range(0, 100)],
                    'min_samples_split': sp_randint(40, 2000),
                    'min_samples_split': sp_randint(40, 2000),
                    'min_samples_split': sp_randint(40, 2000),

                } 