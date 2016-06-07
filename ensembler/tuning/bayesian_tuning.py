'''
Created on 08-Mar-2016

@author: pankajrawat
'''
import math


def getXGBClassifierTuningParams():
    repeat, repeatDict = False, None
    _tuningParams = {
                        'n_estimators': (40, 2000),
                        'learning_rate': (0.001, 0.5),
                        'max_depth': (3, 10),
                        "subsample": (0.6, 1),
                        "colsample_bytree": (0.6, 1),
                    }
    tuningParamsInt = ['n_estimators', 'max_depth']

    tuningParams = {
                'tuned_params': _tuningParams,
                'explore_params': {},
                'tuned_params_int': tuningParamsInt,
                'repeat': repeat,
                'repeat_dict': repeatDict
    }
    return tuningParams


def getExtraTreesClassifierTuningParams(totalFeatures):
    repeat, repeatDict = False, None
    _tuningParams = {
                        'n_estimators': (400, 4000),
                        'max_depth': (3, 50),  # None also an option
                        'min_samples_split': (2, 10),
                        'min_samples_leaf': (1, 10),
                        "max_features": (math.log(totalFeatures, 2), totalFeatures)
                    }
    tuningParamsInt = ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'max_features']

    repeat = True
    repeatDict = {
                    'criterion': ['gini', 'entropy']
                 }
    tuningParams = {
                'tuned_params': _tuningParams,
                'explore_params': {},
                'tuned_params_int': tuningParamsInt,
                'repeat': repeat,
                'repeat_dict': repeatDict
    }

    return tuningParams


def getLogisticRegressionTuningParams():
    repeat, repeatDict = False, None
    _tuningParams = {
                        'C': (0.001, 2),
                    }
    tuningParamsInt = []

    repeat = True
    repeatDict = {
                    #'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag'],
                    #'penalty': ['l1', 'l2']
                }
    tuningParams = {
                'tuned_params': _tuningParams,
                'explore_params': {},
                'tuned_params_int': tuningParamsInt,
                'repeat': repeat,
                'repeat_dict': repeatDict
    }
    return tuningParams
