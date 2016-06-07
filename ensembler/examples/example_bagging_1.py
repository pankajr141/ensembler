'''
Created on 16-Mar-2016

@author: pankajrawat
'''

from multiprocessing import freeze_support
from ensembler.bagging import ensemble


if __name__ == "__main__":
    freeze_support()
    score_dict = {
                    'input1.csv': 0.47317,
                    'input2.csv': 0.45900,
             }
    ensemble.ensemble_byweight_logloss(score_dict)
