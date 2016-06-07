'''
Created on 16-Mar-2016

@author: pankajrawat
'''

from multiprocessing import freeze_support
from sklearn import cross_validation
from sklearn import datasets
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from ensembler.blending import ensemble


if __name__ == "__main__":
    freeze_support()

    x, y = datasets.make_classification(n_samples=1000, n_features=20,
                                        n_informative=2, n_redundant=2)
    clfs = [
              RandomForestClassifier(),
              SVC(probability=True, degree=3, gamma=0.001, kernel='linear'),
              ExtraTreesClassifier(max_depth=6, n_estimators=40, max_features=None),
    ]

    # Creating a ensemble/stack of 3 base level estimator with CV 3
    blendModel = ensemble.BlendModel(clfs, nFoldsBase=3)
    x, xHoldout, y, yHoldout = cross_validation.train_test_split(x, y, test_size=0.33)
    blendModel.fit(x, y)
    predictions = blendModel.predict(xHoldout)
    blendModel.base_score(xHoldout, yHoldout)
    score = blendModel.score(xHoldout, yHoldout)
    print "BlendModel score : ", score
