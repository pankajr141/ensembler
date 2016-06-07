'''
Created on 21-Jan-2016

@author: pankajrawat
'''
import copy
from datetime import datetime
import os

from scipy.optimize import minimize
from sklearn import cross_validation
from sklearn import metrics
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
import numpy as np
import optimize
import pandas as pd


def models_avg(clfs, x_test, y_test):
    print "#-------------- Models Averaging -----------------#"
    predictions = []
    for clf in clfs:
        _prediction = clf.predict_proba(x_test)
        print "Score: ", clf.__class__.__name__, " -> ", metrics.log_loss(y_test, _prediction)
        predictions.append(_prediction)

    def log_loss_func(weights):
        ''' scipy minimize will pass the weights as a numpy array '''
        final_prediction = 0
        for weight, prediction in zip(weights, predictions):
                final_prediction += weight * prediction
        return metrics.log_loss(y_test, final_prediction)

    starting_values = [0.5] * len(predictions)
    cons = ({'type': 'eq', 'fun': lambda w: 1 - sum(w)})
    bounds = [(0, 1)] * len(predictions)
    res = minimize(log_loss_func, starting_values, method='SLSQP', bounds=bounds, constraints=cons)
    print type(res)
    print('Ensemble Score: {best_score}'.format(best_score=res['fun']))
    print('Best Weights: {weights}'.format(weights=res['x']))


class BlendModel():
    """
    Parameters
    ----------
    baseEstimators : list
        Object list of all the classifiers which will be used as a base level estimators.

    nFoldsBase : int, optional (default=3)
        Cross validation for base models, it is neccessary if to train base estimators on partial data, 
        if they see all data then our meta estimator will overfit on train data.

    saveAndPickBaseDump : bool, default: False
        When set to true will save base estimatar and blended data set to disk on the location specified by
        saveAndPickBaseDumpLoc. This is crucial when deciding metaEstimator as their is not need to retrain 
        baseEstimators which could be a time consuming process.

    saveAndPickBaseDumpLoc: string (default=tmp)
        Location of the pickled data, if is used when saveAndPickBaseDump is set to true

    metaEstimator: classifier (default=LogisticRegression)
        2nd level classifier which will train on the data provided by baseEstimiators

    metaTunedParams: dict (default=None)
        tuning parameter for metaEstimator, could be different based on metaOptimizer

    nFoldsMeta: int (default=5)
        cross validation for metaEstimator

    metaOptimizer: string (default=grid)
        'grid': uses GridSearchCV to tune metaEstmator
        'random': uses RandomizedSearchCV to tune metaEstmator
        'bayesian': uses BayesianOptimization to tune metaEstimator

    metaOptimizerJobs: int (default=2)
        no of parallel jobs for metaOptimizer

    metaOptimizerIter: int (default=1)
        used when metaOptimizer=random, number of iteration metaOptimizer will run before halting
        Note: metaTunedParams must have this number of combination else it will throw error

    metaOptimizerInitPoints: int (default=2)
        number of random initialization points, used only when metaOptimizer=bayesian

    scoring: string (default=log_loss)
        scoring method to calculate accuracy
   """
    baseEstimators = None
    nFoldsBase = 3
    saveAndPickBaseDump = None
    saveAndPickBaseDumpLoc = None

    # Meta models params
    metaEstimator = None
    metaTunedParams = None
    nFoldsMeta = 5

    # Internal attributes
    _baseEstimators = None
    _metaEstimator = None

    def __init__(self, baseEstimators, nFoldsBase=3, saveAndPickBaseDump=False, saveAndPickBaseDumpLoc='tmp',
                 metaEstimator=LogisticRegression(), metaTunedParams={}, nFoldsMeta=5,
                 metaOptimizer='grid', metaOptimizerJobs=2, metaOptimizerIter=1, metaOptimizerInitPoints=2,
                 scoring='log_loss'):
        self.baseEstimators = baseEstimators
        self.nFoldsBase = nFoldsBase
        self.saveAndPickBaseDump = saveAndPickBaseDump
        self.saveAndPickBaseDumpLoc = saveAndPickBaseDumpLoc
        self.metaEstimator = metaEstimator
        self.metaTunedParams = metaTunedParams
        self.metaOptimizer = metaOptimizer
        self.metaOptimizerJobs = metaOptimizerJobs
        self.metaOptimizerIter = metaOptimizerIter
        self.metaOptimizerInitPoints = metaOptimizerInitPoints
        self.scoring = scoring

    def _generateBlendData(self, df):
        blend_test = np.zeros((df.shape[0], self.numClasses * len(self.baseEstimators)))
        for cntr, _baseEstimator in enumerate(self._baseEstimators):
            start = (cntr % len(self.baseEstimators)) * self.numClasses
            #end = start + 3
            end = start + self.numClasses
            #print 'test blend ', _baseEstimator.__class__.__name__, start, end
            blend_test[:, start: end] = blend_test[:, start: end] + _baseEstimator.predict_proba(df)
            #print blend_test[0:15]
        blend_test = blend_test / self.nFoldsBase
        #print blend_test[0:15]
        return pd.DataFrame(blend_test)

    def _trainBaseClassifiers(self, df, dfLabel):
        df = pd.DataFrame(df) if isinstance(df, np.ndarray) else df
        dfLabel = pd.Series(dfLabel) if isinstance(dfLabel, np.ndarray) else dfLabel
        numClasses = len(dfLabel.unique())
        # Check if input data is pandas series or numpy array
        numClasses = len(pd.Series(dfLabel).unique()) if isinstance(dfLabel, np.ndarray) else len(dfLabel.unique())
        self.numClasses = numClasses
        baseEstimators = self.baseEstimators
        #cv = cross_validation.StratifiedShuffleSplit(dfLabel, n_iter=7, test_size=0.3, random_state=0)
        cv = cross_validation.StratifiedKFold(dfLabel, n_folds=self.nFoldsBase, shuffle=True, random_state=0)
        # Pre-allocate the data
        blend_train = np.zeros((df.shape[0], numClasses * len(baseEstimators)))

        _baseEstimators = []
        print "Fitting", len(baseEstimators), "classifiers", self.nFoldsBase, "folds, Total", len(baseEstimators) * self.nFoldsBase
        for cvCntr, (train_index, test_index) in enumerate(cv):
            x_train, x_test = df.iloc[train_index], df.iloc[test_index]
            y_train, y_test = dfLabel.iloc[train_index], dfLabel.iloc[test_index]
            print "+++++++++ CV", cvCntr + 1, "++++++++++++"
            for cntr, baseEstimator in enumerate(baseEstimators):
                print "Model => ", cntr + 1, baseEstimator.__class__.__name__,
                #x_train, x_test = x_train.as_matrix(), x_test.as_matrix()
                #y_train, y_test = y_train.as_matrix(), y_test.as_matrix()
                clfTmp = copy.deepcopy(baseEstimator)
                #===============================================================
                # clfTmp = clf.__class__()
                # if clf.__class__.__name__ == 'SVC':
                #     clfTmp = clf.__class__(probability=True)
                #===============================================================
                startTime = datetime.now()
                clfTmp.fit(x_train, y_train)
                endTime = datetime.now()
                start = numClasses * cntr
                #end = start + 3 # check why 3 may be numclasses
                end = start + numClasses  # check why 3 may be numclasses
                #print clf.__class__.__name__, start, end
                #print test_index
                blend_train[test_index, start: end] = clfTmp.predict_proba(x_test)
                print " TestSet : ", metrics.log_loss(y_test, clfTmp.predict_proba(x_test)),
                print endTime - startTime
                #print blend_train[0:30]
                #print " HoldSet : ", metrics.log_loss(dfHoldOutLabel, clfTmp.predict_proba(dfHoldOut))
                _baseEstimators.append(clfTmp)

        self._baseEstimators = _baseEstimators
        print "Total models trained =>", len(_baseEstimators)
        print '@@@@@#####@@@@@'
        dfBlend = pd.DataFrame(blend_train)
        return dfBlend

    def fit(self, df, dfLabel):
        if self.saveAndPickBaseDump:
            # save dfBlend and _baseEstimators as dump
            if not os.path.exists(self.saveAndPickBaseDumpLoc):
                os.makedirs(self.saveAndPickBaseDumpLoc)
            _baseEstimatorsFilePath = os.path.join(self.saveAndPickBaseDumpLoc, '_baseEstimators.pickle')
            dfBlendFilePath = os.path.join(self.saveAndPickBaseDumpLoc, 'dfBlend.pickle')
            if not os.path.exists(self.saveAndPickBaseDumpLoc):
                os.mkdir(self.saveAndPickBaseDumpLoc)
            if not os.path.exists(_baseEstimatorsFilePath):
                dfBlend = self._trainBaseClassifiers(df, dfLabel)
                joblib.dump(self._baseEstimators, _baseEstimatorsFilePath)
                joblib.dump(dfBlend, dfBlendFilePath)
            self._baseEstimators = joblib.load(_baseEstimatorsFilePath)
            dfBlend = joblib.load(dfBlendFilePath)
            numClasses = len(dfLabel.unique())
            self.numClasses = numClasses
        else:
            dfBlend = self._trainBaseClassifiers(df, dfLabel)

        metaEstimator = self.metaEstimator
        metaTunedParams = self.metaTunedParams
        print 'MetaEstimators', metaEstimator, metaTunedParams

        cv = cross_validation.StratifiedShuffleSplit(dfLabel, n_iter=5, test_size=0.3, random_state=2)
        #scv = None
        _metaEstimator = None
        if self.metaOptimizer == 'grid':
            _metaEstimator = optimize.performGridSearchOptimization(metaEstimator, metaTunedParams, cv, dfBlend, dfLabel,
                                                                    n_jobs=self.metaOptimizerJobs, scoring=self.scoring)
        elif self.metaOptimizer == 'random':
            _metaEstimator = optimize.performRandomizedSearchOptimization(metaEstimator, metaTunedParams, cv, dfBlend, dfLabel,
                                                                          n_iter=self.metaOptimizerIter, n_jobs=self.metaOptimizerJobs,
                                                                          scoring=self.scoring)
        elif self.metaOptimizer == 'bayesian':
            _metaEstimator = optimize.performBayesianOptimization(metaEstimator, metaTunedParams, cv, dfBlend, dfLabel,
                                                                  scoring=self.scoring, init_points=self.metaOptimizerInitPoints, 
                                                                  n_iter=self.metaOptimizerIter)
        assert(_metaEstimator != None, 'metaOptimizer is set to none')
        # scv.fit(dfBlend, dfLabel)
        self._metaEstimator = _metaEstimator

    def predict_proba(self, x):
        return self._metaEstimator.predict_proba(self._generateBlendData(x))

    def predict(self, x):
        return self._metaEstimator.predict(self._generateBlendData(x))

    def base_score(self, x, y, scoring=None):
        """Calculate holdout score of base classifiers"""
        scoring = scoring if scoring else self.scoring
        for cntr, _baseEstimator in enumerate(self._baseEstimators):
            print "BaseModel", cntr, "score : ", _baseEstimator.__class__.__name__, metrics.log_loss(y, _baseEstimator.predict_proba(x))

    def score(self, x, y, scoring=None):
        """Calculate holdout score of blended classifier"""
        scoring = scoring if scoring else self.scoring
        if scoring == 'log_loss':
            print "BlendModel score : ", metrics.log_loss(y, self.predict_proba(x))

if __name__ == "__main__":
    #clf1 = joblib.load(os.path.join(pickle_dir, LogisticRegression().__class__.__name__))
    #clf2 = joblib.load(os.path.join(pickle_dir, xgb.XGBClassifier().__class__.__name__))
    #clfs = [clf1, clf2]
    #models_avg(clfs)
    pass
