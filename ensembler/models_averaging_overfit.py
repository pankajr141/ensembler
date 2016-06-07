'''
Created on 21-Jan-2016

@author: pankajrawat
'''
from scipy.optimize import minimize
from sklearn import ensemble, cross_validation
from sklearn import metrics
from sklearn.externals import joblib
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
import numpy as np
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
    clfs = None
    _estimaterModel = None

    def __init__(self, clfs):
        self.clfs = clfs

    def _generateBlendData(self, df):
        dfBlend = None
        for clf in self.clfs:
            clf_proba = clf.predict_proba(df)
            if dfBlend is None:
                dfBlend = pd.DataFrame(clf_proba)
                continue
            dfBlend = pd.concat([dfBlend, pd.DataFrame(clf_proba)], axis=1)
        #dfBlend.columns = ['c1', 'c2', 'c3', 'c4', 'c5', 'c6']
        return dfBlend

    def fit(self, df, dfLabel, numClasses):
        #dfBlend = pd.DataFrame(np.zeros(shape=(len(df), numClasses)))
        #clfs = [ensemble.RandomForestClassifier(), ensemble.GradientBoostingClassifier(), ensemble.ExtraTreesClassifier()]
        #numClasses = dfLabel.unique()
        dfBlend = self._generateBlendData(df)
        print dfBlend.shape
        clf = LogisticRegression()
        #clf = ensemble.ExtraTreesClassifier()

        tuned_params = {}
        cv = cross_validation.StratifiedShuffleSplit(dfLabel, n_iter=5, test_size=0.2, random_state=2)
        gscv = GridSearchCV(clf, param_grid=tuned_params, cv=cv, verbose=3, scoring="log_loss", n_jobs=2)
        gscv.fit(dfBlend, dfLabel)
        print gscv.best_estimator_, gscv.best_score_
        self._estimaterModel = gscv.best_estimator_

    def predict_proba(self, x):
        return self._estimaterModel.predict_proba(self._generateBlendData(x))

    def score(self, x, y):
        xBlend = self._generateBlendData(x)
        print "BlendModel score : ", metrics.log_loss(y, self._estimaterModel.predict_proba(xBlend))

if __name__ == "__main__":
    #clf1 = joblib.load(os.path.join(pickle_dir, LogisticRegression().__class__.__name__))
    #clf2 = joblib.load(os.path.join(pickle_dir, xgb.XGBClassifier().__class__.__name__))
    #clfs = [clf1, clf2]
    #models_avg(clfs)
    pass