'''
Created on 08-Mar-2016

@author: pankajrawat
'''
from bayes_opt import BayesianOptimization
from sklearn import cross_validation
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.grid_search import ParameterGrid


def performRandomizedSearchOptimization(clf, tuned_params, cv, x, y, n_iter=50, n_jobs=-1, scoring='log_loss'):
    print "*" * 60, "RandomSearch", "*" * 60
    rscv = RandomizedSearchCV(clf, param_distributions=tuned_params, cv=cv, verbose=3, scoring=scoring, n_jobs=n_jobs, n_iter=n_iter, error_score=-100)
    rscv.fit(x, y)
    print rscv.best_estimator_, rscv.best_score_
    print "=" * 30, " END ", "=" * 60
    return rscv.best_estimator_


def performGridSearchOptimization(clf, tuned_params, cv, x, y, n_jobs=-1, scoring='log_loss'):
    print "*" * 60, "GridSearch", "*" * 60
    gscv = GridSearchCV(clf, param_grid=tuned_params, cv=cv, verbose=3, scoring=scoring, n_jobs=n_jobs, error_score=-100)
    gscv.fit(x, y)
    print gscv.best_estimator_, gscv.best_score_
    print "=" * 30, " END ", "=" * 60
    return gscv.best_estimator_


def performBayesianOptimization(clf, params, cv, x, y, scoring='log_loss', init_points=2, n_iter=50):
    print "*" * 60, "Bayesian", "*" * 60

    tuned_params = params['tuned_params']
    explore_params = params['explore_params']
    tuned_params_int = params['tuned_params_int']
    repeat = params['repeat']
    repeat_dict = params['repeat_dict']
    repeat_grid = None if not repeat else list(ParameterGrid(repeat_dict))
    cntr = 0
    while  True:
        def _cvfunction(**kwargs):
            try:
                clfT = clf.__class__()
                for key, val in kwargs.items():
                    if key in tuned_params_int:
                        val = int(round(val))
                    setattr(clfT, key, val)
                if repeat:
                    for key, val in repeat_grid[cntr].items():
                        if key in tuned_params_int:
                            val = int(round(val))
                        setattr(clfT, key, val)
                return  cross_val_score(clfT, x.as_matrix(), y.as_matrix(), scoring=scoring, cv=cv).mean()
            except Exception, err:
                print err
                return -999.9  # Return large negative value on error

        if repeat:
            print "\nRepeat => ", cntr, "\nNon numeric Attrib => ", repeat_grid[cntr]
        optimizer = BayesianOptimization(_cvfunction, tuned_params)
        if explore_params:
            optimizer.explore(explore_params)
        optimizer.maximize(init_points=init_points, n_iter=n_iter, acq='ucb', kappa=10)
        print clf.__class__.__name__, optimizer.res['max']['max_val']

        cntr += 1
        if not repeat or (cntr == len(repeat_grid)):
            break

    estimator = clf.__class__()
    for key, val in optimizer.res['max']['max_params'].items():
        if key in tuned_params_int:
            val = int(round(val))
        setattr(estimator, key, val)
    print "=" * 60, " END ", "=" * 60
    estimator.fit(x, y)
    print estimator, cross_validation.cross_val_score(estimator, x, y, scoring=scoring)
    return estimator
