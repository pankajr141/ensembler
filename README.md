Usage
--------
Blending examples:

#### Example 1 - Using default meta estimator (LogisticRegression)
```shell
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from ensembler import blend

clfs = [
          RandomForestClassifier(),
          SVC(probability=True, degree=3, gamma=0.001, kernel='linear'),
          ExtraTreesClassifier(max_depth=6, n_estimators=1000, max_features=None),
]

# Creating a blend/stack of 3 base level estimator with CV 3
blendModel = blend.BlendModel(clfs, nFoldsBase=3)
blendModel.fit(x, y)
predictions = blendModel.predict(xHoldout)
blendModel.score(xHoldout, yHoldOut)
```


#### Example 2 - Using custom meta estimator
```shell
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from ensembler import blend

clfs = [
          RandomForestClassifier(),
          SVC(probability=True, degree=3, gamma=0.001, kernel='linear'),
          ExtraTreesClassifier(max_depth=6, n_estimators=1000, max_features=None),
]

# Tuning parameters for meta estimator
metaTunedParamsRF = {
                          'n_estimators': [2000, 1000, 500],
                          'max_depth': [4, 6]
                     }
metaEstimator = RandomForestClassifier()
metaTunedParams = metaTunedParamsRF

# saveAndPickBaseDump is useful when we dont want to train base level estimator again, 
# it will initially train base estimatior and pickles them to disk. 
# This is handy while trying various metaEstimators
# Creating a blend/stack of 3 base level estimator with CV 4 and custom metaEstimator
 blendModel = blend.BlendModel(clfs, nFoldsBase=4, 
                                saveAndPickBaseDump=True, saveAndPickBaseDumpLoc=r'pickle_dir',
                                metaEstimator=metaEstimator, metaTunedParams=metaTunedParams
                              )
blendModel.fit(x, y)
predictions = blendModel.predict(xHoldout)
blendModel.score(xHoldout, yHoldOut)
```
