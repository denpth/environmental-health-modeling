import unittest
import numpy as np
from sklearn.model_selection import KFold,cross_validate
from evaluate import build_pipeline

class FoldIsolationTests(unittest.TestCase):
    def test_transformations_are_fit_only_on_training_folds(self):
        rng=np.random.default_rng(10); X=rng.normal(size=(36,20)); y=X[:,0]+rng.normal(size=36)
        X[0,0]=10000; X[2,2]=np.nan
        cv=KFold(3,shuffle=True,random_state=42)
        result=cross_validate(build_pipeline(20,trees=5),X,y,cv=cv,return_estimator=True)
        for (train,valid),est in zip(cv.split(X),result['estimator']):
            np.testing.assert_allclose(est['impute'].statistics_,np.nanmedian(X[train],axis=0))
            transformed=est['select'].transform(est['impute'].transform(X[train]))
            np.testing.assert_allclose(est['scale'].mean_,transformed.mean(axis=0))
            self.assertTrue(np.isfinite(est.predict(X[valid])).all())

if __name__=='__main__':unittest.main()
