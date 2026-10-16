"""Fold-contained evaluation. Synthetic fixture results are not health evidence."""
import argparse
import hashlib
import json
import platform
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42

def build_pipeline(features, trees=100):
    return Pipeline([('impute', SimpleImputer(strategy='median',keep_empty_features=True)),
        ('select', SelectKBest(f_regression, k=min(15, features))),
        ('scale', StandardScaler()),
        ('model', RandomForestRegressor(n_estimators=trees,random_state=SEED,n_jobs=1))])

def metric(y, pred):
    return {'rmse':float(np.sqrt(mean_squared_error(y,pred))), 'r2':float(r2_score(y,pred))}

def evaluate(path, folds=5, trials=12, trees=100):
    path=Path(path); df=pd.read_csv(path)
    if 'healthRiskScore' not in df: raise ValueError('Missing healthRiskScore target')
    # Fixed schema exclusions do not learn from targets or validation rows.
    X=df.drop(columns=['healthRiskScore','datetimeEpoch','sunriseEpoch','sunsetEpoch','month','dayOfWeek','isWeekend'],errors='ignore')
    if not len(X.columns): raise ValueError('No predictors')
    X=X.apply(pd.to_numeric,errors='raise'); y=pd.to_numeric(df['healthRiskScore'],errors='raise')
    if y.isna().any() or not np.isfinite(y).all(): raise ValueError('Target must be finite')
    if np.isinf(X.to_numpy()).any(): raise ValueError('Predictors cannot contain infinity')
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.2,random_state=SEED)
    cv=KFold(n_splits=folds,shuffle=True,random_state=SEED)
    baseline=Pipeline([('impute',SimpleImputer(strategy='median',keep_empty_features=True)),('model',RandomForestRegressor(n_estimators=trees,random_state=SEED,n_jobs=1))])
    pipeline=build_pipeline(X.shape[1],trees)
    params=[{'model':[RandomForestRegressor(n_estimators=trees,random_state=SEED,n_jobs=1)],'model__max_depth':[None,6,12], 'model__min_samples_leaf':[1,3,5]},
            {'model':[Ridge()], 'model__alpha':[.1,1.,10.,100.]}]
    search=RandomizedSearchCV(pipeline,params,n_iter=trials,cv=cv,scoring='neg_root_mean_squared_error',random_state=SEED,n_jobs=1,refit=True)
    scoring={'rmse':'neg_root_mean_squared_error','r2':'r2'}
    # Outer CV evaluates the entire inner model-selection procedure.
    base_cv=cross_validate(baseline,X_train,y_train,cv=cv,scoring=scoring)
    nested_cv=cross_validate(search,X_train,y_train,cv=cv,scoring=scoring)
    search.fit(X_train,y_train); baseline.fit(X_train,y_train)
    # Diagnostic comparison only: replicate prefit-transform leakage under fixed
    # selected hyperparameters. It is not a second model-selection estimate.
    chosen=search.best_estimator_
    fixed_cv=cross_validate(clone(chosen),X_train,y_train,cv=cv,scoring=scoring)
    transforms=Pipeline(chosen.steps[:-1]); transformed=transforms.fit_transform(X_train,y_train)
    prefit_cv=cross_validate(clone(chosen[-1]),transformed,y_train,cv=cv,scoring=scoring)
    def cv_metrics(scores):
        rmses=-scores['test_rmse']
        return {'rmse_per_fold':rmses.tolist(),'rmse_mean':float(rmses.mean()),'rmse_std':float(rmses.std()),'r2_per_fold':scores['test_r2'].tolist()}
    return {'dataset_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'rows':len(df),'train_rows':len(X_train),'holdout_rows':len(X_test),'seed':SEED,'folds':folds,'search_trials':trials,'trees':trees,'python':platform.python_version(),'sklearn':sklearn.__version__,
        'baseline_cv':cv_metrics(base_cv),'nested_selection_cv':cv_metrics(nested_cv),
        'fixed_selected_model_fold_contained_cv':cv_metrics(fixed_cv),
        'diagnostic_prefit_transform_cv':cv_metrics(prefit_cv),
        'baseline_holdout':metric(y_test,baseline.predict(X_test)),
        'selected_holdout':metric(y_test,search.predict(X_test)),
        'selected_model':type(chosen[-1]).__name__,
        'best_parameters':{k:repr(v) if k=='model' else v for k,v in search.best_params_.items()},
        'scope':'Random held-out rows from the supplied dataset. Synthetic fixtures do not establish health prediction or reproduce historical results. Nested CV covers tuning; the diagnostic fixed-model comparisons are conditional on hyperparameters chosen using the training partition. No temporal or geographic generalization is established.'}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset',type=Path);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--folds',type=int,default=5);parser.add_argument('--trials',type=int,default=12);parser.add_argument('--trees',type=int,default=100)
    args=parser.parse_args()
    if args.output.exists():parser.error('Refusing to overwrite an existing report')
    if args.folds<2 or args.trials<1 or args.trees<1:parser.error('folds>=2, trials>=1, trees>=1 required')
    result=evaluate(args.dataset,args.folds,args.trials,args.trees)
    args.output.write_text(json.dumps(result,indent=2)+'\n');print(args.output)

if __name__=='__main__': main()
