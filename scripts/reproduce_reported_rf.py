"""Replay the paper's historical RF prediction calculation on its fixed inputs.

Faithful computational extraction from historical/07_verification.py.
It does not establish the chronology of candidate design or fabrication.
"""
import argparse,importlib.util,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from release_paths import REPO_ROOT,PRIVATE_ROOT,OUTPUT_ROOT


def module(name):
    path=REPO_ROOT/'historical'/name
    spec=importlib.util.spec_from_file_location(path.stem,path)
    mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
    return mod


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--include-cv',action='store_true')
    args=parser.parse_args()
    old=module('07_verification.py')
    train=pd.read_csv(PRIVATE_ROOT/'data/02_engineered_features.csv')
    trials=pd.read_csv(PRIVATE_ROOT/'data/verification_holdout.csv')
    assert set(train.sample_id).isdisjoint(trials.sample_id)
    targets=old.TARGETS
    cols=[c for c in train if c not in targets+['sample_id','material_system','alloy_or_coating_name']]
    engineered=old.engineer_holdout(trials)
    scaler=StandardScaler().fit(train[cols].to_numpy(float))
    X=scaler.transform(train[cols].to_numpy(float));Z=scaler.transform(engineered[cols].to_numpy(float))
    params=json.loads((PRIVATE_ROOT/'data/results.json').read_text(encoding='utf-8'))
    reference=pd.read_csv(PRIVATE_ROOT/'data/verification_predictions.csv')
    rows=[];checks=[]
    for target in targets:
        hp=params[target]['RandomForest']['best_params']
        model=RandomForestRegressor(random_state=42,n_jobs=1,**hp)
        y=train[target].to_numpy(float);model.fit(X,y);prediction=model.predict(Z)
        expected=reference[reference.target==target].set_index('sample_id').loc[trials.sample_id]
        np.testing.assert_allclose(prediction,expected.pred_RandomForest,atol=1e-8,rtol=1e-10)
        check={'target':target,'rmse':float(np.sqrt(np.mean((prediction-trials[target].to_numpy(float))**2))),
               'maximum_prediction_difference':float(np.max(np.abs(prediction-expected.pred_RandomForest.to_numpy())))}
        if args.include_cv:
            errors=[]
            for a,b in KFold(n_splits=10,shuffle=True,random_state=42).split(X):
                m=RandomForestRegressor(random_state=42,n_jobs=1,**hp);m.fit(X[a],y[a])
                errors.append(mean_squared_error(y[b],m.predict(X[b])))
            cv=float(np.sqrt(np.mean(errors)))
            np.testing.assert_allclose(cv,expected.cv_rmse.to_numpy(),atol=1e-8,rtol=1e-10)
            check['cv_rmse']=cv
        checks.append(check)
        rows.extend({'sample_id':sid,'target':target,'predicted':float(p),'measured':float(v)} for sid,p,v in zip(trials.sample_id,prediction,trials[target]))
    dest=OUTPUT_ROOT/'reproduction';dest.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(dest/'reported_rf_recomputed.csv',index=False)
    report={'status':'PASS','n_training':len(train),'n_trials':len(trials),'n_features':len(cols),
            'historical_preprocessing_reproduced':True,'prospective_chronology_independently_verified':False,
            'cv_recomputed':args.include_cv,'checks':checks}
    (dest/'rf_replay.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
