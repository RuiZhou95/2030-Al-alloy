"""Verify the historical engineered matrix without overwriting released inputs."""
import importlib.util,sys
from pathlib import Path
import pandas as pd
from release_paths import REPO_ROOT,PRIVATE_ROOT


def main():
    path=REPO_ROOT/'historical/01_feature_screen.py'
    spec=importlib.util.spec_from_file_location('feature_source',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    d=pd.read_csv(PRIVATE_ROOT/'data/01_processed_data.csv')
    for c in m.COMP_COLS+m.PROC_COLS:
        if c not in d:d[c]=0.0
        d[c]=pd.to_numeric(d[c],errors='coerce').fillna(0)
    d['Zn_x_Mg']=d.Zn_wt_pct*d.Mg_wt_pct
    d['Cu_x_Mg']=d.Cu_wt_pct*d.Mg_wt_pct
    d['Mg_x_Si']=d.Mg_wt_pct*d.Si_wt_pct
    d['total_alloy_wt']=d[m.COMP_COLS].drop(columns=['Al_wt_pct']).sum(axis=1)
    cols=m.COMP_COLS+m.PROC_COLS+['Zn_x_Mg','Cu_x_Mg','Mg_x_Si','total_alloy_wt']
    nonconst=[c for c in cols if d[c].nunique()>1]
    actual=d[['sample_id','material_system','alloy_or_coating_name']+nonconst+m.TARGETS]
    expected=pd.read_csv(PRIVATE_ROOT/'data/02_engineered_features.csv')
    pd.testing.assert_frame_equal(actual,expected,check_dtype=False,check_exact=False,rtol=1e-10,atol=1e-10)
    print(f'PASS: all {len(actual)} rows and {len(nonconst)} historical features reproduced.')


if __name__=='__main__':main()
