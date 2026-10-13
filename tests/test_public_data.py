import hashlib,json,importlib.util,sys,ast
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]

def test_data_manifest():
    m=json.loads((ROOT/'data/manifest.json').read_text())
    for r in m['files']:
        assert hashlib.sha256((ROOT/r['path']).read_bytes()).hexdigest()==r['sha256'],r['path']

def test_fixed_split_and_properties():
    a=pd.read_csv(ROOT/'data/01_processed_data.csv');b=pd.read_csv(ROOT/'data/verification_holdout.csv')
    assert len(a)==284 and len(b)==4 and a.sample_id.is_unique and b.sample_id.is_unique
    assert set(a.sample_id).isdisjoint(b.sample_id)
    assert a.alloy_or_coating_name.value_counts().to_dict()=={7075:99,6082:89,2024:64,5083:32}
    for d in [a,b]:
        assert (d.ultimate_tensile_strength_mpa>=d.yield_strength_mpa).all()
        assert (d.elongation_pct>0).all()
    assert set(b.sample_id)=={'al_150','al_191','al_243','al_48'}

def test_reported_rf_metrics_and_source_table():
    d=pd.read_csv(ROOT/'data/verification_predictions.csv')
    p=pd.read_csv(ROOT/'data/reference/paper/fig5_trial_predictions.csv')
    j=d.merge(p,on=['sample_id','target'],suffixes=('_historical','_paper'),validate='one_to_one')
    assert len(j)==12
    np.testing.assert_allclose(j.pred_RandomForest,j.predicted,rtol=1e-12,atol=1e-12)
    expected={'yield_strength_mpa':16.77,'ultimate_tensile_strength_mpa':15.98,'elongation_pct':1.54}
    for t,q in p.groupby('target'):
        assert round(float(np.sqrt(np.mean((q.predicted-q.measured)**2))),2)==expected[t]

def test_descriptor_counts_and_candidate_counts():
    d=pd.read_csv(ROOT/'data/reference/descriptors/descriptor_library.csv')
    assert len(d)==118
    c=pd.read_csv(ROOT/'data/reference/scientific/robust_route_candidates.csv')
    assert len(c)==48000
    assert int((c.supported & c.physically_feasible).sum())==46705

def test_historical_outcome_selection_remains_disclosed():
    source=(ROOT/'historical/00_data_overview.py').read_text(encoding='utf-8')
    tree=ast.parse(source);fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='select_verification_samples')
    scope={'pd':pd,'TARGETS':['yield_strength_mpa','ultimate_tensile_strength_mpa','elongation_pct']}
    exec(compile(ast.Module(body=[fn],type_ignores=[]),'<historical selection>','exec'),scope)
    train=pd.read_csv(ROOT/'data/01_processed_data.csv');trial=pd.read_csv(ROOT/'data/verification_holdout.csv')
    selected,_=scope['select_verification_samples'](pd.concat([train,trial],ignore_index=True))
    assert set(selected.sample_id)==set(trial.sample_id)
    j=selected[['sample_id','balance_score']].merge(trial[['sample_id','balance_score']],on='sample_id')
    np.testing.assert_allclose(j.balance_score_x,j.balance_score_y,rtol=1e-12,atol=1e-12)
    assert 'outcome_based_selector = False' not in (ROOT/'scripts/00_data_quality_audit.py').read_text()
    p=json.loads((ROOT/'data/split_provenance.json').read_text())
    assert p['experimental_chronology_independently_verified'] is False

def test_licence_and_provenance_documents():
    assert 'CC BY 4.0' in (ROOT/'LICENSE-DATA').read_text()
    assert 'MIT License' in (ROOT/'LICENSE').read_text()
    assert (ROOT/'PROVENANCE.md').is_file() and (ROOT/'CODE_PROVENANCE.json').is_file()
