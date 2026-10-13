"""Recompute the previously missing materials-analysis table stage.

Uses the actual materials-analysis implementation, not a new model or selection rule.
"""
import importlib.util,sys
from pathlib import Path


def load(name):
    path=Path(__file__).parent/name;spec=importlib.util.spec_from_file_location(path.stem,path)
    m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m);return m


def main():
    m=load('15_materials_source_analysis.py');m.setup()
    raw,ext,pred,groups=m.read_inputs()
    tr=m.engineer(raw,'modelling');ex=m.engineer(ext,'post-model trial')
    desc=m.mechanism_summary(tr);state=m.state_response_summary(tr,groups)
    design=m.design_validation(tr,ex);rf=m.historical_rf_validation(pred)
    m.write_process_coverage(tr)
    reader=load('12_make_supplementary_figures.py');reader.setup()
    reader.source_tables(tr,ex,desc,state,design,rf)
    print('Regenerated materials and reported-RF source tables; no candidate selection or new model fitting.')


if __name__=='__main__':main()
