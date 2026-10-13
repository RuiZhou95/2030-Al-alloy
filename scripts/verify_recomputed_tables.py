"""Compare recomputed table stages with their released numerical references."""
from pathlib import Path
import json
import pandas as pd
from release_paths import REPO_ROOT,OUTPUT_ROOT


def main():
    checks=[]
    groups={'paper':('paper/data',[
        'mechanism_descriptor_summary.csv','state_response_summary.csv','cross_grade_design_validation.csv',
        'prospective_cross_grade_validation.csv','fig3_state_responses.csv','fig5_trial_predictions.csv']),
        'descriptors':('paper/data_materials_v2',[
        'descriptor_library.csv','descriptor_candidate_matrix_284.csv','descriptor_stability_all.csv',
        'descriptor_selected_by_grade.csv','descriptor_primary_panel.csv','descriptor_role_assignment.csv'])}
    for group,(dest,names) in groups.items():
        for name in names:
            reference=pd.read_csv(REPO_ROOT/'data/reference'/group/name)
            actual=pd.read_csv(OUTPUT_ROOT/dest/name)
            pd.testing.assert_frame_equal(actual,reference,check_dtype=False,check_exact=False,rtol=1e-9,atol=1e-10)
            checks.append({'table':name,'rows':len(actual),'status':'PASS'})
    p=OUTPUT_ROOT/'reproduction';p.mkdir(parents=True,exist_ok=True)
    (p/'table_recomputation.json').write_text(json.dumps({'status':'PASS','tables':checks},indent=2),encoding='utf-8')
    print(json.dumps({'status':'PASS','verified_tables':len(checks)},indent=2))


if __name__=='__main__':main()
