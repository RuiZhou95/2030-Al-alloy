# -*- coding: utf-8 -*-
"""
07_verification.py
实验验证（推断式）：对 5 个试制验证样本（每牌号兼顾度最优 1 名 + 全局次优 1 名，技术实现上
从 289 样本中排除于 284 样本建模集）检验基于 284 样本模型的两条推断。
推断 (i)：六模型（284 样本重训）对试制样本三性质的预测与 ±1.96×CV-RMSE 区间；
推断 (ii)：多目标寻优推断试制样本应位于 YS-EL / UTS-EL / YS-UTS 帕累托前沿兼顾区。
输出：data/verification_predictions.csv、figure/fig8_verification.png（论文图10）
环境：LubLLM
"""
import os
import json
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, 'data')
FIG_DIR = os.path.join(BASE, 'figure')

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.9,
    'legend.frameon': False,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
})

TARGETS = ['yield_strength_mpa', 'ultimate_tensile_strength_mpa', 'elongation_pct']
TARGET_EN = {'yield_strength_mpa': 'YS (MPa)',
             'ultimate_tensile_strength_mpa': 'UTS (MPa)',
             'elongation_pct': 'EL (%)'}
MODEL_NAMES = {'ridge': 'Ridge', 'xgb': 'XGBoost', 'rf': 'RandomForest',
               'svr': 'SVR', 'gpr': 'GPR', 'mlp': 'MLP'}
MODEL_COLORS = {'Ridge': '#0F4D92', 'XGBoost': '#B64342', 'RandomForest': '#2E7D32',
                'SVR': '#7C6CCF', 'GPR': '#E65100', 'MLP': '#8E24AA'}
BEST_SURROGATE = {'yield_strength_mpa': 'RandomForest',
                  'ultimate_tensile_strength_mpa': 'RandomForest',
                  'elongation_pct': 'RandomForest'}

COMP_COLS = ['Al_wt_pct', 'Ti_wt_pct', 'Fe_wt_pct', 'Ni_wt_pct', 'Cr_wt_pct',
             'Mn_wt_pct', 'Si_wt_pct', 'Cu_wt_pct', 'Zn_wt_pct', 'Mg_wt_pct',
             'Mo_wt_pct', 'V_wt_pct', 'Nb_wt_pct', 'Zr_wt_pct', 'Sn_wt_pct',
             'Co_wt_pct', 'C_wt_pct', 'N_wt_pct', 'O_wt_pct', 'B_wt_pct',
             'P_wt_pct', 'S_wt_pct']
PROC_COLS = ['homogenization_temp_c', 'homogenization_time_h',
             'solution_temp_c', 'solution_time_h',
             'aging_temp_c', 'aging_time_h',
             'annealing_temp_c', 'annealing_time_h',
             'rolling_temp_c', 'rolling_reduction_pct', 'specimen_thickness_mm']


def get_model(mname):
    if mname == 'ridge':
        return Ridge(random_state=42)
    if mname == 'xgb':
        return XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=4)
    if mname == 'rf':
        return RandomForestRegressor(random_state=42, n_jobs=4)
    if mname == 'svr':
        return SVR()
    if mname == 'gpr':
        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.01)
        return GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                        n_restarts_optimizer=3, random_state=42)
    if mname == 'mlp':
        return MLPRegressor(random_state=42, max_iter=2000, early_stopping=True)
    raise ValueError(mname)


def engineer_holdout(holdout):
    """按 01 的口径对试制样本做同样的特征工程（类别编码不纳入特征集）。"""
    h = holdout.copy()
    for c in COMP_COLS + PROC_COLS:
        if c not in h.columns:
            h[c] = 0.0
        h[c] = h[c].apply(lambda v: 0 if v in ('无', '不关注', '/', '', None) else v)
        h[c] = pd.to_numeric(h[c], errors='coerce').fillna(0)
    h['Zn_x_Mg'] = h['Zn_wt_pct'] * h['Mg_wt_pct']
    h['Cu_x_Mg'] = h['Cu_wt_pct'] * h['Mg_wt_pct']
    h['Mg_x_Si'] = h['Mg_wt_pct'] * h['Si_wt_pct']
    h['total_alloy_wt'] = h[COMP_COLS].drop(columns=['Al_wt_pct']).sum(axis=1)
    return h


def eval_formula(ftext, vals, names):
    expr = ftext
    for op, py in [('square', '**2'), ('cube', '**3')]:
        while op + '(' in expr:
            i = expr.find(op + '(')
            depth, j = 0, i + len(op)
            while j < len(expr):
                if expr[j] == '(':
                    depth += 1
                elif expr[j] == ')':
                    depth -= 1
                    if depth == 0:
                        break
                j += 1
            expr = expr[:i] + '(' + expr[i + len(op) + 1:j] + ')' + py + expr[j + 1:]
    while 'sqrt(' in expr:
        i = expr.find('sqrt(')
        depth, j = 0, i + len('sqrt')
        while j < len(expr):
            if expr[j] == '(':
                depth += 1
            elif expr[j] == ')':
                depth -= 1
                if depth == 0:
                    break
            j += 1
        expr = expr[:i] + '(abs(' + expr[i + len('sqrt') + 1:j] + '))**0.5' + expr[j + 1:]
    mapping = dict(zip(names, vals))
    for k in sorted(mapping, key=len, reverse=True):
        expr = expr.replace(k, repr(float(mapping[k])))
    return float(eval(expr, {'abs': abs}, {}))


def sr_pred_for_holdout(target, h_features, si):
    feat_file = os.path.join(DATA_DIR, f'sr_{target}_features.txt')
    if target == 'elongation_pct':
        best_file = os.path.join(DATA_DIR, 'sr_scan', 'elongation_pct', 'el_single', 'best.json')
    else:
        best_file = os.path.join(DATA_DIR, f'sr_{target}_best.json')
    if not (os.path.exists(feat_file) and os.path.exists(best_file)):
        return None, None
    top8 = [l.strip() for l in open(feat_file, encoding='utf-8') if l.strip()]
    best = json.load(open(best_file, encoding='utf-8'))
    df284 = pd.read_csv(os.path.join(DATA_DIR, '02_engineered_features.csv'), encoding='utf-8-sig')
    X = df284[top8].values.astype(float)
    X_tr, _ = train_test_split(X, test_size=0.2, random_state=42)
    scaler = StandardScaler().fit(X_tr)
    vals = scaler.transform(h_features[top8].values.astype(float))[si]
    pred = eval_formula(best['formula_text'], vals, top8)
    return pred, best


def main():
    holdout = pd.read_csv(os.path.join(DATA_DIR, 'verification_holdout.csv'), encoding='utf-8-sig')
    feat_df = pd.read_csv(os.path.join(DATA_DIR, '02_engineered_features.csv'), encoding='utf-8-sig')
    feature_cols = [c for c in feat_df.columns
                    if c not in TARGETS + ['sample_id', 'material_system', 'alloy_or_coating_name']]
    h = engineer_holdout(holdout)
    results = json.load(open(os.path.join(DATA_DIR, 'results.json'), encoding='utf-8'))
    samples = [r['sample_id'] for _, r in holdout.iterrows()]
    grades = dict(zip(holdout['sample_id'], holdout['alloy_or_coating_name']))

    rows = []
    for t in TARGETS:
        X284 = feat_df[feature_cols].values.astype(float)
        y284 = feat_df[t].values.astype(float)
        scaler = StandardScaler().fit(X284)
        X284_s = scaler.transform(X284)
        h_s = scaler.transform(h[feature_cols].values.astype(float))
        best_name = BEST_SURROGATE[t]
        for si, sid in enumerate(samples):
            row = {'sample_id': sid, 'target': t, 'measured': float(h[t].values[si])}
            for mname in ['ridge', 'xgb', 'rf', 'svr', 'gpr', 'mlp']:
                name = MODEL_NAMES[mname]
                params = results[t][name]['best_params']
                model = get_model(mname)
                model.set_params(**params)
                model.fit(X284_s, y284)
                row['pred_' + name] = float(model.predict(h_s[[si]])[0])
                if name == best_name:
                    kf = KFold(n_splits=10, shuffle=True, random_state=42)
                    errs = []
                    for i_tr, i_val in kf.split(X284_s):
                        m = get_model(mname)
                        m.set_params(**params)
                        m.fit(X284_s[i_tr], y284[i_tr])
                        errs.append(mean_squared_error(y284[i_val], m.predict(X284_s[i_val])))
                    row['cv_rmse'] = float(np.sqrt(np.mean(errs)))
            pred_sr, sr_best = sr_pred_for_holdout(t, h, si)
            row['pred_SR'] = pred_sr
            rows.append(row)
    ver = pd.DataFrame(rows)

    front_ys = pd.read_csv(os.path.join(DATA_DIR, 'multiobj_front_ys.csv')).values
    front_uts = pd.read_csv(os.path.join(DATA_DIR, 'multiobj_front_uts.csv')).values
    y_span = max(front_ys[:, 0]) - min(front_ys[:, 0])
    u_span = max(front_uts[:, 0]) - min(front_uts[:, 0])
    el_span = feat_df['elongation_pct'].max() - feat_df['elongation_pct'].min()
    dist_ys, dist_uts = {}, {}
    for si, sid in enumerate(samples):
        ym, em = float(h['yield_strength_mpa'].values[si]), float(h['elongation_pct'].values[si])
        um = float(h['ultimate_tensile_strength_mpa'].values[si])
        d1 = np.min(np.sqrt(((front_ys[:, 0] - ym) / y_span) ** 2 +
                            ((front_ys[:, 1] - em) / el_span) ** 2))
        d2 = np.min(np.sqrt(((front_uts[:, 0] - um) / u_span) ** 2 +
                            ((front_uts[:, 1] - em) / el_span) ** 2))
        dist_ys[sid] = float(d1)
        dist_uts[sid] = float(d2)
    ver['dist_front'] = [dist_ys.get(s, np.nan) if t == 'yield_strength_mpa' else
                         (dist_uts.get(s, np.nan) if t == 'ultimate_tensile_strength_mpa' else np.nan)
                         for t, s in zip(ver['target'], ver['sample_id'])]
    ver.to_csv(os.path.join(DATA_DIR, 'verification_predictions.csv'), index=False,
               encoding='utf-8-sig')
    print('saved data/verification_predictions.csv')
    for _, r in ver.iterrows():
        print(f"{r['sample_id']} {r['target']:30s} 实测={r['measured']:7.1f} "
              f"最优({BEST_SURROGATE[r['target']]})={r['pred_'+BEST_SURROGATE[r['target']]]:7.1f} "
              f"SR={r['pred_SR'] if r['pred_SR'] is None else round(r['pred_SR'],1)}")

    # ---------- 图10 ----------
    fig = plt.figure(figsize=(13.5, 8.4))
    gs = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.32,
                  left=0.08, right=0.97, top=0.94, bottom=0.15)
    xpos = np.arange(len(samples))
    xlab = [f'{s} ({grades[s]})' for s in samples]
    for k, t in enumerate(TARGETS):
        ax = fig.add_subplot(gs[0, k])
        sub = ver[ver['target'] == t]
        jitter = {'Ridge': -0.15, 'XGBoost': -0.09, 'RandomForest': -0.03,
                  'SVR': 0.03, 'GPR': 0.09, 'MLP': 0.15}
        for name in MODEL_NAMES.values():
            ax.scatter(xpos + jitter[name], sub['pred_' + name], s=55, c=MODEL_COLORS[name],
                       alpha=0.85, edgecolors='white', linewidths=0.6, label=name, zorder=3)
        best_name = BEST_SURROGATE[t]
        ax.errorbar(xpos, sub['pred_' + best_name], yerr=1.96 * sub['cv_rmse'],
                    fmt='none', ecolor=MODEL_COLORS[best_name], elinewidth=1.8,
                    capsize=6, capthick=1.4, zorder=4,
                    label=best_name + ' $\pm$1.96$\\times$CV-RMSE')
        ax.scatter(xpos, sub['measured'], s=170, marker='*', c='#111111', zorder=6,
                   edgecolors='white', linewidths=0.8, label='Measured')
        if sub['pred_SR'].notna().any():
            ax.scatter(xpos, sub['pred_SR'], s=140, marker='D', facecolors='none',
                       edgecolors='#767676', linewidths=1.6, zorder=5, label='SR formula')
        ax.set_xticks(xpos)
        ax.set_xticklabels(xlab, fontsize=11)
        ax.set_ylabel(TARGET_EN[t], fontsize=14)
        ax.tick_params(labelsize=11)
        ax.text(0.02, 1.02, '(' + 'abc'[k] + ') ' + {'yield_strength_mpa': 'YS',
                                                     'ultimate_tensile_strength_mpa': 'UTS',
                                                     'elongation_pct': 'EL'}[t],
                transform=ax.transAxes, fontsize=15, fontweight='bold', va='bottom')

    train_pts = feat_df[['yield_strength_mpa', 'ultimate_tensile_strength_mpa', 'elongation_pct']].values
    pts_yu = np.vstack([front_ys[:, [0, 2]], front_uts[:, [2, 0]]])
    keep = np.ones(len(pts_yu), bool)
    for i in range(len(pts_yu)):
        if not keep[i]:
            continue
        dom = ((pts_yu[:, 0] >= pts_yu[i, 0]) & (pts_yu[:, 1] >= pts_yu[i, 1]) &
               ((pts_yu[:, 0] > pts_yu[i, 0]) | (pts_yu[:, 1] > pts_yu[i, 1])))
        keep[dom] = False
        keep[i] = True
    front_yu = pts_yu[keep]
    front_yu = front_yu[np.argsort(front_yu[:, 1])]

    panels = [
        (front_ys, 'yield_strength_mpa', 'YS (MPa)', 'YS-EL', 'd'),
        (front_uts, 'ultimate_tensile_strength_mpa', 'UTS (MPa)', 'UTS-EL', 'e'),
    ]
    for k, (front, tgt, ylab, pname, ptag) in enumerate(panels):
        ax = fig.add_subplot(gs[1, k])
        ax.scatter(train_pts[:, 2], train_pts[:, 0 if tgt == 'yield_strength_mpa' else 1], s=40,
                   c='#CFCECE', alpha=0.7, edgecolors='none', label='Training samples')
        order = np.argsort(front[:, 1])
        ax.plot(front[order, 1], front[order, 0], '-', c='#E53935', lw=2.2, zorder=4,
                label='Pareto front (surrogate)')
        ax.scatter(front[:, 1], front[:, 0], s=45, c='#E53935', alpha=0.9, zorder=4,
                   edgecolors='white', linewidths=0.5)
        for si, sid in enumerate(samples):
            em = float(h['elongation_pct'].values[si])
            sm = float(h[tgt].values[si])
            sub = ver[(ver['sample_id'] == sid) & (ver['target'] == tgt)]
            pm = float(sub['pred_' + BEST_SURROGATE[tgt]])
            pe = float(ver[(ver['sample_id'] == sid) & (ver['target'] == 'elongation_pct')]
                       ['pred_' + BEST_SURROGATE['elongation_pct']])
            ax.plot([pe, em], [pm, sm], '-', color='#767676', lw=1.0, zorder=3, alpha=0.8)
            ax.scatter(pe, pm, s=150, facecolors='none', edgecolors='#4D4D4D', linewidths=1.6,
                       zorder=5, label='Surrogate predicted' if si == 0 else None)
            ax.scatter(em, sm, s=260, marker='*', c='#111111', zorder=6,
                       edgecolors='white', linewidths=1.0,
                       label='Measured (verification)' if si == 0 else None)
        ax.set_xlabel('EL (%)', fontsize=14)
        ax.set_ylabel(ylab, fontsize=14)
        ax.tick_params(labelsize=12)
        ax.text(0.02, 1.02, '(' + ptag + ') ' + pname + ' Pareto check',
                transform=ax.transAxes, fontsize=15, fontweight='bold', va='bottom')
        ax.legend(fontsize=10, loc='lower left')

    ax = fig.add_subplot(gs[1, 2])
    ax.scatter(train_pts[:, 1], train_pts[:, 0], s=40, c='#CFCECE', alpha=0.7,
               edgecolors='none', label='Training samples')
    ax.plot(front_yu[:, 1], front_yu[:, 0], '-', c='#E53935', lw=2.2, zorder=4,
            label='Pareto front (surrogate)')
    ax.scatter(front_yu[:, 1], front_yu[:, 0], s=45, c='#E53935', alpha=0.9, zorder=4,
               edgecolors='white', linewidths=0.5)
    for si, sid in enumerate(samples):
        sm = float(h['yield_strength_mpa'].values[si])
        um = float(h['ultimate_tensile_strength_mpa'].values[si])
        pm = float(ver[(ver['sample_id'] == sid) & (ver['target'] == 'yield_strength_mpa')]
                   ['pred_' + BEST_SURROGATE['yield_strength_mpa']])
        pu = float(ver[(ver['sample_id'] == sid) & (ver['target'] == 'ultimate_tensile_strength_mpa')]
                   ['pred_' + BEST_SURROGATE['ultimate_tensile_strength_mpa']])
        ax.plot([pu, um], [pm, sm], '-', color='#767676', lw=1.0, zorder=3, alpha=0.8)
        ax.scatter(pu, pm, s=150, facecolors='none', edgecolors='#4D4D4D', linewidths=1.6,
                   zorder=5, label='Surrogate predicted' if si == 0 else None)
        ax.scatter(um, sm, s=260, marker='*', c='#111111', zorder=6,
                   edgecolors='white', linewidths=1.0,
                   label='Measured (verification)' if si == 0 else None)
    ax.set_xlabel('UTS (MPa)', fontsize=14)
    ax.set_ylabel('YS (MPa)', fontsize=14)
    ax.tick_params(labelsize=12)
    ax.text(0.02, 1.02, '(f) YS-UTS Pareto check',
            transform=ax.transAxes, fontsize=15, fontweight='bold', va='bottom')
    ax.legend(fontsize=10, loc='lower left')

    handles, labels = fig.axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.02),
               ncol=5, fontsize=10, frameon=False)
    out = os.path.join(FIG_DIR, 'fig8_verification.png')
    fig.savefig(out, dpi=600, bbox_inches='tight')
    plt.close(fig)
    print('saved', out)
    print('07 done')


if __name__ == '__main__':
    main()
